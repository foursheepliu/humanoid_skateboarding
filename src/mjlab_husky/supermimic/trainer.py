"""Online coupled-world-model learning and imagined BPTT for the HUSKY task.

The physical environment/reward stays HUSKY. Unlike PPO, policy gradients flow
through a learned augmented-state model, learned task-reward/failure heads, and
the retained differentiable HUSKY adversarial style discriminator.
"""
from dataclasses import asdict
import copy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from .config import SuperMimicConfig
from .env_adapter import CoupledStateAdapter
from .replay import EpisodeSequenceReplay
from .models import WorldModel, world_model_loss
from .imagination import imagine, critic_loss as imagined_critic_loss


class StyleNormalizer(nn.Module):
    """GPU population moments of raw joint states, separate from mixed precision."""
    def __init__(self, dimensions=23):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dimensions))
        self.register_buffer("var", torch.ones(dimensions))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, frames):
        flat = frames.reshape(-1, self.mean.numel())
        mean, var = flat.mean(0), flat.var(0, unbiased=False)
        n = len(flat)
        delta, count = mean - self.mean, self.count + n
        m2 = self.var * self.count + var * n + delta.square() * self.count * n / count
        self.mean.add_(delta * n / count)
        self.var.copy_(m2 / count)
        self.count.copy_(count)

    def forward(self, frames):
        return ((frames - self.mean) / (self.var + 1e-4).sqrt()).clamp(-10, 10)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def checked_step(loss, optimizer, parameters):
    parameters = list(parameters)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite loss; no optimizer step applied")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    optimizer.step()
    return norm.detach()


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def batch_dict(batch):
    if isinstance(batch, dict):
        return batch
    names = {"states": "states", "actions": "actions", "rewards": "rewards",
             "dones": "dones", "valid": "valid", "context": "context",
             "terminated": "terminated", "transition_valid": "transition_valid"}
    return {name: getattr(batch, attr) for name, attr in names.items() if hasattr(batch, attr)}


class SuperMimicTrainer:
    def __init__(self, env, actor_critic, config: SuperMimicConfig, out: Path, run_name="train"):
        from rsl_rl.modules import DiscriminatorMulti
        from rsl_rl.utils import G1_AMPLoader
        config.validate()
        self.env, self.policy, self.cfg, self.out = env, actor_critic, config, Path(out)
        self.device = env.device
        self.out.mkdir(parents=True, exist_ok=True)
        if not run_name or Path(run_name).name != run_name:
            raise ValueError("run_name must be one directory name")
        self.run_name = run_name
        self.paths = {name: self.out / name / run_name for name in ("log", "configs", "data", "checkpoints")}
        for directory in self.paths.values():
            directory.mkdir(parents=True, exist_ok=True)
        self.adapter = CoupledStateAdapter(env)
        self.world = WorldModel(self.adapter.state_dim, env.action_manager.total_action_dim,
                                self.adapter.context_dim,
                                hidden_dims=(config.hidden_width,) * 3).to(self.device)
        self.replay = EpisodeSequenceReplay(env.num_envs, self.adapter.state_dim,
            env.action_manager.total_action_dim, self.adapter.context_dim,
            config.replay_steps, "cpu" if config.replay_device == "cpu" else self.device)
        self.world_optimizer = torch.optim.Adam(self.world.parameters(), lr=config.model_lr)
        self.actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.policy.critic.parameters(), lr=config.critic_lr)
        self.discriminator = DiscriminatorMulti(23, 5.0, (256, 256), str(self.device), 5,
                                               task_reward_lerp=0.7, use_lerp=False)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=config.discriminator_lr)
        self.style_norm = StyleNormalizer().to(self.device)
        self.expert = G1_AMPLoader(str(self.device), time_between_frames=0.02,
            motion_files="dataset/skate_push", preload_transitions=True,
            num_preload_transitions=config.expert_transitions, num_frames=5)
        self.expert_frames = torch.stack(self.expert.preloaded_frames, dim=1).detach()
        split = min(max(1, int(env.num_envs * 0.9)), env.num_envs - 1) if env.num_envs > 1 else 1
        self.train_env_ids = torch.arange(split, device=self.replay.device)
        self.validation_env_ids = torch.arange(split, env.num_envs, device=self.replay.device) if env.num_envs > 1 else self.train_env_ids
        self.validation_batch = None
        self.validation_generator = torch.Generator(device=self.replay.device).manual_seed(config.seed + 104729)
        self.iteration = 0
        self.real_steps = 0
        self.normalization_fitted = False
        self.reference_actor = copy.deepcopy(self.policy.actor).eval().requires_grad_(False)
        self.reference_normalizer = copy.deepcopy(self.policy.actor_obs_normalizer).eval().requires_grad_(False)
        self.reference_std = self.policy.std.detach().abs().clamp_min(0.05).clone()
        self.trust_inputs = None
        self.policy.train()
        layout = {k: [v.start, v.stop] if isinstance(v, slice) else v
                  for k, v in self.adapter.layout.items()}
        (self.paths["configs"] / "config.json").write_text(json.dumps(asdict(config), indent=2))
        (self.paths["configs"] / "state_layout.json").write_text(json.dumps(layout, indent=2))

    def collect(self, steps):
        returns = []
        with torch.no_grad():
            for _ in range(steps):
                state = self.adapter.observe()
                obs = self.adapter.state_to_observations(state)
                if not getattr(self, "warm_started", False):
                    self.policy.update_normalization(obs)
                action = self.policy.act_inference(obs)
                action = action + self.cfg.collection_noise * torch.randn_like(action)
                transition = self.adapter.capture_step(action)
                self.replay.add(transition)
                returns.append(transition.reward.mean())
                self.real_steps += self.env.num_envs
        return torch.stack(returns).mean() if returns else torch.zeros((), device=self.device)

    def sample(self, size, horizon, env_ids, generator=None):
        batch = batch_dict(self.replay.sample(size, horizon, env_ids=env_ids, generator=generator))
        return {key: value.to(self.device, non_blocking=True) for key, value in batch.items()}

    def _style_frames(self, states):
        return states[..., self.adapter.layout["amp_history"]].reshape(-1, 5, 23)

    def style_update(self):
        batch = self.sample(self.cfg.model_batch, 1, self.train_env_ids)
        frames = self._style_frames(batch["states"][:, 0]).detach()
        indices = torch.randint(len(self.expert_frames), (len(frames),), device=self.device)
        expert = self.expert_frames[indices]
        self.style_norm.update(torch.cat((frames, expert), 0))
        frames, expert = self.style_norm(frames), self.style_norm(expert)
        self.discriminator.train()
        policy_d = self.discriminator(frames.flatten(1))
        expert_d = self.discriminator(expert.flatten(1))
        loss = 0.5 * ((policy_d + 1).square().mean() + (expert_d - 1).square().mean())
        loss = loss + self.discriminator.compute_grad_pen(expert, lambda_=5)
        checked_step(loss, self.discriminator_optimizer, self.discriminator.parameters())
        return loss.detach()

    def style_reward(self, state, action, next_state, task_reward, step_index):
        del state, action, step_index
        frames = self._style_frames(next_state)
        self.discriminator.eval()
        score = self.discriminator(self.style_norm(frames).flatten(1)).squeeze(-1)
        # Exactly the released HUSKY non-lerp style reward scaling, in seconds.
        style = 5.0 * 0.02 * (1 - 0.25 * (score - 1).square()).clamp_min(0)
        push = next_state[:, self.adapter.layout["phase"]][:, 0].detach().clamp(0, 1)
        return task_reward + push * style

    def teacher_statistics(self, inputs):
        reference = self.reference_actor(self.reference_normalizer(inputs))
        current = self.policy.actor(self.policy.actor_obs_normalizer(inputs))
        delta = current - reference
        standardized = delta / self.reference_std
        return 0.5 * standardized.square().sum(-1).mean(), delta.abs().max(), standardized.abs().max()

    def actor_step(self, imagined, starts):
        # A learned reward/dynamics model is imperfect. Keep warm-start updates
        # inside a cumulative teacher trust radius measured on REAL observations.
        # The BPTT objective and its state/action Jacobians remain active.
        if not getattr(self, "warm_started", False):
            checked_step(imagined["actor_loss"], self.actor_optimizer, self.policy.actor.parameters())
            return {"actor_step_accepted": torch.ones((), device=self.device),
                    "actor_step_scale": torch.ones((), device=self.device)}
        real_inputs = self.policy.get_actor_obs(self.adapter.state_to_observations(starts)).detach()
        if self.trust_inputs is None:
            self.trust_inputs = real_inputs.clone()
        imagined_inputs = self.policy.get_actor_obs(self.adapter.state_to_observations(
            imagined["states"][:, :-1:4].flatten(0, 1))).detach()
        penalty_inputs = torch.cat((real_inputs, imagined_inputs), 0)
        kl, _, _ = self.teacher_statistics(penalty_inputs)
        loss = imagined["actor_loss"] + self.cfg.teacher_kl_coefficient * kl
        parameters = list(self.policy.actor.parameters())
        before = [parameter.detach().clone() for parameter in parameters]
        old_optimizer = copy.deepcopy(self.actor_optimizer.state_dict())
        checked_step(loss, self.actor_optimizer, parameters)
        proposal = [parameter.detach().clone() for parameter in parameters]
        guard_inputs = torch.cat((self.trust_inputs, real_inputs), 0)
        accepted, scale = False, 1.0
        with torch.no_grad():
            for _ in range(10):
                for parameter, old, new in zip(parameters, before, proposal):
                    parameter.copy_(old + scale * (new - old))
                real_kl, max_delta, max_standardized = self.teacher_statistics(guard_inputs)
                if (real_kl <= self.cfg.teacher_kl_limit and
                    max_delta <= self.cfg.teacher_max_action_delta and
                    max_standardized <= self.cfg.teacher_max_standardized_delta):
                    accepted = True
                    break
                scale *= 0.5
            anchor_projected = False
            if not accepted:
                # Newly visited real observations can expose a teacher-radius
                # violation in the PREVIOUS iterate. Backtracking only toward
                # that iterate cannot repair it; project the proposal toward
                # the fixed teacher, whose radius is exactly zero.
                anchor_projected = True
                reference = list(self.reference_actor.parameters())
                scale = 0.5
                for _ in range(14):
                    for parameter, anchor, new in zip(parameters, reference, proposal):
                        parameter.copy_(anchor + scale * (new - anchor))
                    real_kl, max_delta, max_standardized = self.teacher_statistics(guard_inputs)
                    if (real_kl <= self.cfg.teacher_kl_limit and
                        max_delta <= self.cfg.teacher_max_action_delta and
                        max_standardized <= self.cfg.teacher_max_standardized_delta):
                        accepted = True
                        break
                    scale *= 0.5
            if not accepted:
                for parameter, anchor in zip(parameters, self.reference_actor.parameters()):
                    parameter.copy_(anchor)
                self.actor_optimizer.load_state_dict(old_optimizer)
                scale = 0.0
                real_kl, max_delta, max_standardized = self.teacher_statistics(guard_inputs)
        return {"actor_anchor_projection": torch.tensor(float(anchor_projected), device=self.device),
                "actor_teacher_kl": real_kl.detach(), "actor_max_action_delta": max_delta.detach(),
                "actor_max_standardized_delta": max_standardized.detach(),
                "actor_step_accepted": torch.tensor(float(accepted), device=self.device),
                "actor_step_scale": torch.tensor(scale, device=self.device)}

    def fit_normalization(self):
        batch = self.sample(max(256, self.cfg.model_batch), self.cfg.model_window, self.train_env_ids)
        self.world.fit_normalization(batch["states"][:, :-1].flatten(0, 1),
            batch["actions"].flatten(0, 1), batch["context"].flatten(0, 1),
            next_states=batch["states"][:, 1:].flatten(0, 1),
            rewards=batch["rewards"].flatten(), valid=batch["valid"].flatten())
        self.normalization_fitted = True

    def update(self):
        synchronize(self.device)
        start = time.perf_counter()
        collected_reward = self.collect(self.cfg.collect_steps) if self.iteration % self.cfg.collect_interval == 0 else torch.zeros((), device=self.device)
        synchronize(self.device)
        after_collect = time.perf_counter()
        if not self.normalization_fitted:
            self.fit_normalization()
        if self.validation_batch is None:
            self.validation_batch = self.sample(self.cfg.model_batch, self.cfg.model_window,
                self.validation_env_ids, generator=self.validation_generator)
        diagnostics = {}
        for _ in range(self.cfg.model_updates):
            batch = self.sample(self.cfg.model_batch, self.cfg.model_window, self.train_env_ids)
            losses = world_model_loss(self.world, batch, project=self.adapter.project, horizon=self.cfg.model_window)
            checked_step(losses["loss"], self.world_optimizer, self.world.parameters())
            diagnostics = {"model_" + k: v.detach() for k, v in losses.items()}
        for _ in range(self.cfg.discriminator_updates):
            diagnostics["style_loss"] = self.style_update()
        synchronize(self.device)
        after_model = time.perf_counter()
        self.world_optimizer.zero_grad(set_to_none=True)
        self.discriminator_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        starts = self.sample(self.cfg.imagination_batch, 1, self.train_env_ids)
        imagined = imagine(self.policy, self.world, starts["states"][:, 0],
            self.adapter.state_to_observations, horizon=self.cfg.imagination_horizon,
            gamma=self.cfg.gamma, lam=self.cfg.lam, context=starts["context"][:, 0],
            project=self.adapter.project, reward_callback=self.style_reward,
            additional_frozen_modules=(self.discriminator,))
        diagnostics.update(self.actor_step(imagined, starts["states"][:, 0]))
        values = self.policy.evaluate(imagined["critic_observations"]).reshape_as(imagined["targets"])
        mask = imagined["mask"].to(values.dtype)
        critic_loss = imagined_critic_loss(values, imagined["targets"], mask.bool())
        checked_step(critic_loss, self.critic_optimizer, self.policy.critic.parameters())
        synchronize(self.device)
        end = time.perf_counter()
        self.iteration += 1
        if self.iteration == 1 or self.iteration % self.cfg.log_interval == 0:
            with torch.no_grad():
                validation = world_model_loss(self.world, self.validation_batch, project=self.adapter.project,
                                              horizon=self.cfg.model_window)
                diagnostics.update({"heldout_" + k: v.detach() for k, v in validation.items()})
        synchronize(self.device)
        validation_end = time.perf_counter()
        diagnostics.update(actor_loss=imagined["actor_loss"].detach(), critic_loss=critic_loss.detach(),
                           collected_step_reward=collected_reward.detach())
        values = torch.stack([torch.as_tensor(v, device=self.device) for v in diagnostics.values()]).cpu().tolist()
        record = dict(zip(diagnostics, values))
        record.update(iteration=self.iteration, real_env_steps=self.real_steps,
            collect_seconds=after_collect-start, model_seconds=after_model-after_collect,
            imagination_seconds=end-after_model, validation_seconds=validation_end-end,
            total_seconds=validation_end-start, timestamp=time.time(),
            imagined_actor_transitions=self.cfg.imagination_batch*self.cfg.imagination_horizon,
            model_fitting_transitions=self.cfg.model_updates*self.cfg.model_batch*self.cfg.model_window,
            actor_optimizer_steps=int(record.get("actor_step_accepted", 1)), critic_optimizer_steps=1,
            model_optimizer_steps=self.cfg.model_updates,
            discriminator_optimizer_steps=self.cfg.discriminator_updates)
        record["replay_host_bytes"] = self.replay.host_allocated_bytes
        record["replay_device_bytes"] = self.replay.device_allocated_bytes
        if torch.device(self.device).type == "cuda":
            record["peak_cuda_bytes"] = torch.cuda.max_memory_allocated(self.device)
        with (self.paths["log"] / "updates.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        (self.paths["data"] / "status.json").write_text(json.dumps(record, indent=2))
        return record

    def save(self, path):
        """Restore learning state; simulator continuation starts a new real rollout."""
        checkpoint = {"format": "husky-supermimic-v1", "model_state_dict": self.policy.state_dict(),
            "world": self.world.state_dict(), "discriminator": self.discriminator.state_dict(),
            "style_normalizer": self.style_norm.state_dict(), "config": asdict(self.cfg),
            "iteration": self.iteration, "real_steps": self.real_steps,
            "warm_started": getattr(self, "warm_started", False),
            "normalization_fitted": self.normalization_fitted,
            "optimizers": {name: getattr(self, name + "_optimizer").state_dict()
                           for name in ("world", "actor", "critic", "discriminator")},
            "replay": self.replay.state_dict(), "rng": rng_state(),
            "validation_batch": getattr(self, "validation_batch", None),
            "validation_generator": self.validation_generator.get_state() if hasattr(self, "validation_generator") else None,
            "expert_frames": self.expert_frames,
            "reference_actor": self.reference_actor.state_dict() if getattr(self, "reference_actor", None) is not None else None,
            "reference_normalizer": self.reference_normalizer.state_dict() if getattr(self, "reference_normalizer", None) is not None else None,
            "reference_std": getattr(self, "reference_std", None), "trust_inputs": getattr(self, "trust_inputs", None),
            "resume_scope": "Learning state restored; real simulator resets to fresh trajectories."}
        temporary = Path(path).with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)

    def load(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != "husky-supermimic-v1":
            raise ValueError("Legacy PPO checkpoint is a policy warm start, not a SuperMimic resume")
        saved_config, requested_config = dict(checkpoint["config"]), asdict(self.cfg)
        saved_config.pop("iterations")
        requested_config.pop("iterations")
        if saved_config != requested_config:
            raise ValueError("Resume learning/state config differs; explicit migration is required")
        if self.cfg.iterations < checkpoint["iteration"]:
            raise ValueError("Requested cap is below already completed iterations")
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.world.load_state_dict(checkpoint["world"])
        self.discriminator.load_state_dict(checkpoint["discriminator"])
        self.style_norm.load_state_dict(checkpoint["style_normalizer"])
        for name, state in checkpoint["optimizers"].items():
            getattr(self, name + "_optimizer").load_state_dict(state)
        self.replay.load_state_dict(checkpoint["replay"])
        self.expert_frames = checkpoint["expert_frames"].to(self.device)
        validation = checkpoint.get("validation_batch")
        self.validation_batch = {k: v.to(self.device) for k, v in validation.items()} if validation is not None else None
        self.validation_generator = torch.Generator(device=self.replay.device).manual_seed(self.cfg.seed + 104729)
        if checkpoint.get("validation_generator") is not None:
            self.validation_generator.set_state(checkpoint["validation_generator"].cpu())
        self.iteration, self.real_steps = checkpoint["iteration"], checkpoint["real_steps"]
        self.normalization_fitted = checkpoint["normalization_fitted"]
        restore_rng(checkpoint["rng"])
        self.warm_started = checkpoint.get("warm_started", False)
        if checkpoint.get("reference_actor") is not None:
            self.reference_actor = copy.deepcopy(self.policy.actor).eval().requires_grad_(False)
            self.reference_actor.load_state_dict(checkpoint["reference_actor"])
            self.reference_normalizer = copy.deepcopy(self.policy.actor_obs_normalizer).eval().requires_grad_(False)
            self.reference_normalizer.load_state_dict(checkpoint["reference_normalizer"])
            self.reference_std = checkpoint["reference_std"].to(self.device)
            self.trust_inputs = checkpoint["trust_inputs"].to(self.device) if checkpoint.get("trust_inputs") is not None else None
