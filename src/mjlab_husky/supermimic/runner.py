"""Real physics collection and differentiable learned-world-model optimization.

The simulator is used only under no_grad. Actor gradients pass through a fixed
learned model and fixed AMP/critic networks; they are surrogate gradients, not
derivatives of MuJoCo. Deployment retains the original HUSKY ActorCritic inputs.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from rsl_rl.modules import ActorCritic

from .adapter import SkateWorldModelAdapter, roll_history
from .config import SuperMimicConfig
from .models import WorldModel, frozen_parameters
from .replay import ReplayBuffer


def lambda_returns(rewards: Tensor, continuations: Tensor, values: Tensor,
                   gamma: float, gae_lambda: float) -> Tensor:
    """Detached [batch,time] targets; values includes the terminal bootstrap."""
    if rewards.shape != continuations.shape or values.shape != (
            rewards.shape[0], rewards.shape[1] + 1):
        raise ValueError("Lambda-return trajectory dimensions differ")
    with torch.no_grad():
        result = torch.empty_like(rewards)
        tail = values[:, -1]
        for step in reversed(range(rewards.shape[1])):
            tail = rewards[:, step] + gamma * continuations[:, step] * (
                (1 - gae_lambda) * values[:, step + 1] + gae_lambda * tail)
            result[:, step] = tail
    return result


class SuperMimicRunner:
    """RSL-style runner with independent model, actor, critic and AMP optimizers.

    ``adapter`` and ``amp`` injections allow simulator-free bounded smoke tests.
    A resume without saved replay retains weights/statistics and recollects the
    configured warmup before the next actor update. Simulator state is never
    restored from a training checkpoint.
    """

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None,
                 device="cpu", *, adapter=None, amp=None):
        self.env, self.cfg = env, train_cfg
        self.device = torch.device(device)
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise ValueError("SuperMimicRunner currently supports one training process/GPU")
        self.settings = SuperMimicConfig.from_dict(train_cfg.get("supermimic"))
        self.log_dir = str(log_dir) if log_dir is not None else None
        self.save_interval = int(train_cfg.get("save_interval", 500))
        if self.save_interval < 1:
            raise ValueError("save_interval must be positive")
        self.num_steps_per_env = self.settings.real_steps
        self.num_envs = int(env.num_envs)
        self.current_learning_iteration = 0
        self.tot_timesteps, self.tot_time = 0, 0.0
        self.actor_optimizer_updates = self.critic_optimizer_updates = 0
        self._warmup_collections = 0
        self.git_status_repos: list[str] = []
        self.last_metrics: dict = {}
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self._episode_reward = torch.zeros(self.num_envs, device=self.device)
        self._episode_length = torch.zeros(self.num_envs, device=self.device)
        self._rng = torch.Generator(device=self.device).manual_seed(int(train_cfg.get("seed", 42)))

        self._obs = self._observations(env.get_observations())
        groups = train_cfg.get("obs_groups", {"policy": ("policy",), "critic": ("critic",)})
        if groups != {"policy": ("policy",), "critic": ("critic",)} and groups != {
                "policy": ["policy"], "critic": ["critic"]}:
            raise ValueError("Skater adapter requires the original policy/critic observation groups")
        policy_cfg = dict(train_cfg["policy"])
        class_name = policy_cfg.pop("class_name", "ActorCritic")
        if class_name != "ActorCritic":
            raise ValueError("SuperMimicRunner requires the feed-forward HUSKY ActorCritic")
        self.policy = ActorCritic(self._obs, groups, env.num_actions, **policy_cfg).to(self.device)
        if self.settings.actor_mode == "deterministic" and self.policy.state_dependent_std:
            raise ValueError("Deterministic SuperMimic requires the original state-independent std checkpoint schema")
        self.alg = SimpleNamespace(policy=self.policy)  # Original inference/export tools expect this attribute.
        if amp is None:
            from .amp import AmpLearner
            amp = AmpLearner(env, train_cfg, self.device)
        self.amp = amp
        self._amp_reward_stats = None
        self.discriminator = amp.discriminator
        self.amp_normalizer = amp.normalizer
        self.amp_num_frames = int(amp.num_frames)
        self.amp_observation_dim = int(amp.observation_dim)
        current_amp = env.get_amp_observations().to(self.device).detach()
        self.amp_obs_frames = torch.zeros((self.num_envs, self.amp_num_frames,
                                          self.amp_observation_dim), device=self.device)
        self.amp_obs_frames = roll_history(self.amp_obs_frames, current_amp)
        self.adapter = adapter or SkateWorldModelAdapter(env, self._obs, self.amp_obs_frames)
        if self.adapter.action_dim != env.num_actions:
            raise ValueError("Adapter/action dimensions differ")
        self._critic_transition_target_slices = self._find_critic_transition_target_slices()
        self.critic_transition_target_dim = sum(end - start for start, end in self._critic_transition_target_slices)
        self._project_next = self.adapter.project_next
        if self.settings.compile_projection and self.device.type == "cuda":
            # Fuse the many small history/kinematics kernels, leaving the
            # learned dynamics and all gradients numerically unchanged.
            self._project_next = torch.compile(self._project_next, fullgraph=True, dynamic=False)
        self._state = self.adapter.capture(self._obs, self.amp_obs_frames).to(self.device)
        self.world_model = WorldModel(
            self.adapter.state_dim, self.adapter.action_dim,
            hidden_dims=self.settings.model_hidden_dims,
            learning_rate=self.settings.model_learning_rate, device=self.device,
            gradient_clip=self.settings.max_grad_norm, project_next=self._project_next)
        if getattr(self.adapter, "supports_exogenous_projection", False):
            self.world_model.project_training_next = lambda previous, predicted, action, target: (
                self._project_next(previous, predicted, action,
                                          exogenous={"next_state": target}))
        self.replay = ReplayBuffer(
            self.settings.replay_capacity_steps, self.num_envs, self.adapter.state_dim,
            self.adapter.action_dim, self.device, seed=int(train_cfg.get("seed", 42)) + 1)
        self._actor_parameters = list(self.policy.actor.parameters())
        for name in ("std", "log_std"):
            noise_parameter = getattr(self.policy, name, None)
            if noise_parameter is not None:
                # Keep the original checkpoint/deployment schema. Deterministic
                # SuperMimic uses separate fixed collection noise, not this std.
                noise_parameter.requires_grad_(self.settings.actor_mode == "reparameterized")
                if noise_parameter.requires_grad:
                    self._actor_parameters.append(noise_parameter)
        self.actor_optimizer = torch.optim.Adam(self._actor_parameters, lr=self.settings.actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.policy.critic.parameters(), lr=self.settings.critic_learning_rate)
        self._bound_std()
        if self.log_dir is not None:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            (Path(self.log_dir) / "supermimic_config.json").write_text(
                json.dumps({"supermimic": asdict(self.settings),
                            "adapter": self.adapter.metadata()}, indent=2), encoding="utf-8")

    def _observations(self, observations: Mapping) -> dict[str, Tensor]:
        result = {key: value.to(self.device).detach().clone() for key, value in observations.items()}
        for key, value in result.items():
            self._finite(f"real observation {key}", value)
        return result

    @staticmethod
    def _finite(name: str, value: Tensor) -> None:
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite {name}; stopping before checkpointing invalid parameters")

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _clock(self):
        self._sync()
        return time.perf_counter()

    def _state_obs(self, state: Tensor) -> dict[str, Tensor]:
        return {"policy": self.adapter.actor_obs(state), "critic": self.adapter.critic_obs(state)}

    def _find_critic_transition_target_slices(self):
        """Resolve raw critic coordinates from task metadata, never fixed offsets."""
        wanted = {"trans_target_pos_b", "trans_target_quat_b"}
        group = getattr(self.adapter, "critic_group", "critic")
        terms = getattr(self.adapter, "_obs_terms", {}).get(group, ())
        selected = [term for term in terms if term.name in wanted]
        if selected and {term.name for term in selected} != wanted:
            raise ValueError("Critic transition-target metadata must contain both position and quaternion terms")
        ranges = tuple(sorted((term.start, term.stop) for term in selected))
        previous_end = 0
        for start, end in ranges:
            if start < previous_end or not start < end <= self._obs["critic"].shape[-1]:
                raise ValueError("Critic transition-target slices overlap or exceed the original observation")
            previous_end = end
        return ranges

    def _actor_bootstrap_obs(self, state: Tensor) -> dict[str, Tensor]:
        """Cut only learned transition targets in the actor's terminal value path.

        Forward values and all other critic-input derivatives remain unchanged.
        Critic fitting and model supervision continue using the full observations.
        """
        observations = self._state_obs(state)
        if not self.settings.detach_critic_transition_targets or not self._critic_transition_target_slices:
            return observations
        raw = observations["critic"]
        pieces, offset = [], 0
        for start, end in self._critic_transition_target_slices:
            pieces.extend((raw[..., offset:start], raw[..., start:end].detach()))
            offset = end
        pieces.append(raw[..., offset:])
        observations["critic"] = torch.cat(pieces, -1)
        return observations

    def _action(self, observations: Mapping, stochastic: bool = True, *, imagination: bool = False):
        actor_obs = self.policy.actor_obs_normalizer(self.policy.get_actor_obs(observations))
        if self.settings.actor_mode == "deterministic":
            mean = self.policy.actor(actor_obs)
            if stochastic and not imagination:
                mean = mean + self.settings.real_exploration_std * torch.randn(
                    mean.shape, device=self.device, generator=self._rng)
            clip = getattr(self.env, "clip_actions", self.cfg.get("clip_actions"))
            action = mean if clip is None else mean.clamp(-clip, clip)
            return action, mean.new_zeros(mean.shape[:-1])
        # Retain trial one's Gaussian path, learned std, RNG draws and entropy.
        self.policy.update_distribution(actor_obs)
        mean = self.policy.action_mean
        std = self.policy.action_std.clamp(self.settings.min_noise_std, self.settings.max_noise_std)
        if stochastic:
            mean = mean + std * torch.randn(mean.shape, device=self.device, generator=self._rng)
        clip = getattr(self.env, "clip_actions", self.cfg.get("clip_actions"))
        action = mean if clip is None else mean.clamp(-clip, clip)
        entropy = (std.log() + 0.5 * math.log(2 * math.pi * math.e)).sum(-1)
        return action, entropy

    @torch.no_grad()
    def _bound_std(self):
        if self.settings.actor_mode == "deterministic":
            return
        if hasattr(self.policy, "std"):
            self.policy.std.clamp_(self.settings.min_noise_std, self.settings.max_noise_std)
        elif hasattr(self.policy, "log_std"):
            self.policy.log_std.clamp_(math.log(self.settings.min_noise_std),
                                       math.log(self.settings.max_noise_std))

    def _split_done(self, dones: Tensor, extras: Mapping):
        done = dones.to(self.device).bool().reshape(-1)
        raw = getattr(self.env, "unwrapped", self.env)
        manager = getattr(raw, "termination_manager", None)
        timeouts = extras.get("time_outs")
        if timeouts is None:
            timeouts = getattr(raw, "reset_time_outs", None)
        if timeouts is None and manager is not None:
            timeouts = getattr(manager, "time_outs", None)
        if timeouts is None:
            raise RuntimeError("Environment must expose timeout flags; treating every done as failure is invalid")
        truncated = torch.as_tensor(timeouts, device=self.device).bool().reshape(-1) & done
        terminated = getattr(raw, "reset_terminated", None)
        if terminated is None and manager is not None:
            terminated = getattr(manager, "terminated", None)
        terminated = done & ~truncated if terminated is None else (
            torch.as_tensor(terminated, device=self.device).bool().reshape(-1) & done)
        return terminated, truncated, done

    @torch.no_grad()
    def collect(self) -> dict[str, float]:
        """Collect fresh physics transitions; only this method updates WM/actor stats."""
        self.policy.train()
        task_rewards, total_rewards, amp_histories = [], [], []
        failures = timeouts = 0
        prediction_error = 0.0
        validation_samples = 0
        position_errors: dict[str, float] = {}
        for _ in range(self.settings.real_steps):
            self.policy.update_normalization(self._obs)
            action, _ = self._action(self._obs)
            self._finite("real action", action)
            next_obs, task_reward, dones, extras = self.env.step(action.to(self.env.device))
            next_obs = self._observations(next_obs)
            task_reward = task_reward.to(self.device).reshape(-1).clone()
            self._finite("real task reward", task_reward)
            terminated, truncated, done = self._split_done(dones, extras)
            next_amp = self.env.get_amp_observations().to(self.device).detach().clone()
            self._finite("real AMP joints", next_amp)
            next_frames = roll_history(self.amp_obs_frames, next_amp)
            next_state = self.adapter.capture(next_obs, next_frames).to(self.device)
            valid = ~done  # The wrapper already returned reset observations on done.
            # Prequential validation: predict a fresh real target before it is used
            # for fitting or normalizer updates. Post-reset targets never enter it.
            if valid.any():
                prediction = self.world_model(self._state, action)
                if getattr(self.adapter, "supports_exogenous_projection", False):
                    predicted = self._project_next(
                        self._state, prediction.next_state, action,
                        exogenous={"next_state": next_state})
                else:
                    predicted = self._project_next(self._state, prediction.next_state, action)
                error = (predicted[valid] - next_state[valid]) / self.world_model.delta_norm.scale(next_state)
                self._finite("fresh-transition model error", error)
                prediction_error += float(error.square().mean(-1).sum())
                validation_samples += int(valid.sum())
                for key, label in (("robot.root_link_pos_w", "robot"),
                                   ("skateboard.root_link_pos_w", "board")):
                    if key in self.adapter.layout:
                        absolute_error = (self.adapter.layout.get(predicted, key)[valid]
                                          - self.adapter.layout.get(next_state, key)[valid]).abs()
                        position_errors[label] = position_errors.get(label, 0.0) + float(
                            absolute_error.reshape(int(valid.sum()), -1).mean(-1).sum())
            self.world_model.update_normalization(self._state, action, next_state, task_reward, valid)
            self.replay.append(self._state, action, next_state, task_reward,
                               terminated, truncated, next_state_valid=valid)
            # Reward uses the original post-step history and push gate, before reset history clearing.
            gate = self.adapter.layout.get(next_state, "amp_gate")
            real_reward = self.amp.reward(task_reward, next_frames, gate, differentiable=False)
            self._finite("real AMP reward", real_reward)
            if valid.any():
                amp_histories.append(next_frames[valid].clone())
            self._episode_reward += real_reward
            self._episode_length += 1
            if done.any():
                self.episode_rewards.extend(self._episode_reward[done].cpu().tolist())
                self.episode_lengths.extend(self._episode_length[done].cpu().tolist())
                self._episode_reward[done] = self._episode_length[done] = 0
            failures += int(terminated.sum())
            timeouts += int(truncated.sum())
            task_rewards.append(task_reward)
            total_rewards.append(real_reward)
            self.amp_obs_frames = next_frames
            self.amp_obs_frames[done] = 0
            self._obs = next_obs
            # The next current state must contain the same cleared history as its real AMP input.
            self._state = self.adapter.capture(next_obs, self.amp_obs_frames).to(self.device)
        self._amp_real_histories = (torch.cat(amp_histories) if amp_histories else None)
        self._warmup_collections += 1
        samples = self.num_envs * self.settings.real_steps
        self.tot_timesteps += samples
        return {"real/task_reward": float(torch.cat(task_rewards).mean()),
                "real/reward": float(torch.cat(total_rewards).mean()),
                "real/failure_rate": failures / samples, "real/timeout_rate": timeouts / samples,
                "real/episode_return": sum(self.episode_rewards) / max(1, len(self.episode_rewards)),
                "real/episode_length": sum(self.episode_lengths) / max(1, len(self.episode_lengths)),
                "real/completed_episodes": len(self.episode_rewards),
                "validation/one_step_normalized_rmse": math.sqrt(prediction_error / max(1, validation_samples)),
                "validation/one_step_samples": validation_samples,
                **{f"validation/{name}_position_mae_m": error / max(1, validation_samples)
                   for name, error in position_errors.items()}}

    def fit_world_model(self):
        totals: dict[str, float] = {}
        for _ in range(self.settings.model_updates):
            batch = self.replay.sample_sequences(self.settings.model_batch_size,
                                                 self.settings.model_sequence_length)
            result = self.world_model.fit(batch, self.settings.model_sequence_length)
            for key, value in result.items():
                totals[key] = totals.get(key, 0.0) + value / self.settings.model_updates
        return {f"model/{key}": value for key, value in totals.items()}

    def _imagined_reward(self, state: Tensor, task_reward: Tensor):
        normalizer = self.amp.normalizer
        if self._amp_reward_stats is None:
            self._cache_amp_reward_statistics()
        mean, std = self._amp_reward_stats
        disc = self.amp.discriminator
        return self.adapter.amp_reward(
            state, task_reward, disc, mean, std,
            amp_reward_coef=disc.amp_reward_coef, task_reward_lerp=disc.task_reward_lerp,
            use_lerp=disc.use_lerp, clip_obs=normalizer.clip_obs,
            reward_dt=0.02) * self.settings.reward_scale

    def _cache_amp_reward_statistics(self):
        # Statistics stay fixed across all H steps: copy host AMP stats once,
        # avoiding repeated tiny CPU-to-GPU transfers inside the BPTT loop.
        normalizer = self.amp.normalizer
        mean = torch.as_tensor(normalizer.mean, device=self.device, dtype=torch.float32).detach()
        variance = torch.as_tensor(normalizer.var, device=self.device, dtype=torch.float32).detach()
        self._amp_reward_stats = (mean, (variance + normalizer.epsilon).sqrt())

    def _imagine(self):
        """Keep all H state/action edges attached for the actor's BPTT objective."""
        state = self.replay.sample_states(self.settings.actor_batch_size).detach()
        states, rewards, continuations, weights = [], [], [], []
        survival = torch.ones(len(state), device=self.device)
        objective = torch.zeros_like(survival)
        entropy_total = torch.zeros_like(survival)
        # Command changes and sensor noise are external to the policy. Sample
        # them independently before BPTT so actions cannot control their values.
        exogenous_projection = getattr(self.adapter, "supports_exogenous_projection", False)
        if exogenous_projection:
            command_uniforms = torch.rand(
                self.settings.horizon, len(state), self.adapter.command_noise_dim,
                device=self.device, generator=self._rng)
            sensor_uniforms = torch.rand(
                self.settings.horizon, len(state), self.adapter.sensor_noise_dim,
                device=self.device, generator=self._rng)
        for step in range(self.settings.horizon):
            states.append(state.detach())
            action, entropy = self._action(self._state_obs(state), imagination=True)
            prediction = self.world_model(state, action)
            if exogenous_projection:
                next_state = self._project_next(
                    state, prediction.next_state, action,
                    exogenous={"command_uniforms": command_uniforms[step],
                               "sensor_uniforms": sensor_uniforms[step]})
            else:
                next_state = self._project_next(state, prediction.next_state, action)
            reward = self._imagined_reward(next_state, prediction.reward)
            continuation = prediction.continuation
            weight = self.settings.gamma ** step * survival
            objective = objective + weight * reward
            if self.settings.actor_mode == "reparameterized":
                entropy_total = entropy_total + weight * entropy
            rewards.append(reward.detach())
            continuations.append(continuation.detach())
            weights.append(survival.detach())
            survival = survival * continuation
            state = next_state
        terminal = self.policy.evaluate(self._actor_bootstrap_obs(state)).squeeze(-1)
        objective = objective + self.settings.gamma ** self.settings.horizon * survival * terminal
        states.append(state.detach())
        actor_loss = -objective.mean()
        if self.settings.actor_mode == "reparameterized":
            actor_loss = actor_loss - self.settings.entropy_coef * entropy_total.mean()
        return actor_loss, {
            "state": torch.stack(states, 1), "reward": torch.stack(rewards, 1),
            "continuation": torch.stack(continuations, 1), "weight": torch.stack(weights, 1),
        }, {"actor/objective": float(objective.detach().mean()),
            "actor/imagined_reward": float(torch.stack(rewards, 1).mean()),
            "actor/survival_at_horizon": float(survival.detach().mean()),
            "actor/detached_critic_transition_target_dim": (self.critic_transition_target_dim
                if self.settings.detach_critic_transition_targets else 0),
            "actor/effective_entropy_coef": (self.settings.entropy_coef
                                              if self.settings.actor_mode == "reparameterized" else 0.0)}

    def update_actor_critic(self):
        metrics: dict[str, float] = {}
        traces = []
        self._cache_amp_reward_statistics()
        self.world_model.eval()
        for _ in range(self.settings.actor_updates):
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.world_model.optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            self.amp.optimizer.zero_grad(set_to_none=True)
            with frozen_parameters(self.world_model, self.amp.discriminator, self.policy.critic):
                actor_loss, trace, current = self._imagine()
                self._finite("actor loss", actor_loss)
                actor_loss.backward()
            norm = clip_grad_norm_(self._actor_parameters, self.settings.max_grad_norm,
                                   error_if_nonfinite=True)
            self.actor_optimizer.step()
            self._bound_std()
            self.actor_optimizer_updates += 1
            traces.append(trace)
            current.update({"actor/loss": float(actor_loss.detach()), "actor/gradient_norm": float(norm)})
            for key, value in current.items():
                metrics[key] = metrics.get(key, 0.0) + value / self.settings.actor_updates
        states = torch.cat([trace["state"] for trace in traces])
        rewards = torch.cat([trace["reward"] for trace in traces])
        continuations = torch.cat([trace["continuation"] for trace in traces])
        weights = torch.cat([trace["weight"] for trace in traces]).flatten()
        batch, steps, dim = states.shape
        with torch.no_grad():
            values = self.policy.evaluate(self._state_obs(states.reshape(-1, dim))).reshape(batch, steps)
            targets = lambda_returns(rewards, continuations, values,
                                     self.settings.gamma, self.settings.gae_lambda).flatten()
        inputs = states[:, :-1].reshape(-1, dim)
        for _ in range(self.settings.critic_updates):
            self.critic_optimizer.zero_grad(set_to_none=True)
            prediction = self.policy.evaluate(self._state_obs(inputs)).squeeze(-1)
            loss = (weights * (prediction - targets).square()).sum() / weights.sum().clamp_min(1)
            self._finite("critic loss", loss)
            loss.backward()
            norm = clip_grad_norm_(self.policy.critic.parameters(), self.settings.max_grad_norm,
                                   error_if_nonfinite=True)
            self.critic_optimizer.step()
            self.critic_optimizer_updates += 1
            metrics["critic/loss"] = metrics.get("critic/loss", 0.0) + float(loss.detach()) / self.settings.critic_updates
            metrics["critic/gradient_norm"] = metrics.get("critic/gradient_norm", 0.0) + float(norm) / self.settings.critic_updates
        self.world_model.train()
        return metrics

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        if num_learning_iterations < 0:
            raise ValueError("Learning iteration count cannot be negative")
        if init_at_random_ep_len:
            with torch.no_grad():
                self.env.episode_length_buf = torch.randint_like(
                    self.env.episode_length_buf, high=int(self.env.max_episode_length))
                self._state = self.adapter.capture(self._obs, self.amp_obs_frames).to(self.device)
        for _ in range(num_learning_iterations):
            start = self._clock()
            metrics = self.collect()
            collected = self._clock()
            metrics.update(self.fit_world_model())
            model_fitted = self._clock()
            if self._amp_real_histories is not None:
                amp_metrics = self.amp.update(self._amp_real_histories,
                                               self.settings.amp_updates, self.settings.amp_batch_size)
                metrics.update({f"amp/{key}": value for key, value in amp_metrics.items()})
            amp_fitted = self._clock()
            warming = self._warmup_collections <= self.settings.warmup_iterations
            if not warming:
                metrics.update(self.update_actor_critic())
            finished = self._clock()
            self.current_learning_iteration += 1
            elapsed = finished - start
            self.tot_time += elapsed
            metrics.update({"iteration": self.current_learning_iteration, "total_real_steps": self.tot_timesteps,
                            "policy/actor_mode": self.settings.actor_mode,
                            "policy/real_exploration_std": (self.settings.real_exploration_std
                                if self.settings.actor_mode == "deterministic"
                                else float(self.policy.action_std.detach().mean())),
                            "policy/imagined_stochastic_actions": self.settings.actor_mode == "reparameterized",
                            "warmup": warming, "replay_transitions": len(self.replay),
                            "time/collect": collected - start, "time/model": model_fitted - collected,
                            "time/amp": amp_fitted - model_fitted, "time/actor_critic": finished - amp_fitted,
                            "time/iteration": elapsed,
                            "seconds": elapsed,
                            "performance/real_samples_per_second": self.num_envs * self.settings.real_steps / max(elapsed, 1e-9),
                            "updates/actor": self.actor_optimizer_updates,
                            "updates/critic": self.critic_optimizer_updates,
                            "updates/world_model": self.world_model.optimizer_updates})
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not math.isfinite(value):
                    raise FloatingPointError(f"Nonfinite metric {key}")
            self.last_metrics = metrics
            if self.log_dir is not None:
                with (Path(self.log_dir) / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(metrics, allow_nan=False) + "\n")
            if self.current_learning_iteration % self.settings.log_interval == 0:
                print(f"SuperMimic {self.current_learning_iteration}: real={metrics['real/reward']:.4f} "
                      f"model={metrics['model/loss']:.4f} actor={metrics.get('actor/loss', 0):.4f} "
                      f"time={elapsed:.3f}s warmup={warming}", flush=True)
            if self.log_dir and self.current_learning_iteration % self.save_interval == 0:
                self.save(str(Path(self.log_dir) / f"model_{self.current_learning_iteration}.pt"))
        if self.log_dir:
            self.save(str(Path(self.log_dir) / f"model_{self.current_learning_iteration}.pt"))

    def save(self, path: str, infos=None):
        for label, module in (("policy", self.policy), ("world model", self.world_model),
                              ("AMP", self.amp.discriminator)):
            for name, value in module.state_dict().items():
                self._finite(f"{label} parameter {name}", value)
        payload = {"version": 1, "config": asdict(self.settings),
                   "adapter": self.adapter.metadata(), "world_model": self.world_model.training_state_dict(),
                   "amp": self.amp.state_dict(), "actor_optimizer": self.actor_optimizer.state_dict(),
                   "critic_optimizer": self.critic_optimizer.state_dict(),
                   "actor_optimizer_updates": self.actor_optimizer_updates,
                   "critic_optimizer_updates": self.critic_optimizer_updates,
                   "total_real_steps": self.tot_timesteps, "total_time": self.tot_time,
                   "warmup_collections": self._warmup_collections, "rng": self._rng.get_state(),
                   "git_status_repos": self.git_status_repos}
        if self.settings.checkpoint_replay:
            payload["replay"] = self.replay.state_dict()
        checkpoint = {"model_state_dict": self.policy.state_dict(),
                      "iter": self.current_learning_iteration, "infos": infos,
                      "supermimic": payload}
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(target)

    def load(self, path: str, load_optimizer: bool = True, map_location=None):
        checkpoint = torch.load(path, map_location=map_location or self.device, weights_only=False)
        payload = checkpoint.get("supermimic")
        if payload is None and load_optimizer:
            raise ValueError("Cannot resume PPO as SuperMimic: no learned-model/AMP training state. "
                             "Use load_optimizer=False explicitly to initialize only policy weights.")
        # Validate identity before touching any live parameters or optimizer state.
        def validate_weights(saved, current, label):
            if set(saved) != set(current):
                raise ValueError(f"Checkpoint {label} parameter keys differ")
            for name, value in saved.items():
                if not isinstance(value, Tensor) or value.shape != current[name].shape:
                    raise ValueError(f"Checkpoint {label} shape differs for {name}")
                self._finite(f"checkpoint {label} {name}", value)

        validate_weights(checkpoint["model_state_dict"], self.policy.state_dict(), "policy")
        if payload is not None:
            required = {"version", "config", "adapter", "world_model", "amp", "actor_optimizer",
                        "critic_optimizer", "actor_optimizer_updates", "critic_optimizer_updates",
                        "total_real_steps", "total_time", "warmup_collections", "rng"}
            if not required <= set(payload) or payload["version"] != 1:
                raise ValueError("Unsupported or incomplete SuperMimic checkpoint schema")
            saved_config = SuperMimicConfig.from_checkpoint_dict(payload["config"])
            if load_optimizer and saved_config.actor_mode != self.settings.actor_mode:
                description = ("legacy reparameterized checkpoint (no actor_mode)"
                               if "actor_mode" not in payload["config"] else saved_config.actor_mode)
                raise ValueError(f"Cannot resume {description} into actor_mode={self.settings.actor_mode}. "
                                 "Use load_optimizer=False explicitly to initialize weights without optimizer state.")
            if (load_optimizer and saved_config.detach_critic_transition_targets
                    != self.settings.detach_critic_transition_targets):
                description = ("legacy checkpoint without a detach_critic_transition_targets flag (interpreted as False)"
                               if "detach_critic_transition_targets" not in payload["config"] else
                               f"checkpoint detach_critic_transition_targets={saved_config.detach_critic_transition_targets}")
                raise ValueError(f"Cannot resume {description} into "
                                 f"detach_critic_transition_targets={self.settings.detach_critic_transition_targets}. "
                                 "Use load_optimizer=False explicitly to initialize weights for the changed objective.")
            if load_optimizer and asdict(saved_config) != asdict(self.settings):
                raise ValueError("SuperMimic resume configuration differs from the checkpoint")
            if payload["adapter"] != self.adapter.metadata():
                raise ValueError("Checkpoint training-state layout differs from this environment")
            model = payload["world_model"]
            if (model["state_dim"], model["action_dim"], tuple(model["hidden_dims"])) != (
                    self.adapter.state_dim, self.adapter.action_dim, tuple(self.settings.model_hidden_dims)):
                raise ValueError("Checkpoint world-model architecture differs")
            validate_weights(model["model"], self.world_model.state_dict(), "world model")
            amp_state = payload["amp"]
            validate_weights(amp_state["discriminator"], self.amp.discriminator.state_dict(), "AMP")
            if hasattr(self.amp, "expert_config") and (
                    amp_state["expert_config"] != self.amp.expert_config
                    or amp_state["observation_dim"] != self.amp.observation_dim):
                raise ValueError("Checkpoint AMP expert configuration differs")
            if "replay" in payload:
                replay = payload["replay"]
                for name in ("capacity", "num_envs", "state_dim", "action_dim"):
                    if replay[name] != getattr(self.replay, name):
                        raise ValueError(f"Checkpoint replay {name} differs")
                retained = min(replay["steps"], self.replay.capacity)
                if set(replay["data"]) != set(self.replay.data) or any(
                        value.shape != self.replay.data[name][:retained].shape
                        for name, value in replay["data"].items()):
                    raise ValueError("Checkpoint replay tensor dimensions differ")
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.current_learning_iteration = int(checkpoint.get("iter", 0))
        if payload is not None:
            if load_optimizer:
                self.world_model.load_training_state_dict(payload["world_model"])
            else:
                self.world_model.load_state_dict(payload["world_model"]["model"])
            initial_amp_optimizer = self.amp.optimizer.state_dict()
            self.amp.load_state_dict(payload["amp"])
            if not load_optimizer:
                self.amp.optimizer.load_state_dict(initial_amp_optimizer)
            if load_optimizer:
                self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
                self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
                self.actor_optimizer_updates = int(payload["actor_optimizer_updates"])
                self.critic_optimizer_updates = int(payload["critic_optimizer_updates"])
            self.tot_timesteps = int(payload["total_real_steps"])
            self.tot_time = float(payload["total_time"])
            self._rng.set_state(payload["rng"].cpu())
            if "replay" in payload:
                self.replay.load_state_dict(payload["replay"])
                self._warmup_collections = int(payload["warmup_collections"])
                if len(self.replay):
                    # Physics restarts from reset rather than the saved simulator
                    # state. End old collection windows without labelling failure.
                    last_slot = (self.replay.steps - 1) % self.replay.capacity
                    self.replay.data["truncated"][last_slot] = True
            else:
                self._warmup_collections = 0
        self._bound_std()
        self._amp_reward_stats = None
        return checkpoint.get("infos")

    def get_inference_policy(self, device=None):
        self.eval_mode()
        if device is not None:
            self.policy.to(device)
        return self.policy.act_inference

    def train_mode(self):
        self.policy.train()
        self.world_model.train()

    def eval_mode(self):
        self.policy.eval()
        self.world_model.eval()

    def add_git_repo_to_log(self, repo_file_path: str):
        if repo_file_path not in self.git_status_repos:
            self.git_status_repos.append(repo_file_path)
