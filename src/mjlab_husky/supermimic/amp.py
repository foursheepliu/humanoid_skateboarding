"""HUSKY's original AMP discriminator with differentiable imagined rewards.

Expert loading, LSGAN targets, spectral normalization, gradient penalty and
reward scaling follow the bundled AMP implementation. Only real policy
histories may be passed to ``update``; imagined states never update AMP stats.
"""

from contextlib import nullcontext

import numpy as np
import torch
from torch import Tensor

from rsl_rl.modules import DiscriminatorMulti
from rsl_rl.utils import G1_AMPLoader, Normalizer


class AmpLearner:
    def __init__(self, env, agent_cfg: dict, device="cpu"):
        self.env, self.device = env, torch.device(device)
        self.num_frames = int(agent_cfg.get("amp_num_frames", 5))
        if self.num_frames < 1:
            raise ValueError("AMP requires at least one history frame")
        self.expert_config = {
            "motion_files": agent_cfg.get("amp_motion_files", "dataset/skate_push"),
            "num_preload_transitions": int(agent_cfg.get("amp_num_preload_transitions", 200_000)),
            "num_frames": self.num_frames,
            "time_between_frames": 1 / 50.0,
        }
        if self.expert_config["num_preload_transitions"] < 1:
            raise ValueError("AMP expert preload size must be positive")
        self.loader = G1_AMPLoader(
            device=self.device, preload_transitions=True, **self.expert_config
        )
        # The legacy loader's observation_dim describes an unprocessed motion
        # frame. The actual 23-DOF expert generator slices a different layout.
        self.observation_dim = self.loader.preloaded_frames[0].shape[-1]
        configured_dim = int(agent_cfg.get("amp_num_obs", self.observation_dim))
        if configured_dim not in (0, self.observation_dim):
            raise ValueError(
                f"AMP config expects {configured_dim} values, expert frames have "
                f"{self.observation_dim}"
            )
        self.normalizer = Normalizer(self.observation_dim)
        self.discriminator = DiscriminatorMulti(
            self.observation_dim,
            float(agent_cfg.get("amp_reward_coef", 5.0)),
            agent_cfg.get("amp_discr_hidden_dims", (256, 256)),
            self.device,
            self.num_frames,
            float(agent_cfg.get("amp_task_reward_lerp", 0.7)),
            bool(agent_cfg.get("use_lerp", False)),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.discriminator.trunk.parameters(), "weight_decay": 1e-3},
                {"params": self.discriminator.amp_linear.parameters(), "weight_decay": 1e-1},
            ],
            lr=float(agent_cfg.get("algorithm", {}).get("learning_rate", 1e-3)),
        )
        self.optimizer_updates = 0

    def _check_frames(self, frames: Tensor) -> None:
        if frames.ndim != 3 or tuple(frames.shape[1:]) != (
            self.num_frames, self.observation_dim
        ):
            raise ValueError(
                f"AMP histories must have shape [batch, {self.num_frames}, "
                f"{self.observation_dim}], got {tuple(frames.shape)}"
            )

    def update(self, policy_frames: Tensor, steps: int = 4,
               batch_size: int = 256) -> dict[str, float]:
        """Fit AMP on real history samples and the original expert distribution.

        The original PPO uses all sampled policy histories for discrimination;
        the push-phase mask is applied to rewards, rather than filtering AMP
        replay. ``policy_frames`` must therefore be real, unnormalized histories.
        """
        self._check_frames(policy_frames)
        if steps < 1 or batch_size < 1 or len(policy_frames) < 1:
            raise ValueError("AMP update needs nonempty real samples and positive sizes")
        policy_frames = policy_frames.detach().to(self.device)
        self.discriminator.train()
        totals = dict(amp_loss=0.0, amp_grad_penalty=0.0,
                      amp_policy_score=0.0, amp_expert_score=0.0)
        expert_batches = self.loader.feed_forward_generator_23dof_multi(steps, batch_size)
        for expert_raw in expert_batches:
            indices = torch.randint(len(policy_frames), (batch_size,), device=self.device)
            policy_raw = policy_frames[indices]
            with torch.no_grad():
                policy = self.normalizer.normalize_torch(policy_raw, self.device)
                expert = self.normalizer.normalize_torch(expert_raw.to(self.device), self.device)
            policy_score = self.discriminator(policy.flatten(1))
            expert_score = self.discriminator(expert.flatten(1))
            expert_loss = (expert_score - 1).square().mean()
            policy_loss = (policy_score + 1).square().mean()
            amp_loss = 0.5 * (expert_loss + policy_loss)
            penalty = self.discriminator.compute_grad_pen(expert, lambda_=5)
            self.optimizer.zero_grad(set_to_none=True)
            (amp_loss + penalty).backward()
            self.optimizer.step()
            # Match the bundled AMP_PPO normalizer update, including its use of
            # normalized minibatches. This preserves the baseline's AMP scale.
            self.normalizer.update(policy.detach().cpu().numpy())
            self.normalizer.update(expert.detach().cpu().numpy())
            totals["amp_loss"] += float(amp_loss.detach())
            totals["amp_grad_penalty"] += float(penalty.detach())
            totals["amp_policy_score"] += float(policy_score.detach().mean())
            totals["amp_expert_score"] += float(expert_score.detach().mean())
            self.optimizer_updates += 1
        return {name: value / steps for name, value in totals.items()}

    def reward(self, task_reward: Tensor, frames: Tensor, push_gate: Tensor,
               differentiable: bool = True) -> Tensor:
        """Merge task and AMP rewards exactly as in the original push-phase gate.

        Discriminator parameters should be frozen by the caller during actor
        optimization. Eval mode also holds spectral-normalization buffers fixed,
        while gradients through the input histories remain available.
        """
        self._check_frames(frames)
        if task_reward.numel() != len(frames) or push_gate.numel() != len(frames):
            raise ValueError("AMP rewards, gates and histories need equal batch sizes")
        was_training = self.discriminator.training
        self.discriminator.eval()
        try:
            with nullcontext() if differentiable else torch.no_grad():
                states = self.normalizer.normalize_torch(frames, self.device)
                score = self.discriminator(states.flatten(1)).squeeze(-1)
                disc_reward = self.discriminator.amp_reward_coef * (
                    1 - 0.25 * (score - 1).square()
                ).clamp_min(0)
                task = task_reward.reshape(-1)
                if self.discriminator.use_lerp:
                    ratio = self.discriminator.task_reward_lerp
                    combined = (1 - ratio) * disc_reward + ratio * task
                else:
                    combined = task + 0.02 * disc_reward
                result = torch.where(push_gate.reshape(-1).bool(), combined, task)
                return result.reshape(task_reward.shape)
        finally:
            self.discriminator.train(was_training)

    def state_dict(self) -> dict:
        """Checkpoint trainable AMP state; immutable expert data is reloaded."""
        return {
            "discriminator": self.discriminator.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "normalizer": {
                "mean": torch.as_tensor(np.array(self.normalizer.mean, copy=True)),
                "var": torch.as_tensor(np.array(self.normalizer.var, copy=True)),
                "count": float(self.normalizer.count),
                "epsilon": float(self.normalizer.epsilon),
                "clip_obs": float(self.normalizer.clip_obs),
            },
            "expert_config": dict(self.expert_config),
            "observation_dim": self.observation_dim,
            "optimizer_updates": self.optimizer_updates,
        }

    def load_state_dict(self, state: dict) -> None:
        if state["observation_dim"] != self.observation_dim:
            raise ValueError("AMP checkpoint observation dimension does not match")
        if state["expert_config"] != self.expert_config:
            raise ValueError("AMP checkpoint expert configuration does not match")
        self.discriminator.load_state_dict(state["discriminator"])
        self.optimizer.load_state_dict(state["optimizer"])
        normalizer = state["normalizer"]
        self.normalizer.mean = normalizer["mean"].detach().cpu().numpy().copy()
        self.normalizer.var = normalizer["var"].detach().cpu().numpy().copy()
        self.normalizer.count = float(normalizer["count"])
        self.normalizer.epsilon = float(normalizer["epsilon"])
        self.normalizer.clip_obs = float(normalizer["clip_obs"])
        self.optimizer_updates = int(state["optimizer_updates"])
