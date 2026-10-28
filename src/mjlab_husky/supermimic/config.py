"""Bounded online learned-dynamics training settings, separate from AMP/PPO."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from typing import Literal, Mapping


@dataclass
class SuperMimicConfig:
    actor_mode: Literal["deterministic", "reparameterized"] = "deterministic"
    real_exploration_std: float = 0.1
    detach_critic_transition_targets: bool = True
    analytic_action_cost: bool = False
    real_steps: int = 5
    horizon: int = 32
    model_sequence_length: int = 8
    model_batch_size: int = 256
    actor_batch_size: int = 256
    replay_capacity_steps: int = 128
    warmup_iterations: int = 20
    model_updates: int = 4
    actor_updates: int = 1
    critic_updates: int = 4
    amp_updates: int = 20
    amp_batch_size: int = 256
    model_hidden_dims: tuple[int, ...] = (512, 512, 512)
    model_learning_rate: float = 2e-4
    actor_learning_rate: float = 2e-4
    critic_learning_rate: float = 2e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.005  # Active only in legacy reparameterized mode.
    max_grad_norm: float = 1.0
    min_noise_std: float = 0.05
    max_noise_std: float = 1.0
    reward_scale: float = 1.0
    checkpoint_replay: bool = False
    log_interval: int = 1
    compile_projection: bool = True

    def __post_init__(self):
        if self.actor_mode not in ("deterministic", "reparameterized"):
            raise ValueError("actor_mode must be deterministic or reparameterized")
        if not isinstance(self.detach_critic_transition_targets, bool):
            raise ValueError("detach_critic_transition_targets must be a boolean")
        if not isinstance(self.analytic_action_cost, bool):
            raise ValueError("analytic_action_cost must be a boolean")
        if not math.isfinite(self.real_exploration_std) or self.real_exploration_std < 0:
            raise ValueError("real_exploration_std must be finite and nonnegative")
        positive = ("real_steps", "horizon", "model_sequence_length", "model_batch_size",
                    "actor_batch_size", "replay_capacity_steps", "model_updates",
                    "actor_updates", "critic_updates", "amp_updates", "amp_batch_size",
                    "log_interval")
        if any(getattr(self, name) < 1 for name in positive):
            raise ValueError("SuperMimic step, batch, update and capacity counts must be positive")
        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations cannot be negative")
        if self.replay_capacity_steps < self.model_sequence_length:
            raise ValueError("Replay capacity must hold a world-model training sequence")
        if not self.model_hidden_dims or min(self.model_hidden_dims) < 1:
            raise ValueError("World-model hidden dimensions must be positive")
        for name in ("model_learning_rate", "actor_learning_rate", "critic_learning_rate",
                     "max_grad_norm", "reward_scale"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not (0 <= self.gamma <= 1 and 0 <= self.gae_lambda <= 1):
            raise ValueError("gamma and gae_lambda must lie in [0, 1]")
        if not (0 < self.min_noise_std <= self.max_noise_std):
            raise ValueError("Noise bounds must be positive and ordered")
        if not math.isfinite(self.entropy_coef) or self.entropy_coef < 0:
            raise ValueError("entropy_coef cannot be negative")

    @classmethod
    def from_dict(cls, values: Mapping | "SuperMimicConfig" | None):
        if isinstance(values, cls):
            return cls(**asdict(values))
        values = dict(values or {})
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown SuperMimic configuration: {sorted(unknown)}")
        if "model_hidden_dims" in values:
            values["model_hidden_dims"] = tuple(values["model_hidden_dims"])
        return cls(**values)

    @classmethod
    def from_checkpoint_dict(cls, values: Mapping):
        """Checkpoints before actor_mode used learned stochastic imagination."""
        values = dict(values)
        values.setdefault("actor_mode", "reparameterized")
        values.setdefault("detach_critic_transition_targets", False)
        values.setdefault("analytic_action_cost", False)
        return cls.from_dict(values)


SuperMimicCfg = SuperMimicConfig
