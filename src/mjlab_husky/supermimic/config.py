"""A bounded, 24-GB 4090 starting profile for the coupled-world-model backend."""
from dataclasses import dataclass
import math


@dataclass
class SuperMimicConfig:
    num_envs: int = 4096
    iterations: int = 1000
    seed: int = 42
    replay_steps: int = 384
    replay_device: str = "cpu"
    warmup_steps: int = 320
    collect_steps: int = 5
    collect_interval: int = 2
    model_window: int = 8
    model_updates: int = 10
    model_batch: int = 256
    imagination_horizon: int = 32
    imagination_batch: int = 256
    hidden_width: int = 512
    model_lr: float = 3e-4
    actor_lr: float = 3e-6
    teacher_kl_coefficient: float = 10.0
    teacher_kl_limit: float = 0.003
    teacher_max_action_delta: float = 0.03
    teacher_max_standardized_delta: float = 0.1
    critic_lr: float = 3e-4
    discriminator_lr: float = 1e-4
    discriminator_updates: int = 2
    collection_noise: float = 0.10
    gamma: float = 0.99
    lam: float = 0.95
    save_interval: int = 100
    log_interval: int = 10
    expert_transitions: int = 200000
    mixed_precision: str = "none"
    joint_consistency: bool = False

    def validate(self):
        for name in ("model_lr", "actor_lr", "critic_lr", "discriminator_lr", "teacher_kl_limit",
                     "teacher_max_action_delta", "teacher_max_standardized_delta"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("collection_noise", "teacher_kl_coefficient"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("num_envs", "iterations", "replay_steps", "warmup_steps", "collect_steps",
                     "collect_interval", "model_window", "model_updates", "model_batch",
                     "imagination_horizon", "imagination_batch", "save_interval", "log_interval"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.replay_steps < self.model_window + 1 or self.warmup_steps < self.model_window + 1:
            raise ValueError("Replay/warmup must cover a full autoregressive training window")
        if self.replay_device not in ("cpu", "cuda"):
            raise ValueError("replay_device must be cpu or cuda")
        if self.replay_steps < self.warmup_steps:
            raise ValueError("Replay must retain the complete warmup, including a full phase cycle")
        if self.mixed_precision != "none":
            raise ValueError("The first coupled backend is FP32; BF16 requires a separate numerical validation")
