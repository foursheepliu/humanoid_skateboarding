"""Vectorized real-transition replay with episode-aware autoregressive windows."""

import torch
from torch import Tensor


class ReplayBuffer:
    def __init__(self, capacity: int, num_envs: int, state_dim: int,
                 action_dim: int, device="cpu", seed=0):
        if min(capacity, num_envs, state_dim, action_dim) < 1:
            raise ValueError("Replay dimensions must be positive")
        self.capacity, self.num_envs = capacity, num_envs
        self.state_dim, self.action_dim = state_dim, action_dim
        self.device, self.steps = str(device), 0
        self.generator = torch.Generator(device=device).manual_seed(seed)
        shapes = {"state": (state_dim,), "next_state": (state_dim,), "action": (action_dim,),
                  "reward": (), "terminated": (), "truncated": (), "next_state_valid": ()}
        self.data = {key: torch.empty((capacity, num_envs, *shape), device=device,
                                     dtype=torch.bool if key in ("terminated", "truncated", "next_state_valid")
                                     else torch.float32) for key, shape in shapes.items()}

    def __len__(self):
        return min(self.steps, self.capacity) * self.num_envs

    @torch.no_grad()
    def append(self, state: Tensor, action: Tensor, next_state: Tensor, reward: Tensor,
               terminated: Tensor, truncated: Tensor,
               next_state_valid: Tensor | None = None):
        values = {"state": state, "action": action, "next_state": next_state,
                  "reward": reward.reshape(-1), "terminated": terminated.bool().reshape(-1),
                  "truncated": truncated.bool().reshape(-1)}
        values["next_state_valid"] = (~(values["terminated"] | values["truncated"])
                                     if next_state_valid is None else next_state_valid.bool().reshape(-1))
        slot = self.steps % self.capacity
        # Validate all fields before mutating the ring (append is all-or-nothing).
        for key, value in values.items():
            if value.shape != self.data[key].shape[1:]:
                raise ValueError(f"Replay shape mismatch for {key}: {value.shape}")
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite real transition in {key}")
        for key, value in values.items():
            self.data[key][slot].copy_(value.detach())
        self.steps += 1

    def sample_sequences(self, batch_size: int, horizon: int = 8):
        if not len(self) or min(batch_size, horizon) < 1:
            raise ValueError("Sampling requires nonempty replay and positive dimensions")
        oldest = max(0, self.steps - self.capacity)
        starts = torch.randint(oldest, self.steps, (batch_size,), device=self.device,
                               generator=self.generator)
        envs = torch.randint(self.num_envs, (batch_size,), device=self.device,
                             generator=self.generator)
        times = starts[:, None] + torch.arange(horizon, device=self.device)
        exists = times < self.steps
        # Clamp to an existing transition solely for safe padding; `valid` owns loss coverage.
        slots = times.clamp_max(self.steps - 1) % self.capacity
        data = {key: value[slots, envs[:, None]].clone() for key, value in self.data.items()}
        boundary = data["terminated"] | data["truncated"]
        prior_alive = torch.cat((torch.ones_like(boundary[:, :1]), ~boundary[:, :-1]), 1)
        valid = exists & prior_alive.long().cumprod(1).bool()
        data["valid"] = valid
        return data

    def sample_states(self, batch_size: int) -> Tensor:
        """Uniform current real states for starting independently imagined trajectories."""
        return self.sample_sequences(batch_size, 1)["state"][:, 0]

    def state_dict(self):
        # Unused torch.empty slots are not serialized (nor mistaken for real data).
        return {"capacity": self.capacity, "num_envs": self.num_envs,
                "state_dim": self.state_dim, "action_dim": self.action_dim,
                "steps": self.steps, "generator_state": self.generator.get_state(),
                "data": {key: value[:min(self.steps, self.capacity)].clone()
                         for key, value in self.data.items()}}

    def load_state_dict(self, state):
        identity = ("capacity", "num_envs", "state_dim", "action_dim")
        if any(state[key] != getattr(self, key) for key in identity):
            raise ValueError("Replay checkpoint dimensions differ")
        retained = min(state["steps"], self.capacity)
        for key, value in state["data"].items():
            if key not in self.data or value.shape != self.data[key][:retained].shape:
                raise ValueError(f"Replay checkpoint shape differs for {key}")
        if set(state["data"]) != set(self.data):
            raise ValueError("Replay checkpoint fields differ")
        for key, value in state["data"].items():
            self.data[key][:retained].copy_(value)
        self.steps = state["steps"]
        self.generator.set_state(state["generator_state"].cpu())
