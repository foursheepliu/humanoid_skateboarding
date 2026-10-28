"""Bounded CPU/CUDA sequence replay for coupled-system real transitions."""
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class TransitionBatch:
    state: torch.Tensor
    action: torch.Tensor
    next_state: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    valid_next: torch.Tensor
    context: torch.Tensor
    terminated: torch.Tensor | None = None
    observations: Any = None
    extras: Any = None


@dataclass
class SequenceBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    valid: torch.Tensor
    context: torch.Tensor
    terminated: torch.Tensor


class EpisodeSequenceReplay:
    """Time-major ring; a sampled window cannot bridge an environment reset.

    Both state targets are stored, including diagnostic pre-reset terminal
    physics. Terminal targets with stale actor history are marked invalid;
    reward and failure labels remain usable. Context banks deduplicate immutable
    per-environment physical contexts, while retaining older contexts on resume.
    """

    def __init__(self, num_envs, state_dim, action_dim, context_dim=0,
                 capacity_steps=64, device="cpu", dtype=torch.float32):
        if min(num_envs, state_dim, action_dim, capacity_steps) <= 0 or context_dim < 0:
            raise ValueError("Invalid replay dimensions")
        self.num_envs, self.state_dim, self.action_dim = int(num_envs), int(state_dim), int(action_dim)
        self.context_dim, self.capacity_steps = int(context_dim), int(capacity_steps)
        self.device, self.dtype = torch.device(device), dtype
        shape = (self.capacity_steps, self.num_envs)
        self.states = torch.zeros((*shape, self.state_dim), device=self.device, dtype=dtype)
        self.next_states = torch.zeros_like(self.states)
        self.actions = torch.zeros((*shape, self.action_dim), device=self.device, dtype=dtype)
        self.rewards = torch.zeros(shape, device=self.device, dtype=dtype)
        self.dones = torch.zeros(shape, device=self.device, dtype=torch.bool)
        self.valid = torch.zeros_like(self.dones)
        self.terminated = torch.zeros_like(self.dones)
        self.episode_ids = torch.zeros(shape, device=self.device, dtype=torch.long)
        self.context_ids = torch.zeros((self.capacity_steps,), device=self.device, dtype=torch.long)
        self._episode = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._context_bank = torch.empty((0, self.num_envs, self.context_dim), device=self.device, dtype=dtype)
        self._last_context_ref = None
        self._context_id = -1
        self.total_steps = 0
        self._candidate_cache = {}
        self._filtered_candidate_cache = {}
        self._filter_refs = {}

    def __len__(self):
        return min(self.total_steps, self.capacity_steps) * self.num_envs

    @property
    def allocated_bytes(self):
        tensors = (self.states, self.next_states, self.actions, self.rewards,
                   self.dones, self.valid, self.terminated, self.episode_ids,
                   self.context_ids, self._episode, self._context_bank)
        tensors = (*tensors, *self._candidate_cache.values(), *self._filtered_candidate_cache.values())
        seen, size = set(), 0
        for tensor in tensors:
            # A full-environment selection can reuse the unfiltered tensor.
            pointer = tensor.data_ptr()
            if pointer not in seen:
                seen.add(pointer)
                size += tensor.numel() * tensor.element_size()
        return size

    @property
    def host_allocated_bytes(self):
        return self.allocated_bytes if self.device.type == "cpu" else 0

    @property
    def device_allocated_bytes(self):
        return self.allocated_bytes if self.device.type != "cpu" else 0

    def _invalidate_candidates(self):
        self._candidate_cache.clear()
        self._filtered_candidate_cache.clear()
        self._filter_refs.clear()

    @torch.no_grad()
    def add(self, batch: TransitionBatch):
        if batch.state.shape != (self.num_envs, self.state_dim):
            raise ValueError("Replay state batch shape mismatch")
        if batch.next_state.shape != batch.state.shape or batch.action.shape != (self.num_envs, self.action_dim):
            raise ValueError("Replay action/target shape mismatch")
        if batch.context.shape != (self.num_envs, self.context_dim):
            raise ValueError("Replay physical context shape mismatch")
        # Adapter contexts are immutable; refresh_context returns a new tensor.
        if batch.context is not self._last_context_ref:
            self._context_bank = torch.cat((self._context_bank, batch.context.detach().to(self.device, self.dtype)[None]), 0)
            self._context_id += 1
            self._last_context_ref = batch.context
        slot = self.total_steps % self.capacity_steps
        for name, value in (("states", batch.state), ("next_states", batch.next_state),
                            ("actions", batch.action), ("rewards", batch.reward.reshape(-1)),
                            ("dones", batch.done.reshape(-1)), ("valid", batch.valid_next.reshape(-1))):
            getattr(self, name)[slot].copy_(value)
        terminated = batch.done if batch.terminated is None else batch.terminated
        self.terminated[slot].copy_(terminated.reshape(-1))
        self.episode_ids[slot].copy_(self._episode)
        self.context_ids[slot] = self._context_id
        self._episode.add_(batch.done.reshape(-1).to(device=self.device, dtype=torch.long))
        self.total_steps += 1
        self._invalidate_candidates()

    def break_continuity(self):
        """Call after restoring into newly reset simulators, before collection."""
        self._episode.add_(1)
        self._last_context_ref = None
        self._invalidate_candidates()

    begin_new_episode = break_continuity

    def _candidates(self, horizon):
        if horizon < 1 or horizon > self.capacity_steps:
            raise ValueError("Sequence horizon exceeds replay time capacity")
        cached = self._candidate_cache.get(horizon)
        if cached is not None:
            return cached
        oldest = max(0, self.total_steps - self.capacity_steps)
        stop = self.total_steps - horizon + 1
        if stop <= oldest:
            raise ValueError("Not enough real transitions for this sequence horizon")
        starts = torch.arange(oldest, stop, device=self.device)
        first = self.episode_ids[starts % self.capacity_steps]
        last = self.episode_ids[(starts + horizon - 1) % self.capacity_steps]
        rows = (first == last).nonzero(as_tuple=False)
        if rows.shape[0] == 0:
            raise ValueError("No sequence avoids reset boundaries; collect longer episodes or use shorter horizon")
        candidates = torch.stack((starts[rows[:, 0]], rows[:, 1]), -1)
        self._candidate_cache[horizon] = candidates
        return candidates

    def _filtered_candidates(self, horizon, env_ids):
        choices = self._candidates(horizon)
        if env_ids is None:
            return choices
        if isinstance(env_ids, torch.Tensor):
            # The tensor version changes on in-place edits. Retaining the input
            # below prevents object-id reuse from returning a stale selection.
            key = (horizon, "tensor", id(env_ids), env_ids._version,
                   env_ids.shape, env_ids.dtype, env_ids.device)
        elif isinstance(env_ids, range):
            key = (horizon, "range", env_ids.start, env_ids.stop, env_ids.step)
        else:
            key = (horizon, "ids", tuple(env_ids))
        cached = self._filtered_candidate_cache.get(key)
        if cached is not None:
            return cached
        allowed = torch.as_tensor(list(env_ids) if isinstance(env_ids, range) else env_ids,
                                  device=self.device, dtype=torch.long).reshape(-1)
        if allowed.numel() == 0 or bool(((allowed < 0) | (allowed >= self.num_envs)).any()):
            raise ValueError("Invalid held-out/training environment selection")
        membership = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        membership[allowed] = True
        # Linear lookup avoids torch.isin's repeated sorting of up to 1.5M rows
        # against thousands of environment IDs in the full-cycle CPU replay.
        selected = choices if bool(membership.all()) else choices[membership[choices[:, 1]]]
        if selected.shape[0] == 0:
            raise ValueError("Selected environments have no reset-safe sequence")
        self._filtered_candidate_cache[key] = selected
        if isinstance(env_ids, torch.Tensor):
            self._filter_refs[key] = env_ids
        return selected

    def sample(self, batch_size, horizon=8, env_ids=None, generator=None):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        choices = self._filtered_candidates(int(horizon), env_ids)
        choice = choices[torch.randint(choices.shape[0], (int(batch_size),), device=self.device, generator=generator)]
        slots = (choice[:, :1] + torch.arange(horizon, device=self.device)) % self.capacity_steps
        env = choice[:, 1:2].expand(-1, horizon)
        states = self.states[slots, env]
        endpoint = self.next_states[slots[:, -1], env[:, -1]][:, None]
        context = self._context_bank[self.context_ids[slots], env]
        return SequenceBatch(torch.cat((states, endpoint), 1), self.actions[slots, env],
                             self.rewards[slots, env], self.dones[slots, env],
                             self.valid[slots, env], context, self.terminated[slots, env])

    def state_dict(self):
        keys = ("states", "next_states", "actions", "rewards", "dones", "valid",
                "terminated", "episode_ids", "context_ids", "_episode", "_context_bank")
        tensors = {k: getattr(self, k).detach().to("cpu", copy=True) for k in keys}
        # Unused ring rows are not samples. Never serialize uninitialized bytes
        # or arbitrary NaNs from torch.empty into a partially filled snapshot.
        if self.total_steps < self.capacity_steps:
            for name in keys:
                if name not in ("_episode", "_context_bank"):
                    tensors[name][self.total_steps:].zero_()
        return {"version": 1, "shape": [self.num_envs, self.state_dim, self.action_dim,
                    self.context_dim, self.capacity_steps], "total_steps": self.total_steps,
                "context_id": self._context_id, "tensors": tensors}

    def load_state_dict(self, state):
        shape = [self.num_envs, self.state_dim, self.action_dim, self.context_dim, self.capacity_steps]
        if state["version"] != 1 or state["shape"] != shape:
            raise ValueError("Incompatible replay checkpoint layout")
        for name, value in state["tensors"].items():
            if name == "_context_bank":
                self._context_bank = value.to(self.device, self.dtype).clone()
            else:
                getattr(self, name).copy_(value)
        self.total_steps, self._context_id = int(state["total_steps"]), int(state["context_id"])
        self._last_context_ref = None
        self._invalidate_candidates()
