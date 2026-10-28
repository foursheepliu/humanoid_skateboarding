"""Only the original action-rate/action-acceleration reward terms.

This is a parameter-free callable, not an nn.Module. It uses the pre-step
action history and executed current action, never a post-reset next state.
Live configuration inspection imports no simulation package.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import sys
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class KnownActionRewards:
    layout: Any
    action_rate_weight: float
    action_acc_weight: float
    step_dt: float
    scale_by_dt: bool

    def __post_init__(self):
        if any(isinstance(value, Tensor) for value in (self.action_rate_weight, self.action_acc_weight, self.step_dt)):
            raise ValueError("Known-action configuration must use immutable numeric scalars, not tensors")
        if (not math.isfinite(self.step_dt) or self.step_dt <= 0
                or not math.isfinite(self.action_rate_weight) or not math.isfinite(self.action_acc_weight)):
            raise ValueError("Known-action weights must be finite and step_dt positive/finite")
        if type(self.scale_by_dt) is not bool:
            raise ValueError("RewardManager scale_by_dt must be a boolean")
        current = self.layout.shapes.get("action.action")
        previous = self.layout.shapes.get("action.prev_action")
        if (current is None or len(current) != 1 or type(current[0]) is not int
                or current[0] < 1 or previous != current):
            raise ValueError("Packed action.action/action.prev_action must have the same positive 1D action shape")

    @property
    def action_dim(self) -> int:
        return self.layout.shapes["action.action"][0]

    @property
    def effective_scale(self) -> float:
        return self.step_dt if self.scale_by_dt else 1.0

    @classmethod
    def from_env(cls, env, layout) -> KnownActionRewards:
        """Read active manager weights and scaling; reject changed term semantics."""
        env = getattr(env, "unwrapped", env)
        manager = env.reg_reward_manager
        module_name = "mjlab.envs.mdp.rewards"
        module = sys.modules.get(module_name)
        weights = []
        for name in ("action_rate_l2", "action_acc_l2"):
            cfg = manager.get_term_cfg(name)
            function = cfg.func
            if (not callable(function) or getattr(function, "__module__", None) != module_name
                    or getattr(function, "__name__", None) != name or module is None
                    or getattr(module, name, None) is not function):
                raise ValueError(f"Expected original {module_name}.{name}")
            if dict(cfg.params):
                raise ValueError(f"Original {name} takes no configurable parameters")
            weights.append(float(cfg.weight))
        result = cls(layout, weights[0], weights[1], float(env.step_dt), manager._scale_by_dt)
        if getattr(env.action_manager, "total_action_dim", None) != result.action_dim:
            raise ValueError("Packed action history dimension differs from the live action manager")
        return result

    def __call__(self, state: Tensor, action: Tensor) -> Tensor:
        """A_t in the task's dt-scaled reward units, preserving history gradients.

        Before process_action: action.action=a[t-1], prev_action=a[t-2].
        After process_action: manager(action,prev,prevprev)=(a[t],a[t-1],a[t-2]).
        No division by dt, actuator scaling, clipping or survival factor occurs
        here. ``action`` is the actual already-clipped executed raw input.
        """
        if action.shape != (*state.shape[:-1], self.action_dim):
            raise ValueError("Executed action and packed-state batch/action dimensions differ")
        previous = self.layout.get(state, "action.action")
        previous_previous = self.layout.get(state, "action.prev_action")
        rate = (action - previous).square().sum(-1)
        acceleration = (action - 2 * previous + previous_previous).square().sum(-1)
        # Match pinned RewardManager's per-term weighted/dt-scaled sanitizer.
        # Neither history slice is detached: imagined actions retain their graph.
        rate = torch.nan_to_num(rate * self.action_rate_weight * self.effective_scale,
                                nan=0., posinf=0., neginf=0.)
        acceleration = torch.nan_to_num(acceleration * self.action_acc_weight * self.effective_scale,
                                        nan=0., posinf=0., neginf=0.)
        return rate + acceleration

    def metadata(self) -> dict:
        return {"version": 1, "terms": ["action_rate_l2", "action_acc_l2"],
                "weights": {"action_rate_l2": self.action_rate_weight, "action_acc_l2": self.action_acc_weight},
                "step_dt": self.step_dt, "scale_by_dt": self.scale_by_dt,
                "effective_scale": self.effective_scale, "action_dim": self.action_dim,
                "action_fields": {"a[t-1]": "action.action", "a[t-2]": "action.prev_action"},
                "reduction": "sum over raw executed action coordinates", "depends_on_next_state": False}
