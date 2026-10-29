"""Original board speed/flat reward terms and reward-time collection.

Real labels are observed before resets or post-reward events. Model decoding
uses the same quaternion/phase projection as the skating adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import sys
import types
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class RewardPhysics:
    quaternion_wxyz: Tensor
    linear_velocity_w: Tensor
    gamma: Tensor
    push_steer_phase: Tensor

    def detached_clone(self):
        return RewardPhysics(*(value.detach().clone() for value in (
            self.quaternion_wxyz, self.linear_velocity_w, self.gamma, self.push_steer_phase)))


def unit_quaternion(quaternion):
    """Same operation order/identity fallback as adapter._unit_quaternion."""
    identity = torch.zeros_like(quaternion)
    identity[..., 0] = 1
    norm = quaternion.norm(dim=-1, keepdim=True)
    return torch.where(norm > 1e-8, quaternion / norm.clamp_min(1e-8), identity)


def inverse_rotate(quaternion, vector):
    """Pinned mjlab quat_apply_inverse, full WXYZ quaternion, no yaw reduction."""
    xyz = quaternion[..., 1:]
    cross = 2 * xyz.cross(vector, dim=-1)
    return vector - quaternion[..., :1] * cross + xyz.cross(cross, dim=-1)


def capture_real_reward_physics(raw):
    """Read ten physical/phase values per env, without observation compute/FK.

    Caller chooses the reward-time observation point. In a reset hook this
    reads the complete batch BEFORE original reset, never terminal adapter data.
    A real simulator quaternion is retained exactly as the original task reads
    it; normalization belongs only to the model projection path.
    """
    board = raw.skateboard.data
    return RewardPhysics(board.root_link_quat_w, board.root_link_lin_vel_w,
                         board.joint_pos[..., 0], raw.contact_phase[..., :2]).detached_clone()


def _term(manager, name):
    cfg = manager.get_term_cfg(name)
    module_name = "mjlab_husky.tasks.skater.mdp.rewards"
    module = sys.modules.get(module_name)
    if (module is None or getattr(module, name, None) is not cfg.func
            or getattr(cfg.func, "__module__", None) != module_name
            or getattr(cfg.func, "__name__", None) != name):
        raise ValueError(f"Expected original {module_name}.{name}")
    return cfg


@dataclass(frozen=True)
class NextPhysicalRewards:
    push_weight: float
    flat_weight: float
    push_std: float
    flat_std: float
    step_dt: float
    push_scale_by_dt: bool
    flat_scale_by_dt: bool
    cycle_time: float
    phase_ratios: tuple[float, ...]

    def __post_init__(self):
        values = (self.push_weight, self.flat_weight, self.push_std, self.flat_std,
                  self.step_dt, self.cycle_time, *self.phase_ratios)
        if not all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) for v in values):
            raise ValueError("Reward/phase configuration must contain finite numeric constants")
        if min(self.push_std, self.flat_std, self.step_dt, self.cycle_time) <= 0:
            raise ValueError("Positive reward std, step_dt, and cycle_time required")
        if type(self.push_scale_by_dt) is not bool or type(self.flat_scale_by_dt) is not bool:
            raise ValueError("Manager scaling flags must be bool")
        if (len(self.phase_ratios) != 5 or self.phase_ratios[0] != 0 or self.phase_ratios[-1] != 1
                or any(a >= b for a, b in zip(self.phase_ratios, self.phase_ratios[1:]))
                or int(self.cycle_time / 2 / self.step_dt) < 1):
            raise ValueError("Unsupported original phase-clock configuration")

    @classmethod
    def from_env(cls, env):
        """Read live immutable configuration; imports no environment package."""
        raw = getattr(env, "unwrapped", env)
        push = _term(raw.push_reward_manager, "push_skateboard_lin_vel")
        flat = _term(raw.reg_reward_manager, "board_flat")
        p, f = dict(push.params), dict(flat.params)
        if (set(p) != {"asset_cfg", "command_name", "std"} or set(f) != {"std"}
                or getattr(p["asset_cfg"], "name", None) != "skateboard" or p["command_name"] != "skate"):
            raise ValueError("Original board velocity/flat term parameters required")
        ratios = tuple(float(v) for v in raw.cfg.phase_ratios)
        # Adapter currently projects a shared cfg tuple. Reject per-env phase
        # customization rather than silently computing a different S.
        expected = raw.phase_ratios.new_tensor(ratios).expand_as(raw.phase_ratios)
        if not torch.equal(raw.phase_ratios, expected):
            raise ValueError("Per-env phase ratios differ from the adapter's shared configuration")
        return cls(float(push.weight), float(flat.weight), float(p["std"]), float(f["std"]),
                   float(raw.step_dt), raw.push_reward_manager._scale_by_dt,
                   raw.reg_reward_manager._scale_by_dt, float(raw.cycle_time), ratios)

    @property
    def push_scale(self):
        return self.step_dt if self.push_scale_by_dt else 1.

    @property
    def flat_scale(self):
        return self.step_dt if self.flat_scale_by_dt else 1.

    def reward_phase(self, previous_phase_length, previous_command):
        """Original reward phase: increment clock, use OLD command/still state."""
        phase_steps = previous_phase_length + 1
        still = previous_command[..., 0] < .1
        half_cycle = int(self.cycle_time / 2 / self.step_dt)
        reset_still = still & (torch.remainder(phase_steps - 1, half_cycle) == 0)
        phase_steps = torch.where(reset_still, torch.zeros_like(phase_steps), phase_steps)
        phase = torch.remainder(phase_steps * self.step_dt / self.cycle_time, 1.)
        r = self.phase_ratios
        push = (phase >= r[0]) & (phase < r[1]) & ~still
        steer = (phase >= r[2]) & (phase < r[3]) & ~still
        return torch.stack((push, steer), -1).to(previous_command.dtype)

    def model_physics(self, previous_phase_length, previous_command,
                      raw_quaternion, raw_velocity, raw_gamma):
        """Minimal projection for S; never reads a learned raw contact phase.

        This performs no full packed-state/history/observation projection.
        The existing full adapter may still project its own next-state copy.
        """
        return RewardPhysics(unit_quaternion(raw_quaternion), raw_velocity, raw_gamma,
                             self.reward_phase(previous_phase_length, previous_command))

    def components(self, previous_command, physical):
        leading = previous_command.shape[:-1]
        if (previous_command.shape[-1] != 2 or physical.quaternion_wxyz.shape != (*leading, 4)
                or physical.linear_velocity_w.shape != (*leading, 3)
                or physical.gamma.shape != leading or physical.push_steer_phase.shape != (*leading, 2)):
            raise ValueError("Reward snapshot/old-command batch dimensions differ")
        velocity = inverse_rotate(physical.quaternion_wxyz, physical.linear_velocity_w)
        error = ((previous_command[..., 0] - velocity[..., 0]).square()
                 + velocity[..., 1].square() + velocity[..., 2].square())
        push = torch.exp(-error / self.push_std ** 2) * self.push_weight * self.push_scale
        # Original push manager sanitizes before the task's external phase gate.
        push = torch.nan_to_num(push, nan=0., posinf=0., neginf=0.) * physical.push_steer_phase[..., 0]
        non_steer = physical.push_steer_phase[..., 1] != 1.
        flat = torch.where(non_steer, torch.exp(-physical.gamma.abs() / self.flat_std ** 2), 0.)
        flat = torch.nan_to_num(flat * self.flat_weight * self.flat_scale, nan=0., posinf=0., neginf=0.)
        return push, flat

    def __call__(self, previous_command, physical):
        push, flat = self.components(previous_command, physical)
        return push + flat

    def metadata(self):
        return {"version": 1, "terms": ["push_skateboard_lin_vel", "board_flat"],
                "weights": [self.push_weight, self.flat_weight], "std": [self.push_std, self.flat_std],
                "step_dt": self.step_dt, "manager_scale_by_dt": [self.push_scale_by_dt, self.flat_scale_by_dt],
                "cycle_time": self.cycle_time, "phase_ratios": list(self.phase_ratios),
                "snapshot_values_per_env": 10, "old_command_source": "pre-step command.skate.command_b",
                "phase_source": "after phase clock update, before command resampling",
                "model_quaternion": "adapter unit normalization including identity fallback",
                "reward_representation": "R-S_real side label; total prediction=residual+S_projected",
                "continuation_multiplier": False, "action_cost_decomposition": False}



@dataclass(frozen=True)
class KnownBoardRewards:
    layout: Any
    physical: NextPhysicalRewards

    def __post_init__(self):
        required = {
            "phase_length_buf": (), "command.skate.command_b": (2,),
            "skateboard.root_link_quat_w": (4,),
            "skateboard.root_link_lin_vel_w": (3,),
        }
        for key, shape in required.items():
            if self.layout.shapes.get(key) != shape:
                raise ValueError(f"Unsupported board reward state field: {key}")
        joints = self.layout.shapes.get("skateboard.joint_pos")
        if joints is None or len(joints) != 1 or joints[0] < 1:
            raise ValueError("Board rewards require the original first board joint")

    @classmethod
    def from_env(cls, env, layout):
        return cls(layout, NextPhysicalRewards.from_env(env))

    def __call__(self, state, raw_next_state):
        if state.shape != raw_next_state.shape or state.shape[-1] != self.layout.dim:
            raise ValueError("Board reward state/next-state dimensions differ")
        command = self.layout.get(state, "command.skate.command_b")
        physical = self.physical.model_physics(
            self.layout.get(state, "phase_length_buf"), command,
            self.layout.get(raw_next_state, "skateboard.root_link_quat_w"),
            self.layout.get(raw_next_state, "skateboard.root_link_lin_vel_w"),
            self.layout.get(raw_next_state, "skateboard.joint_pos")[..., 0])
        return self.physical(command, physical)

    def metadata(self):
        names = ("phase_length_buf", "command.skate.command_b",
                 "skateboard.root_link_quat_w", "skateboard.root_link_lin_vel_w",
                 "skateboard.joint_pos")
        return {**self.physical.metadata(), "state_dimension": self.layout.dim,
                "state_fields": {name: list(self.layout.shapes[name]) for name in names},
                "label_observer": "after original regularization compute, before task reset/events"}

    def observe_step(self, env):
        return BoardRewardObserver(self, getattr(env, "unwrapped", env))


class BoardRewardObserver:
    """Observe S at its original reward stage without changing task execution.

    The instance hook forwards each original compute call and its exact return.
    The context must enclose one env.step; finish rejects missing/extra calls.
    Capture errors are delayed until finish so the original environment still
    performs its ordinary reset/events. No terminal observation is fabricated.
    """
    def __init__(self, known, raw):
        self.known, self.raw = known, raw
        self.manager = raw.reg_reward_manager
        self.original = self.manager.compute
        self.had_instance = "compute" in self.manager.__dict__
        self.previous = self.manager.__dict__.get("compute")
        self.calls = self.original_calls = 0
        self.snapshot = self.value = self.error = None
        self.old_command = None
        self.restored = False

    def __enter__(self):
        command = self.raw.command_manager.get_command("skate")
        if command is None or command.shape[-1] != 2:
            raise ValueError("Original skate command is required at reward collection")
        self.old_command = command.detach().clone()
        observer = self
        def compute(_manager, dt):
            observer.calls += 1
            observer.original_calls += 1
            result = observer.original(dt)
            try:
                if observer.calls != 1:
                    raise RuntimeError("Board reward observer requires one regularization compute per step")
                if (isinstance(dt, bool) or not isinstance(dt, (int, float))
                        or not math.isfinite(dt) or float(dt) != observer.known.physical.step_dt):
                    raise ValueError("Reward compute dt differs from the captured live configuration")
                observer.snapshot = capture_real_reward_physics(observer.raw)
                observer.value = observer.known.physical(
                    observer.old_command, observer.snapshot).detach().clone()
            except Exception as exc:
                observer.error = exc
            return result
        self.manager.compute = types.MethodType(compute, self.manager)
        return self

    def __exit__(self, *_):
        if self.had_instance:
            self.manager.compute = self.previous
        else:
            del self.manager.__dict__["compute"]
        self.restored = True

    def finish(self):
        if (not self.restored or self.calls != 1 or self.original_calls != 1
                or self.error is not None or self.value is None):
            raise RuntimeError(f"Board reward capture failed: calls={self.calls}, original={self.original_calls}, error={self.error}")
        if (("compute" in self.manager.__dict__) != self.had_instance
                or self.manager.__dict__.get("compute") is not self.previous):
            raise RuntimeError("Original reward-manager compute was not restored")
        if not torch.isfinite(self.value).all():
            raise FloatingPointError("Nonfinite original board reward side label")
        return self.value
