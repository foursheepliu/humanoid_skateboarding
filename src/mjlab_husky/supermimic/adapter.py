"""Training-only state adapter; the deployed HUSKY actor keeps its original input.

The world model sees privileged physical state and task memory as well as the
original policy/critic observations. Observations MUST be supplied from the
last reset/step: recomputing them here would resample noise or mutate history,
phase and filtered-contact buffers. No simulator/environment methods are
patched. Auto-reset next states must be masked by the replay consumer.

Only torch is imported, so layouts/history/AMP reward can be tested on CPU
without importing MuJoCo, mjlab or the task registry.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from math import pi, prod
from typing import Any

import torch
from torch import Tensor, nn


class StateLayout:
    """Ordered named slices, with arbitrary leading batch dimensions.

    Field shapes describe feature axes only; a scalar field has shape ``()``.
    ``pack`` allocates a new tensor, preventing simulator buffer aliasing.
    """

    def __init__(self, shapes: Mapping[str, Sequence[int] | int]):
        if not shapes:
            raise ValueError("A state layout requires at least one field")
        self.shapes: dict[str, tuple[int, ...]] = {}
        self.slices: dict[str, slice] = {}
        offset = 0
        for name, shape in shapes.items():
            shape = (shape,) if isinstance(shape, int) else tuple(shape)
            if not name or any(size < 1 for size in shape):
                raise ValueError(f"Invalid state field {name!r}: {shape}")
            width = prod(shape)
            self.shapes[name] = shape
            self.slices[name] = slice(offset, offset + width)
            offset += width
        self.dim = self.size = self.total_dim = offset

    @classmethod
    def from_tensors(cls, fields: Mapping[str, Tensor]) -> StateLayout:
        return cls({name: value.shape[1:] for name, value in fields.items()})

    @classmethod
    def from_shapes(cls, shapes: Mapping[str, Sequence[int] | int]) -> StateLayout:
        return cls(shapes)

    def __contains__(self, name: str) -> bool:
        return name in self.shapes

    def __getitem__(self, name: str) -> slice:
        return self.slices[name]

    def field_slice(self, name: str) -> slice:
        return self.slices[name]

    def get(self, state: Tensor, name: str) -> Tensor:
        if state.shape[-1] != self.dim:
            raise ValueError(f"Expected state width {self.dim}, got {state.shape[-1]}")
        return state[..., self.slices[name]].reshape((*state.shape[:-1], *self.shapes[name]))

    def unpack(self, state: Tensor) -> dict[str, Tensor]:
        return {name: self.get(state, name) for name in self.shapes}

    def pack(self, fields: Mapping[str, Tensor] | None = None, **kwargs: Tensor) -> Tensor:
        values = dict(fields or {})
        values.update(kwargs)
        if set(values) != set(self.shapes):
            raise ValueError(f"State keys differ: missing={set(self.shapes) - set(values)}, "
                             f"extra={set(values) - set(self.shapes)}")
        pieces = []
        leading = None
        for name, shape in self.shapes.items():
            value = values[name]
            if shape and tuple(value.shape[-len(shape):]) != shape:
                raise ValueError(f"Wrong shape for {name}: {tuple(value.shape)}; expected *{shape}")
            batch = value.shape[:-len(shape)] if shape else value.shape
            if leading is None:
                leading = batch
            if batch != leading:
                raise ValueError(f"Leading dimensions of {name} differ: {batch} != {leading}")
            pieces.append(value.reshape(*batch, prod(shape)))
        return torch.cat(pieces, dim=-1)

    def metadata(self) -> dict[str, Any]:
        return {"version": 1, "dimension": self.dim,
                "fields": [{"name": name, "shape": list(shape),
                            "start": self.slices[name].start, "stop": self.slices[name].stop}
                           for name, shape in self.shapes.items()]}


def roll_history(history: Tensor, frame: Tensor) -> Tensor:
    """Shift chronological (..., frames, features) history; keep autograd intact."""
    if history.ndim < 2 or history.shape[-1] != frame.shape[-1]:
        raise ValueError("History/frame feature dimensions differ")
    if history.shape[:-2] != frame.shape[:-1]:
        raise ValueError("History/frame batch dimensions differ")
    return torch.cat((history[..., 1:, :], frame.unsqueeze(-2)), dim=-2)


@contextmanager
def _frozen_discriminator(discriminator: nn.Module):
    flags = [(parameter, parameter.requires_grad) for parameter in discriminator.parameters()]
    was_training = discriminator.training
    try:
        discriminator.eval()  # Spectral norm must not mutate its power-iteration buffers.
        for parameter, _ in flags:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, enabled in flags:
            parameter.requires_grad_(enabled)
        discriminator.train(was_training)


def differentiable_amp_reward(
    frames: Tensor,
    task_reward: Tensor,
    push_gate: Tensor,
    discriminator: nn.Module,
    mean: Tensor | Any,
    std: Tensor | Any,
    *,
    amp_reward_coef: float = 5.0,
    task_reward_lerp: float = 0.7,
    use_lerp: bool = False,
    clip_obs: float = 10.0,
    reward_dt: float = 0.02,
    already_frozen: bool = False,
) -> Tensor:
    """Original phase-gated AMP formula with derivatives through predicted joints.

    ``std`` is sqrt(normalizer.var + normalizer.epsilon), not variance.
    Normalizer statistics and discriminator weights stay fixed. Forward values
    match AMP_PPO for real 0/1 gates. Imagined predicted gates are bounded to
    [0,1]; this is a continuous surrogate between the original discrete phases.
    The task reward already includes RewardManager's dt scaling.
    ``already_frozen=True`` requires the caller to freeze parameters and set
    eval mode outside the rollout; it avoids per-step mode/flag mutations.
    """
    if frames.ndim < 3:
        raise ValueError("AMP input must have (..., frames, joint_angles) axes")
    mean = torch.as_tensor(mean, device=frames.device, dtype=frames.dtype).detach()
    std = torch.as_tensor(std, device=frames.device, dtype=frames.dtype).detach().clamp_min(1e-6)
    normalized = ((frames - mean) / std).clamp(-clip_obs, clip_obs)
    with nullcontext() if already_frozen else _frozen_discriminator(discriminator):
        logits = discriminator(normalized.flatten(-2)).squeeze(-1)
    style = amp_reward_coef * (1.0 - (logits - 1.0).square() / 4.0).clamp_min(0.0)
    task_reward = task_reward.reshape(logits.shape)
    gate = push_gate.reshape(logits.shape).clamp(0.0, 1.0)
    if use_lerp:
        mixed = task_reward_lerp * task_reward + (1.0 - task_reward_lerp) * style
    else:
        mixed = task_reward + reward_dt * style
    return task_reward + gate * (mixed - task_reward)


@dataclass(frozen=True)
class _ObservationTerm:
    name: str
    start: int
    stop: int
    history: int
    frame_width: int
    scale: Any = None
    clip: tuple[float, float] | None = None
    noise_min: Any = None
    noise_max: Any = None
    sensor_name: str | None = None


def _unit_quaternion(quaternion: Tensor) -> Tensor:
    """WXYZ quaternions, matching the task's pinned mjlab math functions."""
    identity = torch.zeros_like(quaternion)
    identity[..., 0] = 1
    norm = quaternion.norm(dim=-1, keepdim=True)
    return torch.where(norm > 1e-8, quaternion / norm.clamp_min(1e-8), identity)


def _quat_apply(quaternion: Tensor, vector: Tensor) -> Tensor:
    xyz, vector = torch.broadcast_tensors(quaternion[..., 1:], vector)
    twice_cross = 2 * torch.linalg.cross(xyz, vector, dim=-1)
    return vector + quaternion[..., :1] * twice_cross + torch.linalg.cross(xyz, twice_cross, dim=-1)


def _quat_inverse_apply(quaternion: Tensor, vector: Tensor) -> Tensor:
    return _quat_apply(torch.cat((quaternion[..., :1], -quaternion[..., 1:]), -1), vector)


def _quat_multiply(left: Tensor, right: Tensor) -> Tensor:
    w = left[..., :1] * right[..., :1] - (left[..., 1:] * right[..., 1:]).sum(-1, keepdim=True)
    xyz = (left[..., :1] * right[..., 1:] + right[..., :1] * left[..., 1:]
           + torch.linalg.cross(left[..., 1:], right[..., 1:], dim=-1))
    return torch.cat((w, xyz), -1)


def _wrap_pi(angle: Tensor) -> Tensor:
    wrapped = torch.remainder(angle + pi, 2 * pi)
    # Match mjlab's MATLAB-style convention: +pi stays +pi, -pi stays -pi.
    return torch.where((wrapped == 0) & (angle > 0), pi, wrapped - pi)


class SkateWorldModelAdapter:
    """Discover a fixed training state schema from the live unwrapped task.

    The first fields are exactly ``actor_obs``, ``critic_obs``, ``amp_history``
    and ``amp_gate``. Everything else is privileged model context; never pass
    the packed state directly to the deployed actor. State dimensions are
    inferred at runtime, including the 23-DoF G1 and all transition memories.
    """

    supports_exogenous_projection = True
    command_noise_dim = 6

    def __init__(self, env: Any, observations: Mapping[str, Tensor], amp_frames: Tensor, *,
                 project_transition_memory: bool = False):
        if type(project_transition_memory) is not bool:
            raise ValueError("project_transition_memory must be a boolean")
        self.project_transition_memory = project_transition_memory
        self.env = getattr(env, "unwrapped", env)
        self.num_envs = int(self.env.num_envs)
        self.step_dt = float(self.env.step_dt)
        self.cycle_time = float(getattr(self.env, "cycle_time", 6.0))
        self.amp_num_frames, self.amp_observation_dim = amp_frames.shape[-2:]
        self.actor_group = "policy"
        self.critic_group = "critic"
        self._sources: dict[str, Callable[[], Tensor]] = {}
        self._discover_sources()
        fields = self._fields(observations, amp_frames)
        self.layout = StateLayout.from_tensors(fields)
        self.state_dim = self.layout.dim
        self._transition_memory_fields = (
            ("bezier.push2steer_start_pos_b", 2, 3),
            ("bezier.push2steer_start_quat_b", 2, 4),
            ("bezier.steer2push_start_pos_b", 3, 3),
            ("bezier.steer2push_start_quat_b", 3, 4),
        )
        if self.project_transition_memory:
            body_counts = set()
            for name, _phase_column, width in self._transition_memory_fields:
                shape = self.layout.shapes.get(name)
                if shape is None or len(shape) != 2 or shape[0] < 1 or shape[1] != width:
                    raise ValueError(f"Transition memory requires {name} with shape (num_bodies, {width})")
                body_counts.add(shape[0])
            if (len(body_counts) != 1 or self.layout.shapes.get("contact_phase") != (4,)
                    or self.layout.shapes.get("phase_length_buf") != ()):
                raise ValueError("Transition memory requires matching body counts and original phase buffers")
        self.actor_dim = observations[self.actor_group].shape[-1]
        self.critic_dim = observations[self.critic_group].shape[-1]
        action_dim = getattr(env, "num_actions", None)
        if action_dim is None:
            action_dim = self.env.action_manager.total_action_dim
        self.action_dim = int(action_dim)
        self._obs_terms = {group: self._observation_terms(group, observations[group].shape[-1])
                           for group in (self.actor_group, self.critic_group)}
        self.sensor_noise_dim = sum(term.frame_width for term in self._obs_terms[self.actor_group])
        self._sensor_frames = self._read_sensor_frames()
        command_terms = getattr(getattr(self.env, "command_manager", None), "_terms", {})
        self._command_cfg = getattr(command_terms.get("skate"), "cfg", None)
        if self._command_cfg is None and "command.skate.command_b" in self.layout:
            raise ValueError("The original skate command configuration is required")
        # Every deployed actor input is an analytical function of physical
        # predictions, external command/noise, history and the executed action.
        # Unsupported future actor terms must not silently become model outputs.
        for term in self._obs_terms[self.actor_group]:
            if self._analytic_term(fields, term) is None:
                raise ValueError(f"No physics-derived actor observation for {term.name!r}")

    def _add_source(self, name: str, getter: Callable[[], Any]) -> None:
        def tensor_getter() -> Any:
            value = getter()
            # mjlab 13212ad exposes sim.data/model through zero-copy TorchArray
            # proxies. Unwrap the tensor without importing the simulation stack.
            return value if isinstance(value, Tensor) else getattr(value, "_tensor", None)

        try:
            value = tensor_getter()
        except (AttributeError, KeyError, TypeError, RuntimeError):
            return
        if isinstance(value, Tensor) and value.ndim and value.shape[0] == self.num_envs and value.numel():
            self._sources[name] = tensor_getter

    def _discover_sources(self) -> None:
        env = self.env
        for entity_name in ("robot", "skateboard"):
            entity = getattr(env, entity_name, None)
            if entity is None:
                continue
            for attribute in ("root_link_pos_w", "root_link_quat_w", "root_link_lin_vel_w",
                              "root_link_ang_vel_w", "joint_pos", "joint_vel", "joint_acc",
                              "applied_torque", "default_joint_pos", "default_joint_vel",
                              "gravity_vec_w", "forward_vec_b"):
                self._add_source(f"{entity_name}.{attribute}",
                                 lambda e=entity, a=attribute: getattr(e.data, a))
        for attribute in ("phase_length_buf", "episode_length_buf", "still", "contact_phase",
                          "last_contact_phase", "last_contacts", "last_wheel_contacts",
                          "last_contacts_b", "last_contacts_g", "contact_filt", "wheel_contact_filt",
                          "just_entered_push2steer", "just_entered_steer2push",
                          "just_exited_push2steer", "just_exited_steer2push"):
            self._add_source(attribute, lambda a=attribute: getattr(env, a))
        for name in getattr(env, "body_bezier_buffers", {}):
            self._add_source(f"bezier.{name}", lambda n=name: env.body_bezier_buffers[n])
        manager = getattr(env, "action_manager", None)
        for attribute in ("action", "prev_action", "prev_prev_action"):
            self._add_source(f"action.{attribute}", lambda a=attribute: getattr(manager, a))
        # Direct tensor attributes only: no property getter with task side effects.
        terms = getattr(getattr(env, "command_manager", None), "_terms", {})
        for name, term in terms.items():
            for attribute in ("command_b", "heading_ref_b", "heading_error", "is_heading_env",
                              "is_standing_env", "time_left", "command_counter"):
                self._add_source(f"command.{name}.{attribute}",
                                 lambda t=term, a=attribute: getattr(t, a))
        for name, sensor in getattr(getattr(env, "scene", None), "sensors", {}).items():
            for attribute in ("force", "found", "current_air_time", "last_air_time",
                              "current_contact_time", "last_contact_time"):
                self._add_source(f"sensor.{name}.{attribute}",
                                 lambda s=sensor, a=attribute: getattr(s.data, a))
        # Native mjlab tensor views, where exposed; entity states above also
        # cover robot + the independently moving articulated skateboard.
        sim = getattr(env, "sim", None)
        for attribute in ("qpos", "qvel", "act"):
            self._add_source(f"sim.{attribute}", lambda a=attribute: getattr(sim.data, a))
        # Randomized mass/COM/friction are persistent context. Only batched
        # tensors are captured; shared, non-randomized model constants needn't be.
        for attribute in ("body_mass", "body_inertia", "body_ipos", "geom_friction",
                          "dof_damping", "actuator_gainprm", "actuator_biasprm"):
            self._add_source(f"physics.{attribute}", lambda a=attribute: getattr(sim.model, a))
        event_manager = getattr(env, "event_manager", None)
        for attribute in ("_interval_term_time_left", "_interval_time_left"):
            timers = getattr(event_manager, attribute, ())
            if isinstance(timers, (tuple, list)):
                for index in range(len(timers)):
                    self._add_source(f"event.{attribute}.{index}",
                                     lambda a=attribute, i=index: getattr(event_manager, a)[i])

    def _fields(self, observations: Mapping[str, Tensor], amp_frames: Tensor) -> dict[str, Tensor]:
        if amp_frames.shape != (self.num_envs, self.amp_num_frames, self.amp_observation_dim):
            raise ValueError("AMP history dimensions changed")
        # Clone/cat at capture prevents physical/cache tensors from being
        # overwritten by the next simulator step.
        fields = {"actor_obs": observations[self.actor_group],
                  "critic_obs": observations[self.critic_group],
                  "amp_history": amp_frames,
                  "amp_gate": self.env.contact_phase[:, 0]}
        fields.update({name: getter().to(fields["actor_obs"]) for name, getter in self._sources.items()})
        return fields

    @torch.no_grad()
    def capture(self, observations: Mapping[str, Tensor], amp_frames: Tensor) -> Tensor:
        """Snapshot already-computed observations; never advances/noises history."""
        return self.layout.pack(self._fields(observations, amp_frames)).detach().clone()

    pack = capture

    def actor_obs(self, state: Tensor) -> Tensor:
        return self.layout.get(state, "actor_obs")

    def critic_obs(self, state: Tensor) -> Tensor:
        return self.layout.get(state, "critic_obs")

    def amp_history(self, state: Tensor) -> Tensor:
        return self.layout.get(state, "amp_history")

    def _observation_terms(self, group: str, width: int) -> tuple[_ObservationTerm, ...]:
        manager = getattr(self.env, "observation_manager", None)
        names = getattr(manager, "_group_obs_term_names", {}).get(group, [])
        dimensions = getattr(manager, "group_obs_term_dim", {}).get(group, [])
        cfgs = getattr(manager, "_group_obs_term_cfgs", {}).get(group, [])
        result = []
        offset = 0
        for name, shape, cfg in zip(names, dimensions, cfgs, strict=True):
            size = prod(shape)
            history = max(1, int(getattr(cfg, "history_length", 0)))
            if size % history:
                raise ValueError(f"Observation history shape is unsupported for {group}/{name}")
            noise = getattr(cfg, "noise", None)
            noise_min = getattr(noise, "n_min", None)
            noise_max = getattr(noise, "n_max", None)
            if noise is not None and (noise_min is None or noise_max is None):
                raise ValueError(f"Only original uniform observation noise is supported: {group}/{name}")
            if noise is not None and getattr(noise, "operation", "add") != "add":
                raise ValueError(f"Original additive observation noise is required: {group}/{name}")
            if getattr(cfg, "delay_max_lag", 0):
                raise ValueError(f"Observation delays need an explicit delay-state adapter: {group}/{name}")
            clip = getattr(cfg, "clip", None)
            if noise is not None and clip is not None:
                raise ValueError("Recovering exact external sensor noise requires unclipped original task observations")
            if noise is not None:
                noise_min = torch.as_tensor(noise_min, device=self.env.device, dtype=torch.float32).clone()
                noise_max = torch.as_tensor(noise_max, device=self.env.device, dtype=torch.float32).clone()
            result.append(_ObservationTerm(
                name, offset, offset + size, history, size // history,
                scale=getattr(cfg, "scale", None), clip=clip,
                noise_min=noise_min, noise_max=noise_max,
                sensor_name=getattr(cfg, "params", {}).get("sensor_name")))
            offset += size
        if not result or offset != width:
            raise ValueError(f"Cannot determine original {group} term/history layout ({offset} != {width})")
        return tuple(result)

    def _read_sensor_frames(self) -> dict[str, tuple[Tensor, Tensor]]:
        """Validate the actual compiled IMU attachment; never assume root COM."""
        names = {term.sensor_name for terms in self._obs_terms.values() for term in terms
                 if term.name in ("base_ang_vel", "base_lin_vel") and term.sensor_name is not None}
        supplied = getattr(self.env, "adapter_sensor_frames", {})
        model = getattr(getattr(self.env, "sim", None), "mj_model", None)
        frames = {}
        for name in names:
            if name in supplied:
                position, quaternion = supplied[name]
            elif model is not None:
                sensor = model.sensor(name)
                if int(sensor.objtype[0]) != 6:  # MuJoCo mjOBJ_SITE.
                    raise ValueError(f"Sensor {name} must use a site attachment")
                site_id = int(sensor.objid[0])
                root_indexing = getattr(self.env.robot, "indexing", None)
                if root_indexing is None:
                    root_indexing = getattr(self.env.robot.data, "indexing", None)
                root_id = int(root_indexing.root_body_id)
                if int(model.site_bodyid[site_id]) != root_id:
                    raise ValueError(f"Sensor {name} is not attached to the root body; root-only IMU projection is invalid")
                position, quaternion = model.site_pos[site_id], model.site_quat[site_id]
            else:
                raise ValueError(f"Compiled sensor frame is required for {name}")
            frames[name] = (torch.as_tensor(position, device=self.env.device, dtype=torch.float32).clone(),
                            torch.as_tensor(quaternion, device=self.env.device, dtype=torch.float32).clone())
        return frames

    def _heading(self, fields: Mapping[str, Tensor], entity: str) -> Tensor:
        quaternion = _unit_quaternion(fields[f"{entity}.root_link_quat_w"])
        forward = fields.get(f"{entity}.forward_vec_b")
        if forward is None:
            raise ValueError(f"Missing {entity}.forward_vec_b for the original heading observation")
        direction = _quat_apply(quaternion, forward)
        return torch.atan2(direction[..., 1], direction[..., 0])

    def _analytic_term(self, fields: Mapping[str, Tensor], term: _ObservationTerm) -> Tensor | None:
        name = term.name
        if name == "command":
            return fields["command.skate.command_b"]
        if name == "actions":
            return fields["action.action"]
        if name == "phase":
            return torch.remainder(fields["phase_length_buf"] * self.step_dt / self.cycle_time, 1.0).unsqueeze(-1)
        if name == "joint_pos":
            return fields["robot.joint_pos"] - fields["robot.default_joint_pos"]
        if name == "joint_vel":
            return fields["robot.joint_vel"] - fields["robot.default_joint_vel"]
        if name == "heading":
            return self._heading(fields, "robot").unsqueeze(-1)
        if name == "contact_phase":
            return fields["contact_phase"]
        if name == "skateboard_roll":
            return fields["skateboard.joint_pos"][..., [0, 1, 4]]
        if name in ("l_foot_contact_forces", "r_foot_contact_forces",
                    "l_foot_contact_forces_b", "r_foot_contact_forces_b"):
            force = fields[f"sensor.{term.sensor_name}.force"]
            force = force.reshape(*force.shape[:force.ndim - len(self.layout.shapes[f'sensor.{term.sensor_name}.force'])], -1)
            return force.sign() * force.abs().log1p()
        if name == "heading_error":
            target = _wrap_pi(fields["command.skate.heading_ref_b"] + fields["command.skate.command_b"][..., 1])
            return _wrap_pi(target - self._heading(fields, "skateboard")).unsqueeze(-1)
        if name not in ("base_ang_vel", "base_lin_vel", "projected_gravity", "skate_pose_local",
                        "skate_vel_local", "skate_ang_vel_local"):
            # Transition FK targets remain learned privileged critic features;
            # they are never supplied to the deployed actor.
            return None
        robot_quat = _unit_quaternion(fields["robot.root_link_quat_w"])
        if name == "projected_gravity":
            return _quat_inverse_apply(robot_quat, fields["robot.gravity_vec_w"])
        if name in ("base_ang_vel", "base_lin_vel"):
            angular_b = _quat_inverse_apply(robot_quat, fields["robot.root_link_ang_vel_w"])
            if name == "base_ang_vel":
                value = angular_b
            else:
                value = _quat_inverse_apply(robot_quat, fields["robot.root_link_lin_vel_w"])
            if term.sensor_name is not None:
                position, site_quat = self._sensor_frames[term.sensor_name]
                if name == "base_lin_vel":
                    value = value + torch.linalg.cross(angular_b, position.expand_as(angular_b), dim=-1)
                value = _quat_inverse_apply(site_quat, value)
            return value
        if name == "skate_vel_local":
            return _quat_inverse_apply(robot_quat, fields["skateboard.root_link_lin_vel_w"])
        if name == "skate_ang_vel_local":
            return _quat_inverse_apply(robot_quat, fields["skateboard.root_link_ang_vel_w"])
        relative_position = _quat_inverse_apply(robot_quat, fields["skateboard.root_link_pos_w"]
                                               - fields["robot.root_link_pos_w"])
        conjugate = torch.cat((robot_quat[..., :1], -robot_quat[..., 1:]), -1)
        board_quat = _unit_quaternion(fields["skateboard.root_link_quat_w"])
        relative_quat = _quat_multiply(conjugate, board_quat)
        tangent = relative_position.new_tensor((1., 0., 0.)).expand_as(relative_position)
        normal = relative_position.new_tensor((0., 0., 1.)).expand_as(relative_position)
        return torch.cat((relative_position, _quat_apply(relative_quat, tangent),
                          _quat_apply(relative_quat, normal)), -1)

    @staticmethod
    def _process_term(raw: Tensor, term: _ObservationTerm, uniforms: Tensor | None = None) -> Tensor:
        # Pinned observation_manager.py: compute -> noise -> clip -> scale -> history.
        value = raw
        if term.noise_min is not None and uniforms is not None:
            minimum = torch.as_tensor(term.noise_min, device=raw.device, dtype=raw.dtype)
            maximum = torch.as_tensor(term.noise_max, device=raw.device, dtype=raw.dtype)
            value = value + minimum + uniforms.detach() * (maximum - minimum)
        if term.clip is not None:
            value = value.clamp(*term.clip)
        if term.scale is not None:
            value = value * torch.as_tensor(term.scale, device=raw.device, dtype=raw.dtype)
        return value

    def _project_commands(self, previous: Mapping[str, Tensor], result: dict[str, Tensor],
                          exogenous: Mapping[str, Tensor]) -> None:
        prefix = "command.skate."
        if prefix + "command_b" not in result:
            return
        truth = exogenous.get("next_state")
        if truth is not None:
            for name in self.layout.shapes:
                if name.startswith(prefix):
                    result[name] = self.layout.get(truth.detach(), name)
            return
        # Exogenous draws are supplied by the caller, fixed for this rollout
        # and independent of both the actor and learned dynamics.
        for name in self.layout.shapes:
            if name.startswith(prefix):
                result[name] = previous[name]
        cfg = self._command_cfg
        time_left = previous[prefix + "time_left"] - self.step_dt
        resample = time_left <= 0
        phase_exit = (previous["contact_phase"][..., 2] > 0.5) & (result["contact_phase"][..., 2] < 0.5)
        uniforms = exogenous.get("command_uniforms")
        if uniforms is None:
            raise ValueError("Projection requires independent command_uniforms or a training next_state context")
        if uniforms.shape != (*time_left.shape, self.command_noise_dim):
            raise ValueError("Command uniform-noise dimensions differ")
        uniforms = uniforms.detach()
        command = previous[prefix + "command_b"]
        vx_range = cfg.ranges.lin_vel_x
        vx = torch.where(resample, vx_range[0] + uniforms[..., 0] * (vx_range[1] - vx_range[0]), command[..., 0])
        heading_env = previous[prefix + "is_heading_env"] > .5
        standing_env = previous[prefix + "is_standing_env"] > .5
        heading_env = torch.where(resample, uniforms[..., 2] <= cfg.rel_heading_envs, heading_env)
        standing_env = torch.where(resample, uniforms[..., 3] <= cfg.rel_standing_envs, standing_env)
        relative = command[..., 1]
        reference = previous[prefix + "heading_ref_b"]
        if cfg.heading_command:
            heading_range = cfg.ranges.heading
            sampled_relative = heading_range[0] + uniforms[..., 1] * (heading_range[1] - heading_range[0])
            heading_resample = resample & heading_env
            relative = torch.where(heading_resample, sampled_relative, relative)
            board_heading = self._heading(result, "skateboard")
            reference = torch.where(heading_resample, board_heading, reference)
            # Original _update_command re-samples relative heading once more
            # on exiting push->steer, even when the 20-second timer also fired.
            phase_heading = phase_exit & heading_env
            phase_relative = heading_range[0] + uniforms[..., 5] * (heading_range[1] - heading_range[0])
            relative = torch.where(phase_heading, phase_relative, relative)
            reference = torch.where(phase_heading, board_heading, reference)
        next_command = torch.stack((vx, relative), -1)
        result[prefix + "command_b"] = torch.where(standing_env[..., None], torch.zeros_like(next_command), next_command)
        result[prefix + "heading_ref_b"] = reference
        result[prefix + "is_heading_env"] = heading_env.to(command.dtype)
        result[prefix + "is_standing_env"] = standing_env.to(command.dtype)
        interval = cfg.resampling_time_range
        sampled_time = interval[0] + uniforms[..., 4] * (interval[1] - interval[0])
        result[prefix + "time_left"] = torch.where(resample, sampled_time, time_left)
        if prefix + "command_counter" in result:
            result[prefix + "command_counter"] = previous[prefix + "command_counter"] + resample.to(command.dtype)

    def project_next(self, state: Tensor, predicted_next: Tensor, action: Tensor, *,
                     exogenous: Mapping[str, Tensor] | None = None) -> Tensor:
        """Enforce known history/action/phase updates during imagined rollouts.

        mjlab concatenates term-major histories, not whole-frame-major history.
        Each newest deployed term is derived from physical predictions. Noise
        and command draws are external fixed samples; training can condition on
        their observed next-state realizations without teacher-forcing physics.
        """
        previous = self.layout.unpack(state)
        result = self.layout.unpack(predicted_next)
        exogenous = {} if exogenous is None else exogenous
        # Root rotations live on SO(3); prevent independent quaternion norm
        # drift before deriving gravity, heading, IMU and board observations.
        for entity in ("robot", "skateboard"):
            key = f"{entity}.root_link_quat_w"
            if key in result:
                result[key] = _unit_quaternion(result[key])
        for name in self.layout.shapes:
            if (name.startswith("physics.") or name.endswith((".default_joint_pos", ".default_joint_vel",
                                                              ".gravity_vec_w", ".forward_vec_b"))):
                result[name] = previous[name]
        if "action.prev_prev_action" in result and "action.prev_action" in previous:
            result["action.prev_prev_action"] = previous["action.prev_action"]
        if "action.prev_action" in result and "action.action" in previous:
            result["action.prev_action"] = previous["action.action"]
        if "action.action" in result:
            result["action.action"] = action
        next_phase = None
        if "phase_length_buf" in result:
            phase_steps = previous["phase_length_buf"] + 1
            command = previous.get("command.skate.command_b")
            still = command[..., 0] < 0.1 if command is not None else previous["still"] > 0.5
            half_cycle = int(self.cycle_time / 2 / self.step_dt)
            reset_still = still & (torch.remainder(phase_steps - 1, half_cycle) == 0)
            phase_steps = torch.where(reset_still, torch.zeros_like(phase_steps), phase_steps)
            next_phase = torch.remainder(phase_steps * self.step_dt / self.cycle_time, 1.0)
            result["phase_length_buf"] = phase_steps
            if "still" in result:
                result["still"] = still.to(state.dtype)
            ratios = getattr(self.env.cfg, "phase_ratios", (0.0, 0.4, 0.5, 0.95, 1.0))
            phase_masks = [(next_phase >= ratios[a]) & (next_phase < ratios[b]) & ~still
                           for a, b in ((0, 1), (2, 3), (1, 2), (3, 4))]
            contact_phase = torch.stack(phase_masks, dim=-1).to(state.dtype)
            result["amp_gate"] = contact_phase[..., 0]
            if "contact_phase" in result:
                result["contact_phase"] = contact_phase
            if "last_contact_phase" in result:
                result["last_contact_phase"] = previous["contact_phase"]
        else:
            result["amp_gate"] = result["amp_gate"].clamp(0, 1)
        if self.project_transition_memory:
            # Original _resample_contact_phases writes each board-frame anchor
            # only on entry, before rewards/reset/command resampling. The entry
            # value stays learned: root/joint state is insufficient for exact FK.
            # Use actual old contact flags, including the task's stale flags
            # after reset, rather than reconstructing them from the old clock.
            for name, phase_column, _width in self._transition_memory_fields:
                entry = ((result["contact_phase"][..., phase_column] > .5)
                         & (previous["contact_phase"][..., phase_column] < .5))
                result[name] = torch.where(entry[..., None, None], result[name], previous[name])
        self._project_commands(previous, result, exogenous)
        truth_state = exogenous.get("next_state")
        truth_fields = self.layout.unpack(truth_state.detach()) if truth_state is not None else None
        sensor_uniforms = exogenous.get("sensor_uniforms")
        if sensor_uniforms is not None and sensor_uniforms.shape != (*state.shape[:-1], self.sensor_noise_dim):
            raise ValueError("Sensor uniform-noise dimensions differ")
        for key, group in (("actor_obs", self.actor_group), ("critic_obs", self.critic_group)):
            old, predicted = previous[key], result[key]
            pieces = []
            noise_offset = 0
            for term in self._obs_terms[group]:
                old_frames = old[..., term.start:term.stop].reshape(*old.shape[:-1], term.history, term.frame_width)
                clean = self._analytic_term(result, term)
                if clean is None:
                    if group == self.actor_group:
                        raise ValueError(f"Unsupported deployed observation: {term.name}")
                    newest = predicted[..., term.stop - term.frame_width:term.stop]
                elif term.noise_min is not None and truth_fields is not None:
                    clean_truth = self._analytic_term(truth_fields, term)
                    observed_truth = truth_fields[key][..., term.stop - term.frame_width:term.stop]
                    # The residual is external sensor noise, detached from the
                    # actor/model. Physical targets never replace predictions.
                    residual = (observed_truth - self._process_term(clean_truth, term)).detach()
                    newest = self._process_term(clean, term) + residual
                else:
                    uniforms = None
                    if term.noise_min is not None:
                        if sensor_uniforms is None:
                            raise ValueError("Noisy actor observations require independent sensor_uniforms or training next_state context")
                        uniforms = sensor_uniforms[..., noise_offset:noise_offset + term.frame_width]
                    newest = self._process_term(clean, term, uniforms)
                if newest.shape[-1] != term.frame_width:
                    raise ValueError(f"Analytical observation width differs for {group}/{term.name}")
                pieces.append(roll_history(old_frames, newest).flatten(-2))
                noise_offset += term.frame_width
            result[key] = torch.cat(pieces, dim=-1)
        next_amp = result["amp_history"][..., -1, :]
        if "robot.joint_pos" in result and result["robot.joint_pos"].shape[-1] == self.amp_observation_dim:
            next_amp = result["robot.joint_pos"]
        result["amp_history"] = roll_history(previous["amp_history"], next_amp)
        if "episode_length_buf" in result:
            result["episode_length_buf"] = previous["episode_length_buf"] + 1
        return self.layout.pack(result)

    def amp_reward(self, next_state: Tensor, task_reward: Tensor, discriminator: nn.Module,
                   mean: Any, std: Any, **kwargs) -> Tensor:
        return differentiable_amp_reward(self.amp_history(next_state), task_reward,
                                         self.layout.get(next_state, "amp_gate"),
                                         discriminator, mean, std, **kwargs)

    def metadata(self) -> dict[str, Any]:
        metadata = {**self.layout.metadata(), "version": 2, "actor_group": self.actor_group,
                "critic_group": self.critic_group, "actor_dim": self.actor_dim,
                "critic_dim": self.critic_dim, "action_dim": self.action_dim,
                "amp_num_frames": self.amp_num_frames,
                "amp_observation_dim": self.amp_observation_dim, "step_dt": self.step_dt,
                "sensor_noise_dim": self.sensor_noise_dim, "command_noise_dim": self.command_noise_dim,
                "observation_projection": "physical state plus external commands and sensor noise",
                "history_order": "per-term oldest-to-newest",
                "terminal_next_state": "post-reset next state; dynamics target must be masked"}
        # Preserve legacy False metadata exactly; an enabled projection changes
        # world-model semantics even though packed tensor dimensions stay equal.
        if self.project_transition_memory:
            metadata["transition_memory_projection"] = {
                "version": 1, "mode": "learned on phase entry; otherwise hold previous",
                "fields": {name: {"entry_phase_column": column} for name, column, _width
                           in self._transition_memory_fields}}
        return metadata


# Short alias for callers that select the adapter via the training backend.
SkaterAdapter = SkateWorldModelAdapter
