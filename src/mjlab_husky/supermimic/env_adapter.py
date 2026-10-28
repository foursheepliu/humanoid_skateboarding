"""Augmented coupled robot+skateboard state without altering HUSKY physics.

Terminal physics is captured immediately before the simulator's reset. Its
observation history is necessarily stale: those next-state regression targets
are excluded rather than inventing a reset jump or advancing sensor RNG twice.
"""
import math
from types import MethodType

import torch

from .replay import TransitionBatch


def _tensor(value, device):
    if hasattr(value, "_tensor"):
        value = value._tensor
    return torch.as_tensor(value, device=device)


class CoupledStateAdapter:
    def __init__(self, env):
        self.step_env = env
        self.env = getattr(env, "unwrapped", env)
        self.device = torch.device(self.env.device)
        self.num_envs = int(self.env.num_envs)
        self.action_dim = int(self.env.action_manager.total_action_dim)
        self.layout, self._features = {}, []
        self._last_obs = self._cached_observations()
        self.policy_dim = int(self._last_obs["policy"].shape[-1])
        self.critic_dim = int(self._last_obs["critic"].shape[-1])
        self.state_dim = 0
        self._register("policy", lambda: self._last_obs["policy"])
        self._register("critic", lambda: self._last_obs["critic"])
        self._register("amp_joints", lambda: self.env.robot.data.joint_pos)
        self._amp_history = torch.zeros((self.num_envs, 5, 23), device=self.device)
        self._amp_history[:, -1] = self.env.robot.data.joint_pos
        self._register("amp_history", lambda: self._amp_history)
        data = self.env.sim.data
        for name in ("qpos", "qvel", "act", "ctrl", "qacc_warmstart"):
            if hasattr(data, name):
                self._register(name, lambda name=name: getattr(self.env.sim.data, name))
        task_defaults = {"last_contact_phase": torch.zeros((self.num_envs, 4), device=self.device),
                         **{name: torch.zeros(self.num_envs, device=self.device)
                            for name in ("just_entered_push2steer", "just_entered_steer2push",
                                         "just_exited_push2steer", "just_exited_steer2push")}}
        for name in ("contact_phase", "last_contact_phase", "phase_length_buf", "episode_length_buf",
                     "still", "last_contacts", "last_wheel_contacts", "last_contacts_b", "last_contacts_g",
                     "just_entered_push2steer", "just_entered_steer2push", "just_exited_push2steer", "just_exited_steer2push"):
            if hasattr(self.env, name) or name in task_defaults:
                self._register("phase" if name == "contact_phase" else name,
                               lambda name=name: getattr(self.env, name, task_defaults.get(name)))
        for name in sorted(getattr(self.env, "body_bezier_buffers", {})):
            self._register("bezier_" + name, lambda name=name: self.env.body_bezier_buffers[name])
        self.command_term = self.env.command_manager.get_term("skate")
        for name in ("command_b", "heading_ref_b", "is_heading_env", "is_standing_env", "time_left"):
            if hasattr(self.command_term, name):
                self._register(name, lambda name=name: getattr(self.command_term, name))
        for name in ("action", "prev_action", "prev_prev_action"):
            if hasattr(self.env.action_manager, name):
                self._register(name, lambda name=name: getattr(self.env.action_manager, name))
        # Air/contact durations are task memory, not reconstructible from qpos.
        for sensor_name, sensor in getattr(getattr(self.env, "scene", None), "sensors", {}).items():
            for name in ("current_air_time", "last_air_time", "current_contact_time", "last_contact_time"):
                value = getattr(sensor.data, name, None)
                if value is not None:
                    self._register("sensor_" + sensor_name + "_" + name,
                                   lambda sensor=sensor, name=name: getattr(sensor.data, name))
        timer_memory = getattr(getattr(self.env, "event_manager", None), "_interval_term_time_left", [])
        for index, value in enumerate(timer_memory):
            if isinstance(value, torch.Tensor):
                self._register("event_interval_time_left_" + str(index),
                               lambda index=index: self.env.event_manager._interval_term_time_left[index])
        self.history_terms = self._history_layout("policy")
        self.critic_terms = self._history_layout("critic")
        self.qpos_quaternion_slices = self._quaternion_layout()
        self.board_quaternion_slice = self._root_quaternion_slice("skateboard")
        self.context_layout = {}
        self._context = self._make_context()
        self.context_dim = self._context.shape[-1]
        self._capturing = False
        self._terminal = torch.zeros((self.num_envs, self.state_dim), device=self.device)
        self._terminal_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._original_reset = self.env._reset_idx
        if getattr(self.env, "_supermimic_adapter", None) is not None:
            raise RuntimeError("Environment already has a coupled-state adapter")
        self.env._supermimic_adapter = self

        def before_reset(instance, env_ids=None):
            ids = slice(None) if env_ids is None else env_ids
            if self._capturing:
                # No observation-manager call: never double-update history/noise.
                old = self._amp_history
                terminal_history = torch.cat((old[:, 1:], self.env.robot.data.joint_pos[:, None]), 1)
                self._amp_history = terminal_history
                self._terminal[ids] = self.observe()[ids]
                self._terminal_mask[ids] = True
                self._amp_history = old
            self._original_reset(env_ids)
            self._amp_history[ids] = 0

        self.env._reset_idx = MethodType(before_reset, self.env)

    def _cached_observations(self):
        value = getattr(self.env, "obs_buf", None)
        if not value or "policy" not in value:
            value = self.env.observation_manager.compute(update_history=False)
        return value

    def _batch(self, value):
        t = _tensor(value, self.device)
        if t.ndim == 0:
            t = t.expand(self.num_envs, 1)
        elif t.shape[0] == 1 and self.num_envs != 1:
            t = t.expand(self.num_envs, *t.shape[1:])
        if t.shape[0] != self.num_envs:
            raise ValueError("State feature lacks a leading environment dimension")
        return t.reshape(self.num_envs, -1).float()

    def _register(self, name, getter):
        value = self._batch(getter())
        self.layout[name] = slice(self.state_dim, self.state_dim + value.shape[-1])
        self.state_dim += value.shape[-1]
        self._features.append((name, getter))

    def _history_layout(self, group):
        manager = self.env.observation_manager
        names = getattr(manager, "_group_obs_term_names", {}).get(group, [])
        dimensions = getattr(manager, "_group_obs_term_dim", {}).get(group, [])
        configs = getattr(manager, "_group_obs_term_cfgs", {}).get(group, [])
        offset, result = self.layout[group].start, []
        for name, dim, cfg in zip(names, dimensions, configs):
            width = math.prod(dim)
            history = int(getattr(cfg, "history_length", 0)) or 1
            if width % history:
                raise ValueError("Policy history term shape is not divisible by history length")
            result.append({"name": name, "slice": slice(offset, offset + width),
                           "history": history, "frame_dim": width // history,
                           "scale": getattr(cfg, "scale", None)})
            offset += width
        if result and offset != self.layout[group].stop:
            raise ValueError("Term-major policy history layout does not match policy observation")
        return result

    def _quaternion_layout(self):
        model = self.env.sim.mj_model
        if "qpos" not in self.layout or not hasattr(model, "jnt_type"):
            return []
        base, result = self.layout["qpos"].start, []
        for kind, address in zip(model.jnt_type, model.jnt_qposadr):
            if int(kind) == 0:  # MuJoCo free joint: xyz + wxyz.
                result.append(slice(base + int(address) + 3, base + int(address) + 7))
            elif int(kind) == 1:  # Ball joint.
                result.append(slice(base + int(address), base + int(address) + 4))
        return result

    def _root_quaternion_slice(self, entity_name):
        model = self.env.sim.mj_model
        if "qpos" not in self.layout or not hasattr(model, "jnt_type"):
            return None
        for index, (kind, address) in enumerate(zip(model.jnt_type, model.jnt_qposadr)):
            if int(kind) != 0:
                continue
            if callable(getattr(model, "joint", None)):
                name = model.joint(index).name
            else:
                names = getattr(model, "joint_names", [])
                name = names[index] if index < len(names) else ""
            if entity_name in str(name):
                base = self.layout["qpos"].start + int(address) + 3
                return slice(base, base + 4)
        return None

    def board_yaw_from_state(self, state):
        """Heading of the predicted deck root; no future real physics is used."""
        if self.board_quaternion_slice is None:
            return None
        q = state[..., self.board_quaternion_slice].clone()
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        w, x, y, z = q.unbind(-1)
        return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))

    def _next_phase(self, previous):
        clock = previous[..., self.layout["phase_length_buf"]].squeeze(-1) + 1
        if "command_b" in self.layout:
            still = previous[..., self.layout["command_b"]][..., 0] < 0.1
        else:
            still = previous[..., self.layout["still"]].squeeze(-1) > 0.5
        half = max(1, round(self.env.cycle_time / 2 / self.env.step_dt))
        clock = torch.where(still & ((clock - 1).remainder(half) == 0), torch.zeros_like(clock), clock)
        phase = (clock * self.env.step_dt / self.env.cycle_time).remainder(1)
        ratios = self.env.cfg.phase_ratios
        phases = torch.stack(((phase >= ratios[0]) & (phase < ratios[1]),
                              (phase >= ratios[2]) & (phase < ratios[3]),
                              (phase >= ratios[1]) & (phase < ratios[2]),
                              (phase >= ratios[3]) & (phase <= ratios[4])), -1)
        return clock, phase, phases.to(previous.dtype) * (~still)[..., None], still

    def _command_events(self, previous, next_phase):
        cfg = getattr(self.command_term, "cfg", None)
        shape = previous.shape[:-1]
        timer = (previous[..., self.layout["time_left"]].squeeze(-1) <= self.env.step_dt
                 if "time_left" in self.layout else torch.zeros(shape, device=previous.device, dtype=torch.bool))
        exit_push = ((previous[..., self.layout["phase"]][..., 2] > 0.5)
                     & (next_phase[..., 2] < 0.5))
        if cfg is None or not getattr(cfg, "heading_command", False):
            exit_push = torch.zeros_like(exit_push)
        return cfg, timer, exit_push

    def exogenous_from_state(self, next_state):
        """Condition model fitting on recorded external goals, never physics.

        A heading draw is intrinsically unpredictable. These four fields are the
        only teacher-forced quantities; heading_ref is rebuilt from predicted
        board yaw, so recorded future board poses cannot leak into dynamics.
        """
        return {name: next_state[..., self.layout[name]].detach()
                for name in ("command_b", "is_heading_env", "is_standing_env", "time_left")
                if name in self.layout}

    @torch.no_grad()
    def sample_exogenous(self, previous, generator=None):
        """Sample the HUSKY command process independently of model prediction.

        Heading resamples at push->steer completion (ticks150/450/750 for the
        six-second moving cycle). Ordinary command timers also resample speed
        and standing/heading flags according to the original configuration.
        Random values are generated only when some row has an actual event;
        this preserves the distribution, not the original simulator RNG stream.
        """
        result = {name: previous[..., self.layout[name]].detach().clone()
                  for name in ("command_b", "is_heading_env", "is_standing_env", "time_left")
                  if name in self.layout}
        _, _, next_phase, _ = self._next_phase(previous)
        cfg, timer, exit_push = self._command_events(previous, next_phase)
        if "time_left" in result:
            result["time_left"].sub_(self.env.step_dt)
        if cfg is None:
            return result
        shape = previous.shape[:-1]

        def uniform(low, high):
            if low == high:
                return torch.full(shape, float(low), device=previous.device, dtype=previous.dtype)
            return torch.rand(shape, device=previous.device, dtype=previous.dtype, generator=generator) * (high - low) + low

        if bool(timer.any()):
            speed = uniform(*cfg.ranges.lin_vel_x)
            result["command_b"][..., 0] = torch.where(timer, speed, result["command_b"][..., 0])
            if "is_heading_env" in result:
                heading = uniform(0., 1.) <= cfg.rel_heading_envs
                result["is_heading_env"] = torch.where(timer[..., None], heading[..., None].float(), result["is_heading_env"])
            if "is_standing_env" in result:
                standing = uniform(0., 1.) <= cfg.rel_standing_envs
                result["is_standing_env"] = torch.where(timer[..., None], standing[..., None].float(), result["is_standing_env"])
            if "time_left" in result:
                duration = uniform(*cfg.resampling_time_range)
                result["time_left"] = torch.where(timer[..., None], duration[..., None], result["time_left"])
        heading_env = result.get("is_heading_env", torch.ones((*shape, 1), device=previous.device)).squeeze(-1) > 0.5
        heading_event = (timer | exit_push) & heading_env & bool(getattr(cfg, "heading_command", False))
        if bool(heading_event.any()):
            heading = uniform(*cfg.ranges.heading)
            result["command_b"][..., 1] = torch.where(heading_event, heading, result["command_b"][..., 1])
        if "is_standing_env" in result:
            standing = result["is_standing_env"].squeeze(-1) > 0.5
            result["command_b"] = torch.where(standing[..., None], torch.zeros_like(result["command_b"]), result["command_b"])
        return result

    def _make_context(self):
        pieces, offset = [], 0

        def append(name, value, batched=True):
            nonlocal offset
            t = _tensor(value, self.device)
            if not batched:
                t = t.reshape(1, -1).expand(self.num_envs, -1)
            t = self._batch(t)
            self.context_layout[name] = slice(offset, offset + t.shape[-1])
            offset += t.shape[-1]
            pieces.append(t)

        bridge = self.env.sim.model
        model = self.env.sim.mj_model
        for name in ("body_mass", "body_inertia", "body_ipos", "geom_friction", "dof_damping",
                     "dof_armature", "actuator_gainprm", "actuator_biasprm"):
            if hasattr(bridge, name):
                append(name, getattr(bridge, name))
            elif hasattr(model, name):
                append(name, getattr(model, name), batched=False)
        for name in ("phase_ratios", "steer_init_pos", "push_init_body_pos_b", "push_init_body_quat_b",
                     "steer_init_body_pos_b", "steer_init_body_quat_b"):
            if hasattr(self.env, name):
                append(name, getattr(self.env, name))
        append("timing", [float(self.env.step_dt), float(self.env.cycle_time)], batched=False)
        return torch.cat(pieces, -1).detach().clone()

    def context(self):
        return self._context

    def refresh_context(self):
        """Make a new immutable version after physical-parameter randomization."""
        self._context = self._make_context()
        return self._context

    def observe(self, obs=None):
        if obs is not None:
            self._last_obs = obs
        return torch.cat([self._batch(getter()) for _, getter in self._features], -1)

    def state_to_observations(self, state):
        from tensordict import TensorDict
        return TensorDict({"policy": state[..., self.layout["policy"]],
                           "critic": state[..., self.layout["critic"]]},
                          batch_size=state.shape[:-1])

    observations_from_state = state_to_observations

    def amp_features_from_state(self, state):
        return state[..., self.layout["amp_joints"]]

    def push_mask_from_state(self, state):
        return state[..., self.layout["phase"]][..., 0].clamp(0, 1)

    def project(self, previous, predicted, action=None, *, exogenous=None):
        """Enforce exact history shifts while retaining differentiable new frames."""
        result = predicted.clone()
        for term in self.history_terms:
            if term["history"] <= 1:
                continue
            sl, h, width = term["slice"], term["history"], term["frame_dim"]
            before = previous[..., sl].reshape(*previous.shape[:-1], h, width)
            after = result[..., sl].reshape(*result.shape[:-1], h, width)
            newest = after[..., -1:, :]
            if term["name"] == "command":
                newest = before[..., -1:, :]
            result[..., sl] = torch.cat((before[..., 1:, :], newest), -2).flatten(-2)
        for name in ("command_b", "is_heading_env", "is_standing_env", "still"):
            if name in self.layout:
                result[..., self.layout[name]] = previous[..., self.layout[name]]
        if "amp_history" in self.layout:
            hist = previous[..., self.layout["amp_history"]].reshape(*previous.shape[:-1], 5, 23)
            joints = result[..., self.layout["amp_joints"]][..., None, :]
            result[..., self.layout["amp_history"]] = torch.cat((hist[..., 1:, :], joints), -2).flatten(-2)
        for name in ("episode_length_buf", "phase_length_buf"):
            if name in self.layout:
                result[..., self.layout[name]] = previous[..., self.layout[name]] + 1
        if "phase_length_buf" in self.layout and "phase" in self.layout:
            clock, _, phases, still = self._next_phase(previous)
            result[..., self.layout["phase_length_buf"]] = clock[..., None]
            result[..., self.layout["phase"]] = phases
            result[..., self.layout["still"]] = still[..., None].to(result.dtype)
        allowed_exogenous = {"command_b", "is_heading_env", "is_standing_env", "time_left"}
        if exogenous is not None:
            if set(exogenous) - allowed_exogenous:
                raise ValueError("Only external command fields may be supplied to projection")
            for name, value in exogenous.items():
                if name not in self.layout or value.shape != previous[..., self.layout[name]].shape:
                    raise ValueError("External command field shape differs from the state layout")
        if "phase" in self.layout and "command_b" in self.layout:
            cfg, timer, exit_push = self._command_events(previous, result[..., self.layout["phase"]])
            for name in ("is_heading_env", "is_standing_env"):
                if name in self.layout:
                    target = predicted[..., self.layout[name]].clamp(0, 1) if exogenous is None else exogenous.get(name, previous[..., self.layout[name]])
                    result[..., self.layout[name]] = torch.where(timer[..., None], target, previous[..., self.layout[name]])
            heading_env = (result[..., self.layout["is_heading_env"]].squeeze(-1) > 0.5
                           if "is_heading_env" in self.layout else torch.ones_like(timer))
            heading_event = (timer | exit_push) & heading_env & bool(getattr(cfg, "heading_command", False))
            commands = previous[..., self.layout["command_b"]].clone()
            target = predicted[..., self.layout["command_b"]].clone() if exogenous is None else exogenous.get("command_b", commands)
            if exogenous is None and cfg is not None:
                target = torch.stack((target[..., 0].clamp(*cfg.ranges.lin_vel_x),
                                      target[..., 1].clamp(*cfg.ranges.heading) if cfg.ranges.heading is not None else target[..., 1]), -1)
            commands[..., 0] = torch.where(timer, target[..., 0], commands[..., 0])
            commands[..., 1] = torch.where(heading_event, target[..., 1], commands[..., 1])
            if "is_standing_env" in self.layout:
                standing = result[..., self.layout["is_standing_env"]].squeeze(-1) > 0.5
                commands = torch.where(standing[..., None], torch.zeros_like(commands), commands)
            result[..., self.layout["command_b"]] = commands
            if "time_left" in self.layout:
                before = previous[..., self.layout["time_left"]]
                target_time = (exogenous.get("time_left", predicted[..., self.layout["time_left"]])
                               if exogenous is not None else predicted[..., self.layout["time_left"]])
                if exogenous is None and cfg is not None and cfg.resampling_time_range[0] == cfg.resampling_time_range[1]:
                    target_time = torch.full_like(before, cfg.resampling_time_range[0])
                result[..., self.layout["time_left"]] = torch.where(timer[..., None], target_time, before - self.env.step_dt)
        # Commands stay fixed within a phase. Heading reference can change at
        # push->steer completion, and Bezier starts only at the two entry edges.
        if "phase" in self.layout:
            old_phase = previous[..., self.layout["phase"]]
            new_phase = result[..., self.layout["phase"]]
            if "last_contact_phase" in self.layout:
                result[..., self.layout["last_contact_phase"]] = old_phase
            for name, index, entering in (("just_entered_push2steer", 2, True),
                                           ("just_entered_steer2push", 3, True),
                                           ("just_exited_push2steer", 2, False),
                                           ("just_exited_steer2push", 3, False)):
                if name in self.layout:
                    flag = ((new_phase[..., index] > 0.5) & (old_phase[..., index] < 0.5)
                            if entering else (new_phase[..., index] < 0.5) & (old_phase[..., index] > 0.5))
                    result[..., self.layout[name]] = flag.to(result.dtype)[..., None]
            if "heading_ref_b" in self.layout:
                sl = self.layout["heading_ref_b"]
                board_yaw = self.board_yaw_from_state(result)
                reference = result[..., sl].clone() if board_yaw is None else board_yaw[..., None]
                result[..., sl] = torch.where(heading_event[..., None], reference, previous[..., sl])
            for name, sl in self.layout.items():
                if name.startswith("bezier_"):
                    phase_index = 2 if "push2steer" in name else 3
                    entering = (new_phase[..., phase_index] > 0.5) & (old_phase[..., phase_index] < 0.5)
                    result[..., sl] = torch.where(entering[..., None], result[..., sl], previous[..., sl])
        if action is not None:
            for current, old in (("prev_prev_action", "prev_action"), ("prev_action", "action")):
                if current in self.layout and old in self.layout:
                    result[..., self.layout[current]] = previous[..., self.layout[old]]
            if "action" in self.layout:
                result[..., self.layout["action"]] = action
        known = {}
        if "command_b" in self.layout:
            known["command"] = result[..., self.layout["command_b"]].clone()
        if "action" in self.layout:
            known["actions"] = result[..., self.layout["action"]].clone()
        if "phase_length_buf" in self.layout:
            known["phase"] = (result[..., self.layout["phase_length_buf"]].clone()
                               * self.env.step_dt / self.env.cycle_time).remainder(1)
        if "phase" in self.layout:
            known["contact_phase"] = result[..., self.layout["phase"]].clone()
        if "heading_ref_b" in self.layout and "command_b" in self.layout:
            board_yaw = self.board_yaw_from_state(result)
            if board_yaw is not None:
                error = (result[..., self.layout["heading_ref_b"]].clone()
                         + result[..., self.layout["command_b"]][..., 1:2].clone() - board_yaw[..., None])
                known["heading_error"] = (error + math.pi).remainder(2 * math.pi) - math.pi
        # Actor flattened history is term-major; critic terms have one frame.
        # Known exogenous/action fields must agree with their raw state copies.
        for term in self.history_terms + self.critic_terms:
            if term["name"] not in known:
                continue
            newest = known[term["name"]]
            if newest.shape[-1] != term["frame_dim"]:
                continue
            if term["scale"] is not None:
                newest = newest * torch.as_tensor(term["scale"], device=newest.device)
            sl, h, width = term["slice"], term["history"], term["frame_dim"]
            if h > 1:
                before = previous[..., sl].reshape(*previous.shape[:-1], h, width)
                newest = torch.cat((before[..., 1:, :], newest[..., None, :]), -2).flatten(-2)
            result[..., sl] = newest
        for sl in self.qpos_quaternion_slices:
            q = result[..., sl].clone()
            identity = torch.zeros_like(q)
            identity[..., 0] = 1
            norm = q.norm(dim=-1, keepdim=True)
            result[..., sl] = torch.where(norm > 1e-8, q / norm.clamp_min(1e-8), identity)
        return result

    @torch.no_grad()
    def capture_step(self, actions, obs=None):
        state = self.observe(obs).clone()
        old_history = self._amp_history.clone()
        self._terminal_mask.zero_()
        self._capturing = True
        try:
            result = self.step_env.step(actions)
        finally:
            self._capturing = False
        if len(result) == 5:
            next_obs, reward, terminated, truncated, extras = result
            done = terminated | truncated
        else:
            next_obs, reward, done, extras = result
            truncated = extras.get("time_outs", torch.zeros_like(done))
            terminated = done.bool() & ~truncated.bool()
        done = done.reshape(-1).bool()
        self._amp_history = torch.cat((old_history[:, 1:], self.env.robot.data.joint_pos[:, None]), 1)
        self._amp_history[done] = 0
        self._last_obs = next_obs
        next_state = self.observe().clone()
        next_state[self._terminal_mask] = self._terminal[self._terminal_mask]
        # Every ending target is excluded even if a custom env did not invoke the
        # hook. Reward/done supervision still uses its valid pre-action state.
        return TransitionBatch(state, actions.detach().clone(), next_state, reward.detach().clone(),
                               done, ~done, self.context(), terminated.reshape(-1).bool(), next_obs, extras)

    step = capture_step

    def close(self):
        if getattr(self.env, "_supermimic_adapter", None) is self:
            self.env._reset_idx = self._original_reset
            del self.env._supermimic_adapter
