"""Adapter contracts use PyTorch and a fake task; no mjlab/MuJoCo import."""

from types import SimpleNamespace as NS
from copy import deepcopy
from math import pi
import unittest

import torch
from torch import nn

from mjlab_husky.supermimic.adapter import (
    SkateWorldModelAdapter,
    StateLayout,
    differentiable_amp_reward,
)


class _TorchArrayProxy:
    """The relevant part of pinned mjlab's zero-copy Warp tensor bridge."""

    def __init__(self, tensor):
        self._tensor = tensor


class _ObservationManager:
    # Distinct term widths expose accidental whole-frame-major reshaping.
    _group_obs_term_names = {
        "policy": ["command", "actions", "phase"],
        "critic": ["command", "actions", "phase", "contact_phase"],
    }
    group_obs_term_dim = {
        "policy": [(10,), (115,), (5,)],
        "critic": [(2,), (23,), (1,), (4,)],
    }
    _group_obs_term_cfgs = {
        "policy": [NS(history_length=5)] * 3,
        "critic": [NS(history_length=0)] * 4,
    }

    def __init__(self):
        self.compute_calls = 0

    def compute(self, *args, **kwargs):
        self.compute_calls += 1
        raise AssertionError("Adapter must not resample observations/noise/history")


def _fake_task():
    batch, actions = 2, 23
    robot = NS(data=NS(
        joint_pos=torch.zeros(batch, actions),
        joint_vel=torch.zeros(batch, actions),
        default_joint_pos=torch.full((batch, actions), .2),
        default_joint_vel=torch.full((batch, actions), .1),
        gravity_vec_w=torch.tensor([[0., 0., -1.]]).repeat(batch, 1),
        forward_vec_b=torch.tensor([[1., 0., 0.]]).repeat(batch, 1),
        root_link_pos_w=torch.zeros(batch, 3),
        root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(batch, 1),
        root_link_lin_vel_w=torch.tensor([[1., 0., 0.]]).repeat(batch, 1),
        root_link_ang_vel_w=torch.tensor([[0., 0., 1.]]).repeat(batch, 1),
    ))
    command = NS(command_b=torch.tensor([[1., 0.], [0., 0.]]),
                 heading_ref_b=torch.zeros(batch), time_left=torch.ones(batch),
                 is_heading_env=torch.ones(batch, dtype=torch.bool),
                 is_standing_env=torch.zeros(batch, dtype=torch.bool),
                 command_counter=torch.zeros(batch),
                 cfg=NS(heading_command=True, rel_heading_envs=1., rel_standing_envs=0.,
                        resampling_time_range=(20., 20.),
                        ranges=NS(lin_vel_x=(0., 1.5), heading=(-pi / 4, pi / 4))))
    board = NS(data=NS(root_link_pos_w=torch.tensor([[1., 1., .2]]).repeat(batch, 1),
                       root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(batch, 1),
                       root_link_lin_vel_w=torch.tensor([[2., 0., 0.]]).repeat(batch, 1),
                       root_link_ang_vel_w=torch.zeros(batch, 3), joint_pos=torch.zeros(batch, 5),
                       forward_vec_b=torch.tensor([[1., 0., 0.]]).repeat(batch, 1)))
    env = NS(
        num_envs=batch, num_actions=actions, step_dt=.02, cycle_time=6., device="cpu",
        cfg=NS(phase_ratios=[0., .4, .5, .95, 1.]), robot=robot, skateboard=board,
        action_manager=NS(total_action_dim=actions, action=torch.ones(batch, actions),
                          prev_action=torch.zeros(batch, actions),
                          prev_prev_action=torch.zeros(batch, actions)),
        command_manager=NS(_terms={"skate": command}),
        phase_length_buf=torch.tensor([120, 150]),
        episode_length_buf=torch.ones(batch, dtype=torch.long),
        still=torch.tensor([False, True]),
        contact_phase=torch.tensor([[1., 0., 0., 0.], [0., 0., 0., 0.]]),
        last_contacts=torch.zeros(batch, 2, dtype=torch.bool),
        body_bezier_buffers={"start": torch.zeros(batch, 2, 3)},
        observation_manager=_ObservationManager(),
        scene=NS(sensors={"feet": NS(data=NS(
            force=torch.ones(batch, 1, 3), current_air_time=torch.zeros(batch, 1)))}),
        event_manager=NS(_interval_term_time_left=[torch.ones(batch)]),
        sim=NS(data=NS(qpos=_TorchArrayProxy(torch.zeros(batch, 37))),
               model=NS(body_mass=_TorchArrayProxy(torch.ones(batch, 3)))),
    )
    observations = {"policy": torch.arange(float(batch * 130)).reshape(batch, 130),
                    "critic": torch.zeros(batch, 30)}
    amp = torch.arange(float(batch * 5 * actions)).reshape(batch, 5, actions)
    return env, observations, amp


def _full_task():
    """Original G1 term names, widths, scales and uniform-noise settings."""
    env, _, amp = _fake_task()
    names = ["command", "heading", "base_ang_vel", "projected_gravity", "joint_pos",
             "joint_vel", "actions", "phase"]
    widths = [2, 1, 3, 3, 23, 23, 23, 1]
    scales = [(2., 1.), 1 / pi, .25, None, None, .05, None, None]
    noise_bounds = {"base_ang_vel": (-.2, .2), "projected_gravity": (-.05, .05),
                    "joint_pos": (-.01, .01), "joint_vel": (-1.5, 1.5)}
    policy_cfg = []
    for name, scale in zip(names, scales):
        bounds = noise_bounds.get(name)
        noise = None if bounds is None else NS(n_min=bounds[0], n_max=bounds[1], operation="add")
        params = {"sensor_name": "robot/imu_ang_vel"} if name == "base_ang_vel" else {}
        policy_cfg.append(NS(history_length=5, scale=scale, clip=None, noise=noise, params=params))
    critic_cfg = deepcopy(policy_cfg)
    for cfg in critic_cfg:
        cfg.history_length = 0
        cfg.noise = None
    extra_names = ["base_lin_vel", "heading_error", "skate_pose_local", "skate_vel_local",
                   "skate_ang_vel_local", "skateboard_roll", "trans_target_pos_b",
                   "trans_target_quat_b", "l_foot_contact_forces", "r_foot_contact_forces",
                   "l_foot_contact_forces_b", "r_foot_contact_forces_b", "contact_phase"]
    extra_widths = [3, 1, 9, 3, 3, 3, 42, 56, 3, 3, 3, 3, 4]
    force_names = ["left_feet_ground_contact", "right_feet_ground_contact",
                   "left_feet_board_contact", "right_feet_board_contact"]
    for name, sensor in zip(extra_names[8:12], force_names):
        env.scene.sensors[sensor] = NS(data=NS(force=torch.ones(2, 1, 3),
                                              current_air_time=torch.zeros(2, 1)))
    for name in extra_names:
        params = {}
        if name == "base_lin_vel":
            params = {"sensor_name": "robot/imu_lin_vel"}
        elif name in extra_names[8:12]:
            params = {"sensor_name": force_names[extra_names.index(name) - 8]}
        critic_cfg.append(NS(history_length=0, scale=1 / pi if name == "heading_error" else None,
                             clip=None, noise=None, params=params))
    env.observation_manager._group_obs_term_names = {"policy": names, "critic": names + extra_names}
    env.observation_manager.group_obs_term_dim = {
        "policy": [(width * 5,) for width in widths],
        "critic": [(width,) for width in widths + extra_widths],
    }
    env.observation_manager._group_obs_term_cfgs = {"policy": policy_cfg, "critic": critic_cfg}
    env.adapter_sensor_frames = {
        "robot/imu_ang_vel": ((.1, 0., 0.), (1., 0., 0., 0.)),
        "robot/imu_lin_vel": ((.1, 0., 0.), (1., 0., 0., 0.)),
    }
    return env, {"policy": torch.zeros(2, 395), "critic": torch.zeros(2, 215)}, amp


def _latest(adapter, state, name, group="policy"):
    key = "actor_obs" if group == "policy" else "critic_obs"
    term = next(term for term in adapter._obs_terms[group] if term.name == name)
    return adapter.layout.get(state, key)[..., term.stop - term.frame_width:term.stop]


class LayoutTests(unittest.TestCase):
    def test_roundtrip_scalars_batches_and_shape_errors(self):
        layout = StateLayout({"actor_obs": (2,), "scalar": (), "history": (3, 2)})
        fields = {"actor_obs": torch.arange(4.).reshape(2, 2),
                  "scalar": torch.tensor([2., 3.]),
                  "history": torch.arange(12.).reshape(2, 3, 2)}
        packed = layout.pack(fields)
        self.assertEqual(packed.shape, (2, 9))
        for name, value in fields.items():
            torch.testing.assert_close(layout.get(packed, name), value)
        self.assertEqual(layout.get(packed[0], "scalar").ndim, 0)
        torch.testing.assert_close(layout.pack(layout.unpack(packed)), packed)
        with self.assertRaises(ValueError):
            layout.pack({"actor_obs": fields["actor_obs"]})
        with self.assertRaises(ValueError):
            layout.pack({**fields, "history": torch.ones(2, 4, 2)})


class AdapterTests(unittest.TestCase):
    def test_capture_uses_cached_obs_owns_snapshot_and_keeps_actor_unprivileged(self):
        env, observations, amp = _fake_task()
        adapter = SkateWorldModelAdapter(env, observations, amp)
        state = adapter.capture(observations, amp)
        saved = state.clone()
        torch.testing.assert_close(adapter.actor_obs(state), observations["policy"])
        self.assertEqual(adapter.actor_obs(state).shape, (2, 130))
        self.assertIn("sim.qpos", adapter.layout)
        self.assertIn("physics.body_mass", adapter.layout)
        self.assertIn("bezier.start", adapter.layout)
        self.assertIn("sensor.feet.current_air_time", adapter.layout)
        observations["policy"].zero_()
        amp.zero_()
        env.robot.data.joint_pos.add_(100)
        env.sim.data.qpos._tensor.add_(100)
        torch.testing.assert_close(state, saved)
        self.assertEqual(env.observation_manager.compute_calls, 0)
        fields = adapter.layout.unpack(state)
        fields["sim.qpos"] = fields["sim.qpos"] + 1_000
        fields["bezier.start"] = fields["bezier.start"] + 1_000
        changed = adapter.layout.pack(fields)
        torch.testing.assert_close(adapter.actor_obs(changed), adapter.actor_obs(state))

    def test_term_major_history_action_phase_and_amp_roll(self):
        env, observations, amp = _fake_task()
        adapter = SkateWorldModelAdapter(env, observations, amp)
        state = adapter.capture(observations, amp)
        predicted = state.clone().requires_grad_()
        action = torch.full((2, 23), .25, requires_grad=True)
        next_state = adapter.project_next(state, predicted, action,
                                          exogenous={"command_uniforms": torch.full((2, 6), .5)})
        fields = adapter.layout.unpack(next_state)
        actor = fields["actor_obs"]
        torch.testing.assert_close(actor[:, :8], observations["policy"][:, :10]
                                   .reshape(2, 5, 2)[:, 1:].flatten(1))
        torch.testing.assert_close(actor[:, 10:102], observations["policy"][:, 10:125]
                                   .reshape(2, 5, 23)[:, 1:].flatten(1))
        torch.testing.assert_close(actor[:, 102:125], action)
        # Next tick is inside push->steer; a still environment follows the
        # original phase counter reset at the 151st tick, without mutating env.
        torch.testing.assert_close(actor[:, -1], torch.tensor([121 * .02 / 6, 0.]))
        torch.testing.assert_close(fields["phase_length_buf"], torch.tensor([121., 0.]))
        torch.testing.assert_close(fields["contact_phase"],
                                   torch.tensor([[0., 0., 1., 0.], [0., 0., 0., 0.]]))
        torch.testing.assert_close(fields["amp_gate"], torch.zeros(2))
        torch.testing.assert_close(fields["amp_history"][:, :-1], amp[:, 1:])
        torch.testing.assert_close(fields["amp_history"][:, -1], torch.zeros(2, 23))
        torch.testing.assert_close(fields["action.action"], action)
        torch.testing.assert_close(fields["action.prev_action"], torch.ones(2, 23))
        torch.testing.assert_close(fields["physics.body_mass"], torch.ones(2, 3))
        torch.testing.assert_close(env.phase_length_buf, torch.tensor([120, 150]))
        actor.sum().backward()
        torch.testing.assert_close(action.grad, torch.ones_like(action))

    def test_full_actor_and_board_critic_derive_from_physics_not_free_observation_heads(self):
        env, observations, amp = _full_task()
        adapter = SkateWorldModelAdapter(env, observations, amp)
        self.assertEqual((adapter.actor_dim, adapter.critic_dim, adapter.sensor_noise_dim), (395, 215, 79))
        state = adapter.capture(observations, amp)
        fields = adapter.layout.unpack(state)
        poison = torch.full((2, 395), 1e6, requires_grad=True)
        joints = torch.full((2, 23), 1.2, requires_grad=True)
        fields["actor_obs"] = poison
        fields["robot.joint_pos"] = joints
        fields["robot.joint_vel"] = torch.full((2, 23), 2.)
        fields["robot.root_link_quat_w"] = torch.tensor([[2., 0., 0., 0.]]).repeat(2, 1)
        fields["command.skate.command_b"] = torch.full((2, 2), -999.)
        predicted = adapter.layout.pack(fields)
        projected = adapter.project_next(state, predicted, torch.zeros(2, 23), exogenous={
            "command_uniforms": torch.full((2, 6), .75),
            "sensor_uniforms": torch.full((2, 79), .75),
        })
        # Original noise is added before scale: position noise=.005,
        # velocity noise=.75 then scale=.05, gyro noise=.1 then scale=.25.
        torch.testing.assert_close(_latest(adapter, projected, "joint_pos"), torch.full((2, 23), 1.005))
        torch.testing.assert_close(_latest(adapter, projected, "joint_vel"), torch.full((2, 23), .1325))
        torch.testing.assert_close(_latest(adapter, projected, "base_ang_vel"),
                                   torch.tensor([[.025, .025, .275]]).repeat(2, 1))
        torch.testing.assert_close(_latest(adapter, projected, "projected_gravity"),
                                   torch.tensor([[.025, .025, -.975]]).repeat(2, 1))
        torch.testing.assert_close(_latest(adapter, projected, "command"), torch.tensor([[2., 0.], [0., 0.]]))
        torch.testing.assert_close(_latest(adapter, projected, "heading"), torch.zeros(2, 1))
        # MuJoCo velocimeter is at an offset site: v_site=v_root+omega x r.
        torch.testing.assert_close(_latest(adapter, projected, "base_lin_vel", "critic"),
                                   torch.tensor([[1., .1, 0.]]).repeat(2, 1))
        torch.testing.assert_close(_latest(adapter, projected, "skate_pose_local", "critic"),
                                   torch.tensor([[1., 1., .2, 1., 0., 0., 0., 0., 1.]]).repeat(2, 1))
        torch.testing.assert_close(_latest(adapter, projected, "skate_vel_local", "critic"),
                                   torch.tensor([[2., 0., 0.]]).repeat(2, 1))
        torch.testing.assert_close(_latest(adapter, projected, "l_foot_contact_forces", "critic"),
                                   torch.full((2, 3), torch.log(torch.tensor(2.)).item()))
        adapter.actor_obs(projected).sum().backward()
        self.assertTrue(poison.grad is None or not poison.grad.any())
        torch.testing.assert_close(joints.grad, torch.ones_like(joints))

    def test_command_timer_and_phase_resampling_are_external_not_learned(self):
        env, observations, amp = _full_task()
        env.phase_length_buf[:] = 150
        env.contact_phase[:] = torch.tensor([0., 0., 1., 0.])
        command = env.command_manager._terms["skate"]
        command.command_b[:] = torch.tensor([1., 0.])
        command.time_left[:] = .001
        adapter = SkateWorldModelAdapter(env, observations, amp)
        state = adapter.capture(observations, amp)
        fields = adapter.layout.unpack(state)
        fields["command.skate.command_b"] = torch.full((2, 2), -999., requires_grad=True)
        fields["command.skate.time_left"] = torch.full((2,), -999.)
        fields["skateboard.root_link_quat_w"] = torch.tensor([[2 ** -.5, 0., 0., 2 ** -.5]]).repeat(2, 1)
        uniforms = torch.tensor([[.2, .1, .4, .4, .7, .8]]).repeat(2, 1).requires_grad_()
        projected = adapter.project_next(state, adapter.layout.pack(fields), torch.zeros(2, 23), exogenous={
            "command_uniforms": uniforms, "sensor_uniforms": torch.full((2, 79), .5)})
        after = adapter.layout.unpack(projected)
        # Timer samples vx=.3. Phase exit samples heading a second time using
        # independent u=.8, replacing the timer draw u=.1, exactly as original.
        expected_heading = -pi / 4 + .8 * pi / 2
        torch.testing.assert_close(after["command.skate.command_b"],
                                   torch.tensor([[.3, expected_heading]]).repeat(2, 1))
        torch.testing.assert_close(after["command.skate.heading_ref_b"], torch.full((2,), pi / 2))
        torch.testing.assert_close(after["command.skate.time_left"], torch.full((2,), 20.))
        torch.testing.assert_close(after["command.skate.command_counter"], torch.ones(2))
        torch.testing.assert_close(_latest(adapter, projected, "command"),
                                   torch.tensor([[.6, expected_heading]]).repeat(2, 1))
        head_gradient, noise_gradient = torch.autograd.grad(
            after["command.skate.command_b"].sum(),
            (fields["command.skate.command_b"], uniforms), allow_unused=True)
        self.assertTrue(head_gradient is None or not head_gradient.any())
        self.assertIsNone(noise_gradient)

    def test_training_conditions_on_external_commands_and_noise_without_physics_teacher_forcing(self):
        env, observations, amp = _full_task()
        adapter = SkateWorldModelAdapter(env, observations, amp)
        state = adapter.capture(observations, amp)
        target = adapter.layout.unpack(state)
        target["robot.joint_pos"] = torch.full((2, 23), 4., requires_grad=True)
        target["command.skate.command_b"] = torch.tensor([[.7, .2]]).repeat(2, 1)
        target["actor_obs"] = target["actor_obs"].clone()
        term = next(term for term in adapter._obs_terms["policy"] if term.name == "joint_pos")
        target["actor_obs"][:, term.stop - 23:term.stop] = 3.807  # clean(4-.2) + external .007 noise.
        truth = adapter.layout.pack(target)
        predicted = adapter.layout.unpack(state)
        joints = torch.full((2, 23), 1.2, requires_grad=True)
        predicted["robot.joint_pos"] = joints
        projected = adapter.project_next(state, adapter.layout.pack(predicted), torch.zeros(2, 23),
                                          exogenous={"next_state": truth})
        torch.testing.assert_close(_latest(adapter, projected, "joint_pos"), torch.full((2, 23), 1.007))
        torch.testing.assert_close(_latest(adapter, projected, "command"), torch.tensor([[1.4, .2]]).repeat(2, 1))
        _latest(adapter, projected, "joint_pos").sum().backward()
        torch.testing.assert_close(joints.grad, torch.ones_like(joints))
        self.assertIsNone(target["robot.joint_pos"].grad)


class AmpRewardTests(unittest.TestCase):
    def test_original_formula_phase_gate_and_input_gradient_with_frozen_discriminator(self):
        torch.manual_seed(3)
        discriminator = nn.Linear(5 * 23, 1, bias=False)
        nn.init.constant_(discriminator.weight, .001)
        frames = torch.randn(2, 5, 23, requires_grad=True)
        task_reward = torch.tensor([.2, .3])
        gate = torch.tensor([1., 0.])
        mean, std = torch.ones(23) * .1, torch.ones(23) * 2
        normalized = ((frames.detach() - mean) / std).clamp(-10., 10.)
        score = discriminator(normalized.flatten(1)).squeeze(-1)
        style = 5 * (1 - (score - 1).square() / 4).clamp_min(0)
        actual = differentiable_amp_reward(frames, task_reward, gate, discriminator, mean, std)
        torch.testing.assert_close(actual, task_reward + gate * .02 * style)
        actual.sum().backward()
        self.assertGreater(float(frames.grad[0].abs().sum()), 0)
        self.assertEqual(float(frames.grad[1].abs().sum()), 0)
        self.assertTrue(all(parameter.grad is None for parameter in discriminator.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in discriminator.parameters()))
        self.assertTrue(discriminator.training)
        lerp = differentiable_amp_reward(frames, task_reward, gate, discriminator, mean, std,
                                        use_lerp=True)
        torch.testing.assert_close(lerp, task_reward + gate * (.7 * task_reward + .3 * style - task_reward))


class TransitionMemoryTests(unittest.TestCase):
    @staticmethod
    def task(num_bodies=30):
        env, obs, amp = _full_task()
        env.body_bezier_buffers = {
            f"{direction}_start_{kind}_b": torch.randn(2, num_bodies, width)
            for direction in ("push2steer", "steer2push")
            for kind, width in (("pos", 3), ("quat", 4))}
        return env, obs, amp

    @staticmethod
    def original_contacts(counter):
        # Use the original float32 arithmetic/comparisons, including roundoff
        # at 120 ticks (.399999976 < float32 .4); do not invent integer gates.
        phase = torch.remainder(counter * .02 / 6., 1.)
        return torch.stack(((phase >= 0.) & (phase < .4),
                            (phase >= .5) & (phase < .95),
                            (phase >= .4) & (phase < .5),
                            (phase >= .95) & (phase <= 1.)), -1).float()

    def test_opt_in_schema_and_false_metadata_forward_and_gradients(self):
        env, obs, amp = self.task()
        default = SkateWorldModelAdapter(env, obs, amp)
        disabled = SkateWorldModelAdapter(env, obs, amp, project_transition_memory=False)
        active = SkateWorldModelAdapter(env, obs, amp, project_transition_memory=True)
        self.assertEqual(default.metadata(), disabled.metadata())
        self.assertNotIn("transition_memory_projection", default.metadata())
        self.assertEqual(active.metadata()["transition_memory_projection"]["version"], 1)
        state = default.capture(obs, amp).requires_grad_()
        predicted = torch.randn_like(state, requires_grad=True)
        action = torch.randn(2, 23, requires_grad=True)
        exogenous = {"next_state": state.detach()}
        first = default.project_next(state, predicted, action, exogenous=exogenous)
        second = disabled.project_next(state, predicted, action, exogenous=exogenous)
        self.assertTrue(torch.equal(first, second))
        cotangent = torch.randn_like(first)
        g1 = torch.autograd.grad((first*cotangent).sum(), (state, predicted, action), retain_graph=True)
        g2 = torch.autograd.grad((second*cotangent).sum(), (state, predicted, action))
        for left, right in zip(g1, g2):
            self.assertTrue(torch.equal(left, right))
        for invalid in (1, "true", None):
            with self.assertRaisesRegex(ValueError, "must be a boolean"):
                SkateWorldModelAdapter(env, obs, amp, project_transition_memory=invalid)
        env.body_bezier_buffers["push2steer_start_pos_b"] = torch.zeros(2, 29, 3)
        with self.assertRaisesRegex(ValueError, "matching body counts"):
            SkateWorldModelAdapter(env, obs, amp, project_transition_memory=True)
        del env.body_bezier_buffers["push2steer_start_pos_b"]
        with self.assertRaisesRegex(ValueError, "requires bezier.push2steer_start_pos_b"):
            SkateWorldModelAdapter(env, obs, amp, project_transition_memory=True)

    def test_phase_edges_hold_or_enter_only_the_correct_pair_and_preserve_other_fields(self):
        for old_counter in (119, 120, 121, 149, 150, 200, 284, 285, 299, 300):
            with self.subTest(old_counter=old_counter):
                env, obs, amp = self.task()
                env.phase_length_buf[:] = old_counter
                env.command_manager._terms["skate"].command_b[:, 0] = 1.
                env.contact_phase[:] = self.original_contacts(torch.full((2,), float(old_counter)))
                adapter = SkateWorldModelAdapter(env, obs, amp, project_transition_memory=True)
                plain = SkateWorldModelAdapter(env, obs, amp)
                state = adapter.capture(obs, amp)
                fields = adapter.layout.unpack(torch.randn_like(state))
                for number, (name, _column, _width) in enumerate(adapter._transition_memory_fields):
                    fields[name] = torch.full_like(fields[name], -100-number)
                predicted = adapter.layout.pack(fields)
                action = torch.zeros(2, 23)
                actual = adapter.layout.unpack(adapter.project_next(state, predicted, action,
                                                exogenous={"next_state": state}))
                baseline = plain.layout.unpack(plain.project_next(state, predicted, action,
                                                exogenous={"next_state": state}))
                expected_phase = self.original_contacts(torch.full((2,), float(old_counter+1)))
                torch.testing.assert_close(actual["contact_phase"], expected_phase, rtol=0, atol=0)
                memories = {name for name, _column, _width in adapter._transition_memory_fields}
                for name in actual:
                    if name not in memories:
                        self.assertTrue(torch.equal(actual[name], baseline[name]), name)
                for name, column, _width in adapter._transition_memory_fields:
                    entry = (expected_phase[:, column] > .5) & (env.contact_phase[:, column] < .5)
                    expected = torch.where(entry[:, None, None], fields[name], adapter.layout.get(state, name))
                    torch.testing.assert_close(actual[name], expected, rtol=0, atol=0)

    def test_entry_gradients_use_prediction_and_hold_keeps_previous_graph(self):
        env, obs, amp = self.task()
        env.phase_length_buf[:] = torch.tensor([120, 121])
        env.command_manager._terms["skate"].command_b[:, 0] = 1.
        env.contact_phase[:] = torch.tensor([[1., 0., 0., 0.], [0., 0., 1., 0.]])
        adapter = SkateWorldModelAdapter(env, obs, amp, project_transition_memory=True)
        state = adapter.capture(obs, amp).requires_grad_()
        predicted = torch.randn_like(state, requires_grad=True)
        action = torch.zeros(2, 23)
        result = adapter.project_next(state, predicted, action, exogenous={"next_state": state.detach()})
        # A further non-entry imagined step must retain the first entry's graph.
        next_fields = adapter.layout.unpack(result.detach())
        next_fields["phase_length_buf"] = torch.tensor([121., 122.])
        next_prediction = adapter.layout.pack(next_fields)
        result = adapter.project_next(result, next_prediction, action,
                                      exogenous={"next_state": result.detach()})
        loss = sum(adapter.layout.get(result, name).sum()
                   for name, _column, _width in adapter._transition_memory_fields)
        previous_grad, prediction_grad = torch.autograd.grad(loss, (state, predicted))
        for name, column, _width in adapter._transition_memory_fields:
            expected_prediction = torch.zeros_like(adapter.layout.get(state, name))
            if column == 2:
                expected_prediction[0] = 1.
            torch.testing.assert_close(adapter.layout.get(prediction_grad, name), expected_prediction, rtol=0, atol=0)
            torch.testing.assert_close(adapter.layout.get(previous_grad, name), 1-expected_prediction, rtol=0, atol=0)

    def test_standing_uses_previous_command_and_reset_like_stale_contact_is_safe(self):
        for old_counter, old_vx, next_vx, stale_reset in (
                (120, 1., 0., False), (120, 0., 1., False),
                (0, 1., 1., True), (150, 0., 1., False), (149, -.2, 1., False)):
            with self.subTest(counter=old_counter, old_vx=old_vx, reset=stale_reset):
                env, obs, amp = self.task()
                env.phase_length_buf[:] = old_counter
                env.command_manager._terms["skate"].command_b[:, 0] = old_vx
                env.contact_phase[:] = torch.tensor([[0., 0., 1., 0.]]) if stale_reset else 0.
                if stale_reset:
                    for memory in env.body_bezier_buffers.values():
                        memory.zero_()
                adapter = SkateWorldModelAdapter(env, obs, amp, project_transition_memory=True)
                state = adapter.capture(obs, amp)
                truth = adapter.layout.unpack(state.clone())
                truth["command.skate.command_b"][:, 0] = next_vx
                result = adapter.project_next(state, torch.randn_like(state), torch.zeros(2, 23),
                                             exogenous={"next_state": adapter.layout.pack(truth)})
                new_contact = adapter.layout.get(result, "contact_phase")
                self.assertEqual(bool(new_contact.any()), old_vx >= .1)
                for name, column, _width in adapter._transition_memory_fields:
                    entry = (new_contact[:, column] > .5) & (env.contact_phase[:, column] < .5)
                    if not entry.any():
                        torch.testing.assert_close(adapter.layout.get(result, name), adapter.layout.get(state, name), rtol=0, atol=0)
                if old_vx < .1 and old_counter in (0, 150):
                    self.assertEqual(float(adapter.layout.get(result, "phase_length_buf").abs().sum()), 0.)

    def test_real_reset_next_memory_is_masked_and_windows_do_not_cross_reset(self):
        from mjlab_husky.supermimic.replay import ReplayBuffer
        env, obs, amp = self.task()
        adapter = SkateWorldModelAdapter(env, obs, amp, project_transition_memory=True)
        state = adapter.capture(obs, amp)
        reset = adapter.layout.unpack(state.clone())
        reset["phase_length_buf"].zero_()
        for name, _column, _width in adapter._transition_memory_fields:
            reset[name].zero_()
        reset = adapter.layout.pack(reset)
        replay = ReplayBuffer(4, 2, adapter.state_dim, adapter.action_dim, seed=12)
        for step in range(3):
            terminated = torch.tensor([step == 0, False])
            truncated = torch.tensor([False, step == 1])
            replay.append(state if step == 0 else reset, torch.zeros(2, 23), reset,
                          torch.ones(2), terminated, truncated)
        batch = replay.sample_sequences(128, 3)
        boundary = batch["terminated"] | batch["truncated"]
        self.assertTrue(boundary[batch["valid"]].any())
        self.assertFalse(batch["next_state_valid"][boundary].any())
        for step in (1, 2):
            self.assertFalse((batch["valid"][:, step] & boundary[:, step-1]).any())


if __name__ == "__main__":
    unittest.main()
