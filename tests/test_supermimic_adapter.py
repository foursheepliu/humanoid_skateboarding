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


if __name__ == "__main__":
    unittest.main()
