"""CPU boundary tests for the original HUSKY exogenous command process."""
import math
from types import SimpleNamespace
import unittest

import torch

from test_supermimic_replay import CoupledStateAdapter, FakeEnvironment


class CommandEnvironment(FakeEnvironment):
    def __init__(self, num_envs=2):
        super().__init__(num_envs)
        self.term.cfg = SimpleNamespace(
            heading_command=True, rel_heading_envs=1.0, rel_standing_envs=0.0,
            ranges=SimpleNamespace(lin_vel_x=(0.3, 1.2), heading=(-0.5, 0.5)),
            resampling_time_range=(20.0, 20.0))
        self.term.command_b[:] = torch.tensor([0.8, 0.1])
        self.term.time_left = torch.full((num_envs,), 17.)
        self.term.heading_ref_b[:] = 0.2
        self.sim.mj_model.joint = lambda i: SimpleNamespace(
            name=("robot/floating_base_joint" if i == 0 else "skateboard/floating_base_joint_skateboard"))


class CommandProjectionTests(unittest.TestCase):
    def setup_adapter(self, clocks):
        env = CommandEnvironment(len(clocks))
        env.phase_length_buf[:] = torch.tensor(clocks).float()
        phase = env.phase_length_buf * env.step_dt / env.cycle_time % 1
        env.contact_phase[:] = torch.stack(((phase >= 0) & (phase < .4),
            (phase >= .5) & (phase < .95), (phase >= .4) & (phase < .5),
            (phase >= .95)), -1).float()
        adapter = CoupledStateAdapter(env)
        self.addCleanup(adapter.close)
        return env, adapter

    def test_only_exit_push2steer_resamples_heading_at_150_450_750(self):
        env, adapter = self.setup_adapter([148, 149, 449, 749])
        state = adapter.observe()
        generator = torch.Generator().manual_seed(17)
        external = adapter.sample_exogenous(state, generator)
        torch.testing.assert_close(external["command_b"][:, 0], torch.full((4,), .8))
        self.assertEqual(float(external["command_b"][0, 1]), float(state[0, adapter.layout["command_b"]][1]))
        self.assertTrue(bool((external["command_b"][1:, 1] != .1).all()))
        self.assertTrue(bool((external["command_b"][1:, 1].abs() <= .5).all()))
        projected = adapter.project(state, state + 100, torch.zeros(4, 3), exogenous=external)
        torch.testing.assert_close(projected[:, adapter.layout["phase_length_buf"]].squeeze(-1),
                                   torch.tensor([149., 150., 450., 750.]))
        torch.testing.assert_close(projected[:, adapter.layout["command_b"]], external["command_b"])

    def test_reference_is_predicted_board_yaw_and_non_event_stays_fixed(self):
        env, adapter = self.setup_adapter([149, 148])
        state = adapter.observe()
        predicted = state.detach().clone().requires_grad_(True)
        q = torch.tensor([math.cos(.6), 0., 0., math.sin(.6)])  # yaw1.2rad
        with torch.no_grad():
            predicted[:, adapter.board_quaternion_slice] = q
        external = adapter.exogenous_from_state(state)
        external["command_b"] = torch.tensor([[.8, -.3], [99., 99.]])
        result = adapter.project(state, predicted, torch.zeros(2, 3), exogenous=external)
        torch.testing.assert_close(result[:, adapter.layout["heading_ref_b"]].squeeze(-1), torch.tensor([1.2, .2]))
        torch.testing.assert_close(result[1, adapter.layout["command_b"]], torch.tensor([.8, .1]))
        result[:, adapter.layout["heading_ref_b"]].sum().backward()
        gradient = predicted.grad[:, adapter.board_quaternion_slice]
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient[0].abs().sum()), 0.)
        self.assertEqual(float(gradient[1].abs().sum()), 0.)

    def test_timer_resamples_speed_flags_and_clock_but_not_physics(self):
        env, adapter = self.setup_adapter([100, 100])
        env.term.time_left[:] = torch.tensor([.01, 5.])
        state = adapter.observe()
        external = adapter.sample_exogenous(state, torch.Generator().manual_seed(3))
        self.assertTrue(.3 <= float(external["command_b"][0, 0]) <= 1.2)
        torch.testing.assert_close(external["command_b"][1], torch.tensor([.8, .1]))
        torch.testing.assert_close(external["time_left"].squeeze(-1), torch.tensor([20., 4.98]))
        self.assertEqual(set(external), {"command_b", "is_heading_env", "is_standing_env", "time_left"})
        result = adapter.project(state, state + 1, torch.zeros(2, 3), exogenous=external)
        torch.testing.assert_close(result[:, adapter.layout["time_left"]].squeeze(-1), torch.tensor([20., 4.98]))
        illegal = {**external, "heading_ref_b": torch.zeros(2, 1)}
        with self.assertRaises(ValueError):
            adapter.project(state, state, exogenous=illegal)

    def test_no_event_preserves_commands_and_does_not_consume_sampler_rng(self):
        env, adapter = self.setup_adapter([100, 200])
        state = adapter.observe()
        generator = torch.Generator().manual_seed(4)
        before = generator.get_state().clone()
        external = adapter.sample_exogenous(state, generator)
        torch.testing.assert_close(generator.get_state(), before)
        torch.testing.assert_close(external["command_b"], state[:, adapter.layout["command_b"]])
        predicted = state + 100
        a = adapter.project(state, predicted, torch.zeros(2, 3))
        b = adapter.project(state, predicted, torch.zeros(2, 3))
        torch.testing.assert_close(a, b)  # deterministic fallback, no hidden RNG

    def test_supplied_recorded_goal_updates_term_major_actor_and_critic(self):
        env, adapter = self.setup_adapter([149, 449])
        state = adapter.observe()
        external = adapter.exogenous_from_state(state)
        external["command_b"] = torch.tensor([[.8, -.4], [.8, .35]])
        result = adapter.project(state, state, torch.ones(2, 3), exogenous=external)
        for term in adapter.history_terms:
            if term["name"] == "command":
                history = result[:, term["slice"]].reshape(2, 5, 2)
                old = state[:, term["slice"]].reshape(2, 5, 2)
                torch.testing.assert_close(history[:, :-1], old[:, 1:])
                torch.testing.assert_close(history[:, -1], external["command_b"] * torch.tensor([2., 1.]))
        for term in adapter.critic_terms:
            if term["name"] == "command":
                torch.testing.assert_close(result[:, term["slice"]], external["command_b"] * torch.tensor([2., 1.]))

    def test_timer_low_high_speed_switch_updates_still_one_step_later(self):
        env, adapter = self.setup_adapter([150, 150])
        env.term.cfg.ranges.lin_vel_x = (0., 1.5)
        env.term.command_b[:, 0] = torch.tensor([.05, .8])
        env.still[:] = torch.tensor([False, True])  # Previous command-process lag.
        env.term.time_left[:] = .01
        state = adapter.observe()
        external = adapter.exogenous_from_state(state)
        external["command_b"] = torch.tensor([[.8, .1], [.05, .1]])
        external["time_left"] = torch.full((2, 1), 20.)
        result = adapter.project(state, state, torch.zeros(2, 3), exogenous=external)
        torch.testing.assert_close(result[:, adapter.layout["still"]].squeeze(-1), torch.tensor([1., 0.]))
        torch.testing.assert_close(result[:, adapter.layout["phase_length_buf"]].squeeze(-1), torch.tensor([0., 151.]))
        self.assertEqual(float(result[0, adapter.layout["phase"]].sum()), 0.)
        self.assertEqual(float(result[1, adapter.layout["phase"]][1]), 1.)
        second = adapter.project(result, result, torch.zeros(2, 3), exogenous=adapter.exogenous_from_state(result))
        torch.testing.assert_close(second[:, adapter.layout["still"]].squeeze(-1), torch.tensor([0., 1.]))
        self.assertEqual(float(second[0, adapter.layout["phase"]][0]), 1.)
        self.assertEqual(float(second[1, adapter.layout["phase"]].sum()), 0.)


if __name__ == "__main__":
    unittest.main()
