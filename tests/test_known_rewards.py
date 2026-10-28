"""Only-A reward contracts with PyTorch and fake live manager configuration."""

import inspect
import math
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from mjlab_husky.supermimic.adapter import StateLayout
from mjlab_husky.supermimic.known_rewards import KnownActionRewards


def _layout(dimension=3):
    return StateLayout({"action.action": (dimension,), "action.prev_action": (dimension,),
                        "action.prev_prev_action": (dimension,), "unused_physics": (4,)})


def _state(layout, batch=2):
    return layout.pack({name: torch.zeros(batch, *shape) for name, shape in layout.shapes.items()})


def _live_fixture(dimension=3, scale_by_dt=True, dt=.125):
    module = ModuleType("mjlab.envs.mdp.rewards")
    calls = []

    def action_rate_l2(env):
        calls.append("rate")
        return (env.action_manager.action - env.action_manager.prev_action).square().sum(-1)

    def action_acc_l2(env):
        calls.append("acceleration")
        m = env.action_manager
        return (m.action - 2 * m.prev_action + m.prev_prev_action).square().sum(-1)

    for function in (action_rate_l2, action_acc_l2):
        function.__module__ = module.__name__
        setattr(module, function.__name__, function)
    terms = {"action_rate_l2": NS(func=action_rate_l2, weight=-.25, params={}),
             "action_acc_l2": NS(func=action_acc_l2, weight=-.5, params={})}
    action_manager = NS(total_action_dim=dimension, action=torch.zeros(2, dimension),
                        prev_action=torch.zeros(2, dimension), prev_prev_action=torch.zeros(2, dimension))
    env = NS(step_dt=dt, action_manager=action_manager,
             reg_reward_manager=NS(_scale_by_dt=scale_by_dt, get_term_cfg=terms.__getitem__))
    return module, env, terms, calls


class KnownActionTests(unittest.TestCase):
    def test_live_configuration_sum_and_scaling_without_extra_time_derivatives(self):
        for scaled in (True, False):
            with self.subTest(scale_by_dt=scaled):
                module, env, _, calls = _live_fixture(scale_by_dt=scaled)
                layout = _layout()
                with patch.dict(sys.modules, {module.__name__: module}):
                    known = KnownActionRewards.from_env(NS(unwrapped=env), layout)
                self.assertEqual(calls, [])  # Factory never computes a live reward.
                action = torch.tensor([[1., 2., 3.], [-2., 0., 1.]])
                expected = -.75 * torch.tensor([14., 5.]) * (.125 if scaled else 1.)
                torch.testing.assert_close(known(_state(layout), action), expected)
                metadata = known.metadata()
                self.assertEqual(metadata["weights"], {"action_rate_l2": -.25, "action_acc_l2": -.5})
                self.assertEqual(metadata["step_dt"], .125)
                self.assertEqual(metadata["scale_by_dt"], scaled)
                self.assertEqual(metadata["action_fields"], {"a[t-1]": "action.action", "a[t-2]": "action.prev_action"})
                self.assertEqual(metadata["action_dim"], 3)
                self.assertFalse(metadata["depends_on_next_state"])

    def test_pre_process_history_matches_original_post_process_formula(self):
        module, env, terms, _ = _live_fixture(dimension=2)
        layout = _layout(2)
        m = env.action_manager
        m.action[:] = torch.tensor([.2, -.1])
        m.prev_action[:] = torch.tensor([-.3, .4])
        m.prev_prev_action[:] = 999  # This PRE-state third history must not be used.
        fields = {"action.action": m.action, "action.prev_action": m.prev_action,
                  "action.prev_prev_action": m.prev_prev_action, "unused_physics": torch.zeros(2, 4)}
        before = layout.pack(fields)
        executed = torch.tensor([[.8, -.4]]).repeat(2, 1)
        with patch.dict(sys.modules, {module.__name__: module}):
            known = KnownActionRewards.from_env(env, layout)
        # Exact pinned ActionManager.process_action ordering.
        m.prev_prev_action.copy_(m.prev_action)
        m.prev_action.copy_(m.action)
        m.action.copy_(executed)
        original = sum(terms[name].weight * terms[name].func(env) * env.step_dt
                       for name in ("action_rate_l2", "action_acc_l2"))
        torch.testing.assert_close(known(before, executed), original)
        # Autoreset then clears all manager action buffers. The cached pre-state
        # and executed action still give the real terminal action penalty.
        m.action.zero_(); m.prev_action.zero_(); m.prev_prev_action.zero_()
        torch.testing.assert_close(known(before, executed), original)
        self.assertTrue((original < 0).all())

    def test_current_and_both_imagined_history_gradients_are_preserved(self):
        layout = _layout(2)
        known = KnownActionRewards(layout, -.1, -.1, .02, True)
        previous = torch.tensor([[.2, -.1]], requires_grad=True)
        older = torch.tensor([[-.3, .4]], requires_grad=True)
        wrong_third = torch.full((1, 2), 999., requires_grad=True)
        state = layout.pack({"action.action": previous, "action.prev_action": older,
                             "action.prev_prev_action": wrong_third, "unused_physics": torch.zeros(1, 4)})
        action = torch.tensor([[.8, -.4]], requires_grad=True)
        gradients = torch.autograd.grad(known(state, action).sum(),
                                        (action, previous, older, wrong_third), allow_unused=True)
        torch.testing.assert_close(gradients[0], torch.tensor([[-.0028, .0004]]))
        torch.testing.assert_close(gradients[1], torch.tensor([[.0032, .0004]]))
        torch.testing.assert_close(gradients[2], torch.tensor([[-.0004, -.0008]]))
        self.assertTrue(gradients[3] is None or not gradients[3].any())

    def test_action_gradient_penalizes_changes_and_no_internal_clipping(self):
        layout = _layout(2)
        known = KnownActionRewards(layout, -.1, -.1, .02, True)
        action = torch.tensor([[2., -2.]], requires_grad=True)
        output = known(_state(layout, 1), action)
        torch.testing.assert_close(output, torch.tensor([-.032]))  # Two sums of 8, not mean/clipped 2.
        output.sum().backward()
        torch.testing.assert_close(action.grad, torch.tensor([[-.016, .016]]))

    def test_terminal_reward_needs_no_next_state_or_survival_and_ignores_unrelated_nan(self):
        layout = _layout(2)
        known = KnownActionRewards(layout, -.1, -.1, .02, True)
        fields = layout.unpack(_state(layout, 1))
        fields["unused_physics"] = torch.full((1, 4), math.nan)
        terminal_pre_state = layout.pack(fields)
        action = torch.tensor([[.4, -.3]])
        torch.testing.assert_close(known(terminal_pre_state, action), torch.tensor([-.001]))
        self.assertEqual(tuple(inspect.signature(known).parameters), ("state", "action"))

    def test_sequence_axes_and_parameter_free_callable_do_not_change_state_dict(self):
        layout = _layout(2)
        known = KnownActionRewards(layout, -.1, -.2, .02, True)
        fields = {name: torch.zeros(2, 4, *shape) for name, shape in layout.shapes.items()}
        result = known(layout.pack(fields), torch.ones(2, 4, 2))
        torch.testing.assert_close(result, torch.full((2, 4), -.012))
        owner = torch.nn.Module()
        owner.known_reward = known
        self.assertEqual(owner.state_dict(), {})
        self.assertFalse(isinstance(known, torch.nn.Module))

    def test_rejects_changed_function_params_weights_flags_layout_and_live_dim(self):
        for change in ("function", "params", "weight", "dt", "flag", "live_dim", "history_shape"):
            with self.subTest(change=change):
                module, env, terms, _ = _live_fixture()
                layout = _layout()
                if change == "function":
                    def impostor(env):
                        pass
                    impostor.__name__, impostor.__module__ = "action_rate_l2", module.__name__
                    terms["action_rate_l2"].func = impostor
                elif change == "params":
                    terms["action_acc_l2"].params["dt"] = .02
                elif change == "weight":
                    terms["action_rate_l2"].weight = math.nan
                elif change == "dt":
                    env.step_dt = 0
                elif change == "flag":
                    env.reg_reward_manager._scale_by_dt = 1
                elif change == "live_dim":
                    env.action_manager.total_action_dim = 4
                else:
                    layout = StateLayout({"action.action": 3, "action.prev_action": 2})
                with patch.dict(sys.modules, {module.__name__: module}), self.assertRaises(ValueError):
                    KnownActionRewards.from_env(env, layout)
        with self.assertRaises(ValueError):
            KnownActionRewards(StateLayout({"action.action": 2, "action.prev_prev_action": 2}), -.1, -.1, .02, True)
        with self.assertRaises(ValueError):
            KnownActionRewards(_layout(), -.1, -.1, .02, True)(_state(_layout()), torch.ones(2, 2))
        with self.assertRaises(ValueError):
            KnownActionRewards(_layout(), torch.tensor(-.1), -.1, .02, True)


if __name__ == "__main__":
    unittest.main()
