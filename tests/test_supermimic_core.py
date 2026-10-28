"""Core tests require PyTorch only; no MuJoCo or robot assets are imported."""

import copy
import unittest

import torch

from mjlab_husky.supermimic.models import WorldModel, frozen_parameters
from mjlab_husky.supermimic.replay import ReplayBuffer


class ReplayTests(unittest.TestCase):
    def test_windows_respect_both_reset_types_and_ring_overwrite(self):
        replay = ReplayBuffer(4, 2, 1, 1, seed=1)
        for t in range(8):
            state = torch.tensor([[float(t)], [float(t + 100)]])
            term = torch.tensor([t == 5, False])
            timeout = torch.tensor([False, t == 6])
            replay.append(state, torch.zeros(2, 1), state + 1, torch.ones(2), term, timeout)
        batch = replay.sample_sequences(512, 5)
        self.assertEqual(len(replay), 8)
        self.assertTrue((batch["state"][:, 0, 0].remainder(100) >= 4).all())
        for b in range(512):
            valid = batch["valid"][b]
            self.assertTrue(valid[0])
            for t in range(1, 5):
                if valid[t]:
                    self.assertTrue(valid[t - 1])
                    self.assertFalse(batch["terminated"][b, t - 1] | batch["truncated"][b, t - 1])
                    self.assertEqual(float(batch["state"][b, t, 0]),
                                     float(batch["state"][b, t - 1, 0]) + 1)
        boundary = batch["terminated"] | batch["truncated"]
        self.assertTrue(boundary[batch["valid"]].any())
        self.assertFalse(batch["next_state_valid"][boundary].any())

    def test_checkpoint_restores_sampling_generator_and_partial_ring(self):
        replay = ReplayBuffer(5, 1, 2, 1, seed=5)
        for t in range(2):
            s = torch.ones(1, 2) * t
            replay.append(s, torch.zeros(1, 1), s + 1, torch.ones(1),
                          torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool))
        saved = replay.state_dict()
        restored = ReplayBuffer(5, 1, 2, 1)
        restored.load_state_dict(saved)
        expected = replay.sample_sequences(16, 4)
        actual = restored.sample_sequences(16, 4)
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key])
        self.assertEqual(saved["data"]["state"].shape[0], 2)


class WorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def toy_batch(self, batch_size=48, horizon=4):
        state = torch.randn(batch_size, 2)
        actions = torch.randn(batch_size, horizon, 2)
        states, next_states, rewards = [], [], []
        for t in range(horizon):
            states.append(state)
            state = state + .1 * actions[:, t]
            next_states.append(state)
            rewards.append(-state.square().mean(-1))
        return {"state": torch.stack(states, 1), "next_state": torch.stack(next_states, 1),
                "action": actions, "reward": torch.stack(rewards, 1),
                "terminated": torch.zeros(batch_size, horizon, dtype=torch.bool),
                "truncated": torch.zeros(batch_size, horizon, dtype=torch.bool),
                "valid": torch.ones(batch_size, horizon, dtype=torch.bool),
                "next_state_valid": torch.ones(batch_size, horizon, dtype=torch.bool)}

    def test_autoregressive_loss_falls_and_normalization_counts_real_data_once(self):
        torch.manual_seed(3)
        model = WorldModel(2, 2, hidden_dims=(32, 32), learning_rate=.01)
        batch = self.toy_batch()
        model.update_normalization(batch["state"], batch["action"], batch["next_state"], batch["reward"])
        count = int(model.input_norm.count)
        initial = float(model.sequence_loss(batch)["loss"].detach())
        for _ in range(180):
            metrics = model.fit(batch)
        final = float(model.sequence_loss(batch)["loss"].detach())
        self.assertLess(final, initial * .25)
        self.assertEqual(int(model.input_norm.count), count)
        self.assertEqual(model.optimizer_updates, 180)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))
        clone = WorldModel(2, 2, hidden_dims=(32, 32), learning_rate=.01)
        clone.load_training_state_dict(copy.deepcopy(model.training_state_dict()))
        sample = (batch["state"][:, 0], batch["action"][:, 0])
        for a, b in zip(model(*sample), clone(*sample)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_frozen_model_keeps_32_step_actor_gradient_and_statistics_fixed(self):
        torch.manual_seed(7)
        model = WorldModel(2, 2, hidden_dims=(16, 16))
        torch.nn.init.normal_(model.delta_head.weight, std=.02)
        actor = torch.nn.Linear(2, 2)
        before = copy.deepcopy(model.state_dict())
        state = torch.randn(8, 2)
        objective = 0
        with frozen_parameters(model):
            for _ in range(32):
                prediction = model(state, torch.tanh(actor(state)))
                objective = objective + prediction.reward.mean()
                state = prediction.next_state
            (objective + state.square().mean()).backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in actor.parameters()), 0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_timeout_is_not_positive_termination_and_reset_delta_is_masked(self):
        model = WorldModel(2, 2, hidden_dims=(8,))
        with torch.no_grad():
            model.termination_head.weight.zero_()
            model.termination_head.bias.fill_(8)
        batch = self.toy_batch(batch_size=2, horizon=1)
        batch["truncated"].fill_(True)
        batch["next_state_valid"].fill_(False)
        batch["next_state"].fill_(1e6)  # Deliberately invalid auto-reset jump.
        losses = model.sequence_loss(batch)
        self.assertEqual(float(losses["state_loss"].detach()), 0)
        self.assertGreater(float(losses["termination_loss"].detach()), 7.9)

    def test_task_projection_is_applied_before_loss_and_recursive_transition(self):
        model = WorldModel(2, 2, hidden_dims=(8,))
        batch = self.toy_batch(batch_size=2, horizon=3)
        batch["state"].zero_()
        batch["next_state"].zero_()
        for t in range(3):
            batch["state"][:, t, 0] = t
            batch["next_state"][:, t, 0] = t + 1
        self.assertGreater(float(model.sequence_loss(batch)["state_loss"].detach()), 0)
        calls = []
        def project(previous, predicted, action):
            calls.append(previous.detach().clone())
            return torch.cat((previous[:, :1] + 1, predicted[:, 1:]), -1)
        model.project_next = project
        self.assertEqual(float(model.sequence_loss(batch)["state_loss"].detach()), 0)
        self.assertEqual(len(calls), 3)
        torch.testing.assert_close(calls[-1][:, 0], torch.full((2,), 2.0))

    def test_known_reward_normalizes_residual_but_returns_total_for_all_reward_labels(self):
        known = lambda state, action: -.2 * (action - state[:, :1]).square().sum(-1)
        model = WorldModel(2, 1, hidden_dims=(8,), known_reward=known)
        state = torch.tensor([[1., 2.], [2., 3.], [3., 4.]])
        action = torch.tensor([[0.], [2.], [-1.]])
        residual = torch.tensor([1., 3., 5.])
        reward = known(state, action) + residual
        # Physical terminal and timeout both have invalid reset next states,
        # but both action costs and residual reward labels remain observable.
        next_valid = torch.tensor([True, False, False])
        next_state = torch.where(next_valid[:, None], state, torch.full_like(state, 1e6))
        model.update_normalization(state, action, next_state, reward, next_valid)
        self.assertEqual(int(model.input_norm.count), 3)
        self.assertEqual(int(model.delta_norm.count), 1)
        self.assertEqual(int(model.reward_norm.count), 3)
        torch.testing.assert_close(model.reward_norm.mean.float(), residual.mean().reshape(1))
        torch.testing.assert_close(model.reward_norm.variance.float(), residual.var(unbiased=False).reshape(1))
        with torch.no_grad():
            model.reward_head.weight.zero_()
            model.reward_head.bias.zero_()
        torch.testing.assert_close(model(state, action).reward, known(state, action) + residual.mean())
        plain = WorldModel(2, 1, hidden_dims=(8,))
        self.assertEqual(model.state_dict().keys(), plain.state_dict().keys())
        self.assertNotIn("reward_representation", plain.training_state_dict())
        self.assertEqual(model.training_state_dict()["reward_representation"], "action_residual")

    def test_residual_sequence_loss_keeps_failure_and_timeout_rewards(self):
        known = lambda state, action: -.2 * (action - state[..., :1]).square().sum(-1)
        model = WorldModel(2, 1, hidden_dims=(8,), known_reward=known)
        state = torch.tensor([[[1., 0.]], [[2., 0.]], [[3., 0.]]])
        action = torch.tensor([[[0.]], [[2.]], [[-1.]]])
        batch = {"state": state, "action": action,
                 "next_state": torch.tensor([[[1., 0.]], [[1e6, 1e6]], [[-1e6, -1e6]]]),
                 "reward": known(state, action) + 2.,
                 "terminated": torch.tensor([[False], [True], [False]]),
                 "truncated": torch.tensor([[False], [False], [True]]),
                 "valid": torch.ones(3, 1, dtype=torch.bool),
                 "next_state_valid": torch.tensor([[True], [False], [False]])}
        model.update_normalization(state, action, batch["next_state"], batch["reward"], batch["next_state_valid"])
        with torch.no_grad():
            model.reward_head.weight.zero_()
            model.reward_head.bias.zero_()
        losses = model.sequence_loss(batch)
        self.assertLess(float(losses["reward_loss"].detach()), 1e-8)
        self.assertEqual(float(losses["state_loss"].detach()), 0.)
        batch["reward"][1:] += .01
        self.assertAlmostEqual(float(model.sequence_loss(batch)["reward_loss"].detach()), 2 / 3, places=4)

    def test_known_reward_frozen_h32_gradients_and_representation_checkpoint_guard(self):
        torch.manual_seed(3)
        known = lambda state, action: -.2 * (action - state[..., :1]).square().sum(-1)
        model = WorldModel(2, 1, hidden_dims=(8,), known_reward=known)
        with torch.no_grad():
            model.reward_head.weight.zero_()
            model.reward_head.bias.zero_()
        actor = torch.nn.Linear(2, 1)
        initial = copy.deepcopy(model.state_dict())
        state, objective = torch.randn(4, 2), 0
        with frozen_parameters(model):
            for _ in range(32):
                prediction = model(state, actor(state))
                objective += prediction.reward.mean()
                state = prediction.next_state
            objective.backward()
        self.assertGreater(sum(float(parameter.grad.abs().sum()) for parameter in actor.parameters()), 0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, initial[key]))
        saved = copy.deepcopy(model.training_state_dict())
        restored = WorldModel(2, 1, hidden_dims=(8,), known_reward=known)
        restored.load_training_state_dict(saved)
        sample = torch.randn(4, 2), torch.randn(4, 1)
        self.assertTrue(torch.equal(model(*sample).reward, restored(*sample).reward))
        plain = WorldModel(2, 1, hidden_dims=(8,))
        before = copy.deepcopy(plain.state_dict())
        with self.assertRaisesRegex(ValueError, "reward representation differs"):
            plain.load_training_state_dict(saved)
        for key, value in plain.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))
        with self.assertRaisesRegex(ValueError, "reward representation differs"):
            model.load_training_state_dict(plain.training_state_dict())


if __name__ == "__main__":
    unittest.main()
