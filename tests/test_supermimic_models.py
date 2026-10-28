"""Numerical contracts for coupled learned dynamics and frozen-model BPTT.

These tests prove tensor/gradient/masking contracts, not real skateboard quality.
Run with: python -m unittest discover -s tests -p test_supermimic_models.py
"""
import copy
import unittest

import torch
from torch import nn

from mjlab_husky.supermimic.models import RunningMoments, WorldModel, world_model_loss
from mjlab_husky.supermimic.imagination import imagine, gae_targets, critic_loss


class TinyActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2))
        self.critic = nn.Linear(6, 1)

    def act_inference(self, obs):
        return self.actor(obs["policy"])

    def evaluate(self, obs):
        return self.critic(obs["critic"])


def observations(state):
    # Last two coordinates act as the coupled board state, unavailable to actor.
    return {"policy": state[:, :4], "critic": state}


class CoupledOracle(nn.Module):
    """Known tensor dynamics solely for checking gradient direction and freezing."""
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.1))

    def forward(self, state, action, context=None):
        robot_next = state[:, :4] + torch.cat((action, action), dim=-1) * self.gain
        board_next = state[:, 4:] + action * self.gain + robot_next[:, :2] * 0.02
        next_state = torch.cat((robot_next, board_next), dim=-1)
        return {"next_state": next_state, "reward": -board_next.square().sum(-1),
                "termination_probability": state.new_full((len(state),), 0.01)}


class SuperMimicModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)

    def batch(self, batch=8, horizon=8):
        states = [torch.randn(batch, 6)]
        actions = torch.randn(batch, horizon, 2)
        dynamics = CoupledOracle()
        rewards = []
        with torch.no_grad():
            for step in range(horizon):
                prediction = dynamics(states[-1], actions[:, step])
                states.append(prediction["next_state"])
                rewards.append(prediction["reward"])
        return {"states": torch.stack(states, 1), "actions": actions,
                "rewards": torch.stack(rewards, 1),
                "dones": torch.zeros(batch, horizon, dtype=torch.bool),
                "valid": torch.ones(batch, horizon, dtype=torch.bool)}

    def fitted_model(self, batch):
        model = WorldModel(6, 2, hidden_dims=(32, 32))
        model.fit_normalization(batch["states"][:, :-1].flatten(0, 1),
                                batch["actions"].flatten(0, 1),
                                next_states=batch["states"][:, 1:].flatten(0, 1),
                                rewards=batch["rewards"].flatten(),
                                valid=batch["valid"].flatten())
        return model

    def test_real_normalization_merge_checkpoint_and_input_gradient(self):
        data = torch.randn(43, 6) * 3 + 5
        norm = RunningMoments(6)
        norm.update(data[:13])
        norm.update(data[13:])
        torch.testing.assert_close(norm.mean, data.mean(0), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(norm.variance, data.var(0, unbiased=False), atol=1e-6, rtol=1e-6)
        restored = RunningMoments(6)
        restored.load_state_dict(copy.deepcopy(norm.state_dict()))
        value = data[:2].clone().requires_grad_()
        torch.testing.assert_close(restored(value), norm(value))
        restored(value).sum().backward()
        self.assertGreater(float(value.grad.abs().sum()), 0)
        before = copy.deepcopy(norm.state_dict())
        with self.assertRaises(FloatingPointError):
            norm.update(torch.full((2, 6), float("nan")))
        for name, item in before.items():
            torch.testing.assert_close(norm.state_dict()[name], item)

    def test_eight_step_autoregressive_model_can_fit_coupled_transitions(self):
        batch = self.batch()
        model = self.fitted_model(batch)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        before = float(world_model_loss(model, batch)["loss"].detach())
        for _ in range(60):
            optimizer.zero_grad(set_to_none=True)
            result = world_model_loss(model, batch)
            result["loss"].backward()
            optimizer.step()
        after = float(world_model_loss(model, batch)["loss"].detach())
        self.assertLess(after, before * 0.65)
        restored = WorldModel(**model.architecture())
        restored.load_state_dict(copy.deepcopy(model.state_dict()))
        x, u = batch["states"][:, 0], batch["actions"][:, 0]
        for key, value in model(x, u).items():
            torch.testing.assert_close(value, restored(x, u)[key], rtol=0, atol=0)

    def test_terminal_next_state_excluded_but_terminal_reward_is_trained(self):
        batch = self.batch(horizon=3)
        batch["dones"][:, 1] = True
        batch["valid"][:, 1:] = False
        model = self.fitted_model(batch)
        base = world_model_loss(model, batch, horizon=3)
        changed = copy.deepcopy(batch)
        changed["states"][:, 2:] = 1e5
        changed["actions"][:, 2] = -1e5
        changed["rewards"][:, 2] = 1e5
        changed["dones"][:, 2] = True
        result = world_model_loss(model, changed, horizon=3)
        torch.testing.assert_close(result["loss"], base["loss"])
        self.assertEqual(float(result["state_transitions"]), 8)
        self.assertEqual(float(result["labeled_transitions"]), 16)
        changed["rewards"][:, 1] += 5
        self.assertGreater(float(world_model_loss(model, changed, horizon=3)["reward"].detach()), float(result["reward"].detach()))

    def test_time_limit_cuts_sequence_without_becoming_physical_failure(self):
        batch = self.batch(horizon=3)
        batch["dones"][:, 1] = True
        batch["terminated"] = torch.zeros_like(batch["dones"])
        model = self.fitted_model(batch)
        trunc = world_model_loss(model, batch, horizon=3)
        batch["terminated"][:, 1] = True
        physical = world_model_loss(model, batch, horizon=3)
        self.assertLess(float(trunc["termination"].detach()), float(physical["termination"].detach()))
        self.assertEqual(float(trunc["labeled_transitions"]), 16)

    def test_actor_gradients_flow_through_frozen_world_and_board(self):
        policy, world = TinyActorCritic(), CoupledOracle()
        keys = list(policy.state_dict())
        weights = copy.deepcopy(world.state_dict())
        trajectory = imagine(policy, world, torch.randn(4, 6), observations, horizon=32)
        trajectory["actor_loss"].backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in policy.actor.parameters()), 1e-6)
        self.assertTrue(all(p.grad is None for p in world.parameters()))
        self.assertTrue(all(p.grad is None for p in policy.critic.parameters()))
        self.assertFalse(trajectory["targets"].requires_grad)
        self.assertTrue(all(not x.requires_grad for x in trajectory["critic_observations"].values()))
        self.assertEqual(trajectory["targets"].shape, (4, 32))
        self.assertEqual(trajectory["critic_observations"]["policy"].shape, (128, 4))
        self.assertEqual(keys, list(policy.state_dict()))
        for name, expected in weights.items():
            torch.testing.assert_close(world.state_dict()[name], expected, rtol=0, atol=0)

    def test_terminal_critic_bootstrap_retains_actor_state_gradient(self):
        policy, world = TinyActorCritic(), CoupledOracle()
        with torch.no_grad():
            policy.critic.weight.fill_(1)
            policy.critic.bias.zero_()
        trajectory = imagine(policy, world, torch.randn(3, 6), observations, horizon=5,
                             reward_callback=lambda state, action, nxt, reward, step: reward * 0)
        trajectory["actor_loss"].backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in policy.actor.parameters()), 1e-6)
        self.assertTrue(all(p.grad is None for p in policy.critic.parameters()))

    def test_adversarial_callback_keeps_style_input_gradient_but_freezes_discriminator(self):
        policy, world, discriminator = TinyActorCritic(), CoupledOracle(), nn.Linear(6, 1)
        callback = lambda state, action, nxt, reward, step: reward + discriminator(nxt).squeeze(-1)
        trajectory = imagine(policy, world, torch.randn(3, 6), observations, horizon=4,
                             reward_callback=callback, additional_frozen_modules=(discriminator,))
        trajectory["actor_loss"].backward()
        self.assertTrue(all(p.grad is None for p in discriminator.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in policy.actor.parameters()), 1e-6)

    def test_gae_terminal_and_padding_boundaries_and_detached_targets(self):
        reward = torch.tensor([[1., 2., 1000.]], requires_grad=True)
        continuation = torch.tensor([[1., 0., 1.]])
        values = torch.tensor([[0., 0., 0., 99.]], requires_grad=True)
        valid = torch.tensor([[True, True, False]])
        result = gae_targets(reward, continuation, values, gamma=1, lam=1, valid=valid)
        torch.testing.assert_close(result, torch.tensor([[3., 2., 0.]]))
        self.assertFalse(result.requires_grad)
        loss = critic_loss(torch.tensor([[4., 1., 9000.]]), result, valid)
        torch.testing.assert_close(loss, torch.tensor(2.))

    def test_context_required_and_all_predictions_remain_fp32(self):
        model = WorldModel(6, 2, context_dim=3, hidden_dims=(16,), mixed_precision="bf16")
        with self.assertRaises(ValueError):
            model(torch.randn(2, 6), torch.randn(2, 2))
        result = model(torch.randn(2, 6), torch.randn(2, 2), torch.randn(2, 3))
        self.assertTrue(all(x.dtype == torch.float32 for x in result.values()))


if __name__ == "__main__":
    unittest.main()
