"""Numerical contracts for coupled learned dynamics and frozen-model BPTT.

These tests prove tensor/gradient/masking contracts, not real skateboard quality.
Run with: python -m unittest discover -s tests -p test_supermimic_models.py
"""
import copy
import unittest

import torch
from torch import nn

from mjlab_husky.supermimic.models import RunningMoments, WorldModel, world_model_loss, projection_function
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


class ExternalCommandWorld(nn.Module):
    """Five physical coordinates and one externally randomized command."""
    def __init__(self, gain=0.1):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(float(gain)))
        self.delta_norm = RunningMoments(6)
        self.reward_norm = RunningMoments(1)

    def forward(self, state, action, context=None):
        increment = torch.cat((action, action[:, :1] * 0.5,
                               torch.zeros_like(action)), dim=-1)
        physical = state[:, :5] + self.gain * increment
        nxt = torch.cat((physical, state[:, 5:6]), dim=-1)
        zero_reward = physical.sum(-1) * 0
        logits = state.new_full((len(state),), -4.0)
        return {"next_state": nxt, "reward": zero_reward,
                "normalized_reward": zero_reward, "termination_logit": logits,
                "termination_probability": torch.sigmoid(logits)}


def command_projection(previous, predicted, action=None, *, exogenous=None):
    command = previous[:, 5:6] if exogenous is None else exogenous["command"]
    # Only the command is supplied externally; all physical coordinates remain
    # the world's own autoregressive prediction, including any model error.
    return torch.cat((predicted[:, :5], command), dim=-1)


def command_observations(state):
    return {"policy": torch.cat((state[:, :3], state[:, 5:6]), -1), "critic": state}


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

    def command_batch(self):
        batch, horizon = 4, 8
        actions = torch.randn(batch, horizon, 2)
        state = torch.zeros(batch, 6)
        state[:, 5] = torch.randn(batch)
        states = [state]
        true_world = ExternalCommandWorld(gain=0.2)
        with torch.no_grad():
            for step in range(horizon):
                nxt = true_world(state, actions[:, step])["next_state"]
                nxt[:, 5] = torch.randn(batch)  # Unpredictable external draws.
                states.append(nxt)
                state = nxt
        return {"states": torch.stack(states, 1), "actions": actions,
                "rewards": torch.zeros(batch, horizon),
                "dones": torch.zeros(batch, horizon, dtype=torch.bool),
                "valid": torch.ones(batch, horizon, dtype=torch.bool)}

    def test_projection_signature_compatibility_with_optional_external_commands(self):
        previous, predicted, action = torch.zeros(2, 6), torch.ones(2, 6), torch.ones(2, 2)
        external = {"command": torch.full((2, 1), 9.0)}
        two = projection_function(lambda old, pred: pred + 1)
        three = projection_function(lambda old, pred, act: pred + act.sum(-1, keepdim=True))
        torch.testing.assert_close(two(previous, predicted, action, external), predicted + 1)
        torch.testing.assert_close(three(previous, predicted, action, external), predicted + 2)
        four = projection_function(command_projection)
        torch.testing.assert_close(four(previous, predicted, action, external)[:, -1], external["command"].squeeze(-1))
        torch.testing.assert_close(four(previous, predicted, action)[:, -1], previous[:, -1])
        torch.testing.assert_close(projection_function(None)(previous, predicted, action, external), predicted)

    def test_random_command_targets_do_not_pollute_physical_model_loss_or_gradient(self):
        batch = self.command_batch()
        changed_commands = copy.deepcopy(batch)
        changed_commands["states"][:, :, 5] = torch.randn_like(batch["states"][:, :, 5]) * 100
        first, second = ExternalCommandWorld(0.1), ExternalCommandWorld(0.1)
        extract = lambda recorded: {"command": recorded[:, 5:6]}
        loss1 = world_model_loss(first, batch, project=command_projection, exogenous_from_state=extract)
        loss2 = world_model_loss(second, changed_commands, project=command_projection, exogenous_from_state=extract)
        torch.testing.assert_close(loss1["state"], loss2["state"], rtol=0, atol=0)
        loss1["loss"].backward()
        loss2["loss"].backward()
        torch.testing.assert_close(first.gain.grad, second.gain.grad, rtol=0, atol=0)
        self.assertGreater(float(first.gain.grad.abs()), 1e-6)

    def test_external_command_replay_does_not_teacher_force_physical_targets(self):
        batch = self.command_batch()
        model = ExternalCommandWorld(gain=0.2)
        extract = lambda recorded: {"command": recorded[:, 5:6]}
        matched = world_model_loss(model, batch, project=command_projection, exogenous_from_state=extract)
        self.assertLess(float(matched["state"].detach()), 1e-12)
        default = world_model_loss(model, batch, project=command_projection)
        self.assertGreater(float(default["state"].detach()), 0.01)
        corrupted_physics = copy.deepcopy(batch)
        corrupted_physics["states"][:, -1, 2] += 3
        mismatch = world_model_loss(model, corrupted_physics, project=command_projection, exogenous_from_state=extract)
        self.assertGreater(float(mismatch["state"].detach()), 0.01)
        # The target physical discrepancy remains visible in validation instead
        # of being hidden by replacing the whole next state with real data.

    def test_imagined_external_sampling_is_detached_and_keeps_actor_bptt(self):
        policy, world = TinyActorCritic(), ExternalCommandWorld(gain=0.2)
        with torch.no_grad():
            policy.critic.weight.zero_()
            policy.critic.bias.zero_()
        batch, horizon = 3, 5
        commands = [nn.Parameter(torch.randn(batch, 1)) for _ in range(horizon)]
        seen_flags = []

        def sample(previous):
            seen_flags.append(previous.requires_grad)
            # Returning existing leaf parameters ensures the learner itself
            # detaches external values, not merely callback-created operations.
            return {"command": commands[len(seen_flags) - 1]}

        trajectory = imagine(
            policy, world, torch.randn(batch, 6), command_observations,
            horizon=horizon, project=command_projection, exogenous_sampler=sample,
            reward_callback=lambda state, action, nxt, reward, step: -nxt[:, 2].square(),
        )
        trajectory["actor_loss"].backward()
        self.assertEqual(seen_flags, [False] * horizon)
        self.assertTrue(all(value.grad is None for value in commands))
        self.assertTrue(all(p.grad is None for p in world.parameters()))
        self.assertTrue(all(p.grad is None for p in policy.critic.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in policy.actor.parameters()), 1e-6)
        for step, command in enumerate(commands):
            torch.testing.assert_close(trajectory["states"][:, step + 1, 5:6], command.detach())

    def test_omitting_external_callbacks_preserves_existing_behavior(self):
        policy, world = TinyActorCritic(), CoupledOracle()
        starts = torch.randn(3, 6)
        implicit = imagine(policy, world, starts, observations, horizon=4)
        explicit = imagine(policy, world, starts, observations, horizon=4, exogenous_sampler=None)
        for name in ("objective", "states", "targets", "rewards"):
            torch.testing.assert_close(implicit[name], explicit[name], rtol=0, atol=0)
        batch = self.batch()
        model = self.fitted_model(batch)
        implicit_loss = world_model_loss(model, batch)
        explicit_loss = world_model_loss(model, batch, exogenous_from_state=None)
        for key in implicit_loss:
            torch.testing.assert_close(implicit_loss[key], explicit_loss[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
