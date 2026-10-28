"""CPU learning-state checkpoint equivalence, without constructing a simulator.

Tests instantiate SuperMimicTrainer.__new__ and use its REAL save/load methods,
real ActorCritic/WorldModel/replay/StyleNormalizer, all four Adam optimizers and
Python/NumPy/Torch RNGs. This proves continuation of a learning update on saved
replay, not an identical physical trajectory after a new simulator is reset.
"""
import contextlib
import copy
from dataclasses import replace
import io
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from rsl_rl.modules import ActorCritic
from mjlab_husky.supermimic.config import SuperMimicConfig
from mjlab_husky.supermimic.imagination import imagine, critic_loss
from mjlab_husky.supermimic.models import WorldModel, world_model_loss
from mjlab_husky.supermimic.replay import EpisodeSequenceReplay, TransitionBatch
from mjlab_husky.supermimic.trainer import (
    SuperMimicTrainer, StyleNormalizer, checked_step, rng_state,
)


def observations(state):
    return {"policy": state[:, :4], "critic": state}


def make_policy():
    with contextlib.redirect_stdout(io.StringIO()):
        return ActorCritic(
            observations(torch.zeros(2, 6)), {"policy": ("policy",), "critic": ("critic",)},
            2, actor_hidden_dims=(8,), critic_hidden_dims=(8,),
            actor_obs_normalization=True, critic_obs_normalization=True,
        )


def make_trainer(directory, *, initialization_seed=7, config=None, warm_started=False):
    """Fully initialized finite toy replay, including every physical ring slot."""
    random.seed(initialization_seed)
    np.random.seed(initialization_seed)
    torch.manual_seed(initialization_seed)
    cfg = config or SuperMimicConfig(
        num_envs=2, iterations=6, replay_steps=12, warmup_steps=12,
        model_window=3, model_updates=1, model_batch=4,
        imagination_horizon=4, imagination_batch=4, hidden_width=16,
        discriminator_updates=1, expert_transitions=12,
    )
    trainer = SuperMimicTrainer.__new__(SuperMimicTrainer)
    trainer.cfg, trainer.device, trainer.out = cfg, "cpu", Path(directory)
    trainer.env = SimpleNamespace(device="cpu", num_envs=2)
    trainer.policy = make_policy()
    trainer.world = WorldModel(6, 2, 2, hidden_dims=(16, 16))
    trainer.discriminator = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 1))
    trainer.style_norm = StyleNormalizer(2)
    trainer.expert_frames = torch.randn(12, 5, 2)
    trainer.replay = EpisodeSequenceReplay(2, 6, 2, 2, cfg.replay_steps, "cpu")
    context = torch.tensor([[0.4, -0.3], [0.7, 0.1]])
    state = torch.randn(2, 6)
    # Every ring row is explicitly filled; checkpoint comparisons never read
    # uninitialized torch.empty storage or compare undefined bytes.
    for _ in range(cfg.replay_steps):
        action = torch.randn(2, 2) * 0.1
        robot = state[:, :4] + torch.cat((action, action), -1) * 0.05
        board = state[:, 4:] + action * 0.05 + robot[:, :2] * 0.01
        nxt = torch.cat((robot, board), -1)
        reward = -board.square().sum(-1)
        trainer.replay.add(TransitionBatch(
            state, action, nxt, reward, torch.zeros(2, dtype=torch.bool),
            torch.ones(2, dtype=torch.bool), context,
            terminated=torch.zeros(2, dtype=torch.bool),
        ))
        state = nxt
    batch = trainer.replay.sample(8, 3)
    trainer.world.fit_normalization(
        batch.states[:, :-1].flatten(0, 1), batch.actions.flatten(0, 1),
        batch.context.flatten(0, 1), next_states=batch.states[:, 1:].flatten(0, 1),
        rewards=batch.rewards.flatten(), valid=batch.valid.flatten(),
    )
    trainer.policy.update_normalization(observations(batch.states[:, 0]))
    trainer.style_norm.update(trainer.expert_frames)
    trainer.world_optimizer = torch.optim.Adam(trainer.world.parameters(), lr=cfg.model_lr)
    trainer.actor_optimizer = torch.optim.Adam(trainer.policy.actor.parameters(), lr=cfg.actor_lr)
    trainer.critic_optimizer = torch.optim.Adam(trainer.policy.critic.parameters(), lr=cfg.critic_lr)
    trainer.discriminator_optimizer = torch.optim.Adam(trainer.discriminator.parameters(), lr=cfg.discriminator_lr)
    trainer.iteration, trainer.real_steps = 0, cfg.replay_steps * 2
    trainer.normalization_fitted, trainer.warm_started = True, warm_started
    return trainer


def learning_step(trainer):
    """A real differentiable update of all four optimizers from saved replay."""
    # All three RNG streams affect updates, making a missing RNG restore visible
    # in model parameters rather than only in a superficial RNG-state assertion.
    offset = random.random() + float(np.random.uniform())
    batch = trainer.replay.sample(4, 3)
    loss = world_model_loss(trainer.world, batch, horizon=3)["loss"] * (1 + offset * 0.02)
    checked_step(loss, trainer.world_optimizer, trainer.world.parameters())
    trainer.world_optimizer.zero_grad(set_to_none=True)

    frames = torch.randn(4, 5, 2) + offset * 0.03
    trainer.style_norm.update(frames)
    normalized = trainer.style_norm(frames).mean(1)
    expert_indices = torch.randint(len(trainer.expert_frames), (4,))
    expert = trainer.style_norm(trainer.expert_frames[expert_indices]).mean(1)
    policy_score = trainer.discriminator(torch.cat((normalized, normalized), -1))
    expert_score = trainer.discriminator(torch.cat((expert, expert), -1))
    style_loss = (policy_score + 1).square().mean() + (expert_score - 1).square().mean()
    checked_step(style_loss, trainer.discriminator_optimizer, trainer.discriminator.parameters())
    trainer.discriminator_optimizer.zero_grad(set_to_none=True)

    starts = trainer.replay.sample(4, 1)

    def style_reward(state, action, nxt, task_reward, step):
        del state, step
        feature = torch.cat((nxt[:, -2:], action), -1)
        return task_reward + 0.01 * trainer.discriminator(feature).squeeze(-1)

    imagined = imagine(
        trainer.policy, trainer.world, starts.states[:, 0], observations,
        horizon=4, context=starts.context[:, 0], reward_callback=style_reward,
        additional_frozen_modules=(trainer.discriminator,),
    )
    checked_step(imagined["actor_loss"], trainer.actor_optimizer, trainer.policy.actor.parameters())
    values = trainer.policy.evaluate(imagined["critic_observations"]).reshape_as(imagined["targets"])
    value_loss = critic_loss(values, imagined["targets"], imagined["mask"])
    checked_step(value_loss, trainer.critic_optimizer, trainer.policy.critic.parameters())
    trainer.iteration += 1
    return torch.stack((loss.detach(), style_loss.detach(), imagined["actor_loss"].detach(), value_loss.detach()))


def snapshot(trainer):
    return copy.deepcopy({
        "policy": trainer.policy.state_dict(), "world": trainer.world.state_dict(),
        "discriminator": trainer.discriminator.state_dict(),
        "style_normalizer": trainer.style_norm.state_dict(),
        "expert_frames": trainer.expert_frames,
        "optimizers": {name: getattr(trainer, name + "_optimizer").state_dict()
                       for name in ("world", "actor", "critic", "discriminator")},
        "replay": trainer.replay.state_dict(), "rng": rng_state(),
        "iteration": trainer.iteration, "real_steps": trainer.real_steps,
        "normalization_fitted": trainer.normalization_fitted,
        "warm_started": trainer.warm_started,
    })


class SuperMimicTrainerCheckpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_nested_equal(self, expected, actual, path="checkpoint"):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=path)
        elif isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected, err_msg=path)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected), path)
            for key in expected:
                self.assert_nested_equal(expected[key], actual[key], path + "/" + str(key))
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(actual), len(expected), path)
            for index, value in enumerate(expected):
                self.assert_nested_equal(value, actual[index], path + "/" + str(index))
        else:
            self.assertEqual(actual, expected, path)

    def test_full_learning_state_resume_matches_uninterrupted_next_update(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = make_trainer(directory, warm_started=True)
            learning_step(trainer)  # Adam moments must be non-empty at save time.
            path = Path(directory) / "full.pt"
            trainer.save(path)
            self.assertTrue(path.exists())
            self.assertFalse(path.with_suffix(".tmp").exists())
            expected_losses = learning_step(trainer)
            expected = snapshot(trainer)

            resumed = make_trainer(directory, initialization_seed=99, warm_started=False)
            resumed.load(path)
            actual_losses = learning_step(resumed)
            torch.testing.assert_close(actual_losses, expected_losses, rtol=0, atol=0)
            self.assert_nested_equal(expected, snapshot(resumed))
            for name in ("world", "actor", "critic", "discriminator"):
                self.assertTrue(getattr(resumed, name + "_optimizer").state)

    def test_iteration_extension_allowed_but_learning_config_changes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            original = make_trainer(directory)
            learning_step(original)
            path = Path(directory) / "full.pt"
            original.save(path)
            extended = make_trainer(directory, config=replace(original.cfg, iterations=20))
            extended.load(path)
            self.assertEqual(extended.cfg.iterations, 20)
            self.assertEqual(extended.iteration, 1)
            for changed in (replace(original.cfg, model_window=4),
                            replace(original.cfg, collection_noise=0.2)):
                incompatible = make_trainer(directory, config=changed)
                before = copy.deepcopy(incompatible.policy.state_dict())
                with self.assertRaisesRegex(ValueError, "config differs"):
                    incompatible.load(path)
                self.assert_nested_equal(before, incompatible.policy.state_dict())
            below_completed = make_trainer(directory, config=replace(original.cfg, iterations=0))
            with self.assertRaisesRegex(ValueError, "below already completed"):
                below_completed.load(path)

    def test_warm_start_flag_is_preserved_for_both_true_and_false(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "full.pt"
            for value in (False, True):
                with self.subTest(warm_started=value):
                    source = make_trainer(directory, warm_started=value)
                    source.save(path)
                    restored = make_trainer(directory, initialization_seed=13, warm_started=not value)
                    restored.load(path)
                    self.assertEqual(restored.warm_started, value)
                    self.assertTrue(restored.normalization_fitted)

    def test_policy_state_dict_remains_legacy_actor_critic_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = make_trainer(directory)
            learning_step(trainer)
            path = Path(directory) / "full.pt"
            trainer.save(path)
            saved = torch.load(path, weights_only=False, map_location="cpu")
            policy = make_policy()
            policy.load_state_dict(saved["model_state_dict"])
            policy.eval()
            trainer.policy.eval()
            obs = observations(torch.tensor([[0.1, 0.2, -0.1, 0.4, 0.8, -0.2]]))
            torch.testing.assert_close(policy.act_inference(obs), trainer.policy.act_inference(obs), rtol=0, atol=0)
            self.assertFalse(any("_orig_mod" in key for key in saved["model_state_dict"]))
            self.assertTrue(any("actor_obs_normalizer" in key for key in saved["model_state_dict"]))
            self.assertIn("world", saved)
            self.assertIn("discriminator", saved)
            self.assertIn("style_normalizer", saved)

    def test_legacy_policy_checkpoint_is_not_silently_used_as_full_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = make_trainer(directory)
            path = Path(directory) / "legacy.pt"
            torch.save({"model_state_dict": trainer.policy.state_dict(), "iter": 10000}, path)
            with self.assertRaisesRegex(ValueError, "policy warm start"):
                trainer.load(path)


if __name__ == "__main__":
    unittest.main()
