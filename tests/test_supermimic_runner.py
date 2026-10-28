"""Actual bundled ActorCritic with small deterministic physics/AMP fixtures.

These exercise learning/checkpoint contracts without MuJoCo. Full task/AMP
expert-loader integration is covered separately by bounded GPU runs.
"""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

try:
    from mjlab_husky.supermimic.runner import SuperMimicRunner, lambda_returns
except (ModuleNotFoundError, ImportError) as exc:
    raise unittest.SkipTest(f"Runner's bundled RSL-RL dependencies unavailable: {exc}") from exc

from mjlab_husky.supermimic.adapter import StateLayout, roll_history, differentiable_amp_reward


class ToyEnvironment:
    def __init__(self):
        self.num_envs, self.num_actions, self.device = 4, 2, "cpu"
        self.clip_actions = None
        self.x = torch.zeros(4, 2)
        self.episode_length_buf = torch.zeros(4, dtype=torch.long)
        self.max_episode_length, self.steps = 10, 0
        self.unwrapped = self

    def get_observations(self):
        return {"policy": self.x.clone(),
                "critic": torch.cat((self.x, self.x.sum(-1, keepdim=True)), 1)}

    def get_amp_observations(self):
        return self.x.clone()

    def step(self, action):
        self.steps += 1
        self.x = 0.9 * self.x + 0.1 * action
        self.episode_length_buf += 1
        task_reward = 1 - self.x.square().sum(-1)
        done = torch.zeros(4, dtype=torch.bool)
        timeout = torch.zeros_like(done)
        if self.steps % 3 == 0:
            done[0] = timeout[0] = True
        if self.steps % 5 == 0:
            done[1] = True
        self.x[done] = 0
        self.episode_length_buf[done] = 0
        return self.get_observations(), task_reward, done, {"time_outs": timeout}


class ToyAdapter:
    def __init__(self):
        self.action_dim, self.step_dt = 2, 0.02
        self.layout = StateLayout({"actor_obs": 2, "critic_obs": 3,
                                   "amp_history": (2, 2), "amp_gate": ()})
        self.state_dim = self.layout.dim

    def capture(self, obs, frames):
        return self.layout.pack(actor_obs=obs["policy"], critic_obs=obs["critic"],
                                amp_history=frames, amp_gate=torch.ones(len(frames)))

    def actor_obs(self, state):
        return self.layout.get(state, "actor_obs")

    def critic_obs(self, state):
        return self.layout.get(state, "critic_obs")

    def project_next(self, state, predicted, action):
        fields = self.layout.unpack(predicted)
        fields["amp_history"] = roll_history(self.layout.get(state, "amp_history"), fields["actor_obs"])
        fields["amp_gate"] = torch.ones(len(state))
        return self.layout.pack(fields)

    def amp_reward(self, state, reward, disc, mean, std, **kwargs):
        return differentiable_amp_reward(self.layout.get(state, "amp_history"), reward,
                                         self.layout.get(state, "amp_gate"), disc, mean, std, **kwargs)

    def metadata(self):
        return self.layout.metadata()


class ToyDiscriminator(nn.Linear):
    def __init__(self):
        super().__init__(4, 1)
        self.amp_reward_coef, self.task_reward_lerp, self.use_lerp = 5.0, 0.7, False


class ToyAmp:
    def __init__(self):
        self.num_frames, self.observation_dim = 2, 2
        self.discriminator = ToyDiscriminator()
        self.normalizer = SimpleNamespace(mean=torch.zeros(2), var=torch.ones(2),
                                          epsilon=1e-4, clip_obs=10.0)
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(), 1e-3)

    def reward(self, reward, frames, gate, differentiable=True):
        return differentiable_amp_reward(frames, reward, gate, self.discriminator,
                                         self.normalizer.mean, (self.normalizer.var + 1e-4).sqrt())

    def update(self, frames, steps, batch_size):
        for _ in range(steps):
            self.optimizer.zero_grad(set_to_none=True)
            loss = (self.discriminator(frames.flatten(1)) + 1).square().mean()
            loss.backward()
            self.optimizer.step()
        return {"loss": float(loss.detach())}

    def state_dict(self):
        return {"discriminator": self.discriminator.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state):
        self.discriminator.load_state_dict(state["discriminator"])
        self.optimizer.load_state_dict(state["optimizer"])


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(4)

    def config(self, **settings):
        return {"policy": {"class_name": "ActorCritic", "actor_hidden_dims": (16,),
                           "critic_hidden_dims": (16,), "actor_obs_normalization": True,
                           "critic_obs_normalization": True},
                "supermimic": {"warmup_iterations": 0, "model_hidden_dims": (16,),
                               "model_updates": 1, "actor_updates": 1, "critic_updates": 1,
                               "amp_updates": 1, "model_batch_size": 8, "actor_batch_size": 4,
                               "amp_batch_size": 8, **settings}}

    def make_runner(self, config=None, log_dir=None):
        return SuperMimicRunner(ToyEnvironment(), config or self.config(), log_dir,
                                adapter=ToyAdapter(), amp=ToyAmp())

    def test_32_step_actor_updates_through_frozen_world_and_discriminator(self):
        runner = self.make_runner()
        metrics = runner.collect()
        self.assertEqual(metrics["validation/one_step_samples"], 18)
        runner.fit_world_model()
        before_actor = [p.detach().clone() for p in runner.policy.actor.parameters()]
        before_world = copy.deepcopy(runner.world_model.state_dict())
        before_disc = copy.deepcopy(runner.amp.discriminator.state_dict())
        count = int(runner.policy.actor_obs_normalizer.count)
        metrics = runner.update_actor_critic()
        self.assertTrue(any(not torch.equal(before, after) for before, after in
                            zip(before_actor, runner.policy.actor.parameters(), strict=True)))
        self.assertGreater(metrics["actor/gradient_norm"], 0)
        self.assertEqual(runner.settings.horizon, 32)
        self.assertEqual(int(runner.policy.actor_obs_normalizer.count), count)
        self.assertEqual(int(runner.world_model.input_norm.count), 20)
        for name, value in runner.world_model.state_dict().items():
            torch.testing.assert_close(value, before_world[name], rtol=0, atol=0)
        for name, value in runner.amp.discriminator.state_dict().items():
            torch.testing.assert_close(value, before_disc[name], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in runner.world_model.parameters()))
        self.assertTrue(all(p.grad is None for p in runner.amp.discriminator.parameters()))

    def test_lambda_returns_bootstrap_and_physical_terminal(self):
        rewards = torch.ones(2, 2, requires_grad=True)
        continuation = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
        values = torch.tensor([[0.0, 2.0, 4.0], [0.0, 2.0, 4.0]], requires_grad=True)
        targets = lambda_returns(rewards, continuation, values, 0.9, 1.0)
        torch.testing.assert_close(targets, torch.tensor([[5.14, 4.6], [1.9, 1.0]]))
        self.assertFalse(targets.requires_grad)
        # lambda=0 reduces to the one-step TD bootstrap, including timeout continuation.
        torch.testing.assert_close(lambda_returns(rewards, continuation, values, 0.9, 0.0),
                                   torch.tensor([[2.8, 4.6], [2.8, 1.0]]))

    def test_checkpoint_resume_preserves_deployable_policy_and_training_state(self):
        config = self.config(checkpoint_replay=True)
        with tempfile.TemporaryDirectory() as directory:
            runner = self.make_runner(config, directory)
            runner.learn(2)
            expected = runner.policy.act_inference(runner._obs).detach()
            path = str(Path(directory) / "checkpoint.pt")
            runner.save(path, {"fixture": True})
            resumed = self.make_runner(config)
            self.assertEqual(resumed.load(path), {"fixture": True})
            torch.testing.assert_close(resumed.policy.act_inference(runner._obs), expected)
            self.assertEqual(len(resumed.replay), len(runner.replay))
            self.assertEqual(resumed.current_learning_iteration, 2)
            self.assertEqual(resumed.actor_optimizer_updates, 2)
            self.assertEqual(resumed.world_model.optimizer_updates, 2)
            self.assertTrue(resumed.replay.data["truncated"][(resumed.replay.steps - 1) % resumed.replay.capacity].all())
            rows = [json.loads(line) for line in (Path(directory) / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[-1]["seconds"], rows[-1]["time/iteration"])
            resumed.learn(1)
            self.assertEqual(resumed.actor_optimizer_updates, 3)

    def test_resume_without_replay_recollects_warmup(self):
        config = self.config(warmup_iterations=1, checkpoint_replay=False)
        with tempfile.TemporaryDirectory() as directory:
            runner = self.make_runner(config)
            runner.learn(2)
            path = str(Path(directory) / "checkpoint.pt")
            runner.save(path)
            resumed = self.make_runner(config)
            resumed.load(path)
            self.assertEqual(len(resumed.replay), 0)
            self.assertEqual(resumed._warmup_collections, 0)
            updates = resumed.actor_optimizer_updates
            resumed.learn(1)
            self.assertTrue(resumed.last_metrics["warmup"])
            self.assertEqual(resumed.actor_optimizer_updates, updates)
            resumed.learn(1)
            self.assertEqual(resumed.actor_optimizer_updates, updates + 1)

    def test_ppo_resume_and_incompatible_schema_rejected_before_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.make_runner()
            original = copy.deepcopy(runner.policy.state_dict())
            path = str(Path(directory) / "ppo.pt")
            changed = {key: value + 1 for key, value in original.items()}
            torch.save({"model_state_dict": changed, "iter": 4}, path)
            with self.assertRaisesRegex(ValueError, "Cannot resume PPO"):
                runner.load(path)
            for name, value in runner.policy.state_dict().items():
                torch.testing.assert_close(value, original[name])
            runner.load(path, load_optimizer=False)
            for name, value in runner.policy.state_dict().items():
                if name != "std":  # Noise bounds intentionally constrain warm-start noise.
                    torch.testing.assert_close(value, changed[name])
            valid = str(Path(directory) / "valid.pt")
            runner.save(valid)
            incompatible = self.make_runner(self.config(horizon=16))
            initial = copy.deepcopy(incompatible.policy.state_dict())
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                incompatible.load(valid)
            for name, value in incompatible.policy.state_dict().items():
                torch.testing.assert_close(value, initial[name])


if __name__ == "__main__":
    unittest.main()
