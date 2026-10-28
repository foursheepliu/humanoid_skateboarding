"""Actual bundled ActorCritic with small deterministic physics/AMP fixtures.

These exercise learning/checkpoint contracts without MuJoCo. Full task/AMP
expert-loader integration is covered separately by bounded GPU runs.
"""

import copy
import json
import math
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
from mjlab_husky.supermimic.known_rewards import KnownActionRewards
from mjlab_husky.supermimic.models import frozen_parameters


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


class RichToyEnvironment(ToyEnvironment):
    def __init__(self, target_dims=(42, 56)):
        super().__init__()
        self.target_dims = target_dims

    def get_observations(self):
        target_width = sum(self.target_dims)
        targets = self.x.repeat(1, math.ceil(target_width / 2))[:, :target_width]
        return {"policy": self.x.clone(),
                "critic": torch.cat((self.x, self.x.sum(-1, keepdim=True), targets), -1)}


class RichToyAdapter(ToyAdapter):
    """Physical critic features plus independently predicted transition targets."""
    def __init__(self, target_dims=(42, 56)):
        super().__init__()
        self.critic_group = "critic"
        self.layout = StateLayout({"actor_obs": 2, "critic_obs": 3 + sum(target_dims),
                                   "amp_history": (2, 2), "amp_gate": ()})
        self.state_dim = self.layout.dim
        self._obs_terms = {"critic": (
            SimpleNamespace(name="physical", start=0, stop=3),
            SimpleNamespace(name="trans_target_pos_b", start=3, stop=3 + target_dims[0]),
            SimpleNamespace(name="trans_target_quat_b", start=3 + target_dims[0], stop=3 + sum(target_dims)))}

    def project_next(self, state, predicted, action):
        fields = self.layout.unpack(super().project_next(state, predicted, action))
        physical = fields["actor_obs"]
        fields["critic_obs"] = torch.cat((physical, physical.sum(-1, keepdim=True),
                                          fields["critic_obs"][..., 3:]), -1)
        return self.layout.pack(fields)


class ActionToyEnvironment(ToyEnvironment):
    """Real task reward contains the exact two known pre-step action costs."""
    def __init__(self):
        super().__init__()
        self.last_action, self.previous_action = torch.zeros(4, 2), torch.zeros(4, 2)
        self.recorded_rewards, self.recorded_costs = [], []

    def step(self, action):
        cost = -.1 * .02 * ((action - self.last_action).square().sum(-1)
                           + (action - 2 * self.last_action + self.previous_action).square().sum(-1))
        observations, physical_reward, done, extras = super().step(action)
        self.previous_action = self.last_action.clone()
        self.last_action = action.detach().clone()
        self.previous_action[done] = self.last_action[done] = 0
        reward = physical_reward + cost
        self.recorded_rewards.append(reward.clone())
        self.recorded_costs.append(cost.clone())
        return observations, reward, done, extras


class ActionToyAdapter(ToyAdapter):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.layout = StateLayout({"actor_obs": 2, "critic_obs": 3,
                                   "amp_history": (2, 2), "amp_gate": (),
                                   "action.action": 2, "action.prev_action": 2})
        self.state_dim = self.layout.dim

    def capture(self, obs, frames):
        return self.layout.pack(actor_obs=obs["policy"], critic_obs=obs["critic"],
                                amp_history=frames, amp_gate=torch.ones(len(frames)),
                                **{"action.action": self.env.last_action,
                                   "action.prev_action": self.env.previous_action})

    def project_next(self, state, predicted, action):
        fields = self.layout.unpack(super().project_next(state, predicted, action))
        fields["action.action"] = action
        fields["action.prev_action"] = self.layout.get(state, "action.action")
        return self.layout.pack(fields)


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

    def make_rich_runner(self, config=None, target_dims=(42, 56)):
        return SuperMimicRunner(RichToyEnvironment(target_dims), config or self.config(),
                                adapter=RichToyAdapter(target_dims), amp=ToyAmp())

    def make_action_runner(self, config=None, *, weight=-.1, step_dt=.02):
        config = config or self.config(analytic_action_cost=True)
        env = ActionToyEnvironment()
        adapter = ActionToyAdapter(env)
        known = (KnownActionRewards(adapter.layout, weight, weight, step_dt, True)
                 if config["supermimic"].get("analytic_action_cost", False) else None)
        return SuperMimicRunner(env, config, adapter=adapter, amp=ToyAmp(), known_reward=known)

    def assert_tree_identical(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_identical(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for first, second in zip(left, right, strict=True):
                self.assert_tree_identical(first, second)
        else:
            self.assertEqual(left, right)

    def test_bootstrap_cut_uses_metadata_preserves_values_and_physical_gradients(self):
        for dimensions in ((42, 56), (3, 4)):
            with self.subTest(target_dims=dimensions):
                runner = self.make_rich_runner(target_dims=dimensions)
                self.assertEqual(runner.critic_transition_target_dim, sum(dimensions))
                with torch.no_grad():
                    for parameter in runner.policy.critic.parameters():
                        parameter.fill_(0.01)
                physical = torch.randn(4, 2, requires_grad=True)
                auxiliary = torch.randn(4, sum(dimensions), requires_grad=True)
                state = runner.adapter.layout.pack(
                    actor_obs=physical,
                    critic_obs=torch.cat((physical, physical.sum(-1, keepdim=True), auxiliary), -1),
                    amp_history=torch.zeros(4, 2, 2), amp_gate=torch.ones(4))
                full_value = runner.policy.evaluate(runner._state_obs(state))
                cut_value = runner.policy.evaluate(runner._actor_bootstrap_obs(state))
                self.assertTrue(torch.equal(full_value, cut_value))
                full_physical, full_auxiliary = torch.autograd.grad(
                    full_value.sum(), (physical, auxiliary), retain_graph=True)
                cut_physical, cut_auxiliary = torch.autograd.grad(
                    cut_value.sum(), (physical, auxiliary), retain_graph=True)
                self.assertGreater(float(full_physical.norm()), 0)
                self.assertGreater(float(full_auxiliary.norm()), 0)
                torch.testing.assert_close(cut_physical, full_physical, rtol=0, atol=0)
                self.assertEqual(int(torch.count_nonzero(cut_auxiliary)), 0)
                # Ordinary evaluate, used by critic fitting/targets, still has
                # full transition-target input gradients after the actor call.
                normal_value = runner.policy.evaluate(runner._state_obs(state))
                normal_auxiliary = torch.autograd.grad(normal_value.sum(), auxiliary)[0]
                torch.testing.assert_close(normal_auxiliary, full_auxiliary, rtol=0, atol=0)
        toy = self.make_runner()
        self.assertEqual(toy.critic_transition_target_dim, 0)
        sample = toy._state.clone().requires_grad_(True)
        self.assertTrue(torch.equal(toy.policy.evaluate(toy._state_obs(sample)),
                                    toy.policy.evaluate(toy._actor_bootstrap_obs(sample))))

    def test_h32_cut_changes_only_bootstrap_gradient_not_rollout_or_model_loss(self):
        config = self.config(checkpoint_replay=True, detach_critic_transition_targets=False)
        baseline = self.make_rich_runner(config)
        baseline.collect()
        baseline.fit_world_model()
        with torch.no_grad():
            baseline.world_model.termination_head.bias.fill_(-2)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "baseline.pt")
            baseline.save(path)
            cut = self.make_rich_runner(self.config(checkpoint_replay=True))
            cut.load(path, load_optimizer=False)
            with frozen_parameters(baseline.world_model, baseline.amp.discriminator, baseline.policy.critic):
                base_loss, base_trace, base_metrics = baseline._imagine()
            with frozen_parameters(cut.world_model, cut.amp.discriminator, cut.policy.critic):
                cut_loss, cut_trace, cut_metrics = cut._imagine()
            self.assertTrue(torch.equal(base_loss, cut_loss))
            for name in base_trace:
                self.assertTrue(torch.equal(base_trace[name], cut_trace[name]), name)
            self.assertEqual(base_metrics["actor/detached_critic_transition_target_dim"], 0)
            self.assertEqual(cut_metrics["actor/detached_critic_transition_target_dim"], 98)
            batch = baseline.replay.sample_sequences(8, 8)
            base_model_loss = baseline.world_model.sequence_loss(batch)
            cut_model_loss = cut.world_model.sequence_loss(batch)
            for name in base_model_loss:
                self.assertTrue(torch.equal(base_model_loss[name], cut_model_loss[name]))
            baseline.actor_optimizer.zero_grad(set_to_none=True)
            cut.actor_optimizer.zero_grad(set_to_none=True)
            base_loss.backward()
            cut_loss.backward()
            self.assertTrue(any(not torch.equal(first.grad, second.grad) for first, second in
                                zip(baseline.policy.actor.parameters(), cut.policy.actor.parameters(), strict=True)))

    def test_actor_modes_real_noise_imagined_actions_rng_and_std_gradients(self):
        for mode in ("deterministic", "reparameterized"):
            for noise_type in ("scalar", "log"):
                with self.subTest(actor_mode=mode, noise_std_type=noise_type):
                    config = self.config(actor_mode=mode, real_exploration_std=0.1)
                    config["policy"].update(noise_std_type=noise_type, init_noise_std=0.6)
                    runner = self.make_runner(config)
                    noise_parameter = getattr(runner.policy, "std" if noise_type == "scalar" else "log_std")
                    self.assertEqual(noise_parameter.requires_grad, mode == "reparameterized")
                    self.assertEqual(any(parameter is noise_parameter for parameter in runner._actor_parameters),
                                     mode == "reparameterized")
                    self.assertIn("std" if noise_type == "scalar" else "log_std", runner.policy.state_dict())
                    mean = runner.policy.act_inference(runner._obs)
                    generator = torch.Generator().set_state(runner._rng.get_state())
                    expected_noise = torch.randn(mean.shape, generator=generator)
                    real_action, _ = runner._action(runner._obs)
                    scale = 0.1 if mode == "deterministic" else 0.6
                    torch.testing.assert_close(real_action, mean + scale * expected_noise)
                    self.assertTrue(torch.equal(generator.get_state(), runner._rng.get_state()))
                    state_before = runner._rng.get_state().clone()
                    imagined_action, entropy = runner._action(runner._obs, imagination=True)
                    if mode == "deterministic":
                        torch.testing.assert_close(imagined_action, mean)
                        self.assertTrue(torch.equal(state_before, runner._rng.get_state()))
                        self.assertTrue((entropy == 0).all())
                    else:
                        expected_noise = torch.randn(mean.shape, generator=generator)
                        torch.testing.assert_close(imagined_action, mean + 0.6 * expected_noise)
                        self.assertTrue(torch.equal(generator.get_state(), runner._rng.get_state()))
                        expected_entropy = 2 * (math.log(0.6) + 0.5 * math.log(2 * math.pi * math.e))
                        torch.testing.assert_close(entropy, torch.full_like(entropy, expected_entropy))
                    imagined_action.square().mean().backward()
                    self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                                        for parameter in runner.policy.actor.parameters()))
                    if mode == "deterministic":
                        self.assertIsNone(noise_parameter.grad)
                    else:
                        self.assertIsNotNone(noise_parameter.grad)
                        self.assertGreater(float(noise_parameter.grad.abs().sum()), 0)

    def test_h32_modes_keep_normalization_fixed_and_deterministic_mode_ignores_entropy(self):
        for mode in ("deterministic", "reparameterized"):
            with self.subTest(actor_mode=mode):
                runner = self.make_runner(self.config(actor_mode=mode, entropy_coef=10.0))
                runner.collect()
                runner.fit_world_model()
                norm_states = {
                    name: copy.deepcopy(module.state_dict()) for name, module in (
                        ("actor", runner.policy.actor_obs_normalizer),
                        ("critic", runner.policy.critic_obs_normalizer),
                        ("input", runner.world_model.input_norm),
                        ("delta", runner.world_model.delta_norm),
                        ("reward", runner.world_model.reward_norm))}
                generator_state = runner._rng.get_state().clone()
                runner.world_model.optimizer.zero_grad(set_to_none=True)
                with frozen_parameters(runner.world_model, runner.amp.discriminator, runner.policy.critic):
                    loss, trace, metrics = runner._imagine()
                    self.assertEqual(trace["state"].shape[1], 33)
                    if mode == "deterministic":
                        self.assertAlmostEqual(float(loss.detach()), -metrics["actor/objective"], places=6)
                        self.assertEqual(metrics["actor/effective_entropy_coef"], 0)
                        self.assertTrue(torch.equal(generator_state, runner._rng.get_state()))
                    else:
                        self.assertLess(float(loss.detach()), -metrics["actor/objective"])
                        self.assertEqual(metrics["actor/effective_entropy_coef"], 10)
                        self.assertFalse(torch.equal(generator_state, runner._rng.get_state()))
                    loss.backward()
                self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                                    for parameter in runner.policy.actor.parameters()))
                self.assertTrue(all(parameter.grad is None for parameter in runner.amp.discriminator.parameters()))
                self.assertTrue(all(parameter.grad is None for parameter in runner.world_model.parameters()))
                for name, module in (("actor", runner.policy.actor_obs_normalizer),
                                     ("critic", runner.policy.critic_obs_normalizer),
                                     ("input", runner.world_model.input_norm),
                                     ("delta", runner.world_model.delta_norm),
                                     ("reward", runner.world_model.reward_norm)):
                    for field, value in module.state_dict().items():
                        torch.testing.assert_close(value, norm_states[name][field], rtol=0, atol=0)

    def test_32_step_actor_updates_through_frozen_world_and_discriminator(self):
        runner = self.make_runner()
        metrics = runner.collect()
        self.assertEqual(metrics["validation/one_step_samples"], 18)
        runner.fit_world_model()
        before_actor = [p.detach().clone() for p in runner.policy.actor.parameters()]
        before_std = runner.policy.std.detach().clone()
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
        torch.testing.assert_close(runner.policy.std, before_std, rtol=0, atol=0)
        self.assertIsNone(runner.policy.std.grad)

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
            self.assertFalse(resumed.policy.std.requires_grad)
            self.assertNotIn(resumed.policy.std, resumed.actor_optimizer.state)
            expected_action, _ = runner._action(runner._obs)
            resumed_action, _ = resumed._action(runner._obs)
            torch.testing.assert_close(resumed_action, expected_action, rtol=0, atol=0)
            self.assertTrue(torch.equal(resumed._rng.get_state(), runner._rng.get_state()))
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

    def test_reparameterized_native_and_legacy_checkpoint_modes_and_rng(self):
        config = self.config(actor_mode="reparameterized", checkpoint_replay=True,
                             detach_critic_transition_targets=False)
        with tempfile.TemporaryDirectory() as directory:
            runner = self.make_runner(config)
            runner.learn(2)
            path = str(Path(directory) / "native.pt")
            runner.save(path)
            native = self.make_runner(config)
            native.load(path)
            self.assertEqual(native.actor_optimizer_updates, 2)
            self.assertEqual(len(native.actor_optimizer.state), len(runner.actor_optimizer.state))
            expected_action, _ = runner._action(runner._obs)
            actual_action, _ = native._action(runner._obs)
            torch.testing.assert_close(actual_action, expected_action, rtol=0, atol=0)
            self.assertTrue(torch.equal(native._rng.get_state(), runner._rng.get_state()))
            self.assertTrue(native.policy.std.requires_grad)
            for key, parameter in native.policy.state_dict().items():
                torch.testing.assert_close(parameter, runner.policy.state_dict()[key], rtol=0, atol=0)
            for field in ("exp_avg", "exp_avg_sq", "step"):
                native_state = native.actor_optimizer.state[native.policy.std][field]
                saved_state = runner.actor_optimizer.state[runner.policy.std][field]
                torch.testing.assert_close(native_state, saved_state, rtol=0, atol=0)
            checkpoint = torch.load(path, weights_only=False)
            checkpoint["supermimic"]["config"].pop("actor_mode")
            checkpoint["supermimic"]["config"].pop("real_exploration_std")
            checkpoint["supermimic"]["config"].pop("detach_critic_transition_targets")
            legacy_path = str(Path(directory) / "legacy.pt")
            torch.save(checkpoint, legacy_path)
            deterministic = self.make_runner(self.config(checkpoint_replay=True))
            initial_weights = copy.deepcopy(deterministic.policy.state_dict())
            with self.assertRaisesRegex(ValueError, "legacy reparameterized.*actor_mode=deterministic"):
                deterministic.load(legacy_path)
            for key, parameter in deterministic.policy.state_dict().items():
                torch.testing.assert_close(parameter, initial_weights[key], rtol=0, atol=0)
            deterministic.load(legacy_path, load_optimizer=False)
            self.assertFalse(deterministic.policy.std.requires_grad)
            self.assertEqual(len(deterministic.actor_optimizer.state), 0)
            for key, parameter in deterministic.policy.state_dict().items():
                torch.testing.assert_close(parameter, checkpoint["model_state_dict"][key], rtol=0, atol=0)
            legacy = self.make_runner(config)
            legacy.load(legacy_path)
            self.assertEqual(legacy.actor_optimizer_updates, 2)
            self.assertTrue(torch.equal(legacy._rng.get_state(), checkpoint["supermimic"]["rng"]))
            self.assertEqual(len(legacy.actor_optimizer.state), len(native.actor_optimizer.state))

    def test_missing_cut_flag_is_legacy_false_and_full_resume_rejects_changed_objective(self):
        config = self.config(checkpoint_replay=True, detach_critic_transition_targets=False)
        with tempfile.TemporaryDirectory() as directory:
            runner = self.make_rich_runner(config)
            runner.learn(1)
            native_path = str(Path(directory) / "native-false.pt")
            legacy_path = str(Path(directory) / "legacy-missing-flag.pt")
            runner.save(native_path)
            checkpoint = torch.load(native_path, weights_only=False)
            checkpoint["supermimic"]["config"].pop("detach_critic_transition_targets")
            torch.save(checkpoint, legacy_path)
            native = self.make_rich_runner(config)
            legacy = self.make_rich_runner(config)
            native.load(native_path)
            legacy.load(legacy_path)
            native.learn(1)
            legacy.learn(1)
            for name in ("policy", "world_model"):
                self.assert_tree_identical(getattr(native, name).state_dict(), getattr(legacy, name).state_dict())
            for name in ("actor_optimizer", "critic_optimizer"):
                self.assert_tree_identical(getattr(native, name).state_dict(), getattr(legacy, name).state_dict())
            self.assert_tree_identical(native.amp.state_dict(), legacy.amp.state_dict())
            self.assertTrue(torch.equal(native._rng.get_state(), legacy._rng.get_state()))
            changed = self.make_rich_runner(self.config(checkpoint_replay=True))
            before = copy.deepcopy(changed.policy.state_dict())
            with self.assertRaisesRegex(ValueError, "legacy.*detach_critic_transition_targets.*False"):
                changed.load(legacy_path)
            self.assert_tree_identical(before, changed.policy.state_dict())
            self.assertEqual(len(changed.actor_optimizer.state), 0)
            changed.load(legacy_path, load_optimizer=False)
            self.assert_tree_identical(changed.policy.state_dict(), checkpoint["model_state_dict"])
            changed.learn(1)
            cut_native_path = str(Path(directory) / "native-cut.pt")
            changed.save(cut_native_path)
            native_cut = self.make_rich_runner(self.config(checkpoint_replay=True))
            native_cut.load(cut_native_path)
            self.assertTrue(native_cut.settings.detach_critic_transition_targets)
            self.assertEqual(native_cut.critic_transition_target_dim, 98)
            self.assert_tree_identical(native_cut.policy.state_dict(), changed.policy.state_dict())
            self.assert_tree_identical(native_cut.world_model.training_state_dict(),
                                        changed.world_model.training_state_dict())
            self.assert_tree_identical(native_cut.actor_optimizer.state_dict(), changed.actor_optimizer.state_dict())
            self.assert_tree_identical(native_cut.critic_optimizer.state_dict(), changed.critic_optimizer.state_dict())
            self.assertTrue(torch.equal(native_cut._rng.get_state(), changed._rng.get_state()))

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

    def test_analytic_action_cost_keeps_real_total_labels_and_h32_frozen_statistics(self):
        runner = self.make_action_runner()
        metrics = runner.collect()
        data = runner.replay.state_dict()["data"]
        raw_reward = torch.stack(runner.env.recorded_rewards)
        known = runner.known_action_reward(data["state"], data["action"])
        torch.testing.assert_close(data["reward"], raw_reward, rtol=0, atol=0)
        torch.testing.assert_close(known, torch.stack(runner.env.recorded_costs))
        self.assertAlmostEqual(metrics["real/task_reward"], float(raw_reward.mean()))
        self.assertTrue(data["terminated"].any())
        self.assertTrue(data["truncated"].any())
        self.assertEqual(int(runner.world_model.reward_norm.count), raw_reward.numel())
        expected_residual = raw_reward - known
        torch.testing.assert_close(runner.world_model.reward_norm.mean.float(),
                                   expected_residual.mean().reshape(1))
        torch.testing.assert_close(runner.world_model.reward_norm.variance.float(),
                                   expected_residual.var(unbiased=False).reshape(1), atol=1e-8, rtol=1e-4)
        runner.fit_world_model()
        world = copy.deepcopy(runner.world_model.state_dict())
        disc = copy.deepcopy(runner.amp.discriminator.state_dict())
        policy_norms = copy.deepcopy((runner.policy.actor_obs_normalizer.state_dict(),
                                      runner.policy.critic_obs_normalizer.state_dict()))
        actor = [parameter.detach().clone() for parameter in runner.policy.actor.parameters()]
        result = runner.update_actor_critic()
        self.assertGreater(result["actor/gradient_norm"], 0)
        self.assertTrue(all(math.isfinite(value) for value in result.values()))
        self.assertTrue(any(not torch.equal(before, after) for before, after in
                            zip(actor, runner.policy.actor.parameters(), strict=True)))
        self.assert_tree_identical(world, runner.world_model.state_dict())
        self.assert_tree_identical(disc, runner.amp.discriminator.state_dict())
        self.assert_tree_identical(policy_norms, (runner.policy.actor_obs_normalizer.state_dict(),
                                                  runner.policy.critic_obs_normalizer.state_dict()))
        self.assertTrue(all(parameter.grad is None for parameter in runner.world_model.parameters()))

    def test_analytic_native_checkpoint_and_metadata_validate_before_any_mutation(self):
        config = self.config(analytic_action_cost=True, checkpoint_replay=True)
        with tempfile.TemporaryDirectory() as directory:
            runner = self.make_action_runner(config)
            runner.learn(2)
            path = str(Path(directory) / "residual.pt")
            runner.save(path)
            checkpoint = torch.load(path, weights_only=False)
            self.assertEqual(checkpoint["supermimic"]["known_action_reward"], runner.known_action_reward.metadata())
            native = self.make_action_runner(config)
            native.load(path)
            self.assert_tree_identical(native.policy.state_dict(), runner.policy.state_dict())
            self.assert_tree_identical(native.world_model.training_state_dict(), runner.world_model.training_state_dict())
            self.assert_tree_identical(native.actor_optimizer.state_dict(), runner.actor_optimizer.state_dict())
            self.assert_tree_identical(native.critic_optimizer.state_dict(), runner.critic_optimizer.state_dict())
            self.assert_tree_identical(native.amp.state_dict(), runner.amp.state_dict())
            self.assertTrue(torch.equal(native._rng.get_state(), runner._rng.get_state()))
            self.assertEqual(len(native.replay), len(runner.replay))
            weights_only = self.make_action_runner(config)
            weights_only.load(path, load_optimizer=False)
            self.assert_tree_identical(weights_only.world_model.state_dict(), runner.world_model.state_dict())
            self.assertEqual(len(weights_only.world_model.optimizer.state), 0)
            for changes in ({"weight": -.2}, {"step_dt": .04}):
                for load_optimizer in (True, False):
                    altered = self.make_action_runner(config, **changes)
                    before = copy.deepcopy((altered.policy.state_dict(), altered.world_model.training_state_dict(),
                                             altered.amp.state_dict(), altered.actor_optimizer.state_dict(),
                                             altered._rng.get_state(), altered.replay.state_dict()))
                    with self.assertRaisesRegex(ValueError, "Known-action reward metadata differs"):
                        altered.load(path, load_optimizer=load_optimizer)
                    self.assert_tree_identical(before, (altered.policy.state_dict(), altered.world_model.training_state_dict(),
                                                        altered.amp.state_dict(), altered.actor_optimizer.state_dict(),
                                                        altered._rng.get_state(), altered.replay.state_dict()))

    def test_analytic_cross_representation_rejects_even_weights_only_and_legacy_is_total(self):
        with tempfile.TemporaryDirectory() as directory:
            plain = self.make_action_runner(self.config(checkpoint_replay=True))
            plain.learn(1)
            total_path = str(Path(directory) / "total.pt")
            plain.save(total_path)
            legacy = torch.load(total_path, weights_only=False)
            legacy["supermimic"]["config"].pop("analytic_action_cost")
            legacy_path = str(Path(directory) / "legacy-total.pt")
            torch.save(legacy, legacy_path)
            plain_legacy = self.make_action_runner(self.config(checkpoint_replay=True))
            plain_legacy.load(legacy_path)
            self.assert_tree_identical(plain_legacy.world_model.training_state_dict(), plain.world_model.training_state_dict())
            analytic = self.make_action_runner(self.config(analytic_action_cost=True, checkpoint_replay=True))
            analytic.learn(1)
            residual_path = str(Path(directory) / "residual.pt")
            analytic.save(residual_path)
            for active, source in ((analytic, legacy_path), (plain, residual_path)):
                for load_optimizer in (True, False):
                    before = copy.deepcopy((active.policy.state_dict(), active.world_model.training_state_dict(),
                                             active.amp.state_dict(), active.actor_optimizer.state_dict(),
                                             active._rng.get_state(), active.replay.state_dict()))
                    with self.assertRaisesRegex(ValueError, "across analytic_action_cost reward representations"):
                        active.load(source, load_optimizer=load_optimizer)
                    self.assert_tree_identical(before, (active.policy.state_dict(), active.world_model.training_state_dict(),
                                                        active.amp.state_dict(), active.actor_optimizer.state_dict(),
                                                        active._rng.get_state(), active.replay.state_dict()))
            # Explicit bare policy initialization imports no incompatible WM,
            # normalizer, discriminator, optimizer or replay representation.
            policy_path = str(Path(directory) / "policy-only.pt")
            torch.save({"model_state_dict": plain.policy.state_dict()}, policy_path)
            before = copy.deepcopy((analytic.world_model.training_state_dict(), analytic.amp.state_dict(),
                                     analytic.replay.state_dict(), analytic._rng.get_state()))
            analytic.load(policy_path, load_optimizer=False)
            self.assert_tree_identical(analytic.policy.state_dict(), plain.policy.state_dict())
            self.assert_tree_identical(before, (analytic.world_model.training_state_dict(), analytic.amp.state_dict(),
                                                analytic.replay.state_dict(), analytic._rng.get_state()))


if __name__ == "__main__":
    unittest.main()
