"""CPU-only contracts for the bounded original PPO continuation harness."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from continue_author_ppo import (UpdateBudget, assert_finite, deadline_timestamp,
                                 filtered_policy_adam, learning_snapshot, warm_start)
import continue_author_ppo as harness


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Linear(2, 2)
        self.critic = nn.Linear(2, 1)
        self.std = nn.Parameter(torch.ones(2))
        self.register_buffer("actor_obs_mean", torch.tensor([3., 7.]))


def runner(seed, rates=(.001, .002, .003)):
    torch.manual_seed(seed)
    policy = Policy()
    discriminator = nn.Module()
    discriminator.trunk = nn.Linear(2, 3)
    discriminator.amp_linear = nn.Linear(3, 1)
    optimizer = torch.optim.Adam([
        {"params": policy.parameters(), "name": "policy", "lr": rates[0]},
        {"params": discriminator.trunk.parameters(), "name": "amp_trunk", "lr": rates[1], "weight_decay": .001},
        {"params": discriminator.amp_linear.parameters(), "name": "amp_head", "lr": rates[2], "weight_decay": .01},
    ])
    alg = SimpleNamespace(policy=policy, discriminator=discriminator, optimizer=optimizer,
        learning_rate=.1, amp_normalizer=SimpleNamespace(mean=np.zeros((5, 23)), var=np.ones((5, 23)),
            count=1e-4, epsilon=1e-4, clip_obs=10.),
        amp_storage=SimpleNamespace(states=torch.zeros(4, 5, 23), step=0, num_samples=0, buffer_size=4,
                                    num_amp_frames=5, device="cpu"),
        amp_data=SimpleNamespace(preloaded_frames=[torch.ones(4, 23)*i for i in range(5)],
            preloaded_s=torch.ones(4, 36), trajectory_names=["motion.npy"], device="cpu"), rnd=None)
    return SimpleNamespace(alg=alg, current_learning_iteration=0, cfg={"num_steps_per_env": 24},
                           amp_obs_frames=torch.zeros(2, 5, 23), tot_timesteps=0, tot_time=0.)


def adam_step(value):
    value.alg.optimizer.zero_grad(set_to_none=True)
    inputs = torch.tensor([[.2, -.8], [.6, .1]])
    p, d = value.alg.policy, value.alg.discriminator
    # Policy and discriminator objectives are independent, letting an original
    # policy-Adam continuation serve as an exact update reference after reset.
    loss = p.actor(inputs).square().sum()+p.critic(inputs).square().sum()+p.std.square().sum()
    loss = loss+d.amp_linear(d.trunk(inputs)).square().sum()
    loss.backward()
    value.alg.optimizer.step()


class AuthorContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def parent(self):
        old = runner(11, rates=(.000123, .000456, .000789))
        for _ in range(3):
            adam_step(old)
        return old, {"iter": 9999, "model_state_dict": copy.deepcopy(old.alg.policy.state_dict()),
                     "optimizer_state_dict": copy.deepcopy(old.alg.optimizer.state_dict())}

    def test_real_adam_policy_moments_survive_amp_moments_reset_and_next_update_matches(self):
        old, checkpoint = self.parent()
        new = runner(77)
        adam_step(new)  # Also prove preexisting optimizer AMP state is removed.
        fresh_disc = copy.deepcopy(new.alg.discriminator.state_dict())
        receipt = warm_start(new, checkpoint)
        self.assertEqual(new.current_learning_iteration, 10000)
        self.assertEqual(new.alg.learning_rate, .000123)
        self.assertEqual([group["lr"] for group in new.alg.optimizer.param_groups], [.000123, .000456, .000789])
        self.assertEqual(receipt["parent_checkpoint_iter"], 9999)
        for name, value in new.alg.policy.state_dict().items():
            torch.testing.assert_close(value, checkpoint["model_state_dict"][name], rtol=0, atol=0)
        for name, value in new.alg.discriminator.state_dict().items():
            torch.testing.assert_close(value, fresh_disc[name], rtol=0, atol=0)
        for left, right in zip(old.alg.policy.parameters(), new.alg.policy.parameters()):
            for field in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(old.alg.optimizer.state[left][field], new.alg.optimizer.state[right][field], rtol=0, atol=0)
        self.assertTrue(all(parameter not in new.alg.optimizer.state for group in new.alg.optimizer.param_groups[1:]
                            for parameter in group["params"]))
        adam_step(old)
        adam_step(new)
        for left, right in zip(old.alg.policy.parameters(), new.alg.policy.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        for group in new.alg.optimizer.param_groups[1:]:
            for parameter in group["params"]:
                self.assertEqual(float(new.alg.optimizer.state[parameter]["step"]), 1.)

    def test_reject_incompatible_optimizer_identity_shape_and_parent_cursor(self):
        _, checkpoint = self.parent()
        new = runner(3)
        bad = copy.deepcopy(checkpoint["optimizer_state_dict"])
        bad["param_groups"][1]["name"] = "unrelated"
        with self.assertRaises(ValueError):
            filtered_policy_adam(bad, new.alg.optimizer)
        bad = copy.deepcopy(checkpoint["optimizer_state_dict"])
        first = bad["param_groups"][0]["params"][0]
        bad["state"][first]["exp_avg"] = torch.zeros(100)
        with self.assertRaises(ValueError):
            filtered_policy_adam(bad, new.alg.optimizer)
        with self.assertRaises(ValueError):
            warm_start(new, dict(checkpoint, iter=10000))

    def test_exact_update_cap_ledger_and_milestones(self):
        budget = UpdateBudget(10000, 2000, deadline=1e20)
        milestones = []
        records = []
        while budget.stop_reason(0.) is None:
            records.append(budget.complete(budget.next_iteration))
            if budget.checkpoint_due():
                milestones.append(budget.completed)
        self.assertEqual(milestones, [50, 500, 1000, 1500, 2000])
        self.assertEqual((records[0]["iteration"], records[0]["additional_completed"], records[0]["cumulative_updates"]),
                         (10000, 1, 10001))
        self.assertEqual((records[-1]["iteration"], records[-1]["additional_completed"], records[-1]["cumulative_updates"]),
                         (11999, 2000, 12000))
        with self.assertRaises(RuntimeError):
            budget.complete(12000)
        with self.assertRaises(ValueError):
            UpdateBudget(10000, 2001, 1e20)

    def test_deadline_stops_at_first_completed_boundary_without_an_extra_update(self):
        budget = UpdateBudget(10000, 2000, deadline=10.)
        calls = []
        for now in (9., 10., 11.):
            calls.append(budget.complete(budget.next_iteration))
            if budget.stop_reason(now):
                break
        self.assertEqual(len(calls), 2)
        self.assertEqual((budget.completed, budget.next_iteration, budget.stop_reason(11.)), (2, 10002, "deadline"))
        not_started = UpdateBudget(10000, 2000, 10.)
        self.assertEqual(not_started.stop_reason(10.), "deadline")
        self.assertEqual(not_started.completed, 0)
        self.assertEqual(deadline_timestamp("2026-10-30T04:20:00+08:00"), deadline_timestamp("2026-10-29T20:20:00Z"))
        with self.assertRaises(ValueError):
            deadline_timestamp("2026-10-30T04:20:00")

    def test_snapshot_captures_missing_legacy_amp_learning_state_and_roundtrips(self):
        old, checkpoint = self.parent()
        warm_start(old, checkpoint)
        budget = UpdateBudget(10000, 2, deadline=1e10)
        budget.complete(10000)
        old.alg.amp_storage.states[0] = 9.
        old.alg.amp_storage.num_samples = 1
        old.alg.amp_storage.step = 1
        old.alg.amp_normalizer.count = 123.
        old.amp_obs_frames[:] = 4.
        snapshot = learning_snapshot(old, budget, {"parent_sha256": "test"})
        stream = io.BytesIO()
        torch.save(snapshot, stream)
        stream.seek(0)
        restored = torch.load(stream, weights_only=False)
        self.assertEqual((restored["iter"], restored["next_iteration"]), (10000, 10001))
        self.assertEqual(set(restored["amp_normalizer"]), {"mean", "var", "count", "epsilon", "clip_obs"})
        self.assertEqual(restored["amp_normalizer"]["count"], 123.)
        self.assertEqual(restored["amp_replay"]["num_samples"], 1)
        self.assertEqual(len(restored["amp_expert"]["preloaded_frames"]), 5)
        self.assertEqual(float(restored["amp_history"].mean()), 4.)
        self.assertIn("discriminator_state_dict", restored)
        self.assertEqual(set(restored["rng"]), {"python", "numpy", "torch_cpu", "torch_cuda"})
        old.alg.amp_storage.states.zero_()
        self.assertEqual(float(restored["amp_replay"]["states"][0].mean()), 9.)
        self.assertIn("Fresh-simulator", restored["resume_scope"])

    def test_recursive_finite_reports_nested_tensor_array_and_scalar(self):
        for value in (torch.tensor([float("inf")]), np.array([[float("nan")]]), float("inf")):
            with self.assertRaisesRegex(FloatingPointError, r"root.nested\[0\]"):
                assert_finite({"nested": [value]})
        assert_finite({"nested": [torch.ones(2), np.zeros((2, 2)), 1.]})

    def run_cpu_harness(self, directory, *, initialization_deadline=False, milestone_save_deadline=False,
                        post_log_deadline=False, mid_rollout_deadline=False):
        """Exercise actual init/log/save hooks with real CPU Adam, no simulator."""
        _, checkpoint = self.parent()
        parent = directory/"parent.pt"
        torch.save(checkpoint, parent)
        events = []
        clock = [0.]
        additional = 51 if milestone_save_deadline else 2
        instances, original_actions = [], []

        class FakeAMP:
            def __init__(self, env, config, *args):
                self.__dict__.update(vars(runner(71)))
                self.env, self.is_distributed, self.writer = env, False, None
                self.alg.storage = SimpleNamespace(step=0)

                def act(*args, **kwargs):
                    events.append(("act", self.alg.storage.step))
                    if mid_rollout_deadline and self.alg.storage.step == 0:
                        clock[0] = 2.

                self.alg.act = act
                instances.append(self)
                original_actions.append(act)

            def log(self, locs):
                events.append(("logged", locs["it"]))

            def save(self, path, infos=None):
                raise AssertionError("Original unbounded save must be hooked")

        class FakeSkater(FakeAMP):
            def save(self, path, infos=None):
                super().save(path, infos)
                self.asserted_logger = self.logger_type  # Actual Skater.save also reads this.

        config = SimpleNamespace(env=SimpleNamespace(scene=SimpleNamespace(num_envs=4096)),
            agent=SimpleNamespace(algorithm=SimpleNamespace(class_name="AMP_PPO"), num_steps_per_env=24,
                amp_num_obs=23, amp_num_frames=5, max_iterations=50000, resume=False, logger="wandb", run_name=""))

        def run_train(task, cfg, output):
            env = SimpleNamespace(num_envs=4096, num_actions=23, close=lambda: events.append(("closed",)))
            value = FakeSkater(env, {})
            self.assertEqual(value.current_learning_iteration, 10000)
            self.assertEqual(cfg.agent.max_iterations, additional)
            self.assertFalse(cfg.agent.resume)
            for it in range(value.current_learning_iteration, value.current_learning_iteration+cfg.agent.max_iterations):
                for _ in range(2):
                    value.alg.act(None, None)
                    value.alg.storage.step += 1
                adam_step(value)
                value.alg.storage.step = 0
                events.append(("updated", it))
                value.current_learning_iteration = it
                value.log({"it": it, "loss_dict": {"policy": 1.}, "collection_time": .1, "learn_time": .2})
                if it % 500 == 0:
                    value.save(str(output/f"model_{it}.pt"))
                if post_log_deadline and it == 10000:
                    # Original learn() can perform store_code_state/writer work here.
                    clock[0] = 2.

        modules = {}
        for name in ("mjlab", "mjlab.tasks", "mjlab_husky", "mjlab_husky.scripts", "mjlab_husky.scripts.train", "rsl_rl", "rsl_rl.runners"):
            modules[name] = ModuleType(name)
            modules[name].__path__ = []
        modules["mjlab_husky.scripts.train"].TrainConfig = SimpleNamespace(from_task=lambda task: config)
        modules["mjlab_husky.scripts.train"].run_train = run_train
        modules["rsl_rl.runners"].AMPOnPolicyRunner = FakeAMP
        controlled_clock = initialization_deadline or milestone_save_deadline or post_log_deadline or mid_rollout_deadline
        deadline = "1970-01-01T00:00:01Z" if controlled_clock else "2100-01-01T00:00:00Z"
        args = ["continue_author_ppo.py", "--checkpoint", str(parent), "--parent-updates", "10000",
                "--additional-updates", str(additional), "--deadline", deadline, "--out", str(directory/"output"), "--run-name", "cpu"]
        original = (FakeAMP.__init__, FakeAMP.log, FakeAMP.save)
        original_save = torch.save

        def delayed_save(payload, *args, **kwargs):
            result = original_save(payload, *args, **kwargs)
            if milestone_save_deadline and payload.get("format") == "husky-author-ppo-learning-state-v1":
                clock[0] = 2.
            return result

        with patch.dict(sys.modules, modules), patch.object(sys, "argv", args), \
                patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "-1", "WORLD_SIZE": "1"}), \
                patch.object(harness, "require_cuda"), \
                patch.object(harness, "source_identity", return_value={"manifest_sha256": "test"}), \
                patch("torch.save", side_effect=delayed_save), redirect_stdout(io.StringIO()):
            if initialization_deadline:
                with patch.object(harness.time, "time", side_effect=[0., 0.]+[2.]*30):
                    harness.main()
            elif controlled_clock:
                with patch.object(harness.time, "time", side_effect=lambda: clock[0]):
                    harness.main()
            else:
                harness.main()
        self.assertEqual((FakeAMP.__init__, FakeAMP.log, FakeAMP.save), original)
        self.assertIs(instances[0].alg.act, original_actions[0])
        self.assertIn(("closed",), events)
        data = directory/"output/data/cpu"
        return json.loads((data/"exit.json").read_text()), events, directory/"output"

    def test_actual_hooks_cap_save_sha_ledger_and_restore_with_cpu_adam(self):
        with tempfile.TemporaryDirectory() as temporary:
            final, events, output = self.run_cpu_harness(Path(temporary))
            self.assertEqual(final["stop_reason"], "additional_update_cap")
            self.assertEqual((final["additional_completed"], final["cumulative_updates"]), (2, 10002))
            self.assertEqual([x for x in events if x[0] == "updated"], [("updated", 10000), ("updated", 10001)])
            path = output/"checkpoints/cpu/model_10001.pt"
            self.assertTrue(path.exists())
            self.assertEqual(final["checkpoints"][0]["sha256"], harness.sha(path))
            ledger = [json.loads(line) for line in (output/"log/cpu/updates.jsonl").read_text().splitlines()]
            self.assertEqual([row["additional_completed"] for row in ledger], [1, 2])
            self.assertEqual(torch.load(path, weights_only=False)["next_iteration"], 10002)

    def test_deadline_during_initialization_saves_zero_updates_and_never_calls_learn(self):
        with tempfile.TemporaryDirectory() as temporary:
            final, events, output = self.run_cpu_harness(Path(temporary), initialization_deadline=True)
            self.assertEqual(final["stop_reason"], "deadline_during_initialization")
            self.assertEqual(final["additional_completed"], 0)
            self.assertFalse(any(event[0] == "updated" for event in events))
            saved = torch.load(output/"checkpoints/cpu/model_9999.pt", weights_only=False)
            self.assertEqual((saved["iter"], saved["next_iteration"]), (9999, 10000))

    def test_milestone_snapshot_crossing_deadline_cannot_start_update51(self):
        with tempfile.TemporaryDirectory() as temporary:
            final, events, output = self.run_cpu_harness(Path(temporary), milestone_save_deadline=True)
            self.assertEqual(final["stop_reason"], "deadline_during_checkpoint")
            self.assertEqual(final["additional_completed"], 50)
            self.assertEqual(len([event for event in events if event[0] == "act"]), 100)
            self.assertEqual(len([event for event in events if event[0] == "updated"]), 50)
            checkpoint = json.loads((output/"data/cpu/checkpoint_50.json").read_text())
            self.assertTrue(checkpoint["deadline_crossed_while_saving"])
            self.assertEqual(checkpoint["reason"], "deadline_during_checkpoint")
            self.assertEqual(final["checkpoints"][0]["reason"], "deadline_during_checkpoint")

    def test_post_log_code_state_delay_stops_at_next_rollout_entry_before_act(self):
        with tempfile.TemporaryDirectory() as temporary:
            final, events, output = self.run_cpu_harness(Path(temporary), post_log_deadline=True)
            self.assertEqual(final["stop_reason"], "deadline_before_rollout")
            self.assertEqual(final["additional_completed"], 1)
            self.assertEqual([event for event in events if event[0] == "act"], [("act", 0), ("act", 1)])
            self.assertEqual([event for event in events if event[0] == "updated"], [("updated", 10000)])
            saved = torch.load(output/"checkpoints/cpu/model_10000.pt", weights_only=False)
            self.assertEqual(saved["next_iteration"], 10001)

    def test_deadline_inside_rollout_finishes_that_update_without_partial_stop(self):
        with tempfile.TemporaryDirectory() as temporary:
            final, events, _ = self.run_cpu_harness(Path(temporary), mid_rollout_deadline=True)
            self.assertEqual(final["stop_reason"], "deadline")
            self.assertEqual(final["additional_completed"], 1)
            self.assertEqual([event for event in events if event[0] == "act"], [("act", 0), ("act", 1)])
            self.assertEqual([event for event in events if event[0] == "updated"], [("updated", 10000)])


if __name__ == "__main__":
    unittest.main()
