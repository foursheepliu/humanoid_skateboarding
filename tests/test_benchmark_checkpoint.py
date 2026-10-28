"""CPU regressions for adaptive PPO checkpoint continuation bookkeeping."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from mjlab_husky.scripts import benchmark_backends as benchmark


class _Normalizer:
    def __init__(self):
        self.mean, self.var = torch.zeros(3).numpy(), torch.ones(3).numpy()
        self.count, self.epsilon, self.clip_obs = 0.0, 1e-4, 10.0


class _Environment:
    def __init__(self, cfg, device):
        self.num_envs = cfg.scene.num_envs

    def close(self):
        pass


class _NativeLikePPO:
    """Mirror native load's Adam/iteration behavior while keeping tensors on CPU."""

    def __init__(self, env, agent, log_dir, device):
        self.device, self.env = "cpu", env
        self.num_steps_per_env = agent["num_steps_per_env"]
        self.current_learning_iteration = self.tot_timesteps = 0
        self.tot_time = 0.0
        policy, self.discriminator = nn.Linear(3, 2), nn.Linear(3, 1)
        self.amp_normalizer = _Normalizer()
        self.alg = SimpleNamespace(
            policy=policy, learning_rate=1e-3, rnd=None,
            optimizer=torch.optim.Adam([
                {"params": policy.parameters()},
                {"params": self.discriminator.parameters()},
            ], lr=1e-3))

    def load(self, path, load_optimizer=True, map_location=None):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.alg.policy.load_state_dict(payload["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(payload["optimizer_state_dict"])
        # Native ActorCritic.load_state_dict returns True, so the native runner
        # always restores this iteration even when load_optimizer is False.
        self.current_learning_iteration = payload["iter"]
        return payload["infos"]

    def learn(self, iterations, init_at_random_ep_len=False):
        if self.alg.learning_rate != 1e-5 or any(
                group["lr"] != 1e-5 for group in self.alg.optimizer.param_groups):
            raise AssertionError("First continued update lost its saved adaptive learning rate")
        import time
        for iteration in range(self.current_learning_iteration,
                               self.current_learning_iteration + iterations):
            self.current_learning_iteration = iteration
            self.log({"collection_time": 0.001, "learn_time": 0.001,
                      "stop": time.time(), "it": iteration,
                      "rewbuffer": [], "lenbuffer": [], "loss_dict": {}})


class BenchmarkCheckpointTests(unittest.TestCase):
    def test_rollout_override_is_typed_and_ppo_only(self):
        self.assertIsNone(benchmark.BenchmarkConfig().ppo_steps_per_env)
        self.assertEqual(benchmark.BenchmarkConfig(ppo_steps_per_env=6).ppo_steps_per_env, 6)
        for value in (True, False, 0, -1, 6.0, "6"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                benchmark.BenchmarkConfig(ppo_steps_per_env=value)
        with self.assertRaises(ValueError):
            benchmark.BenchmarkConfig(backend="supermimic", ppo_steps_per_env=6)

    def test_equal_batch_size_does_not_allow_geometry_resume(self):
        original = benchmark._ppo_rollout_geometry(1024, 24)
        changed = benchmark._ppo_rollout_geometry(4096, 6)
        self.assertEqual(original["transitions_per_update"], changed["transitions_per_update"])
        payload = {"iter": 7, "optimizer_state_dict": {"param_groups": [{"lr": 1e-5}]},
                   "benchmark_amp": {"completed_iterations": 7, "total_real_steps": 7 * 24576,
                                     "rollout_geometry": original}}
        benchmark._validated_ppo_resume_state(payload, original)
        with self.assertRaisesRegex(ValueError, "geometry differs"):
            benchmark._validated_ppo_resume_state(payload, changed)
        payload["benchmark_amp"].pop("rollout_geometry")
        benchmark._validated_ppo_resume_state(payload, original)
        with self.assertRaisesRegex(ValueError, "Legacy PPO checkpoint"):
            benchmark._validated_ppo_resume_state(payload, changed)

    def test_rollout_receipt_and_count_validation(self):
        geometry = benchmark._ppo_rollout_geometry(2, 6)
        payload = {"iter": 7, "optimizer_state_dict": {"param_groups": [{"lr": 1e-5}]},
                   "benchmark_amp": {"completed_iterations": 7, "total_real_steps": 84,
                                     "rollout_geometry": dict(geometry)}}
        benchmark._validated_ppo_resume_state(payload, geometry)
        for key, value in (("version", True), ("num_envs", True), ("steps_per_env", 0),
                           ("transitions_per_update", 13)):
            invalid = json.loads(json.dumps(payload))
            invalid["benchmark_amp"]["rollout_geometry"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                benchmark._validated_ppo_resume_state(invalid, geometry)
        for value in (True, 85, 84.0):
            invalid = json.loads(json.dumps(payload))
            invalid["benchmark_amp"]["total_real_steps"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                benchmark._validated_ppo_resume_state(invalid, geometry)

    def test_legacy_and_explicit_learning_rate(self):
        optimizer = {"param_groups": [{"lr": 1e-5}, {"lr": 1e-5}, {"lr": 1e-5}]}
        self.assertEqual(benchmark._validated_ppo_learning_rate(optimizer), 1e-5)
        self.assertEqual(benchmark._validated_ppo_learning_rate(optimizer, 1e-5), 1e-5)
        with self.assertRaises(ValueError):
            benchmark._validated_ppo_learning_rate(optimizer, 1e-3)
        with self.assertRaises(ValueError):
            benchmark._validated_ppo_learning_rate({"param_groups": [{"lr": 1e-5}, {"lr": 1e-3}]})

    def test_invalid_rates_and_groups(self):
        for rate in (0.0, -1.0, float("nan"), float("inf"), True, None, "1e-5"):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                benchmark._validated_ppo_learning_rate({"param_groups": [{"lr": rate}]})
            with self.subTest(explicit=rate), self.assertRaises(ValueError):
                benchmark._validated_ppo_learning_rate({"param_groups": [{"lr": 1e-5}]}, rate)
        for groups in (None, [], [None]):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                benchmark._validated_ppo_learning_rate({"param_groups": groups})

    def test_resume_rejects_inconsistent_iteration(self):
        for iteration, completed in ((5, 4), (-1, -1), (True, True), (1.5, 1.5)):
            with self.subTest(iteration=iteration), self.assertRaises(ValueError):
                benchmark._validated_ppo_resume_state({
                    "iter": iteration, "benchmark_amp": {"completed_iterations": completed},
                    "optimizer_state_dict": {"param_groups": [{"lr": 1e-5}]}})

    def test_legacy_resume_first_update_and_new_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _NativeLikePPO(SimpleNamespace(num_envs=2), {"num_steps_per_env": 24}, None, "cpu")
            for group in source.alg.optimizer.param_groups:
                group["lr"] = 1e-5
            checkpoint = root / "legacy.pt"
            torch.save({"model_state_dict": source.alg.policy.state_dict(),
                        "optimizer_state_dict": source.alg.optimizer.state_dict(),
                        "iter": 7, "infos": None,
                        "benchmark_amp": {"discriminator": source.discriminator.state_dict(),
                                          "normalizer": benchmark._normalizer_state(source.amp_normalizer),
                                          "completed_iterations": 7,
                                          "total_real_steps": 336, "total_time": 3.0}}, checkpoint)
            cfg = benchmark.BenchmarkConfig(num_envs=2, iterations=1, discard_iterations=0,
                                            output=str(root / "continued"), resume_checkpoint=str(checkpoint))
            with (patch.object(benchmark.tyro, "cli", return_value=cfg),
                  patch("mjlab_husky.envs.G1SkaterManagerBasedRlEnv", _Environment),
                  patch("mjlab_husky.rl.RslRlVecEnvWrapper", side_effect=lambda env, **_: env),
                  patch("mjlab_husky.tasks.registry.load_runner_cls", return_value=_NativeLikePPO),
                  patch("mjlab.utils.torch.configure_torch_backends"),
                  patch.object(torch.cuda, "is_available", return_value=True),
                  patch.object(torch.cuda, "synchronize"),
                  patch.object(torch.cuda, "get_device_name", return_value="MOCK_CPU_ONLY"),
                  patch.object(torch.cuda, "max_memory_allocated", return_value=0),
                  contextlib.redirect_stdout(io.StringIO())):
                benchmark.main()
            report = json.loads((root / "continued/ppo/benchmark.json").read_text())
            self.assertEqual(report["total_iterations"], 8)
            self.assertEqual(report["completed_iterations_this_run"], 1)
            self.assertEqual(report["total_real_steps"], 384)
            saved = torch.load(report["final_checkpoint"], map_location="cpu", weights_only=False)
            self.assertEqual(saved["iter"], 8)
            self.assertEqual(saved["benchmark_amp"]["learning_rate"], 1e-5)
            self.assertEqual(report["effective_steps_per_env"], 24)
            self.assertEqual(saved["benchmark_amp"]["rollout_geometry"], benchmark._ppo_rollout_geometry(2, 24))
            self.assertTrue(all(group["lr"] == 1e-5
                                for group in saved["optimizer_state_dict"]["param_groups"]))

    def test_native_six_step_resume_keeps_rate_and_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _NativeLikePPO(SimpleNamespace(num_envs=2), {"num_steps_per_env": 6}, None, "cpu")
            for group in source.alg.optimizer.param_groups:
                group["lr"] = 1e-5
            checkpoint = root / "six.pt"
            torch.save({"model_state_dict": source.alg.policy.state_dict(),
                        "optimizer_state_dict": source.alg.optimizer.state_dict(), "iter": 7, "infos": None,
                        "benchmark_amp": {"discriminator": source.discriminator.state_dict(),
                            "normalizer": benchmark._normalizer_state(source.amp_normalizer),
                            "completed_iterations": 7, "total_real_steps": 84, "total_time": 3.0,
                            "learning_rate": 1e-5, "rollout_geometry": benchmark._ppo_rollout_geometry(2, 6)}}, checkpoint)
            cfg = benchmark.BenchmarkConfig(num_envs=2, ppo_steps_per_env=6, iterations=1, discard_iterations=0,
                                            output=str(root / "continued"), resume_checkpoint=str(checkpoint))
            with (patch.object(benchmark.tyro, "cli", return_value=cfg),
                  patch("mjlab_husky.envs.G1SkaterManagerBasedRlEnv", _Environment),
                  patch("mjlab_husky.rl.RslRlVecEnvWrapper", side_effect=lambda env, **_: env),
                  patch("mjlab_husky.tasks.registry.load_runner_cls", return_value=_NativeLikePPO),
                  patch("mjlab.utils.torch.configure_torch_backends"),
                  patch.object(torch.cuda, "is_available", return_value=True),
                  patch.object(torch.cuda, "synchronize"),
                  patch.object(torch.cuda, "get_device_name", return_value="MOCK_CPU_ONLY"),
                  patch.object(torch.cuda, "max_memory_allocated", return_value=0),
                  contextlib.redirect_stdout(io.StringIO())):
                benchmark.main()
            report = json.loads((root / "continued/ppo/benchmark.json").read_text())
            self.assertEqual(report["effective_steps_per_env"], 6)
            self.assertEqual(report["total_real_steps"], 96)
            self.assertEqual(report["rows"][0]["real_transitions"], 12)
            saved = torch.load(report["final_checkpoint"], map_location="cpu", weights_only=False)
            self.assertEqual(saved["benchmark_amp"]["rollout_geometry"], benchmark._ppo_rollout_geometry(2, 6))
            self.assertEqual(saved["benchmark_amp"]["learning_rate"], 1e-5)


if __name__ == "__main__":
    unittest.main()
