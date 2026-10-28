"""CPU-only checks for generic paired-trial SuperMimic CLI/config identity."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

import torch
import tyro

from mjlab_husky.scripts.benchmark_backends import BenchmarkConfig
from mjlab_husky.scripts.paired_trial import (
    PairedTrialConfig, _benchmark_command, _validate_benchmark,
)
from mjlab_husky.supermimic.config import SuperMimicConfig


class PairedTrialConfigTests(unittest.TestCase):
    def parse(self, cfg, backend="supermimic"):
        initialized = torch.cuda.is_initialized()
        command = _benchmark_command(cfg, Path("/tmp/output"), Path("/tmp/initial.pt"), backend)
        parsed = tyro.cli(BenchmarkConfig, args=command[3:],
                          config=(tyro.conf.AvoidSubcommands,))
        self.assertEqual(torch.cuda.is_initialized(), initialized)
        return command, parsed

    def test_default_commands_and_independent_factory(self):
        first, second = (PairedTrialConfig(output_dir="/tmp/output") for _ in range(2))
        self.assertIsNot(first.supermimic, second.supermimic)
        for backend in ("ppo", "supermimic"):
            command, parsed = self.parse(first, backend)
            self.assertFalse(any(flag.startswith("--supermimic.") for flag in command))
            self.assertEqual(parsed.supermimic, SuperMimicConfig())

    def test_scalars_tuples_and_literal_parse_as_requested(self):
        settings = replace(SuperMimicConfig(), model_updates=16,
                           model_hidden_dims=(2048, 2048, 2048),
                           actor_mode="reparameterized", real_exploration_std=0.15)
        cfg = PairedTrialConfig(output_dir="/tmp/output", supermimic=settings)
        command, parsed = self.parse(cfg)
        self.assertIn("--supermimic.model-updates", command)
        offset = command.index("--supermimic.model-hidden-dims")
        self.assertEqual(command[offset + 1:offset + 4], ["2048", "2048", "2048"])
        self.assertEqual(parsed.supermimic, settings)
        ppo_command, ppo = self.parse(cfg, "ppo")
        self.assertFalse(any(flag.startswith("--supermimic.") for flag in ppo_command))
        self.assertEqual(ppo.supermimic, SuperMimicConfig())

    def test_both_bool_directions_parse_without_scalar_values(self):
        # checkpoint_replay's False default exercises the same generic route as
        # future opt-in bool fields. No field-name allowlist is maintained.
        settings = replace(SuperMimicConfig(), checkpoint_replay=True,
                           compile_projection=False, detach_critic_transition_targets=False)
        command, parsed = self.parse(PairedTrialConfig(output_dir="/tmp/output", supermimic=settings))
        self.assertIn("--supermimic.checkpoint-replay", command)
        self.assertIn("--supermimic.no-compile-projection", command)
        self.assertIn("--supermimic.no-detach-critic-transition-targets", command)
        self.assertNotIn("True", command)
        self.assertNotIn("False", command)
        self.assertEqual(parsed.supermimic, settings)

    def fixture(self, root, settings):
        cfg = PairedTrialConfig(output_dir=str(root), supermimic=settings)
        checkpoint = root / "checkpoint.pt"
        checkpoint.write_bytes(b"CPU config validation fixture")
        return cfg, {
            "benchmark": {"backend": "supermimic", "task": cfg.task,
                          "num_envs": cfg.num_envs, "seed": cfg.seed,
                          "wall_seconds": cfg.seconds_per_backend,
                          "snapshot_seconds": cfg.snapshot_seconds,
                          "iterations": cfg.hard_cap, "amp_preload": cfg.amp_preload,
                          "supermimic": json.loads(json.dumps(asdict(settings)))},
            "stop_reason": "wall_budget", "end_to_end_seconds": 600.1,
            "completed_iterations_this_run": 1, "real_steps_this_run": 1280,
            "final_checkpoint": str(checkpoint), "final_checkpoint_sha256": "a" * 64,
            "shared_initial_state": {"sha256": "b" * 64},
        }

    def test_report_tuple_json_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, report = self.fixture(Path(directory), replace(SuperMimicConfig(), model_updates=16))
            self.assertIsInstance(report["benchmark"]["supermimic"]["model_hidden_dims"], list)
            _validate_benchmark(report, cfg, "supermimic")

    def test_missing_extra_or_different_full_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, report = self.fixture(Path(directory), replace(SuperMimicConfig(), model_updates=16))
            changes = [lambda config: config.update(model_updates=4),
                       lambda config: config.pop("model_updates"),
                       lambda config: config.update(unrequested_field=True),
                       lambda config: config.update(compile_projection=1)]
            for change in changes:
                invalid = json.loads(json.dumps(report))
                change(invalid["benchmark"]["supermimic"])
                with self.subTest(change=change), self.assertRaises(ValueError):
                    _validate_benchmark(invalid, cfg, "supermimic")
            # PPO reports intentionally ignore SM-only settings, including old
            # reports whose unused SM metadata predates new configuration fields.
            report["benchmark"]["backend"] = "ppo"
            report["benchmark"].pop("supermimic")
            _validate_benchmark(report, cfg, "ppo")


if __name__ == "__main__":
    unittest.main()
