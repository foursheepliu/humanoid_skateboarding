"""Bounded comparisons on the unchanged skating task, one backend per process.

Without --wall-seconds this retains the short iteration-throughput check. A wall
budget includes warmup, collection, every optimizer, logging and checkpoint IO;
it stops at a complete update boundary. Neither mode establishes convergence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from numbers import Real
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab_husky.scripts.train import TrainConfig
from mjlab_husky.supermimic.config import SuperMimicConfig


@dataclass
class BenchmarkConfig:
    backend: Literal["ppo", "supermimic"] = "ppo"
    task: str = "Mjlab-Skater-Flat-Unitree-G1"
    num_envs: int = 256
    ppo_steps_per_env: int | None = None
    """PPO-only rollout override. None retains the registered task default."""
    iterations: int = 12
    """Hard cap on complete updates in this invocation, including warmup."""
    wall_seconds: float | None = None
    snapshot_seconds: float = 120.0
    discard_iterations: int = 3
    amp_preload: int = 4096
    seed: int = 42
    output: str = "logs/backend_benchmark"
    initial_state_file: str | None = None
    """First case creates this shared policy/discriminator initialization."""
    resume_checkpoint: str | None = None
    supermimic: SuperMimicConfig = field(default_factory=SuperMimicConfig)

    def __post_init__(self):
        if self.ppo_steps_per_env is not None:
            _positive_integer(self.ppo_steps_per_env, "ppo_steps_per_env")
            if self.backend != "ppo":
                raise ValueError("ppo_steps_per_env applies only to the PPO backend")


def _positive_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _ppo_rollout_geometry(num_envs, steps):
    num_envs = _positive_integer(num_envs, "PPO num_envs")
    steps = _positive_integer(steps, "PPO steps_per_env")
    return {"version": 1, "num_envs": num_envs, "steps_per_env": steps,
            "transitions_per_update": num_envs * steps}


def _validate_ppo_rollout_counts(geometry, completed, total_steps):
    if (isinstance(completed, bool) or not isinstance(completed, int) or completed < 0
            or isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps < 0):
        raise ValueError("PPO rollout counts must be nonnegative integers")
    if total_steps != completed * geometry["transitions_per_update"]:
        raise ValueError("PPO transition count differs from completed updates * num_envs * steps_per_env")


class BudgetExceeded(Exception):
    """A wall budget expired after a complete learning update."""


_UNSPECIFIED = object()


def _validated_ppo_learning_rate(optimizer_state: dict, explicit=_UNSPECIFIED) -> float:
    """Legacy checkpoints derive the adaptive rate from consistent Adam groups."""
    groups = optimizer_state.get("param_groups") if isinstance(optimizer_state, dict) else None
    if not isinstance(groups, (list, tuple)) or not groups:
        raise ValueError("PPO checkpoint needs nonempty optimizer parameter groups")
    if any(not isinstance(group, dict) for group in groups):
        raise ValueError("PPO checkpoint optimizer parameter groups must be mappings")

    def checked(value):
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(value) or value <= 0):
            raise ValueError("PPO learning rate must be a finite positive number")
        return float(value)

    rates = [checked(group.get("lr")) for group in groups]
    if any(rate != rates[0] for rate in rates[1:]):
        raise ValueError("PPO checkpoint optimizer groups have inconsistent learning rates")
    if explicit is not _UNSPECIFIED and checked(explicit) != rates[0]:
        raise ValueError("PPO checkpoint explicit learning rate differs from optimizer groups")
    return rates[0]


def _validated_ppo_resume_state(payload: dict, expected_rollout=None, *, legacy_steps_per_env=24):
    """Validate resume bookkeeping before the native loader touches parameters."""
    amp = payload.get("benchmark_amp")
    if not isinstance(amp, dict):
        raise ValueError("PPO resume needs a benchmark checkpoint with AMP state")
    learning_rate = _validated_ppo_learning_rate(
        payload.get("optimizer_state_dict"), amp.get("learning_rate", _UNSPECIFIED))
    iteration = payload.get("iter")
    completed = amp.get("completed_iterations")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in (iteration, completed)) or iteration != completed:
        raise ValueError("PPO checkpoint iteration must equal its nonnegative completed update count")
    geometry = amp.get("rollout_geometry")
    if geometry is not None:
        if not isinstance(geometry, dict):
            raise ValueError("PPO rollout_geometry must be a mapping")
        checked = _ppo_rollout_geometry(geometry.get("num_envs"), geometry.get("steps_per_env"))
        if (type(geometry.get("version")) is not int or type(geometry.get("transitions_per_update")) is not int
                or geometry != checked):
            raise ValueError("PPO checkpoint rollout geometry is inconsistent")
        if expected_rollout is not None and checked != expected_rollout:
            raise ValueError("PPO resume rollout geometry differs; start a fresh profile instead")
        _validate_ppo_rollout_counts(checked, completed, amp.get("total_real_steps"))
    elif expected_rollout is not None:
        # Historical benchmark code had no rollout override. Equal batch size
        # does not make 1024x24 and 4096x6 compatible optimizer continuations.
        if expected_rollout["steps_per_env"] != legacy_steps_per_env:
            raise ValueError("Legacy PPO checkpoint has the task-default rollout; an explicit different rollout cannot resume it")
        _validate_ppo_rollout_counts(expected_rollout, completed, amp.get("total_real_steps"))
    return amp, learning_rate


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}\n".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _normalizer_state(normalizer) -> dict:
    return {"mean": torch.as_tensor(normalizer.mean).detach().cpu().clone(),
            "var": torch.as_tensor(normalizer.var).detach().cpu().clone(),
            "count": float(normalizer.count), "epsilon": float(normalizer.epsilon),
            "clip_obs": float(normalizer.clip_obs)}


def _restore_normalizer(normalizer, state: dict):
    normalizer.mean = state["mean"].detach().cpu().numpy().copy()
    normalizer.var = state["var"].detach().cpu().numpy().copy()
    normalizer.count = float(state["count"])
    normalizer.epsilon = float(state["epsilon"])
    normalizer.clip_obs = float(state["clip_obs"])


def _atomic_save(payload: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _shared_initial_state(runner, path: str | None, *, resume: bool = False,
                          expected: dict | None = None) -> dict:
    """Load only fresh initializations; a resume must retain trained weights."""
    policy, discriminator = runner.alg.policy, runner.discriminator
    if resume:
        if path is not None:
            target = Path(path).resolve()
            actual = _file_hash(target)
            if expected is not None and actual != expected.get("sha256"):
                raise ValueError("Resume and shared initialization SHA256 differ")
            if expected is None:
                return {"path": str(target), "sha256": actual,
                        "resume_initialization_identity_unverified": True}
        return expected or {"path": None, "sha256": None,
                            "resume_initialization_identity_unverified": True}
    target = Path(path).resolve() if path is not None else None
    if target is not None:
        if not target.exists():
            payload = {"version": 1,
                       "policy": {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()},
                       "discriminator": {k: v.detach().cpu().clone()
                                         for k, v in discriminator.state_dict().items()},
                       "normalizer": _normalizer_state(runner.amp_normalizer)}
            _atomic_save(payload, target)
        payload = torch.load(target, map_location="cpu", weights_only=False)
        if payload.get("version") != 1:
            raise ValueError("Unsupported shared initialization schema")
        policy.load_state_dict(payload["policy"])
        discriminator.load_state_dict(payload["discriminator"])
        _restore_normalizer(runner.amp_normalizer, payload["normalizer"])
    return {"path": str(target) if target is not None else None,
            "sha256": _file_hash(target) if target is not None else None,
            "policy_sha256": _state_hash(policy.state_dict()),
            "discriminator_sha256": _state_hash(discriminator.state_dict())}


class TrainingClock:
    """Track inclusive training time and save only at complete update boundaries."""

    def __init__(self, cfg: BenchmarkConfig, output: Path):
        self.cfg, self.output = cfg, output
        self.runner = None
        self.start: float | None = None
        self.rows: list[dict] = []
        self.snapshots: list[dict] = []
        self.shared_initial_state: dict = {}
        self.next_snapshot = cfg.snapshot_seconds
        self.initial_iterations = self.initial_real_steps = 0

    def elapsed(self) -> float:
        if self.start is None:
            return 0.0
        if torch.device(self.runner.device).type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - self.start

    def begin(self, runner):
        self.runner = runner
        self.initial_iterations = int(runner.current_learning_iteration)
        self.initial_real_steps = int(runner.tot_timesteps)
        if torch.device(runner.device).type == "cuda":
            torch.cuda.synchronize()
        self.start = time.perf_counter()

    def metadata(self) -> dict:
        return {"shared_initial_state": self.shared_initial_state,
                "total_iterations": int(self.runner.current_learning_iteration),
                "total_real_steps": int(self.runner.tot_timesteps),
                "training_wall_seconds": self.elapsed()}

    def save(self, path: Path) -> dict:
        previous = self.runner._benchmark_force_save
        self.runner._benchmark_force_save = True
        try:
            self.runner.save(str(path), infos={"benchmark": self.metadata()})
        finally:
            self.runner._benchmark_force_save = previous
        return {"checkpoint": str(path), "sha256": _file_hash(path), **self.metadata()}

    def record(self, row: dict):
        row = dict(row)
        row["wall_seconds"] = self.elapsed()
        row["total_real_steps"] = int(self.runner.tot_timesteps)
        self.rows.append(row)
        if self.cfg.backend == "ppo":
            with (self.output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row, allow_nan=False), flush=True)

    def record_supermimic(self):
        metrics = self.runner.last_metrics
        if not metrics:
            return
        iteration = int(metrics["iteration"])
        if iteration <= self.initial_iterations or (self.rows and iteration <= self.rows[-1]["iteration"]):
            return
        self.record({**metrics, "real_transitions": self.cfg.num_envs * self.runner.num_steps_per_env})

    def boundary(self):
        elapsed = self.elapsed()
        if self.cfg.wall_seconds is not None and elapsed >= self.next_snapshot:
            path = self.output / (f"snapshot_{len(self.snapshots) + 1:04d}_"
                                  f"iter_{self.runner.current_learning_iteration:06d}.pt")
            self.snapshots.append(self.save(path))
            elapsed = self.elapsed()
            while self.next_snapshot <= elapsed:
                self.next_snapshot += self.cfg.snapshot_seconds
        if self.cfg.wall_seconds is not None and elapsed >= self.cfg.wall_seconds:
            raise BudgetExceeded


def main():
    cfg = tyro.cli(BenchmarkConfig, config=(tyro.conf.AvoidSubcommands,))
    if cfg.iterations < 1 or cfg.discard_iterations < 0:
        raise ValueError("iterations must be positive and discard count nonnegative")
    if cfg.wall_seconds is None and cfg.iterations <= cfg.discard_iterations:
        raise ValueError("Short benchmarks need iterations greater than discard count")
    if cfg.wall_seconds is not None and (not math.isfinite(cfg.wall_seconds) or cfg.wall_seconds <= 0):
        raise ValueError("wall-seconds must be finite and positive")
    if not math.isfinite(cfg.snapshot_seconds) or cfg.snapshot_seconds <= 0:
        raise ValueError("snapshot-seconds must be finite and positive")
    if not torch.cuda.is_available():
        raise RuntimeError("The A5000 comparison requires a CUDA device")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("WANDB_MODE", "disabled")
    from mjlab.utils.torch import configure_torch_backends
    from mjlab_husky.envs import G1SkaterManagerBasedRlEnv
    from mjlab_husky.rl import RslRlVecEnvWrapper
    from mjlab_husky.tasks.registry import load_runner_cls

    configure_torch_backends()
    torch.manual_seed(cfg.seed)
    original = TrainConfig.from_task(cfg.task)
    legacy_ppo_steps = _positive_integer(original.agent.num_steps_per_env, "Registered PPO rollout steps")
    if cfg.ppo_steps_per_env is not None:
        original.agent.num_steps_per_env = cfg.ppo_steps_per_env
    original.env.scene.num_envs = cfg.num_envs
    original.env.seed = original.agent.seed = cfg.seed
    original.agent.amp_num_preload_transitions = cfg.amp_preload
    original.agent.logger = "tensorboard"
    original.agent.save_interval = cfg.iterations + 1
    output = Path(cfg.output).resolve() / cfg.backend
    output.mkdir(parents=True, exist_ok=True)
    if (output / "benchmark.json").exists() or (output / "metrics.jsonl").exists():
        raise FileExistsError(f"Use a new --output directory for each comparison: {output}")
    clock = TrainingClock(cfg, output)
    env = G1SkaterManagerBasedRlEnv(cfg=original.env, device="cuda:0")
    env = RslRlVecEnvWrapper(env, clip_actions=original.agent.clip_actions)
    agent = asdict(original.agent)

    try:
        if cfg.backend == "ppo":
            base = load_runner_cls(cfg.task)

            class TimedPPO(base):
                _benchmark_force_save = False

                def save(self, path, infos=None):
                    if not self._benchmark_force_save:
                        return
                    # Original AMP/PPO saves its combined policy+D optimizer but
                    # omits D weights and normalization; complete those here.
                    optimizer_state = self.alg.optimizer.state_dict()
                    learning_rate = _validated_ppo_learning_rate(optimizer_state, self.alg.learning_rate)
                    geometry = _ppo_rollout_geometry(self.env.num_envs, self.num_steps_per_env)
                    _validate_ppo_rollout_counts(geometry, self.current_learning_iteration, self.tot_timesteps)
                    payload = {"model_state_dict": self.alg.policy.state_dict(),
                               "optimizer_state_dict": optimizer_state,
                               "iter": self.current_learning_iteration, "infos": infos,
                               "benchmark_amp": {
                                   "discriminator": self.discriminator.state_dict(),
                                   "normalizer": _normalizer_state(self.amp_normalizer),
                                   "total_real_steps": self.tot_timesteps,
                                   "total_time": self.tot_time,
                                   "learning_rate": learning_rate,
                                   "rollout_geometry": geometry,
                                   "completed_iterations": self.current_learning_iteration}}
                    if hasattr(self.alg, "rnd") and self.alg.rnd:
                        payload["rnd_state_dict"] = self.alg.rnd.state_dict()
                        payload["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
                    _atomic_save(payload, Path(path))

                def load(self, path, load_optimizer=True, map_location=None):
                    payload = torch.load(path, map_location=map_location or self.device,
                                         weights_only=False)
                    amp, learning_rate = _validated_ppo_resume_state(
                        payload, _ppo_rollout_geometry(self.env.num_envs, self.num_steps_per_env),
                        legacy_steps_per_env=legacy_ppo_steps)
                    result = super().load(path, load_optimizer=load_optimizer, map_location=map_location)
                    if load_optimizer:
                        # Native RSL restores Adam's group rates but leaves this
                        # Python attribute at the constructor default. Adaptive
                        # PPO uses it to overwrite the group rates next update.
                        self.alg.learning_rate = learning_rate
                    self.discriminator.load_state_dict(amp["discriminator"])
                    _restore_normalizer(self.amp_normalizer, amp["normalizer"])
                    self.tot_timesteps = int(amp["total_real_steps"])
                    self.tot_time = float(amp["total_time"])
                    self.current_learning_iteration = int(amp["completed_iterations"])
                    return result

                def log(self, locs, *args, **kwargs):
                    torch.cuda.synchronize()
                    seconds = locs["collection_time"] + locs["learn_time"]
                    seconds += time.time() - locs["stop"]
                    # Checkpoint iter is completed updates, rather than the
                    # original zero-based current loop index.
                    self.current_learning_iteration = locs["it"] + 1
                    samples = cfg.num_envs * self.num_steps_per_env
                    self.tot_timesteps += samples
                    self.tot_time += seconds
                    rewards, lengths = locs["rewbuffer"], locs["lenbuffer"]
                    clock.record({"iteration": self.current_learning_iteration,
                                  "seconds": seconds,
                                  "collection_seconds": locs["collection_time"],
                                  "real_transitions": samples, "losses": locs["loss_dict"],
                                  "real/episode_return": statistics.mean(rewards) if rewards else None,
                                  "real/episode_length": statistics.mean(lengths) if lengths else None})
                    clock.boundary()

            runner = TimedPPO(env, agent, str(output), "cuda:0")
        else:
            from mjlab_husky.supermimic.runner import SuperMimicRunner

            class TimedSuperMimic(SuperMimicRunner):
                _benchmark_force_save = False

                def save(self, path, infos=None):
                    if self._benchmark_force_save:
                        return super().save(path, infos)

                def collect(self):
                    # The previous iteration has completed all optimizer work.
                    clock.record_supermimic()
                    clock.boundary()
                    return super().collect()

            agent["supermimic"] = asdict(cfg.supermimic)
            runner = TimedSuperMimic(env, agent, str(output), "cuda:0")

        resumed = None
        if cfg.resume_checkpoint is not None:
            resumed = torch.load(cfg.resume_checkpoint, map_location="cpu", weights_only=False)
            if ("supermimic" in resumed) != (cfg.backend == "supermimic"):
                raise ValueError("Resume checkpoint backend differs from --backend")
            runner.load(cfg.resume_checkpoint)
        previous_initial = (resumed or {}).get("infos") or {}
        previous_initial = previous_initial.get("benchmark", {}).get("shared_initial_state")
        clock.shared_initial_state = _shared_initial_state(
            runner, cfg.initial_state_file, resume=resumed is not None, expected=previous_initial)
        starting_policy_hash = _state_hash(runner.alg.policy.state_dict())
        starting_discriminator_hash = _state_hash(runner.discriminator.state_dict())
        clock.begin(runner)
        stop_reason = "iteration_cap"
        try:
            runner.learn(cfg.iterations, init_at_random_ep_len=True)
        except BudgetExceeded:
            stop_reason = "wall_budget"
        if cfg.backend == "supermimic":
            clock.record_supermimic()
        if cfg.wall_seconds is not None and clock.elapsed() >= cfg.wall_seconds:
            stop_reason = "wall_budget"
        final_checkpoint = output / f"model_{runner.current_learning_iteration}.pt"
        final_record = clock.save(final_checkpoint)
        end_to_end = clock.elapsed()  # Includes final checkpoint writing and hashing.
        kept = clock.rows[cfg.discard_iterations:]
        seconds = [row["seconds"] for row in kept]
        result = {"benchmark": asdict(cfg), "gpu": torch.cuda.get_device_name(),
                  "effective_steps_per_env": int(runner.num_steps_per_env),
                  "ppo_rollout_geometry": _ppo_rollout_geometry(cfg.num_envs, runner.num_steps_per_env)
                  if cfg.backend == "ppo" else None,
                  "ppo_update_geometry": {
                      "epochs": agent["algorithm"]["num_learning_epochs"],
                      "minibatches_per_epoch": agent["algorithm"]["num_mini_batches"],
                      "minibatch_size": cfg.num_envs * runner.num_steps_per_env // agent["algorithm"]["num_mini_batches"],
                      "gamma": agent["algorithm"]["gamma"], "lambda": agent["algorithm"]["lam"]}
                  if cfg.backend == "ppo" else None,
                  "scope": ("fixed wall budget on the training task; quality evaluated separately"
                            if cfg.wall_seconds is not None else
                            "short iteration throughput; quality/convergence unverified"),
                  "stop_reason": stop_reason,
                  "mean_seconds": statistics.mean(seconds) if seconds else None,
                  "median_seconds": statistics.median(seconds) if seconds else None,
                  "min_seconds": min(seconds) if seconds else None,
                  "max_seconds": max(seconds) if seconds else None,
                  "measured_iterations": len(kept), "end_to_end_seconds": end_to_end,
                  "training_clock_includes": ["warmup", "collection", "world_model_fit", "AMP",
                                               "policy", "critic", "logging", "checkpoint_IO"],
                  "iteration_timers_exclude_checkpoint_io": True,
                  "budget_overrun_seconds": max(0, end_to_end - cfg.wall_seconds)
                  if cfg.wall_seconds is not None else None,
                  "total_iterations": int(runner.current_learning_iteration),
                  "total_real_steps": int(runner.tot_timesteps),
                  "completed_iterations_this_run": int(runner.current_learning_iteration) - clock.initial_iterations,
                  "real_steps_this_run": int(runner.tot_timesteps) - clock.initial_real_steps,
                  "shared_initial_state": clock.shared_initial_state,
                  "starting_policy_sha256": starting_policy_hash,
                  "starting_discriminator_sha256": starting_discriminator_hash,
                  "resume_checkpoint": str(Path(cfg.resume_checkpoint).resolve())
                  if cfg.resume_checkpoint is not None else None,
                  "resume_semantics": ("weights, optimizers and normalization restored; "
                                       "simulator and unsaved replay restarted")
                  if resumed is not None else None,
                  "final_checkpoint": str(final_checkpoint), "final_checkpoint_sha256": final_record["sha256"],
                  "snapshots": clock.snapshots,
                  "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                  "rows": clock.rows}
        (output / "benchmark.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
