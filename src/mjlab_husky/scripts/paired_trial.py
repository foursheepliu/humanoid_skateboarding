"""One supervised wall-budget screening round, with serial GPU subprocesses.

Example (run from the repository in the prepared training environment)::

  nohup python -m mjlab_husky.scripts.paired_trial --output-dir logs/paired_001 \
    > paired_001_supervisor.log 2>&1 &

This executes one configuration once. A subsequent round needs a new output
directory and explicit resume checkpoints; there is no restart or tuning loop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import traceback

from mjlab_husky.supermimic.config import SuperMimicConfig


@dataclass(frozen=True)
class PairedTrialConfig:
    output_dir: str
    seconds_per_backend: float = 600.0
    num_envs: int = 256
    ppo_steps_per_env: int | None = None
    """PPO-only rollout override; SuperMimic keeps its own real_steps setting."""
    seed: int = 42
    hard_cap: int = 50_000
    initial_state_file: str | None = None
    resume_ppo: str | None = None
    resume_supermimic: str | None = None
    reuse_ppo_benchmark: str | None = None
    """Reuse a verified fresh PPO benchmark; still run both evaluations anew."""
    task: str = "Mjlab-Skater-Flat-Unitree-G1"
    snapshot_seconds: float = 120.0
    amp_preload: int = 4096
    supermimic: SuperMimicConfig = field(default_factory=SuperMimicConfig)

    def __post_init__(self):
        if self.ppo_steps_per_env is not None and (
                isinstance(self.ppo_steps_per_env, bool) or not isinstance(self.ppo_steps_per_env, int)
                or self.ppo_steps_per_env < 1):
            raise ValueError("ppo_steps_per_env must be a positive integer or None")


class TrialInterrupted(RuntimeError):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(f"Supervisor received signal {signum}; stopping its active subprocess")


def _interrupt(signum, frame):
    raise TrialInterrupted(signum)


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _resolve_file(path: str, label: str):
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _Supervisor:
    def __init__(self, cfg: PairedTrialConfig, output: Path):
        self.output = output
        self.cwd = Path(__file__).resolve().parents[3]
        self.started = time.monotonic()
        self.status = {
            "schema_version": 1, "status": "running", "phase": "initializing",
            "supervisor_pid": os.getpid(), "pid": None, "command": None,
            "start": _utc(), "end": None, "exitcode": None,
            "configuration": asdict(cfg), "processes": [], "error": None,
        }
        self.flush()

    def flush(self):
        self.status["updated_at"] = _utc()
        self.status["supervisor_wall_seconds"] = time.monotonic() - self.started
        _write_json(self.output / "status.json", self.status)

    def stage(self, name: str):
        self.status.update(phase=name, pid=None, command=None)
        self.flush()

    @staticmethod
    def _stop(process):
        if process.poll() is not None:
            return
        # Each owned child has its own session. Terminate its compiler/workers
        # as well, so a failed supervisor cannot leave GPU work running behind it.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)

    def run(self, phase: str, command: list[str], timeout_seconds: float):
        log_path = self.output / f"{phase}.log"
        event = {"phase": phase, "pid": None, "command": command,
                 "command_display": shlex.join(command), "cwd": str(self.cwd),
                 "stdout_log": str(log_path), "start": _utc(), "end": None,
                 "exitcode": None, "timeout_seconds": timeout_seconds}
        self.status["processes"].append(event)
        self.status.update(phase=phase, pid=None, command=command)
        self.flush()
        started = time.monotonic()
        process = None
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment.setdefault("MUJOCO_GL", "egl")
        environment.setdefault("WANDB_MODE", "disabled")
        search_paths = [str(self.cwd), str(self.cwd / "src")]
        if environment.get("PYTHONPATH"):
            search_paths.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(search_paths)
        print(f"[{_utc()}] {phase}: {shlex.join(command)}", flush=True)
        try:
            with log_path.open("xb") as log:
                process = subprocess.Popen(command, cwd=self.cwd, env=environment,
                                           stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                event["pid"] = process.pid
                self.status["pid"] = process.pid
                self.flush()
                while True:
                    remaining = timeout_seconds - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError(f"{phase} exceeded supervisor watchdog {timeout_seconds:g}s; see {log_path}")
                    try:
                        code = process.wait(timeout=min(5, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        self.flush()
                event["exitcode"] = code
                if code:
                    raise subprocess.CalledProcessError(code, command)
        except BaseException as exc:
            if process is not None:
                self._stop(process)
                event["exitcode"] = process.poll()
            event["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            event["end"] = _utc()
            event["process_wall_seconds"] = time.monotonic() - started
            self.status["pid"] = None
            self.flush()
        return event


def _benchmark_command(cfg, output, initial, backend):
    command = [sys.executable, "-m", "mjlab_husky.scripts.benchmark_backends",
               "--backend", backend, "--task", cfg.task,
               "--wall-seconds", str(cfg.seconds_per_backend),
               "--snapshot-seconds", str(cfg.snapshot_seconds),
               "--iterations", str(cfg.hard_cap),
               "--discard-iterations", "3" if backend == "ppo" else "23",
               "--num-envs", str(cfg.num_envs), "--seed", str(cfg.seed),
               "--amp-preload", str(cfg.amp_preload),
               "--initial-state-file", str(initial), "--output", str(output)]
    resume = cfg.resume_ppo if backend == "ppo" else cfg.resume_supermimic
    if resume is not None:
        command.extend(("--resume-checkpoint", str(Path(resume).expanduser().resolve())))
    if backend == "ppo" and cfg.ppo_steps_per_env is not None:
        command.extend(("--ppo-steps-per-env", str(cfg.ppo_steps_per_env)))
    if backend == "supermimic":
        defaults = asdict(SuperMimicConfig())
        for name, value in asdict(cfg.supermimic).items():
            if value == defaults[name]:
                continue
            flag = name.replace("_", "-")
            if isinstance(value, bool):
                command.append(f"--supermimic.{'' if value else 'no-'}{flag}")
            elif isinstance(value, (tuple, list)):
                command.extend((f"--supermimic.{flag}", *(str(item) for item in value)))
            else:
                command.extend((f"--supermimic.{flag}", str(value)))
    return command


def _validate_benchmark(report, cfg, backend):
    requested = report["benchmark"]
    expected = {"backend": backend, "task": cfg.task, "num_envs": cfg.num_envs,
                "seed": cfg.seed, "wall_seconds": cfg.seconds_per_backend,
                "snapshot_seconds": cfg.snapshot_seconds, "iterations": cfg.hard_cap,
                "amp_preload": cfg.amp_preload}
    if any(requested.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{backend} benchmark configuration does not match the supervised round")
    if backend == "supermimic" and json.dumps(
            requested.get("supermimic"), sort_keys=True, allow_nan=False) != json.dumps(
                asdict(cfg.supermimic), sort_keys=True, allow_nan=False):
        # JSON represents tuples as lists; compare the full configuration while
        # preserving bool/numeric types and detecting missing or extra fields.
        raise ValueError("SuperMimic benchmark full configuration differs from the supervised round")
    if backend == "ppo":
        from mjlab_husky.tasks.registry import load_rl_cfg
        default_steps = load_rl_cfg(cfg.task).num_steps_per_env
        expected_steps = default_steps if cfg.ppo_steps_per_env is None else cfg.ppo_steps_per_env
        override = requested.get("ppo_steps_per_env")
        effective = report.get("effective_steps_per_env")
        if effective is None:
            # Historical reports predate the override and always used the
            # registered rollout. None must never be interpreted as requested6.
            if override is not None:
                raise ValueError("PPO override report is missing its effective rollout")
            effective = default_steps
        if (type(effective) is not int or effective < 1 or effective != expected_steps
                or (override is not None and (type(override) is not int or override != effective))):
            raise ValueError("PPO effective rollout differs from the supervised round")
        if effective != (default_steps if override is None else override):
            raise ValueError("PPO declared override/default differs from its effective rollout")
        geometry = {"version": 1, "num_envs": cfg.num_envs, "steps_per_env": effective,
                    "transitions_per_update": cfg.num_envs * effective}
        actual = report.get("ppo_rollout_geometry")
        if actual is not None and (not isinstance(actual, dict)
                or any(type(actual.get(key)) is not int for key in geometry) or actual != geometry):
            raise ValueError("PPO reported rollout geometry is inconsistent")
        if report.get("effective_steps_per_env") is not None and actual is None:
            raise ValueError("PPO effective rollout is missing its geometry receipt")
        for iterations_key, steps_key in (("completed_iterations_this_run", "real_steps_this_run"),
                                           ("total_iterations", "total_real_steps")):
            if iterations_key not in report and steps_key not in report:
                continue
            iterations, steps = report.get(iterations_key), report.get(steps_key)
            if (type(iterations) is not int or iterations < 0 or type(steps) is not int or steps < 0
                    or steps != iterations * geometry["transitions_per_update"]):
                raise ValueError("PPO reported counts differ from num_envs * effective rollout * updates")
        for row in report.get("rows", []):
            if (type(row.get("real_transitions")) is not int
                    or row["real_transitions"] != geometry["transitions_per_update"]):
                raise ValueError("PPO row transition count differs from its effective rollout")
    if report.get("stop_reason") not in ("wall_budget", "iteration_cap"):
        raise ValueError(f"{backend} benchmark has no valid bounded stop reason")
    for key in ("end_to_end_seconds", "completed_iterations_this_run", "real_steps_this_run"):
        if not isinstance(report.get(key), (int, float)) or not math.isfinite(report[key]) or report[key] < 0:
            raise ValueError(f"{backend} benchmark has invalid {key}")
    checkpoint = _resolve_file(report["final_checkpoint"], f"{backend} final checkpoint")
    if not report.get("final_checkpoint_sha256") or not report.get("shared_initial_state", {}).get("sha256"):
        raise ValueError(f"{backend} benchmark is missing checkpoint/initialization identity")
    return checkpoint


def _reused_ppo(cfg: PairedTrialConfig, initial: Path):
    """Verify historical inputs without changing its report or checkpoint."""
    path = _resolve_file(cfg.reuse_ppo_benchmark, "Reused PPO benchmark report")
    report_hash = _sha256(path)
    report = _read_json(path)
    checkpoint = _validate_benchmark(report, cfg, "ppo")
    if report["benchmark"].get("discard_iterations") != 3:
        raise ValueError("Reused PPO discard_iterations differs from the supervised PPO configuration")
    if report.get("resume_checkpoint") is not None or any(
            report.get(total) != report.get(current) for total, current in (
                ("total_iterations", "completed_iterations_this_run"),
                ("total_real_steps", "real_steps_this_run"))):
        raise ValueError("Reused PPO must be a fresh run with zero prior training")
    identity = report["shared_initial_state"]
    if identity.get("resume_initialization_identity_unverified", False):
        raise ValueError("Reused PPO initialization identity was not verified")
    initial_hash = _sha256(initial)
    if identity["sha256"] != initial_hash:
        raise ValueError("Reused PPO shared initialization SHA256 differs from --initial-state-file")
    for starting, initial_key in (("starting_policy_sha256", "policy_sha256"),
                                  ("starting_discriminator_sha256", "discriminator_sha256")):
        if not identity.get(initial_key) or report.get(starting) != identity[initial_key]:
            raise ValueError(f"Reused PPO {starting} does not match its shared initialization")
    checkpoint_hash = _sha256(checkpoint)
    if checkpoint_hash != report["final_checkpoint_sha256"]:
        raise ValueError("Reused PPO final checkpoint SHA256 does not match its benchmark report")
    commit = report.get("code", {}).get("commit") or report.get("source_commit")
    warning = (
        "Reused PPO benchmark has no recorded source commit; full source identity is not verified."
        if commit is None else
        "A historical PPO source commit is recorded; equivalence to the current source is not verified."
    )
    receipt = {"benchmark_report": str(path), "benchmark_report_sha256": report_hash,
               "final_checkpoint": str(checkpoint), "final_checkpoint_sha256": checkpoint_hash,
               "shared_initial_state_file": str(initial), "shared_initial_state_sha256": initial_hash,
               "starting_policy_sha256": report["starting_policy_sha256"],
               "starting_discriminator_sha256": report["starting_discriminator_sha256"],
               "reported_source_commit": commit, "full_source_identity_verified": False,
               "warnings": [warning]}
    return report, receipt


def _check_reuse_unchanged(receipt):
    for path_key, hash_key in (("benchmark_report", "benchmark_report_sha256"),
                               ("final_checkpoint", "final_checkpoint_sha256"),
                               ("shared_initial_state_file", "shared_initial_state_sha256")):
        if _sha256(Path(receipt[path_key])) != receipt[hash_key]:
            raise ValueError(f"Reused PPO source changed during this round: {receipt[path_key]}")


def _validate_evaluation(report, cfg, backend, benchmark):
    protocol = report["configuration"]
    if report.get("schema_version") != 2 or report.get("quality_evaluation") is not True:
        raise ValueError(f"{backend} evaluation must use original-task metric schema 2")
    if report.get("finite_measurements") is not True:
        raise ValueError(f"{backend} evaluation contains nonfinite physical measurements")
    expected = {"task": cfg.task, "num_envs": 64, "seconds": 20.0,
                "seed": cfg.seed, "presentation_mode": False, "video": False}
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{backend} evaluation protocol differs from the supervised round")
    if report["checkpoint"]["sha256"] != benchmark["final_checkpoint_sha256"]:
        raise ValueError(f"{backend} evaluation did not load its final training checkpoint")


def _metric_comparison(evaluations):
    selectors = {
        "task_reward_mean": (lambda item: item["measurements"]["task_reward"]["mean"], "higher"),
        "survival_20s_env_fraction": (lambda item: item["episodes"]["survival_20s_env_fraction"], "higher"),
        "physical_failures": (lambda item: item["episodes"]["physical_failures"], "lower"),
        "envs_with_failure": (lambda item: item["episodes"]["envs_with_failure"], "lower"),
        "completed_episode_seconds_mean": (lambda item: item["measurements"]["completed_episode_seconds"]["mean"], "higher"),
        "push_velocity_mae_mps": (lambda item: item["measurements"]["push_velocity_abs_error_mps"]["mean"], "lower"),
        "steer_heading_mae_rad": (lambda item: item["measurements"]["steer_heading_abs_error_rad"]["mean"], "lower"),
    }
    result = {}
    for name, (select, direction) in selectors.items():
        ppo, supermimic = (select(evaluations[backend]) for backend in ("ppo", "supermimic"))
        result[name] = {"ppo": ppo, "supermimic": supermimic, "better_direction": direction,
                        "supermimic_minus_ppo": supermimic - ppo
                        if ppo is not None and supermimic is not None else None}
    return result


def run_trial(cfg: PairedTrialConfig):
    if not math.isfinite(cfg.seconds_per_backend) or cfg.seconds_per_backend <= 0:
        raise ValueError("seconds_per_backend must be positive and finite")
    if not math.isfinite(cfg.snapshot_seconds) or cfg.snapshot_seconds <= 0:
        raise ValueError("snapshot_seconds must be positive and finite")
    if min(cfg.num_envs, cfg.hard_cap, cfg.amp_preload) < 1:
        raise ValueError("num_envs, hard_cap and amp_preload must be positive")
    output = Path(cfg.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".paired_trial.lock"
    lock_descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(lock_descriptor, f"{os.getpid()}\n".encode())
    os.close(lock_descriptor)
    supervisor = None
    old_handlers = {}
    try:
        # Never reinterpret a partially completed round as authorization to run
        # it twice. Resuming training always starts a new supervised round.
        if (output / "status.json").exists() or (output / "paired_report.json").exists():
            raise FileExistsError(f"Use a new --output-dir for this single round: {output}")
        supervisor = _Supervisor(cfg, output)
        for signum in (signal.SIGTERM, signal.SIGINT):
            old_handlers[signum] = signal.signal(signum, _interrupt)
        resume_paths = {"ppo": cfg.resume_ppo, "supermimic": cfg.resume_supermimic}
        if cfg.reuse_ppo_benchmark is not None:
            if cfg.initial_state_file is None:
                raise ValueError("--reuse-ppo-benchmark requires an explicit --initial-state-file")
            if any(resume_paths.values()):
                raise ValueError("PPO reuse compares fresh runs and cannot be combined with resume checkpoints")
        for backend, path in resume_paths.items():
            if path is not None:
                _resolve_file(path, f"{backend} resume checkpoint")
        if any(resume_paths.values()) and cfg.initial_state_file is None:
            raise ValueError("A continuation round needs --initial-state-file pointing to its original shared initialization")
        initial = (Path(cfg.initial_state_file).expanduser().resolve() if cfg.initial_state_file
                   else output / "shared_initial_state.pt")
        if any(resume_paths.values()) or cfg.reuse_ppo_benchmark is not None:
            _resolve_file(str(initial), "Original shared initialization")
        elif initial.exists() and not initial.is_file():
            raise ValueError(f"Shared initialization path is not a file: {initial}")
        supervisor.status["shared_initial_state_file"] = str(initial)
        supervisor.flush()
        benchmarks, evaluations, process_records, benchmark_paths = {}, {}, {}, {}
        reused_ppo = None
        if cfg.reuse_ppo_benchmark is not None:
            supervisor.stage("validate_reused_ppo")
            benchmarks["ppo"], reused_ppo = _reused_ppo(cfg, initial)
            benchmark_paths["ppo"] = reused_ppo["benchmark_report"]
            supervisor.status.update(reused_ppo_training=True, reused_ppo_source=reused_ppo)
            supervisor.flush()
            for warning in reused_ppo["warnings"]:
                print(f"[{_utc()}] WARNING: {warning}", flush=True)
        for backend in ("ppo", "supermimic"):
            if backend == "ppo" and reused_ppo is not None:
                continue
            command = _benchmark_command(cfg, output, initial, backend)
            # The benchmark enforces its inclusive training budget. This extra
            # watchdog bounds a hung import, compilation or simulator startup.
            process_records[f"train_{backend}"] = supervisor.run(
                f"train_{backend}", command, cfg.seconds_per_backend + 600.0)
            supervisor.stage(f"read_benchmark_{backend}")
            path = output / backend / "benchmark.json"
            benchmark = _read_json(path)
            _validate_benchmark(benchmark, cfg, backend)
            benchmarks[backend] = benchmark
            benchmark_paths[backend] = str(path)
        identities = [benchmarks[backend]["shared_initial_state"]["sha256"]
                      for backend in ("ppo", "supermimic")]
        if identities[0] != identities[1]:
            raise ValueError("The two backends did not use the same shared initialization file")
        both_fresh = not any(resume_paths.values())
        if both_fresh:
            for field in ("starting_policy_sha256", "starting_discriminator_sha256"):
                if not benchmarks["ppo"].get(field) or benchmarks["ppo"][field] != benchmarks["supermimic"].get(field):
                    raise ValueError(f"Fresh backends did not have identical {field}")
        for backend in ("ppo", "supermimic"):
            (output / backend).mkdir(parents=True, exist_ok=True)
            evaluation_path = output / backend / "evaluation.json"
            command = [sys.executable, "-m", "mjlab_husky.scripts.evaluate_checkpoint",
                       "--checkpoint-file", benchmarks[backend]["final_checkpoint"],
                       "--task", cfg.task, "--num-envs", "64", "--seconds", "20",
                       "--seed", str(cfg.seed), "--output", str(evaluation_path)]
            process_records[f"evaluate_{backend}"] = supervisor.run(
                f"evaluate_{backend}", command, 300.0)
            supervisor.stage(f"read_evaluation_{backend}")
            evaluation = _read_json(evaluation_path)
            _validate_evaluation(evaluation, cfg, backend, benchmarks[backend])
            evaluations[backend] = evaluation
        if evaluations["ppo"]["comparison_config_sha256"] != evaluations["supermimic"]["comparison_config_sha256"]:
            raise ValueError("The two physical evaluations used different comparison configurations")
        if reused_ppo is not None:
            _check_reuse_unchanged(reused_ppo)
        cases = {}
        for backend in ("ppo", "supermimic"):
            benchmark, evaluation = benchmarks[backend], evaluations[backend]
            reused = backend == "ppo" and reused_ppo is not None
            process_seconds = (0.0 if reused else
                               process_records[f"train_{backend}"]["process_wall_seconds"])
            new_training_seconds = 0.0 if reused else benchmark["end_to_end_seconds"]
            cases[backend] = {
                "benchmark_report": benchmark_paths[backend],
                "reused_ppo_training": reused,
                "training_executed_this_round": not reused,
                "benchmark_source": reused_ppo if reused else None,
                "evaluation_report": str(output / backend / "evaluation.json"),
                "training_configuration": benchmark["benchmark"],
                "requested_training_wall_seconds": cfg.seconds_per_backend,
                "actual_inclusive_training_wall_seconds": benchmark["end_to_end_seconds"],
                "training_wall_seconds_spent_this_round": new_training_seconds,
                "budget_overrun_seconds": benchmark.get("budget_overrun_seconds"),
                "training_subprocess_wall_seconds": process_seconds,
                "setup_and_process_overhead_seconds": max(0, process_seconds - new_training_seconds),
                "evaluation_subprocess_wall_seconds": process_records[f"evaluate_{backend}"]["process_wall_seconds"],
                "stop_reason": benchmark["stop_reason"],
                "completed_iterations_this_run": benchmark["completed_iterations_this_run"],
                "real_steps_this_run": benchmark["real_steps_this_run"],
                "training_counters_origin": "reused_benchmark" if reused else "new_benchmark",
                "completed_iterations_this_round": 0 if reused else benchmark["completed_iterations_this_run"],
                "real_steps_this_round": 0 if reused else benchmark["real_steps_this_run"],
                "prior_iterations": benchmark["total_iterations"] - benchmark["completed_iterations_this_run"],
                "prior_real_steps": benchmark["total_real_steps"] - benchmark["real_steps_this_run"],
                "total_iterations": benchmark["total_iterations"], "total_real_steps": benchmark["total_real_steps"],
                "steady_seconds_per_update": benchmark.get("mean_seconds"),
                "final_checkpoint": benchmark["final_checkpoint"],
                "final_checkpoint_sha256": benchmark["final_checkpoint_sha256"],
                "resume_checkpoint": benchmark.get("resume_checkpoint"),
                "resume_semantics": benchmark.get("resume_semantics"),
                "snapshots": benchmark.get("snapshots", []),
                "episodes": evaluation["episodes"], "measurements": evaluation["measurements"],
            }
        report = {
            "schema_version": 1, "scope": "single-round fixed-wall-budget screening",
            "interpretation": (
                "PPO training is reused from a matching historical benchmark; only SuperMimic training is executed this round. Both final checkpoints are evaluated anew. Full source identity is not verified. "
                if reused_ppo is not None else ""
            ) + "One seed and one bounded round. This does not establish convergence or time to target quality. Per-update durations use different numbers of real transitions.",
            "configuration": asdict(cfg), "shared_initial_state_sha256": identities[0],
            "reused_ppo_training": reused_ppo is not None,
            "reused_ppo_source": reused_ppo,
            "initialization_identity_verified": not any(
                benchmarks[backend]["shared_initial_state"].get("resume_initialization_identity_unverified", False)
                for backend in ("ppo", "supermimic")),
            "continuation_mode": {"ppo": cfg.resume_ppo is not None, "supermimic": cfg.resume_supermimic is not None},
            "prior_budget_comparability": ("Reused PPO and new SuperMimic both started fresh from identical initialization; matching requested training budgets are verified. PPO training time was spent in the source run."
                                          if reused_ppo is not None else
                                          "Both fresh from identical initialization; zero prior training."
                                          if both_fresh else
                                          "Continuation counters are reported; equality of earlier wall budgets is not verified."),
            "evaluation_comparison_config_sha256": evaluations["ppo"]["comparison_config_sha256"],
            "budget": {"requested_training_seconds_per_backend": cfg.seconds_per_backend,
                       "requested_total_training_seconds": (1 if reused_ppo is not None else 2) * cfg.seconds_per_backend,
                       "requested_comparison_training_seconds": 2 * cfg.seconds_per_backend,
                       "new_training_backends": ["supermimic"] if reused_ppo is not None else ["ppo", "supermimic"],
                       "both_reached_wall_budget": all(item["stop_reason"] == "wall_budget" for item in benchmarks.values()),
                       "actual_total_inclusive_training_seconds": sum(case["training_wall_seconds_spent_this_round"] for case in cases.values()),
                       "comparison_total_inclusive_training_seconds": sum(case["actual_inclusive_training_wall_seconds"] for case in cases.values()),
                       "reused_training_seconds": cases["ppo"]["actual_inclusive_training_wall_seconds"] if reused_ppo is not None else 0.0,
                       "supervisor_wall_seconds": time.monotonic() - supervisor.started,
                       "training_clock_includes": benchmarks["ppo"].get("training_clock_includes"),
                       "setup_and_evaluation_are_outside_training_budget": True,
                       "training_process_watchdog_seconds": cfg.seconds_per_backend + 600.0,
                       "evaluation_process_watchdog_seconds": 300.0},
            "cases": cases, "paired_evaluation": _metric_comparison(evaluations),
        }
        supervisor.stage("write_paired_report")
        report_path = output / "paired_report.json"
        _write_json(report_path, report)
        supervisor.status.update(status="complete", phase="complete", end=_utc(), exitcode=0,
                                 report=str(report_path), pid=None, command=None)
        supervisor.flush()
        print(json.dumps({"status": "complete", "report": str(report_path),
                          "paired_evaluation": report["paired_evaluation"]}, indent=2), flush=True)
        return report
    except BaseException as exc:
        if supervisor is not None:
            supervisor.status.update(status="failed", end=_utc(), exitcode=(128 + exc.signum)
                                     if isinstance(exc, TrialInterrupted) else 1,
                                     error={"type": type(exc).__name__, "message": str(exc)})
            supervisor.flush()
        raise
    finally:
        for signum, previous in old_handlers.items():
            signal.signal(signum, previous)
        lock.unlink(missing_ok=True)


def main():
    import tyro
    cfg = tyro.cli(PairedTrialConfig, config=(tyro.conf.AvoidSubcommands,))
    try:
        run_trial(cfg)
    except TrialInterrupted as exc:
        print(str(exc), file=sys.stderr, flush=True)
        raise SystemExit(128 + exc.signum) from exc
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
