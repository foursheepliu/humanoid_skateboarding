"""One supervised wall-budget screening round, with serial GPU subprocesses.

Example (run from the repository in the prepared training environment)::

  nohup python -m mjlab_husky.scripts.paired_trial --output-dir logs/paired_001 \
    > paired_001_supervisor.log 2>&1 &

This executes one configuration once. A subsequent round needs a new output
directory and explicit resume checkpoints; there is no restart or tuning loop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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


@dataclass(frozen=True)
class PairedTrialConfig:
    output_dir: str
    seconds_per_backend: float = 600.0
    num_envs: int = 256
    seed: int = 42
    hard_cap: int = 50_000
    initial_state_file: str | None = None
    resume_ppo: str | None = None
    resume_supermimic: str | None = None
    task: str = "Mjlab-Skater-Flat-Unitree-G1"
    snapshot_seconds: float = 120.0
    amp_preload: int = 4096


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
    return command


def _validate_benchmark(report, cfg, backend):
    requested = report["benchmark"]
    expected = {"backend": backend, "task": cfg.task, "num_envs": cfg.num_envs,
                "seed": cfg.seed, "wall_seconds": cfg.seconds_per_backend,
                "snapshot_seconds": cfg.snapshot_seconds, "iterations": cfg.hard_cap,
                "amp_preload": cfg.amp_preload}
    if any(requested.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{backend} benchmark configuration does not match the supervised round")
    if report.get("stop_reason") not in ("wall_budget", "iteration_cap"):
        raise ValueError(f"{backend} benchmark has no valid bounded stop reason")
    for key in ("end_to_end_seconds", "completed_iterations_this_run", "real_steps_this_run"):
        if not isinstance(report.get(key), (int, float)) or not math.isfinite(report[key]) or report[key] < 0:
            raise ValueError(f"{backend} benchmark has invalid {key}")
    checkpoint = _resolve_file(report["final_checkpoint"], f"{backend} final checkpoint")
    if not report.get("final_checkpoint_sha256") or not report.get("shared_initial_state", {}).get("sha256"):
        raise ValueError(f"{backend} benchmark is missing checkpoint/initialization identity")
    return checkpoint


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
        for backend, path in resume_paths.items():
            if path is not None:
                _resolve_file(path, f"{backend} resume checkpoint")
        if any(resume_paths.values()) and cfg.initial_state_file is None:
            raise ValueError("A continuation round needs --initial-state-file pointing to its original shared initialization")
        initial = (Path(cfg.initial_state_file).expanduser().resolve() if cfg.initial_state_file
                   else output / "shared_initial_state.pt")
        if any(resume_paths.values()):
            _resolve_file(str(initial), "Original shared initialization")
        elif initial.exists() and not initial.is_file():
            raise ValueError(f"Shared initialization path is not a file: {initial}")
        supervisor.status["shared_initial_state_file"] = str(initial)
        supervisor.flush()
        benchmarks, evaluations, process_records = {}, {}, {}
        for backend in ("ppo", "supermimic"):
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
        cases = {}
        for backend in ("ppo", "supermimic"):
            benchmark, evaluation = benchmarks[backend], evaluations[backend]
            process_seconds = process_records[f"train_{backend}"]["process_wall_seconds"]
            cases[backend] = {
                "benchmark_report": str(output / backend / "benchmark.json"),
                "evaluation_report": str(output / backend / "evaluation.json"),
                "training_configuration": benchmark["benchmark"],
                "requested_training_wall_seconds": cfg.seconds_per_backend,
                "actual_inclusive_training_wall_seconds": benchmark["end_to_end_seconds"],
                "budget_overrun_seconds": benchmark.get("budget_overrun_seconds"),
                "training_subprocess_wall_seconds": process_seconds,
                "setup_and_process_overhead_seconds": max(0, process_seconds - benchmark["end_to_end_seconds"]),
                "evaluation_subprocess_wall_seconds": process_records[f"evaluate_{backend}"]["process_wall_seconds"],
                "stop_reason": benchmark["stop_reason"],
                "completed_iterations_this_run": benchmark["completed_iterations_this_run"],
                "real_steps_this_run": benchmark["real_steps_this_run"],
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
            "interpretation": "One seed and one bounded round. This does not establish convergence or time to target quality. Per-update durations use different numbers of real transitions.",
            "configuration": asdict(cfg), "shared_initial_state_sha256": identities[0],
            "initialization_identity_verified": not any(
                benchmarks[backend]["shared_initial_state"].get("resume_initialization_identity_unverified", False)
                for backend in ("ppo", "supermimic")),
            "continuation_mode": {"ppo": cfg.resume_ppo is not None, "supermimic": cfg.resume_supermimic is not None},
            "prior_budget_comparability": ("Both fresh from identical initialization; zero prior training."
                                          if both_fresh else
                                          "Continuation counters are reported; equality of earlier wall budgets is not verified."),
            "evaluation_comparison_config_sha256": evaluations["ppo"]["comparison_config_sha256"],
            "budget": {"requested_training_seconds_per_backend": cfg.seconds_per_backend,
                       "requested_total_training_seconds": 2 * cfg.seconds_per_backend,
                       "both_reached_wall_budget": all(item["stop_reason"] == "wall_budget" for item in benchmarks.values()),
                       "actual_total_inclusive_training_seconds": sum(case["actual_inclusive_training_wall_seconds"] for case in cases.values()),
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
