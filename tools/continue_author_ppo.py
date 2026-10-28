"""Bounded original HUSKY PPO continuation from the legacy10K checkpoint.

This is a policy/critic/observation-normalizer/std and policy-Adam warm start.
Legacy checkpoints omit discriminator, AMP normalization/replay and RNG: those
start fresh. New snapshots include their learning state, but do not serialize
the real simulator, rollout-local variables or claim trajectory continuation.
Runner init/log/save hooks and an action-boundary guard leave task, reward and
PPO/AMP update code intact.
Invocation is a training action; preparation/testing must not invoke main().
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import time


TASK = "Mjlab-Skater-Flat-Unitree-G1"
GROUPS = ["policy", "amp_trunk", "amp_head"]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def deadline_timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Deadline must be timezone-aware ISO8601, for example2026-10-29T20:20:00Z")
    return parsed.timestamp()


@dataclass
class UpdateBudget:
    parent_updates: int
    additional_updates: int
    deadline: float
    completed: int = 0

    def __post_init__(self):
        if self.parent_updates != 10000 or not 1 <= self.additional_updates <= 2000:
            raise ValueError("This authorization permits parent10000 and only1..2000 additional updates")
        if not math.isfinite(self.deadline):
            raise ValueError("Deadline must be finite")

    @property
    def next_iteration(self):
        return self.parent_updates + self.completed

    def complete(self, iteration):
        if self.completed >= self.additional_updates or iteration != self.next_iteration:
            raise RuntimeError("Update cursor/order or authorized cap violated")
        self.completed += 1
        return {"additional_completed": self.completed, "iteration": iteration,
                "cumulative_updates": self.parent_updates + self.completed,
                "next_iteration": self.next_iteration}

    def stop_reason(self, now):
        if self.completed >= self.additional_updates:
            return "additional_update_cap"
        if now >= self.deadline:
            return "deadline"
        return None

    def checkpoint_due(self):
        return self.completed == 50 or self.completed > 0 and self.completed % 500 == 0


def filtered_policy_adam(saved, optimizer):
    """Preserve policy moments and ALL group hyperparameters, discard AMP state."""
    old_groups = saved["param_groups"]
    new_groups = optimizer.param_groups
    if [group.get("name") for group in old_groups] != GROUPS or [group.get("name") for group in new_groups] != GROUPS:
        raise ValueError("Expected original named policy/amp_trunk/amp_head Adam groups in order")
    for old, new in zip(old_groups, new_groups):
        if len(old["params"]) != len(new["params"]):
            raise ValueError("Checkpoint/current optimizer group parameter counts differ")
        if not math.isfinite(float(old["lr"])) or float(old["lr"]) <= 0:
            raise ValueError("Invalid saved optimizer learning rate")
    policy_ids = set(old_groups[0]["params"])
    state = {key: copy.deepcopy(value) for key, value in saved["state"].items() if key in policy_ids}
    for old_id, parameter in zip(old_groups[0]["params"], new_groups[0]["params"]):
        for key, value in state.get(old_id, {}).items():
            if key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq") and value.shape != parameter.shape:
                raise ValueError("Saved policy Adam moment shape differs from its parameter")
    return {"state": state, "param_groups": copy.deepcopy(old_groups)}


def warm_start(runner, checkpoint, parent_updates=10000):
    if parent_updates != 10000 or int(checkpoint["iter"]) != parent_updates-1:
        raise ValueError("Require the exact legacy iter9999/10000-update parent")
    if checkpoint.get("format") == "husky-author-ppo-learning-state-v1":
        raise ValueError("This tool starts from legacy10K; enhanced-snapshot restore is not this warm-start path")
    runner.alg.policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    selected = filtered_policy_adam(checkpoint["optimizer_state_dict"], runner.alg.optimizer)
    runner.alg.optimizer.load_state_dict(selected)
    runner.alg.learning_rate = float(runner.alg.optimizer.param_groups[0]["lr"])
    runner.current_learning_iteration = parent_updates
    policy = set(runner.alg.optimizer.param_groups[0]["params"])
    if any(parameter not in policy for parameter in runner.alg.optimizer.state):
        raise AssertionError("Legacy AMP Adam state was attached to a fresh discriminator")
    return {"scope": "policy/critic/actor+critic normalization/std plus policy Adam warm start; fresh simulation",
            "parent_checkpoint_iter": int(checkpoint["iter"]), "parent_updates": parent_updates,
            "next_iteration": runner.current_learning_iteration,
            "restored_policy_adam_parameter_states": len(runner.alg.optimizer.state),
            "reset": ["discriminator weights", "AMP normalizer", "AMP replay", "AMP Adam moments", "RNG/trajectory"],
            "group_learning_rates": {group["name"]: float(group["lr"]) for group in runner.alg.optimizer.param_groups},
            "algorithm_learning_rate": runner.alg.learning_rate}


def assert_finite(value, path="root"):
    """Check nested tensors, NumPy arrays and scalar numbers, with useful paths."""
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"Nonfinite tensor at {path}")
    elif isinstance(value, np.ndarray):
        if value.dtype.kind in "fc" and not bool(np.isfinite(value).all()):
            raise FloatingPointError(f"Nonfinite array at {path}")
    elif isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite(item, f"{path}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise FloatingPointError(f"Nonfinite scalar at {path}")


def cpu_copy(value):
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    if isinstance(value, torch.device):
        return str(value)
    return copy.deepcopy(value)


def capture_rng():
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def learning_snapshot(runner, budget, lineage, infos=None):
    """Capture the actual AMP learning state; no simulator/trajectory promise."""
    alg = runner.alg
    optimizers = {name: cpu_copy(value.state_dict()) for name, value in vars(alg).items()
                  if "optimizer" in name and value is not None and hasattr(value, "state_dict")}
    payload = {"format": "husky-author-ppo-learning-state-v1",
        "model_state_dict": cpu_copy(alg.policy.state_dict()), "optimizer_state_dict": optimizers["optimizer"],
        "iter": budget.next_iteration-1, "next_iteration": budget.next_iteration, "infos": infos,
        "discriminator_state_dict": cpu_copy(alg.discriminator.state_dict()),
        "amp_normalizer": cpu_copy(vars(alg.amp_normalizer)),
        "amp_replay": cpu_copy(vars(alg.amp_storage)), "amp_expert": cpu_copy(vars(alg.amp_data)),
        "amp_history": cpu_copy(getattr(runner, "amp_obs_frames", None)),
        "optimizers": optimizers, "algorithm_learning_rate": float(alg.learning_rate),
        "rng": cpu_copy(capture_rng()), "runner_config": cpu_copy(runner.cfg),
        "runner_tot_timesteps": runner.tot_timesteps, "runner_tot_time": runner.tot_time,
        "lineage": dict(lineage, parent_updates=budget.parent_updates, additional_completed=budget.completed,
            cumulative_updates=budget.next_iteration, deadline_utc=datetime.fromtimestamp(budget.deadline, timezone.utc).isoformat()),
        "resume_scope": "All captured learning states retained for future restoration work. Fresh-simulator restart only: simulator state, rollout-local obs/reward histories and learn() initialization are not restored by this tool. The legacy parent was not an exact resume."}
    if getattr(alg, "rnd", None) is not None:
        payload["rnd_state_dict"] = cpu_copy(alg.rnd.state_dict())
    assert_finite(payload)
    return payload


def canonical_config(value):
    if is_dataclass(value):
        return canonical_config(asdict(value))
    if isinstance(value, dict):
        return {str(key): canonical_config(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [canonical_config(item) for item in value]
    if isinstance(value, functools.partial):
        return {"partial": canonical_config(value.func), "args": canonical_config(value.args),
                "keywords": canonical_config(value.keywords)}
    if callable(value):
        return f"{getattr(value, '__module__', type(value).__module__)}.{getattr(value, '__qualname__', type(value).__qualname__)}"
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "tolist"):
        return canonical_config(value.tolist())
    return str(value)


def source_identity(root):
    paths = []
    for prefix in ("src/mjlab_husky/tasks/skater", "src/mjlab_husky/envs", "src/mjlab_husky/rl", "rsl_rl"):
        paths.extend((root/prefix).rglob("*.py"))
    paths.extend([root/"src/mjlab_husky/scripts/train.py", Path(__file__).resolve()])
    manifest = {str(path.relative_to(root)): sha(path) for path in sorted(set(paths))}
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return {"git_commit": git.stdout.strip() if git.returncode == 0 else None,
            "source_sha256": manifest,
            "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()}


class CompletedBoundaryStop(Exception):
    pass


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("This4096-environment training harness requires the authorized CUDA host")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parent-updates", type=int, required=True)
    parser.add_argument("--additional-updates", type=int, required=True)
    parser.add_argument("--deadline", required=True, help="Timezone-aware ISO8601/UTC absolute stop deadline")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        parser.error("run-name must be a fresh ASCII letters/digits/underscore/hyphen component")
    try:
        budget = UpdateBudget(args.parent_updates, args.additional_updates, deadline_timestamp(args.deadline))
    except ValueError as error:
        parser.error(str(error))
    if budget.stop_reason(time.time()):
        parser.error("Deadline already reached; no environment or training initialized")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("WANDB_MODE", "disabled")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if os.environ.get("WORLD_SIZE", "1") != "1":
        parser.error("Only the authorized single-process4096-environment run is supported")
    import mjlab.tasks  # Initialize task discovery before the custom plugin.
    import numpy as np
    import torch
    from mjlab_husky.scripts.train import TrainConfig, run_train
    from rsl_rl.runners import AMPOnPolicyRunner
    require_cuda()
    args.checkpoint = args.checkpoint.resolve()
    parent = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert_finite(parent, "parent")
    if int(parent["iter"]) != 9999:
        raise ValueError("Use the verified exact10K legacy model9999 checkpoint")
    cfg = TrainConfig.from_task(TASK)
    if (cfg.agent.algorithm.class_name != "AMP_PPO" or cfg.agent.num_steps_per_env != 24 or
            cfg.agent.amp_num_obs != 23 or cfg.agent.amp_num_frames != 5):
        raise ValueError("Original author task/PPO/AMP configuration differs")
    cfg.env.scene.num_envs = 4096
    cfg.agent.max_iterations = budget.additional_updates
    cfg.agent.resume = False  # Original runner.load would restore invalid AMP moments and cursor9999.
    cfg.agent.logger = "tensorboard"
    cfg.agent.run_name = args.run_name
    root = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    directories = {kind: out/kind/args.run_name for kind in ("data", "log", "checkpoints")}
    if any(path.exists() for path in directories.values()):
        raise FileExistsError("Run name already has output; choose a fresh name, never overwrite or replay it")
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=False)
    config = {"env": canonical_config(cfg.env), "agent": canonical_config(cfg.agent)}
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()
    write_json(directories["data"]/"config_identity.json", {"sha256": config_sha, "config": config})
    lineage = {"parent_path": str(args.checkpoint), "parent_sha256": sha(args.checkpoint),
               "task": TASK, "run_name": args.run_name, "config_sha256": config_sha}
    receipt = {"status": "initializing", "pid": os.getpid(), "started_at": time.time(), **lineage,
        "parent_updates": budget.parent_updates, "additional_cap": budget.additional_updates,
        "cumulative_cap": budget.parent_updates+budget.additional_updates,
        "deadline_utc": datetime.fromtimestamp(budget.deadline, timezone.utc).isoformat(),
        "deadline_rule": "Check completed-update boundaries, after checkpoint calls, and before alg.act at storage.step==0. Stop before a new rollout; never interrupt storage.step>0. Initialization deadline is also checked.",
        "source_identity": source_identity(root), "cfg_max_iterations": cfg.agent.max_iterations,
        "resume_scope": "Legacy policy/critic/norm/std plus filtered policy Adam warm start. AMP state/RNG fresh; no exact legacy or simulator-trajectory resume."}
    write_json(directories["data"]/"launch.json", receipt)
    ledger_path = directories["log"]/"updates.jsonl"
    ledger_path.touch(exist_ok=False)
    holder, saves = {}, {}
    original_init, original_log, original_save = AMPOnPolicyRunner.__init__, AMPOnPolicyRunner.log, AMPOnPolicyRunner.save
    force_reason = None

    def save_at_boundary(runner, reason):
        nonlocal force_reason
        started = time.time()
        force_reason = reason
        try:
            runner.save(str(directories["checkpoints"]/f"model_{budget.next_iteration-1}.pt"))
        finally:
            force_reason = None
        finished = time.time()
        saved = saves.get(budget.completed)
        if saved is not None:
            saved["save_call_completed_at"] = finished
            saved["deadline_reached_at_save_completion"] = finished >= budget.deadline
            saved["deadline_crossed_while_saving"] = started < budget.deadline <= finished
            if reason == "new_update_milestone" and finished >= budget.deadline:
                saved["reason"] = "deadline_during_checkpoint"
            write_json(directories["data"]/f"checkpoint_{budget.completed}.json", saved)

    def observed_init(self, *positional, **keywords):
        holder["environment"] = positional[0] if positional else keywords.get("env")
        original_init(self, *positional, **keywords)
        holder["runner"] = self
        self.logger_type = "tensorboard"  # Also defined before an initialization-deadline save.
        if self.env.num_envs != 4096 or self.env.num_actions != 23 or self.is_distributed:
            raise ValueError("Require original4096-env/23-action single-process runner")
        warm = warm_start(self, parent, budget.parent_updates)
        write_json(directories["data"]/"warm_start.json", warm)
        # Original learn() performs writer/code-state work after log(). Guard
        # the first actual rollout action as well, so that work cannot make us
        # start a further update after the deadline. A partial rollout completes.
        original_act = self.alg.act
        holder["original_act"] = original_act

        def guarded_act(*act_args, **act_kwargs):
            if int(self.alg.storage.step) == 0:
                boundary_reason = budget.stop_reason(time.time())
                if boundary_reason:
                    stop = "deadline_before_rollout" if boundary_reason == "deadline" else boundary_reason
                    save_at_boundary(self, stop)
                    raise CompletedBoundaryStop(stop)
            return original_act(*act_args, **act_kwargs)

        self.alg.act = guarded_act
        if budget.stop_reason(time.time()):
            save_at_boundary(self, "deadline_during_initialization")
            raise CompletedBoundaryStop("deadline_during_initialization")

    def observed_save(self, path, infos=None):
        # Original runner autosaves by absolute index; use actual new-update milestones.
        if force_reason is None or budget.completed in saves:
            return
        destination = directories["checkpoints"]/f"model_{budget.next_iteration-1}.pt"
        if destination.exists():
            raise FileExistsError("Checkpoint destination already exists")
        payload = learning_snapshot(self, budget, lineage, infos)
        temporary = destination.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
        temporary.replace(destination)
        saved = {"additional_completed": budget.completed, "cumulative_updates": budget.next_iteration,
                 "iteration": budget.next_iteration-1, "path": str(destination), "sha256": sha(destination),
                 "reason": force_reason, "timestamp": time.time(), "all_learning_state_finite": True}
        saves[budget.completed] = saved
        write_json(directories["data"]/f"checkpoint_{budget.completed}.json", saved)

    def observed_log(self, locs, *positional, **keywords):
        row = budget.complete(int(locs["it"]))
        try:
            assert_finite({"losses": locs["loss_dict"], "policy": self.alg.policy.state_dict(),
                "discriminator": self.alg.discriminator.state_dict(), "adam": self.alg.optimizer.state_dict(),
                "amp_normalizer": vars(self.alg.amp_normalizer)}, "completed_update")
        except FloatingPointError as error:
            with ledger_path.open("a") as stream:
                stream.write(json.dumps({**row, "finite": False, "error": str(error), "timestamp": time.time()})+"\n")
            raise
        row.update(timestamp=time.time(), finite=True,
                   losses={key: float(value) for key, value in locs["loss_dict"].items()},
                   seconds=float(locs["collection_time"]+locs["learn_time"]),
                   algorithm_learning_rate=float(self.alg.learning_rate),
                   group_learning_rates={group["name"]: float(group["lr"]) for group in self.alg.optimizer.param_groups})
        with ledger_path.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
        original_log(self, locs, *positional, **keywords)
        reason = budget.stop_reason(time.time())
        if reason or budget.checkpoint_due():
            save_at_boundary(self, reason or "new_update_milestone")
            # Snapshot copying, hashing and subclass save hooks can themselves
            # cross the absolute deadline. Do not return to learn() in that case.
            after_save_reason = budget.stop_reason(time.time())
            if reason is None and after_save_reason:
                reason = ("deadline_during_checkpoint" if saves.get(budget.completed, {}).get("deadline_reached_at_save_completion")
                          else after_save_reason)
        write_json(directories["data"]/"status.json", {**row, "status": "stopping" if reason else "running"})
        if reason:
            raise CompletedBoundaryStop(reason)

    AMPOnPolicyRunner.__init__, AMPOnPolicyRunner.log, AMPOnPolicyRunner.save = observed_init, observed_log, observed_save
    error = None
    reason = None
    try:
        run_train(TASK, cfg, directories["checkpoints"])
        raise RuntimeError("Runner returned without the bounded completed-update stop hook")
    except CompletedBoundaryStop as stopped:
        reason = str(stopped)
    except BaseException as caught:
        error = caught
    finally:
        AMPOnPolicyRunner.__init__, AMPOnPolicyRunner.log, AMPOnPolicyRunner.save = original_init, original_log, original_save
        runner = holder.get("runner")
        cleanup_errors = []
        if runner is not None:
            if "original_act" in holder:
                runner.alg.act = holder["original_act"]
            if runner.writer is not None:
                try:
                    runner.writer.flush()
                    runner.writer.close()
                except Exception as cleanup_error:
                    cleanup_errors.append(repr(cleanup_error))
        environment = runner.env if runner is not None else holder.get("environment")
        if environment is not None:
            try:
                environment.close()
            except Exception as cleanup_error:
                cleanup_errors.append(repr(cleanup_error))
        final = {"status": "failed" if error else "stopped", "stop_reason": reason,
            "error": repr(error) if error else None, "additional_completed": budget.completed,
            "cumulative_updates": budget.next_iteration, "last_completed_iteration": budget.next_iteration-1,
            "parent_sha256": lineage["parent_sha256"], "finished_at": time.time(),
            "cleanup_errors": cleanup_errors,
            "checkpoints": list(saves.values()), "unexpected_error_has_no_unverified_final_snapshot": bool(error)}
        write_json(directories["data"]/"exit.json", final)
        write_json(directories["data"]/"status.json", final)
    if error:
        raise error
    print(json.dumps(final, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
