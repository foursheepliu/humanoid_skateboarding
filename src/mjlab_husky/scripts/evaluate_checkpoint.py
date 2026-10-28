"""Bounded headless physics evaluation of the original deployed ActorCritic.

No training runner, discriminator or AMP expert dataset is constructed. Default
evaluation uses the unchanged training task, including its failures, observation
noise and perturbations. Presentation mode is a separate, explicitly labelled
visualization setting and must not be used to claim training-task quality.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class EvaluationConfig:
    checkpoint_file: str
    task: str = "Mjlab-Skater-Flat-Unitree-G1"
    num_envs: int = 64
    seconds: float = 20.0
    seed: int = 42
    output: str = "logs/checkpoint_evaluation.json"
    device: str | None = None
    presentation_mode: bool = False
    video: bool = False
    video_file: str | None = None
    video_fps: int = 25
    video_width: int = 1280
    video_height: int = 720


class _Moments:
    """Streaming scalar moments on the environment device, without per-step IO."""

    def __init__(self, device):
        self.count = torch.zeros((), device=device, dtype=torch.long)
        self.nonfinite = torch.zeros_like(self.count)
        self.total = torch.zeros((), device=device, dtype=torch.float64)
        self.squares = torch.zeros_like(self.total)
        self.minimum = torch.full_like(self.total, math.inf)
        self.maximum = torch.full_like(self.total, -math.inf)

    @torch.no_grad()
    def add(self, values: Tensor):
        values = values.detach().flatten().double()
        if not values.numel():
            return
        finite = torch.isfinite(values)
        safe = torch.where(finite, values, 0)
        self.count += finite.sum()
        self.nonfinite += (~finite).sum()
        self.total += safe.sum()
        self.squares += safe.square().sum()
        self.minimum = torch.minimum(self.minimum, torch.where(finite, values, math.inf).amin())
        self.maximum = torch.maximum(self.maximum, torch.where(finite, values, -math.inf).amax())

    def result(self):
        count, nonfinite, total, squares, minimum, maximum = torch.stack((
            self.count.double(), self.nonfinite.double(), self.total,
            self.squares, self.minimum, self.maximum)).cpu().tolist()
        count = int(count)
        mean = total / count if count else None
        return {"samples": count, "nonfinite_samples": int(nonfinite), "mean": mean,
                "std": math.sqrt(max(0, squares / count - mean * mean)) if count else None,
                "min": minimum if count else None, "max": maximum if count else None}


def _json_config(value):
    """Retain full dataclass settings and name their executable callables."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (Path, torch.device)):
        return str(value)
    if isinstance(value, Enum):
        return _json_config(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_config(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_config(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_config(item) for item in value]
    if isinstance(value, set):
        return [_json_config(item) for item in sorted(value, key=str)]
    if isinstance(value, (Tensor, np.ndarray)):
        return value.detach().cpu().tolist() if isinstance(value, Tensor) else value.tolist()
    if isinstance(value, partial):
        return {"partial": _json_config(value.func), "args": _json_config(value.args),
                "keywords": _json_config(value.keywords)}
    if callable(value):
        return f"{getattr(value, '__module__', type(value).__module__)}.{getattr(value, '__qualname__', type(value).__qualname__)}"
    return {"object_type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision():
    root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL)
        return {"commit": commit, "working_tree_dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "working_tree_dirty": None}


def _done_flags(env, dones, extras, device):
    done = dones.to(device).bool().reshape(-1)
    raw = env.unwrapped
    manager = raw.termination_manager
    timeout = extras.get("time_outs", getattr(raw, "reset_time_outs", None))
    if timeout is None:
        timeout = manager.time_outs
    failure = getattr(raw, "reset_terminated", manager.terminated)
    return (torch.as_tensor(failure, device=device).bool().reshape(-1) & done,
            torch.as_tensor(timeout, device=device).bool().reshape(-1) & done, done)


def run_evaluation(cfg: EvaluationConfig) -> dict:
    if cfg.num_envs < 1 or not math.isfinite(cfg.seconds) or cfg.seconds <= 0:
        raise ValueError("num_envs and seconds must be positive")
    if cfg.video_fps < 1 or min(cfg.video_width, cfg.video_height) < 1:
        raise ValueError("Video dimensions and fps must be positive")
    checkpoint_path = Path(cfg.checkpoint_file).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    output_path = Path(cfg.output).expanduser().resolve()
    if output_path == checkpoint_path:
        raise ValueError("Evaluation output must not overwrite its checkpoint")
    os.environ.setdefault("MUJOCO_GL", "egl")
    # Native viewers and keyboard input are deliberately not imported here.
    from mjlab.utils.torch import configure_torch_backends
    from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat
    from mjlab_husky.envs import G1SkaterManagerBasedRlEnv
    from mjlab_husky.rl import RslRlVecEnvWrapper
    from mjlab_husky.tasks.registry import load_env_cfg, load_rl_cfg
    from mjlab_husky.tasks.skater.mdp.observations import heading_error
    from rsl_rl.modules import ActorCritic

    configure_torch_backends()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device(cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    env_cfg = load_env_cfg(cfg.task, play=cfg.presentation_mode)
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.seed = cfg.seed
    if cfg.video:
        env_cfg.viewer.width = cfg.video_width
        env_cfg.viewer.height = cfg.video_height
    agent_cfg = load_rl_cfg(cfg.task)
    environment_config = _json_config(env_cfg)
    agent_config = _json_config(agent_cfg)
    config_fingerprint = hashlib.sha256(json.dumps(
        {"environment": environment_config, "policy": agent_config["policy"],
         "clip_actions": agent_cfg.clip_actions,
         "protocol": {"seconds": cfg.seconds, "seed": cfg.seed, "num_envs": cfg.num_envs,
                      "presentation_mode": cfg.presentation_mode, "deterministic_actor": True,
                      "metrics_version": 2}},
        sort_keys=True, allow_nan=False).encode()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint must contain the original ActorCritic model_state_dict")
    writer = None
    video_path = None
    if cfg.video:
        import imageio.v2 as imageio
        video_path = (Path(cfg.video_file).expanduser().resolve() if cfg.video_file
                      else output_path.with_suffix(".mp4"))
        if video_path in (output_path, checkpoint_path):
            raise ValueError("Video path must differ from checkpoint and report paths")
        video_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    env = None
    try:
        env = G1SkaterManagerBasedRlEnv(
            cfg=env_cfg, device=str(device), render_mode="rgb_array" if cfg.video else None)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        raw = env.unwrapped
        # Wrapper construction already reset the task and computed initial
        # observations. Never recompute them: that may change history/noise.
        observations = raw.obs_buf
        if not isinstance(observations, Mapping) or not observations:
            raise RuntimeError("Environment reset did not retain its initial observation dictionary")
        policy_config = _json_config(agent_cfg.policy)
        policy_config.pop("class_name", None)
        policy = ActorCritic(observations, agent_cfg.obs_groups, env.num_actions,
                             **policy_config).to(device)
        policy.load_state_dict(checkpoint["model_state_dict"])
        policy.eval()
        step_dt = float(raw.step_dt)
        steps = math.ceil(cfg.seconds / step_dt)
        stride = max(1, round(1 / (step_dt * cfg.video_fps)))
        if cfg.video:
            writer = imageio.get_writer(str(video_path), fps=1 / (step_dt * stride), codec="libx264")
        names = ("task_reward", "completed_episode_return", "completed_episode_steps",
                 "completed_episode_seconds", "push_velocity_abs_error_mps",
                 "steer_heading_abs_error_rad", "board_world_speed_mps", "board_world_vx_mps",
                 "board_world_vy_mps", "board_world_vz_mps", "board_local_vx_mps",
                 "board_local_vy_mps", "board_local_vz_mps", "robot_root_height_m",
                 "push_phase_task_reward", "steer_phase_task_reward",
                 "push2steer_phase_task_reward", "steer2push_phase_task_reward",
                 "still_phase_task_reward", "push_reward_contribution", "steer_reward_contribution",
                 "transition_reward_contribution", "regularization_reward_contribution")
        stats = {name: _Moments(device) for name in names}
        episode_steps = torch.zeros(cfg.num_envs, device=device, dtype=torch.long)
        episode_returns = torch.zeros(cfg.num_envs, device=device)
        total_returns = torch.zeros_like(episode_returns)
        failure_counts = torch.zeros_like(episode_steps)
        timeout_counts = torch.zeros_like(episode_steps)
        completed_counts = torch.zeros_like(episode_steps)
        first_failure_step = torch.full_like(episode_steps, -1)
        termination_manager = raw.termination_manager
        reason_names = list(getattr(termination_manager, "active_terms", ()))
        reason_counts = {name: torch.zeros((), device=device, dtype=torch.long) for name in reason_names}
        reason_is_timeout = {
            name: bool(termination_manager.get_term_cfg(name).time_out) for name in reason_names}
        reward_managers = {
            "push": getattr(raw, "push_reward_manager", None),
            "steer": getattr(raw, "steer_reward_manager", None),
            "transition": getattr(raw, "transition_reward_manager", None),
            "regularization": getattr(raw, "reg_reward_manager", None),
        }
        reward_buffer_sources = [name for name, manager in reward_managers.items()
                                 if isinstance(getattr(manager, "_reward_buf", None), Tensor)]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        setup_seconds = time.perf_counter() - started
        rollout_start = time.perf_counter()
        with torch.inference_mode():
            for step in range(steps):
                action = policy.act_inference(observations)
                if not torch.isfinite(action).all():
                    raise FloatingPointError(f"Nonfinite deterministic action at step {step}")
                observations, task_reward, dones, extras = env.step(action)
                task_reward = task_reward.to(device).reshape(-1)
                failure, timeout, done = _done_flags(env, dones, extras, device)
                stats["task_reward"].add(task_reward)
                episode_steps += 1
                episode_returns += task_reward
                total_returns += task_reward
                failure_counts += failure
                timeout_counts += timeout
                completed_counts += done
                first_failure_step[failure & (first_failure_step < 0)] = step + 1
                for name in reason_names:
                    reason_counts[name] += (termination_manager.get_term(name).bool() & done).sum()
                stats["completed_episode_steps"].add(episode_steps[done])
                stats["completed_episode_seconds"].add(episode_steps[done] * step_dt)
                stats["completed_episode_return"].add(episode_returns[done])
                episode_steps[done] = 0
                episode_returns[done] = 0
                # Done states have already been reset. Exclude them from state
                # metrics while retaining their actual reward/failure event.
                valid = ~done
                phase = raw.contact_phase
                phase_masks = {name: phase[:, index] > 0.5 for index, name in enumerate(
                    ("push", "steer", "push2steer", "steer2push"))}
                phase_masks["still"] = ~(phase > 0.5).any(-1)
                for name, mask in phase_masks.items():
                    stats[f"{name}_phase_task_reward"].add(task_reward[mask])
                component_gates = {
                    "push": phase_masks["push"], "steer": phase_masks["steer"],
                    "transition": phase_masks["push2steer"] | phase_masks["steer2push"],
                    "regularization": torch.ones_like(done),
                }
                # Read the original manager's cached output, never call compute()
                # again (reward functions may mutate contact/episode state).
                for name in reward_buffer_sources:
                    cached = reward_managers[name]._reward_buf.reshape(-1)
                    stats[f"{name}_reward_contribution"].add(cached * component_gates[name])
                commands = raw.command_manager.get_command("skate")
                board_world = raw.skateboard.data.root_link_lin_vel_w.reshape(cfg.num_envs, -1)
                board_local = raw.skateboard.data.root_link_lin_vel_b.reshape(cfg.num_envs, -1)
                board_quat = raw.skateboard.data.root_link_quat_w.reshape(cfg.num_envs, 4)
                board_yaw_velocity = quat_apply_inverse(yaw_quat(board_quat), board_world)
                robot_pos = raw.robot.data.root_link_pos_w.reshape(cfg.num_envs, -1)
                push = valid & (phase[:, 0] > 0.5)
                steer = valid & (phase[:, 1] > 0.5)
                stats["push_velocity_abs_error_mps"].add((board_yaw_velocity[:, 0] - commands[:, 0]).abs()[push])
                # This is the task's original pure observation/reward formula,
                # not the command term's unused cached heading_error buffer.
                stats["steer_heading_abs_error_rad"].add(heading_error(raw, "skate").reshape(-1).abs()[steer])
                stats["board_world_speed_mps"].add(board_world.norm(dim=-1)[valid])
                for index, axis in enumerate("xyz"):
                    stats[f"board_world_v{axis}_mps"].add(board_world[:, index][valid])
                    stats[f"board_local_v{axis}_mps"].add(board_local[:, index][valid])
                stats["robot_root_height_m"].add(robot_pos[:, 2][valid])
                if writer is not None and step % stride == 0:
                    frame = env.render() if hasattr(env, "render") else raw.render()
                    if frame is None:
                        raise RuntimeError("Offscreen renderer returned no video frame")
                    writer.append_data(frame)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        rollout_seconds = time.perf_counter() - rollout_start
        measurements = {name: metric.result() for name, metric in stats.items()}
        failures = int(failure_counts.sum())
        timeouts = int(timeout_counts.sum())
        completed = int(completed_counts.sum())
        first_failure = first_failure_step.cpu().tolist()
        failure_free_envs = sum(step < 0 for step in first_failure)
        survival_20s = (sum(step < 0 or step * step_dt > 20.0 for step in first_failure)
                        if steps * step_dt >= 20.0 else None)
        config_json = _json_config(cfg)
        report = {
            "schema_version": 2,
            "quality_evaluation": not cfg.presentation_mode,
            "evaluation_mode": "presentation" if cfg.presentation_mode else "original_training_task",
            "configuration": config_json,
            "invocation": list(sys.argv),
            "environment_config": environment_config,
            "agent_config": agent_config,
            "comparison_config_sha256": config_fingerprint,
            "code": _git_revision(),
            "checkpoint": {"file": str(checkpoint_path), "sha256": _sha256(checkpoint_path),
                           "size_bytes": checkpoint_path.stat().st_size,
                           "iteration": checkpoint.get("iter"),
                           "backend": "supermimic" if "supermimic" in checkpoint else "rsl_ppo_or_amp"},
            "runtime": {"device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                        "torch_version": torch.__version__, "step_dt": step_dt, "control_steps": steps,
                        "simulated_seconds_per_env": steps * step_dt,
                        "total_real_transitions": steps * cfg.num_envs,
                        "setup_wall_seconds": setup_seconds, "rollout_wall_seconds": rollout_seconds,
                        "observation_shapes": {key: list(value.shape) for key, value in observations.items()},
                        "video_file": str(video_path) if video_path else None,
                        "video_fps": 1 / (step_dt * stride) if cfg.video else None},
            "episodes": {"completed": completed, "physical_failures": failures, "timeouts": timeouts,
                         "physical_failures_per_env": failure_counts.cpu().tolist(),
                         "timeouts_per_env": timeout_counts.cpu().tolist(),
                         "envs_with_failure": int((failure_counts > 0).sum()),
                         "failure_free_entire_horizon_envs": failure_free_envs,
                         "failure_free_entire_horizon_env_fraction": failure_free_envs / cfg.num_envs,
                         "survival_20s_envs": survival_20s,
                         "survival_20s_env_fraction": survival_20s / cfg.num_envs if survival_20s is not None else None,
                         "first_physical_failure_seconds_per_env": [step * step_dt if step >= 0 else None for step in first_failure],
                         "termination_reasons": {name: {"events": int(count), "time_out": reason_is_timeout[name]}
                                                 for name, count in reason_counts.items()},
                         "unfinished_episode_steps": episode_steps.cpu().tolist(),
                         "unfinished_episode_task_returns": episode_returns.cpu().tolist(),
                         "total_task_return_per_env": total_returns.cpu().tolist()},
            "measurements": measurements,
            "metric_definitions": {
                "task_reward": "Unmodified dt-scaled environment task reward; no AMP discriminator reward.",
                "push_velocity_abs_error_mps": "abs(quat_apply_inverse(yaw_quat(board.root_link_quat_w), board.root_link_lin_vel_w)[:,0] - command[:,0]), matching original command metrics; only push phase and non-reset post-step states.",
                "steer_heading_abs_error_rad": "Original mdp.heading_error(..., 'skate').abs(), only original steer phase and non-reset post-step states.",
                "board_local_velocity": "Full board quaternion frame (root_link_lin_vel_b), distinct from the yaw-only push tracking metric.",
                "phase_task_rewards": "Original total task reward conditioned on the original contact_phase; includes terminal-transition rewards, excludes AMP.",
                "reward_contributions": {"definition": "Original cached RewardManager._reward_buf times its original phase gate; averaged over all transitions, including zero contribution outside its phase.",
                                         "available_cached_managers": reward_buffer_sources},
                "survival": "Fraction of environment slots with no physical failure in the first 20 seconds (or throughout the full horizon); timeouts do not count as physical failure. 20s fields are null for shorter evaluations.",
                "termination_reasons": "Counts read from original TerminationManager.get_term caches on done transitions; multiple reasons can co-occur, so their sum may exceed completed episodes.",
                "state_statistics": "Non-reset post-step states; terminal physical states are unavailable after task autoreset.",
                "completed_episodes": "Failure and timeout events are counted separately; both may occur on one transition.",
                "unfinished_episodes": "Right-censored at the fixed evaluation horizon; not counted as completed episodes.",
                "reproducibility": "Same task configuration and seed; policy-dependent resets can change subsequent random draws. CUDA physics may not be bitwise deterministic.",
            },
        }
        report["finite_measurements"] = all(item["nonfinite_samples"] == 0 for item in measurements.values())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(output_path)
        print(json.dumps({"report": str(output_path), "quality_evaluation": report["quality_evaluation"],
                          "task_reward": measurements["task_reward"]["mean"],
                          "physical_failures": failures, "timeouts": timeouts,
                          "push_velocity_mae_mps": measurements["push_velocity_abs_error_mps"]["mean"],
                          "steer_heading_mae_rad": measurements["steer_heading_abs_error_rad"]["mean"],
                          "rollout_wall_seconds": rollout_seconds}, indent=2), flush=True)
        return report
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()


def main():
    import tyro
    run_evaluation(tyro.cli(EvaluationConfig, config=(tyro.conf.AvoidSubcommands,)))


if __name__ == "__main__":
    main()
