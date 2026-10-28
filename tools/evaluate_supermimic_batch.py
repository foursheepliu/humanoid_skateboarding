"""Multi-initial-state real-physics diagnostics for original HUSKY policies.

Each checkpoint runs in its own process. Match seed/num-envs/steps/noise and
compare initial_conditions_sha256 before comparing metrics. This is a play-mode
diagnostic with failure terminations disabled, not a formal task success rate.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("WANDB_MODE", "disabled")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value):
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": list(value.shape), "dtype": value.dtype.str}, sort_keys=True).encode())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def optional_ratio(total, count):
    return float(total / count) if count > 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Experiment root; writes only data/")
    parser.add_argument("--label", default="batch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--training-noise", action="store_true",
                        help="Restore training observation corruption and interval pushes; actor stays deterministic")
    parser.add_argument("--trace", action="store_true", help="Also save raw per-step diagnostics and qpos in NPZ")
    args = parser.parse_args()
    if min(args.num_envs, args.steps) < 1:
        parser.error("num-envs and steps must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        parser.error("label must use ASCII letters, digits, underscore or hyphen")
    args.out = args.out.resolve()
    args.policy = args.policy.resolve()
    data_dir = args.out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    report_path = data_dir / f"{args.label}_batch_evaluation.json"
    trace_path = data_dir / f"{args.label}_batch_trace.npz"
    if report_path.exists() or (args.trace and trace_path.exists()):
        raise FileExistsError("Use a new label; evaluation evidence is not overwritten")

    # Initialize base task discovery before importing the custom environment.
    import mjlab.tasks  # noqa: F401
    import numpy as np
    import torch
    from tensordict import TensorDict
    from rsl_rl.modules import ActorCritic
    from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat, wrap_to_pi
    from mjlab.utils.torch import configure_torch_backends
    from mjlab_husky.envs import G1SkaterManagerBasedRlEnv
    from mjlab_husky.tasks.registry import load_env_cfg, load_rl_cfg
    from mjlab_husky.supermimic.env_adapter import CoupledStateAdapter

    configure_torch_backends()
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    task = "Mjlab-Skater-Flat-Unitree-G1"
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.commands["skate"].ranges.lin_vel_x = (0.0, 1.5)
    cfg.commands["skate"].ranges.heading = (-math.pi / 4, math.pi / 4)
    if args.training_noise:
        cfg.observations["policy"].enable_corruption = True
        cfg.events["push_robot"] = load_env_cfg(task, play=False).events["push_robot"]
    env = G1SkaterManagerBasedRlEnv(cfg, device=args.device)
    adapter = None
    try:
        obs, _ = env.reset()
        agent = load_rl_cfg(task)
        pcfg = asdict(agent.policy)
        pcfg.pop("class_name")
        policy = ActorCritic(TensorDict(obs, batch_size=[args.num_envs]), agent.obs_groups,
                            env.action_manager.total_action_dim, **pcfg).to(args.device)
        # Full learning snapshots can contain a host replay; do not map that
        # replay to CUDA merely to extract compatible ActorCritic weights.
        saved = torch.load(args.policy, map_location="cpu", weights_only=False)
        policy.load_state_dict(saved["model_state_dict"])
        checkpoint_iteration = saved.get("iter", saved.get("iteration"))
        checkpoint_format = saved.get("format", saved.get("algorithm", "legacy_husky"))
        del saved
        policy.eval().requires_grad_(False)
        adapter = CoupledStateAdapter(env)

        def cpu_array(value):
            if hasattr(value, "_tensor"):
                value = value._tensor
            return value.detach().cpu().contiguous().numpy().copy()

        def normalizer_identity(module):
            return canonical_sha256({name: array_sha256(cpu_array(value))
                                     for name, value in module.state_dict().items()})

        normalizers_before = {
            "actor": normalizer_identity(policy.actor_obs_normalizer),
            "critic": normalizer_identity(policy.critic_obs_normalizer),
        }
        if policy.actor_obs_normalizer.training or policy.critic_obs_normalizer.training:
            raise RuntimeError("Evaluation normalizers must be frozen in eval mode")
        rollout_rng = {"torch_cpu_sha256": array_sha256(cpu_array(torch.get_rng_state()))}
        if torch.device(env.device).type == "cuda":
            rollout_rng["torch_cuda_sha256"] = array_sha256(cpu_array(torch.cuda.get_rng_state(env.device)))
        qpos0 = cpu_array(env.sim.data.qpos)
        qvel0 = cpu_array(env.sim.data.qvel)
        context0 = cpu_array(adapter.context())
        command0 = cpu_array(env.command_manager.get_command("skate"))
        target0 = cpu_array(env.get_heading_target_w("skate"))
        robot_pose0 = np.concatenate((cpu_array(env.robot.data.root_link_pos_w),
                                      cpu_array(env.robot.data.root_link_quat_w)), axis=-1)
        board_pose0 = np.concatenate((cpu_array(env.skateboard.data.root_link_pos_w),
                                      cpu_array(env.skateboard.data.root_link_quat_w)), axis=-1)
        raw_obs0 = cpu_array(obs["policy"])
        initial = []
        for index in range(args.num_envs):
            item = {"env_index": index, "command": command0[index].tolist(),
                    "target_heading_world_rad": float(target0[index]),
                    "robot_root_pose_xyz_wxyz": robot_pose0[index].tolist(),
                    "board_root_pose_xyz_wxyz": board_pose0[index].tolist(),
                    "qpos": qpos0[index].tolist(), "qvel": qvel0[index].tolist(),
                    "physical_context_sha256": array_sha256(context0[index]),
                    "policy_observation_sha256": array_sha256(raw_obs0[index])}
            item["initial_state_sha256"] = canonical_sha256(item)
            initial.append(item)
        protocol = {"task": task, "seed": args.seed, "num_envs": args.num_envs,
                    "steps_requested": args.steps, "control_dt_seconds": env.step_dt,
                    "speed_command_range_mps": [0.0, 1.5],
                    "relative_heading_range_rad": [-math.pi / 4, math.pi / 4],
                    "policy": "deterministic mean", "training_noise": args.training_noise,
                    "observation_corruption": bool(cfg.observations["policy"].enable_corruption),
                    "interval_robot_push": "push_robot" in cfg.events,
                    "episode_time_limit_seconds": cfg.episode_length_s,
                    "fall_terminations_enabled": False,
                    "raw_board_contact_threshold_newtons": 5.0,
                    "tilt_threshold_degrees": 70.0}
        initial_identity = canonical_sha256({"protocol": protocol, "environments": initial})
        device = env.device
        n = args.num_envs
        min_height = env.robot.data.root_link_pos_w[:, 2].clone()
        max_tilt = torch.zeros(n, device=device)
        fall_frames = torch.zeros(n, device=device, dtype=torch.long)
        first_fall = torch.full((n,), -1, device=device, dtype=torch.long)
        reset_counts = torch.zeros(n, device=device, dtype=torch.long)
        nonfinite_frames = torch.zeros(n, device=device, dtype=torch.long)
        push_frames = torch.zeros(n, device=device, dtype=torch.long)
        steer_frames = torch.zeros(n, device=device, dtype=torch.long)
        velocity_error_sum = torch.zeros(n, device=device, dtype=torch.float64)
        heading_error_sum = torch.zeros(n, device=device, dtype=torch.float64)
        bilateral_frames = torch.zeros(n, device=device, dtype=torch.long)
        commands = torch.empty((args.steps, n, 2), device=device)
        trace = {key: [] for key in ("qpos", "root_height", "root_tilt_degrees", "phase",
                                      "commands", "board_forward_speed", "heading_error",
                                      "left_board_force_norm", "right_board_force_norm", "resets")} if args.trace else None

        def board_force_norm(sensor_name):
            force = env.scene.sensors[sensor_name].data.force
            if force is None:
                raise RuntimeError("Raw board contact forces unavailable: " + sensor_name)
            return force.reshape(n, -1, 3).norm(dim=-1).amax(dim=-1)

        completed = 0
        failure = None
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                for step in range(args.steps):
                    action = policy.act_inference(TensorDict(obs, batch_size=[n]))
                    if not bool(torch.isfinite(action).all()):
                        raise FloatingPointError("Nonfinite policy action at step " + str(step))
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done = (terminated | truncated).reshape(-1)
                    command = env.command_manager.get_command("skate")
                    commands[step] = command
                    gravity = quat_apply_inverse(env.robot.data.root_link_quat_w,
                                                 action.new_tensor([0.0, 0.0, -1.0]).expand(n, 3))
                    tilt = torch.rad2deg(torch.acos((-gravity[:, 2]).clamp(-1, 1)))
                    height = env.robot.data.root_link_pos_w[:, 2]
                    velocity = quat_apply_inverse(yaw_quat(env.skateboard.data.root_link_quat_w),
                                                  env.skateboard.data.root_link_lin_vel_w)[:, 0]
                    yaw_error = wrap_to_pi(env.get_heading_target_w("skate") - env.skateboard.data.heading_w)
                    left_force = board_force_norm("left_feet_board_contact")
                    right_force = board_force_norm("right_feet_board_contact")
                    finite = (torch.isfinite(height) & torch.isfinite(tilt) & torch.isfinite(velocity)
                              & torch.isfinite(yaw_error) & torch.isfinite(left_force) & torch.isfinite(right_force))
                    nonfinite_frames += (~finite).long()
                    min_height = torch.minimum(min_height, torch.where(finite, height, min_height))
                    max_tilt = torch.maximum(max_tilt, torch.where(finite, tilt, max_tilt))
                    fell = finite & (tilt > 70.0)
                    first_fall = torch.where(fell & (first_fall < 0), step + 1, first_fall)
                    fall_frames += fell.long()
                    reset_counts += done.long()
                    pushing = (env.contact_phase[:, 0] > 0.5) & finite
                    steering = (env.contact_phase[:, 1] > 0.5) & finite
                    push_frames += pushing.long()
                    steer_frames += steering.long()
                    velocity_error_sum += torch.where(pushing, (velocity - command[:, 0]).abs(), 0).double()
                    heading_error_sum += torch.where(steering, yaw_error.abs(), 0).double()
                    bilateral_frames += (steering & (left_force > 5.0) & (right_force > 5.0)).long()
                    if trace is not None:
                        values = {"qpos": env.sim.data.qpos, "root_height": height, "root_tilt_degrees": tilt,
                                  "phase": env.contact_phase, "commands": command,
                                  "board_forward_speed": velocity, "heading_error": yaw_error,
                                  "left_board_force_norm": left_force, "right_board_force_norm": right_force,
                                  "resets": done}
                        for name, value in values.items():
                            trace[name].append(cpu_array(value))
                    completed = step + 1
        except BaseException as error:
            failure = error
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        host = {name: cpu_array(value) for name, value in {
            "minimum_root_height_m": min_height, "maximum_root_tilt_degrees": max_tilt,
            "fell_over_frames_above_70deg": fall_frames, "first_fell_over_step": first_fall,
            "resets": reset_counts, "nonfinite_frames": nonfinite_frames,
            "pushing_frames": push_frames, "steering_frames": steer_frames,
            "velocity_error_sum": velocity_error_sum, "heading_error_sum": heading_error_sum,
            "steering_bilateral_contact_frames": bilateral_frames}.items()}
        command_array = cpu_array(commands[:completed])
        per_env = []
        for index in range(n):
            item = {"env_index": index,
                    "minimum_root_height_m": float(host["minimum_root_height_m"][index]),
                    "maximum_root_tilt_degrees": float(host["maximum_root_tilt_degrees"][index]),
                    "fell_over_frames_above_70deg": int(host["fell_over_frames_above_70deg"][index]),
                    "ever_fell_over_70deg": bool(host["fell_over_frames_above_70deg"][index] > 0),
                    "first_fell_over_step": int(host["first_fell_over_step"][index]),
                    "resets": int(host["resets"][index]), "nonfinite_frames": int(host["nonfinite_frames"][index]),
                    "pushing_frames": int(host["pushing_frames"][index]),
                    "steering_frames": int(host["steering_frames"][index]),
                    "pushing_velocity_mae_mps": optional_ratio(host["velocity_error_sum"][index], host["pushing_frames"][index]),
                    "steering_heading_mae_rad": optional_ratio(host["heading_error_sum"][index], host["steering_frames"][index]),
                    "steering_both_feet_board_fraction": optional_ratio(host["steering_bilateral_contact_frames"][index], host["steering_frames"][index]),
                    "raw_command_sequence_sha256": array_sha256(command_array[:, index])}
            per_env.append(item)
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip()
        except (OSError, subprocess.SubprocessError):
            commit = None
        normalizers_after = {
            "actor": normalizer_identity(policy.actor_obs_normalizer),
            "critic": normalizer_identity(policy.critic_obs_normalizer),
        }
        if normalizers_before != normalizers_after and failure is None:
            failure = RuntimeError("Observation normalization changed during evaluation")
        report = {"status": "failed" if failure else "completed", "error": repr(failure) if failure else None,
                  "label": args.label, "policy_file": args.policy.name, "policy_sha256": file_sha256(args.policy),
                  "checkpoint_iteration": checkpoint_iteration, "checkpoint_format": checkpoint_format,
                  "ppo_updates": 0, "protocol": protocol, "steps_completed": completed,
                  "evaluation_seconds": elapsed, "initial_conditions_sha256": initial_identity,
                  "rollout_rng_identity": rollout_rng,
                  "normalizers_eval_mode": True, "normalizers_before": normalizers_before,
                  "normalizers_after": normalizers_after, "normalizers_unchanged": normalizers_before == normalizers_after,
                  "initial_conditions": initial, "raw_command_sequence_sha256": array_sha256(command_array),
                  "physical_context_layout": {k: [v.start, v.stop] for k, v in adapter.context_layout.items()},
                  "repo_commit": commit, "evaluator_sha256": file_sha256(__file__),
                  "aggregate": {
                      "num_envs": n, "num_ever_fell_over_70deg": sum(x["ever_fell_over_70deg"] for x in per_env),
                      "diagnostic_ever_fell_fraction": sum(x["ever_fell_over_70deg"] for x in per_env) / n,
                      "minimum_root_height_m": min(x["minimum_root_height_m"] for x in per_env),
                      "maximum_root_tilt_degrees": max(x["maximum_root_tilt_degrees"] for x in per_env),
                      "total_resets": int(host["resets"].sum()), "nonfinite_envs": sum(x["nonfinite_frames"] > 0 for x in per_env),
                      "pooled_pushing_velocity_mae_mps": optional_ratio(host["velocity_error_sum"].sum(), host["pushing_frames"].sum()),
                      "pooled_steering_heading_mae_rad": optional_ratio(host["heading_error_sum"].sum(), host["steering_frames"].sum()),
                      "pooled_steering_both_feet_board_fraction": optional_ratio(host["steering_bilateral_contact_frames"].sum(), host["steering_frames"].sum())},
                  "per_env": per_env,
                  "limits": ["Play mode disables physical failure terminations; zero resets is not a success criterion.",
                             "Ever-fell is an independent root-tilt diagnostic above70degrees, not a formal task success label.",
                             "MAE/contact statistics use only their active phase; no active phase gives null, not zero error.",
                             "Each checkpoint is evaluated in an independent process; compare protocol and initial-condition identities.",
                             "Physical context captures the adapter's represented model parameters, not every simulator internal state."]}
        if trace is not None:
            np.savez_compressed(trace_path, **{name: np.stack(value) for name, value in trace.items() if value},
                                initial_conditions_sha256=initial_identity)
            report["trace_file"], report["trace_sha256"] = trace_path.name, file_sha256(trace_path)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"report_file": str(report_path), "initial_conditions_sha256": initial_identity,
                          "aggregate": report["aggregate"]}, indent=2))
        if failure is not None:
            raise failure
    finally:
        if adapter is not None:
            adapter.close()
        env.close()


if __name__ == "__main__":
    main()
