"""Fresh held-out, recorded-action world-model rollout diagnostic.

Policy, model, AMP, normalizers and optimizers remain frozen. This measures
physical autoregression conditional on recorded actions/exogenous samples;
it is neither policy quality nor counterfactual-action model accuracy.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time


SCALES = {"position": .3, "orientation": .4, "joint_position": .3,
          "linear_velocity": 1., "angular_velocity": math.pi, "joint_velocity": 5.}


def _hash_tree(value) -> str:
    """Stable state fingerprint including tensor bytes, without torch.save IDs."""
    import numpy as np
    import torch
    digest = hashlib.sha256()

    def add(item):
        if isinstance(item, torch.Tensor):
            digest.update(f"tensor:{item.dtype}:{tuple(item.shape)}".encode())
            digest.update(item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(f"numpy:{item.dtype}:{item.shape}".encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                digest.update(repr(key).encode())
                add(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            for child in item:
                add(child)
        else:
            digest.update(repr(item).encode())
    add(value)
    return digest.hexdigest()


def _frozen_hashes(runner) -> dict:
    return {"policy": _hash_tree(runner.policy.state_dict()),
            "world_model_and_norms": _hash_tree(runner.world_model.state_dict()),
            "amp_and_norms_and_optimizer": _hash_tree(runner.amp.state_dict()),
            "actor_optimizer": _hash_tree(runner.actor_optimizer.state_dict()),
            "critic_optimizer": _hash_tree(runner.critic_optimizer.state_dict()),
            "model_optimizer": _hash_tree(runner.world_model.optimizer.state_dict())}


def _physical_errors(adapter, predicted, target):
    import torch
    result, normalized = {}, {}
    for entity in ("robot", "skateboard"):
        get = lambda state, name: adapter.layout.get(state, f"{entity}.{name}")
        position = get(predicted, "root_link_pos_w") - get(target, "root_link_pos_w")
        joint = get(predicted, "joint_pos") - get(target, "joint_pos")
        velocity = get(predicted, "joint_vel") - get(target, "joint_vel")
        qp, qt = get(predicted, "root_link_quat_w"), get(target, "root_link_quat_w")
        qp_norm, qt_norm = qp.norm(dim=-1, keepdim=True), qt.norm(dim=-1, keepdim=True)
        qp = qp / qp_norm.clamp_min(1e-8)
        qt = qt / qt_norm.clamp_min(1e-8)
        # Quaternion sign is a gauge. Squared angle/3 equals mean squared
        # rotation-vector error, matching the reference's 3D orientation group.
        relative_xyz = (qt[..., :1] * qp[..., 1:] - qp[..., :1] * qt[..., 1:]
                        - torch.linalg.cross(qt[..., 1:], qp[..., 1:], dim=-1))
        relative_w = (qp * qt).sum(-1)
        angle = 2 * torch.atan2(relative_xyz.norm(dim=-1), relative_w.abs())
        angle = torch.where((qp_norm[..., 0] < 1e-8) | (qt_norm[..., 0] < 1e-8), math.pi, angle)
        groups = {"position": position, "joint_position": joint,
                  "joint_velocity": velocity,
                  "linear_velocity": get(predicted, "root_link_lin_vel_w") - get(target, "root_link_lin_vel_w"),
                  "angular_velocity": get(predicted, "root_link_ang_vel_w") - get(target, "root_link_ang_vel_w")}
        for name, error in groups.items():
            normalized[f"{entity}/{name}"] = (error / SCALES[name]).square().mean(-1)
        normalized[f"{entity}/orientation"] = angle.square() / (3 * SCALES["orientation"] ** 2)
        label = "board" if entity == "skateboard" else entity
        result[f"{label}_position_mae_m"] = position.abs().mean(-1)
        if entity == "robot":
            result["robot_joint_position_mae_rad"] = joint.abs().mean(-1)
            result["robot_joint_velocity_mae_rad_s"] = velocity.abs().mean(-1)
    result["normalized_physical_group_mse"] = torch.stack(list(normalized.values()), -1).mean(-1)
    return result, normalized


def evaluate_horizon(runner, records: dict, horizon: int) -> dict:
    import torch
    steps, envs = records["reward"].shape
    starts = steps - horizon + 1
    empty_metrics = {name: None for name in (
        "robot_position_mae_m", "board_position_mae_m", "robot_joint_position_mae_rad",
        "robot_joint_velocity_mae_rad_s", "normalized_physical_group_mse",
        "normalized_physical_group_rmse", "task_reward_mae", "termination_brier")}
    if starts <= 0:
        return {"available": False, "candidate_windows": 0, "physical_windows": 0,
                "reason": "fewer recorded steps than horizon", "metrics": empty_metrics,
                "normalized_group_mse": None, "terminal_inclusive_prefix": None}

    def windows(name):
        value = records[name]
        return torch.stack([value[start:start + horizon] for start in range(starts)]) \
            .transpose(1, 2).reshape(starts * envs, horizon, *value.shape[2:])

    actions, targets, rewards = windows("action"), windows("next_state"), windows("reward")
    terminated, truncated = windows("terminated"), windows("truncated")
    done = terminated | truncated
    full_alive = ~done.any(1)
    state = records["state"][:starts].reshape(starts * envs, -1).clone()
    alive = torch.ones(len(state), device=state.device, dtype=torch.bool)
    prediction_rewards, briers, prefix_masks = [], [], []
    finite_prefix = True
    for step in range(horizon):
        prediction = runner.world_model(state, actions[:, step])
        reward_error = (prediction.reward - rewards[:, step]).abs()
        brier = (prediction.termination_logit.sigmoid() - terminated[:, step].float()).square()
        prefix_masks.append(alive.clone())
        prediction_rewards.append(reward_error)
        briers.append(brier)
        finite_prefix &= bool(torch.isfinite(prediction.reward[alive]).all()
                              & torch.isfinite(prediction.termination_logit[alive]).all())
        # Only commands and observed noise are conditioned on the held-out
        # target. Its physical state NEVER replaces a model prediction.
        projected = runner.adapter.project_next(
            state, prediction.next_state, actions[:, step], exogenous={"next_state": targets[:, step]})
        active_next = alive & ~done[:, step]
        finite_prefix &= bool(torch.isfinite(projected[active_next]).all())
        state = torch.where(active_next[:, None], projected, state)
        alive = active_next
    prefix = torch.stack(prefix_masks, 1)
    reward_error, brier = torch.stack(prediction_rewards, 1), torch.stack(briers, 1)

    def mean(value):
        return float(value.mean()) if value.numel() and torch.isfinite(value).all() else None

    endpoint, groups = _physical_errors(runner.adapter, state[full_alive], targets[full_alive, -1])
    metrics = {name: mean(value) for name, value in endpoint.items()}
    mse = metrics["normalized_physical_group_mse"]
    metrics["normalized_physical_group_rmse"] = math.sqrt(mse) if mse is not None else None
    metrics.update(task_reward_mae=mean(reward_error[full_alive]), termination_brier=mean(brier[full_alive]))
    return {"available": bool(full_alive.any()), "candidate_windows": starts * envs,
            "physical_windows": int(full_alive.sum()), "windowed_transitions": int(full_alive.sum()) * horizon,
            "reason": None if full_alive.any() else "no complete non-reset prefix survives this horizon",
            "finite_prefix_predictions": finite_prefix, "metrics": metrics,
            "normalized_group_mse": {name: mean(value) for name, value in groups.items()} if full_alive.any() else None,
            "terminal_inclusive_prefix": {
                "windowed_transitions": int(prefix.sum()), "physical_failures": int(terminated[prefix].sum()),
                "timeouts": int(truncated[prefix].sum()),
                "task_reward_mae": mean(reward_error[prefix]), "termination_brier": mean(brier[prefix]),
                "definition": "Includes a terminal last step, excludes every later step after that reset; overlapping windows repeat samples."}}


def run(args) -> dict:
    import numpy as np
    import torch
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("WANDB_MODE", "disabled")
    import mjlab_husky.tasks  # noqa: F401
    from mjlab.utils.torch import configure_torch_backends
    from mjlab_husky.envs import G1SkaterManagerBasedRlEnv
    from mjlab_husky.rl import RslRlVecEnvWrapper
    from mjlab_husky.supermimic.config import SuperMimicConfig
    from mjlab_husky.supermimic.runner import SuperMimicRunner
    from mjlab_husky.supermimic.adapter import roll_history
    from mjlab_husky.tasks.registry import load_env_cfg, load_rl_cfg
    from mjlab_husky.scripts.evaluate_checkpoint import _json_config, _sha256, _git_revision

    checkpoint_path = Path(args.checkpoint_file).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output == checkpoint_path:
        raise ValueError("Report must not overwrite the checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload = checkpoint.get("supermimic")
    if payload is None:
        raise ValueError("A complete SuperMimic checkpoint (not policy-only PPO weights) is required")
    settings = SuperMimicConfig.from_checkpoint_dict(payload["config"])
    configure_torch_backends()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    env_cfg, agent_cfg = load_env_cfg(args.task, play=False), load_rl_cfg(args.task)
    env_cfg.scene.num_envs, env_cfg.seed = args.num_envs, args.seed
    agent = asdict(agent_cfg)
    agent["seed"], agent["supermimic"] = args.seed, asdict(settings)
    expert = payload["amp"]["expert_config"]
    agent.update(amp_num_frames=expert["num_frames"], amp_num_obs=payload["amp"]["observation_dim"],
                 amp_num_preload_transitions=expert["num_preload_transitions"], amp_motion_files=expert["motion_files"])
    env = None
    started = time.perf_counter()
    try:
        env = RslRlVecEnvWrapper(G1SkaterManagerBasedRlEnv(cfg=env_cfg, device=str(device)),
                                 clip_actions=agent_cfg.clip_actions)
        runner = SuperMimicRunner(env, agent, log_dir=None, device=device)
        # Saved training replay is irrelevant to fresh held-out evaluation, and
        # may have a different training num_envs. Never edit the source artifact.
        with tempfile.TemporaryDirectory(prefix="husky-wm-probe-") as temporary:
            load_path = checkpoint_path
            if "replay" in payload:
                copy = {**checkpoint, "supermimic": {key: value for key, value in payload.items() if key != "replay"}}
                load_path = Path(temporary) / "weights-without-training-replay.pt"
                torch.save(copy, load_path)
            runner.load(str(load_path), load_optimizer=False, map_location=device)
        runner.eval_mode(); runner.amp.discriminator.eval()
        for module in (runner.policy, runner.world_model, runner.amp.discriminator):
            module.requires_grad_(False)
        before = _frozen_hashes(runner)
        # The AMP preload is restored exactly, but its random draws must not
        # determine fresh evaluation reset/noise sequences.
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        observations, _ = env.reset()
        observations = {name: value.detach().clone() for name, value in observations.items()}
        current_amp = env.get_amp_observations().detach().clone()
        history = current_amp.new_zeros((args.num_envs, runner.amp_num_frames, runner.amp_observation_dim))
        history = roll_history(history, current_amp)
        collected = {name: [] for name in ("state", "action", "next_state", "reward", "terminated", "truncated")}
        with torch.inference_mode():
            for _ in range(args.steps):
                state = runner.adapter.capture(observations, history)
                action, _ = runner._action(observations, stochastic=False)
                # Autoreset clears ActionManager.action on terminal rows. Save
                # the executed input BEFORE step so terminal labels retain the
                # real action rather than the cleared post-reset zero buffer.
                executed = action.to(env.device).detach().clone()
                if env.clip_actions is not None:
                    executed = executed.clamp(-env.clip_actions, env.clip_actions)
                next_obs, reward, dones, extras = env.step(executed)
                next_obs = {name: value.detach().clone() for name, value in next_obs.items()}
                terminated, truncated, done = runner._split_done(dones, extras)
                next_history = roll_history(history, env.get_amp_observations().detach().clone())
                next_state = runner.adapter.capture(next_obs, next_history)
                values = (state, executed.to(device).clone(), next_state,
                          reward.detach().reshape(-1).clone(), terminated.clone(), truncated.clone())
                for name, value in zip(collected, values):
                    if not torch.isfinite(value).all():
                        raise FloatingPointError(f"Nonfinite fresh recorded {name}")
                    collected[name].append(value)
                history = next_history
                history[done] = 0
                observations = next_obs
            records = {name: torch.stack(value) for name, value in collected.items()}
            horizons = {str(h): evaluate_horizon(runner, records, h) for h in (1, 8, 32)}
        after = _frozen_hashes(runner)
        source_root = Path(__file__).resolve().parents[1]
        report = {"schema_version": 1, "scope": "fresh held-out recorded-action physical autoregression",
                  "not_policy_quality_or_counterfactual_accuracy": True,
                  "protocol": {"task": args.task, "seed": args.seed, "num_envs": args.num_envs,
                               "real_steps": args.steps, "step_dt": runner.adapter.step_dt,
                               "deterministic_policy": True, "original_training_task": True,
                               "conditioned_only_on_gt_commands_and_sensor_noise": True,
                               "projection_execution": "eager; saved compile setting retained but not invoked"},
                  "checkpoint": {"path": str(checkpoint_path), "sha256": _sha256(checkpoint_path),
                                 "iteration": checkpoint.get("iter"), "saved_settings": _json_config(settings),
                                 "restored_amp_expert_config": _json_config(expert), "training_replay_ignored": "replay" in payload},
                  "code": {**_git_revision(), "script_sha256": _sha256(Path(__file__)),
                           "source_sha256": {name: _sha256(source_root / "src/mjlab_husky/supermimic" / name)
                                             for name in ("adapter.py", "models.py", "runner.py", "amp.py")}},
                  "environment_config": _json_config(env_cfg), "adapter": runner.adapter.metadata(),
                  "record_shapes": {name: list(value.shape) for name, value in records.items()},
                  "collection_events": {"physical_failures": int(records["terminated"].sum()),
                                        "timeouts": int(records["truncated"].sum())},
                  "horizons": horizons,
                  "metric_definition": {"physical_metrics": "Endpoint component MAE; complete H-step non-reset windows only.",
                                        "reward_and_main_brier": "Mean across H recorded steps of complete non-reset windows; main Brier therefore has negative labels only.",
                                        "normalized_physical_groups": "Equal-weight mean of 12 groups: six groups each for robot/board, using fixed reference units (not D2730 mean or learned delta variance); RMSE is sqrt of the final mean MSE.",
                                        "normalization_scales": SCALES,
                                        "no_available_horizon": "All main metrics are null, never zero.",
                                        "terminal_physics": "Autoreset terminal physical next states are unavailable and excluded; terminal rewards/failure labels remain in separate prefix diagnostics.",
                                        "exogenous_conditioning": "Ground-truth next state supplies only command/timer fields and detached observed sensor-noise residuals. Physical states autoregress from the world model.",
                                        "coverage": "Fresh reset collection of 32 steps is 0.64 seconds at dt=.02 and may not cover later skating phases."},
                  "passed_definition": "Frozen-state hashes unchanged and valid-prefix predictions finite; no accuracy or policy-quality threshold is imposed.",
                  "frozen_state": {"passed": before == after, "before": before, "after": after},
                  "wall_seconds": time.perf_counter() - started}
        report["passed"] = before == after and all(row.get("finite_prefix_predictions", True) for row in horizons.values())
        if args.record_file:
            record_path = Path(args.record_file).expanduser().resolve()
            if record_path in (checkpoint_path, output):
                raise ValueError("Recorded transitions must not overwrite checkpoint/report")
            record_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"records": {name: value.cpu() for name, value in records.items()},
                        "checkpoint_sha256": report["checkpoint"]["sha256"], "protocol": report["protocol"]}, record_path)
            report["record_file"] = {"path": str(record_path), "sha256": _sha256(record_path)}
        return report
    finally:
        if env is not None:
            env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-file", required=True)
    parser.add_argument("--task", default="Mjlab-Skater-Flat-Unitree-G1")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="logs/world_model_rollout.json")
    parser.add_argument("--record-file", default=None)
    args = parser.parse_args()
    if min(args.num_envs, args.steps) < 1:
        parser.error("num-envs and steps must be positive")
    try:
        report = run(args)
    except Exception as error:
        report = {"passed": False, "error": type(error).__name__, "message": str(error)}
    output = Path(args.output).expanduser().resolve()
    if output == Path(args.checkpoint_file).expanduser().resolve():
        raise ValueError("Report must not overwrite checkpoint")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), "passed": report["passed"],
                      "horizons": report.get("horizons"), "error": report.get("error")}, indent=2), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
