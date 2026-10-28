"""Audit a frozen model on one reusable real-action, 32-step audit corpus.

No optimizer is created. Fresh collection retains the HUSKY training commands,
randomization and disturbances. --batch reuses exactly the same real data.
"""
import argparse
import ast
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
from types import MethodType, SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("WANDB_MODE", "disabled")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def decode_slice(value):
    return slice(*value) if isinstance(value, (list, tuple)) else value


def canonical_metadata(value):
    """Compare saved layouts by value, independent of tensor device/storage."""
    import torch
    if isinstance(value, torch.Tensor):
        return canonical_metadata(value.detach().cpu().tolist())
    if isinstance(value, slice):
        return ("slice", value.start, value.stop, value.step)
    if isinstance(value, dict):
        return {key: canonical_metadata(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(canonical_metadata(item) for item in value)
    return value


def load_joint_metadata(snapshot, argument):
    if argument is None:
        return None
    if argument == "snapshot":
        metadata = snapshot.get("joint_projection_metadata")
        if metadata is None:
            raise ValueError("Explicit snapshot joint projection requested but metadata is absent")
        return metadata
    return json.loads(Path(argument).read_text())


def projection_adapter(snapshot, collection, joint_metadata=None):
    from mjlab_husky.supermimic.env_adapter import CoupledStateAdapter, joint_projection_spec_sha256
    adapter = CoupledStateAdapter.__new__(CoupledStateAdapter)
    adapter.layout = {name: decode_slice(sl) for name, sl in snapshot["layout"].items()}
    adapter.history_terms = [dict(term, slice=decode_slice(term["slice"]))
                             for term in snapshot["history_terms"]]
    adapter.critic_terms = [dict(term, slice=decode_slice(term["slice"]))
                            for term in snapshot["critic_terms"]]
    adapter.qpos_quaternion_slices = [decode_slice(sl) for sl in snapshot["qpos_quaternion_slices"]]
    adapter.board_quaternion_slice = decode_slice(collection.get("board_quaternion_slice"))
    settings = dict(collection.get("command_config", {}))
    settings["ranges"] = SimpleNamespace(**settings.get("ranges", {}))
    adapter.command_term = SimpleNamespace(cfg=SimpleNamespace(**settings))
    adapter.env = SimpleNamespace(step_dt=collection["step_dt"], cycle_time=collection["cycle_time"],
        cfg=SimpleNamespace(phase_ratios=collection["phase_ratios"]))
    adapter.joint_consistency = False
    adapter.joint_projection_variant = "none"
    adapter.joint_projection_metadata = None
    if joint_metadata is not None:
        spec = joint_metadata["spec"]
        if joint_metadata.get("spec_sha256") != joint_projection_spec_sha256(spec):
            raise ValueError("Joint metadata spec SHA differs")
        adapter.set_joint_projection_metadata(spec)
        adapter.joint_projection_metadata["provenance"] = joint_metadata.get("provenance", {})
    return adapter


def legacy_projection(adapter):
    """Load only the exact old projection function, never the old module init."""
    import torch
    root = Path(__file__).resolve().parents[1]
    relative = "src/mjlab_husky/supermimic/env_adapter.py"
    commit = subprocess.check_output(["git", "rev-parse", "c3c7da9^{commit}"], cwd=root, text=True).strip()
    source = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=root, text=True)
    tree = ast.parse(source)
    cls = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "CoupledStateAdapter")
    method = next(item for item in cls.body if isinstance(item, ast.FunctionDef) and item.name == "project")
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {"torch": torch}
    exec(compile(module, f"git:{commit}:{relative}", "exec"), namespace)
    return MethodType(namespace["project"], adapter), {
        "name": "legacy", "source_commit": commit,
        "adapter_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "scope": "Exact legacy project from c3c7da9; command_b stays fixed through heading resampling events."}


def physics_groups(model):
    """Keep metres, radians, quaternion coordinates and velocities separate."""
    import mujoco
    groups = {}

    def add(name, field, indices, units):
        item = groups.setdefault(name, {"field": field, "indices": [], "units": units})
        item["indices"].extend(indices)

    for joint in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) or f"joint_{joint}"
        entity = name.split("/")[0] if "/" in name else "system"
        q, v, kind = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint]), int(model.jnt_type[joint])
        if kind == int(mujoco.mjtJoint.mjJNT_FREE):
            add(entity + "_root_position", "qpos", range(q, q + 3), "m")
            add(entity + "_root_quaternion", "qpos", range(q + 3, q + 7), "quaternion coordinates")
            add(entity + "_root_linear_velocity", "qvel", range(v, v + 3), "m/s")
            add(entity + "_root_angular_velocity", "qvel", range(v + 3, v + 6), "rad/s")
        elif kind == int(mujoco.mjtJoint.mjJNT_BALL):
            add(entity + "_ball_quaternion", "qpos", range(q, q + 4), "quaternion coordinates")
            add(entity + "_ball_angular_velocity", "qvel", range(v, v + 3), "rad/s")
        elif kind == int(mujoco.mjtJoint.mjJNT_SLIDE):
            add(entity + "_slide_position", "qpos", [q], "m")
            add(entity + "_slide_velocity", "qvel", [v], "m/s")
        else:
            add(entity + "_joint_angle", "qpos", [q], "rad")
            add(entity + "_joint_velocity", "qvel", [v], "rad/s")
    return groups


def collect_batch(args, snapshot):
    # Initialize mjlab before importing its custom environment plugin.
    import mjlab.tasks  # noqa: F401
    import numpy as np
    import torch
    from tensordict import TensorDict
    from mjlab.utils.torch import configure_torch_backends
    from mjlab_husky.envs import G1SkaterManagerBasedRlEnv
    from mjlab_husky.tasks.registry import load_env_cfg, load_rl_cfg
    from mjlab_husky.supermimic.env_adapter import CoupledStateAdapter
    from mjlab_husky.supermimic.replay import EpisodeSequenceReplay
    from rsl_rl.modules import ActorCritic

    initialize_started = time.perf_counter()
    configure_torch_backends()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = load_env_cfg("Mjlab-Skater-Flat-Unitree-G1")
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    env = G1SkaterManagerBasedRlEnv(cfg, device=args.device)
    adapter = None
    try:
        obs, _ = env.reset()
        agent = load_rl_cfg("Mjlab-Skater-Flat-Unitree-G1")
        policy_cfg = asdict(agent.policy)
        policy_cfg.pop("class_name")
        teacher = ActorCritic(TensorDict(obs, batch_size=[env.num_envs]), agent.obs_groups,
                             env.action_manager.total_action_dim, **policy_cfg).to(args.device)
        checkpoint = torch.load(args.policy, map_location="cpu", weights_only=False)
        teacher.load_state_dict(checkpoint["model_state_dict"])
        teacher.eval()
        adapter = CoupledStateAdapter(env)
        expected = {name: decode_slice(sl) for name, sl in snapshot["layout"].items()}
        if adapter.layout != expected or adapter.state_dim != snapshot["world_architecture"]["state_dim"]:
            raise ValueError("Fresh environment state layout differs from the diagnostic model")
        replay = EpisodeSequenceReplay(env.num_envs, adapter.state_dim,
            env.action_manager.total_action_dim, adapter.context_dim,
            capacity_steps=args.collect_steps, device="cpu")
        groups = physics_groups(env.sim.mj_model)
        if torch.device(args.device).type == "cuda":
            torch.cuda.synchronize(args.device)
        initialized = time.perf_counter()
        with torch.inference_mode():
            for _ in range(args.collect_steps):
                obs = adapter.state_to_observations(adapter.observe())
                action = teacher.act_inference(obs)
                if args.action_noise:
                    action = action + args.action_noise * torch.randn_like(action)
                if not bool(torch.isfinite(action).all()):
                    raise FloatingPointError("Teacher emitted nonfinite actions")
                transition = adapter.capture_step(action)
                if not bool(torch.isfinite(transition.state).all()):
                    raise FloatingPointError("Nonfinite real state in audit collection")
                replay.add(transition)
        if torch.device(args.device).type == "cuda":
            torch.cuda.synchronize(args.device)
        collected = time.perf_counter()
        split = min(max(1, int(0.8 * env.num_envs)), env.num_envs - 1)
        audit_envs = torch.arange(split, env.num_envs, dtype=torch.long)
        generator = torch.Generator(device="cpu").manual_seed(args.seed + 104729)
        # Capture provenance using the same one-draw sampling operation as the
        # public replay API. The draw is then replayed to obtain the actual batch.
        choices = replay._filtered_candidates(args.horizon, audit_envs)
        generator_state = generator.get_state()
        selected = choices[torch.randint(len(choices), (args.batch_size,), generator=generator)]
        generator.set_state(generator_state)
        sampled = replay.sample(args.batch_size, args.horizon, env_ids=audit_envs, generator=generator)
        batch = {name: getattr(sampled, name).cpu() for name in
                 ("states", "actions", "rewards", "dones", "valid", "context", "terminated")}
        if bool(batch["dones"][:, :-1].any()):
            raise AssertionError("Audit replay crossed an environment reset")
        command_cfg = adapter.command_term.cfg
        payload = {"format": "husky-supermimic-real-action-audit-v1", "batch": batch,
            "layout": adapter.layout, "history_terms": adapter.history_terms,
            "critic_terms": adapter.critic_terms,
            "qpos_quaternion_slices": adapter.qpos_quaternion_slices,
            "physics_groups": groups,
            "collection": {"seed": args.seed, "num_envs": env.num_envs,
                "steps_per_env": args.collect_steps, "real_transitions": env.num_envs * args.collect_steps,
                "action_noise_std": args.action_noise, "policy_mode": "teacher mean + explicit Gaussian collection noise",
                "environment_config": "original HUSKY training config; commands, randomization and disturbances retained",
                "audit_environment_ids": audit_envs.tolist(), "unused_environment_ids": list(range(split)),
                "sampled_start_time_and_environment": selected.tolist(),
                "candidate_windows": len(choices), "sampled_windows": args.batch_size,
                "unique_sampled_windows": len(set(map(tuple, selected.tolist()))),
                "sampling_with_replacement": True,
                "split_scope": "Fresh audit trajectories; first80% unused by this batch. This split does not establish disjoint historical training identities.",
                "initialization_seconds": initialized-initialize_started,
                "collection_seconds": collected-initialized,
                "sampling_seconds": time.perf_counter()-collected,
                "step_dt": env.step_dt, "cycle_time": env.cycle_time,
                "phase_ratios": list(cfg.phase_ratios),
                "board_quaternion_slice": ([adapter.board_quaternion_slice.start,
                    adapter.board_quaternion_slice.stop, adapter.board_quaternion_slice.step]
                    if getattr(adapter, "board_quaternion_slice", None) is not None else None),
                "command_config": {"heading_command": command_cfg.heading_command,
                    "rel_heading_envs": command_cfg.rel_heading_envs,
                    "rel_standing_envs": command_cfg.rel_standing_envs,
                    "resampling_time_range": tuple(command_cfg.resampling_time_range),
                    "ranges": {"lin_vel_x": tuple(command_cfg.ranges.lin_vel_x),
                               "heading": command_cfg.ranges.heading}}},
            "teacher_sha256": sha(args.policy), "diagnostic_at_collection_sha256": sha(args.model_diagnostic)}
        if args.save_replay_corpus:
            # Persist the original real transitions before destroying the
            # collector. The fixed audit batch remains a separate artifact.
            capture_started = time.perf_counter()
            payload["_full_replay_state_dict"] = replay.state_dict()
            payload["collection"]["replay_capture_seconds"] = time.perf_counter()-capture_started
        return payload
    finally:
        if adapter is not None:
            adapter.close()
        env.close()


def finite_mean(values):
    import torch
    if values.numel() == 0 or not bool(torch.isfinite(values).all()):
        return None
    return float(values.double().mean())


def residual_stats(error, mask, scale=None, state_scale=None):
    """Never silently replace nonfinite predictions with successful numbers."""
    import torch
    selected = error[mask].double()
    count = selected.numel()
    nonfinite = int((~torch.isfinite(selected)).sum())
    result = {"transitions": int(mask.sum()), "dimensions": error.shape[-1],
              "scalar_count": count, "nonfinite_scalars": nonfinite}
    if not count or nonfinite:
        return dict(result, raw_mse=None, raw_rmse=None, raw_mae=None,
                    normalized_mse=None, state_normalized_mse=None)
    mse = float(selected.square().mean())
    result.update(raw_mse=mse, raw_rmse=mse**0.5, raw_mae=float(selected.abs().mean()))
    result["normalized_mse"] = (float((selected / scale.double()).square().mean())
                                if scale is not None else None)
    result["state_normalized_mse"] = (float((selected / state_scale.double()).square().mean())
                                      if state_scale is not None else None)
    return result


def audit(world, adapter, project, protocol, payload, horizons, device):
    import torch
    batch = {key: value.to(device) for key, value in payload["batch"].items()}
    source_states, actions = batch["states"], batch["actions"]
    phase_slice = adapter.layout["phase"]
    real_boundary = (source_states[:, 1:, phase_slice] != source_states[:, :-1, phase_slice]).any(-1)
    raw_groups = {name: (sl, "mixed raw simulator units" if name in ("qpos", "qvel")
                        else "scaled observation units" if name in ("policy", "critic")
                        else "feature coordinates; see state layout")
                  for name, sl in adapter.layout.items()}
    delta_scale = world.delta_norm.std.detach().cpu()
    state_scale = world.state_norm.std.detach().cpu()
    reward_scale = world.reward_norm.std.detach().cpu().reshape(1)
    result = {}
    with torch.inference_mode():
        for horizon in horizons:
            current = source_states[:, 0]
            active = torch.ones(len(current), dtype=torch.bool, device=device)
            errors, reward_errors, masks, reward_masks, done_probabilities = [], [], [], [], []
            for step in range(horizon):
                prediction = world(current, actions[:, step], batch["context"][:, step])
                if protocol == "conditional":
                    exogenous = adapter.exogenous_from_state(source_states[:, step+1])
                    if set(exogenous) - {"command_b", "is_heading_env", "is_standing_env", "time_left"}:
                        raise AssertionError("Conditional audit cannot inject future physics")
                    predicted = project(current, prediction["next_state"], actions[:, step], exogenous=exogenous)
                else:
                    predicted = project(current, prediction["next_state"], actions[:, step])
                valid = active & batch["valid"][:, step].bool()
                errors.append((predicted-source_states[:, step+1]).cpu())
                reward_errors.append((prediction["reward"]-batch["rewards"][:, step]).cpu())
                done_probabilities.append(prediction["termination_probability"].cpu())
                masks.append(valid.cpu())
                reward_masks.append(active.cpu())
                active = active & ~batch["dones"][:, step].bool()
                current = torch.where(active[:, None], predicted, source_states[:, step+1])
            error = torch.stack(errors, 1)
            reward_error = torch.stack(reward_errors, 1)[..., None]
            mask = torch.stack(masks, 1)
            reward_mask = torch.stack(reward_masks, 1)
            boundary = real_boundary[:, :horizon].cpu()
            window_boundary = (boundary & mask).any(-1)[:, None].expand_as(mask)
            selections = {"all_valid": mask, "boundary_transition": mask & boundary,
                          "nonboundary_transition": mask & ~boundary,
                          "windows_with_boundary": mask & window_boundary,
                          "windows_without_boundary": mask & ~window_boundary}
            summary = {}
            for condition, chosen in selections.items():
                group_stats = {"full_state": residual_stats(error, chosen, delta_scale, state_scale)}
                for name, (sl, units) in raw_groups.items():
                    group_stats[name] = dict(residual_stats(error[..., sl], chosen, delta_scale[sl], state_scale[sl]),
                                             raw_units=units)
                for name, metadata in payload.get("physics_groups", {}).items():
                    field = adapter.layout[metadata["field"]]
                    indices = torch.tensor(metadata["indices"], dtype=torch.long) + field.start
                    group_stats[name] = dict(residual_stats(error[..., indices], chosen, delta_scale[indices], state_scale[indices]),
                                             raw_units=metadata["units"])
                group_stats["task_reward"] = residual_stats(reward_error, chosen, reward_scale)
                summary[condition] = group_stats
            result[str(horizon)] = {"open_loop_real_actions": True, "horizon_steps": horizon,
                "horizon_seconds": horizon * payload["collection"]["step_dt"],
                "across_horizon": summary,
                "at_final_step": {"full_state": residual_stats(error[:, -1:], mask[:, -1:], delta_scale, state_scale),
                                  "task_reward": residual_stats(reward_error[:, -1:], mask[:, -1:], reward_scale)},
                "task_reward_including_terminal": residual_stats(reward_error, reward_mask, reward_scale),
                "predicted_failure_probability_mean": finite_mean(torch.stack(done_probabilities, 1)[reward_mask]),
                "nonfinite_predictions": int((~torch.isfinite(error[mask])).sum())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-diagnostic", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--batch", type=Path, help="Reuse a saved 32-step audit batch; no new environment collection")
    parser.add_argument("--save-replay-corpus", action="store_true",
                        help="Fresh collection only: also preserve complete CPU replay for train/held-out world fitting")
    parser.add_argument("--joint-metadata", nargs="?", const="snapshot",
                        help="Explicit joint consistency opt-in: JSON sidecar path, or omit value to use snapshot metadata")
    parser.add_argument("--projection-protocol", choices=("legacy", "conditional"), default="legacy",
                        help="Legacy c3c7da9 projection, or new projection conditioned only on recorded external commands")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--collect-steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-noise", type=float, default=0.1)
    args = parser.parse_args()
    if Path(args.label).name != args.label or not args.label:
        parser.error("label must be one nonempty filename component")
    if args.num_envs < 2 or args.horizon < 32 or args.collect_steps < args.horizon or args.batch_size < 1:
        parser.error("Use>=2 environments, horizon>=32 and enough real steps")
    if not math.isfinite(args.action_noise) or args.action_noise < 0:
        parser.error("action-noise must be finite and nonnegative")
    if args.batch and args.save_replay_corpus:
        parser.error("A sampled --batch cannot reconstruct its complete replay; collect a fresh corpus with a new label")
    if args.joint_metadata is not None and args.projection_protocol != "conditional":
        parser.error("--joint-metadata requires the conditional projection protocol")
    import mjlab.tasks  # Initialize base tasks before custom plugin discovery, as in training.
    import torch
    from mjlab_husky.supermimic.models import WorldModel
    torch.set_num_threads(1)
    args.out = args.out.resolve()
    (args.out / "data").mkdir(parents=True, exist_ok=True)
    output = args.out / "data" / f"{args.label}_model_audit.json"
    if output.exists():
        raise FileExistsError("Preserve existing audit reports; choose a fresh label")
    corpus_path = args.out / "data" / f"{args.label}_replay.pt" if args.save_replay_corpus else None
    if corpus_path is not None and corpus_path.exists():
        raise FileExistsError("Preserve existing real replay; choose a fresh label")
    corpus_metadata = None
    snapshot = torch.load(args.model_diagnostic, map_location="cpu", weights_only=False)
    if args.batch:
        batch_path = args.batch.resolve()
        payload = torch.load(batch_path, map_location="cpu", weights_only=False)
    else:
        batch_path = args.out / "data" / f"{args.label}_audit_batch.pt"
        if batch_path.exists():
            raise FileExistsError("Preserve the original audit corpus; reuse it with --batch")
        payload = collect_batch(args, snapshot)
        replay_state = payload.pop("_full_replay_state_dict", None)
        temporary = batch_path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(batch_path)
        if corpus_path is not None:
            if replay_state is None:
                raise AssertionError("Complete replay was requested but not captured")
            corpus = {key: payload[key] for key in ("layout", "history_terms", "critic_terms",
                "qpos_quaternion_slices", "physics_groups", "teacher_sha256", "diagnostic_at_collection_sha256")}
            corpus.update(format="husky-supermimic-real-replay-corpus-v1", replay_state_dict=replay_state,
                metadata={"collection": payload["collection"], "saved_audit_batch_path": str(batch_path),
                    "saved_audit_batch_sha256": sha(batch_path),
                    "audit_tool_sha256": sha(__file__),
                    "split_scope": "First80% environments are exclusively for model fitting; last20% exclusively for held-out audit. No fitting is performed by this collector."},
                train_environment_ids=payload["collection"]["unused_environment_ids"],
                heldout_environment_ids=payload["collection"]["audit_environment_ids"])
            temporary = corpus_path.with_suffix(".tmp")
            save_started = time.perf_counter()
            torch.save(corpus, temporary)
            temporary.replace(corpus_path)
            corpus_metadata = {"path": str(corpus_path), "sha256": sha(corpus_path),
                "save_and_hash_seconds": time.perf_counter()-save_started,
                "train_environment_ids": corpus["train_environment_ids"],
                "heldout_environment_ids": corpus["heldout_environment_ids"],
                "replay_shape": replay_state["shape"], "real_steps_per_environment": replay_state["total_steps"]}
            del corpus, replay_state
    if payload.get("format") != "husky-supermimic-real-action-audit-v1":
        raise ValueError("Unsupported audit batch schema")
    if payload["batch"]["actions"].shape[1] < 32:
        raise ValueError("An8-step diagnostic batch is not a32-step audit corpus")
    expected = {name: decode_slice(sl) for name, sl in snapshot["layout"].items()}
    if canonical_metadata(payload["layout"]) != canonical_metadata(expected):
        raise ValueError("Audit batch layout differs from this model snapshot")
    for key in ("history_terms", "critic_terms", "qpos_quaternion_slices"):
        if canonical_metadata(payload[key]) != canonical_metadata(snapshot[key]):
            raise ValueError(f"Audit batch {key} differs from this model snapshot")
    if payload["teacher_sha256"] != sha(args.policy):
        raise ValueError("Requested teacher differs from the saved audit collection teacher")
    batch = payload["batch"]
    dimensions = snapshot["world_architecture"]
    n, horizon, action_dim = batch["actions"].shape
    if (action_dim != dimensions["action_dim"]
            or batch["states"].shape != (n, horizon+1, dimensions["state_dim"])
            or batch["context"].shape != (n, horizon, dimensions["context_dim"])
            or any(batch[name].shape != (n, horizon) for name in ("rewards", "dones", "valid", "terminated"))):
        raise ValueError("Audit tensor dimensions differ from this model snapshot")
    if bool(batch["dones"][:, :-1].any()):
        raise ValueError("Saved audit batch crosses an environment reset")
    if any(not bool(torch.isfinite(value).all()) for value in batch.values()):
        raise FloatingPointError("Audit corpus contains nonfinite real data")
    initialization_started = time.perf_counter()
    world = WorldModel(**snapshot["world_architecture"])
    world.load_state_dict(snapshot["world"])
    world.to(args.device).eval().requires_grad_(False)
    timing = payload["collection"]
    joint_metadata = load_joint_metadata(snapshot, args.joint_metadata)
    adapter = projection_adapter(snapshot, timing, joint_metadata)
    if args.projection_protocol == "legacy":
        project, protocol_metadata = legacy_projection(adapter)
    else:
        if "command_config" not in timing or adapter.board_quaternion_slice is None:
            raise ValueError("Conditional protocol needs collection command config and board quaternion layout")
        project = adapter.project
        protocol_metadata = {"name": "conditional", "exogenous_fields":
            [name for name in ("command_b", "is_heading_env", "is_standing_env", "time_left") if name in expected],
            "scope": "Only recorded external commands are supplied. Physical states, heading_ref and observations remain model-predicted or reconstructed from predicted physics."}
    normalizers = {name: value.detach().clone() for name, value in world.state_dict().items() if "_norm." in name}
    if torch.device(args.device).type == "cuda":
        torch.cuda.synchronize(args.device)
    started = time.perf_counter()
    metrics = audit(world, adapter, project, args.projection_protocol, payload, [1, 8, 16, 32], args.device)
    if torch.device(args.device).type == "cuda":
        torch.cuda.synchronize(args.device)
    if not all(torch.equal(world.state_dict()[name], value) for name, value in normalizers.items()):
        raise AssertionError("Audit changed model normalization")
    report = {"algorithm_updates": 0, "optimizer_steps": 0, "model_frozen": True,
        "label": args.label, "normalizers_unchanged": True, "projection_protocol": protocol_metadata,
        "joint_projection": {"enabled": adapter.joint_consistency, "variant": adapter.joint_projection_variant,
            "spec_sha256": adapter.joint_projection_metadata["spec_sha256"] if joint_metadata is not None else None,
            "metadata_argument": args.joint_metadata,
            "metadata_file_sha256": sha(args.joint_metadata) if args.joint_metadata not in (None, "snapshot") else None},
        "normalization": {"normalized_mse": "Residual divided by frozen delta_norm.std, matching model-fitting loss.",
            "state_normalized_mse": "Residual divided by frozen state_norm.std. Both scales are reported without refitting.",
            "task_reward": "Reward residual divided by frozen reward_norm.std."},
        "created_at": time.time(), "model_diagnostic_sha256": sha(args.model_diagnostic),
        "audit_tool_sha256": sha(__file__), "current_adapter_sha256": sha(
            Path(__file__).resolve().parents[1] / "src/mjlab_husky/supermimic/env_adapter.py"),
        "teacher_sha256": sha(args.policy), "audit_batch_path": str(batch_path),
        "audit_batch_sha256": sha(batch_path), "reused_batch": bool(args.batch),
        "saved_replay_corpus": corpus_metadata,
        "collection": timing, "audit_initialization_seconds": started-initialization_started,
        "audit_seconds": time.perf_counter()-started,
        "horizons": metrics,
        "scope": "Open-loop predictions with recorded real actions, not an imagined-policy rollout or real policy success test. Normalizers are unchanged. Qpos/qvel aggregates mix units; physical subgroups separate units. Null statistics flag empty or nonfinite data."}
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"report": str(output), "report_sha256": sha(output),
        "audit_batch": str(batch_path), "audit_batch_sha256": report["audit_batch_sha256"],
        "saved_replay_corpus": corpus_metadata,
        "summary": {h: {"state": x["across_horizon"]["all_valid"]["full_state"],
                         "reward": x["across_horizon"]["all_valid"]["task_reward"]}
                    for h, x in metrics.items()}}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
