"""Refine only a conditional SuperMimic world model on saved training data.

The original normalizers stay fixed. A separate, hash-bound 32-step held-out
batch is used only for diagnostics; no actor or critic is instantiated.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("WANDB_MODE", "disabled")


def synchronize(device):
    import torch
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def tensor_hash(values):
    """Stable identity across tensor storage devices and dictionary ordering."""
    digest = hashlib.sha256()
    for name, tensor in sorted(values.items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def normalizer_buffers(world):
    return {name: value.detach().clone() for name, value in world.state_dict().items()
            if "_norm." in name}


def assert_normalizers_unchanged(world, original):
    import torch
    current = world.state_dict()
    if not all(torch.equal(current[name], value) for name, value in original.items()):
        raise AssertionError("Refinement changed a frozen original normalizer")


def validate_inputs(snapshot, corpus, payload, batch_sha, canonical):
    """Reject overlapping splits, mismatched artifacts and stale data schemas."""
    import torch
    if snapshot.get("dynamics_protocol") != "conditional-commands-v2":
        raise ValueError("Refinement requires a trained conditional-commands-v2 diagnostic")
    if corpus.get("format") != "husky-supermimic-real-replay-corpus-v1":
        raise ValueError("Unsupported replay corpus schema")
    if payload.get("format") != "husky-supermimic-real-action-audit-v1":
        raise ValueError("Unsupported fixed audit batch schema")
    if corpus["metadata"]["saved_audit_batch_sha256"] != batch_sha:
        raise ValueError("The held-out batch SHA is not bound to this replay corpus")
    for key in ("layout", "history_terms", "critic_terms", "qpos_quaternion_slices"):
        if canonical(snapshot[key]) != canonical(corpus[key]) or canonical(snapshot[key]) != canonical(payload[key]):
            raise ValueError(f"Model, corpus and held-out {key} differ")
    for key in ("teacher_sha256", "diagnostic_at_collection_sha256"):
        if corpus[key] != payload[key]:
            raise ValueError(f"Replay and fixed audit {key} differ")
    collection = corpus["metadata"]["collection"]
    if canonical(collection) != canonical(payload["collection"]):
        raise ValueError("Replay and held-out collection metadata differ")
    n, state_dim, action_dim, context_dim, _ = corpus["replay_state_dict"]["shape"]
    arch = snapshot["world_architecture"]
    if (state_dim, action_dim, context_dim) != (arch["state_dim"], arch["action_dim"], arch["context_dim"]):
        raise ValueError("Replay dimensions differ from the frozen world architecture")
    train_ids = [int(value) for value in canonical(corpus["train_environment_ids"])]
    heldout_ids = [int(value) for value in canonical(corpus["heldout_environment_ids"])]
    split = min(max(1, int(0.8*n)), n-1)
    if n < 2 or train_ids != list(range(split)) or heldout_ids != list(range(split, n)):
        raise ValueError("Require disjoint first80% training and remaining held-out environment IDs")
    if (int(collection["num_envs"]) != n
            or list(canonical(collection["unused_environment_ids"])) != train_ids
            or list(canonical(collection["audit_environment_ids"])) != heldout_ids):
        raise ValueError("Replay split differs from the original collection metadata")
    if set(train_ids) & set(heldout_ids):
        raise ValueError("Training and held-out environment IDs overlap")
    starts = payload["collection"]["sampled_start_time_and_environment"]
    selected_ids = [int(row[1]) for row in starts]
    if not selected_ids or any(value not in set(heldout_ids) for value in selected_ids):
        raise ValueError("The fixed audit batch contains training environments")
    batch = payload["batch"]
    b, horizon, actions = batch["actions"].shape
    if (horizon < 32 or actions != action_dim or len(starts) != b
            or batch["states"].shape != (b, horizon+1, state_dim)
            or batch["context"].shape != (b, horizon, context_dim)
            or any(batch[name].shape != (b, horizon) for name in ("rewards", "dones", "valid", "terminated"))):
        raise ValueError("Fixed audit tensors do not contain compatible real 32-step windows")
    if bool(batch["dones"][:, :-1].any()):
        raise ValueError("The held-out batch crosses a reset boundary")
    if any(not bool(torch.isfinite(value).all()) for value in batch.values()):
        raise FloatingPointError("Nonfinite real data in the fixed held-out batch")
    if not collection.get("command_config") or collection.get("board_quaternion_slice") is None:
        raise ValueError("Conditional projection needs command configuration and board quaternion layout")
    return train_ids, heldout_ids


def compact_metrics(metrics, physics_groups):
    """Keep raw physical errors visible beside full audit results."""
    names = ("full_state", "qpos", "qvel", "policy", "task_reward", *physics_groups)
    return {horizon: {condition: {name: values[name] for name in names if name in values}
                     for condition, values in item["across_horizon"].items()}
            for horizon, item in metrics.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-diagnostic", required=True, type=Path)
    parser.add_argument("--replay-corpus", required=True, type=Path)
    parser.add_argument("--audit-batch", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model-window", type=int, choices=(8, 16), default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--max-seconds", type=float, default=120.0,
                        help="Soft wall limit after initialization, including held-out diagnostics; final export follows")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--joint-metadata", nargs="?", const="snapshot",
                        help="Explicit joint consistency opt-in: JSON sidecar path, or omit value for snapshot metadata")
    args = parser.parse_args()
    if not args.label or args.label in (".", "..") or Path(args.label).name != args.label:
        parser.error("label must be one nonempty filename component")
    if args.steps < 1 or args.batch_size < 1:
        parser.error("steps and batch-size must be positive")
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("max-seconds must be finite and positive")

    initialized_started = time.perf_counter()
    import mjlab.tasks  # Initialize base tasks before custom plugin discovery.
    import torch
    import audit_supermimic_model as audit_tool
    from mjlab.utils.torch import configure_torch_backends
    from mjlab_husky.supermimic.models import WorldModel, world_model_loss
    from mjlab_husky.supermimic.replay import EpisodeSequenceReplay
    configure_torch_backends()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    args.out = args.out.resolve()
    paths = {kind: args.out / kind for kind in ("data", "checkpoints")}
    for directory in paths.values():
        directory.mkdir(parents=True, exist_ok=True)
    run_checkpoints = paths["checkpoints"] / args.label
    updates_path = paths["data"] / f"{args.label}_world_updates.jsonl"
    audits_path = paths["data"] / f"{args.label}_world_audits.jsonl"
    summary_path = paths["data"] / f"{args.label}_world_refinement.json"
    if run_checkpoints.exists() or any(path.exists() for path in (updates_path, audits_path, summary_path)):
        raise FileExistsError("Preserve previous refinements; use a fresh label")
    snapshot = torch.load(args.model_diagnostic, map_location="cpu", weights_only=False)
    corpus = torch.load(args.replay_corpus, map_location="cpu", weights_only=False)
    payload = torch.load(args.audit_batch, map_location="cpu", weights_only=False)
    identities = {"initial_diagnostic_sha256": audit_tool.sha(args.model_diagnostic),
        "replay_corpus_sha256": audit_tool.sha(args.replay_corpus),
        "audit_batch_sha256": audit_tool.sha(args.audit_batch),
        "tool_sha256": audit_tool.sha(__file__), "audit_tool_sha256": audit_tool.sha(audit_tool.__file__)}
    root = Path(__file__).resolve().parents[1]
    for name in ("models", "env_adapter", "replay"):
        identities[name + "_source_sha256"] = audit_tool.sha(root / "src/mjlab_husky/supermimic" / f"{name}.py")
    train_ids, heldout_ids = validate_inputs(snapshot, corpus, payload,
        identities["audit_batch_sha256"], audit_tool.canonical_metadata)
    shape = corpus["replay_state_dict"]["shape"]
    replay = EpisodeSequenceReplay(*shape[:4], capacity_steps=shape[4], device="cpu")
    replay.load_state_dict(corpus["replay_state_dict"])
    train_ids_tensor = torch.tensor(train_ids, dtype=torch.long)
    generator_seed = args.seed + 32452843  # Separate from audit/validation seed+104729.
    generator = torch.Generator(device="cpu").manual_seed(generator_seed)
    world = WorldModel(**snapshot["world_architecture"]).to(args.device)
    world.load_state_dict(snapshot["world"])
    world.train().requires_grad_(True)
    original_normalizers = normalizer_buffers(world)
    identities["initial_world_state_sha256"] = tensor_hash(world.state_dict())
    identities["frozen_normalizers_sha256"] = tensor_hash(original_normalizers)
    lr = float(snapshot["config"]["model_lr"])
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("Original model learning rate must be finite and positive")
    optimizer = torch.optim.Adam(world.parameters(), lr=lr)
    joint_metadata = audit_tool.load_joint_metadata(snapshot, args.joint_metadata)
    adapter = audit_tool.projection_adapter(snapshot, corpus["metadata"]["collection"], joint_metadata)
    joint_projection = {"enabled": adapter.joint_consistency, "variant": adapter.joint_projection_variant,
        "spec_sha256": adapter.joint_projection_metadata["spec_sha256"] if joint_metadata is not None else None,
        "metadata_argument": args.joint_metadata,
        "metadata_file_sha256": audit_tool.sha(args.joint_metadata) if args.joint_metadata not in (None, "snapshot") else None}
    # Fail before training if this split cannot supply the requested window.
    candidates = replay._filtered_candidates(args.model_window, train_ids_tensor)
    if not bool(torch.isin(candidates[:, 1], train_ids_tensor).all()):
        raise AssertionError("Training candidates escaped their environment split")
    run_checkpoints.mkdir()
    synchronize(args.device)
    initialization_seconds = time.perf_counter() - initialized_started
    actual_steps = nominal_transitions = valid_state_targets = 0
    training_seconds = validation_seconds = export_seconds = 0.0
    records = []
    budget_started = time.perf_counter()

    def export(name):
        nonlocal export_seconds
        started = time.perf_counter()
        assert_normalizers_unchanged(world, original_normalizers)
        contents = {key: snapshot[key] for key in ("world_architecture", "layout", "history_terms",
            "critic_terms", "qpos_quaternion_slices", "config", "iteration") if key in snapshot}
        contents.update(format="husky-supermimic-world-refinement-diagnostic-v1",
            world={key: value.detach().cpu() for key, value in world.state_dict().items()},
            dynamics_protocol="conditional-commands-v2", refinement_step=actual_steps,
            joint_consistency=adapter.joint_consistency, projection_variant=adapter.joint_projection_variant,
            joint_projection_metadata=adapter.joint_projection_metadata,
            world_refinement={"model_window": args.model_window, "learning_rate": lr,
                "optimizer_initialization": "Fresh Adam at original model_lr; diagnostic contains no optimizer resume state.",
                "nominal_fitting_transitions": nominal_transitions, "valid_state_targets": valid_state_targets,
                "actor_optimizer_steps": 0, "critic_optimizer_steps": 0, "normalizers_unchanged": True,
                "joint_projection": joint_projection,
                **identities}, scope="World-only diagnostic; no policy, replay or optimizer resume state")
        path = run_checkpoints / name
        temporary = path.with_suffix(".tmp")
        torch.save(contents, temporary)
        temporary.replace(path)
        export_seconds += time.perf_counter() - started
        return {"path": str(path), "sha256": audit_tool.sha(path)}

    def evaluate():
        nonlocal validation_seconds
        synchronize(args.device)
        started = time.perf_counter()
        world.eval()
        training_generator_state = generator.get_state().clone()
        metrics = audit_tool.audit(world, adapter, adapter.project, "conditional", payload,
                                   [1, 8, 16, 32], args.device)
        if not torch.equal(training_generator_state, generator.get_state()):
            raise AssertionError("Held-out evaluation advanced the independent training generator")
        synchronize(args.device)
        elapsed = time.perf_counter() - started
        validation_seconds += elapsed
        world.train()
        assert_normalizers_unchanged(world, original_normalizers)
        record = {"world_optimizer_steps": actual_steps, "actor_optimizer_steps": 0,
            "critic_optimizer_steps": 0, "nominal_fitting_transitions": nominal_transitions,
            "valid_state_targets": valid_state_targets, "validation_seconds": elapsed,
            "elapsed_after_initialization_seconds": time.perf_counter()-budget_started,
            "normalizers_unchanged": True, "joint_projection": joint_projection, "horizons": metrics,
            "raw_core_metrics": compact_metrics(metrics, payload.get("physics_groups", {})), **identities}
        with audits_path.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False)+"\n")
        records.append({"world_optimizer_steps": actual_steps,
                        "raw_core_metrics": record["raw_core_metrics"]})
        return record

    status, error = "failed", None
    try:
        evaluate()  # Step zero uses exactly the common trained checkpoint.
        while actual_steps < args.steps:
            if time.perf_counter()-budget_started >= args.max_seconds:
                break
            started = time.perf_counter()
            sample = replay.sample(args.batch_size, args.model_window, env_ids=train_ids_tensor,
                                   generator=generator)
            batch = {key: getattr(sample, key).to(args.device) for key in
                     ("states", "actions", "rewards", "dones", "valid", "context", "terminated")}
            losses = world_model_loss(world, batch, project=adapter.project,
                exogenous_from_state=adapter.exogenous_from_state, horizon=args.model_window)
            if not bool(torch.isfinite(losses["loss"])):
                raise FloatingPointError("Nonfinite world loss; optimizer step skipped")
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            norm = torch.nn.utils.clip_grad_norm_(world.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            synchronize(args.device)
            elapsed = time.perf_counter()-started
            training_seconds += elapsed
            actual_steps += 1
            nominal_transitions += args.batch_size*args.model_window
            valid_state_targets += int(sample.valid.sum())
            record = {"world_optimizer_steps": actual_steps, "seconds": elapsed,
                "nominal_fitting_transitions": nominal_transitions,
                "losses": {key: float(value.detach().cpu()) for key, value in losses.items()},
                "gradient_norm": float(norm.detach().cpu())}
            with updates_path.open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False)+"\n")
            if actual_steps % 100 == 0:
                evaluate()
                checkpoint = export(f"world_{actual_steps}.pt")
                print(json.dumps({"step": actual_steps, "checkpoint": checkpoint,
                    "raw_core_metrics": records[-1]["raw_core_metrics"]}, allow_nan=False), flush=True)
        status = "completed" if actual_steps == args.steps else "stopped_time_limit"
        if records[-1]["world_optimizer_steps"] != actual_steps:
            evaluate()
    except BaseException as caught:
        error = repr(caught)
        raise
    finally:
        final = export("world_final.pt")
        summary = {"status": status, "error": error, "label": args.label,
            "joint_projection": joint_projection,
            "world_optimizer_steps": actual_steps, "actor_optimizer_steps": 0, "critic_optimizer_steps": 0,
            "requested_steps_cap": args.steps, "max_seconds": args.max_seconds,
            "time_limit_scope": "Soft wall time after initialization, including baseline/periodic validation and intermediate exports; final validation/export may overrun.",
            "model_window": args.model_window, "batch_size": args.batch_size, "model_learning_rate": lr,
            "optimizer_initialization": "Fresh Adam at original model_lr; no original optimizer moments restored.",
            "initialization_seconds": initialization_seconds, "training_seconds": training_seconds,
            "validation_seconds": validation_seconds, "export_seconds": export_seconds,
            "elapsed_after_initialization_seconds": time.perf_counter()-budget_started,
            "nominal_fitting_transitions": nominal_transitions, "valid_state_targets": valid_state_targets,
            "generator_seed": generator_seed, "generator_device": "cpu",
            "training_environment_ids": train_ids, "heldout_environment_ids": heldout_ids,
            "training_candidate_windows": len(candidates), "normalizers_unchanged": True,
            "final_diagnostic": final, "audit_ledger": str(audits_path), "update_ledger": str(updates_path),
            "audits": records, **identities,
            "scope": "World-only held-out model refinement. Fixed audit data never enters fitting. This is not policy training or a policy-quality result."}
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False))
        print(json.dumps({"summary": str(summary_path), "summary_sha256": audit_tool.sha(summary_path),
            "status": status, "actual_world_steps": actual_steps, "final_diagnostic": final}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
