"""Offline FP32 eager versus compiled-world-forward numerical/latency audit.

Use the same trained diagnostic, fixed32 real corpus and original teacher as
audit_supermimic_precision.py. Only a bound WorldModel.forward is compiled;
actor/critic, projection, autoregressive loops, losses and optimizers stay out
of the compiled region. No optimizer or real environment is created.

Run under an external timeout (for example240s). A progress JSON preserves the
last started workload if cold compilation exceeds that process budget.
"""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import time

from audit_supermimic_precision import (
    actor_gradient, finite_scalar, fixed_exogenous, gradient_comparison,
    load_policy, matmul_precision_metadata, prediction_comparison,
    recorded_action_rollout, rng_state, same_rng, tensor_dict_sha256,
    validate_inputs,
)
from audit_supermimic_model import projection_adapter, sha


def install_compiled_forward(world, *, backend="inductor"):
    """Retain the original module tree, parameters and checkpoint key names."""
    import torch
    keys = tuple(world.state_dict())
    original = world.forward
    arguments = {"backend": backend, "fullgraph": True, "dynamic": False}
    if backend == "inductor":
        # First isolate kernel fusion; repeated autoregressive outputs stay live
        # until backward, so CUDAGraph lifetime/capture is a separate experiment.
        arguments["options"] = {"triton.cudagraphs": False}
    world.forward = torch.compile(original, **arguments)
    if tuple(world.state_dict()) != keys:
        raise AssertionError("Compiling forward changed checkpoint keys")
    return original


def compiler_counters():
    """Best-effort diagnostics only; never clear or change compiler state."""
    try:
        from torch._dynamo.utils import counters
        return {str(group): {str(key): int(value) for key, value in values.items()
                            if isinstance(value, (int, bool))}
                for group, values in counters.items()}
    except (ImportError, AttributeError, RuntimeError) as error:
        return {"unavailable_reason": str(error)}


def tf32_compatibility_metadata():
    """Check Inductor's legacy reads without invoking any legacy setter.

    PyTorch2.10's new precision setter leaves a separate legacy enum at its
    default HIGHEST, while Inductor still queries legacy getters. On that build,
    launch a FRESH process with TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1: Context.h
    initializes both representations consistently, then pinned mjlab's usual
    new-API settings retain TF32 permission. Setting only the new global flag
    does not update the legacy enum and cannot repair this incompatibility.
    """
    import torch
    override = os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")
    nvidia_override = os.environ.get("NVIDIA_TF32_OVERRIDE")
    if nvidia_override == "0":
        raise ValueError("NVIDIA_TF32_OVERRIDE=0 disables TF32 and invalidates this training-matched speed comparison")
    try:
        legacy_matmul = torch.backends.cuda.matmul.allow_tf32
        legacy_global = torch.get_float32_matmul_precision()
        legacy_cudnn = torch.backends.cudnn.allow_tf32
    except RuntimeError as error:
        raise RuntimeError(
            "PyTorch/Inductor TF32 getters disagree before compilation. On pinned PyTorch2.10, "
            "start a fresh process with TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 before importing torch, "
            "then retain the original mjlab configure_torch_backends() new-API settings. "
            "Do not disable TF32 or change one variant's math. Original error: " + str(error)
        ) from error
    new_cuda = torch.backends.cuda.matmul.fp32_precision
    new_cudnn = torch.backends.cudnn.fp32_precision
    if new_cuda != "tf32" or new_cudnn != "tf32" or not legacy_matmul or not legacy_cudnn:
        raise ValueError("Both eager/compiled must retain the HUSKY training configuration's CUDA/cuDNN TF32 permission")
    return {"TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": override, "NVIDIA_TF32_OVERRIDE": nvidia_override,
        "new_global_precision": torch.backends.fp32_precision, "new_cuda_matmul_precision": new_cuda,
        "new_cudnn_precision": new_cudnn, "new_cudnn_conv_precision": torch.backends.cudnn.conv.fp32_precision,
        "new_cudnn_rnn_precision": torch.backends.cudnn.rnn.fp32_precision,
        "legacy_matmul_allow_tf32": legacy_matmul, "legacy_global_matmul_precision": legacy_global,
        "legacy_cudnn_allow_tf32": legacy_cudnn, "legacy_and_new_getters_compatible": True,
        "relative_to_original_training": "Original pinned mjlab helper sets CUDA matmul and cuDNN new precision fields to tf32, benchmark=True, deterministic=False. Those math permissions are retained for BOTH variants. The fresh-process environment compatibility bootstrap changes the legacy matmul enum initialization (HIGH instead of default HIGHEST) so Inductor's legacy reads agree; no legacy setter is called here. Flag agreement alone does not prove original-vs-bootstrap eager output/gradient equality; verify that separately before production adoption."}


def rename_variants(value):
    """Reuse precision's pairwise math, without labeling either path BF16."""
    if isinstance(value, dict):
        return {key.replace("bf16", "compiled").replace("fp32", "eager"):
                rename_variants(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rename_variants(item) for item in value]
    return value


def numerical_gates(horizons, gradient):
    """Provisional FP32 compiler tolerances, stricter than the BF16 proposal."""
    checks = []
    for horizon, comparison in horizons.items():
        for location in ("across_horizon", "at_final_step"):
            for group, eager in comparison["eager_true_error"][location].items():
                compiled = comparison["compiled_true_error"][location][group]
                if group == "physical_failure_probability":
                    continue
                for metric, fraction in (("raw_rmse", .01), ("raw_absolute_p95", .02)):
                    left, right = eager[metric], compiled[metric]
                    checks.append({"horizon": int(horizon), "location": location, "group": group,
                        "metric": metric, "eager": left, "compiled": right,
                        "passed": left is not None and right is not None and right <= left*(1+fraction)+1e-8})
    relative, cosine = gradient["gradient_relative_l2"], gradient["gradient_cosine"]
    gradient_pass = (gradient["all_gradients_finite"] and relative is not None and cosine is not None and
                     relative <= .001 and cosine >= .9999)
    return {"suggested_limits": {"actor_gradient_relative_l2_maximum": .001,
            "actor_gradient_cosine_minimum": .9999, "true_rmse_relative_increase_maximum": .01,
            "true_absolute_p95_relative_increase_maximum": .02, "absolute_error_slack": 1e-8},
        "actor_gradient_passed": gradient_pass, "true_error_checks": checks,
        "all_true_error_checks_passed": all(item["passed"] for item in checks),
        "all_suggested_numerical_gates_passed": gradient_pass and all(item["passed"] for item in checks),
        "scope": "Suggested same-FP32 arithmetic gates, not validated physical tolerances. Near-zero gradients require the reported absolute differences. Equal prediction error does not establish equal learning or real-policy quality."}


def run_world_backward(world, adapter, batch):
    from mjlab_husky.supermimic.models import world_model_loss
    losses = world_model_loss(world, batch, project=adapter.project, horizon=8,
                              exogenous_from_state=adapter.exogenous_from_state)
    losses["loss"].backward()
    return losses


def run_imagined_backward(policy, world, adapter, batch, commands, gamma, lam):
    from mjlab_husky.supermimic.imagination import imagine
    counter = 0

    def command_sampler(_detached_state):
        nonlocal counter
        fields = commands[counter]
        counter += 1
        return fields

    imagined = imagine(policy, world, batch["states"][:, 0], adapter.state_to_observations,
        horizon=32, gamma=gamma, lam=lam, context=batch["context"][:, :32],
        project=adapter.project, exogenous_sampler=command_sampler)
    if counter != 32:
        raise AssertionError("Expected exactly32 fixed command inputs")
    imagined["actor_loss"].backward()
    return {name: imagined[name] for name in ("objective", "actor_loss", "reward_objective", "bootstrap_objective")}


def timed_workload(kind, policy, world, adapter, batch, commands, gamma, lam, device):
    """Gradient clearing/metric transfers are excluded; all Python loss code included."""
    import torch
    policy.zero_grad(set_to_none=True)
    world.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    else:
        begin = end = None
    started = time.perf_counter()
    if begin is not None:
        begin.record()
    if kind == "world8":
        values = run_world_backward(world, adapter, batch)
    elif kind == "imagination32":
        values = run_imagined_backward(policy, world, adapter, batch, commands, gamma, lam)
    else:
        raise ValueError(f"Unknown workload {kind}")
    if end is not None:
        end.record()
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter()-started
    event = begin.elapsed_time(end) / 1000 if begin is not None else None
    finite_gradients = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                           for module in (policy, world) for parameter in module.parameters())
    if kind == "imagination32" and any(parameter.grad is not None for parameter in world.parameters()):
        raise AssertionError("Imagined BPTT changed the world-weight gradient scope")
    if any(value.dtype != torch.float32 for value in values.values()):
        raise AssertionError("Loss/objective arithmetic did not remain FP32")
    losses = {key: finite_scalar(value.detach()) for key, value in values.items()}
    return {"wall_seconds": elapsed, "cuda_event_seconds": event,
            "losses": losses, "gradients_finite": finite_gradients}


def latency_summary(samples):
    wall = [item["wall_seconds"] for item in samples]
    return {"wall_seconds": wall, "wall_median_seconds": statistics.median(wall),
            "wall_mean_seconds": statistics.mean(wall), "wall_min_seconds": min(wall),
            "cuda_event_seconds": [item["cuda_event_seconds"] for item in samples],
            "all_gradients_finite": all(item["gradients_finite"] for item in samples),
            "last_losses": samples[-1]["losses"]}


def progress_write(path, progress, stage, **details):
    progress.update(stage=stage, updated_at=time.time(), **details)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(progress, indent=2, allow_nan=False))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-diagnostic", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        parser.error("label must contain ASCII letters, digits, underscore or hyphen")
    if args.warmup < 1 or args.repeats < 2:
        parser.error("Use at least one warmup and two alternating repeats per variant")
    import torch
    from mjlab.utils.torch import configure_torch_backends
    from mjlab_husky.supermimic.models import WorldModel
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("This benchmark requires CUDA; CPU helper tests do not establish CUDA compile speed")
    torch.cuda.set_device(device)
    configure_torch_backends()
    tf32_compatibility = tf32_compatibility_metadata()
    torch.set_num_threads(1)
    data_dir = args.out.resolve() / "data"
    output = data_dir / f"{args.label}_compile_audit.json"
    progress_path = data_dir / f"{args.label}_compile_progress.json"
    if output.exists() or progress_path.exists():
        raise FileExistsError("Preserve existing compile evidence; choose a new label")
    snapshot = torch.load(args.model_diagnostic, map_location="cpu", weights_only=False)
    payload = torch.load(args.batch, map_location="cpu", weights_only=False)
    teacher_sha = sha(args.policy)
    heads = validate_inputs(snapshot, payload, teacher_sha)
    adapter = projection_adapter(snapshot, payload["collection"])
    batch = {key: value.to(device) for key, value in payload["batch"].items()}
    commands = fixed_exogenous(adapter, batch["states"])
    gamma, lam = snapshot.get("config", {}).get("gamma", .99), snapshot.get("config", {}).get("lam", .95)
    progress = {"format": "husky-supermimic-compile-progress-v1", "label": args.label,
        "algorithm_updates": 0, "optimizer_steps": 0, "started_at": time.time(),
        "tf32_compatibility": tf32_compatibility,
        "model_diagnostic_sha256": sha(args.model_diagnostic), "audit_batch_sha256": sha(args.batch),
        "teacher_sha256": teacher_sha, "cold_workloads": {}, "warmup_workloads": {}}
    data_dir.mkdir(parents=True, exist_ok=True)
    progress_write(progress_path, progress, "constructing_worlds")
    original_rng = rng_state(device)
    started_all = time.perf_counter()
    try:
        with torch.random.fork_rng(devices=[device.index if device.index is not None else torch.cuda.current_device()]):
            worlds, initial_hashes = {}, {}
            for variant in ("eager", "compiled"):
                world = WorldModel(**dict(snapshot["world_architecture"], mixed_precision="none"))
                world.load_state_dict(snapshot["world"], strict=True)
                worlds[variant] = world.to(device=device, dtype=torch.float32).eval()
                initial_hashes[variant] = tensor_dict_sha256(world.state_dict())
            if initial_hashes["eager"] != initial_hashes["compiled"]:
                raise AssertionError("Variants must share exact weights and normalizers")
            initial_keys = tuple(worlds["compiled"].state_dict())
            wrapper_started = time.perf_counter()
            original_forward = install_compiled_forward(worlds["compiled"])
            wrapper_seconds = time.perf_counter()-wrapper_started
            policy = load_policy(torch.load(args.policy, map_location="cpu", weights_only=False), adapter,
                                 worlds["eager"].action_dim, device)
            policy_hash = tensor_dict_sha256(policy.state_dict())
            audit_rng = rng_state(device)
            counters_before = compiler_counters()
            predictions, cold = {}, {}
            cold_started = time.perf_counter()
            # Run complete first workloads so every real requires_grad/stride
            # specialization compiles outside measured steady-state repetitions.
            for variant, world in worlds.items():
                progress_write(progress_path, progress, f"cold_{variant}_prediction32")
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                predictions[variant] = recorded_action_rollout(world, adapter, batch, commands)
                torch.cuda.synchronize(device)
                cold[f"{variant}_prediction32"] = {"wall_seconds": time.perf_counter()-started,
                                                      "compiler_counters": compiler_counters()}
                progress["cold_workloads"] = cold
                for kind in ("world8", "imagination32"):
                    progress_write(progress_path, progress, f"cold_{variant}_{kind}")
                    item = timed_workload(kind, policy, world, adapter, batch, commands, gamma, lam, device)
                    cold[f"{variant}_{kind}"] = dict(item, compiler_counters=compiler_counters())
                    progress["cold_workloads"] = cold
            cold_seconds = time.perf_counter()-cold_started
            warmups = {}
            for kind in ("world8", "imagination32"):
                for variant, world in worlds.items():
                    progress_write(progress_path, progress, f"warmup_{variant}_{kind}")
                    warmups[f"{variant}_{kind}"] = [timed_workload(kind, policy, world, adapter, batch,
                        commands, gamma, lam, device) for _ in range(args.warmup)]
                    progress["warmup_workloads"] = {key: latency_summary(values) for key, values in warmups.items()}
            counters_after_warmup = compiler_counters()
            warmup_complete = time.perf_counter()
            progress_write(progress_path, progress, "steady_state_benchmarks",
                           counters_after_warmup=counters_after_warmup)
            measurements = {}
            for kind in ("world8", "imagination32"):
                samples = {variant: [] for variant in worlds}
                for repeat in range(args.repeats):
                    for variant in (tuple(worlds) if repeat % 2 == 0 else tuple(reversed(worlds))):
                        samples[variant].append(timed_workload(kind, policy, worlds[variant], adapter,
                            batch, commands, gamma, lam, device))
                measurements[kind] = {variant: latency_summary(values) for variant, values in samples.items()}
                measurements[kind]["median_wall_speed_ratio_eager_over_compiled"] = (
                    measurements[kind]["eager"]["wall_median_seconds"] /
                    measurements[kind]["compiled"]["wall_median_seconds"])
            counters_after_timing = compiler_counters()
            progress_write(progress_path, progress, "numerical_and_actor_gradient_comparison",
                           counters_after_timing=counters_after_timing)
            horizons = rename_variants(prediction_comparison(worlds["eager"], adapter, payload,
                {"fp32": predictions["eager"], "bf16": predictions["compiled"]}))
            gradients, gradient_summaries = {}, {}
            for old_name, variant in (("fp32", "eager"), ("bf16", "compiled")):
                gradients[old_name], gradient_summaries[old_name] = actor_gradient(policy,
                    worlds[variant], adapter, batch, commands, gamma=gamma, lam=lam)
            gradient = rename_variants(gradient_comparison(gradients, gradient_summaries))
            final_hashes = {variant: tensor_dict_sha256(world.state_dict()) for variant, world in worlds.items()}
            if initial_hashes != final_hashes or tensor_dict_sha256(policy.state_dict()) != policy_hash:
                raise AssertionError("Audit changed model/policy weights or normalizers")
            if tuple(worlds["compiled"].state_dict()) != initial_keys:
                raise AssertionError("Compiled checkpoint keys changed")
            if not same_rng(audit_rng, rng_state(device)):
                raise AssertionError("Fixed-data audit consumed RNG")
            # Keep the owning module serializable even in an interactive caller.
            worlds["compiled"].forward = original_forward
        if not same_rng(original_rng, rng_state(device)):
            raise AssertionError("Audit did not restore construction RNG")
        before_graphs = counters_after_warmup.get("stats", {}).get("unique_graphs")
        after_graphs = counters_after_timing.get("stats", {}).get("unique_graphs")
        cache_stable = before_graphs == after_graphs if before_graphs is not None and after_graphs is not None else None
        source_root = Path(__file__).resolve().parents[1] / "src/mjlab_husky/supermimic"
        report = {"format": "husky-supermimic-compile-audit-v1", "label": args.label, "created_at": time.time(),
            "algorithm_updates": 0, "optimizer_steps": 0, "model_and_policy_state_unchanged": True,
            "state_dict_keys_unchanged": True, "normalizers_unchanged": True, "rng_unchanged": True,
            "model_diagnostic_path": str(args.model_diagnostic.resolve()), "model_diagnostic_sha256": sha(args.model_diagnostic),
            "world_state_sha256": initial_hashes["eager"], "teacher_path": str(args.policy.resolve()),
            "teacher_sha256": teacher_sha, "policy_state_sha256": policy_hash,
            "audit_batch_path": str(args.batch.resolve()), "audit_batch_sha256": sha(args.batch),
            "audit_tool_sha256": sha(__file__), "pairwise_audit_tool_sha256": sha(Path(__file__).with_name("audit_supermimic_precision.py")),
            "source_sha256": {name: sha(source_root / f"{name}.py") for name in ("models", "imagination", "env_adapter")},
            "diagnostic_iteration": snapshot.get("iteration"), "trained_head_maximum_absolute_parameter": heads,
            "world_architecture": dict(snapshot["world_architecture"], mixed_precision="none"),
            "collection": payload["collection"], "batch_size": batch["actions"].shape[0],
            "compile": {"backend": "inductor", "mode": "default", "fullgraph": True, "dynamic": False,
                "options": {"triton.cudagraphs": False}, "compiled_region": "Original bound WorldModel.forward only",
                "wrapper_creation_seconds": wrapper_seconds, "cold_workload_seconds": cold_seconds,
                "cold_workloads": cold, "extra_warmup_per_variant_workload": args.warmup,
                "warmup_workloads": {key: latency_summary(values) for key, values in warmups.items()},
                "cold_and_warmup_seconds": warmup_complete-cold_started,
                "counters_before_cold": counters_before, "counters_after_warmup": counters_after_warmup,
                "counters_after_timing": counters_after_timing, "steady_state_unique_graph_count_stable": cache_stable,
                "cold_cost_scope": "First complete workload includes compilation and first execution, not a pure compiler-only duration. State/input requires_grad and frozen world-parameter variants are all exercised before timing. Inductor disk caches may reduce cold cost; this tool never deletes shared caches."},
            "arithmetic": {"all_model_parameters_actor_critic_physics_losses": "torch.float32", **matmul_precision_metadata(),
                "tf32_compatibility": tf32_compatibility,
                "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                "device": str(device), "device_name": torch.cuda.get_device_name(device),
                "scope": "Same configured matmul backend in both variants, including TF32 if training enables it. Compiler fusion may change FP32 reduction order; numerical checks are still required."},
            "prediction_horizons": horizons, "actor_bptt": dict(gradient, gamma=gamma, lam=lam),
            "benchmarks": {"repeats_per_variant": args.repeats, "workloads": measurements,
                "scope": "Fixed resident data. Each repeat clears gradients outside timing, then times complete eager Python world8 or imagination32 forward+backward; only compiled-world forward differs. No optimizer, replay sampling, environment or AMP style reward. Order alternates per repeat."},
            "suggested_quality_gates": numerical_gates(horizons, gradient),
            "audit_seconds": time.perf_counter()-started_all,
            "scope": "Offline kernel-fusion and gradient-equivalence diagnostic. True prediction errors use recorded actions; actor/imagination uses the same FP32 teacher and fixed four-field external goal sequence without future physics. This does not establish end-to-end learning or real-policy quality."}
        output.write_text(json.dumps(report, indent=2, allow_nan=False))
        progress_write(progress_path, progress, "completed", report_path=str(output), report_sha256=sha(output))
        print(json.dumps({"report": str(output), "report_sha256": sha(output),
            "gradient_relative_l2": gradient["gradient_relative_l2"], "gradient_cosine": gradient["gradient_cosine"],
            "world8_speed_ratio": measurements["world8"]["median_wall_speed_ratio_eager_over_compiled"],
            "imagination32_speed_ratio": measurements["imagination32"]["median_wall_speed_ratio_eager_over_compiled"],
            "cold_and_warmup_seconds": report["compile"]["cold_and_warmup_seconds"],
            "steady_state_cache_stable": cache_stable,
            "numerical_gates_passed": report["suggested_quality_gates"]["all_suggested_numerical_gates_passed"]},
            indent=2, allow_nan=False))
    except Exception as error:
        progress_write(progress_path, progress, "failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
