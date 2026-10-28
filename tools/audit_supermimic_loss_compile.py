"""Frozen-weight whole-world-loss8 Inductor audit on the existing fixed corpus.

Compile only the original world_model_loss closure, including the original
conditional adapter projection. Return its entire loss dictionary and run its
backward. World.forward, actor, imagined rollout and optimizer remain ordinary
eager code. No optimizer or simulator is created. Launch in a fresh process
with TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 on pinned PyTorch2.10; an external300s
timeout may bound compilation. Progress is saved before expensive work.
"""
import argparse
import json
from pathlib import Path
import re
import statistics
import time

from audit_supermimic_compile import (
    compiler_counters, progress_write, run_imagined_backward,
    tf32_compatibility_metadata,
)
from audit_supermimic_precision import (
    finite_scalar, fixed_exogenous, load_policy, matmul_precision_metadata,
    rng_state, same_rng, tensor_dict_sha256, validate_inputs,
)
from audit_supermimic_model import projection_adapter, sha


def contiguous_window(batch, horizon=8):
    """Replay production samples have H+1/H contiguous storage, not H32 strides."""
    window = {name: value[:, :horizon+1 if name == "states" else horizon].contiguous()
              for name, value in batch.items()}
    if any(not value.is_contiguous() for value in window.values()):
        raise AssertionError("Every resident input must have production-like contiguous H8 storage")
    return window


def loss_function(world, adapter):
    from mjlab_husky.supermimic.models import world_model_loss

    def original_loss(batch):
        return world_model_loss(world, batch, project=adapter.project,
            exogenous_from_state=adapter.exogenous_from_state, horizon=8)

    return original_loss


def inductor_options(*, max_fusion_size=None, epilogue_fusion=None):
    import torch._inductor as inductor
    options = {"triton.cudagraphs": False}
    if max_fusion_size is not None:
        if max_fusion_size < 1:
            raise ValueError("max_fusion_size must be positive")
        options["max_fusion_size"] = int(max_fusion_size)
    if epilogue_fusion is not None:
        options["epilogue_fusion"] = bool(epilogue_fusion)
    unknown = set(options) - set(inductor.list_options())
    if unknown:
        raise ValueError(f"Installed PyTorch does not support Inductor options: {sorted(unknown)}")
    return options


def compile_loss(function, *, backend="inductor", max_fusion_size=None, epilogue_fusion=None):
    import torch
    arguments = {"backend": backend, "fullgraph": True, "dynamic": False}
    if backend == "inductor":
        arguments["options"] = inductor_options(max_fusion_size=max_fusion_size, epilogue_fusion=epilogue_fusion)
    return torch.compile(function, **arguments)


def parameter_gradients(world):
    import torch
    return {name: parameter.grad.detach().float().cpu().clone() if parameter.grad is not None else None
            for name, parameter in world.named_parameters()}


def vector_difference(reference, measured):
    import torch
    reference, measured = reference.double().flatten(), measured.double().flatten()
    if reference.shape != measured.shape:
        raise ValueError("Gradient shape mismatch")
    nonfinite = int((~torch.isfinite(reference)).sum() + (~torch.isfinite(measured)).sum())
    result = {"elements": reference.numel(), "nonfinite_elements": nonfinite,
              "reference_l2": None, "measured_l2": None, "difference_l2": None,
              "relative_l2": None, "cosine": None, "maximum_absolute_difference": None}
    if not nonfinite:
        left, right, difference = float(reference.norm()), float(measured.norm()), float((measured-reference).norm())
        result.update(reference_l2=left, measured_l2=right, difference_l2=difference,
                      relative_l2=difference/left if left > 0 else None,
                      cosine=float(torch.dot(reference, measured)/(reference.norm()*measured.norm()))
                          if left > 0 and right > 0 else None,
                      maximum_absolute_difference=float((measured-reference).abs().max()) if reference.numel() else 0.)
    return result


def gradient_comparison(reference, measured):
    import torch
    if tuple(reference) != tuple(measured):
        raise ValueError("Compared parameter names/order differ")
    missing = {"reference": [name for name, value in reference.items() if value is None],
               "measured": [name for name, value in measured.items() if value is None]}
    per_parameter = {name: vector_difference(reference[name], measured[name])
                     for name in reference if reference[name] is not None and measured[name] is not None}
    if per_parameter:
        summary = vector_difference(torch.cat([reference[name].flatten() for name in per_parameter]),
                                    torch.cat([measured[name].flatten() for name in per_parameter]))
    else:
        summary = {"elements": 0, "nonfinite_elements": 0, "relative_l2": None, "cosine": None}
    return {"all_parameter_gradients_present": not missing["reference"] and not missing["measured"],
            "missing": missing, "all_world_parameters": summary, "per_parameter": per_parameter}


def loss_comparison(reference, measured):
    if set(reference) != set(measured):
        raise ValueError("Compiled closure did not return the full original loss dictionary")
    result = {}
    for name, left in reference.items():
        right = measured[name]
        difference = right-left if left is not None and right is not None else None
        result[name] = {"eager": left, "compiled": right, "compiled_minus_eager": difference,
                       "relative_absolute_difference": abs(difference)/abs(left)
                           if difference is not None and left != 0 else None,
                       "suggested_gate_passed": difference is not None and abs(difference) <= 1e-7+1e-4*abs(left)}
    return result


def timed_backward(function, world, batch, device):
    import torch
    world.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    else:
        begin = end = None
    started = time.perf_counter()
    if begin is not None:
        begin.record()
    losses = function(batch)
    losses["loss"].backward()
    if end is not None:
        end.record()
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter()-started
    if any(value.dtype != torch.float32 for value in losses.values()):
        raise AssertionError("Full loss dictionary must retain FP32 arithmetic")
    finite_gradients = all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                           for parameter in world.parameters())
    return {"wall_seconds": elapsed,
            "cuda_event_seconds": begin.elapsed_time(end)/1000 if begin is not None else None,
            "losses": {name: finite_scalar(value.detach()) for name, value in losses.items()},
            "all_parameter_gradients_present_and_finite": finite_gradients}


def summarize_samples(samples):
    values = [item["wall_seconds"] for item in samples]
    return {"wall_seconds": values, "wall_median_seconds": statistics.median(values),
            "wall_mean_seconds": statistics.mean(values), "wall_min_seconds": min(values),
            "cuda_event_seconds": [item["cuda_event_seconds"] for item in samples],
            "all_gradients_present_and_finite": all(item["all_parameter_gradients_present_and_finite"] for item in samples),
            "last_losses": samples[-1]["losses"]}


def unique_graph_count(counters):
    return counters.get("stats", {}).get("unique_graphs")


def counts_equal(before, after):
    left, right = unique_graph_count(before), unique_graph_count(after)
    return left == right if left is not None and right is not None else None


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
    parser.add_argument("--max-fusion-size", type=int,
                        help="Optional Inductor fusion node limit; omitted retains the installed default")
    parser.add_argument("--epilogue-fusion", action=argparse.BooleanOptionalAction, default=None,
                        help="Optional Inductor epilogue fusion override; --no-epilogue-fusion disables it")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        parser.error("label must contain ASCII letters, digits, underscore or hyphen")
    if args.warmup < 1 or args.repeats < 2:
        parser.error("Use at least one warmup and two alternating timed repeats per variant")
    if args.max_fusion_size is not None and args.max_fusion_size < 1:
        parser.error("max-fusion-size must be positive")
    import torch
    from mjlab.utils.torch import configure_torch_backends
    from mjlab_husky.supermimic.models import WorldModel
    from torch._inductor import config as inductor_config
    options = inductor_options(max_fusion_size=args.max_fusion_size, epilogue_fusion=args.epilogue_fusion)
    effective_fusion = {"max_fusion_size": options.get("max_fusion_size", inductor_config.max_fusion_size),
                        "epilogue_fusion": options.get("epilogue_fusion", inductor_config.epilogue_fusion)}
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("CUDA is required for this Inductor benchmark; CPU helper tests do not establish CUDA speed")
    configure_torch_backends()
    compatibility = tf32_compatibility_metadata()
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    data_dir = args.out.resolve() / "data"
    output = data_dir / f"{args.label}_loss_compile_audit.json"
    progress_path = data_dir / f"{args.label}_loss_compile_progress.json"
    if output.exists() or progress_path.exists():
        raise FileExistsError("Choose a fresh label; preserve existing results and partial compile progress")
    snapshot = torch.load(args.model_diagnostic, map_location="cpu", weights_only=False)
    payload = torch.load(args.batch, map_location="cpu", weights_only=False)
    teacher_sha = sha(args.policy)
    head_sizes = validate_inputs(snapshot, payload, teacher_sha)
    adapter = projection_adapter(snapshot, payload["collection"])
    batch8 = {name: value.to(device) for name, value in contiguous_window(payload["batch"]).items()}
    batch32 = {name: value.to(device) for name, value in contiguous_window(payload["batch"], 32).items()}
    commands = fixed_exogenous(adapter, batch32["states"])
    gamma, lam = snapshot.get("config", {}).get("gamma", .99), snapshot.get("config", {}).get("lam", .95)
    progress = {"format": "husky-supermimic-loss-compile-progress-v1", "label": args.label,
        "algorithm_updates": 0, "optimizer_steps": 0, "started_at": time.time(),
        "model_diagnostic_sha256": sha(args.model_diagnostic), "audit_batch_sha256": sha(args.batch),
        "teacher_sha256": teacher_sha, "tf32_compatibility": compatibility,
        "inductor_options": options, "effective_fusion_settings": effective_fusion,
        "cold_workloads": {}, "warmup_workloads": {}}
    data_dir.mkdir(parents=True, exist_ok=True)
    progress_write(progress_path, progress, "constructing_worlds")
    original_rng = rng_state(device)
    started_all = time.perf_counter()
    try:
        with torch.random.fork_rng(devices=[device.index if device.index is not None else torch.cuda.current_device()]):
            worlds, initial_hashes, functions = {}, {}, {}
            for variant in ("eager", "compiled"):
                model = WorldModel(**dict(snapshot["world_architecture"], mixed_precision="none"))
                model.load_state_dict(snapshot["world"], strict=True)
                worlds[variant] = model.to(device=device, dtype=torch.float32).eval()
                initial_hashes[variant] = tensor_dict_sha256(model.state_dict())
                functions[variant] = loss_function(model, adapter)
            if initial_hashes["eager"] != initial_hashes["compiled"]:
                raise AssertionError("Variants must share exact weights and normalization buffers")
            initial_keys = tuple(worlds["compiled"].state_dict())
            wrapper_started = time.perf_counter()
            functions["compiled"] = compile_loss(functions["compiled"], max_fusion_size=args.max_fusion_size,
                                                 epilogue_fusion=args.epilogue_fusion)
            wrapper_seconds = time.perf_counter()-wrapper_started
            policy = load_policy(torch.load(args.policy, map_location="cpu", weights_only=False), adapter,
                                 worlds["eager"].action_dim, device)
            policy_hash = tensor_dict_sha256(policy.state_dict())
            audit_rng = rng_state(device)
            cold, cold_gradients = {}, {}
            counters_before = compiler_counters()
            cold_started = time.perf_counter()
            for variant in worlds:
                progress_write(progress_path, progress, f"cold_{variant}_loss8")
                cold[variant] = timed_backward(functions[variant], worlds[variant], batch8, device)
                cold_gradients[variant] = parameter_gradients(worlds[variant])
                progress["cold_workloads"] = cold
            cold_seconds = time.perf_counter()-cold_started
            losses = loss_comparison(cold["eager"]["losses"], cold["compiled"]["losses"])
            gradients = gradient_comparison(cold_gradients["eager"], cold_gradients["compiled"])
            warmups = {}
            for variant in worlds:
                progress_write(progress_path, progress, f"warmup_{variant}_loss8")
                warmups[variant] = [timed_backward(functions[variant], worlds[variant], batch8, device)
                                    for _ in range(args.warmup)]
                progress["warmup_workloads"] = {name: summarize_samples(values) for name, values in warmups.items()}
            counters_after_warmup = compiler_counters()
            warmup_complete = time.perf_counter()
            progress_write(progress_path, progress, "eager_imagination32_frozen_world",
                           counters_after_warmup=counters_after_warmup)
            compiled_world = worlds["compiled"]
            compiled_world.zero_grad(set_to_none=True)
            policy.zero_grad(set_to_none=True)
            before_requires_grad = [parameter.requires_grad for parameter in compiled_world.parameters()]
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            imagined = run_imagined_backward(policy, compiled_world, adapter, batch32, commands, gamma, lam)
            torch.cuda.synchronize(device)
            imagination_seconds = time.perf_counter()-started
            if any(parameter.grad is not None for parameter in compiled_world.parameters()):
                raise AssertionError("Eager imagination accumulated frozen world parameter gradients")
            if before_requires_grad != [parameter.requires_grad for parameter in compiled_world.parameters()]:
                raise AssertionError("Eager imagination did not restore world parameter requires_grad")
            imagination_losses = {name: finite_scalar(value.detach()) for name, value in imagined.items()}
            policy.zero_grad(set_to_none=True)
            progress_write(progress_path, progress, "compiled_loss8_after_eager_imagination")
            after_freeze = timed_backward(functions["compiled"], compiled_world, batch8, device)
            after_freeze_gradients = parameter_gradients(compiled_world)
            freeze_comparison = gradient_comparison(cold_gradients["compiled"], after_freeze_gradients)
            counters_after_freeze = compiler_counters()
            progress_write(progress_path, progress, "steady_state_alternating_loss8",
                           counters_after_freeze=counters_after_freeze)
            samples = {variant: [] for variant in worlds}
            for repeat in range(args.repeats):
                for variant in (tuple(worlds) if repeat % 2 == 0 else tuple(reversed(worlds))):
                    samples[variant].append(timed_backward(functions[variant], worlds[variant], batch8, device))
            timings = {name: summarize_samples(values) for name, values in samples.items()}
            counters_after_timing = compiler_counters()
            if initial_hashes != {name: tensor_dict_sha256(world.state_dict()) for name, world in worlds.items()}:
                raise AssertionError("Audit changed world weights or normalization buffers")
            if tuple(compiled_world.state_dict()) != initial_keys or tensor_dict_sha256(policy.state_dict()) != policy_hash:
                raise AssertionError("Audit changed model structure, policy or its normalizer")
            if not same_rng(audit_rng, rng_state(device)):
                raise AssertionError("Fixed-data audit consumed RNG")
        if not same_rng(original_rng, rng_state(device)):
            raise AssertionError("Audit did not restore construction RNG")
        overall = gradients["all_world_parameters"]
        gradient_gate = (gradients["all_parameter_gradients_present"] and overall["nonfinite_elements"] == 0 and
                         overall["relative_l2"] is not None and overall["relative_l2"] <= .001 and
                         overall["cosine"] is not None and overall["cosine"] >= .9999)
        resumed = freeze_comparison["all_world_parameters"]
        freeze_gate = (freeze_comparison["all_parameter_gradients_present"] and resumed["nonfinite_elements"] == 0 and
                       resumed["difference_l2"] is not None and resumed["difference_l2"] <= 1e-7 + 1e-7*resumed["reference_l2"])
        speed_ratio = timings["eager"]["wall_median_seconds"] / timings["compiled"]["wall_median_seconds"]
        source_root = Path(__file__).resolve().parents[1] / "src/mjlab_husky/supermimic"
        report = {"format": "husky-supermimic-loss-compile-audit-v1", "label": args.label, "created_at": time.time(),
            "algorithm_updates": 0, "optimizer_steps": 0, "world_policy_normalizers_unchanged": True,
            "state_dict_keys_unchanged": True, "rng_unchanged": True,
            "model_diagnostic_path": str(args.model_diagnostic.resolve()), "model_diagnostic_sha256": sha(args.model_diagnostic),
            "world_state_sha256": initial_hashes["eager"], "teacher_sha256": teacher_sha,
            "policy_state_sha256": policy_hash, "audit_batch_path": str(args.batch.resolve()),
            "audit_batch_sha256": sha(args.batch), "audit_tool_sha256": sha(__file__),
            "source_sha256": {name: sha(source_root / f"{name}.py") for name in ("models", "imagination", "env_adapter")},
            "diagnostic_iteration": snapshot.get("iteration"), "trained_head_maximum_absolute_parameter": head_sizes,
            "world_architecture": dict(snapshot["world_architecture"], mixed_precision="none"),
            "input_window": {"steps": 8, "batch_size": batch8["actions"].shape[0], "all_tensors_contiguous": True,
                "shapes": {name: list(value.shape) for name, value in batch8.items()},
                "strides": {name: list(value.stride()) for name, value in batch8.items()},
                "scope": "Both variants use the same fixed32 source cut to H8/H+1 states and contiguous storage, matching replay's production H8 samples."},
            "compile": {"backend": "inductor", "mode": "default", "fullgraph": True, "dynamic": False,
                "options": options, "effective_fusion_settings": effective_fusion,
                "compiled_region": "Original world_model_loss closure with horizon8, bound adapter.project and exogenous_from_state; full original loss dict returned.",
                "world_forward_wrapped_or_changed": False, "wrapper_creation_seconds": wrapper_seconds,
                "cold_workload_seconds": cold_seconds, "cold_workloads": cold,
                "extra_warmup_per_variant": args.warmup,
                "warmup_workloads": {name: summarize_samples(values) for name, values in warmups.items()},
                "cold_and_warmup_seconds": warmup_complete-cold_started,
                "counters_before_cold": counters_before, "counters_after_warmup": counters_after_warmup,
                "counters_after_freeze": counters_after_freeze, "counters_after_timing": counters_after_timing,
                "no_new_graphs_after_freeze": counts_equal(counters_after_warmup, counters_after_freeze),
                "steady_state_graph_count_stable": counts_equal(counters_after_freeze, counters_after_timing),
                "cold_scope": "First forward+backward includes compilation plus execution. Disk caches are preserved and may lower cold cost; no pure compiler-only duration is claimed."},
            "losses": losses, "world_parameter_gradients": gradients,
            "eager_freeze_resume": {"imagination_steps": 32, "imagination_seconds": imagination_seconds,
                "task_reward_and_critic_objectives": imagination_losses, "gamma": gamma, "lam": lam,
                "world_gradients_absent_during_imagination": True, "requires_grad_restored": True,
                "compiled_loss_after_freeze": after_freeze, "same_compiled_gradient_after_freeze": freeze_comparison,
                "scope": "An ordinary eager imagine32 forward+backward freezes this same world's weights, then returns to its original compiled fitting closure. Same four-field recorded external command sequence; no future physical target and no AMP callback."},
            "timing": {"repeats_per_variant": args.repeats, "variants": timings,
                "median_wall_speed_ratio_eager_over_compiled": speed_ratio,
                "scope": "Same resident contiguous H8 data, full original loss forward+backward, alternating variant order. Gradient zeroing, finite checks and host metrics transfers are outside timing. No optimizer or replay/simulator cost."},
            "arithmetic": {"dtype": "torch.float32", **matmul_precision_metadata(), "tf32_compatibility": compatibility,
                "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                "device": str(device), "device_name": torch.cuda.get_device_name(device)},
            "suggested_quality_gates": {"loss_relative_tolerance": 1e-4, "loss_absolute_tolerance": 1e-7,
                "gradient_relative_l2_maximum": .001, "gradient_cosine_minimum": .9999,
                "all_losses_passed": all(item["suggested_gate_passed"] for item in losses.values()),
                "all_world_parameters_gradient_passed": gradient_gate, "post_freeze_gradient_passed": freeze_gate,
                "all_suggested_numerical_gates_passed": gradient_gate and freeze_gate and all(item["suggested_gate_passed"] for item in losses.values()),
                "scope": "Provisional numerical tolerances only. See all parameter-level differences and absolute gradient norms; undefined relative/cosine metrics at zero are not an automatic pass."},
            "audit_seconds": time.perf_counter()-started_all,
            "scope": "Offline loss-fusion latency and gradient contract only. This report does not establish equal real-policy quality or end-to-end learning/convergence speed."}
        output.write_text(json.dumps(report, indent=2, allow_nan=False))
        progress_write(progress_path, progress, "completed", report_path=str(output), report_sha256=sha(output))
        print(json.dumps({"report": str(output), "report_sha256": sha(output), "world_gradient_relative_l2": overall["relative_l2"],
            "world_gradient_cosine": overall["cosine"], "world8_speed_ratio": speed_ratio,
            "cold_and_warmup_seconds": report["compile"]["cold_and_warmup_seconds"],
            "no_new_graphs_after_freeze": report["compile"]["no_new_graphs_after_freeze"],
            "steady_state_cache_stable": report["compile"]["steady_state_graph_count_stable"],
            "numerical_gates_passed": report["suggested_quality_gates"]["all_suggested_numerical_gates_passed"]}, indent=2, allow_nan=False))
    except Exception as error:
        progress_write(progress_path, progress, "failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
