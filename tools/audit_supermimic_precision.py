"""Frozen-weight CUDA FP32 versus world-MLP BF16 numerical and latency audit.

Reuse a trained diagnostic and the existing fixed 32-step real-action corpus.
No simulator, optimizer, normalizer fitting or policy update is run. A passing
numerical audit does not establish equal real-task learning or policy quality.
"""
import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import time

from audit_supermimic_model import canonical_metadata, decode_slice, projection_adapter, sha


EXOGENOUS_KEYS = {"command_b", "is_heading_env", "is_standing_env", "time_left"}


def tensor_dict_sha256(values):
    """Hash values rather than torch.save's storage identities or ZIP timestamps."""
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def rng_state(device):
    import torch
    return {"cpu": torch.random.get_rng_state().clone(),
            "cuda": torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None}


def same_rng(left, right):
    import torch
    return torch.equal(left["cpu"], right["cpu"]) and (
        left["cuda"] is None and right["cuda"] is None or
        left["cuda"] is not None and right["cuda"] is not None and
        torch.equal(left["cuda"], right["cuda"]))


def finite_scalar(value):
    result = float(value)
    return result if math.isfinite(result) else None


def matmul_precision_metadata():
    """Read backend metadata without mixing legacy/new precision-setting APIs."""
    import torch
    result = {}
    for name, query in (("fp32_matmul_precision", lambda: torch.backends.cuda.matmul.fp32_precision),
                        ("float32_matmul_precision_setting", torch.get_float32_matmul_precision)):
        try:
            result[name] = query()
        except (AttributeError, RuntimeError) as error:
            result[name] = None
            result[name + "_unavailable_reason"] = str(error)
    return result


def validate_inputs(snapshot, payload, policy_sha):
    import torch
    if snapshot.get("joint_consistency", False):
        raise ValueError("This arithmetic audit supports conditional-v2 without joint projection; use the joint-aware model audit for P1 snapshots")
    if payload.get("format") != "husky-supermimic-real-action-audit-v1":
        raise ValueError("Use an existing husky-supermimic-real-action-audit-v1 corpus")
    if payload.get("teacher_sha256") != policy_sha:
        raise ValueError("Requested policy differs from the fixed audit collection teacher")
    for field in ("layout", "history_terms", "critic_terms", "qpos_quaternion_slices"):
        if canonical_metadata(snapshot[field]) != canonical_metadata(payload[field]):
            raise ValueError(f"Diagnostic/corpus {field} mismatch")
    timing = payload["collection"]
    if not timing.get("command_config") or timing.get("board_quaternion_slice") is None:
        raise ValueError("Conditional projection needs command_config and board_quaternion_slice")
    if not payload.get("physics_groups"):
        raise ValueError("Corpus must include physical groups that separate metres, radians and velocities")
    batch, dimensions = payload["batch"], snapshot["world_architecture"]
    n, horizon, action_dim = batch["actions"].shape
    if (n < 1 or horizon < 32 or action_dim != dimensions["action_dim"] or
            batch["states"].shape != (n, horizon+1, dimensions["state_dim"]) or
            batch["context"].shape != (n, horizon, dimensions["context_dim"]) or
            any(batch[key].shape != (n, horizon) for key in ("rewards", "dones", "valid", "terminated"))):
        raise ValueError("Need complete, dimension-compatible fixed32 trajectories")
    if bool(batch["dones"][:, :-1].any()):
        raise ValueError("Audit windows must not cross resets")
    if any(not bool(torch.isfinite(value).all()) for value in batch.values()):
        raise FloatingPointError("Corpus contains nonfinite real data")
    weights = snapshot["world"]
    head_sizes = {}
    for head in ("delta_head", "reward_head"):
        maximum = max(float(weights[f"{head}.{name}"].abs().max()) for name in ("weight", "bias"))
        if maximum == 0:
            raise ValueError(f"{head} is zero initialized; use a nonzero trained world model")
        head_sizes[head] = maximum
    for name in ("state", "action", "delta", "reward"):
        if float(weights[f"{name}_norm.count"]) <= 0:
            raise ValueError(f"{name} normalizer has never seen real training data")
    if any(not bool(torch.isfinite(value).all()) for value in weights.values()):
        raise FloatingPointError("Diagnostic contains nonfinite world weights/buffers")
    return head_sizes


def load_policy(checkpoint, adapter, action_dim, device):
    """Infer layer sizes from the original HUSKY ELU teacher; load strictly."""
    import torch
    from rsl_rl.modules import ActorCritic
    weights = checkpoint["model_state_dict"]

    def linear_weights(prefix):
        names = sorted((name for name in weights if re.fullmatch(prefix + r"\.\d+\.weight", name)),
                       key=lambda name: int(name.split(".")[1]))
        layers = [weights[name] for name in names]
        if len(layers) < 2 or any(layer.ndim != 2 for layer in layers):
            raise ValueError(f"Unsupported {prefix} checkpoint architecture")
        return layers

    actor, critic = linear_weights("actor"), linear_weights("critic")
    observations = adapter.state_to_observations(torch.zeros(1, max(sl.stop for sl in adapter.layout.values())))
    if (actor[0].shape[1] != observations["policy"].shape[-1] or actor[-1].shape[0] != action_dim or
            critic[0].shape[1] != observations["critic"].shape[-1] or critic[-1].shape[0] != 1):
        raise ValueError("Teacher does not match adapter actor/critic/action dimensions")
    with redirect_stdout(io.StringIO()):
        policy = ActorCritic(observations, {"policy": ["policy"], "critic": ["critic"]}, action_dim,
            actor_obs_normalization=any(name.startswith("actor_obs_normalizer.") for name in weights),
            critic_obs_normalization=any(name.startswith("critic_obs_normalizer.") for name in weights),
            actor_hidden_dims=tuple(layer.shape[0] for layer in actor[:-1]),
            critic_hidden_dims=tuple(layer.shape[0] for layer in critic[:-1]), activation="elu",
            noise_std_type="log" if "log_std" in weights else "scalar")
    policy.load_state_dict(weights, strict=True)
    policy.to(device=device, dtype=torch.float32).eval()
    if any(not bool(torch.isfinite(value).all()) for value in policy.state_dict().values()):
        raise FloatingPointError("Policy contains nonfinite weights/buffers")
    return policy


def fixed_exogenous(adapter, states):
    values = []
    for step in range(32):
        fields = adapter.exogenous_from_state(states[:, step+1].detach())
        if set(fields) - EXOGENOUS_KEYS:
            raise AssertionError("Future physical state may not be supplied to projection")
        values.append({key: value.detach().clone() for key, value in fields.items()})
    return values


def real_masks(batch, horizon):
    import torch
    active = torch.ones_like(batch["dones"][:, :1], dtype=torch.bool)
    before = []
    for step in range(horizon):
        before.append(active[:, 0])
        active = active & ~batch["dones"][:, step:step+1].bool()
    reward_mask = torch.stack(before, 1)
    return reward_mask & batch["valid"][:, :horizon].bool(), reward_mask


def recorded_action_rollout(world, adapter, batch, commands):
    """Free-run all physical fields; condition only on four external goal fields."""
    import torch
    state = batch["states"][:, 0]
    states, rewards, failures = [], [], []
    with torch.inference_mode():
        for step in range(32):
            prediction = world(state, batch["actions"][:, step], batch["context"][:, step])
            state = adapter.project(state, prediction["next_state"], batch["actions"][:, step],
                                    exogenous=commands[step])
            if any(value.dtype != torch.float32 for value in prediction.values()) or state.dtype != torch.float32:
                raise AssertionError("Physical predictions, rewards, logits and projected state must remain FP32")
            states.append(state.cpu())
            rewards.append(prediction["reward"].cpu())
            failures.append(prediction["termination_probability"].cpu())
    return {"states": torch.stack(states, 1), "rewards": torch.stack(rewards, 1),
            "failure_probability": torch.stack(failures, 1)}


def error_statistics(error, mask, scale=None):
    """Raw units and fixed training scales; p95 is over scalar absolute residuals."""
    import torch
    selected = error[mask].double()
    nonfinite = int((~torch.isfinite(selected)).sum())
    result = {"transitions": int(mask.sum()), "dimensions": error.shape[-1],
              "scalar_count": selected.numel(), "nonfinite_scalars": nonfinite,
              "raw_rmse": None, "raw_mae": None, "raw_absolute_p95": None,
              "delta_normalized_rmse": None}
    if selected.numel() and not nonfinite:
        result.update(raw_rmse=float(selected.square().mean().sqrt()),
                      raw_mae=float(selected.abs().mean()),
                      raw_absolute_p95=float(torch.quantile(selected.abs().reshape(-1), .95)))
        if scale is not None:
            result["delta_normalized_rmse"] = float((selected / scale.double()).square().mean().sqrt())
    return result


def prediction_comparison(world, adapter, payload, predictions):
    import torch
    source = payload["batch"]
    true = {"states": source["states"][:, 1:33].cpu(), "rewards": source["rewards"][:, :32].cpu(),
            "failure_probability": source["terminated"][:, :32].float().cpu()}
    state_mask, reward_mask = [value.cpu() for value in real_masks(source, 32)]
    delta_scale = world.delta_norm.std.detach().cpu()
    reward_scale = world.reward_norm.std.detach().cpu()
    groups = {}
    for name in ("policy", "critic", "qpos", "qvel"):
        if name in adapter.layout:
            groups[name] = (adapter.layout[name], "scaled observation units" if name in ("policy", "critic") else
                            "mixed units; consult physical subgroups")
    for name, metadata in payload["physics_groups"].items():
        indices = torch.tensor(metadata["indices"], dtype=torch.long) + adapter.layout[metadata["field"]].start
        groups[name] = (indices, metadata["units"])
    results = {}
    for horizon in (1, 8, 32):
        pairs = (("fp32_true_error", predictions["fp32"], true),
                 ("bf16_true_error", predictions["bf16"], true),
                 ("bf16_minus_fp32", predictions["bf16"], predictions["fp32"]))
        summaries = {}
        for label, predicted, reference in pairs:
            state_error = predicted["states"][:, :horizon] - reference["states"][:, :horizon]
            reward_error = (predicted["rewards"][:, :horizon] - reference["rewards"][:, :horizon])[..., None]
            failure_error = (predicted["failure_probability"][:, :horizon] - reference["failure_probability"][:, :horizon])[..., None]
            selections = {"across_horizon": slice(0, horizon), "at_final_step": slice(horizon-1, horizon)}
            summary = {}
            for location, selection in selections.items():
                mask = state_mask[:, selection]
                metrics = {}
                for name, (indices, units) in groups.items():
                    metrics[name] = dict(error_statistics(state_error[:, selection, indices], mask, delta_scale[indices]),
                                         raw_units=units)
                metrics["task_reward"] = dict(error_statistics(reward_error[:, selection],
                    reward_mask[:, selection], reward_scale), raw_units="HUSKY task reward per real control step")
                metrics["physical_failure_probability"] = dict(error_statistics(failure_error[:, selection],
                    reward_mask[:, selection]), raw_units="probability; true target is physical termination label")
                summary[location] = metrics
            summaries[label] = summary
        results[str(horizon)] = {"steps": horizon, "seconds": horizon * payload["collection"]["step_dt"],
                                 **summaries}
    return results


def actor_gradient(policy, world, adapter, batch, commands, *, gamma, lam):
    import torch
    from mjlab_husky.supermimic.imagination import imagine
    counter = 0

    def command_sampler(_detached_current_state):
        nonlocal counter
        fields = commands[counter]
        counter += 1
        return fields

    policy.zero_grad(set_to_none=True)
    world.zero_grad(set_to_none=True)
    imagined = imagine(policy, world, batch["states"][:, 0], adapter.state_to_observations,
        horizon=32, gamma=gamma, lam=lam, context=batch["context"][:, :32],
        project=adapter.project, exogenous_sampler=command_sampler)
    if counter != 32:
        raise AssertionError("External command replay did not consume exactly 32 steps")
    imagined["actor_loss"].backward()
    parameters = list(policy.actor.named_parameters())
    gradients = torch.cat([(parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)).detach().flatten()
                           for _, parameter in parameters]).float().cpu()
    if any(parameter.grad is not None for parameter in world.parameters()):
        raise AssertionError("Actor BPTT unexpectedly accumulated world weight gradients")
    metadata = {name: finite_scalar(imagined[name].detach()) for name in
                ("objective", "actor_loss", "reward_objective", "bootstrap_objective")}
    metadata.update(gradient_elements=gradients.numel(), nonfinite_gradient_elements=int((~torch.isfinite(gradients)).sum()),
                    missing_gradient_parameters=[name for name, parameter in parameters if parameter.grad is None],
                    actor_dtype=str(next(policy.actor.parameters()).dtype), horizon=32,
                    scope="Raw -objective gradient before trust regularization, clipping or fallback. Task reward + frozen teacher critic only; AMP style reward is omitted.")
    policy.zero_grad(set_to_none=True)
    return gradients, metadata


def gradient_comparison(gradients, summaries):
    import torch
    left, right = gradients["fp32"].double(), gradients["bf16"].double()
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    norm = float(left.norm()) if finite else None
    difference = float((right-left).norm()) if finite else None
    other_norm = float(right.norm()) if finite else None
    objective_fp32, objective_bf16 = summaries["fp32"]["objective"], summaries["bf16"]["objective"]
    delta_objective = objective_bf16-objective_fp32 if objective_fp32 is not None and objective_bf16 is not None else None
    return {"modes": summaries, "all_gradients_finite": finite,
            "fp32_gradient_l2": norm, "bf16_gradient_l2": other_norm,
            "gradient_difference_l2": difference,
            "gradient_relative_l2": difference/norm if norm is not None and norm > 0 else None,
            "gradient_cosine": float(torch.dot(left, right) / (left.norm()*right.norm()))
                if finite and norm > 0 and other_norm > 0 else None,
            "objective_bf16_minus_fp32": delta_objective,
            "objective_relative_absolute_difference": abs(delta_objective)/abs(objective_fp32)
                if delta_objective is not None and abs(objective_fp32) > 0 else None,
            "near_zero_guard": "Relative metrics are undefined at zero. Interpret the reported absolute gradient/objective differences alongside relative gates."}


def benchmark_world_loss(worlds, adapter, batch, *, warmup, repeats, device):
    """Same8-step full batch; time forward+backward and never take a step."""
    import torch
    from mjlab_husky.supermimic.models import world_model_loss
    timings = {mode: [] for mode in worlds}
    kernel_timings = {mode: [] for mode in worlds}
    losses = {}

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def run(mode, timed):
        world = worlds[mode]
        world.zero_grad(set_to_none=True)
        begin = torch.cuda.Event(enable_timing=True) if timed and device.type == "cuda" else None
        end = torch.cuda.Event(enable_timing=True) if begin is not None else None
        synchronize()
        started = time.perf_counter()
        if begin is not None:
            begin.record()
        loss = world_model_loss(world, batch, project=adapter.project, horizon=8,
                                exogenous_from_state=adapter.exogenous_from_state)
        loss["loss"].backward()
        if end is not None:
            end.record()
        synchronize()
        elapsed = time.perf_counter()-started
        if timed:
            timings[mode].append(elapsed)
            if begin is not None:
                kernel_timings[mode].append(begin.elapsed_time(end) / 1000)
        losses[mode] = {key: finite_scalar(value.detach()) for key, value in loss.items()}

    for _ in range(warmup):
        for mode in worlds:
            run(mode, False)
    for repeat in range(repeats):
        for mode in (tuple(worlds) if repeat % 2 == 0 else tuple(reversed(worlds))):
            run(mode, True)
    summaries = {}
    for mode, values in timings.items():
        finite_gradients = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                               for parameter in worlds[mode].parameters())
        summaries[mode] = {"wall_seconds": values, "wall_mean_seconds": statistics.mean(values),
            "wall_median_seconds": statistics.median(values), "wall_min_seconds": min(values),
            "cuda_event_seconds": kernel_timings[mode], "last_losses": losses[mode],
            "last_gradients_finite": finite_gradients}
        worlds[mode].zero_grad(set_to_none=True)
    return {"warmup_per_mode": warmup, "repeats_per_mode": repeats, "window_steps": 8,
            "batch_size": batch["actions"].shape[0], "modes": summaries,
            "median_wall_speed_ratio_fp32_over_bf16": statistics.median(timings["fp32"])/statistics.median(timings["bf16"]),
            "scope": "Fixed resident batch, exact conditional projection, world_model_loss forward+backward. No optimizer, replay sampling, environment collection or normalizer fitting. Gradient clearing is outside timing; precision order alternates each measured repeat."}


def suggested_quality_gates(horizons, gradients):
    """Transparent starting thresholds, not established real-task tolerances."""
    limits = {"actor_gradient_cosine_minimum": .99, "actor_gradient_relative_l2_maximum": .05,
              "true_error_rmse_relative_increase_maximum": .05,
              "true_error_absolute_p95_relative_increase_maximum": .10,
              "true_error_absolute_slack": 1e-8}
    checks = []
    for horizon, report in horizons.items():
        for location in ("across_horizon", "at_final_step"):
            reference = report["fp32_true_error"][location]
            for group, baseline in reference.items():
                if group == "physical_failure_probability":
                    continue
                measured = report["bf16_true_error"][location][group]
                for metric, relative in (("raw_rmse", .05), ("raw_absolute_p95", .10)):
                    left, right = baseline[metric], measured[metric]
                    passed = (left is not None and right is not None and
                              right <= left*(1+relative)+limits["true_error_absolute_slack"])
                    checks.append({"horizon": int(horizon), "location": location, "group": group,
                                   "metric": metric, "fp32": left, "bf16": right, "passed": passed})
    relative, cosine = gradients["gradient_relative_l2"], gradients["gradient_cosine"]
    gradient_pass = (gradients["all_gradients_finite"] and relative is not None and cosine is not None and
                     relative <= .05 and cosine >= .99)
    return {"suggested_limits": limits, "actor_gradient_passed": gradient_pass,
            "true_error_checks": checks, "all_true_error_checks_passed": all(item["passed"] for item in checks),
            "all_suggested_numerical_gates_passed": gradient_pass and all(item["passed"] for item in checks),
            "scope": "Provisional numerical gates for this same-weight corpus only. Review FP32 absolute physical errors first: matching an inaccurate FP32 model is insufficient. No trained-policy quality or equal-convergence-speed conclusion follows; matched real-data training and real-physics checks remain necessary."}


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
        parser.error("Use at least one warmup and two interleaved timed repeats per mode")
    import torch
    from mjlab.utils.torch import configure_torch_backends
    from mjlab_husky.supermimic.models import WorldModel
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("The real BF16 autocast branch requires CUDA; CPU fallback cannot validate BF16")
    with torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported():
            raise ValueError("Requested CUDA device does not support BF16")
    torch.cuda.set_device(device)
    configure_torch_backends()
    torch.set_num_threads(1)
    args.out = args.out.resolve()
    output = args.out / "data" / f"{args.label}_precision_audit.json"
    if output.exists():
        raise FileExistsError("Preserve existing reports; choose a new label")
    snapshot = torch.load(args.model_diagnostic, map_location="cpu", weights_only=False)
    payload = torch.load(args.batch, map_location="cpu", weights_only=False)
    policy_sha = sha(args.policy)
    head_sizes = validate_inputs(snapshot, payload, policy_sha)
    adapter = projection_adapter(snapshot, payload["collection"])
    batch = {name: value.to(device) for name, value in payload["batch"].items()}
    commands = fixed_exogenous(adapter, batch["states"])
    initialization_started = time.perf_counter()
    original_rng = rng_state(device)
    with torch.random.fork_rng(devices=[device.index if device.index is not None else torch.cuda.current_device()]):
        worlds = {}
        initial_hashes = {}
        for mode, precision in (("fp32", "none"), ("bf16", "bf16")):
            architecture = dict(snapshot["world_architecture"], mixed_precision=precision)
            world = WorldModel(**architecture)
            world.load_state_dict(snapshot["world"], strict=True)
            worlds[mode] = world.to(device=device, dtype=torch.float32).eval()
            initial_hashes[mode] = tensor_dict_sha256(world.state_dict())
        if initial_hashes["fp32"] != initial_hashes["bf16"]:
            raise AssertionError("Arithmetic variants do not have identical weights and normalizers")
        policy = load_policy(torch.load(args.policy, map_location="cpu", weights_only=False), adapter,
                             worlds["fp32"].action_dim, device)
        policy_hash = tensor_dict_sha256(policy.state_dict())
        forward_dtypes = {}
        for mode, world in worlds.items():
            def record_dtype(_module, _inputs, value, key=mode):
                forward_dtypes[key] = str(value.dtype)
            handle = world.delta_head.register_forward_hook(record_dtype)
            with torch.inference_mode():
                world(batch["states"][:, 0], batch["actions"][:, 0], batch["context"][:, 0])
            handle.remove()
        if forward_dtypes != {"fp32": "torch.float32", "bf16": "torch.bfloat16"}:
            raise AssertionError(f"Requested arithmetic did not execute: {forward_dtypes}")
        initialized = time.perf_counter()
        comparison_rng = rng_state(device)
        predictions = {mode: recorded_action_rollout(world, adapter, batch, commands) for mode, world in worlds.items()}
        horizons = prediction_comparison(worlds["fp32"], adapter, payload, predictions)
        gradients, gradient_summaries = {}, {}
        gamma, lam = snapshot.get("config", {}).get("gamma", .99), snapshot.get("config", {}).get("lam", .95)
        for mode, world in worlds.items():
            gradients[mode], gradient_summaries[mode] = actor_gradient(policy, world, adapter, batch, commands, gamma=gamma, lam=lam)
        gradient_report = gradient_comparison(gradients, gradient_summaries)
        benchmark = benchmark_world_loss(worlds, adapter, batch, warmup=args.warmup, repeats=args.repeats, device=device)
        final_hashes = {mode: tensor_dict_sha256(world.state_dict()) for mode, world in worlds.items()}
        if final_hashes != initial_hashes or tensor_dict_sha256(policy.state_dict()) != policy_hash:
            raise AssertionError("Audit changed model/policy weights or normalization buffers")
        if not same_rng(comparison_rng, rng_state(device)):
            raise AssertionError("Fixed arithmetic comparison consumed RNG")
    if not same_rng(original_rng, rng_state(device)):
        raise AssertionError("Audit did not restore construction-time RNG")
    model_source = Path(__file__).resolve().parents[1] / "src/mjlab_husky/supermimic"
    report = {"format": "husky-supermimic-precision-audit-v1", "label": args.label, "created_at": time.time(),
        "algorithm_updates": 0, "optimizer_steps": 0, "same_world_weights_and_normalizers": True,
        "world_and_policy_state_unchanged": True, "rng_unchanged": True,
        "model_diagnostic_path": str(args.model_diagnostic.resolve()), "model_diagnostic_sha256": sha(args.model_diagnostic),
        "world_state_sha256": initial_hashes["fp32"], "teacher_path": str(args.policy.resolve()),
        "teacher_sha256": policy_sha, "policy_state_sha256": policy_hash,
        "audit_batch_path": str(args.batch.resolve()), "audit_batch_sha256": sha(args.batch),
        "audit_tool_sha256": sha(__file__), "projection_audit_tool_sha256": sha(Path(__file__).with_name("audit_supermimic_model.py")),
        "source_sha256": {name: sha(model_source / f"{name}.py") for name in ("models", "imagination", "env_adapter")},
        "trained_head_maximum_absolute_parameter": head_sizes,
        "diagnostic_iteration": snapshot.get("iteration"), "diagnostic_dynamics_protocol": snapshot.get("dynamics_protocol"),
        "world_architecture": snapshot["world_architecture"], "collection": payload["collection"],
        "arithmetic": {"world_head_forward_dtypes": forward_dtypes,
            "physical_state_reward_logit_and_loss_dtype": "torch.float32", "actor_and_critic_dtype": "torch.float32",
            **matmul_precision_metadata(),
            "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "device": str(device), "device_name": torch.cuda.get_device_name(device),
            "scope": "Training backend settings retained, including TF32 if enabled. FP32 denotes FP32 tensors and this recorded matmul backend, not necessarily IEEE-only GEMMs. Only world trunk/heads enter BF16 autocast."},
        "projection_protocol": {"name": "conditional-commands-v2", "exogenous_fields": sorted(EXOGENOUS_KEYS),
            "scope": "Same recorded external command sequence in both variants and actor BPTT. No future qpos/qvel, observations, heading_ref or physical target is fed into prediction. Heading reference is reconstructed from predicted board yaw on legal events."},
        "prediction_horizons": horizons, "actor_bptt": dict(gradient_report, gamma=gamma, lam=lam),
        "world_loss_benchmark": benchmark, "suggested_quality_gates": suggested_quality_gates(horizons, gradient_report),
        "initialization_seconds": initialized-initialization_started,
        "audit_seconds": time.perf_counter()-initialized,
        "scope": "Same-weight numerical sensitivity and isolated fixed-data latency. Prediction truth errors use recorded real actions; actor BPTT uses policy-generated actions and task reward + teacher critic without AMP reward. Absolute FP32 physical errors are reported by unit-separated groups. No real policy quality or convergence-speed claim is made."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"report": str(output), "report_sha256": sha(output),
        "actor_gradient_relative_l2": gradient_report["gradient_relative_l2"],
        "actor_gradient_cosine": gradient_report["gradient_cosine"],
        "objective_difference": gradient_report["objective_bf16_minus_fp32"],
        "world_loss_median_wall_speed_ratio": benchmark["median_wall_speed_ratio_fp32_over_bf16"],
        "suggested_numerical_gates_passed": report["suggested_quality_gates"]["all_suggested_numerical_gates_passed"]},
        indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
