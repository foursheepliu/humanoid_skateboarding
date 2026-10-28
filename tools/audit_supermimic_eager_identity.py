"""Save FP32 eager rollout and actor gradients for fresh-process comparisons.

Run this worker once with the original environment and once with an externally
supplied TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1. It never sets math environment
variables, compiles code, creates an optimizer or modifies model parameters.
"""
import argparse
import json
import os
from pathlib import Path
import re
import time


MATH_ENVIRONMENT_KEYS = (
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE",
    "CUBLAS_WORKSPACE_CONFIG", "CUDNN_CONV_WSCAP_DBG", "CUDA_VISIBLE_DEVICES",
    "TORCHINDUCTOR_CACHE_DIR", "TORCHINDUCTOR_FORCE_DISABLE_CACHES",
)
PRECISION_ENVIRONMENT_KEYS = MATH_ENVIRONMENT_KEYS[:4]


def math_environment():
    return {key: os.environ.get(key) for key in MATH_ENVIRONMENT_KEYS}


def precision_metadata():
    """Read old/new getters independently; their disagreement is evidence."""
    import torch
    from audit_supermimic_precision import matmul_precision_metadata
    result = matmul_precision_metadata()
    getters = {
        "new_global_precision": lambda: torch.backends.fp32_precision,
        "new_cuda_matmul_precision": lambda: torch.backends.cuda.matmul.fp32_precision,
        "new_cudnn_precision": lambda: torch.backends.cudnn.fp32_precision,
        "new_cudnn_conv_precision": lambda: torch.backends.cudnn.conv.fp32_precision,
        "new_cudnn_rnn_precision": lambda: torch.backends.cudnn.rnn.fp32_precision,
        "legacy_matmul_allow_tf32": lambda: torch.backends.cuda.matmul.allow_tf32,
        "legacy_global_matmul_precision": torch.get_float32_matmul_precision,
        "legacy_cudnn_allow_tf32": lambda: torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": lambda: torch.backends.cudnn.benchmark,
        "cudnn_deterministic": lambda: torch.backends.cudnn.deterministic,
    }
    for name, getter in getters.items():
        try:
            result[name] = getter()
        except (AttributeError, RuntimeError) as error:
            result[name] = None
            result[name + "_unavailable_reason"] = str(error)
    return result


def synchronize(device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-diagnostic", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        parser.error("label must contain ASCII letters, digits, underscore or hyphen")
    environment_before = math_environment()
    started = time.perf_counter()
    import torch
    from mjlab.utils.torch import configure_torch_backends
    from mjlab_husky.supermimic.models import WorldModel
    from audit_supermimic_model import projection_adapter, sha
    from audit_supermimic_precision import (
        actor_gradient, fixed_exogenous, load_policy, recorded_action_rollout,
        rng_state, same_rng, tensor_dict_sha256, validate_inputs,
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; no CPU fallback is used")
        torch.cuda.set_device(device)
    elif device.type != "cpu":
        raise ValueError("Only explicit CUDA or CPU execution is supported")
    precision_before = precision_metadata()
    configure_torch_backends()
    torch.set_num_threads(1)
    precision_configured = precision_metadata()
    args.out = args.out.resolve()
    tensor_path = args.out / "data" / f"{args.label}_eager_identity.pt"
    report_path = tensor_path.with_suffix(".json")
    if tensor_path.exists() or report_path.exists():
        raise FileExistsError("Preserve fresh-process comparison artifacts; use a new label")
    snapshot = torch.load(args.model_diagnostic, map_location="cpu", weights_only=False)
    payload = torch.load(args.batch, map_location="cpu", weights_only=False)
    policy_sha = sha(args.policy)
    head_sizes = validate_inputs(snapshot, payload, policy_sha)
    adapter = projection_adapter(snapshot, payload["collection"])
    batch = {name: value.to(device) for name, value in payload["batch"].items()}
    commands = fixed_exogenous(adapter, batch["states"])
    original_rng = rng_state(device)
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        architecture = dict(snapshot["world_architecture"], mixed_precision="none")
        world = WorldModel(**architecture)
        world.load_state_dict(snapshot["world"], strict=True)
        world.to(device=device).eval()
        if any(parameter.dtype != torch.float32 for parameter in world.parameters()):
            raise AssertionError("Eager identity worker requires FP32 world parameters")
        policy = load_policy(torch.load(args.policy, map_location="cpu", weights_only=False),
                             adapter, world.action_dim, device)
        world_hash = tensor_dict_sha256(world.state_dict())
        policy_hash = tensor_dict_sha256(policy.state_dict())
        comparison_rng = rng_state(device)
        synchronize(device)
        initialized = time.perf_counter()
        prediction_started = time.perf_counter()
        prediction = recorded_action_rollout(world, adapter, batch, commands)
        synchronize(device)
        rollout_seconds = time.perf_counter()-prediction_started
        gradient_started = time.perf_counter()
        gamma, lam = snapshot.get("config", {}).get("gamma", .99), snapshot.get("config", {}).get("lam", .95)
        gradient, gradient_metadata = actor_gradient(policy, world, adapter, batch, commands,
                                                     gamma=gamma, lam=lam)
        synchronize(device)
        gradient_seconds = time.perf_counter()-gradient_started
        if tensor_dict_sha256(world.state_dict()) != world_hash or tensor_dict_sha256(policy.state_dict()) != policy_hash:
            raise AssertionError("Eager evaluation changed model, policy or normalizer state")
        if not same_rng(comparison_rng, rng_state(device)):
            raise AssertionError("Fixed eager rollout/gradient consumed RNG")
    if not same_rng(original_rng, rng_state(device)):
        raise AssertionError("Model construction RNG was not restored")
    environment_after = math_environment()
    if any(environment_before[key] != environment_after[key] for key in PRECISION_ENVIRONMENT_KEYS):
        raise AssertionError("Worker or imported helpers changed a recorded math environment variable")
    outputs = {"recorded_action_rollout32": prediction, "actor_gradient32": gradient}
    flat = {"rollout_"+name: value for name, value in prediction.items()}
    flat["actor_gradient32"] = gradient
    if any(not bool(torch.isfinite(value).all()) for value in flat.values()):
        raise FloatingPointError("Nonfinite eager outputs or actor gradient")
    output_hashes = {name: tensor_dict_sha256({name: value}) for name, value in flat.items()}
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tensor_path.with_suffix(".tmp")
    torch.save({"format": "husky-supermimic-eager-identity-tensors-v1", **outputs}, temporary)
    temporary.replace(tensor_path)
    source = Path(__file__).resolve().parents[1] / "src/mjlab_husky/supermimic"
    report = {
        "format": "husky-supermimic-eager-identity-v1", "label": args.label,
        "created_at": time.time(), "execution": "eager FP32; no compilation",
        "optimizer_steps": 0, "algorithm_updates": 0,
        "world_and_policy_state_unchanged": True, "rng_unchanged": True,
        "model_diagnostic_sha256": sha(args.model_diagnostic), "audit_batch_sha256": sha(args.batch),
        "teacher_sha256": policy_sha, "world_state_sha256": world_hash, "policy_state_sha256": policy_hash,
        "tool_sha256": sha(__file__),
        "helper_source_sha256": {name: sha(Path(__file__).with_name(name)) for name in
            ("audit_supermimic_model.py", "audit_supermimic_precision.py")},
        "source_sha256": {name: sha(source / f"{name}.py") for name in ("models", "imagination", "env_adapter")},
        "diagnostic_iteration": snapshot.get("iteration"),
        "diagnostic_dynamics_protocol": snapshot.get("dynamics_protocol"),
        "trained_head_maximum_absolute_parameter": head_sizes,
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "math_environment_before": environment_before, "math_environment_after": environment_after,
        "environment_changes": {key: {"before": environment_before[key], "after": environment_after[key]}
            for key in MATH_ENVIRONMENT_KEYS if environment_before[key] != environment_after[key]},
        "precision_before_configure": precision_before, "precision_after_configure": precision_configured,
        "precision_after_execution": precision_metadata(),
        "world_architecture": architecture, "actor_gradient": dict(gradient_metadata, gamma=gamma, lam=lam),
        "tensor_artifact": str(tensor_path), "tensor_artifact_sha256": sha(tensor_path),
        "tensor_value_sha256": output_hashes, "combined_tensor_value_sha256": tensor_dict_sha256(flat),
        "tensors": {name: {"shape": list(value.shape), "dtype": str(value.dtype),
            "all_finite": True, "l2_norm": float(value.double().norm()),
            "maximum_absolute_value": float(value.double().abs().max())} for name, value in flat.items()},
        "initialization_seconds": initialized-started, "rollout_seconds": rollout_seconds,
        "actor_gradient_seconds": gradient_seconds, "total_seconds": time.perf_counter()-started,
        "scope": "Compare tensor values across fresh processes using identical source/artifact hashes. File SHA includes serialization and is not an output equality test. FP32 is the tensor dtype with the recorded original mjlab matmul permissions, including TF32. This is no policy-quality or convergence-speed result.",
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"report": str(report_path), "report_sha256": sha(report_path),
        "tensor_artifact": str(tensor_path), "tensor_artifact_sha256": report["tensor_artifact_sha256"],
        "combined_tensor_value_sha256": report["combined_tensor_value_sha256"],
        "optimizer_steps": 0, "actor_gradient_elements": gradient.numel()}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
