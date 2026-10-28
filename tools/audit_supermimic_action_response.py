"""Bounded, fresh-process finite-amplitude real action-response audit.

For each requested phase, independent plus/minus/repeated-plus workers follow
the same unperturbed teacher prefix, then each world probes action i % 23 once.
Exact precondition equality is required before interpreting a real difference.
No optimizer, rendering, simulator restoration or training is performed.
"""
import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time
import traceback

from evaluate_supermimic_batch import canonical_sha256, file_sha256


PHASES = {"push": (0, 1, 0), "push2steer": (1, 2, 2),
          "steer": (2, 3, 1), "steer2push": (3, 4, 3)}
VARIANTS = ("plus", "minus", "repeat_plus")


def write_json(path, contents):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(contents, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def tensor_sha(value):
    """Hash dtype/shape/value, independently of serialization storage IDs."""
    import torch
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(json.dumps([str(value.dtype), list(value.shape)]).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def per_env_hashes(fields):
    n = next(iter(fields.values())).shape[0]
    return [canonical_sha256({name: tensor_sha(value[index]) for name, value in sorted(fields.items())})
            for index in range(n)]


def host_fields(fields):
    return {name: value.detach().cpu().contiguous().clone() for name, value in fields.items()}


def comparison_metrics(plus, minus, repeat, predicted, epsilon, selected=None):
    """CPU-only statistics; eligibility is determined by REAL response/noise.

    A response must exceed ten times repeated-input noise AND eight FP32 ULPs
    at the output scale. Near-zero and noisy responses stay explicitly counted.
    Different physical units must be supplied as different calls/groups.
    """
    import torch
    plus, minus, repeat, predicted = [x.detach().cpu().float() for x in (plus, minus, repeat, predicted)]
    if plus.ndim != 2 or any(x.shape != plus.shape for x in (minus, repeat, predicted)):
        raise ValueError("Response groups must have identical [env, features] shapes")
    n = plus.shape[0]
    selected = torch.ones(n, dtype=torch.bool) if selected is None else selected.cpu().bool()
    signal_delta, noise_delta = plus - minus, plus - repeat
    derivative = signal_delta / (2 * epsilon)
    scale = torch.maximum(torch.maximum(plus.abs(), minus.abs()), torch.ones_like(plus))
    floor = 8 * torch.finfo(torch.float32).eps * scale
    finite = torch.isfinite(signal_delta).all(1) & torch.isfinite(noise_delta).all(1) & torch.isfinite(predicted).all(1)
    response_rms = signal_delta.double().square().mean(1).sqrt()
    noise_rms = noise_delta.double().square().mean(1).sqrt()
    floor_rms = floor.double().square().mean(1).sqrt()
    near_zero = response_rms <= floor_rms
    interpretable = selected & finite & ~near_zero & (response_rms > 10 * noise_rms)
    component_mask = (selected[:, None] & finite[:, None] & (signal_delta.abs() > floor)
                      & (signal_delta.abs() > 10 * noise_delta.abs()))
    real_norm = derivative.double().norm(dim=1)
    model_norm = predicted.double().norm(dim=1)
    cosine = (derivative.double() * predicted.double()).sum(1) / (real_norm * model_norm).clamp_min(1e-30)
    cosine_mask = interpretable & (model_norm > 0)

    def distribution(value, mask):
        chosen = value[mask]
        return {"count": int(chosen.numel()), "mean": float(chosen.mean()) if chosen.numel() else None,
                "median": float(chosen.median()) if chosen.numel() else None,
                "minimum": float(chosen.min()) if chosen.numel() else None,
                "maximum": float(chosen.max()) if chosen.numel() else None}

    matched = (torch.sign(derivative) == torch.sign(predicted)) & component_mask
    component_count = int(component_mask.sum())
    summary = {"selected_envs": int(selected.sum()), "nonfinite_envs": int((selected & ~finite).sum()),
        "near_zero_envs": int((selected & finite & near_zero).sum()),
        "noise_dominated_envs": int((selected & finite & ~near_zero & ~interpretable).sum()),
        "interpretable_envs": int(interpretable.sum()), "interpretable_components": component_count,
        "sign_agreement": float(matched.sum() / component_count) if component_count else None,
        "cosine": distribution(cosine, cosine_mask),
        "predicted_to_real_norm_ratio": distribution(model_norm / real_norm.clamp_min(1e-30), interpretable),
        "derivative_rmse": distribution((predicted.double()-derivative.double()).square().mean(1).sqrt(), interpretable),
        "real_response_rms": distribution(response_rms, selected & finite),
        "repeat_noise_rms": distribution(noise_rms, selected & finite),
        "eligibility": "real plus-minus RMS > 10*repeat-plus noise RMS and > 8 FP32 ULPs at output scale; component signs use the same per-component gate"}
    tensors = {"real_finite_difference": derivative, "model_finite_difference": predicted,
               "repeat_noise_delta": noise_delta, "response_delta": signal_delta,
               "interpretable_env_mask": interpretable, "interpretable_component_mask": component_mask,
               "selected_env_mask": selected}
    return summary, tensors


def compare_preconditions(left, right):
    import torch
    report, exact = {}, True
    for name in sorted(set(left["pre"]) | set(right["pre"])):
        a, b = left["pre"].get(name), right["pre"].get(name)
        same_shape = a is not None and b is not None and a.shape == b.shape and a.dtype == b.dtype
        same = same_shape and torch.equal(a, b)
        item = {"exact": bool(same)}
        if same_shape and not same:
            delta = (a.double()-b.double()).abs()
            finite = torch.isfinite(delta)
            item.update(maximum_absolute_difference=float(delta[finite].max()) if bool(finite.any()) else None,
                        nonfinite_difference_elements=int((~finite).sum()),
                        differing_env_indices=torch.nonzero((a != b).reshape(a.shape[0], -1).any(1)).flatten().tolist())
        report[name] = item
        exact = exact and bool(same)
    prefix_same = (left["prefix_sha256_per_env"] == right["prefix_sha256_per_env"]
                   and left["initial_sha256_per_env"] == right["initial_sha256_per_env"])
    rng_same = left["pre_rng_sha256"] == right["pre_rng_sha256"]
    metadata_same = left["comparison_identity"] == right["comparison_identity"]
    pre_hash_same = left["pre_sha256_per_env"] == right["pre_sha256_per_env"]
    return {"exact": exact and prefix_same and rng_same and metadata_same and pre_hash_same,
            "prefix_hashes_exact": prefix_same, "rng_hashes_exact": rng_same,
            "runtime_metadata_exact": metadata_same, "fields": report,
            "precondition_hashes_exact": pre_hash_same}


def worker_paths(args, phase, variant):
    stem = f"{args.label}_{phase}_{variant}_action_response"
    return {"json": args.out / "data" / (stem + ".json"),
            "pt": args.out / "data" / (stem + ".pt"),
            "log": args.out / "log" / (stem + ".log")}


def worker(args):
    # Base tasks must initialize before custom plugins. These imports are only
    # in a worker; parent orchestration and --help never construct a simulator.
    import mjlab.tasks  # noqa: F401
    import mujoco
    import mujoco_warp as mjwarp
    import numpy as np
    import torch
    import warp as wp
    from tensordict import TensorDict
    from rsl_rl.modules import ActorCritic
    from mjlab.utils.torch import configure_torch_backends
    from mjlab_husky.envs import G1SkaterManagerBasedRlEnv
    from mjlab_husky.tasks.registry import load_env_cfg, load_rl_cfg
    from mjlab_husky.supermimic.env_adapter import CoupledStateAdapter
    from mjlab_husky.supermimic.models import WorldModel
    from audit_supermimic_model import canonical_metadata, load_joint_metadata, physics_groups

    paths = worker_paths(args, args.worker_phase, args.worker_variant)
    started = time.monotonic()
    status = {"format": "husky-supermimic-action-response-worker-v1", "phase": args.worker_phase,
              "variant": args.worker_variant, "status": "starting", "optimizer_steps": 0}

    def checkpoint(stage):
        status.update(status=stage, elapsed_seconds=time.monotonic()-started)
        write_json(paths["json"], status)

    checkpoint("loading")
    configure_torch_backends()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model_sha = file_sha256(args.model_diagnostic)
    snapshot = torch.load(args.model_diagnostic, map_location="cpu", weights_only=False)
    if model_sha != file_sha256(args.model_diagnostic):
        raise ValueError("Model diagnostic changed while it was loading")
    if snapshot.get("dynamics_protocol") != "conditional-commands-v2":
        raise ValueError("Expected a conditional-commands-v2 diagnostic")
    task = "Mjlab-Skater-Flat-Unitree-G1"
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs, cfg.seed = args.num_envs, args.seed
    cfg.observations["policy"].enable_corruption = False
    # Preserve the training adapter's interval-timer feature. Removing this
    # event in play mode changes the state layout from 1443 to 1442. Its
    # original velocity-setting function/parameters are retained, but a fixed
    # ten-second timer cannot fire during any prefix/probe (asserted below).
    push_event = load_env_cfg(task, play=False).events["push_robot"]
    original_push_interval = list(push_event.interval_range_s)
    push_event.interval_range_s = (10.0, 10.0)
    cfg.events["push_robot"] = push_event
    cfg.commands["skate"].ranges.lin_vel_x = (0.0, 1.5)
    cfg.commands["skate"].ranges.heading = (-math.pi/4, math.pi/4)
    env = G1SkaterManagerBasedRlEnv(cfg, device=args.device)
    adapter = None
    try:
        obs, _ = env.reset()
        agent = load_rl_cfg(task)
        policy_cfg = asdict(agent.policy)
        policy_cfg.pop("class_name")
        teacher = ActorCritic(TensorDict(obs, batch_size=[env.num_envs]), agent.obs_groups,
                             env.action_manager.total_action_dim, **policy_cfg).to(args.device)
        policy_sha = file_sha256(args.policy)
        saved = torch.load(args.policy, map_location="cpu", weights_only=False)
        if policy_sha != file_sha256(args.policy):
            raise ValueError("Teacher checkpoint changed while it was loading")
        teacher.load_state_dict(saved["model_state_dict"])
        del saved
        teacher.eval().requires_grad_(False)
        adapter = CoupledStateAdapter(env, joint_consistency=False)
        for name in ("layout", "history_terms", "critic_terms", "qpos_quaternion_slices"):
            if canonical_metadata(getattr(adapter, name)) != canonical_metadata(snapshot[name]):
                raise ValueError("Live adapter differs from diagnostic: " + name)
        if adapter.state_dim != snapshot["world_architecture"]["state_dim"] or adapter.context_dim != snapshot["world_architecture"]["context_dim"]:
            raise ValueError("Diagnostic dimensions differ from live environment")
        metadata = load_joint_metadata(snapshot, args.joint_metadata)
        if metadata is not None:
            if adapter.joint_projection_metadata is None or canonical_metadata(metadata["spec"]) != canonical_metadata(adapter.joint_projection_metadata["spec"]):
                raise ValueError("Explicit joint metadata differs from live robot APIs")
            if metadata.get("spec_sha256") != adapter.joint_projection_metadata["spec_sha256"]:
                raise ValueError("Joint metadata SHA differs from live canonical spec")
            adapter.set_joint_projection_metadata(metadata["spec"])
        n, action_dim = env.num_envs, env.action_manager.total_action_dim
        if action_dim != 23:
            raise ValueError("This audit requires the original 23-action HUSKY policy")
        qids = torch.as_tensor(env.robot.data.indexing.joint_q_adr, device=env.device, dtype=torch.long)
        vids = torch.as_tensor(env.robot.data.indexing.joint_v_adr, device=env.device, dtype=torch.long)
        joint_names = list(env.robot.joint_names)
        term = env.action_manager._terms["joint_pos"]
        action_names = list(term.target_names)
        if len(joint_names) != 23 or len(action_names) != 23 or set(joint_names) != set(action_names):
            raise ValueError("Action names do not map exactly to robot joint APIs")
        action_to_robot = [joint_names.index(name) for name in action_names]
        action_scale = torch.as_tensor(term.scale, device=env.device, dtype=torch.float32).expand(n, 23).clone()
        action_offset = torch.as_tensor(term.offset, device=env.device, dtype=torch.float32).expand(n, 23).clone()
        low, high, phase_index = PHASES[args.worker_phase]
        center = (cfg.phase_ratios[low]+cfg.phase_ratios[high])/2
        phase_center_steps = max(1, int(math.floor(center*env.cycle_time/env.step_dt+.5)))
        prefix_steps = phase_center_steps if args.prefix_steps is None else args.prefix_steps
        if not (prefix_steps+1)*env.step_dt < 10.0:
            raise AssertionError("Prefix plus real probe must finish before the fixed ten-second push timer")
        context = adapter.context().detach().clone()
        prefix_digests = [hashlib.sha256() for _ in range(n)]
        initial_hashes = per_env_hashes(host_fields({"state": adapter.observe(), "context": context}))
        normalizers_before = {name: tensor_sha(value) for name, value in teacher.state_dict().items() if "_obs_normalizer." in name}
        if teacher.actor_obs_normalizer.training or teacher.critic_obs_normalizer.training:
            raise AssertionError("Teacher observation normalizers must be frozen")
        checkpoint("teacher_prefix")
        with torch.no_grad():
            for step in range(prefix_steps):
                state = adapter.observe()
                action = teacher.act_inference(adapter.state_to_observations(state))
                rows = torch.cat((state, action), -1).detach().cpu().contiguous().numpy()
                for index in range(n):
                    prefix_digests[index].update(rows[index].tobytes())
                transition = adapter.capture_step(action)
                if bool(transition.done.any()):
                    raise AssertionError("Play prefix unexpectedly reset an environment")
            state = adapter.observe().detach().clone()
            base_action = teacher.act_inference(adapter.state_to_observations(state)).detach().clone().float()
        if not torch.equal(state[:, adapter.layout["qpos"]][:, qids], env.robot.data.joint_pos):
            raise AssertionError("Live robot joint-position addresses do not match simulator qpos")
        if not torch.equal(state[:, adapter.layout["qvel"]][:, vids], env.robot.data.joint_vel):
            raise AssertionError("Live robot joint-velocity addresses do not match simulator qvel")

        # This exact API/shape is exercised by the pinned Warp support_test.py.
        # Unsupported user/plugin data must not silently produce an incomplete identity.
        for field in ("nuserdata", "npluginstate"):
            if int(getattr(env.sim.mj_model, field, 0)):
                raise ValueError("INTEGRATION identity does not support nonzero " + field)
        size = mujoco.mj_stateSize(env.sim.mj_model, mujoco.mjtState.mjSTATE_INTEGRATION)
        with wp.ScopedDevice(env.sim.wp_device):
            integration_wp = wp.zeros((n, size), dtype=float)
            mjwarp.get_state(env.sim.wp_model, env.sim.wp_data, integration_wp, mjwarp.State.INTEGRATION)
        wp.synchronize_device(env.sim.wp_device)
        integration = wp.to_torch(integration_wp).detach().clone()
        pre = host_fields({"state": state, "context": context, "raw_policy": adapter._last_obs["policy"],
            "raw_critic": adapter._last_obs["critic"], "command": env.command_manager.get_command("skate"),
            "teacher_action": base_action, "integration": integration,
            "action_scale": action_scale, "action_offset": action_offset,
            "encoder_bias": env.robot.data.encoder_bias,
            "joint_position_target": env.robot.data.joint_pos_target})
        pre_rng = {"cpu": tensor_sha(torch.get_rng_state()),
                   "cuda": tensor_sha(torch.cuda.get_rng_state(env.device)) if torch.device(env.device).type == "cuda" else None}
        with torch.no_grad():
            exogenous = {name: value.detach().clone() for name, value in adapter.sample_exogenous(state).items()}
        after_exo_rng = {"cpu": tensor_sha(torch.get_rng_state()),
                         "cuda": tensor_sha(torch.cuda.get_rng_state(env.device)) if torch.device(env.device).type == "cuda" else None}
        if pre_rng != after_exo_rng:
            raise AssertionError("Phase-center exogenous sampling unexpectedly consumed RNG")
        if set(exogenous)-{"command_b", "is_heading_env", "is_standing_env", "time_left"}:
            raise AssertionError("Only the four external command fields may condition the model")
        pre.update({"exogenous_"+name: value.detach().cpu().clone() for name, value in exogenous.items()})
        direction = torch.zeros_like(base_action)
        action_indices = torch.arange(n, device=env.device) % 23
        direction.scatter_(1, action_indices[:, None], 1.)
        sign = -1 if args.worker_variant == "minus" else 1
        probe_action = base_action + sign*args.epsilon*direction
        checkpoint("real_probe")
        with torch.no_grad():
            transition = adapter.capture_step(probe_action)
        if bool(transition.done.any()):
            raise AssertionError("Real probe unexpectedly reset an environment")

        groups = {"physical_robot_qpos": {"indices": (qids+adapter.layout["qpos"].start).cpu().tolist(), "units": "rad"},
                  "physical_robot_qvel": {"indices": (vids+adapter.layout["qvel"].start).cpu().tolist(), "units": "rad/s"},
                  "amp_joint_pos": {"indices": list(range(adapter.layout["amp_joints"].start, adapter.layout["amp_joints"].stop)), "units": "rad"}}
        for group_name, terms in (("actor", adapter.history_terms), ("critic", adapter.critic_terms)):
            for observation in terms:
                if observation["name"] in ("joint_pos", "joint_vel", "base_ang_vel", "projected_gravity"):
                    end, width = observation["slice"].stop, observation["frame_dim"]
                    scale = torch.as_tensor(1. if observation["scale"] is None else observation["scale"]).flatten().tolist()
                    groups[group_name+"_"+observation["name"]] = {"indices": list(range(end-width, end)),
                        "units": "original scaled observation coordinates", "observation_scale": scale}
        for name, group in physics_groups(env.sim.mj_model).items():
            if "root_" in name:
                groups["physical_"+name] = {"indices": [adapter.layout[group["field"]].start+i for i in group["indices"]],
                                            "units": group["units"]}
        selectors = {name: torch.tensor(group["indices"], device=env.device) for name, group in groups.items()}

        def extract(value):
            return {name: value.index_select(-1, indices) for name, indices in selectors.items()}

        checkpoint("model_response")
        # Model construction happens AFTER real probing so initialization draws
        # cannot alter the real prefix, command events or measured transition.
        world = WorldModel(**dict(snapshot["world_architecture"], mixed_precision="none"))
        world.load_state_dict(snapshot["world"])
        world.to(args.device).eval().requires_grad_(False)
        world_before = {name: tensor_sha(value) for name, value in world.state_dict().items()}

        def predicted_next(action):
            prediction = world(state, action, context)
            return adapter.project(state, prediction["next_state"], action, exogenous=exogenous).float()

        with torch.no_grad():
            model_base = predicted_next(base_action)
            model_plus = predicted_next(base_action+args.epsilon*direction)
            model_minus = predicted_next(base_action-args.epsilon*direction)
            model_fd = (model_plus-model_minus)/(2*args.epsilon)
        with torch.enable_grad(), torch.autocast(device_type=torch.device(env.device).type, enabled=False):
            _, jvp = torch.autograd.functional.jvp(predicted_next, base_action.clone(), direction, create_graph=False)
        if any(value.dtype != torch.float32 or not bool(torch.isfinite(value).all()) for value in
               (state, base_action, transition.next_state, model_base, model_plus, model_minus, model_fd, jvp)):
            raise FloatingPointError("All audited states/actions/model outputs must be finite FP32")
        if world_before != {name: tensor_sha(value) for name, value in world.state_dict().items()}:
            raise AssertionError("Frozen world/normalizer state changed")
        if normalizers_before != {name: tensor_sha(value) for name, value in teacher.state_dict().items() if "_obs_normalizer." in name}:
            raise AssertionError("Teacher observation normalizers changed")
        actual_phase = state[:, adapter.layout["phase"]][:, phase_index] > .5
        standing = state[:, adapter.layout["command_b"]][:, 0] < .1
        if args.prefix_steps is None and bool((~actual_phase & ~standing).any()):
            raise AssertionError("Requested runtime phase center differs from actual non-standing phase")
        # With an explicit override, --phases is only the output namespace.
        # Reset states can have all-zero phase flags: retain every environment
        # without claiming that these are samples of the requested phase.
        selected = actual_phase if args.prefix_steps is None else torch.ones_like(actual_phase)
        coverage = ("phase_center" if args.prefix_steps is None else
                    "reset_only" if prefix_steps == 0 else "explicit_nonphase_prefix")
        identity = {"task": task, "seed": args.seed, "num_envs": n, "phase": args.worker_phase,
            "phase_ratios": list(cfg.phase_ratios), "prefix_steps": prefix_steps, "phase_center_fraction": center,
            "prefix_steps_override": args.prefix_steps, "phase_center_default_prefix_steps": phase_center_steps,
            "coverage_mode": coverage, "probe_is_phase_center": args.prefix_steps is None,
            "coverage_limit": ("Reset-only states before any teacher physics prefix; phase label is a filename namespace, not demonstrated phase coverage."
                if coverage == "reset_only" else "Explicit-prefix states; phase-center coverage is not asserted."
                if coverage == "explicit_nonphase_prefix" else "Only sampled teacher phase-center states are covered."),
            "step_dt_seconds": env.step_dt, "physics_dt_seconds": env.physics_dt, "decimation": cfg.decimation,
            "cycle_time_seconds": env.cycle_time, "epsilon_raw_action": args.epsilon,
            "joint_names_robot_order": joint_names, "joint_names_action_order": action_names,
            "action_order_to_robot_joint": action_to_robot, "groups": groups,
            "model_diagnostic_sha256": model_sha, "policy_sha256": policy_sha,
            "source_sha256": {"tool": file_sha256(__file__),
                **{name: file_sha256(inspect.getsourcefile(cls)) for name, cls in
                   (("world", WorldModel), ("adapter", CoupledStateAdapter),
                    ("environment", G1SkaterManagerBasedRlEnv), ("teacher", ActorCritic))}},
            "projection_variant": adapter.joint_projection_variant,
            "joint_projection_spec_sha256": adapter.joint_projection_metadata["spec_sha256"] if metadata is not None else None,
            "dynamics_protocol": snapshot["dynamics_protocol"], "tf32": False,
            "observation_corruption": False, "physical_parameters": "original HUSKY play configuration",
            "push_robot_event_present": True, "push_robot_triggered_during_prefix_or_probe": False,
            "push_robot_original_interval_range_seconds": original_push_interval,
            "push_robot_diagnostic_interval_range_seconds": [10.0, 10.0],
            "push_robot_function_and_velocity_parameters": "original training event, unchanged",
            "prefix_plus_probe_seconds": (prefix_steps+1)*env.step_dt,
            "integration_signature": int(mjwarp.State.INTEGRATION), "integration_state_size": size}
        payload = {"format": "husky-supermimic-action-response-tensors-v1", "comparison_identity": identity,
            "pre": pre, "pre_sha256_per_env": per_env_hashes(pre), "pre_rng_sha256": pre_rng,
            "initial_sha256_per_env": initial_hashes,
            "prefix_sha256_per_env": [digest.hexdigest() for digest in prefix_digests],
            "requested_phase_mask": selected.cpu(), "actual_requested_phase_mask": actual_phase.cpu(),
            "standing_mask": standing.cpu(),
            "action_joint_indices": action_indices.cpu(),
            "output_joint_indices": torch.tensor(action_to_robot)[action_indices.cpu()],
            "probe_action": probe_action.cpu(), "probe_direction": direction.cpu(),
            "real_next_state": transition.next_state.cpu(), "real_reward": transition.reward.cpu(),
            "real_outputs": host_fields(extract(transition.next_state)),
            "model": {name: host_fields(extract(value)) for name, value in
                      (("base", model_base), ("plus", model_plus), ("minus", model_minus), ("finite_difference", model_fd), ("jvp", jvp))},
            "model_full_state": host_fields({"base": model_base, "plus": model_plus, "minus": model_minus, "jvp": jvp})}
        temporary = paths["pt"].with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(paths["pt"])
        status.update(status="completed", elapsed_seconds=time.monotonic()-started,
            comparison_identity=identity, pre_sha256_per_env=payload["pre_sha256_per_env"],
            prefix_sha256_per_env=payload["prefix_sha256_per_env"], initial_sha256_per_env=initial_hashes,
            tensor_file=paths["pt"].name, tensor_sha256=file_sha256(paths["pt"]),
            selected_envs=int(selected.sum()), actual_requested_phase_envs=int(actual_phase.sum()),
            requested_phase_envs=int(actual_phase.sum()), standing_envs=int(standing.sum()),
            normalizers_unchanged=True, frozen_world_unchanged=True)
        write_json(paths["json"], status)
    finally:
        if adapter is not None:
            adapter.close()
        env.close()


def aggregate_phase(args, phase):
    import torch
    payloads = {}
    for variant in VARIANTS:
        paths = worker_paths(args, phase, variant)
        if not paths["json"].exists() or not paths["pt"].exists():
            return {"status": "incomplete", "interpretation": "No real finite difference calculated."}, None
        report = json.loads(paths["json"].read_text())
        if report.get("status") != "completed" or report.get("tensor_sha256") != file_sha256(paths["pt"]):
            return {"status": "incomplete_or_invalid_evidence", "variant": variant}, None
        payloads[variant] = torch.load(paths["pt"], map_location="cpu", weights_only=False)
    plus, minus, repeat = [payloads[variant] for variant in VARIANTS]
    checks = {"plus_vs_minus": compare_preconditions(plus, minus),
              "plus_vs_repeat": compare_preconditions(plus, repeat)}
    report = {"precondition_comparisons": checks, "comparison_identity": plus["comparison_identity"]}
    if not all(check["exact"] for check in checks.values()):
        report.update(status="precondition_mismatch", interpretation="No real finite difference or sign/cosine/norm ratio calculated. Inspect saved precondition tensors and differences.")
        return report, None
    summaries, tensors = {}, {}
    selected = plus["requested_phase_mask"]
    for name, value in plus["real_outputs"].items():
        summary, raw = comparison_metrics(value, minus["real_outputs"][name], repeat["real_outputs"][name],
            plus["model"]["finite_difference"][name], args.epsilon, selected)
        jvp, fd = plus["model"]["jvp"][name], plus["model"]["finite_difference"][name]
        summary["model_jvp_vs_finite_difference"] = {
            "maximum_absolute_difference": float((jvp-fd).abs().max()),
            "rmse": float((jvp.double()-fd.double()).square().mean().sqrt())}
        raw["model_jvp"] = jvp
        if name in ("physical_robot_qpos", "physical_robot_qvel", "amp_joint_pos",
                    "actor_joint_pos", "actor_joint_vel", "critic_joint_pos", "critic_joint_vel"):
            indices = plus["output_joint_indices"][:, None]
            diagonal, _ = comparison_metrics(*(item.gather(1, indices) for item in
                (value, minus["real_outputs"][name], repeat["real_outputs"][name], plus["model"]["finite_difference"][name])),
                args.epsilon, selected)
            summary["actuated_joint_component"] = diagonal
        summaries[name], tensors[name] = summary, raw
    consistency = {}
    for name, physical in (("amp_joint_pos", "physical_robot_qpos"), ("actor_joint_pos", "physical_robot_qpos"),
                           ("critic_joint_pos", "physical_robot_qpos"), ("actor_joint_vel", "physical_robot_qvel"),
                           ("critic_joint_vel", "physical_robot_qvel")):
        if name not in tensors:
            continue
        scales = plus["comparison_identity"]["groups"][name].get("observation_scale", [1.])
        scale = torch.tensor(scales).expand(tensors[name]["model_jvp"].shape[-1])
        if bool((scale == 0).any()):
            continue
        consistency[name] = {}
        for kind in ("real_finite_difference", "model_finite_difference", "model_jvp"):
            difference = tensors[name][kind]/scale-tensors[physical][kind]
            consistency[name][kind+"_rmse_against_raw_physics"] = float(difference.double().square().mean().sqrt())
    report.update(status="completed", selected_envs=int(selected.sum()),
        requested_phase_envs=int(plus["actual_requested_phase_mask"].sum()),
        actual_requested_phase_envs=int(plus["actual_requested_phase_mask"].sum()),
        coverage_mode=plus["comparison_identity"]["coverage_mode"],
        standing_envs=int(plus["standing_mask"].sum()), groups=summaries,
        redundant_joint_channel_derivative_consistency=consistency,
        interpretation="Finite-amplitude one-control-step action response at the explicitly recorded coverage mode; signs/cosines/norm ratios only use noise-eligible responses. "
            + plus["comparison_identity"]["coverage_limit"])
    return report, tensors


def orchestrate(args):
    started = time.monotonic()
    deadline = started+args.max_seconds
    report_path = args.out / "data" / f"{args.label}_action_response.json"
    tensor_path = report_path.with_suffix(".pt")
    for path in (report_path, tensor_path, *(path for phase in args.phases for variant in VARIANTS
                                            for path in worker_paths(args, phase, variant).values())):
        if path.exists():
            raise FileExistsError("Use a new label; audit evidence is not overwritten: " + str(path))
    for name in ("data", "log"):
        (args.out/name).mkdir(parents=True, exist_ok=True)
    report = {"format": "husky-supermimic-action-response-v1", "status": "running", "label": args.label,
        "created_at_unix": time.time(), "max_seconds": args.max_seconds, "worker_max_seconds": args.worker_seconds,
        "phases_requested": args.phases, "workers": [], "optimizer_steps": 0, "training_updates": 0,
        "prefix_steps_override": args.prefix_steps,
        "tool_sha256": file_sha256(__file__), "model_diagnostic_sha256": file_sha256(args.model_diagnostic),
        "policy_sha256": file_sha256(args.policy),
        "scope": ["Finite-amplitude action responses in original play physics with observation noise disabled and deterministic teacher mean.",
                  "The original training push event and velocity parameters are retained to preserve the 1443-dimensional timer layout, with its diagnostic timer fixed at 10 seconds (original range 5..10 seconds). Every prefix plus probe is asserted shorter than 10 seconds; all default probes finish within the first 6 seconds, before any push can fire.",
                  "Each phase has independent fresh plus/minus/repeated-plus processes with an unperturbed common teacher prefix.",
                  "Exact captured preconditions are a prerequisite; adapter and INTEGRATION identities do not assert an exhaustive simulator snapshot/restore implementation.",
                  "Only sampled states and one-joint, one-control-step perturbations are covered. Default sampling uses phase centers; explicit prefix overrides, especially reset-only zero, do not establish phase-center or later-phase coverage. This is not a task success rate, 32-step return-gradient validation or proof of improved policy learning."]}
    write_json(report_path, report)
    timed_out = False
    failed_worker = False
    for phase in args.phases:
        for variant in VARIANTS:
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            paths = worker_paths(args, phase, variant)
            command = [sys.executable, str(Path(__file__).resolve()), "--model-diagnostic", str(args.model_diagnostic),
                "--policy", str(args.policy), "--out", str(args.out), "--label", args.label,
                "--seed", str(args.seed), "--num-envs", str(args.num_envs), "--epsilon", str(args.epsilon),
                "--device", args.device, "--worker-phase", phase, "--worker-variant", variant]
            if args.joint_metadata is not None:
                command.extend(["--joint-metadata", args.joint_metadata])
            if args.prefix_steps is not None:
                command.extend(["--prefix-steps", str(args.prefix_steps), "--phases", phase])
            run_started = time.monotonic()
            limit = min(args.worker_seconds, remaining)
            with paths["log"].open("w") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                report["active_worker"] = {"phase": phase, "variant": variant, "pid": process.pid,
                                           "time_limit_seconds": limit, "status": "running"}
                write_json(report_path, report)
                timeout = False
                try:
                    try:
                        returncode = process.wait(timeout=limit)
                    except subprocess.TimeoutExpired:
                        timeout = True
                        os.killpg(process.pid, signal.SIGKILL)
                        returncode = process.wait()
                finally:
                    # Also clean up a worker if the overall alarm interrupts
                    # wait(), hashing, or orchestration. Never leave GPU work.
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    report["active_worker"].update(status="exited", returncode=process.returncode,
                                                   elapsed_seconds=time.monotonic()-run_started)
                    write_json(report_path, report)
            report["workers"].append({"phase": phase, "variant": variant, "returncode": returncode,
                "timeout": timeout, "time_limit_seconds": limit, "elapsed_seconds": time.monotonic()-run_started,
                "log_file": paths["log"].name, "report_file": paths["json"].name,
                "tensor_file": paths["pt"].name if paths["pt"].exists() else None})
            write_json(report_path, report)
            if returncode != 0 and not timeout:
                failed_worker = True
                report["stopped_after_failed_worker"] = {"phase": phase, "variant": variant,
                    "returncode": returncode,
                    "reason": "Nonzero worker exit; later variants and phases were not started. Inspect this worker JSON/log before a new-label rerun."}
                write_json(report_path, report)
                break
            if timeout and time.monotonic() >= deadline:
                timed_out = True
                break
        if timed_out or failed_worker:
            break
    phase_reports, raw = {}, {}
    for phase in args.phases:
        try:
            phase_reports[phase], values = aggregate_phase(args, phase)
        except Exception as error:
            phase_reports[phase], values = {"status": "aggregation_failed", "error": repr(error),
                                           "traceback": traceback.format_exc()}, None
        if values is not None:
            raw[phase] = values
    if raw:
        import torch
        torch.save({"format": "husky-supermimic-action-response-aggregate-tensors-v1", "phases": raw}, tensor_path)
        report.update(aggregate_tensor_file=tensor_path.name, aggregate_tensor_sha256=file_sha256(tensor_path))
    complete = all(value["status"] == "completed" for value in phase_reports.values())
    report.update(status="completed" if complete else "deadline_reached" if timed_out else "worker_failed" if failed_worker else "incomplete_or_invalid",
                  phases=phase_reports, elapsed_seconds=time.monotonic()-started, deadline_reached=timed_out)
    write_json(report_path, report)
    print(json.dumps({"status": report["status"], "report": str(report_path),
                      "elapsed_seconds": report["elapsed_seconds"]}, allow_nan=False), flush=True)
    return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-diagnostic", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Existing experiment root; writes data/ and log/")
    parser.add_argument("--label", default="action_response")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--epsilon", type=float, default=.03, help="Perturbation in original policy-action units")
    parser.add_argument("--phases", nargs="+", choices=list(PHASES), default=list(PHASES))
    parser.add_argument("--prefix-steps", type=int,
                        help="Explicit single-phase prefix override; 0 probes reset-only states, not phase-center coverage")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-seconds", type=float, default=300., help="Overall wall-time cap, at most 300 seconds")
    parser.add_argument("--worker-seconds", type=float, default=60., help="Each worker wall-time cap, at most 60 seconds")
    parser.add_argument("--joint-metadata", help="Explicit P1 opt-in: snapshot or metadata JSON; default uses original conditional-v2 projection")
    parser.add_argument("--worker-phase", choices=list(PHASES), help=argparse.SUPPRESS)
    parser.add_argument("--worker-variant", choices=VARIANTS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        parser.error("label must use ASCII letters, digits, underscore or hyphen")
    if args.num_envs < 1 or not math.isfinite(args.epsilon) or args.epsilon <= 0:
        parser.error("num-envs and finite epsilon must be positive")
    if not (0 < args.max_seconds <= 300 and 0 < args.worker_seconds <= 60):
        parser.error("max-seconds must be in (0,300], worker-seconds in (0,60]")
    if len(set(args.phases)) != len(args.phases):
        parser.error("phases must be unique")
    if args.prefix_steps is not None and (args.prefix_steps < 0 or len(args.phases) != 1):
        parser.error("prefix-steps must be nonnegative and explicitly used with exactly one --phases value")
    if bool(args.worker_phase) != bool(args.worker_variant):
        parser.error("internal worker arguments must occur together")
    args.out, args.policy, args.model_diagnostic = args.out.resolve(), args.policy.resolve(), args.model_diagnostic.resolve()
    if args.worker_phase:
        paths = worker_paths(args, args.worker_phase, args.worker_variant)
        try:
            worker(args)
        except BaseException as error:
            previous = json.loads(paths["json"].read_text()) if paths["json"].exists() else {}
            previous.update(status="failed", error=repr(error), traceback=traceback.format_exc())
            write_json(paths["json"], previous)
            raise
        return 0
    def overall_deadline(signum, frame):
        raise TimeoutError("Overall action-response audit wall-time cap reached")

    old_handler = signal.signal(signal.SIGALRM, overall_deadline)
    signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
    try:
        return orchestrate(args)
    except TimeoutError as error:
        report_path = args.out / "data" / f"{args.label}_action_response.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        report.update(status="deadline_reached", deadline_reached=True, error=str(error),
                      scope_at_deadline="Worker group killed; partial JSON/PT/log evidence retained. No retry or training.")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(report_path, report)
        print(json.dumps({"status": "deadline_reached", "report": str(report_path)}), flush=True)
        return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


if __name__ == "__main__":
    raise SystemExit(main())
