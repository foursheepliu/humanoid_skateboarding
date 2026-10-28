"""Bounded live adapter identity check, using true next physics as prediction.

This checks observation frames, noise conditioning, commands and history
projection. It does not measure learned-model accuracy, policy quality or speed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time


def check_compiled_projection(adapter, captured_state, args) -> dict:
    """Value and vector-Jacobian parity, including external-input isolation.

    Both paths receive independent leaves with identical values and the same
    fixed packed cotangent. The synthetic boundary cases test the projection,
    without advancing or modifying the live simulator.
    """
    import torch

    device = captured_state.device
    generator = torch.Generator(device=device).manual_seed(args.seed + 101)
    cotangent = torch.randn(captured_state.shape, generator=generator, device=device)
    cotangent = cotangent / math.sqrt(adapter.state_dim)
    action = .1 * torch.randn((*captured_state.shape[:-1], adapter.action_dim),
                               generator=generator, device=device)
    command_uniforms = torch.rand((*captured_state.shape[:-1], adapter.command_noise_dim),
                                  generator=generator, device=device)
    sensor_uniforms = torch.rand((*captured_state.shape[:-1], adapter.sensor_noise_dim),
                                 generator=generator, device=device)

    def actor_projection(state, predicted, action, command_uniforms, sensor_uniforms):
        return adapter.project_next(state, predicted, action, exogenous={
            "command_uniforms": command_uniforms, "sensor_uniforms": sensor_uniforms})

    def training_projection(state, predicted, action, target):
        return adapter.project_next(state, predicted, action, exogenous={"next_state": target})

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def evaluate(function, inputs):
        leaves = tuple(value.detach().clone().requires_grad_(True) for value in inputs)
        synchronize()
        started = time.perf_counter()
        output = function(*leaves)
        gradients = torch.autograd.grad((output * cotangent).sum(), leaves,
                                        allow_unused=True)
        synchronize()
        return output.detach(), tuple(None if value is None else value.detach()
                                      for value in gradients), time.perf_counter() - started

    def comparison(eager, compiled, tolerance):
        if eager is None or compiled is None:
            return {"passed": eager is None and compiled is None,
                    "eager_unused": eager is None, "compiled_unused": compiled is None,
                    "max_abs_diff": None}
        finite = bool(torch.isfinite(eager).all() & torch.isfinite(compiled).all())
        return {"passed": finite and bool(torch.allclose(eager, compiled,
                                                         atol=tolerance, rtol=tolerance)),
                "finite": finite,
                "max_abs_diff": float((eager - compiled).abs().max()) if finite else None}

    # A clock just before each boundary exercises the increment into that tick.
    cases = [(f"next_tick_{tick}", tick - 1, False, 20.) for tick in (120, 150, 285, 300)]
    cases += [("standing_reset_151", 150, True, 20.),
              ("timer_equal_step_dt", 10, False, adapter.step_dt),
              ("timer_less_than_step_dt", 10, False, adapter.step_dt / 2)]
    rows = []
    with torch.enable_grad():  # check() deliberately runs under global no_grad.
        compiled_functions = {
            "actor_uniforms": torch.compile(actor_projection, backend=args.compile_backend,
                                             fullgraph=True, dynamic=False),
            "training_next_state": torch.compile(training_projection, backend=args.compile_backend,
                                                  fullgraph=True, dynamic=False),
        }
        for case_name, clock, standing, timer in cases:
            fields = adapter.layout.unpack(captured_state.detach().clone())
            fields["phase_length_buf"] = torch.full_like(fields["phase_length_buf"], clock)
            fields["still"] = torch.full_like(fields["still"], float(standing))
            phase = torch.remainder(fields["phase_length_buf"] * adapter.step_dt / adapter.cycle_time, 1.)
            ratios = adapter.env.cfg.phase_ratios
            phase_masks = [(phase >= ratios[left]) & (phase < ratios[right])
                           for left, right in ((0, 1), (2, 3), (1, 2), (3, 4))]
            fields["contact_phase"] = torch.stack(phase_masks, -1).to(captured_state.dtype)
            if standing:
                fields["contact_phase"] = torch.zeros_like(fields["contact_phase"])
            fields["amp_gate"] = fields["contact_phase"][..., 0]
            command = fields["command.skate.command_b"].clone()
            command[..., 0], command[..., 1] = (0. if standing else 1.), .2
            fields["command.skate.command_b"] = command
            fields["command.skate.time_left"] = torch.full_like(fields["command.skate.time_left"], timer)
            fields["command.skate.is_heading_env"] = torch.ones_like(fields["command.skate.is_heading_env"])
            fields["command.skate.is_standing_env"] = torch.full_like(
                fields["command.skate.is_standing_env"], float(standing))
            state = adapter.layout.pack(fields).detach()
            # Use the same valid captured physical predictions, but independent
            # previous/prediction/target leaves for the derivative comparison.
            predicted, target = captured_state.detach(), captured_state.detach()
            for path, eager_function, inputs, external_names in (
                ("actor_uniforms", actor_projection,
                 (state, predicted, action, command_uniforms, sensor_uniforms),
                 ("command_uniforms", "sensor_uniforms")),
                ("training_next_state", training_projection,
                 (state, predicted, action, target), ("target_next_state",)),
            ):
                row = {"case": case_name, "path": path, "previous_phase_tick": clock,
                       "standing": standing, "command_time_left": timer}
                try:
                    eager_value, eager_grad, eager_seconds = evaluate(eager_function, inputs)
                    compiled_value, compiled_grad, compiled_seconds = evaluate(compiled_functions[path], inputs)
                    row["output"] = comparison(eager_value, compiled_value, args.tolerance)
                    row["gradients"] = {
                        name: comparison(eager_grad[index], compiled_grad[index], args.gradient_tolerance)
                        for index, name in enumerate(("previous_state", "predicted_next_state", "action"))}
                    for index, name in enumerate(("previous_state", "predicted_next_state", "action")):
                        present = eager_grad[index] is not None and compiled_grad[index] is not None
                        row["gradients"][name]["present"] = present
                        row["gradients"][name]["passed"] &= present
                    row["external_inputs"] = {
                        name: {"passed": eager_grad[index + 3] is None and compiled_grad[index + 3] is None,
                               "eager_gradient_none": eager_grad[index + 3] is None,
                               "compiled_gradient_none": compiled_grad[index + 3] is None}
                        for index, name in enumerate(external_names)}
                    row["passed"] = (row["output"]["passed"]
                                     and all(value["passed"] for value in row["gradients"].values())
                                     and all(value["passed"] for value in row["external_inputs"].values()))
                    row["eager_seconds"] = eager_seconds
                    row["compiled_seconds_including_first_compile"] = compiled_seconds
                except Exception as error:
                    row.update(passed=False, error=type(error).__name__, message=str(error))
                rows.append(row)
    return {"enabled": True, "passed": all(row["passed"] for row in rows),
            "backend": args.compile_backend, "fullgraph": True,
            "output_atol_rtol": args.tolerance, "gradient_atol_rtol": args.gradient_tolerance,
            "scope": "projection values and fixed-cotangent input VJPs; no simulator or policy update",
            "cases": rows}


def check(args) -> dict:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("WANDB_MODE", "disabled")
    import torch
    import mjlab_husky.tasks  # noqa: F401 -- register the original task.
    from mjlab.utils.torch import configure_torch_backends
    from mjlab_husky.envs import G1SkaterManagerBasedRlEnv
    from mjlab_husky.supermimic.adapter import SkateWorldModelAdapter, roll_history
    from mjlab_husky.tasks.registry import load_env_cfg, load_rl_cfg

    configure_torch_backends()
    torch.manual_seed(args.seed)
    env_cfg = load_env_cfg(args.task, play=False)
    agent_cfg = load_rl_cfg(args.task)
    env_cfg.scene.num_envs, env_cfg.seed = args.num_envs, args.seed
    env = G1SkaterManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    try:
        with torch.no_grad():
            # Keep exactly the observations returned by reset/step. Never call
            # get_observations()/compute() again to build a second history/noise.
            observations, _ = env.reset()
            observations = {name: value.detach().clone() for name, value in observations.items()}
            current_amp = env.get_amp_observations().detach().clone()
            amp_history = current_amp.new_zeros((env.num_envs, agent_cfg.amp_num_frames,
                                                current_amp.shape[-1]))
            amp_history = roll_history(amp_history, current_amp)
            adapter = SkateWorldModelAdapter(env, observations, amp_history)
            generator = torch.Generator(device=env.device).manual_seed(args.seed + 1)
            groups = {}
            for group, terms in adapter._obs_terms.items():
                groups[group] = {term.name: {
                    "width": term.stop - term.start,
                    "frame_width": term.frame_width,
                    "history": term.history,
                    "projection": ("learned privileged target; unchanged"
                                   if term.name in ("trans_target_pos_b", "trans_target_quat_b")
                                   else "analytic physical state / external context / history"),
                    "max_abs_diff": 0.0,
                    "passed": True,
                } for term in terms}
            field_errors = {name: 0.0 for name in adapter.layout.shapes}
            valid_total = reset_total = 0
            finite = True
            for _ in range(args.steps):
                state = adapter.capture(observations, amp_history)
                action = args.action_noise * torch.randn(
                    (env.num_envs, env.action_manager.total_action_dim),
                    generator=generator, device=env.device)
                next_obs, _, terminated, truncated, _ = env.step(action)
                next_obs = {name: value.detach().clone() for name, value in next_obs.items()}
                next_amp = env.get_amp_observations().detach().clone()
                next_history = roll_history(amp_history, next_amp)
                target = adapter.capture(next_obs, next_history)
                # The environment manager owns the actually executed action.
                executed = env.action_manager.action.detach().clone()
                projected = adapter.project_next(state, target, executed,
                                                  exogenous={"next_state": target})
                done = (terminated | truncated).bool().reshape(-1)
                valid = ~done
                valid_total += int(valid.sum())
                reset_total += int(done.sum())
                if valid.any():
                    for key, group in (("actor_obs", adapter.actor_group),
                                       ("critic_obs", adapter.critic_group)):
                        actual = adapter.layout.get(projected, key)[valid]
                        expected = adapter.layout.get(target, key)[valid]
                        for term in adapter._obs_terms[group]:
                            error = (actual[..., term.start:term.stop]
                                     - expected[..., term.start:term.stop]).abs()
                            is_finite = bool(torch.isfinite(error).all())
                            finite &= is_finite
                            difference = float(error.max()) if is_finite else None
                            row = groups[group][term.name]
                            row["max_abs_diff"] = (max(row["max_abs_diff"], difference)
                                                   if difference is not None and row["max_abs_diff"] is not None else None)
                            row["passed"] &= is_finite and difference <= args.tolerance
                    for name in adapter.layout.shapes:
                        difference = (adapter.layout.get(projected, name)[valid]
                                      - adapter.layout.get(target, name)[valid]).abs()
                        is_finite = bool(torch.isfinite(difference).all())
                        finite &= is_finite
                        field_errors[name] = (max(field_errors[name], float(difference.max()))
                                              if is_finite and field_errors[name] is not None else None)
                amp_history = next_history
                amp_history[done] = 0  # Same reset convention as the real runner.
                observations = next_obs
            actor_pass = all(row["passed"] for row in groups[adapter.actor_group].values())
            critic_pass = all(row["passed"] for row in groups[adapter.critic_group].values())
            identity_pass = finite and all(value is not None and value <= args.tolerance
                                          for value in field_errors.values())
            compiled_report = {"enabled": False, "passed": True}
            if args.compile_projection:
                captured_state = adapter.capture(observations, amp_history)
                compiled_report = check_compiled_projection(adapter, captured_state, args)
            passed = bool(valid_total and actor_pass and critic_pass and identity_pass
                          and compiled_report["passed"])
            return {
                "passed": passed,
                "scope": "projection identity given true next physics; learned dynamics and policy quality not tested",
                "task": args.task, "device": args.device, "seed": args.seed,
                "num_envs": args.num_envs, "steps": args.steps,
                "action_noise_std": args.action_noise, "tolerance": args.tolerance,
                "valid_transitions": valid_total, "reset_transitions_excluded": reset_total,
                "schema": adapter.metadata(),
                "dimensions": {"actor": adapter.actor_dim, "critic": adapter.critic_dim,
                               "packed_state": adapter.state_dim,
                               "privileged_outside_actor": adapter.state_dim - adapter.actor_dim},
                "actor_passed": actor_pass, "critic_passed": critic_pass,
                "terms": groups,
                "compiled_projection": compiled_report,
                "model_strict_identity": {
                    "passed": identity_pass,
                    "definition": "project(previous, true_next, executed_action, observed_exogenous) equals true_next",
                    "learned_model_not_tested": True,
                    "max_abs_diff": max(field_errors.values()) if finite else None,
                    "fields": field_errors,
                },
            }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Mjlab-Skater-Flat-Unitree-G1")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--action-noise", type=float, default=.1)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--compile-projection", action="store_true")
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--gradient-tolerance", type=float, default=1e-4)
    parser.add_argument("--output", default="logs/adapter_identity_check.json")
    args = parser.parse_args()
    if (min(args.num_envs, args.steps) < 1 or not math.isfinite(args.action_noise)
            or not math.isfinite(args.tolerance) or not math.isfinite(args.gradient_tolerance)
            or args.action_noise < 0 or min(args.tolerance, args.gradient_tolerance) <= 0):
        parser.error("positive num-envs/steps/tolerance and nonnegative action-noise required")
    try:
        report = check(args)
    except Exception as error:
        report = {"passed": False, "error": type(error).__name__, "message": str(error)}
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(encoded, encoding="utf-8")
    print(encoded, end="", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
