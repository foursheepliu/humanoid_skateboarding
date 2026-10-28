"""Bounded real-task only-A integration check, never a policy-quality run.

This script is prepared separately from training. Running it requires an idle
GPU; it changes only its own fresh environment/runner and writes one JSON report.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from mjlab.utils.torch import configure_torch_backends
from mjlab_husky.envs import G1SkaterManagerBasedRlEnv
from mjlab_husky.rl import RslRlVecEnvWrapper
from mjlab_husky.supermimic.config import SuperMimicConfig
from mjlab_husky.supermimic.runner import SuperMimicRunner
from mjlab_husky.tasks.registry import load_env_cfg, load_rl_cfg


def fingerprint(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def amp_normalizer_fingerprint(runner):
    return fingerprint({name: torch.as_tensor(value) for name, value in
                        runner.amp.state_dict()["normalizer"].items()})


def main(output):
    started = time.perf_counter()
    configure_torch_backends()
    torch.manual_seed(42)
    cfg = load_env_cfg("Mjlab-Skater-Flat-Unitree-G1", play=False)
    cfg.scene.num_envs, cfg.seed = 16, 42
    agent = asdict(load_rl_cfg("Mjlab-Skater-Flat-Unitree-G1"))
    agent["seed"], agent["amp_num_preload_transitions"] = 42, 128
    settings = SuperMimicConfig(
        analytic_action_cost=True, warmup_iterations=0, model_batch_size=16,
        actor_batch_size=8, critic_updates=1, amp_updates=1, amp_batch_size=16)
    agent["supermimic"] = asdict(settings)
    env = None
    try:
        print("STAGE create fresh N16 CUDA environment; only-A enabled", flush=True)
        env = RslRlVecEnvWrapper(G1SkaterManagerBasedRlEnv(cfg=cfg, device="cuda:0"),
                                clip_actions=agent.get("clip_actions"))
        runner = SuperMimicRunner(env, agent, None, "cuda:0")
        assert runner.settings.horizon == 32
        assert runner.settings.model_hidden_dims == (512, 512, 512)
        assert runner.settings.compile_projection and runner.device.type == "cuda"
        assert runner.settings.detach_critic_transition_targets
        assert runner.critic_transition_target_dim == 98
        raw, layout = env.unwrapped, runner.adapter.layout
        known = runner.known_action_reward
        metadata = known.metadata()
        manager = raw.reg_reward_manager
        term_indices = [manager._term_names.index(name) for name in metadata["terms"]]
        scale = raw.step_dt if manager._scale_by_dt else 1.0
        assert scale == metadata["effective_scale"]

        # Advance only the clock of two fresh environments to exercise the
        # original timeout condition. No physical state/reward/termination
        # function is changed. Capture once from cached observations afterward.
        raw.episode_length_buf[:2] = raw.max_episode_length - 1
        runner._state = runner.adapter.capture(runner._obs, runner.amp_obs_frames).to(runner.device)
        original_step = env.step
        records = []
        post_failure_resets = 0
        post_timeout_resets = 0
        previous_terminated = torch.zeros(16, device=runner.device, dtype=torch.bool)
        previous_truncated = torch.zeros_like(previous_terminated)

        def checked_step(action):
            nonlocal post_failure_resets, post_timeout_resets
            nonlocal previous_terminated, previous_truncated
            pre_state = runner._state.detach().clone()
            previous_done = previous_terminated | previous_truncated
            if previous_done.any():
                # The following transition must start from reset action history,
                # while its predecessor's reward retained its pre-reset history.
                for field in ("action.action", "action.prev_action"):
                    assert torch.count_nonzero(layout.get(pre_state, field)[previous_done]).item() == 0
                post_failure_resets += int(previous_terminated.sum())
                post_timeout_resets += int(previous_truncated.sum())
            clip = env.clip_actions
            assert clip is None or bool((action.abs() <= clip).all())
            analytic = known(pre_state, action).detach().clone()
            result = original_step(action)
            _, task_reward, done, extras = result
            # _step_reward is weighted, unscaled-by-dt, and is not cleared by
            # RewardManager.reset. Read immediately, including terminal rows.
            cached = manager._step_reward[:, term_indices].sum(-1).detach().clone() * scale
            torch.testing.assert_close(analytic, cached, rtol=1e-5, atol=2e-7)
            terminated, truncated, boundary = runner._split_done(done, extras)
            records.append({"state": pre_state, "action": action.detach().clone(),
                            "reward": task_reward.detach().clone(), "known": analytic,
                            "cached": cached, "terminated": terminated.clone(),
                            "truncated": truncated.clone(), "next_valid": ~boundary})
            previous_terminated, previous_truncated = terminated.clone(), truncated.clone()
            return result

        collect_metrics = []
        env.step = checked_step
        try:
            print("STAGE collect real labels, forced timeouts and natural failures (max120 steps)", flush=True)
            for _ in range(24):
                collect_metrics.append(runner.collect())
                print(f"COLLECT steps={len(records)} post_failure={post_failure_resets} "
                      f"post_timeout={post_timeout_resets}", flush=True)
                if post_failure_resets and post_timeout_resets:
                    break
        finally:
            env.step = original_step
        rewards = torch.stack([record["reward"] for record in records])
        known_values = torch.stack([record["known"] for record in records])
        cached_values = torch.stack([record["cached"] for record in records])
        terminated = torch.stack([record["terminated"] for record in records])
        truncated = torch.stack([record["truncated"] for record in records])
        next_valid = torch.stack([record["next_valid"] for record in records])
        assert bool(truncated[0, :2].all())
        assert bool(terminated.any()) and post_failure_resets > 0 and post_timeout_resets > 0
        assert len(runner.replay) == rewards.numel()
        replay = runner.replay.state_dict()["data"]
        torch.testing.assert_close(replay["reward"], rewards, rtol=0, atol=0)
        torch.testing.assert_close(known(replay["state"], replay["action"]), known_values, rtol=0, atol=0)
        residual = (rewards - known_values).double()
        assert int(runner.world_model.reward_norm.count) == rewards.numel()
        assert int(runner.world_model.input_norm.count) == rewards.numel()
        assert int(runner.world_model.delta_norm.count) == int(next_valid.sum())
        torch.testing.assert_close(runner.world_model.reward_norm.mean, residual.mean().reshape(1),
                                   rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(runner.world_model.reward_norm.variance,
                                   residual.var(unbiased=False).reshape(1), rtol=1e-6, atol=1e-10)
        counts = {"alive": int(next_valid.sum()), "physical_terminal": int(terminated.sum()),
                  "timeout": int(truncated.sum()), "total_reward_labels": rewards.numel(),
                  "simultaneous_terminal_and_timeout": int((terminated & truncated).sum()),
                  "post_failure_reset_transitions": post_failure_resets,
                  "post_timeout_reset_transitions": post_timeout_resets}

        # Prediction.reward must keep total task units for existing probes/AMP.
        with torch.no_grad():
            state, action = replay["state"][0], replay["action"][0]
            model = runner.world_model
            hidden = model.trunk(model.input_norm(torch.cat((state, action), -1)))
            residual_prediction = model.reward_norm.decode(model.reward_head(hidden)).squeeze(-1)
            prediction = model(state, action)
            total = residual_prediction + known(state, action)
            assert torch.equal(prediction.reward, total)
            assert bool((known(state, action) != 0).any())
            assert bool((prediction.reward != residual_prediction).any())
        print("STAGE fit4 world updates and update1 AMP", flush=True)
        model_metrics = runner.fit_world_model()
        assert runner.world_model.optimizer_updates == 4
        assert runner._amp_real_histories is not None
        amp_metrics = runner.amp.update(runner._amp_real_histories, 1, 16)
        before_actor = {name: parameter.detach().clone() for name, parameter in runner.policy.actor.named_parameters()}
        before_critic = {name: parameter.detach().clone() for name, parameter in runner.policy.critic.named_parameters()}
        model_before = fingerprint(runner.world_model.state_dict())
        disc_before = fingerprint(runner.amp.discriminator.state_dict())
        policy_norms = (fingerprint(runner.policy.actor_obs_normalizer.state_dict()),
                        fingerprint(runner.policy.critic_obs_normalizer.state_dict()),
                        amp_normalizer_fingerprint(runner))
        print("STAGE H32 finite actor+critic update; fixed WM/D/normalizers", flush=True)
        updated = runner.update_actor_critic()
        assert model_before == fingerprint(runner.world_model.state_dict())
        assert disc_before == fingerprint(runner.amp.discriminator.state_dict())
        assert policy_norms == (fingerprint(runner.policy.actor_obs_normalizer.state_dict()),
                                fingerprint(runner.policy.critic_obs_normalizer.state_dict()),
                                amp_normalizer_fingerprint(runner))
        actor_delta = max(float((parameter.detach() - before_actor[name]).abs().max())
                          for name, parameter in runner.policy.actor.named_parameters())
        critic_delta = max(float((parameter.detach() - before_critic[name]).abs().max())
                           for name, parameter in runner.policy.critic.named_parameters())
        assert actor_delta > 0 and critic_delta > 0 and updated["actor/gradient_norm"] > 0
        assert all(math.isfinite(value) for value in updated.values())
        assert all(torch.isfinite(value).all() for module in (runner.policy, runner.world_model, runner.amp.discriminator)
                   for value in module.state_dict().values())
        assert runner.policy.std.grad is None and not runner.policy.std.requires_grad
        assert all(parameter.grad is None for parameter in runner.world_model.parameters())
        assert all(parameter.grad is None for parameter in runner.amp.discriminator.parameters())
        torch.cuda.synchronize()
        report = {"passed": True, "scope": "bounded fresh only-A integration, not training quality",
                  "settings": asdict(settings), "seed": 42, "num_envs": 16,
                  "gpu": torch.cuda.get_device_name(), "amp_expert_preload": 128,
                  "state_dim": runner.adapter.state_dim, "action_dim": runner.adapter.action_dim,
                  "actor_dim": runner.adapter.actor_dim, "critic_dim": runner.adapter.critic_dim,
                  "dynamic_critic_cut_dimensions": runner.critic_transition_target_dim,
                  "known_action_reward": metadata, "label_counts": counts,
                  "cached_action_cost_max_abs_error": float((known_values - cached_values).abs().max()),
                  "all_reward_labels_used_for_residual_normalization": True,
                  "prediction_reward_is_total_bitwise": True,
                  "real_total_reward_replay_unchanged": True,
                  "world_and_discriminator_frozen_during_actor_update": True,
                  "normalizers_unchanged_during_imagination": True,
                  "actor_max_parameter_delta": actor_delta, "critic_max_parameter_delta": critic_delta,
                  "collect": collect_metrics, "model": model_metrics, "amp": amp_metrics,
                  "actor_critic_update": updated, "wall_seconds": time.perf_counter() - started,
                  "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        main(output_path)
    except Exception as error:
        output_path.write_text(json.dumps({"passed": False, "error": type(error).__name__,
                                           "message": str(error)}, indent=2) + "\n")
        raise
