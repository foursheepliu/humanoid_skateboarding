"""Bounded fresh 2048x3 BF16 world-model / Bezier-hold integration, no quality claim."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.lab_api.math import subtract_frame_transforms
from mjlab_husky.envs import G1SkaterManagerBasedRlEnv
from mjlab_husky.rl import RslRlVecEnvWrapper
from mjlab_husky.supermimic.adapter import roll_history
from mjlab_husky.supermimic.config import SuperMimicConfig
from mjlab_husky.supermimic.runner import SuperMimicRunner
from mjlab_husky.tasks.registry import load_env_cfg, load_rl_cfg


def digest(value):
    h = hashlib.sha256()
    def visit(v):
        if isinstance(v, torch.Tensor):
            h.update(str((v.dtype, tuple(v.shape))).encode())
            h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str):
                h.update(str(k).encode()); visit(v[k])
        elif isinstance(v, (tuple, list)):
            for item in v: visit(item)
        else:
            h.update(repr(v).encode())
    visit(value)
    return h.hexdigest()


def training_bundle(runner):
    return {"policy": runner.policy.state_dict(), "world": runner.world_model.training_state_dict(),
            "actor_optimizer": runner.actor_optimizer.state_dict(),
            "critic_optimizer": runner.critic_optimizer.state_dict(), "amp": runner.amp.state_dict(),
            "rng": runner._rng.get_state()}


def main(output):
    started = time.perf_counter(); configure_torch_backends(); torch.manual_seed(42)
    cfg = load_env_cfg("Mjlab-Skater-Flat-Unitree-G1", play=False)
    cfg.scene.num_envs, cfg.seed = 32, 42
    agent = asdict(load_rl_cfg("Mjlab-Skater-Flat-Unitree-G1"))
    agent.update(seed=42, amp_num_preload_transitions=128)
    settings = SuperMimicConfig(project_transition_memory=True, model_precision="bfloat16",
        model_hidden_dims=(2048, 2048, 2048), analytic_action_cost=False, warmup_iterations=0,
        model_batch_size=32, actor_batch_size=32, amp_updates=1, amp_batch_size=32, critic_updates=1)
    agent["supermimic"] = asdict(settings)
    env = None
    try:
        print("STAGE create N32/B32 BF16 WM2048x3/H32, only-A off", flush=True)
        env = RslRlVecEnvWrapper(G1SkaterManagerBasedRlEnv(cfg=cfg, device="cuda:0"),
                                clip_actions=agent.get("clip_actions"))
        runner = SuperMimicRunner(env, agent, None, "cuda:0")
        raw, adapter = env.unwrapped, runner.adapter
        assert runner.critic_transition_target_dim == 98 and runner.known_action_reward is None
        # Prime clock/contact phase only to cover both real phase entries and
        # ordinary holds in one fresh step. Physical state/termination is intact.
        counters = torch.tensor([10, 120, 284, 200], device=runner.device).repeat(8)
        raw.phase_length_buf[:] = counters
        phase = torch.remainder(counters * raw.step_dt / raw.cycle_time, 1.)
        moving = raw.command_manager.get_command("skate")[:, 0] >= .1
        raw.contact_phase[:] = torch.stack(((phase < .4), (phase >= .5) & (phase < .95),
            (phase >= .4) & (phase < .5), phase >= .95), -1).float() * moving[:, None]
        runner._state = adapter.capture(runner._obs, runner.amp_obs_frames)
        original_step, record = env.step, {}
        def capture_first(action):
            previous = runner._state.detach().clone()
            result = original_step(action)
            if not record:
                obs, _, done, _ = result
                frames = roll_history(runner.amp_obs_frames, env.get_amp_observations())
                position, rotation = subtract_frame_transforms(
                    raw.skateboard.data.root_link_pos_w[:, None, :].expand(-1, raw.robot.num_bodies, -1),
                    raw.skateboard.data.root_link_quat_w[:, None, :].expand(-1, raw.robot.num_bodies, -1),
                    raw.robot.data.body_link_pos_w, raw.robot.data.body_link_quat_w)
                record.update(previous=previous, truth=adapter.capture(obs, frames), action=action.clone(),
                              valid=~done.bool(), position=position.clone(), rotation=rotation.clone())
            return result
        env.step = capture_first
        try:
            print("STAGE collect5, capture real anchor hold/entry", flush=True)
            collected = runner.collect()
        finally:
            env.step = original_step
        old, truth, action = record["previous"], record["truth"], record["action"]
        entries = {}
        for name, column, width in adapter._transition_memory_fields:
            entry = ((adapter.layout.get(truth, "contact_phase")[:, column] > .5)
                     & (adapter.layout.get(old, "contact_phase")[:, column] < .5) & record["valid"])
            hold = ~entry & record["valid"]
            before, after = adapter.layout.get(old, name), adapter.layout.get(truth, name)
            torch.testing.assert_close(after[hold], before[hold], rtol=0, atol=0)
            physics = record["position"] if width == 3 else record["rotation"]
            torch.testing.assert_close(after[entry], physics[entry], rtol=1e-6, atol=1e-6)
            assert entry.any() and hold.any(), (name, int(entry.sum()), int(hold.sum()))
            entries[name] = {"entry": int(entry.sum()), "hold": int(hold.sum())}
        print("STAGE compiled/eager projection values and VJP", flush=True)
        leaves = tuple(value.detach().clone().requires_grad_(True) for value in (old, truth, action))
        exogenous = {"next_state": truth.detach()}
        eager = adapter.project_next(*leaves, exogenous=exogenous)
        compiled = runner._project_next(*leaves, exogenous=exogenous)
        torch.testing.assert_close(compiled, eager, rtol=1e-5, atol=1e-5)
        cotangent = torch.randn_like(eager)
        ge = torch.autograd.grad((eager * cotangent).sum(), leaves)
        gc = torch.autograd.grad((compiled * cotangent).sum(), leaves)
        for left, right in zip(ge, gc, strict=True):
            torch.testing.assert_close(left, right, rtol=1e-4, atol=2e-5)
        projection_error = float((compiled - eager).abs().max().detach())
        vjp_error = max(float((left - right).abs().max()) for left, right in zip(ge, gc, strict=True))
        heads, other = [], []
        handles = [head.register_forward_hook(lambda _m, _i, out: heads.append(out.dtype))
                   for head in (runner.world_model.delta_head, runner.world_model.reward_head,
                                runner.world_model.termination_head)]
        handles += [network.register_forward_hook(lambda _m, _i, out: other.append(out.dtype))
                    for network in (runner.policy.actor, runner.policy.critic, runner.amp.discriminator)]
        try:
            prediction = runner.world_model(old, action)
            assert all(value.dtype == torch.float32 for value in prediction)
            print("STAGE 4 WM / 1 AMP / H32 actor+critic updates", flush=True)
            model_metrics = runner.fit_world_model()
            amp_metrics = runner.amp.update(runner._amp_real_histories, 1, 32)
            before_actor = digest(runner.policy.actor.state_dict())
            before_critic = digest(runner.policy.critic.state_dict())
            fixed = digest({"world": runner.world_model.state_dict(), "D": runner.amp.state_dict(),
                "actor_norm": runner.policy.actor_obs_normalizer.state_dict(),
                "critic_norm": runner.policy.critic_obs_normalizer.state_dict()})
            updated = runner.update_actor_critic()
            assert before_actor != digest(runner.policy.actor.state_dict())
            assert before_critic != digest(runner.policy.critic.state_dict())
            assert fixed == digest({"world": runner.world_model.state_dict(), "D": runner.amp.state_dict(),
                "actor_norm": runner.policy.actor_obs_normalizer.state_dict(),
                "critic_norm": runner.policy.critic_obs_normalizer.state_dict()})
            assert set(heads) == {torch.bfloat16} and set(other) == {torch.float32}
            assert all(math.isfinite(value) for value in updated.values())
            assert updated["actor/gradient_norm"] > 0
            for network in (runner.world_model, runner.policy, runner.amp.discriminator):
                assert all(parameter.dtype == torch.float32 and torch.isfinite(parameter).all()
                           for parameter in network.parameters())
            for state in runner.world_model.optimizer.state.values():
                assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
        finally:
            for handle in handles: handle.remove()
        print("STAGE complete native checkpoint roundtrip", flush=True)
        checkpoint_path = output.with_name("native_checkpoint.pt")
        runner.save(str(checkpoint_path))
        expected = digest(training_bundle(runner))
        # Resume constructs a fresh simulator. Reusing this stepped simulator
        # would expose lazily created task caches absent at initial construction
        # and therefore create a different adapter layout in this checker.
        resume_cfg = load_env_cfg("Mjlab-Skater-Flat-Unitree-G1", play=False)
        resume_cfg.scene.num_envs, resume_cfg.seed = 32, 42
        resume_env = RslRlVecEnvWrapper(G1SkaterManagerBasedRlEnv(cfg=resume_cfg, device="cuda:0"),
                                       clip_actions=agent.get("clip_actions"))
        try:
            restored = SuperMimicRunner(resume_env, agent, None, "cuda:0")
            restored.load(str(checkpoint_path))
            assert expected == digest(training_bundle(restored))
        finally:
            resume_env.close()
        torch.cuda.synchronize()
        report = {"passed": True, "scope": "bounded integration; no throughput or quality conclusion",
            "settings": asdict(settings), "num_envs": 32, "batch_size": 32,
            "state_dim": adapter.state_dim, "action_dim": adapter.action_dim,
            "dynamic_cut_dimensions": runner.critic_transition_target_dim, "real_anchor_counts": entries,
            "projection_max_abs_error": projection_error, "VJP_max_abs_error": vjp_error,
            "head_dtype": "bfloat16", "state_reward_termination_dtype": "float32",
            "actor_critic_discriminator_dtype": "float32", "master_weight_adam_dtype": "float32",
            "normalizer_accumulation_dtype": str(runner.world_model.input_norm.mean.dtype),
            "fixed_WM_D_all_normalizers_during_actor_update": True, "native_checkpoint_roundtrip": True,
            "native_checkpoint": str(checkpoint_path), "checkpoint_bundle_sha256": expected,
            "collect": collected, "model": model_metrics, "amp": amp_metrics, "actor_critic": updated,
            "wall_seconds": time.perf_counter()-started, "peak_memory_gib": torch.cuda.max_memory_allocated()/2**30}
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        if env is not None: env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", required=True)
    path = Path(parser.parse_args().output); path.parent.mkdir(parents=True, exist_ok=True)
    try:
        main(path)
    except Exception as error:
        path.write_text(json.dumps({"passed": False, "error": type(error).__name__, "message": str(error)}, indent=2)+"\n")
        raise
