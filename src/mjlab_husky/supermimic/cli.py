"""Explicit bounded entry point for HUSKY coupled-world-model/BPTT training."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("WANDB_MODE", "disabled")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-name", default="train")
    parser.add_argument("--iterations", type=int, required=True,
                        help="Explicit total SuperMimic iterations; these are not PPO updates")
    parser.add_argument("--max-seconds", type=float,
                        help="Optional wall-time limit from warmup onward; stops after a completed update")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--replay-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--skip-final-checkpoint", action="store_true", help="Diagnostic runs: export actor only; no resume snapshot")
    parser.add_argument("--diagnostic-interval", type=int, default=0,
                        help="Optional policy/world diagnostic export interval; does not include replay or optimizer resume")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warm-start", type=Path, help="Policy-only warm start from legacy PPO checkpoint")
    parser.add_argument("--world-start", type=Path,
                        help="Explicit world weights and frozen-normalizer warm start; all learning/replay state starts fresh")
    parser.add_argument("--resume", type=Path, help="Full learning-state SuperMimic checkpoint")
    parser.add_argument("--joint-consistency", action="store_true",
                        help="Experimental exact AMP/critic robot-joint projection from predicted qpos/qvel")
    for name, default in [("actor-lr", 3e-6), ("teacher-kl-coefficient", 10.0),
            ("teacher-kl-limit", 0.003), ("teacher-max-action-delta", 0.03),
            ("teacher-max-standardized-delta", 0.1)]:
        parser.add_argument("--" + name, type=float, default=default)
    for name, default in [("warmup-steps", 320), ("replay-steps", 384), ("collect-steps", 5),
            ("collect-interval", 2), ("model-window", 8), ("model-updates", 10),
            ("model-batch", 256), ("imagination-horizon", 32), ("imagination-batch", 256),
            ("hidden-width", 512), ("expert-transitions", 200000), ("save-interval", 100)]:
        parser.add_argument("--" + name, type=int, default=default)
    args = parser.parse_args()
    if args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds <= 0):
        parser.error("--max-seconds must be finite and positive")
    if args.diagnostic_interval < 0:
        parser.error("--diagnostic-interval must be nonnegative")
    if args.warm_start and args.resume:
        parser.error("Choose warm start or resume")
    if args.world_start and args.resume:
        parser.error("Choose world-weight warm start or full resume")
    import mjlab.tasks  # Initialize the base package before custom plugin discovery.
    import numpy as np
    import torch
    from mjlab.utils.torch import configure_torch_backends
    from mjlab_husky.envs import G1SkaterManagerBasedRlEnv
    from mjlab_husky.tasks.registry import load_env_cfg, load_rl_cfg
    from rsl_rl.modules import ActorCritic
    from tensordict import TensorDict
    from mjlab_husky.supermimic.config import SuperMimicConfig
    from mjlab_husky.supermimic.trainer import SuperMimicTrainer

    configure_torch_backends()
    torch.set_num_threads(1)
    fields = ("iterations", "num_envs", "seed", "replay_device", "warmup_steps", "replay_steps", "collect_steps",
        "collect_interval", "model_window", "model_updates", "model_batch", "imagination_horizon",
        "imagination_batch", "hidden_width", "expert_transitions", "save_interval", "actor_lr",
        "teacher_kl_coefficient", "teacher_kl_limit", "teacher_max_action_delta", "teacher_max_standardized_delta",
        "joint_consistency")
    if args.resume:
        import sys
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved.get("format") != "husky-supermimic-v1":
            raise ValueError("Use --warm-start for a legacy PPO policy")
        if saved.get("dynamics_protocol") != "conditional-commands-v2":
            raise ValueError("World-model command protocol differs; explicit migration is required")
        settings = dict(saved["config"])
        settings.setdefault("joint_consistency", False)
        for name in fields:
            flag = "--" + name.replace("_", "-")
            if name != "iterations" and any(x == flag or x.startswith(flag + "=") for x in sys.argv[1:]):
                if getattr(args, name) != settings[name]:
                    raise ValueError(f"Resume cannot change {name}")
        settings["iterations"] = args.iterations
        cfg = SuperMimicConfig(**settings)
        del saved
    else:
        cfg = SuperMimicConfig(**{name: getattr(args, name) for name in fields})
    cfg.validate()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    args.out = args.out.resolve()
    if (args.out / "log" / args.run_name / "updates.jsonl").exists():
        raise FileExistsError("Use a fresh run name, including when resuming, to preserve the previous ledger")
    env_cfg = load_env_cfg("Mjlab-Skater-Flat-Unitree-G1")
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.seed = cfg.seed
    env = G1SkaterManagerBasedRlEnv(env_cfg, device=args.device)
    obs, _ = env.reset()
    agent = load_rl_cfg("Mjlab-Skater-Flat-Unitree-G1")
    pcfg = asdict(agent.policy)
    pcfg.pop("class_name")
    policy = ActorCritic(TensorDict(obs, batch_size=[cfg.num_envs]), agent.obs_groups,
                        env.action_manager.total_action_dim, **pcfg).to(args.device)
    if args.warm_start:
        saved = torch.load(args.warm_start, map_location=args.device, weights_only=False)
        policy.load_state_dict(saved["model_state_dict"])
    trainer = SuperMimicTrainer(env, policy, cfg, args.out, run_name=args.run_name)
    trainer.warm_started = bool(args.warm_start)
    if args.world_start:
        saved_world = torch.load(args.world_start, map_location="cpu", weights_only=False)
        if saved_world.get("world_architecture") != trainer.world.architecture() or saved_world.get("layout") != trainer.adapter.layout:
            raise ValueError("World warm-start architecture or augmented state layout differs")
        if saved_world.get("joint_consistency", False) and not cfg.joint_consistency:
            raise ValueError("This world used joint projection; explicitly enable --joint-consistency")
        if saved_world.get("joint_consistency", False):
            previous_spec = saved_world.get("joint_projection_metadata") or {}
            current_spec = trainer.adapter.joint_projection_metadata or {}
            if previous_spec.get("spec_sha256") != current_spec.get("spec_sha256"):
                raise ValueError("World warm-start joint geometry/defaults metadata differs")
        trainer.world.load_state_dict(saved_world["world"])
        if not all(bool(torch.isfinite(value).all()) for value in trainer.world.state_dict().values()):
            raise FloatingPointError("World warm-start contains nonfinite parameters or normalization")
        if any(float(getattr(trainer.world, name).count) <= 0 for name in
               ("state_norm", "action_norm", "delta_norm", "reward_norm")):
            raise ValueError("World warm-start must contain fitted real-data normalizers")
        trainer.normalization_fitted = True
        del saved_world

    def export_diagnostic(directory):
        directory.mkdir(parents=True, exist_ok=True)
        exports = {
            "policy_final.pt": {"model_state_dict": policy.state_dict(), "algorithm": "supermimic",
                "iteration": trainer.iteration, "resume_scope": "policy-only diagnostic export"},
            "model_diagnostic.pt": {"world": trainer.world.state_dict(), "world_architecture": trainer.world.architecture(),
                "model_state_dict": policy.state_dict(),
                "discriminator": trainer.discriminator.state_dict(),
                "style_normalizer": trainer.style_norm.state_dict(),
                "warm_started": trainer.warm_started,
                "reference_actor": trainer.reference_actor.state_dict(),
                "reference_normalizer": trainer.reference_normalizer.state_dict(),
                "reference_std": trainer.reference_std,
                "validation_batch": trainer.validation_batch, "layout": trainer.adapter.layout,
                "history_terms": trainer.adapter.history_terms, "critic_terms": trainer.adapter.critic_terms,
                "qpos_quaternion_slices": trainer.adapter.qpos_quaternion_slices,
                "config": asdict(cfg), "iteration": trainer.iteration,
                "dynamics_protocol": "conditional-commands-v2",
                "joint_consistency": cfg.joint_consistency,
                "projection_variant": trainer.adapter.joint_projection_variant,
                "joint_projection_metadata": trainer.adapter.joint_projection_metadata,
                "scope": "model, policy and style diagnostics only; no replay or optimizer resume"},
        }
        for name, contents in exports.items():
            temporary = (directory / name).with_suffix(".tmp")
            torch.save(contents, temporary)
            temporary.replace(directory / name)

    started = time.time()
    terminal = {"status": "failed", "iterations": 0, "error": "Initialization interrupted"}
    try:
        if args.resume:
            trainer.load(args.resume)
            if hasattr(trainer.replay, "break_continuity"):
                trainer.replay.break_continuity()
        else:
            trainer.collect(cfg.warmup_steps)
        warmup_finished = time.time()
        while trainer.iteration < cfg.iterations:
            if args.max_seconds is not None and time.time() - started >= args.max_seconds:
                break
            record = trainer.update()
            if trainer.iteration % cfg.log_interval == 0 or trainer.iteration == 1:
                print(json.dumps(record), flush=True)
            if trainer.iteration % cfg.save_interval == 0 and trainer.iteration < cfg.iterations:
                trainer.save(trainer.paths["checkpoints"] / f"training_{trainer.iteration}.pt")
            if args.diagnostic_interval and trainer.iteration % args.diagnostic_interval == 0:
                export_diagnostic(trainer.paths["checkpoints"] / f"diagnostic_{trainer.iteration}")
        if not args.skip_final_checkpoint:
            trainer.save(trainer.paths["checkpoints"] / "final.pt")
        export_diagnostic(trainer.paths["checkpoints"])
        terminal = {"status": "completed" if trainer.iteration >= cfg.iterations else "stopped_time_limit",
                    "iterations": trainer.iteration,
                    "real_env_steps": trainer.real_steps, "elapsed_seconds": time.time()-started,
                    "warmup_seconds": warmup_finished-started,
                    "max_seconds": args.max_seconds,
                    "world_warm_start": str(args.world_start.resolve()) if args.world_start else None,
                    "budget": cfg.iterations, "algorithm": "coupled-world-model+BPTT+adversarial-style"}
    except BaseException as error:
        terminal = {"status": "failed", "iterations": trainer.iteration, "error": repr(error)}
        raise
    finally:
        (trainer.paths["data"] / "exit.json").write_text(json.dumps(terminal, indent=2))
        env.close()


if __name__ == "__main__":
    main()
