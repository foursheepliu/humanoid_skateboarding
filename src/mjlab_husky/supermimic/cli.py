"""Explicit bounded entry point for HUSKY coupled-world-model/BPTT training."""
import argparse
from dataclasses import asdict
import json
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
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--replay-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--skip-final-checkpoint", action="store_true", help="Diagnostic runs: export actor only; no resume snapshot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warm-start", type=Path, help="Policy-only warm start from legacy PPO checkpoint")
    parser.add_argument("--resume", type=Path, help="Full learning-state SuperMimic checkpoint")
    for name, default in [("warmup-steps", 320), ("replay-steps", 384), ("collect-steps", 5),
            ("collect-interval", 2), ("model-window", 8), ("model-updates", 10),
            ("model-batch", 256), ("imagination-horizon", 32), ("imagination-batch", 256),
            ("hidden-width", 512), ("expert-transitions", 200000), ("save-interval", 100)]:
        parser.add_argument("--" + name, type=int, default=default)
    args = parser.parse_args()
    if args.warm_start and args.resume:
        parser.error("Choose warm start or resume")
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
        "imagination_batch", "hidden_width", "expert_transitions", "save_interval")
    if args.resume:
        import sys
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved.get("format") != "husky-supermimic-v1":
            raise ValueError("Use --warm-start for a legacy PPO policy")
        settings = dict(saved["config"])
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
    if (args.out / "log" / args.run_name / "updates.jsonl").exists() and not args.resume:
        raise FileExistsError("Use a fresh output directory or explicitly resume")
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
            record = trainer.update()
            if trainer.iteration % cfg.log_interval == 0 or trainer.iteration == 1:
                print(json.dumps(record), flush=True)
            if trainer.iteration % cfg.save_interval == 0:
                trainer.save(trainer.paths["checkpoints"] / f"training_{trainer.iteration}.pt")
        if not args.skip_final_checkpoint:
            trainer.save(trainer.paths["checkpoints"] / "final.pt")
        else:
            torch.save({"model_state_dict": policy.state_dict(), "algorithm": "supermimic", "iteration": trainer.iteration,
                        "resume_scope": "policy-only diagnostic export"}, trainer.paths["checkpoints"] / "policy_final.pt")
            torch.save({"world": trainer.world.state_dict(), "world_architecture": trainer.world.architecture(),
                        "validation_batch": trainer.validation_batch, "layout": trainer.adapter.layout,
                        "history_terms": trainer.adapter.history_terms, "critic_terms": trainer.adapter.critic_terms,
                        "qpos_quaternion_slices": trainer.adapter.qpos_quaternion_slices,
                        "config": asdict(cfg), "iteration": trainer.iteration,
                        "scope": "model diagnostics only, no replay or optimizer resume"},
                        trainer.paths["checkpoints"] / "model_diagnostic.pt")
        terminal = {"status": "completed", "iterations": trainer.iteration,
                    "real_env_steps": trainer.real_steps, "elapsed_seconds": time.time()-started,
                    "warmup_seconds": warmup_finished-started,
                    "budget": cfg.iterations, "algorithm": "coupled-world-model+BPTT+adversarial-style"}
    except BaseException as error:
        terminal = {"status": "failed", "iterations": trainer.iteration, "error": repr(error)}
        raise
    finally:
        (trainer.paths["data"] / "exit.json").write_text(json.dumps(terminal, indent=2))
        env.close()


if __name__ == "__main__":
    main()
