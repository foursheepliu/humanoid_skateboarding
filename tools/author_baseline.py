"""Run the released HUSKY baseline with bounded training and local evidence.

Run from the repository root. This harness does not modify the task or PPO/AMP.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("WANDB_MODE", "disabled")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def sha(path):
    return hashlib.file_digest(open(path, "rb"), "sha256").hexdigest()


def smoke(out):
    import mjlab.tasks
    import torch
    from mjlab_husky.envs import G1SkaterManagerBasedRlEnv
    from mjlab_husky.tasks.registry import load_env_cfg
    cfg = load_env_cfg("Mjlab-Skater-Flat-Unitree-G1")
    cfg.scene.num_envs = 4096
    cfg.seed = 42
    env = G1SkaterManagerBasedRlEnv(cfg, device="cuda:0")
    obs, _ = env.reset()
    for i in range(200):
        obs, reward, terminated, truncated, extras = env.step(
            torch.zeros((4096, env.action_manager.total_action_dim), device="cuda:0")
        )
        if not torch.isfinite(reward).all() or any(not torch.isfinite(x).all() for x in obs.values()):
            raise RuntimeError(f"Nonfinite smoke state at step {i}")
    write_json(out / "data/smoke.json", {
        "status": "passed", "num_envs": 4096, "steps": 200, "ppo_updates": 0,
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "obs_shapes": {k: list(v.shape) for k, v in obs.items()},
        "actions": env.action_manager.total_action_dim,
    })
    env.close()


def evaluate(out, checkpoint, steps, label):
    import mjlab.tasks
    import dataclasses
    import mujoco
    import numpy as np
    import torch
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    from rsl_rl.modules import ActorCritic
    from mjlab_husky.envs import G1SkaterManagerBasedRlEnv
    from mjlab_husky.rl import RslRlVecEnvWrapper
    from mjlab_husky.tasks.registry import load_env_cfg, load_rl_cfg
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()
    task = "Mjlab-Skater-Flat-Unitree-G1"
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = 1
    cfg.seed = 42
    # Fixed speed and relative heading, with the author's phase controller intact.
    cfg.commands["skate"].ranges.lin_vel_x = (0.8, 0.8)
    cfg.commands["skate"].ranges.heading = (0.5, 0.5)
    env = G1SkaterManagerBasedRlEnv(cfg, device="cuda:0")
    wrapped = RslRlVecEnvWrapper(env)
    agent = load_rl_cfg(task)
    policy_cfg = dataclasses.asdict(agent.policy)
    policy_cfg.pop("class_name")
    policy = ActorCritic(wrapped.get_observations(), agent.obs_groups, wrapped.num_actions, **policy_cfg).to("cuda:0")
    saved = torch.load(checkpoint, map_location="cuda:0", weights_only=False)
    policy.load_state_dict(saved["model_state_dict"])
    policy.eval()
    obs = wrapped.get_observations()
    qpos, phases, positions, commands, resets, heights = [], [], [], [], [], []
    with torch.inference_mode():
        for _ in range(steps):
            action = policy.act_inference(obs)
            obs, reward, done, _ = wrapped.step(action)
            if not torch.isfinite(obs["policy"]).all() or not torch.isfinite(reward).all():
                raise RuntimeError("Nonfinite evaluation state")
            qpos.append(env.sim.data.qpos[0].cpu().numpy().copy())
            phases.append(env.contact_phase[0].cpu().numpy().copy())
            positions.append(env.skateboard.data.root_link_pos_w[0].cpu().numpy().copy())
            commands.append(env.command_manager.get_command("skate")[0].cpu().numpy().copy())
            resets.append(bool(done[0]))
            heights.append(float(env.robot.data.root_link_pos_w[0, 2]))
    out.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(exist_ok=True)
    (out / "vedio").mkdir(exist_ok=True)
    (out / "image").mkdir(exist_ok=True)
    np.savez_compressed(out / "data" / f"{label}_trace.npz", qpos=qpos, phases=phases,
                        board_position=positions, commands=commands, root_height=heights, resets=resets)
    mujoco.mj_saveModel(env.sim.mj_model, str(out / "data" / f"{label}_scene.mjb"))
    video = render_trace(out, label, env.sim.mj_model, qpos, env.step_dt)
    write_json(out / "data" / f"{label}_evaluation.json", {
        "checkpoint": str(checkpoint), "sha256": sha(checkpoint),
        "checkpoint_iteration": saved.get("iter"), "steps": steps, "dt": env.step_dt,
        "ppo_updates": 0, "conditions": "author play config, seed42, speed0.8m/s, relative heading0.5rad",
        "phase_frame_counts": np.asarray(phases).sum(0).tolist(),
        "resets": sum(resets), "minimum_root_height_m": min(heights),
        "board_displacement_m": float(np.linalg.norm(np.asarray(positions)[-1, :2] - np.asarray(positions)[0, :2])),
        "limits": "One fixed-condition play rollout; play mode disables fall terminations. Not a success rate.",
        "camera_azimuths_relative_torso": [210, 120], "video": str(video), "video_sha256": sha(video),
    })
    wrapped.close()


def render_trace(out, label, model, qpos, dt):
    import mujoco
    import numpy as np
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    data = mujoco.MjData(model)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 1280)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 640)
    renderer = mujoco.Renderer(model, height=640, width=640)
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/torso_link")
    if root_id < 0:
        root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/pelvis")
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation = 3.2, -15
    steps = len(qpos)
    samples = set(np.linspace(0, min(steps - 1, round(6 / dt) - 1), 6).round().astype(int).tolist())
    thumbs = []
    video = out / "vedio" / f"{label}_front_side.mp4"
    fps = round(1 / dt)
    with imageio.get_writer(video, fps=fps, codec="libx264", quality=7) as writer:
        for i, q in enumerate(qpos):
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            camera.lookat[:] = data.xpos[root_id] if root_id >= 0 else q[:3]
            views = []
            torso_forward = data.xmat[root_id].reshape(3, 3)[:, 0]
            yaw = np.degrees(np.arctan2(torso_forward[1], torso_forward[0]))
            for azimuth in (yaw + 210, yaw + 120):
                camera.azimuth = azimuth
                renderer.update_scene(data, camera=camera)
                views.append(renderer.render().copy())
            frame = np.concatenate(views, axis=1)
            writer.append_data(frame)
            if i in samples:
                pic = Image.fromarray(views[0]).resize((320, 320))
                ImageDraw.Draw(pic).text((8, 8), f"t={i / fps:.2f}s", fill="black", stroke_width=1, stroke_fill="white")
                thumbs.append(pic)
    sheet = Image.new("RGB", (960, 640), "white")
    for i, pic in enumerate(thumbs):
        sheet.paste(pic, ((i % 3) * 320, (i // 3) * 320))
    sheet.save(out / "image" / f"{label}_six_frames.jpg")
    renderer.close()
    return video


def train(out):
    import mjlab.tasks
    import math
    import torch
    from mjlab_husky.scripts.train import TrainConfig, run_train
    from rsl_rl.runners import AMPOnPolicyRunner
    task = "Mjlab-Skater-Flat-Unitree-G1"
    cfg = TrainConfig.from_task(task)
    assert cfg.agent.max_iterations == 50000 and not cfg.agent.resume
    cfg.env.scene.num_envs = 4096
    # Local logging changes neither the algorithm nor task behavior.
    cfg.agent.logger = "tensorboard"
    cfg.agent.run_name = "author_baseline_20261028"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    run_dir = out / "checkpoints/author_baseline"
    run_dir.mkdir(parents=True, exist_ok=False)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "log").mkdir(parents=True, exist_ok=True)
    original_log = AMPOnPolicyRunner.log
    times = []
    start = time.time()

    def observed_log(self, locs, *args, **kwargs):
        update = int(locs["it"]) + 1
        if update != len(times) + 1 or update > 50000:
            raise RuntimeError(f"Update ledger or budget violation: {update}")
        losses = {k: float(v) for k, v in locs["loss_dict"].items()}
        if any(not math.isfinite(x) for x in losses.values()):
            raise RuntimeError(f"Nonfinite loss at update {update}: {losses}")
        seconds = float(locs["collection_time"] + locs["learn_time"])
        times.append(seconds)
        original_log(self, locs, *args, **kwargs)
        record = {"completed_updates": update, "iteration": update - 1, "seconds": seconds,
                  "losses": losses, "timestamp": time.time(), "budget": 50000}
        with (out / "log/updates.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if update == 50 or update % 50 == 0:
            if any(not torch.isfinite(p).all() for p in self.alg.policy.parameters()):
                raise RuntimeError(f"Nonfinite policy weights at {update}")
            mean = sum(times[-30:]) / len(times[-30:])
            write_json(out / "data/status.json", {**record, "status": "running",
                "mean_last30_seconds": mean, "eta_timestamp": time.time() + (50000-update)*mean,
                "elapsed_seconds": time.time()-start, "pid": os.getpid()})
        if update == 50 or update % 5000 == 0:
            cp = run_dir / f"model_{update-1}.pt"
            self.save(str(cp))
            write_json(out / f"data/checkpoint_{update}.json", {"updates": update, "path": str(cp), "sha256": sha(cp)})
            if update % 5000 == 0:
                eval_out = out
                with (out / "log" / f"eval_{update}.log").open("w") as stream:
                    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "eval",
                        "--out", str(eval_out), "--checkpoint", str(cp), "--steps", "1500", "--label", f"update{update}"],
                        stdout=stream, stderr=subprocess.STDOUT, timeout=1800)
                write_json(out / f"data/evaluation_{update}_exit.json", {"returncode": result.returncode})

    AMPOnPolicyRunner.log = observed_log
    try:
        run_train(task, cfg, run_dir)
    except BaseException as error:
        write_json(out / "data/exit.json", {"status": "failed", "error": repr(error), "timestamp": time.time()})
        raise
    final = {"status": "completed", "updates": 50000, "timestamp": time.time()}
    write_json(out / "data/exit.json", final)
    write_json(out / "data/status.json", final)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["smoke", "eval", "render", "train"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--label", default="author_pretrained")
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.mode == "smoke":
        smoke(args.out)
    elif args.mode == "eval":
        evaluate(args.out, args.checkpoint.resolve(), args.steps, args.label)
    elif args.mode == "render":
        import mujoco
        import numpy as np
        trace = np.load(args.out / "data" / f"{args.label}_trace.npz")
        model = mujoco.MjModel.from_binary_path(str(args.out / "data" / f"{args.label}_scene.mjb"))
        video = render_trace(args.out, args.label, model, trace["qpos"], 0.02)
        report = args.out / "data" / f"{args.label}_evaluation.json"
        if report.exists():
            data = json.loads(report.read_text())
            data.pop("camera_azimuths", None)
            data["camera_azimuths_relative_torso"] = [210, 120]
            data["video_sha256"] = sha(video)
            write_json(report, data)
    else:
        train(args.out)
