# HUSKY coupled-world-model / BPTT backend

This branch adapts the training method from [Breezewrf/supermimic_mjlab](https://github.com/Breezewrf/supermimic_mjlab), pinned at `77183062eb92026bfbee2daf276f87cad36fd957`, to the HUSKY robot–skateboard system. It adds a genuinely learned dynamics/reward/failure model, autoregressive model fitting, and actor gradients through frozen learned dynamics. It does **not** rename or compile PPO and call it SuperMimic.

The original HUSKY environment, physics, rewards, command ranges, randomization, actor/critic architecture and reference data remain available. The new learner is a separate entry point and algorithm. The upstream humanoid-only model and its reported convergence numbers cannot be reused as skateboarding evidence.

## Train on the 4090 D

From the repository root:

```sh
uv sync --python 3.12 --locked
uv run --no-sync python tools/train_supermimic.py \
  --out /absolute/path/to/experiment --run-name train \
  --iterations 1000 --num-envs 4096 \
  --warm-start /absolute/path/to/legacy_husky_policy.pt
```

Every invocation requires an explicit iteration cap. `--warm-start` imports only the original policy and observation-normalizer weights; the world model, style discriminator and optimizers start fresh. It is not exact continuation of PPO. The original HUSKY checkpoint format omits discriminator weights and AMP normalization state.

The default starting workload is FP32, 5 real steps every 2 learner iterations, an 8-step model-fitting window, 10 model batches of 256, and a 32-step imagined actor rollout with 256 starts. PPO updates and these iterations have different definitions and are not comparable units of progress.

The default 320-step warmup covers a complete 6-second phase cycle. The 384-step FP32 replay lives in **host RAM**, while model and policy batches run on CUDA. At 4096 environments this needs roughly 18–20 GB of host storage plus sampling/checkpoint overhead; allow at least 48 GB available host RAM. This avoids filling a 24-GB GPU with augmented-state history. `--replay-device cuda` is available for a smaller replay or a larger-memory GPU. The default does not pin the entire host ring.

A bounded wiring test can explicitly use a shorter warmup; it does not validate the complete phase distribution:

```sh
uv run --no-sync python tools/train_supermimic.py \
  --out /absolute/path/to/experiment --run-name smoke \
  --iterations 2 --num-envs 128 --warmup-steps 32 --replay-steps 64 \
  --model-updates 2 --model-batch 64 --imagination-batch 32 \
  --hidden-width 128 --expert-transitions 2000 \
  --warm-start /absolute/path/to/legacy_husky_policy.pt
```

## What is learned and what remains exact

The adapter includes robot and skateboard qpos/qvel/actuation, actor/critic observation history, phase clocks, cached Bezier endpoints, action history, contact timers and physical randomization context. Actor histories are **term-major**, as in the pinned mjlab implementation. Projection preserves known history shifts, command/action copies, phase clocks and quaternion normalization. It is an augmented-state approximation; not every internal simulator or stochastic state is guaranteed to be represented.

Real rewards come from the unchanged HUSKY reward calculation. The model learns task reward and physical-failure heads from that data. Episode reset and physical failure have separate masks: time-outs cut real replay sequences but do not become learned physical failures. Terminal physics is captured before reset; stale terminal observation-history targets are excluded from state regression. Eight-step windows never join two episodes.

Actor BPTT freezes world, critic and style-discriminator weights while retaining gradients with respect to their inputs. The terminal critic bootstrap keeps a state Jacobian; critic targets and input trajectories are detached before critic fitting. The actor reuses the original HUSKY network and compatible `model_state_dict` keys.

HUSKY **AMP means Adversarial Motion Prior**. Its discriminator architecture, expert pushing motions, five-frame joint windows, least-squares objective, gradient penalty and push-only style coefficient `5 * 0.02` are retained. The new learner has its own GPU raw-joint moment normalization, separate discriminator optimizer and explicit collection noise. Those are new-backend choices, not a claim of PPO numerical equivalence. SuperMimic's unrelated `amp=bf16` option means automatic mixed precision; this first profile uses FP32 and does not enable it.

## Resume and export

```sh
uv run --no-sync python tools/train_supermimic.py \
  --out /absolute/path/to/experiment --run-name resumed \
  --iterations 2000 --resume /absolute/path/to/final.pt
```

Resume infers the saved learning workload and permits an explicitly extended total iteration cap; incompatible learning/state changes fail. Full snapshots include world, original actor/critic, discriminator, all normalization statistics, four optimizers, episode-aware replay/context versions, expert frames, counters and Python/NumPy/Torch/CUDA RNG. New real simulators start fresh trajectories and replay continuity is broken, so exact simulator trajectories are not claimed. The actor export remains usable by the original HUSKY play/evaluation path.

Full replay snapshots can be large and need additional host RAM and disk space. Diagnostic speed tests may use `--skip-final-checkpoint` to export only `policy_final.pt`; that file is deliberately not accepted as a full resume snapshot.

## Tests and measurement scope

```sh
CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m unittest discover -s tests -p 'test_supermimic_*.py' -v
```

Tests cover frozen-model actor gradients (including the board), terminal bootstrap, adversarial gradients, no reset leakage, term-major history projection, autoregressive fitting, holdout filtering, replay/context restore, and learning-state checkpoint roundtrips.

The last 10% of environment IDs are held out from model/actor fitting. A fixed held-out eight-step batch reports dynamics, reward and physical-failure errors. Normalization is fitted to the warmup distribution and then frozen; distribution drift and learned-reward exploitation remain important limitations. Short gradient/throughput tests do not establish stable long-run policy convergence.

Timing reports must show real simulator steps, imagined actor transitions, model fitting transitions, optimizer counts, initialization, validation, GPU memory and host memory separately. Lower latency per learner iteration does not prove less time to the same skateboarding success rate. The required quality comparison is a later real-simulation evaluation at matched conditions/returns.

## Attribution

Adapted protocol/code provenance and Apache-2.0 license are in `third_party/supermimic/`. HUSKY task/assets keep their original license terms.
