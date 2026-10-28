# HUSKY coupled-world-model / BPTT backend

Branch: `suppermimic_4090`. Evidence snapshot: 2026-10-29.

This branch adapts the training method from [Breezewrf/supermimic_mjlab](https://github.com/Breezewrf/supermimic_mjlab), pinned at `77183062eb92026bfbee2daf276f87cad36fd957`, to the HUSKY robot–skateboard system. It learns augmented robot–board dynamics, task reward and physical failure, fits autoregressive real-data windows, and trains the actor through frozen learned dynamics. This is a separate model-based algorithm; its gradients do not pass through the real MuJoCo simulator.

The original HUSKY environment, physics, reward calculation, command ranges, randomization, actor/critic architecture and reference data remain available. Upstream humanoid-only convergence results are not skateboard results. The measurements below show lower **latency per learner iteration**, with different sampling and optimization work from PPO; they do not establish faster convergence to equal skateboarding quality.

## Native command

From the repository root:

```sh
uv sync --python 3.12 --locked
uv run --no-sync husky-supermimic --help
uv run --no-sync husky-supermimic \
  --out /absolute/path/to/experiment --run-name train \
  --iterations 1000 --num-envs 4096 \
  --warm-start /absolute/path/to/legacy_husky_policy.pt
```

`husky-supermimic` is the installed entry point. `python tools/train_supermimic.py` remains a compatibility wrapper around the same implementation.

Every invocation requires an explicit total iteration cap. `--warm-start` imports the original actor/critic and observation-normalizer weights. The world model, style discriminator and optimizers start fresh; this is not full continuation of PPO. The legacy HUSKY checkpoint omits discriminator weights and AMP normalization state. Omitting `--warm-start` starts a fresh policy, but cold-start skateboarding convergence has **not** been demonstrated by these tests.

The default workload uses FP32, five real steps every two learner iterations, an eight-step autoregressive model window, ten model batches of 256, and a 32-step imagined actor rollout with 256 starts. Actor LR is `3e-6`; a warm-started actor also uses the teacher constraints below. PPO updates and these iterations are different units of training progress.

The default 320-step warmup spans 6.4 seconds, longer than the nominal six-second phase cycle; individual resets can still truncate a trajectory. The 384-step replay lives in **host RAM**, while sampled batches and networks run on CUDA. The measured replay allocation was 18,447,171,088 bytes, approximately 18.45 GB / 17.18 GiB, before sampling/checkpoint overhead. Allow at least 48 GB available host RAM. The whole ring is not pinned. `--replay-device cuda` is available for smaller rings or larger-memory GPUs.

A bounded wiring test can use a shorter warmup; it does not validate complete phase coverage:

```sh
uv run --no-sync husky-supermimic \
  --out /absolute/path/to/experiment --run-name smoke \
  --iterations 2 --num-envs 128 --warmup-steps 32 --replay-steps 64 \
  --model-updates 2 --model-batch 64 --imagination-batch 32 \
  --hidden-width 128 --expert-transitions 2000 \
  --warm-start /absolute/path/to/legacy_husky_policy.pt
```

## Learned state, task labels and actor gradients

The adapter captures robot and skateboard qpos/qvel/actuation, actor/critic observation history, phase clocks, cached Bezier endpoints, action history, contact timers and physical randomization context. Actor histories are **term-major**, matching pinned mjlab. Projection enforces known history shifts, command/action copies, phase clocks and quaternion normalization. This is an augmented-state approximation; it does not guarantee capture of every internal solver, sensor, actuator or stochastic state.

Real task rewards come from the unchanged HUSKY calculation. The model learns reward and physical-failure heads from those labels. Time-outs cut real replay sequences but are not physical-failure labels. Terminal physics is captured before reset; stale terminal observation-history targets are excluded from state regression, while terminal reward/failure labels remain usable. Eight-step windows never join two episodes.

Actor BPTT freezes world, critic and style-discriminator parameters while retaining their state/action Jacobians. Terminal critic bootstrap remains differentiable with respect to its input state. Critic trajectories and lambda-return targets are detached before fitting. The actor reuses the original HUSKY network and compatible `model_state_dict` keys.

HUSKY **AMP means Adversarial Motion Prior**. Its discriminator architecture, expert pushing motions, five-frame joint windows, least-squares objective, gradient penalty and push-only style coefficient `5 * 0.02` are retained. This learner uses raw-joint device moments, a separate discriminator optimizer and explicit collection noise. Those are new-backend choices, not PPO numerical equivalence. SuperMimic's unrelated `amp=bf16` option means automatic mixed precision; this profile uses FP32.

## Warm-start teacher constraints

An initial unguarded 50-iteration adaptation degraded the 10K teacher and fell during the fixed replay, reaching a minimum root height of **0.1157 m**. Lower iteration latency alone was therefore insufficient.

The warm-start path now freezes a teacher actor, its observation normalizer and action standard deviations. Its actor objective includes a coefficient-10 teacher KL penalty on real and detached imagined observation inputs. After an Adam proposal, parameter backtracking checks the fixed teacher on the first cached real-input batch plus the current real-input batch:

| Constraint | Limit |
| --- | ---: |
| Mean fixed-covariance teacher KL | 0.003 |
| Maximum raw action-output difference | 0.03 |
| Maximum difference divided by teacher std | 0.1 |

The KL is `0.5 * sum(((mu - mu_teacher) / std_teacher)^2)`, averaged over the checked inputs. Raw differences are in actor-output units. Backtracking tries at most ten scales. An accepted scaled proposal retains the full Adam moment update: this is **projected Adam**, not an unchanged Adam step or a faithful reproduction of upstream SuperMimic optimization. A rejected proposal restores both actor parameters and Adam state.

The limits constrain cumulative displacement from the frozen teacher on these sampled inputs. They do not provide a global state-space safety guarantee, and they can restrict further improvement. A cold-start policy does not use this warm-start teacher guard. BPTT gradients remain active; accepted nonzero actor steps must be reported so unchanged teacher playback is not mistaken for learning.

## Measured 4090 D iteration latency

The public, path-free data snapshot is [benchmarks/4090d.json](benchmarks/4090d.json). The run warm-started from the original HUSKY 10K policy. It used 4,096 environments, FP32 world layers `[512,512,512]`, model batch 256 × eight steps × ten updates, imagined batch 256 × 32 steps, and 20,000 expert preload windows. The measured expert preload is smaller than the CLI default of 200,000.

| Measurement | Original HUSKY PPO+AMP | Coupled model + BPTT + teacher constraints |
| --- | ---: | ---: |
| Timing window | PPO updates 7,530–8,529 | Learner iterations 21–50 |
| Mean iteration latency | 1.92694 s | **0.62010 s** |
| Real simulator transitions / iteration | 98,304 | 10,240 steady average |
| Imagined actor transitions / iteration | — | 8,192 |
| Model-fitting transitions / iteration | — | 20,480 |
| Actor optimizer updates / iteration | 5 epochs × 4 minibatches | 1 proposal, possibly projected |

The latency ratio is **3.107×**, for these different workloads. The real-sample count is 9.6× smaller per learner iteration; neither ratio is an equal-quality convergence speedup.

The stable model-based iteration consists of approximately 0.13882 s collection, 0.36620 s model/style fitting, 0.11417 s imagination/actor/critic, and 0.00092 s validation. Initialization and 1,310,720 warmup simulator transitions are excluded from that stable timing. The 50-iteration run used 1,822,720 real transitions including warmup. All 50 actor proposals were accepted; four required backtracking. Maximum recorded teacher KL/raw difference/standardized difference were 0.00299996 / 0.0299436 / 0.0992156.

Peak reported PyTorch allocation was 622,908,928 bytes. This counts the PyTorch allocator only, not total device usage from MuJoCo/Warp, CUDA context or driver allocations. These are RTX **4090 D** measurements; no A5000 measurement is implied.

## Fixed-condition physical replay

Both final policies below were evaluated for 1,500 steps × 0.02 seconds = 30 seconds, using the author's play configuration, seed 42, speed command 0.8 m/s and relative heading command 0.5 rad.

| Replay metric | Original 10K teacher | After 50 projected learner iterations |
| --- | ---: | ---: |
| Minimum root height | 0.66058 m | 0.66150 m |
| Frames with body tilt above 70° | 0 | 0 |
| Both feet on board during steering, raw contact threshold 5 N | 1.000 | 1.000 |
| Pushing velocity MAE | 0.13504 m/s | 0.12247 m/s |
| Steering heading MAE | 0.29379 rad | 0.27287 rad |
| Start-to-end board displacement | 26.132 m | 24.288 m |

The projected policy retained a no-fall replay under this **one fixed condition**. Play mode disables fall termination, so zero resets alone is not evidence of success. The tilt, root-height and contact diagnostics supplement the actual rollout. Small differences in the MAEs are not statistically established improvements. Displacement is not path length or velocity accuracy. The four phase occupancy counts are clock-driven and cannot establish successful physical transitions by themselves.

Teacher checkpoint SHA256: `528560e39ab45fe25c08fa4bac37c5795fe2fdd0054fb644f208e1e33d77275d` (legacy `iter=9999`, after 10,000 PPO updates).

Projected diagnostic policy SHA256: `54ab6deda873a6d844716fd27bdabdd6b9c2a0eed0d7d7585a0327ff4535a27e` (50 model-based learner iterations after warm start, not 50 PPO updates).

## Model validation and remaining evidence gaps

The last 10% of environment IDs are held out from fitting. A fixed, independently sampled eight-step validation batch is checkpointed with its own RNG. In the measured run, normalized autoregressive state loss fell from 6.0708 to 0.8193 and normalized reward loss from 0.53768 to 0.07603. These are fixed-batch model errors, not a skateboarding success rate. Normalization is fitted to the warmup distribution and then frozen; distribution drift and learned-reward exploitation remain limitations.

The following are **not established** by these measurements:

- Useful cold-start learning from a fresh policy.
- Held-out 32-step model accuracy matching the actor's imagination horizon; only eight-step model validation is reported.
- Less wall-clock time to the same success rate/return as PPO, including initialization, warmup and evaluation.
- Broad randomized robustness, multi-seed convergence, or real-robot deployment of this backend.

## Resume, export and tests

```sh
uv run --no-sync husky-supermimic \
  --out /absolute/path/to/experiment --run-name resumed \
  --iterations 2000 --resume /absolute/path/to/final.pt

CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m unittest discover \
  -s tests -p 'test_supermimic_*.py' -v
```

Resume infers the saved workload and permits an explicitly changed total cap above already completed iterations; incompatible learning/state changes fail. Full snapshots include world, actor/critic, discriminator, normalizers, four optimizers, episode-aware replay/context, expert frames, counters, teacher anchors/trust inputs, fixed validation data/RNG and Python/NumPy/Torch/CUDA RNG. Fresh real simulators reset and replay continuity is broken; exact physical trajectory continuation is not claimed. The compatible policy export is usable by the original HUSKY play/evaluation path.

Full replay checkpoints need additional host RAM and disk space. `--skip-final-checkpoint` produces a policy-only export and a separate model-diagnostic artifact; neither is a full resumable snapshot.

The tests cover gradients through the frozen coupled model/critic/style callback, reset masks, term-major history, model fitting, replay/context restoration, teacher acceptance/backtracking/rollback, independent validation sampling, compatible policy export and learning-state restore. The final remote log confirms **28 tests passed in 0.409 seconds**, including GPU-to-CPU replay transport. The installed native CLI completed a two-iteration real-environment smoke run and a full snapshot resume from iteration two to three. These checks validate wiring and learning-state restoration, not long-run convergence.

## Attribution

Adapted protocol/code provenance and the Apache-2.0 license are in `third_party/supermimic/`. HUSKY task/assets retain their original terms.

## Conditional command protocol (nightly follow-up)

The task samples a new relative heading when push-to-steer ends. The previous model projection incorrectly kept that command fixed. The conditional protocol supplies only recorded external command values during fitting and validation; imagined rollouts sample commands from the original task ranges. Physical state, heading reference and observations are never copied from a future real target. The heading reference is reconstructed from predicted board orientation. Still/phase transitions use the preceding command, matching the task's update order.

Full snapshots now carry `dynamics_protocol=conditional-commands-v2`. Earlier snapshots require explicit migration and are rejected as full resumes. Policy warm starts remain compatible. Resumed runs require a fresh run name to preserve prior ledgers.

The conditional implementation passed **39 tests**, including CUDA replay transport, and real-environment smoke, diagnostic exports and full learning-state resume. Its 50-step diagnostic loop averaged **0.71747 seconds** over the last 30 steps. The earlier 0.62010-second result above describes the previous implementation; the correctness fix adds work.

In a separate 64-environment, 30-second play evaluation over the original command ranges, teacher and conditional-50 policy had identical initial-state and complete command-sequence hashes. Both had five environments with observed root tilt above 70 degrees. Pooled pushing velocity MAE was 0.13532/0.13286 m/s and heading MAE 0.22158/0.22071 rad (teacher/candidate). This preserves behavior in this diagnostic sample; it does not establish a statistically significant improvement or a formal success rate.

`tools/audit_supermimic_model.py` measures 1/8/16/32-step open-loop predictions against the same recorded real actions, with physical-unit groups, phase-boundary groups and frozen normalizers. Changing only command projection improves command/observation consistency, with a small physical-state effect. A separately retrained 50-step model has mixed physical errors; comparing losses under different normalizers cannot establish a better model. Model refinement and precision measurements are ongoing.

`--max-seconds` limits learning time from warmup onward and checks between completed updates. Initialization, an in-progress update and checkpoint writing can extend wall time; allow a margin before an independent overall deadline. `--diagnostic-interval` exports policy/world snapshots without replay or optimizer resume state.
