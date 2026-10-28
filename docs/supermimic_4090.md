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

The default workload uses eager FP32, five real steps every two learner iterations, an eight-step autoregressive model window, ten model batches of 256, and a 32-step imagined actor rollout with 256 starts. Actor LR is `3e-6`; a warm-started actor also uses the teacher constraints below. `joint_consistency=False` remains the default. PPO updates and these iterations are different units of training progress.

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

The KL is `0.5 * sum(((mu - mu_teacher) / std_teacher)^2)`, averaged over the checked inputs. Raw differences are in actor-output units. Backtracking tries at most ten scales toward the previous actor. If new real inputs expose an existing violation, it also tries fourteen scales toward the teacher. An accepted scaled proposal retains the full Adam moment update: this is **projected Adam**, not an unchanged Adam step or a faithful reproduction of upstream SuperMimic optimization. The final fallback resets the actor to its teacher and restores the pre-proposal Adam state.

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

The online learner holds out the last 10% of environment IDs. A fixed, independently sampled eight-step validation batch is checkpointed with its own RNG. In the initial measured run, normalized autoregressive state loss fell from 6.0708 to 0.8193 and normalized reward loss from 0.53768 to 0.07603. These are fixed-batch model errors, not a skateboarding success rate. Normalization is fitted to the warmup distribution and then frozen; distribution drift and learned-reward exploitation remain limitations. The separate fixed32 study below uses a different, explicit 80/20 environment split.

The following are **not established** by these measurements:

- Useful cold-start learning from a fresh policy.
- Model accuracy sufficient for reliable policy improvement through a 32-step imagined rollout. Fixed32 errors are now measured, but lower prediction errors have not established better real behavior.
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

The tests cover gradients through the frozen coupled model/critic/style callback, reset masks, term-major history, model fitting, replay/context restoration, teacher acceptance/backtracking/rollback, independent validation sampling, compatible policy export and learning-state restore. The initial backend passed 28 checks; the conditional-command revision passed 39. Joint-consistency integration also completed CUDA tests, a two-iteration real-environment smoke and full learning-state resume from two to three. A regression test covers inference-first cache warming followed by world-loss and actor-BPTT backward. These checks validate wiring and restoration, not long-run convergence.

## Attribution

Adapted protocol/code provenance and the Apache-2.0 license are in `third_party/supermimic/`. HUSKY task/assets retain their original terms.

## Conditional command protocol (nightly follow-up)

The task samples a new relative heading when push-to-steer ends. The previous model projection incorrectly kept that command fixed. The conditional protocol supplies only recorded external command values during fitting and validation; imagined rollouts sample commands from the original task ranges. Physical state, heading reference and observations are never copied from a future real target. The heading reference is reconstructed from predicted board orientation. Still/phase transitions use the preceding command, matching the task's update order.

Full snapshots now carry `dynamics_protocol=conditional-commands-v2`. Earlier snapshots require explicit migration and are rejected as full resumes. Policy warm starts remain compatible. Resumed runs require a fresh run name to preserve prior ledgers.

The conditional implementation passed **39 tests**, including CUDA replay transport, and real-environment smoke, diagnostic exports and full learning-state resume. Its 50-step diagnostic loop averaged **0.71747 seconds** over the last 30 steps. The earlier 0.62010-second result above describes the previous implementation; the correctness fix adds work.

In a separate 64-environment, 30-second play evaluation over the original command ranges, teacher and conditional-50 policy had identical initial-state and complete command-sequence hashes. Both had five environments with observed root tilt above 70 degrees. Pooled pushing velocity MAE was 0.13532/0.13286 m/s and heading MAE 0.22158/0.22071 rad (teacher/candidate). This preserves behavior in this diagnostic sample; it does not establish a statistically significant improvement or a formal success rate.

`tools/audit_supermimic_model.py` measures 1/8/16/32-step open-loop predictions against the same recorded real actions, with physical-unit groups, phase-boundary groups and frozen normalizers. Changing only command projection improves command/observation consistency, with a small physical-state effect. A separately retrained 50-step model had mixed physical errors; comparing losses under different normalizers cannot establish a better model.

`--max-seconds` limits learning time from warmup onward and checks between completed updates. Initialization, an in-progress update and checkpoint writing can extend wall time; allow a margin before an independent overall deadline. `--diagnostic-interval` exports policy/world snapshots without replay or optimizer resume state.

## Fixed-data world refinement and optional joint consistency

A separate real corpus contains 128 environments × 640 control steps, with the original training commands, observation noise and disturbances. The first 102 environments are used for fitting; the remaining 26 supply a fixed batch of 256 reset-safe, 32-step windows. The complete CPU replay and its audit batch are bound by SHA256. The held-out batch never enters fitting.

`tools/refine_supermimic_world.py` initializes each run from the same trained world weights, freezes the original normalizers, and uses a fresh Adam at the original model LR. It updates only the world model. Each run is capped at 1,000 optimizer steps or 120 seconds after initialization, including periodic audits. Final audit/export can extend that soft time limit. The eight-step run fits 2,048,000 nominal transitions; the sixteen-step run fits 4,096,000. They therefore have equal optimizer counts, but different fitting work and approximately 39.72/78.21 seconds of optimizer workload.

The following values are raw RMSE averaged over all valid transitions of the fixed32 rollout, with recorded real actions. Norms are identical across the two fitting runs.

| Fixed32 prediction | Initial world | After 1,000 × window8 | After 1,000 × window16 |
| --- | ---: | ---: | ---: |
| Robot root position, m | 0.04185 | 0.02631 | 0.02444 |
| Board root position, m | 0.03815 | 0.02262 | 0.02077 |
| Robot joint velocity, rad/s | 0.93441 | 0.82464 | 0.77580 |
| Policy observations, scaled units | 0.05235 | 0.04402 | 0.04061 |
| Task reward per control step | 0.03817 | 0.03470 | 0.03295 |

The optional P1 projection removes independent predictions for AMP joints and the critic's noise-free joint observations. It selects robot joints from **predicted** qpos/qvel, subtracts the original default joint pose/velocity and applies the critic scales. AMP's newest history frame then uses those same joints. Actor joint observations and their noisy history remain unchanged. No future real physical state is supplied; only the existing four external command fields are replayed during fitting.

Joint addresses, names, defaults, slices and scales are constant metadata, with a separate provenance record and a semantic spec SHA. State/context widths remain 1,443/1,645. `tools/derive_supermimic_joint_metadata.py` establishes the unique mapping on training environments, then verifies it on held-out environments. For this corpus, AMP and critic formulas reproduce all 163,832 stored/valid-next rows with maximum error zero. The spec SHA is `1e4568b8931bf71972256b047aa401d2bdf1d33fc0b72f6f662d4296b62b39c7`. These constants do not represent a learned or future physical trajectory.

A separate P1 window16 run completed 1,000 world steps with unchanged norms. Its fixed32 robot/board position RMSE changed from 0.04188/0.03826 m to 0.02315/0.01990 m, and joint velocity from 0.93069 to 0.74510 rad/s. Derived-field error reductions are partly structural; raw qpos/qvel errors and real policy behavior remain the relevant checks. P1 is experimental and stays **off by default**.

```sh
uv run --no-sync python tools/derive_supermimic_joint_metadata.py \
  --replay-corpus /absolute/path/to/corpus_replay.pt \
  --out /absolute/path/to/experiment --label robot_joint_mapping

uv run --no-sync python tools/refine_supermimic_world.py \
  --model-diagnostic /absolute/path/to/trained_model_diagnostic.pt \
  --replay-corpus /absolute/path/to/corpus_replay.pt \
  --audit-batch /absolute/path/to/fixed32_audit_batch.pt \
  --joint-metadata /absolute/path/to/robot_joint_mapping_joint_metadata.json \
  --out /absolute/path/to/experiment --label p1_refine16 \
  --model-window 16 --steps 1000 --max-seconds 120
```

Audit/refinement tools enable the joint projection only with explicit `--joint-metadata`: a JSON path, or the flag without a value to restore metadata from the supplied snapshot. A snapshot carrying metadata does not silently enable it. New online runs opt in with `--joint-consistency`; a full resume keeps its saved configuration. `--world-start` imports world weights/frozen norms and starts other learning state fresh; importing a P1 world requires the explicit joint-consistency flag. World-only refinement artifacts are diagnostics, not full resumable learner snapshots.

## Precision and compilation experiments

The default remains eager FP32 with the pinned mjlab TF32 permissions. These experiments used the same trained weights, real-action batch and fixed command sequence, with no optimizer steps. Numerical agreement on this batch is not proof of equivalent learned policies.

| Isolated workload, median | Eager FP32 | Alternative | Result |
| --- | ---: | ---: | --- |
| World8 forward + backward | 32.778 ms | BF16 world MLP: 34.825 ms | 6.25% slower |
| World8 forward + backward | 32.492 ms | Compiled world forward: 33.801 ms | 4.03% slower |
| Imagination32 forward + backward | 127.335 ms | Compiled world forward: 130.558 ms | 2.53% slower |

BF16's actor-gradient cosine was 0.999399 and relative L2 difference 0.038801; provisional same-weight numerical gates passed. The compiled-forward gradient cosine was 0.999999888 and relative L2 difference 0.000522. Neither alternative improved measured latency, so neither is adopted as the default.

PyTorch 2.10 Inductor initially conflicted with mjlab's new precision-setting API. A fresh-process TF32 compatibility environment was tested separately: original-environment and compatibility-environment eager outputs and actor-gradient tensor-value hashes were identical for the recorded inputs. This is a bounded arithmetic check, not a general compatibility guarantee. Only the world forward was compiled in the successful timing experiment; Python autoregressive loops, projection and loss remained eager.

Compiling the complete eight-step loss failed in Inductor/Triton (`PassManager::run failed`) before a completed compiled forward/backward result. A retry with fusion size 16 and epilogue fusion disabled failed as well. There is no valid full-loss compilation speedup or gradient-equivalence result to report. Compiler failure evidence is retained; production training remains eager.

## Paired 500-iteration policy diagnostic

The P1 learner completed 500 iterations, all with finite logged scalars and accepted actor steps. It started from the window16-refined world, then used the default online window8 fitting profile and tight teacher limits. It used 6,430,720 real simulator transitions including warmup. Reported time after initialization was 405.11 seconds, including 16.91 seconds of warmup. The last 30 iterations averaged 0.76686 seconds. Initialization, separate corpus collection and world-only refinement are outside that reported time.

The updated evaluator sums the unchanged environment task reward, excluding the adversarial style reward added by the training runner. In this paired test, all three policies have identical initial-condition and complete command-sequence hashes: 64 environments × 30 seconds, seed 42, deterministic means, random speed/heading commands, no observation corruption or interval pushes, and fall termination disabled.

| Paired diagnostic | Original 10K teacher | P1 + refined world, 50 iterations | P1 + refined world, 500 iterations |
| --- | ---: | ---: | ---: |
| Environments ever above 70° root tilt | 6/64 | 5/64 | 6/64 |
| Mean task return, excluding AMP style | 304.3453 | 305.4450 | 304.6647 |
| Pushing speed MAE, m/s | 0.133605 | 0.138106 | 0.136403 |
| Steering heading MAE, rad | 0.221456 | 0.225937 | 0.222293 |
| Steering both-feet board-contact fraction | 0.996381 | 0.998677 | 0.996106 |

The 500-iteration return difference is only +0.105%; fall count is unchanged and tracking errors are slightly worse. This test shows no meaningful real-policy improvement. The 50-iteration sample's one fewer tilted environment is not attributable evidence with one seed. These **64 × 30-second play diagnostics are not a task success rate** or a multi-seed learning result. Earlier 64-environment reports above used a previous evaluator; compare paired results within each recorded evaluator/protocol.

Strict teacher constraints remain in place. An earlier exploratory widening to KL/raw/std limits 0.01/0.1/0.3 increased observed falls from 5/64 to 7/64 and worsened pushing speed MAE from 0.13457 to 0.14688 m/s in that earlier paired protocol. Together with the original unguarded failure, this supports retaining the tighter defaults for this adaptation; it is not a global stability guarantee. Improved world-model errors and lower iteration latency have **not established faster convergence to equal task quality**.
