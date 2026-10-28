# SuperMimic training backend on RTX A5000

This branch adds an optional `--backend supermimic` to the original skating
task. `--backend ppo` remains the default. The robot, skateboard, task rewards,
termination rules, actor inputs and actor/critic network sizes are unchanged.

The learned-model backend is experimental. Fixed-budget A5000 screens have
not demonstrated faster learning than PPO: at about 1200 training seconds,
PPO averaged 3.865 seconds per completed episode, versus 1.634 for the original
uncut SuperMimic profile and 1.254 for its critic-cut variant. All had 0/64
failure-free 20-second evaluations. These single-seed screens establish neither
usable skating nor convergence speed; update throughput alone is insufficient.

The implementation follows the learned-dynamics, short differentiable rollout
and actor-gradient idea in [Breezewrf/supermimic_mjlab](https://github.com/Breezewrf/supermimic_mjlab)
(reference revision `77183062eb92026bfbee2daf276f87cad36fd957`). It is a new skating
adapter rather than a strict reproduction of that motion-tracking task. The
LAFAN policy and world-model checkpoints cannot be used for skating.

## What changes in training

1. Collect real MuJoCo transitions using the original actor mean plus fixed
   Gaussian exploration with standard deviation 0.1, matching the reference
   SuperMimic collector. The original PPO backend retains its learned noise.
2. Fit a normalized delta world model on short replay sequences. Its reward head
   predicts the original task reward, and its termination head predicts physical
   failure. Time limits are kept separate from physical failure.
3. Starting from real replay states, unroll a differentiable model for 32 steps.
   Use deterministic actor means during imagination. Update the actor by
   backpropagating the imagined discounted return through
   the frozen model and frozen value/discriminator networks. Fit the critic to
   detached lambda-return targets. No entropy loss or learned exploration std
   is used in this default SuperMimic mode; the original std checkpoint key is
   retained for playback compatibility and is frozen during this training.
4. Train the original AMP discriminator only with real histories and its
   original skating expert data. Compute the AMP reward differentiably from
   predicted joint histories; a stale combined AMP reward is not a model label.

The actor still receives its original 395-dimensional term-major history and
outputs 23 joint actions. Only the training world model sees privileged robot,
board, contact, command, transition-memory and randomized-physics context.
History shifts, previous actions and the phase clock are projected explicitly
at every imagined step. New actor observations are derived from predicted
physical states, including the actual compiled IMU site transform. Commands
follow the original timer/phase resampling rules using independent external
random samples. Sensor noise is also independent of the actor; model fitting
conditions on observed external command/noise samples without replacing the
predicted physical state. Model and policy normalizers only use real data.

The privileged critic's Bezier transition targets and the scalar task reward /
termination remain learned predictions. Their accuracy still needs to be
checked in real rollouts; constraining actor observations does not eliminate
all possible model exploitation.

By default, `detach_critic_transition_targets=True` stops the actor's terminal
value gradient through the independently predicted Bezier target inputs. The
target slices are resolved from the task's observation metadata (98 coordinates
for this task). Their forward values, critic fitting and model supervision are
unchanged; other value inputs retain their gradients. Outside transitions,
these 98 coordinates are the current poses of 14 body links relative to the
board, so this cut also removes a legitimate geometry gradient route. Their
independent prediction creates a possible error source, not proof of a harmful
shortcut. This is an experimental gradient ablation, not improved-skating evidence.
Set `--supermimic.no-detach-critic-transition-targets` to reproduce the earlier
objective. Checkpoints without the flag are interpreted as `False`; a full
optimizer resume across this change is rejected. Explicit weights-only loading
can initialize a changed objective, but is not a same-algorithm continuation.

The original environment returns an already-reset next observation on terminal
steps. Those next states are excluded from dynamics regression and delta
statistics, and replay sequences stop at both failure and timeout boundaries.
Terminal task rewards and true failure labels are retained. Unknown pre-reset
terminal joints are not fabricated for model training.

This task uses contacts, discrete phase changes, resets and random pushes. The
learned model is a differentiable surrogate, not a differentiable MuJoCo solver.
An actor can exploit model errors; imagined returns alone are not evidence that
real skating improved. Short iteration speed does not establish time to equal
policy quality. Compare real-simulator evaluations at fixed wall-time budgets.

The earlier stochastic-actor experiment is reproducible with
`--supermimic.actor-mode reparameterized`. Checkpoints without an `actor_mode`
field are recognized as that legacy mode. A full optimizer resume into a
different actor mode is rejected rather than silently mixing algorithms.

## Install and run

Use a separate Python 3.12 environment. The original custom `rsl_rl` and pinned
`mjlab` revision are required; the newer environment used for LAFAN SuperMimic
is incompatible. The top-level project pins MuJoCo-Warp to the API used by
`mjlab` revision `13212ad`.

```bash
uv sync --python /usr/bin/python3.12

# Original PPO (unchanged default).
uv run python -m mjlab_husky.scripts.train Mjlab-Skater-Flat-Unitree-G1 \
  --env.scene.num-envs 256 --agent.max-iterations 12 \
  --agent.logger tensorboard --agent.amp-num-preload-transitions 4096

# Bounded integration check. Warm-up fits the model before actor updates.
uv run python -m mjlab_husky.scripts.train Mjlab-Skater-Flat-Unitree-G1 --backend supermimic \
  --env.scene.num-envs 256 --agent.max-iterations 24 \
  --agent.logger tensorboard --agent.amp-num-preload-transitions 4096
```

The expert preload reduction above is for the short checks only, applied to
both backends. It is not the default full training configuration. Inspect
`husky-train ... --help` for individual `--supermimic.*` settings. The explicit
Python module commands avoid the upstream mjlab package's `train`/`play` entry
point name collision; `husky-train` and `husky-play` aliases are also provided. Single GPU is
supported. Start from a fresh run; resuming PPO optimizer state into this
backend is invalid.

On CUDA, `compile_projection=True` compiles the observation/history projection
to fuse small kernels. The first no-grad, model-gradient and actor-gradient
variants compile separately. Include that startup in wall-budget comparisons.
A C/C++ toolchain is required by PyTorch Inductor. Eager execution can be
selected with `--supermimic.compile-projection False` in the training CLI.
Compiled and eager floating-point results need not be bitwise identical.

Checkpoints retain the original `model_state_dict` so existing `play`/export
can load the actor without optimizer state. Training checkpoints also contain
the model, independent optimizers, AMP statistics, replay schema and sampler
state. A resumed process starts a fresh physical simulator episode; it does not
restore an exact in-progress MuJoCo trajectory.

The paired benchmark's PPO checkpoints also preserve the adaptive learning-rate
attribute alongside Adam's parameter-group rates. Older benchmark checkpoints
derive that attribute from their consistent saved group rates. Invalid or
inconsistent rates are rejected before restoring policy weights. This avoids
silently resetting the continued PPO run's adaptive rate to its constructor
default. Fresh PPO updates are unchanged.

## Reproducible short timing check

Run each case in its own process on an idle GPU, with identical environment
count, seed, actor architecture and expert preload. Discard compilation/warm-up
separately from the learned-model warm-up. Include the latter in end-to-end
time-to-quality experiments.

```bash
uv run python -m mjlab_husky.scripts.benchmark_backends \
  --backend ppo --num-envs 256 --iterations 12 --discard-iterations 3
uv run python -m mjlab_husky.scripts.benchmark_backends \
  --backend supermimic --num-envs 256 --iterations 32 --discard-iterations 23
```

`logs/backend_benchmark/<backend>/benchmark.json` records the exact configuration,
per-iteration measurements, GPU name and memory. The SuperMimic run also writes
`metrics.jsonl`. Compare real-step budgets and model/actor/critic/AMP update
counts alongside seconds/update: one model-based update and one PPO update do
different amounts of work. Report any configuration reductions explicitly.

CPU invariant checks do not load MuJoCo:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_supermimic_*.py' -v
```

The live adapter identity check uses true next physical states and tests the
observation reconstruction; it does not evaluate world-model prediction:

```bash
uv run python -m mjlab_husky.scripts.check_world_model_adapter \
  --num-envs 32 --steps 8 --output logs/adapter_identity.json
```

For a bounded paired experiment, run each backend for the same inclusive
training time from the same saved random policy/discriminator initialization:

```bash
uv run python -m mjlab_husky.scripts.paired_trial \
  --output-dir logs/paired_trial_001 --seconds-per-backend 600
```

The supervisor runs GPU jobs serially and writes `status.json`, complete
configuration/checkpoint hashes, and final original-task evaluations. An
evaluation uses deterministic policy means but retains the original observation
noise, pushes, and failure conditions. Presentation mode disables some of those
conditions and is labelled separately; do not use it as training success proof.
Fixed-start survival, task reward and speed/heading errors must be considered
together, with the same evaluation protocol and seed. A single short paired
run screens the approach; it is not a multi-seed convergence result.

If only the SuperMimic implementation changes, an already completed PPO
training run can be reused with `--reuse-ppo-benchmark <ppo/benchmark.json>`
and `--initial-state-file <original-shared-initial-state.pt>`. The supervisor
checks the configuration, original report/checkpoint hashes and common weight
identity, skips duplicate PPO training, and evaluates both policies again.
Historical PPO training time and newly consumed training time are reported
separately. This check does not by itself prove unchanged task/PPO source code;
verify that separately before interpreting the comparison.

A frozen world-model probe uses new real data, recorded actions and externally
observed command/noise samples, without fitting or updating statistics:

```bash
uv run python scripts/check_world_model_rollout.py \
  --checkpoint-file logs/<run>/model_<iteration>.pt --num-envs 64 --steps 32 \
  --output logs/world_model_probe.json
```

It reports H1/H8/H32 errors and usable window counts. If no real trajectory
survives a horizon, its physical accuracy is reported as unavailable, not zero.
The probe's `passed` flag checks frozen state and finite predictions; it is not
an accuracy or policy-success gate.

## Optional exact action-cost component

`--supermimic.analytic-action-cost` in the benchmark/paired CLI enables an
experimental, parameter-free decomposition of the two original action penalties.
It defaults to off. For pre-step state `s` and the executed raw action `a`,

```text
A(s,a) = scale * [w_rate * sum((a - a_previous)^2)
                 + w_acc * sum((a - 2*a_previous + a_previous_previous)^2)]
```

Weights and time scaling come from the live original regularization manager.
The packed pre-step `action.action` is `a_previous`; `action.prev_action` is
`a_previous_previous`. This uses sums over action coordinates and no division by
the timestep. The model normalizes/learns `R - A`, while `Prediction.reward`
always returns `A + learned_residual`, in original total task-reward units.
Replay, AMP inputs and real reward metrics retain `R`. Failure and timeout reward
labels all participate; this component does not require a post-reset next state
or a survival multiplier. Other task rewards and termination predictions remain
learned. This is an integration option, not a measured learning-speed improvement.

The checkpoint records the reward representation, weights, scaling and history
schema. Loading a world model across this flag is rejected even with
`load_optimizer=False`, because total-reward and residual heads have different
meanings. Use a fresh run for the ablation. A policy-only initialization must
explicitly omit the old world model and its normalizer.

The paired supervisor forwards and verifies the full requested SuperMimic
configuration. For example, a separate model-fitting-budget experiment can use
`--supermimic.model-updates 16`. Keep one factor changed at a time and include
its extra model fitting, imagination and checkpoint costs in the wall budget.

## Optional transition-memory projection and WM precision

`--supermimic.project-transition-memory` keeps the four Bezier start-pose
buffers unchanged between their original phase-entry events. New entry values
remain learned predictions; no forward kinematics or simulator state is
teacher-forced. The entry mask uses the projected new contact phase and actual
previous phase, including the original standing/reset behavior and float32
boundary arithmetic. It does not change the original real task, the learned
98 critic targets, or raw simulator qpos/qvel. The option defaults to off.

`--supermimic.model-precision bfloat16` enables autocast only for the world-model
MLP and its three heads. Master weights, Adam state, decoded state/reward/logit
outputs and normalization arithmetic stay FP32; the original normalization
accumulators remain float64. Actor, critic, AMP discriminator and simulation
precision are unchanged. CUDA BF16 support is checked before use. This allows a
separate larger-model profile, for example
`--supermimic.model-hidden-dims 2048 2048 2048`; it does not establish that BF16
is faster for every width/batch. The default remains `float32`.

Both options passed bounded A5000 integration with a 2048x3 world model and
H32, including actual phase-entry/hold targets, compiled/eager value and input
gradient parity, finite updates, precision boundaries and native checkpoint
restoration. That small integration check is not a training-quality benchmark.
The default off/FP32 path was checked against the earlier implementation with
identical seeded tensors and optimizer/RNG states.

Missing checkpoint flags are interpreted as off/FP32. A full optimizer resume
across precision modes is rejected; explicit weights-only loading can reuse the
same FP32 master-parameter format. A world model cannot be loaded across
transition-memory projection modes, including weights-only loading. Use fresh
runs when comparing these profiles and include compilation/startup in the budget.

## Critic objective alignment

`--supermimic.critic-loss-profile reference_time_sum` matches the reference's
critic regression objective: sum squared errors over all imagined time steps,
then average over trajectories. Squaring and reduction use FP64 intermediates;
the final scalar returns to the prediction dtype. The default
`legacy_survival_mean` retains the earlier survival-weighted mean exactly.
Loss magnitudes across the two profiles are not directly comparable.

This switch changes the critic objective only. Actor survival gradients,
lambda-return targets, reward/termination, Adam parameters, clipping, update
counts and learning-rate scheduling are unchanged. In particular, it does not
claim complete alignment with all reference optimizer/scheduling choices.
Checkpoints without the setting use the legacy objective. Full optimizer resume
across objectives is rejected before state mutation; explicit weights-only
initialization is allowed. Use fresh equal-budget runs for the comparison.

## Optional reference actor optimizer

`--supermimic.actor-adam-beta2 0.95 --supermimic.actor-element-clip 0.1`
selects the reference actor Adam beta2 and elementwise gradient bound. The
existing global norm clip runs first, followed by the elementwise clamp. Adam
beta1 remains 0.9, epsilon 1e-8 and weight decay zero. The default beta2 is
0.999 with no elementwise clamp, preserving the earlier update. Critic and
world-model optimizers, task rewards, termination and policy inputs are unchanged.

Missing checkpoint settings use the earlier defaults. Full optimizer resumes
require matching configuration and saved actor optimizer hyperparameters before
any state mutation; explicit weights-only initialization retains the new run's
optimizer settings. CPU tests cover binding clamp/order and Adam moments; a
bounded A5000 BF16/H32 update and native restoration passed with the reference
settings. These checks establish integration, not faster learning or complete
alignment with the reference training schedule.

## Independent PPO rollout profiles

The benchmark accepts `--ppo-steps-per-env 6 --num-envs 4096` for PPO. Omitting
the override retains the registered task default (24 steps). The original
1024x24 and candidate 4096x6 profiles both collect 24576 transitions per update;
with five epochs and four minibatches, the optimizer batch sizes are unchanged.
Fewer serial collection calls may improve throughput, but shorter GAE
trajectories change the bootstrap bias. This is an independent PPO experiment
and must not be reported as a SuperMimic gain.

Reports and checkpoints record the actual environment count, rollout length,
transition counts and PPO update configuration. Full optimizer resumes across
different environment/rollout geometry are rejected even when the total batch
size matches. Legacy checkpoints require the task-default rollout and matching
counts. Paired trials forward this option only to PPO and reject reused reports
whose effective rollout differs. Compare fresh profiles from a shared initial
state and include a matched quality evaluation before claiming a speed gain.

The source task and bundled RSL-RL retain their existing licenses. Reference
links describe algorithm provenance; the new backend does not bundle LAFAN
assets or pretrained checkpoints from the reference project.
