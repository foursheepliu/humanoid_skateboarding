# SuperMimic training backend on RTX A5000

This branch adds an optional `--backend supermimic` to the original skating
task. `--backend ppo` remains the default. The robot, skateboard, task rewards,
termination rules, actor inputs and actor/critic network sizes are unchanged.

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
unchanged; physical-state-derived value inputs retain their gradients. This
is a targeted model-exploitation ablation, not a claim of improved skating.
Set `--supermimic.detach-critic-transition-targets False` to reproduce the earlier
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

The source task and bundled RSL-RL retain their existing licenses. Reference
links describe algorithm provenance; the new backend does not bundle LAFAN
assets or pretrained checkpoints from the reference project.
