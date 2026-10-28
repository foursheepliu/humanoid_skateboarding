This backend adapts the online world-model training, frozen-model BPTT,
soft-survival objective and detached lambda-return critic design from
Breezewrf/supermimic_mjlab, commit 77183062eb92026bfbee2daf276f87cad36fd957.
Source: https://github.com/Breezewrf/supermimic_mjlab
The upstream reproduction is Apache-2.0 (LICENSE in this directory).
The HUSKY task and original assets retain their original license terms.

Modifications: robot+skateboard augmented state, episode-safe HUSKY replay,
learned HUSKY task reward/termination heads, retained HUSKY adversarial motion
prior, compatible ActorCritic policy export and 24-GB-oriented workload.
This is a task adaptation; upstream humanoid-only tracking checkpoints and
reported convergence numbers do not apply to skateboarding.
