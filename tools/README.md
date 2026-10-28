# Author baseline reproduction

Run these commands from the repository root on a CUDA Linux machine:

```sh
uv sync --python 3.12 --locked
uv run --no-sync python tools/author_baseline.py smoke --out /absolute/evidence/path
uv run --no-sync python tools/author_baseline.py eval --out /absolute/evidence/path --checkpoint ckpts/test.pt --label author_pretrained
uv run --no-sync python tools/author_baseline.py train --out /absolute/evidence/path
```

The baseline uses the released task, AMP data, seed 42, 4096 environments, and 50,000 updates, starting from scratch. The harness changes only logging/output destinations and adds finite-value checks, an update ledger, and evaluation checkpoints after 50 and each 5,000 updates. It uses local TensorBoard logging. Evaluation runs in a separate process, and training waits for it without recreating the training environment or reloading the model. Evaluation time is additional to the update-speed estimate. No extra PPO updates are authorized or started after the cap. A failed run is not resumed automatically.

`model_4999.pt` from the harness contains the policy after 5,000 updates; the original save-every-500 schedule is also preserved. The original checkpoint format does not save every AMP/environment/RNG state, so these files are not claimed to support exact training continuation. All 50,000 updates run in one continuous process.

The original repository does not include a full dependency lock and its dependency indexes select moving development wheels. The upstream source also imports SciPy without declaring it; this branch adds that dependency. It provides `husky-train` / `husky-play` to avoid entry-point name collisions with mjlab. The harness imports mjlab tasks first so plugin discovery cannot leave the custom registry empty after a circular import.

This branch fixes MuJoCo 3.4.0, Warp 1.11.0, PyTorch 2.10.0 and TorchVision 0.25.0, preserving the released mjlab and mujoco-warp source pins. These runtime choices are compatibility adaptations, not verified author environment versions. The generated `uv.lock` records the remaining resolved packages.

Evaluation uses the author's play configuration, one environment and seed 42, fixing speed to 0.8 m/s and relative heading to 0.5 rad. It retains the author's phase transitions. Play mode removes fall terminations, so root height and phase occupancy are diagnostics, not a success rate. The video shows the actual simulated rollout, not reference motion. Verify camera orientation visually before labeling a view as front-facing.

Outputs use `checkpoints/`, `configs/`, `data/`, `log/`, `image/`, and `vedio/`. No credentials are stored. This workstation is not a GPUFree rental instance and the harness never shuts down the host.
