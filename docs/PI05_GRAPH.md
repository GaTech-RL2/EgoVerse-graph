# PI0.5 campaigns on the graph runtime

The port of `aidan/abc-stationery-pi@d5f72068` uses `PI05Stage` inside
`PipelineAlgo`. The model adapter lives in `egomimic/models/pi05/`. Data loading and normalization
use `rldb/zarr/zarr_dataset_multi.py`; pose transforms use the shared embodiment
and transform modules. Calibration matrices and dataset corrections are in YAML.
The old `campaigns/pi05` paths forward imports to these shared components;
new code and configurations should use their canonical homes. Historical YAML
using `mode`, `cartesian_pi`, or `fix_mecka_left_wrist` must migrate to the explicit
`action_mode`/`coord_frame`/`rotation_mode`, `camera_keys`, and
`local_frame_rotations` settings shown in the current recipes.

## Environment and entry point

Use a Linux GPU environment with the source OpenPI fork at
[`981483dca0fd9acba698fea00aa6e52d56a66c58`](https://github.com/GaTech-RL2/openpi/tree/981483dca0fd9acba698fea00aa6e52d56a66c58).
The `pi05` extra pins that dependency. Its upstream environment includes CUDA
JAX and additional workspace dependencies; follow that fork's environment
setup when preparing a fresh cluster environment. Normal graph configurations
do not import OpenPI or download model/tokenizer weights.

Activate the project environment before running Python. Request a GPU before
training. From the graph checkout, the stationery recipe is:

```bash
python egomimic/trainHydra.py --config-name=train_zarr_cartesian_pi \
  pi05.pretrained_weights=/path/to/pi05_base_pytorch \
  norm_stats.precomputed_norm_path=/path/to/abc_stationery_rl2yam/norm_stats
```

This selects the full stationery SQL filter, Eva encoding of the mirror's
metadata, wrist-relative continuous-6D actions, the `rl2yam` camera calibration,
a 100-step / 32D OpenPI action expert, LR `5e-5`, warmup 2,000 steps and 400
epochs of 100 batches. The source launchers are under `scripts/pi05/`; they
accept `WT`, `PY`, `PI05_WEIGHTS`, and their existing norm/output/resume options.
Their lab filesystem defaults and Slurm partitions still need to match the
cluster. Checkpoints are retained. No launcher was submitted during this port.

Data, model and evaluator groups are prefixed with `pi05/`, for example
`data=pi05/mecka_abc_pi_6d`. The older `aria_bimanual` configuration slot maps
to `human_bimanual`; the SQL filters and episode exclusions retain their source
values. Some historical filters intentionally describe older episode-table
snapshots and may require a matching snapshot to reproduce their experiments.

## Graph boundaries and checkpoints

The training entry point binds the stats-only normalizer to the pipeline
before constructing an optimizer or loading a checkpoint. This materializes
the optional PI network and registers every parameter on the graph. Call
`algo.bind_data_context(normalizer=stats)` at the same point in custom loaders.
The PI stage receives normalized data. Its backend produces native decoded
actions, which the stage normalizes once into graph `pred_action`.

Graph checkpoints load strictly after binding. For weights-only initialization
from the original PI runtime or another graph PI run, set
`model.pipeline.stages.0.init_weights_ckpt=/path/to/checkpoint`. This translates
the policy prefix and requires all policy tensors to match; optimizer and epoch
state start fresh. Use `ckpt_path` for a full resume of a graph run. Legacy HPT
and PI checkpoints are not interchangeable with arbitrary graph models.

HPT and PI use `BimanualCartesianEval` over graph `pred_action`, with shared
`EvalVideo` buffering. Pose, DTW, Fréchet and optional multi-sample metrics are
configured on that evaluator; it never inspects PI stages or calls a model backend.
Predictions and targets are unnormalized once. The `train_viz` validation group
uses the standard `Valid_train_viz/` metric prefix and its own video directory;
its multi-sample metrics stay off. Other group names work the same way.
Only rank zero writes videos; playback FPS accounts for distributed sampling.
The stationery `rl2yam` calibration intentionally does not geometrically match
ABC images, as documented in the source recipe.

`python scripts/data/precompute_norm_stats.py --data <recipe> --model <recipe>
--out <directory>` is available to any graph model. `trainHydra.py
norm_stats_only=true norm_stats.save_cache_dir=<directory>` uses the same loader.

## Validation

The 54 source numerical tests pass for rotations, action packing, wrist/frame
round trips, bounds, norm cache provenance and metrics. Graph integration tests
exercise the actual PI adapter with a tiny replacement OpenPI network and
tokenizer: parameter registration, backward, normalized inference, strict
checkpoint loading and Lightning evaluator binding. All ported config groups
compose without importing OpenPI. Python and shell source tools pass syntax
checks. Full pretrained-model training, distributed CUDA execution, real
dataset access and robot actuation remain cluster validation work.
