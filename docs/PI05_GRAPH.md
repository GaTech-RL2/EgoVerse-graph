# PI0.5 campaigns on the graph runtime

The port of `aidan/abc-stationery-pi@d5f72068` uses `PI05Stage` inside
`PipelineAlgo`. It retains the source PI adapter, action converters, camera
calibration, normalization checks, metrics and campaign filters in
`egomimic/campaigns/pi05/`. This scope keeps the source's continuous-6D
conventions separate from the current graph ARC transform API.

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

The PI evaluator consumes the source backend's native outputs so it preserves
the original metric definitions and performs frame reversion once. The source
train visualization loader is a graph validation group named `train_viz`;
its metrics remain separately named, and multi-sample metrics stay off there.
Only rank zero writes videos. The stationery `rl2yam` calibration intentionally
does not geometrically match ABC images, as documented in the source recipe.

## Validation

The 54 source numerical tests pass for rotations, action packing, wrist/frame
round trips, bounds, norm cache provenance and metrics. Graph integration tests
exercise the actual PI adapter with a tiny replacement OpenPI network and
tokenizer: parameter registration, backward, normalized inference, strict
checkpoint loading and Lightning evaluator binding. All ported config groups
compose without importing OpenPI. Python and shell source tools pass syntax
checks. Full pretrained-model training, distributed CUDA execution, real
dataset access and robot actuation remain cluster validation work.
