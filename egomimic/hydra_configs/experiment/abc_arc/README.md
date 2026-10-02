# Supported visual ABC / human BC experiments

Launch from this experiment group, not from retired model/data templates:

| Group | Dataset | Paired recipes |
| --- | --- | --- |
| `robot_bc` | ABC stationery slice + RL2 updated organize-stationery; RL2-only validation | `stationery_rl2_hpt300_visual_{baseline,hybrid}_openloop` |
| `robot_bc` | ABC fold and stack the towels | `abc_towels_hpt180_{baseline,hybrid}_visual_openloop` |
| `robot_bc` | ABC four-task multitask | `abc_multitask4_hpt300_{baseline,hybrid}_visual_openloop` |
| `human_bc` | MECKA folding clothes, 40-hour subset | `mecka_fold_clothes_40h_human_visual_{baseline,hybrid}_openloop` |

Brace notation in this table denotes two separate YAML files, not Hydra syntax.
`robot_bc/abc_visual_hpt300_base` is an abstract shared template with a required
task predicate; `cotrain` currently has no supported launch recipe.

All retained recipes are visual/proprioceptive HPT, without a text encoder or
language conditioning. They share executed-prefix open-loop evaluation, full
frame validation overlays, and 10% normalization sampling. Every ARC recipe has
hybrid rotation, a separate R target/clock, and per-waypoint velocity.

Select translation semantics with `abc.arc_chunking_mode=joint_distance`,
`race`, or `multistream`. The default is `joint_distance`. ARC reads up to 600
native source frames, then caps/resamples to M=100 waypoints; per-waypoint rate
rows make the model token horizon 200. This is distinct from both the raw source
buffer and the baseline's 100 control frames. Execution defaults to the first
30 waypoints, or the first 30 control frames for the baseline. See
`docs/ARC_CHUNKING_MODES.md` for short-arm holds and normalization compatibility.

## Model names and checkpoint compatibility

`hpt180` and `hpt300` are historical family names kept for command and checkpoint
compatibility. They are not exact total-parameter counts. No weights, layer
dimensions, or checkpoint tensor shapes were resized by this cleanup.

| Historical family | Trunk width / blocks / heads | Flow blocks / width |
| --- | --- | --- |
| HPT180 | 640 / 19 / 8 | 10 / 384 |
| HPT300 | 840 / 19 / 10 | 6 / 320 |

The robot variants have roughly 232M and 249M total parameters, respectively.
Human and robot stem/domain choices can change totals. To print exact composed
robot counts without starting training, activate the project environment and run:

```bash
PYTHONPATH=. python -m pytest -s -q tests/test_abc_visual_hybrid_configs.py -k model_instantiates
```

Precomputed caches must match the dataset, split, embodiment, sampling fraction,
and ARC action contract. Never relabel a 20% or different-mode cache as 10% or
as another mode. Changing these settings creates a new training signature and
requires a new W&B identity, not a resume of a different experiment.

The stationery template's training split currently includes both a deterministic
ABC slice and RL2; its validation split is RL2-only. This cleanup preserves that
data contract. A strictly RL2-only training campaign needs an explicit filter
override and matching normalization cache; the experiment name alone does not
establish that restriction.

## Robot inference export

The newer `main` exports model-owned robot inference sidecars for Cartesian and
E1 codecs. The three ABC hybrid ARC modes require their own mode-aware robot
adapter, so automatic robot export explicitly reports `unsupported` for them.
It must never label the 200 waypoint/rate rows as 200 Cartesian control frames.
This guard does not disable ARC training, detokenization, validation metrics,
or validation videos.
