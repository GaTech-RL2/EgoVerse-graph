# Supported visual ABC / human BC experiments

Launch from this experiment group, not from retired model/data templates:

| Group | Dataset | Paired recipes |
| --- | --- | --- |
| `robot_bc` | ABC stationery slice + RL2 updated organize-stationery; RL2-only validation | `stationery_rl2_hpt300_visual_{baseline,hybrid}_openloop` |
| `robot_bc` | ABC fold and stack the towels | `abc_towels_hpt180_{baseline,hybrid}_visual_openloop` |
| `robot_bc` | ABC four-task multitask | `abc_multitask4_hpt300_{baseline,hybrid}_visual_openloop` |
| `human_bc` | MECKA folding clothes, 40-hour subset | `mecka_fold_clothes_40h_human_visual_{baseline,hybrid}_openloop` |

Brace notation denotes separate YAML files, not Hydra syntax.

`robot_bc/abc_visual_hpt300_base` is a shared template requiring a task
predicate. Cotraining has four explicit
`organize_clock_{velocity,duration}_{full_separation,translation_rotation}`
leaves. See [the clock contract](../../../../docs/ARC_CLOCK_CONTRACT.md).

Select `abc.arc_clock_mode=velocity|duration` and
`abc.arc_streams=full_separation|translation_rotation`, with token dimension
20 or 18 respectively. All tokens have M rows. Row zero stores start delay;
later clocks describe incoming intervals. Duration means log seconds, not
cumulative time. The clock contract describes gripper and hold semantics.

Source-window size is embodiment/recipe-owned. Execution keeps a waypoint
prefix (normally 30 of M100), then replans when the first moving stream ends.
Baselines execute 30 control frames. Legacy distance-prefix ARC and E1
fixed-tick execution are no longer selectable.

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

HPT visual model declarations select `FirstStreamArcDecoder` for Cartesian ARC.
The native M×18/M×20 output is decoded to Cartesian control frames, and the
adapter reports the actual execution boundary. This does not authorize robot
operation. Historical checkpoints require their recorded source and normalizer;
they must not be silently migrated to this representation.
