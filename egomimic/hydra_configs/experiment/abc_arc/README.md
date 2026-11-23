# ABC-lab visual experiments

The rl2 YAM task leaves live in `arc_real/` (see [the experiment menu](../README.md)).
This folder keeps the ABC-lab visual experiments on the `abc_visual` pool:
`robot_bc/abc_towels_hpt180_{baseline,hybrid}_visual_openloop`, the ABC
four-task `robot_bc/multitask/{baseline,arc_vel,arc_dur}` and the ABC-DiT
`robot_bc/multitask/variants/abc_multitask4_abcdit_{baseline,hybrid}_visual_openloop`,
plus the independent MECKA human-only folding experiments
`human_bc/mecka_fold_clothes_40h_human_visual_{baseline,hybrid}_openloop`.
Brace notation denotes two YAML files, not Hydra syntax.

ABC seven-task pretraining lives in `robot_bc/multitask_pretrain/{hpt,rdt}/{baseline,arc_dur}`
on `data/abc_arc/multitask_pretrain`: the `abc/arm_local_fk_v1` conversion with its
published `is_eval` split (14,494 train / 1,166 validation episodes), sampled on one
30 Hz clock. These runs use D = 0.519 m, the training-only median of larger-arm command
translation travel over complete 100-row windows.
`rdt/arc_dur_m200` keeps D and measures R the same way (1.787 rad, larger-arm SO(3)
rotation travel over the same windows) and uses M = 200 waypoints. The other arc_dur runs keep
the 0.419 rad default, which ends most executed prefixes on the rotation stream after ~9 frames.

`robot_bc/multitask_finetune/rdt/{baseline,arc_dur,arc_dur_m200}` fine-tune those three RDT
runs from their 240k weights (weights-only, sha256-pinned) on `data/abc_arc/rl2_mixed_e526_t476_s376`:
Aidan's 2026-11-22 rl2 pool of 526 earbuds, 476 towels and 376 stationery takes, with 56 frozen
validation takes. rl2 `processed_v3/yam` poses are `tcp_site` FK with both arms in `left_base`;
`scripts/data/rl2_tcp_to_arm_local.py` adds `<arm>.{obs,cmd}_ee_pose_arm_local` arrays to
the local episode copies in the ABC convention (per-arm base, grasp site), so the data config reuses
the pretraining loader unchanged. Token format, optimizer and batch follow the pretraining run.
Normalization is selected at launch: the pretraining data context or one recomputed on this pool.

See [the clock contract](../../../../docs/ARC_CLOCK_CONTRACT.md).

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
