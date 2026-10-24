# Task-first experiments

The Cartesian YAM/ABC experiment menu is:

```text
experiment/
  real/                   the nine rl2 YAM task leaves (Aidan's frozen pools)
    stationary_{time,duration,velocity}.yaml
    earbuds_{time,duration,velocity}.yaml
    towels_{time,duration,velocity}.yaml
  abc_arc/                ABC-lab visual experiments
    robot_bc/             ABC towels HPT180, ABC multitask4 HPT300 and ABC-DiT
    human_bc/             MECKA human-only folding
  sim/
    libero/
    libero_historical/
    oat/
    pusht/
    synthetic/
  _shared/                shared defaults, not standalone runs
```

Select one with `+experiment=real/stationary_duration` (or `_velocity` or
`_time`). Cluster resources still come from the explicit `hydra/launcher=...`
selection and require a separate launch preflight.

Each leaf is a **complete experiment entry point**: its defaults select the
model graph, data filters/splits/transforms, evaluator, optimizer, trainer,
callbacks and logging. No additional `model=` or `data=` selection is needed.
Shared parents avoid copying those definitions into every leaf. Supply
machine-specific data/weight paths, credentials, a fresh W&B identity
(`logger.wandb.id` defaults to `<name>_s<seed>`) and the reviewed cluster
allocation at launch.

Example (submits a run; execute only after launch preflight):

```bash
python egomimic/trainHydra.py -m \
  +experiment=real/stationary_duration \
  hydra/launcher=submitit_pace_h100
```

## The real task leaves

| Task | Frozen pool (train / val), `data/abc_arc/<task>` | D |
| --- | --- | --- |
| stationary | slow-pace 366 + the 51 organize_stationary episodes Elmo recorded on 2026-10-19/20 (417 / 24) | 0.22 m |
| earbuds | `wireless_earbuds_updated` + Elmo's 49 takes of 2026-10-20 and 20 of 2026-10-21 (339 / 15) | 0.22 m |
| towels | `since0924plus74` + Aidan's 19 fold_towels takes of 2026-10-21 and 37 of 2026-10-24 (499 / 17) | 0.33 m |

The pools point at their SQL tasks (lab `rl2`, `yam_bimanual`): on 2026-10-22 the
episodes not in these pools were moved to `organize_stationary_retired`,
`fold_towels_retired` and `wireless_earbuds_retired` (and the former
`wireless_earbuds_updated` became `wireless_earbuds`), so each task holds exactly
its pool. A pool YAML lists only its frozen validation hashes; the train rule is
the task minus that list, and the count / SHA-256 pins guard the membership. To
add data, add it to the task and update the pins.

All nine share one graph and schedule (`_shared/frozen_yam/hpt300_openloop`:
HPT-300M, embed 840, trunk 19 x 10 heads, flow 6 x 320, 240k steps, cosine
with 10k warmup) and one scorer (the complete-episode open-loop evaluator on
the 25 % prefix; rank on `Valid/open_loop_sim/{XYZ,YPR,Grip}_MSE`). The three
leaves of a task select exactly the same episodes.

- `<task>_time`: the baseline, 100 x 14 time-indexed actions, deployed through
  `TimeChunkRetimer`.
- `<task>_duration`: absolute-duration single-clock ARC
  (`_shared/arc_single_clock`): M 200, one clock per arm
  (`abc.arc_streams=translation`, M x 16; rotation and gripper ride the
  translation supports), absolute waypoint seconds decoded by speed-bounded
  isotonic regression, 100-frame native source window, deployed 50 % waypoint
  prefix with first-moving-stream replanning.
- `<task>_velocity`: the same recipe with interval arcvel clocks
  (`abc.arc_clock_mode=velocity`, each waypoint segment's mean speed).

Use `abc.arc_streams=full_separation` (M x 20) or `translation_rotation`
(M x 18) on an ARC leaf for more clocks; the token width follows automatically,
including the model and decoder. ARC Projection lives on its own branch.
These are fresh-run recipes: checkpoints trained before the first-delay token
(2026-10-21) keep their recorded source and are not loadable here.

ARC uses the canonical [clock contract](../../../docs/ARC_CLOCK_CONTRACT.md)
(repository path `docs/ARC_CLOCK_CONTRACT.md`): velocity rates or **absolute**
seconds from the chunk anchor (decoded by speed-bounded isotonic regression),
first-row start delay, first-moving-stream termination.
No tokenizer or detokenizer algorithm is changed by this YAML reorganization.

## ABC-lab menu

`abc_arc/robot_bc/` keeps the ABC-lab visual experiments on the `abc_visual`
pool (ABC towels HPT180 baseline / hybrid, ABC four-task HPT300
`multitask/{baseline,arc_vel,arc_dur}` and the ABC-DiT `multitask/variants`).
`abc_arc/human_bc/` keeps the MECKA human-only folding experiments. They read
400 native frames with the full-separation clock contract and are not
interchangeable with the real task leaves.

## Retired on 2026-10-22

The per-task folders (`real/{organize_stationary,wireless_earbuds,
fold_and_stack_towels}/{baseline,arc_vel,arc_dur}` with their `variants/`),
`real/arcdur_abstime/`, `real/multitask/` (moved to `abc_arc/robot_bc/`), the
`_shared/frozen_yam/{towels_hpt300,earbuds_updated_hpt300}` recipes and the
dated `data/abc_arc` pools they selected are removed in favour of the nine
leaves above. `docs/experiment_config_migration.json` maps every old path to
its replacement, or to `null` when it was retired (Elmo's organize
robot + human cotrain leaves, the HPT180 / ABC-DiT rl2 towel variants and the
`new290plus74` pool). All of it is recoverable from Git history (`8f5eeb7a`).
PushT, LIBERO, OAT/LIBERO and synthetic experiments live under `sim/`.

The `e1`, `ravens` and `stationary_recipe` configuration menus are removed.
E1 scripts and runtime compatibility entry points are also removed. Shared
components required by current real recipes have neutral names:
`model/abc_arc/frozen_yam_*`, `model_contract/frozen_yam_wrists`,
`embodiment_resolvers`, and `progress_sampler`. Their behavior is unchanged.

Old CLI paths under `abc_arc/{robot_bc,cotrain}` and `yam_arc_grid` are removed,
not silently redirected. Original jobs/checkpoints must keep their recorded
source/config/normalization. No cluster jobs are migrated by changing these
files. Deleted configurations and scripts remain recoverable from Git history.
