# Task-first experiments

The Cartesian YAM/ABC experiment menu is:

```text
experiment/
  real/
    organize_stationary/   {baseline,arc_vel,arc_dur}.yaml
    fold_and_stack_towels/  {baseline,arc_vel,arc_dur}.yaml
    wireless_earbuds/       {baseline,arc_vel,arc_dur}.yaml
    multitask/             {baseline,arc_vel,arc_dur}.yaml
  sim/
    libero/
    libero_historical/
    oat/
    pusht/
    synthetic/
  _shared/               shared defaults, not standalone runs
```

Select one with `+experiment=real/organize_stationary/arc_vel` (or `arc_dur`
or `baseline`). Cluster resources still come from the explicit
`hydra/launcher=...` selection and require a separate launch preflight.

Each public task YAML is a **complete experiment entry point**: its defaults
select the model graph, data filters/splits/transforms, evaluator, optimizer,
trainer, callbacks and logging. No additional `model=` or `data=` selection
is needed. Shared parents avoid copying those definitions into every leaf.
Supply machine-specific data/weight paths, credentials, a fresh W&B identity
and the reviewed cluster allocation at launch.

Example (submits a run; execute only after launch preflight):

```bash
python egomimic/trainHydra.py -m \
  +experiment=real/organize_stationary/arc_vel \
  hydra/launcher=submitit_lambda_h100_shared
```

| Folder | Default model / data | ARC M / D / R |
| --- | --- | --- |
| organize_stationary | HPT300, RL2 Elmo robot + human organize stationery | 100 / 0.42 m / 24° |
| fold_and_stack_towels | HPT300, frozen RL2 `since0924plus74` pool | 200 / 0.33 m / 100° |
| wireless_earbuds | HPT300, frozen RL2 `wireless_earbuds_updated` pool | 200 / 0.33 m / 100° |
| multitask | HPT300, existing ABC four-task pool | 100 / 0.81 m / 24° |

These are explicit recipe defaults, not a new distance/rotation calibration.
The four-task pool is towels, fake fruits, stationery and earbuds, **not**
the separate seven-task ABC-130k proposal.
Each task's three entries use exactly the same filters, episode lists and split.
Baseline geometry is 100 × 14 time-indexed actions.

ARC uses the canonical [clock contract](../../../docs/ARC_CLOCK_CONTRACT.md)
(repository path `docs/ARC_CLOCK_CONTRACT.md`): velocity rates or **absolute**
seconds from the chunk anchor (decoded by speed-bounded isotonic regression),
first-row start delay, first-moving-stream termination.
Full separation is the default. Use `abc.arc_streams=translation_rotation` for
four clocks (M × 18), `translation` for one clock per arm (M × 16, rotation and
gripper ride the translation supports) or `projection` (M × 16, progress
measured 2 cm off the end effector); token width follows automatically,
including the model and decoder.
No tokenizer or detokenizer algorithm is changed by this YAML reorganization.

The frozen towel/earbud ARC entries are **new canonical-codec recipes** built
on their retained baseline network/optimizer/pool. They read 400 native
frames, use the shared open-loop evaluator, and require fresh normalization
and training. They do not reproduce retired E1 ARC checkpoints.
The baselines retain their original evaluator/training schedule.

## Retained variants

Distinct existing HPT180, robot-only, ABC-DiT, ABC-only/ABC+RL2 towel,
and `new290plus74` pool choices live under each task's `variants/`.
They retain their model/data settings and are not interchangeable aliases for
the primary menu. PushT, LIBERO, OAT/LIBERO and synthetic experiments now live
under `sim/`; retained ABC and human-only menus are not simulation.

The `e1`, `ravens` and `stationary_recipe` configuration menus are removed.
E1 scripts and runtime compatibility entry points are also removed. Shared
components required by current real recipes have neutral names:
`model/abc_arc/frozen_yam_*`, `model_contract/frozen_yam_wrists`,
`embodiment_resolvers`, and `progress_sampler`. Their behavior is unchanged.

Duplicate organize clock-layout leaves were consolidated: choose
`real/organize_stationary/arc_vel` or `arc_dur` and set `abc.arc_streams` instead
of choosing a second file. Distinct old experiment paths are mapped in
`docs/experiment_config_migration.json`; the two retired
`*_translation_rotation` aliases additionally require
`abc.arc_streams=translation_rotation`.

Old CLI paths under `abc_arc/{robot_bc,cotrain}` and `yam_arc_grid` are removed,
not silently redirected. Original jobs/checkpoints must keep their recorded
source/config/normalization. No cluster jobs are migrated by changing these
files. Deleted configurations and scripts remain recoverable from Git history.
