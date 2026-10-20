# Task-first experiments

The Cartesian YAM/ABC experiment menu is:

```text
experiment/
  organize_stationary/   {baseline,arc_vel,arc_dur}.yaml
  fold_and_stack_towels/  {baseline,arc_vel,arc_dur}.yaml
  wireless_earbuds/       {baseline,arc_vel,arc_dur}.yaml
  multitask/             {baseline,arc_vel,arc_dur}.yaml
  _shared/               shared defaults, not standalone runs
```

Select one with `+experiment=organize_stationary/arc_vel` (or `arc_dur`
or `baseline`). Cluster resources still come from the explicit
`hydra/launcher=...` selection and require a separate launch preflight.

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
(repository path `docs/ARC_CLOCK_CONTRACT.md`): velocity or log **interval**
seconds, first-row start delay, first-moving-stream termination.
It is not E1 or cumulative absolute seconds. Full separation is the default.
Use `abc.arc_streams=translation_rotation` for four clocks; token width
automatically changes from M × 20 to M × 18, including the model and decoder.
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
the primary menu. Other task/model families (PushT, Libero, human-only,
Ravens, etc.) are untouched.

Duplicate organize clock-layout leaves were consolidated: choose
`organize_stationary/arc_vel` or `arc_dur` and set `abc.arc_streams` instead
of choosing a second file. Distinct old experiment paths are mapped in
`docs/experiment_config_migration.json`; the two retired
`*_translation_rotation` aliases additionally require
`abc.arc_streams=translation_rotation`.

Old CLI paths under `abc_arc/{robot_bc,cotrain}` and `yam_arc_grid` are removed,
not silently redirected. Original jobs/checkpoints must keep their recorded
source/config/normalization. No cluster jobs are migrated by changing these
files. Historical E1 recipe-generation scripts are not a supported way to
regenerate this task menu.
