# Absolute-duration single-clock ARC: Aidan's defaults per task

Isolated entry points for the codec protocol the 2026-10-20 Phoenix runs train
(vault note *ARC Abstime And Interval Arcvel Run Protocol 10-20*), on the stack's
shared frozen-YAM ARC plumbing. They are separate from the task folders' `arc_dur`
/ `arc_vel` entries, which keep Aniketh's full-separation defaults.

| Leaf | Pool (train / val) | Graph | M / D | Source window |
| --- | --- | --- | --- | --- |
| `organize_stationary` | slow-pace 366 + Elmo 51 of 2026-10-19 (417 / 24), `data/abc_arc/stationery_sp366e51_bc_time` | HPT 640 x 8, flow 384 x 10 | 200 / 0.22 m | 100 frames |
| `wireless_earbuds` | `earbuds_updated` + Elmo's 49 takes of 2026-10-20 (319 / 15), `data/abc_arc/earbuds_plus49_20261020` | HPT300 | 200 / 0.22 m | 100 frames |
| `fold_and_stack_towels` | `since0924plus74` (443 / 17) | HPT300 | 200 / 0.33 m | 100 frames |

Shared defaults (`_base/arcdur_abstime.yaml`): `clock_mode: duration` (absolute
seconds from the anchor, speed-bounded isotonic decode), `streams: translation`
(M x 16, one clock per arm; rotation and gripper ride the translation supports),
open-loop validation on the 25 % prefix, deployed 50 % waypoint prefix with
first-moving-stream replanning. R (100 deg) is required by the codec but inert
for single-clock streams.

```bash
python egomimic/trainHydra.py -m +experiment=real/arcdur_abstime/organize_stationary \
  hydra/launcher=... logger.wandb.id=<leaf>_<date>_s42
```

Variant by override: interval arcvel is `abc.arc_clock_mode=velocity`. ARC
Projection lives on its own branch off this stack. Override the W&B id on every launch.
These are fresh-run recipes: checkpoints trained on `aidan/yam-tasks-consolidated-20261018`
keep their recorded source (their token has no start-delay row).
