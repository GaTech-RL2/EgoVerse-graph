# Current real-task recipes

The current ARC cleanup stack is based on PR #263. Its public experiment
entry points live under `egomimic/hydra_configs/experiment/arc_real/`:

- `stationary_{time,velocity,duration}` — 417 training / 24 validation episodes
- `towels_{time,velocity,duration}` — 462 training / 17 validation episodes
- `earbuds_{time,velocity,duration}` — 339 training / 15 validation episodes

ABC-lab multitask recipes remain under `experiment/abc_arc/robot_bc/multitask/`,
separate from the nine rl2 real-task recipes introduced in PR #277.

Each YAML selects a complete model/data/evaluator/training/logging configuration.
See the [experiment menu](../egomimic/hydra_configs/experiment/README.md) for
defaults, pinned dataset pools, variants and launch selection, and the
[configuration guide](CONFIGURATION.md) for the shared-component layout.

The canonical [clock contract](ARC_CLOCK_CONTRACT.md) has incoming waypoint
velocity or absolute waypoint seconds, with start delay in row zero. The
`full_separation`, `translation_rotation`, `translation` and `projection`
layouts use six, four, two and two clocks respectively. `_duration` selects the
absolute-seconds duration representation; the separate `arc_real/arcdur_abstime`
experiment menu is removed, not required to enable it. Retired codecs are not
selectable.

All nine use HPT300, 240k optimizer steps and 25% open-loop evaluation. ARC
defaults to M200, a 100-frame source window, one translation clock per arm and
D=0.22 m for stationery/earbuds or D=0.33 m for towels. The rollout contract
executes 50% of the ARC waypoints; these distinct evaluation/deployment prefixes
are preserved from PR #277.

YAM task recipes retain their HPT300 architecture and episode
membership. They link directly to `model/hpt/yam_300m` (baseline) or
`model/hpt/yam_300m_arc` (ARC), their matching data adapter, and the
open-loop evaluator. Reusable settings live in `model/base/`, `trainer/` and
`model_contract/`; no real recipe imports another experiment or `_shared/`.

Older experiments, removed launchers and checkpoint layouts remain recoverable
from their original Git commits. Preserve every running job's recorded source,
config and normalizer. Do not switch a checkout used by active jobs.
