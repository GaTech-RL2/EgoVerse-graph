# Current real-task recipes

The current ARC cleanup stack is based on PR #263. Its public experiment
entry points live under `egomimic/hydra_configs/experiment/real/`:

- `organize_stationary/{baseline,arc_vel,arc_dur}`
- `fold_and_stack_towels/{baseline,arc_vel,arc_dur}`
- `wireless_earbuds/{baseline,arc_vel,arc_dur}`
- `multitask/{baseline,arc_vel,arc_dur}`

Each YAML selects a complete model/data/evaluator/training/logging configuration.
See the [experiment menu](../egomimic/hydra_configs/experiment/README.md) for
defaults, frozen dataset pools, variants and launch selection.

The canonical [clock contract](ARC_CLOCK_CONTRACT.md) has velocity or log
interval duration, first-row delay, and four or six independent clocks.
Neither retired codecs nor absolute-seconds layouts are selectable here.

Frozen towel/earbud recipes retain their HPT300 architecture and episode
membership. The shared definitions are `model/abc_arc/frozen_yam_hpt300_time`,
`model/abc_arc/frozen_yam_hpt_flow`, `model_contract/frozen_yam_wrists`,
and `experiment/_shared/frozen_yam/`. These are dependencies, not extra
public run menus. ARC leaves override the baseline codec and evaluator.

Older experiments, removed launchers and checkpoint layouts remain recoverable
from their original Git commits. Preserve every running job's recorded source,
config and normalizer. Do not switch a checkout used by active jobs.
