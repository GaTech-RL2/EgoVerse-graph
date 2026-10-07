# Aidan ARC stack additions

Source repository: `GaTech-RL2/EgoVerse-graph`. Integration starts at PR #198,
`7ccb609626c62fafce5c1181bc79f576e0829f2b`, preserving its existing Graphite parents.
The additions are runtime children; validation artifacts remain on this companion.

## BC recipes and hybrid token contracts

Donor: `aidan/arc-bc-consolidated`, PR #160,
`8ff8da40fbb22dba5ae36fc0d85e4e1fd15ebc20`.

The first child adds 12 training recipes: Elmo/Aidan 218/11 time, duration,
velocity and hybrid tokens; HPT300 time/duration twins; DP180 time; and five
YAM/Aria co-training variants. Co-training uses 37 frozen human episodes plus
217 slow-pace YAM training episodes and the shared 24-episode YAM validation
set. Source actions use 100-frame windows at 30 Hz.

All 12 composed neural stage graphs, optimizer and scheduler configs match
the donor, apart from stable `sampler` stage-ID metadata. Exact episode sets
match the donor. Receipts are under `evidence/aidan-bc-*`.

The port retains current ZarrDataModule/data requirements and model-owned
inference contracts. E1 duration/velocity hybrid tokens have 18 channels and
four independent timing streams. Their rotation budget is explicit (2π for
these recipes) and is bound into the inference artifact. The existing plain
16D contracts and generic pipeline inference implementation are preserved.

Generators now share checkout-local builders, reproduce inherited YAMLs, and
retain the correct split and rotation budget. PACE Python runs through `srun`.
The open-loop evaluator recognizes hybrid tokens without losing start delays.

The BC layer separately resolves all 466 of its shipped configs. A fresh
checkout restored its pinned validation artifacts and passed 143 focused CPU
tests. Additional launcher-guard tests cover short `atl1-login` PACE hostnames.

## Rollout layer

Donor: `aidan/arc-token-shapes-20261001`,
`b1ecba6310fac41e7441a743f756362a7ca65517`.

The second child adds token layouts, motion/hold speed controls, fastest-stream
prefix execution, rollout episode recording, checkpoint substring selection,
HDF5 tempo diagnostics, teleop gripper force controls and camera shutdown cleanup.
Every donor runtime path has an explicit disposition in
`evidence/aidan-rollout-dispositions.json`. ImageNet image normalization already
exists in the parent stack and needs no duplicate change.

Current Cartesian profiles explicitly select `canonical200`. The frozen M28
codec requires `m28_99be4af0`; PR #193 uses an explicit restricted compatibility
wrapper; E1 uses `e1_8ff`, and historical tri layouts are decode-only. Native
shapes alone cannot identify their timing semantics. Current per-arm rotation
clocks remain independent, and decoder implementations validate declared native
and Cartesian shapes before strict model loading.

ARC flow rollout declarations use 20 sampling iterations, a 50% execution
prefix and fastest-stream mode. Time baselines retain their declared sampling
defaults. These defaults change inference behavior; they do not change the
training optimizer, scheduler or neural graph. All 12 BC recipes retain donor
neural/optimizer/scheduler parity after the rollout port, as recorded in
`evidence/aidan-rollout-profile-composition.json`.

Generic pipeline declarations bind typed public control paths, including nested
decoder sequences. Owners preflight updates, and failures roll back all touched
values. The graph loader and policy use declared interfaces rather than model
class or name discovery. Shared YAML declarations avoid repeating profiles.

Recording initializes HDF5 in a background worker, keeps terminal requests
outside the bounded row queue, and records planned versus actually written
command rows separately. Gripper updates preflight every arm, enforce the
supported force cap and verify rollback, including cancellation. Failed model
replacement clears prior observation history.

## Validation and reproduction

The final affected CPU regression suite and exact source content hashes are in
`evidence/aidan-final-validation.json` and `evidence/aidan-final-cpu-junit.xml`.
All 469 final shipped YAML configs resolve. Ruff 0.8.6 lint/format checks and
JavaScript syntax checks cover the source changes. Linux CI on the runtime PRs
retains the complete existing config, component, CPU and installed-wheel gates.

This branch is a validation archive based on the existing companion branch. Its
CI checks complete artifact inventory, hashes, Python syntax and immutable
source pins. To run functional tests, check out the corresponding runtime child,
activate its locked environment and run `.github/scripts/restore_validation.py`;
the runtime's `.github/validation-ref` pins this archive. Do not merge the
companion's historical runtime tree into the maintained stack.

No training allocation, historical checkpoint load, physical robot rollout or
station deployment was performed. These checks establish source and config
compatibility; they do not measure physical rollout performance. The original
donor branches and the existing 13 stack PR heads remain unchanged.

Read-only integration with current `main` at
`0dc31eff5fd9beb5370f1840ed366ae7218f3f97` finds four existing modify/delete
conflicts in `tests/test_action_flow_{diagnostics,stages,training_behavior,unite_recipe}.py`.
Main modified these tests after the existing stack moved them to its validation
archive. The two new runtime layers remain parented above #198; review readiness
does not establish merge readiness of the complete old stack against main.
