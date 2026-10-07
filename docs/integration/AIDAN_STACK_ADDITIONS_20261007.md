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

Validation: 104 focused CPU tests passed, including existing E1, inference,
open-loop and Cartesian hybrid coverage. No training allocation or physical
robot execution was performed. Original donor branches remain untouched.

## Rollout layer

Donor: `aidan/arc-token-shapes-20261001`,
`b1ecba6310fac41e7441a743f756362a7ca65517`.

This independent branch is ported by capability rather than by replacing the
runtime tree. Its old inference class discovery, shared rotation-clock behavior,
invalid station YAML and unrelated removals must not overwrite the maintained
stack. Rollout completion receipts and detailed dispositions are added with the
second runtime child.
