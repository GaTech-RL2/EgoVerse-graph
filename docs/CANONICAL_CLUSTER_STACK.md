# Canonical cluster application stack

The shared application stack is rooted on GitHub `main`. Each branch represents
one user-visible feature; follow-up fixes stay on that feature branch.

Bottom to top:

1. `canonical/arc-data-runtime`: native 100-frame baseline windows, distance-aware
   ARC source windows, and timed reconstruction.
2. `canonical/hpt-clock-heads`: separate clock heads and shared-stem mixtures.
3. `canonical/open-loop-evaluation`: executed-prefix metrics, per-frame capped
   GT/prediction videos, joint-distance semantics, and trajectory comparisons.
4. `canonical/checkpoint-validation`: offline checkpoint sweeps and their logs.
5. `canonical/robot-bc-recipes`: RL2 stationery and ABC towels BC recipes.
6. `canonical/hybrid-arc`: independent translation/rotation caps and clocks with
   per-waypoint velocity.
7. `canonical/human-bc-recipes`: human-bimanual MECKA folding-clothes recipes.
8. `canonical/visual-abc-multitask`: no-language ABC campaigns, 30 Hz sampling,
   joint-distance calibration, and 20% normalization.
9. `canonical/run-resume`: stable W&B identities for resumed training, while
   offline checkpoint evaluation retains a separate W&B identity.

`cluster/pace`, `cluster/sky2`, and `cluster/ice` are sibling configuration
overlays above the same application tip. Do not put shared model, data, or
evaluator implementation changes in those overlays.

The original cluster commits are preserved by the
`snapshots/cluster-sync-20260922/{pace,sky2,ice}` tags. Existing PR branches are
not deleted or force-pushed by this consolidation.

Syncing a branch ref does not deploy code into an active job's checkout. Running
jobs remain pinned to their recorded source commits; promotion requires a clean
canonical checkout and the usual launch/cache/validation gates. ICE's existing
untracked files must not be overwritten.

Known pre-existing limitation: the visual-only recipes currently labeled
HPT180/HPT300M instantiate approximately 232M/249M total parameters. This stack
reconciliation preserves the submitted experiments; it does not silently resize
their models.
