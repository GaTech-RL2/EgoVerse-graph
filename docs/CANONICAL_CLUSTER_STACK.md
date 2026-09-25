# Canonical cluster application stack

## September 25 feature tip

The canonical layers have been reconciled with `main` at
`161e3a0c40182ba003434d3adaeaa6cbd12107d8` using history-preserving merges.
Each canonical parent remains an ancestor of its child; published commits were
not rebased or force-pushed. The cluster overlays are siblings immediately above
`canonical/arc-chunking-modes`, with application trees identical to that tip.

The stack below now continues above `canonical/run-resume` with:

10. `canonical/distance-budget-global-dtw`: episode-distance rollout budgets and
    bidirectional, GT-frame-balanced DTW.
11. `canonical/visual-bc-launch-readiness`: visual-only robot/human recipes and
    corrected BC targets.
12. `canonical/open-loop-full-frame-video-wandb`: per-frame validation video,
    fresh-run logging, and organized retained experiments.
13. `canonical/arc-chunking-modes`: race, multistream, and joint-distance
    translation; separate hybrid R clocks; matching evaluation and cache
    provenance. See `ARC_CHUNKING_MODES.md`.

Pace, Sky2, ICE, and Lambda each retain a cluster-specific configuration overlay
above the shared application tip. Updating refs does not change active checkouts
or running jobs. Lambda's original launcher addition remains in the historical
application ancestry; this feature does not rewrite that published history.

## Original canonical layers

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
   joint-distance calibration, and normalization (historically 20%; retained
   recipes now default to 10%).
9. `canonical/run-resume`: stable W&B identities for resumed training, while
   offline checkpoint evaluation retains a separate W&B identity.

`cluster/pace`, `cluster/sky2`, `cluster/ice`, and `cluster/lambda` are sibling configuration
overlays above the same application tip. Do not put shared model, data, or
evaluator implementation changes in those overlays.

The original cluster commits are preserved by the
`snapshots/cluster-sync-20260922/{pace,sky2,ice}` tags. Existing PR branches are
not deleted or force-pushed by this consolidation.

Syncing a branch ref does not deploy code into an active job's checkout. Running
jobs remain pinned to their recorded source commits; promotion requires a clean
canonical checkout and the usual launch/cache/validation gates. ICE's existing
untracked files must not be overwritten.

The `hpt180`/`hpt300` names are historical profile identifiers, not claims of
exact total parameter counts. The robot visual profiles instantiate roughly
232M/249M parameters, respectively; the bimanual human profile also has its own
stem/domain parameters. Architecture and checkpoint shapes are preserved by
this cleanup. See the experiment-directory README for the retained profile
dimensions and how to measure a composed model.
