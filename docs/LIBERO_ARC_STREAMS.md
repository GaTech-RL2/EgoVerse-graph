# LIBERO ARC grouping and timing sweep

This is a fixed categorical experiment, not another R/D/M search. All arms use
R=384 degrees, D=1.6 metres, M=36, the frozen shared STK/DUR profile selected on
2026-09-21. R and D are accumulated horizon caps, not minimum emission quanta.
The existing LIBERO encoder retains sub-R rotation. Cross-window rotation
carry is pending an explicit residual/quantum/state definition; no duplicate
of the reference is presented as a new carry experiment.

The runtime branch is `codex/libero-arc-streams-20261001`, based on the proven
LIBERO runtime at `5c162c13` (PR #195). It does not overwrite the consolidated
graph work in PR #198. Tests, this protocol, historical benchmark records and
new receipts live on `codex/libero-stream-validation-20261001`. The runtime
pins an immutable validation commit and can restore these files with
`python .github/scripts/restore_validation.py`.

## Matrix

Every row is run with STK and duration, on Spatial, Object, Goal and LIBERO-10.
LIBERO-90 remains deferred. There are 88 training cells, each followed by the
same 2,500-episode evaluation. One training seed (42) is used; the five eval
repetitions are not independent training seeds.

| Variant | Support groups | Timing/clock change | Main comparison |
| --- | --- | --- | --- |
| reference | XYZ + gripper; SO(3) | one rate/duration per group | existing LIBERO codec, timing columns moved to the end |
| gripper | XYZ; SO(3); gripper events | own gripper distance and clock | reference |
| x_yz | X; YZ; SO(3); gripper | group clocks | gripper |
| y_xz | Y; XZ; SO(3); gripper | group clocks | gripper |
| z_xy | Z; XY; SO(3); gripper | group clocks | gripper |
| xyz_scalar | X; Y; Z; SO(3); gripper | group clocks | gripper and each axis-vs-pair arm |
| component_velocity | XYZ; SO(3); gripper | component rate labels, shared group clock; duration labels averaged into shared group clock | gripper |
| component_time | XYZ; SO(3); gripper | component rate/duration labels and independent component clocks | component_velocity |
| angular_driver | XYZ; integrated world angular commands; gripper | group clocks | rotation representation control |
| angular_scalar | XYZ; separate angular-command axes; gripper | group clocks | angular_driver |
| all_scalar | X; Y; Z; each angular-command axis; gripper | independent supports and clocks | angular_scalar and xyz_scalar, accounting for rotation representation |

Per-dimension **time** does not mean per-dimension support sampling. With shared
supports its duration targets are equal across components; independent predictions
can run those components at different speeds. The shared-clock component-duration
arm averages those predictions, providing a matched-channel-count control. STK
component targets are signed displacement / seconds. A shared group clock uses
the ratio of displacement and velocity norms; independent clocks invert each
component separately. Zero displacement and zero rate cannot encode a dwell.
There are no hidden time channels repairing that limitation.

SO(3) shape remains rotation6d in the main arms. Independent angular clocks
control the components of world-frame log increments between those supports.
The decoder composes rotation increments at the union of clock breakpoints.
It does not independently interpolate Euler coordinates or add rotations.
Fully scalar support sampling requires an explicit angular-command representation;
the angular_driver control separates that change from stream grouping.

Gripper is a held controller command. Its own distance is total variation in
command units, with support at each command event, and no D/R cap. It never
invents intermediate apertures. If event count exceeds M, that reconstruction
loss is measured. A constant gripper has no movement clock and remains held.

## Matched training and evaluation

The first campaign uses the current U-Net DP (widths 256/512/1024, 100 DDIM steps),
because all four suites have working prior scores. The prior OAT-DP Object anomaly
is unresolved. This campaign does not change that backbone or its checkpoints.
All input/output channels are active. Parameter counts vary slightly with
representation width and are recorded in the parameter manifest and rollout
metadata; no unused parameters or channels are added to claim exact equality.
The existing mean epsilon MSE is retained, so additional meaningful timing
channels also change the fraction of that loss assigned to timing. Treat this
as a representation-level comparison, not proof of an isolated architectural effect.

Each run uses the decoded loader, AdamW 5e-5 (observation encoder 1e-5), betas
0.9/0.95, no weight decay, EMA, global batch 1024, and 5001 epochs. Full optimizer
budgets are Spatial 270054, Object 325065, Goal 280056, and LIBERO-10 605121.
There are two observation frames, 32 predicted actions, 16 executed actions per
replan, and a maximum of 550 evaluation steps. Evaluation uses 10 tasks × 50
trials × 5 repetitions, with exact episode IDs and initial-state hashes paired.
Checkpoint source, normalization, stream schema, geometry, optimizer and EMA
counts are checked before evaluation. Different representations are allowed;
different datasets, initial states, geometry or rollout protocols are rejected.

All 22 representations first replay the same 30 demonstrations per suite
(demos 0, 1, 2 of every task). These demos have been used previously and are
not claimed to be a fresh holdout. Raw, repeated raw, original-precision raw,
and dense float32 controls accompany each replay. Simulator repeatability and
dense reconstruction must pass. Poor candidate success or timing coverage is
reported, not used to silently remove an unfavorable arm from the experiment.

The OSMO DAG limits each suite to one 4-L40S training task and one 1-L40S
evaluation task at a time: at most 16 training GPUs plus four evaluation GPUs
across the campaign. Training starts only after paired replay. Every training
arm runs a separate two-update GPU preflight at the full per-rank batch size,
reloads its checkpoint and exercises all ten tasks before fresh full training.
Evaluation depends on that arm's final checkpoint and is serialized separately,
so it can overlap the next training arm. A failure stops dependent work for
inspection. Jobs are pinned to a full Git SHA and never overwrite run artifacts.

This runtime does not automatically resume quota-preempted jobs. Existing
content-addressed checkpoints are preserved for an explicit recovery workflow;
restart from scratch must not be labeled a resume.

## Validation

CPU coverage includes legacy column-permutation equivalence, diagonal motion,
stationary actions, gripper events and overflow, sub-R rotation, noncommuting
rotations, per-component time changes, duration-vs-velocity dwell behavior,
normalization bounds, invalid partitions, cache identity, all 88 Hydra cells,
workflow dependencies, stale replay rejection, and real shared training / EMA
reload / target-free inference for three representative codecs.

Paired replay success and trained-policy success are distinct measurements.
No full policy scores are claimed until the full optimizer/EMA and rollout
receipts have completed.
