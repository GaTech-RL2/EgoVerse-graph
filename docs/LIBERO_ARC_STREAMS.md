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
and dense float32 controls accompany each replay. Decoded actions are sent to
the simulator as float32, matching GraphPolicy inference; the legacy helper's
default float64 transport remains available for reproducing historical reports.
Simulator repeatability and
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

## Parallel expansion, October 2

The user authorized queuing more work. The four running reference-STK policies
retain their original workflows, checkpoints and automatic evaluation. The other
84 cells move into four additional workflows generated by
`scripts/benchmarks/launch_libero_stream_parallel.py`, with three independent
training lanes per suite. This allows 16 concurrent training jobs (64 L40S GPUs),
plus at most eight evaluation GPUs during overlap with the original workflows.
The first new cells in each suite are reference-DUR, gripper-STK and
component-time-DUR. All 88 unique cells and their training budgets remain intact.

New jobs pin the same training source, `69ff3fab69c2d322b590809461c80d70878ba71f`,
and embed that suite's completed replay receipt. Replay source hashes, geometry,
candidates and control results are revalidated by the unchanged training runner.
Training lanes do not wait for reference evaluations. A separate CPU task waits
for the original reference-STK evaluation; the new reference-DUR evaluation runs
first in the new workflow. Candidate evaluations therefore retain their exact
paired comparisons without occupying GPUs while waiting for reference scores.

OSMO does not support editing the submitted serial DAG. Once replacement jobs
are submitted, the launcher checks that every old tail task is still WAITING and
the old reference-DUR artifact prefix is empty, then writes a checksummed handoff
marker there using a conditional put. The original ArtifactUploader refuses an
occupied prefix before training starts. Thus old `train-01` deliberately stops
at its no-overwrite check, and its dependent tail becomes FAILED_UPSTREAM.
Original `train-00` and `evaluate-00` are independent of that tail and finish
normally under OSMO's [upstream failure rules](https://nvidia.github.io/OSMO/release/6.2/user_guide/workflows/lifecycle/index.html).
The old wrapper can consequently show FAILED even when its retained STK training
and evaluation completed successfully. Treat only its handed-off tasks as retired,
and use the new workflow IDs for those cells. The receipt records the exact old
and new IDs; no in-progress training or checkpoint is overwritten or interrupted.

## Checkpoint recovery, October 3

Shared-pool quota enforcement canceled all active workflows overnight, after
Spatial and Goal reference-STK training finished. Their standalone evaluations
resume checksummed, paired rollout episodes from the same final checkpoints.
Object, LIBERO-10 and the first twelve additional cells have partial checkpoints.
`launch_libero_stream_recovery.py` resumes these fourteen cells and preserves the
remaining seventy-two unstarted cells in the parallel queue.

The recovery helper runs against the same checked-out training source,
`69ff3fab69c2d322b590809461c80d70878ba71f`. Its own source hash is recorded separately.
It verifies the source runtime, replay, representation, original preflight,
checkpoint checksum, optimizer, normalizer, EMA and original four-GPU budget
before passing `ckpt_path` to the unchanged trainer. It performs a simulator
smoke test of the recovered checkpoint and never falls back to fresh training
when recovery fails. Lightning's periodic saves contain the final batch of an
epoch before incrementing its completed-epoch counter; the helper validates this
boundary and preserves Lightning's loop state. A real train/save/resume test
verifies that optimizer and EMA updates continue without repeating that epoch.

CPU waiter images use the fully qualified `docker.io/library/python:3.11-slim`
name, since L40S-01 rejects ambiguous short image names. Recovered reference
evaluations explicitly identify their original training run. Object and LIBERO-10
reference training/evaluation are dependencies within their recovery workflows;
Spatial and Goal wait for the separate evaluations of their completed policies.
All recovery submissions retain NORMAL priority and remain subject to shared-pool
preemption. This is explicit recovery, not automatic requeue or protected capacity.

## Additional GPU slots, October 3

`launch_libero_stream_expansion.py` moves the 72 unstarted cells into six training
lanes per suite. Its four workflows request 24 additional four-L40S jobs (96 GPUs),
while the 14 checkpoint recoveries continue. The overlap permits 38 training jobs
(152 GPUs); once the recoveries finish, the new workflows retain 24 training slots.
These are concurrency ceilings, not promises of allocated capacity. All work uses
NORMAL priority on L40S-01 and remains subject to shared-pool scheduling.

Each new workflow contains two CPU waiters for the existing STK and DUR reference
evaluations. Candidate evaluations require the matching reference's complete,
checksummed 2,500-episode result, exact source revision and original training run.
Training does not depend on either waiter. The expansion does not rerun reference
training, checkpoint recoveries or replay, and retains the pinned model and budget.

After each replacement is accepted, the handoff verifies that old tasks 03..20
remain WAITING and that retiring training heads 03, 04 and 05 affects exactly those
cells. Conditional checksummed markers reserve only their empty artifact prefixes.
The no-overwrite guards stop the superseded tails; existing training tasks 00..02,
their evaluations, and the reference-STK jobs remain independent. Old workflow
wrappers may eventually report FAILED for these intentional retirements. The
canonical expansion manifest records all 88 unique cells and replacement IDs.
The maximum simultaneous evaluation allocation during overlap is 12 L40S GPUs.

## Isolated task repair, October 4 UTC

A GitHub dependency fetch timed out before Spatial XYZ-scalar-DUR training
started. Its two dependent training cells and sixteen evaluations were marked
FAILED_UPSTREAM, while the other training lanes continued. The repair launcher
selects only terminal failed tasks and requires every affected artifact prefix
to be empty; existing checkpoint data instead requires explicit resume.

The three training tasks retain their source, recipes, run IDs and four-L40S
allocation. Dependency installation gets three bounded attempts. Thirteen CPU
tasks wait for policies being trained in the healthy original lanes, validate
checksummed completion/optimizer/EMA/layout receipts, and reconstruct the pinned
evaluation requests. The native evaluator independently validates the downloaded
checkpoint and complete training budget before rollouts. Sixteen recovered
evaluations run in up to four independent lanes, retaining their STK/DUR reference
gates. No healthy job is canceled, and already successful training is not repeated.

## Validation

CPU coverage includes legacy column-permutation equivalence, diagonal motion,
stationary actions, gripper events and overflow, sub-R rotation, noncommuting
rotations, per-component time changes, duration-vs-velocity dwell behavior,
normalization bounds, invalid partitions, cache identity, all 88 Hydra cells,
workflow dependencies, stale replay rejection, and real shared training / EMA
reload / target-free inference for the reference and three representative codecs.
The reference uses the full production U-Net and the same checkpoint verifier
as the GPU preflight. All 88 configuration checks use the model-only tree
serialized by ModelWrapper. The seed, geometry and loader provenance required
for checkpoint verification are resolved into the saved benchmark protocol;
verification never assumes the top-level Hydra configuration is in a checkpoint.

Paired replay success and trained-policy success are distinct measurements.
No full policy scores are claimed until the full optimizer/EMA and rollout
receipts have completed.
