# AstraPush implementation and evidence

User request: implement the supplied `astra-hpt-libero-signs-of-life.zip` in
EgoVerse-graph. The unchanged version 2 specification is in
`specification-v2/`. Archive SHA-256:
`635de89644434766dd8d11b8b56ad33262b0637b2a0e5c9f53642e25853f0be7`.
All 17 packaged checksums passed before implementation.

## Repository decision

Use EgoVerse-graph, as explicitly requested, instead of the specification's
proposed separate AstraExps repository. Start from `eda6f8d1`, the integration
stack's HPT graph parity layer (PR #159), in an isolated worktree. This preserves
the existing experiment and integration checkouts. Nothing is deleted.

Reuse `PipelineAlgo`, HPT stems/trunk, and the flow denoiser. New reusable
language/masking capabilities belong in their existing model/stage folders.
The AstraPush scene grammar, simulator adapter, acquisition and matched
curriculum experiment belong under `egomimic/experiments/astra_push/`, with
declared YAML model semantics. Do not restore the legacy algorithm runtime.
The simulator gets its own dependency lock and process. The learner uses the
existing locked graph environment. GPU work uses one OSMO L40/L40S; no sky1/sky2,
distributed learner, pretrained weights or DQC restart.

## Work sequence

1. W01: pin sources/environments, verify actual Astra model access, audit fully
   random graph construction, execute an engineering update and render.
2. W02–W05: typed scene/task/teacher contracts, independent trajectory semantics,
   LIBERO compiler and assets, commissioning gates, HDF5/observation transport.
3. W06–W08: freeze partitions, collect verified common data, warm-start, then
   run the two budget-matched arms with exact checkpoint/ledger resume.
4. W09–W10: sealed paired evaluation and videos, clustered uncertainty, then
   the fixed twelve-candidate gradient diagnostic and actual cost report.

Production training waits for the six-scene novelty/witness gate. Each gate
requires measured receipts, not a synthetic test or illustrative JSON. When a
required service is unavailable, record the exact blocker and continue only
independent implementation; never substitute another generator or curriculum.

## Current status

Specification imported and checksums verified; implementation in progress.
No AstraPush demonstrations, production updates, or evaluation results yet.
