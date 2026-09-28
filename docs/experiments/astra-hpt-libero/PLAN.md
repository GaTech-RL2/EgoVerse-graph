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

Foundation is published in draft PR #178 on top of integration PR #159.
The one-L40 OSMO v1 run passed random full-model update/inference; provider
catalog access returned HTTP 429 and simulator imports failed. All fourteen
outputs were preserved on R2, including the model audit receipt. OSMO's own
output collector could not read one mode-0600 JSON file; publication now uses
mode 0644 for credential-free artifacts.

The follow-up implements closed-loop typed teacher control and full simulator
state snapshots. Local calibration passed 60/60 resets, exact state/image
replay, world-frame axis and camera-left checks, and gripper polarity. Each
stage has a successful hand-authored engineering teacher fixture; failures
and controller iterations remain under scratch/astra_push. These episodes
are excluded from production replay and from Astra novelty/commissioning.

The next source-pinned OSMO preflight rechecks the catalog with bounded retries,
uses LIBERO's explicit source import path, and measures the real three-stage
HDF5 batch update plus all physical checks in the Linux lock. No Astra-authored
commissioning demonstrations, production updates, or policy evaluation results
exist yet. W03 split freezing, W04 novelty/commissioning, W06–W10 remain open.
