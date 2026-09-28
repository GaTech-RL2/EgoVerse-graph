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

Full learner training and adaptive-versus-uniform evaluation have now been
authorized. The coordinator implementation and current launch status are in
[FULL-RUN-PLAN.md](FULL-RUN-PLAN.md), on the fifth stack layer. The measured
commissioning results below remain the completed evidence until production
receipts arrive; source implementation alone is not a learner result.

The user-authorized Codex `gpt-6-astra` route has passed the generated-data
commissioning and learner integration test on one OSMO L40. The corrected
source is `41778ab32bb84d10bea8745763bfe8107f152b56`, in draft PR #181 on top of
#178 foundation → #179 simulator → #180 protocol. Read
[SUBAGENT-PILOT.md](SUBAGENT-PILOT.md) for the transport amendment and measured
receipts; the unavailable gateway is no longer a blocker for this route.

Four actual, context-isolated Astra children returned valid responses without
repair: six commissioning templates, two adaptive synthetic-report decisions,
and one blinded uniform decision. Corrected commissioning produced 30 accepted
demonstrations in 31 attempts, five per scene and ten per stage. All six exact
fresh-process restores and the full random HPT real-data update/inference
passed. Visual review prompted a distinct gray optional fixture while retaining
the yellow reference marker. The first run remains preserved and charged:
31 earlier plus 31 corrected acquisitions use 62 of the 120-attempt cap.

Production optimizer updates and student policy rollouts are **zero**. Frozen
evaluation partitions, common seed collection/training, complete resumable
round orchestration, matched U/A learner evaluation and the gradient diagnostic
remain open. Teacher acceptance and synthetic decision fixtures do not measure
learner competence or an adaptive-curriculum advantage. Commissioning episodes
are excluded from production imitation replay.

## Historical engineering gates and gateway blocker

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

OSMO v2 confirmed a concrete external blocker: the existing Astra credential
had already exceeded its configured gateway budget (HTTP 429,
`budget_exceeded`). Three catalog attempts were archived; zero generation
calls occurred. The user has been asked to replenish that credential or name
another approved credential. Do not retry generation until access changes.
The simulator lock also required explicit matplotlib/imageio dependencies;
those are now pinned. Run independent engineering checks with
`build_workflow --skip-provider`, which omits the Astra credential and writes
an explicit failed/not-run provider gate receipt.

The protocol layer now has tested typed round decisions, a blinded uniform
generation context, fixed quota validation and a single-writer SQLite ledger
that preserves incurred attempts across crashes. It does not yet constitute
the complete round runner or satisfy W07's actual-provider fixture test.

## Earlier measured calibration gate (2026-09-28)

OSMO `astra-hpt-calibration-20260928-v3-1` completed on one L40 in 488.54
seconds. All 60 reset checks and three fresh-process replay checks passed;
physics states and camera frames matched exactly. All three engineering
teachers succeeded. The real three-stage HDF5 batch completed a full random
HPT update and 50-step flow inference, with gradients in every component.
There are 244 preserved artifact files; hashes and the compute receipt are
under `evidence/osmo-calibration-v3/`.

At this earlier point the simulator/learner engineering checks had passed,
but authenticated Astra generation and generated commissioning were absent.
The later user-authorized subagent route addresses those two limitations while
leaving the full curriculum runner, frozen holdouts and final analyses open.
The original gateway credential remains exhausted and must not be retried.

Published stack: #178 foundation → #179 simulator → #180 protocol. The original
OSMO source commits are retained under `astra-hpt-source-*` tags. A later
metadata correction names the verified controller coordinate frame `world`
instead of `controller_input`; its dedicated deployment tests passed. No
model weights, action scaling or physical controller behavior changed in that
correction.

The self-contained OSMO teacher-video preview is at
`/Users/rpunamiya/Desktop/GEAR/handover_figures/astra-hpt-libero-20260928/engineering-preview.html`.

Repository CI passed on all three layers. The protocol code at `5e7bd472`
passed 1,481 CPU tests plus configuration, installed-wheel, static and lock
checks in GitHub Actions run `36456471469`. The pending checkpoint/round
coordinator and final analyses are not covered by that claim. W08 cannot begin
before the remaining W01–W07 prerequisites pass under the amended manifest.

## User-authorized subagent test

The user subsequently requested testing the pilot with Astra directly as a
Codex subagent. [SUBAGENT-PILOT.md](SUBAGENT-PILOT.md) records this new transport
version, its limits, and the first live commissioning test. An isolated
`gpt-6-astra` child returned six valid scene/task/teacher templates in one
logical request. Its raw response and exact request are preserved under
`evidence/subagent-pilot-v1/generation/commissioning-001/`.

The generated layouts pass typed geometry, unique-signature and composition
checks. The pinned stock inventory contains 130 BDDL files; all have different
physical asset composition. Ten files retain explicitly recorded dangling,
unused affordance-region references. This does not claim those region graphs
were fully normalized. The corrected one-L40 OSMO run measured actual teacher
yield, all thirty commissioning episodes, six fresh-process reloads, frozen
normalization and a discarded HPT engineering update; those checks passed.

The generator exchange and collector passed 29 focused CPU checks, including
induced interruption without repeated acquisitions, exact five-per-scene
quotas, immutable request/response binding, and commissioning exclusion from
production imitation replay. No production optimizer updates have occurred.
