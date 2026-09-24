# Graph integration work plan

Requested 2026-09-24 via `~/Downloads/Untitled (3)`; the source contract is
preserved in `../GENERIC_PIPELINE_CONTRACT.md`. This is implementation work,
not a claim that the integration or cutover has passed.

Pinned live audit heads:
- EgoVerse: `ec5c903c067bf1b29bbff781bc414c6df2bcf1f2`.
- EgoVerse-graph: `161e3a0c40182ba003434d3adaeaa6cbd12107d8`.

Work is isolated in `EgoVerse-graph-integration-20260924`, initially based on
graph main. Existing worktrees, branches, datasets and checkpoints are retained.
The stopped DQC experiment remains stopped. Use OSMO L40/L40S for any required
GPU validation; do not use sky1/sky2. No legacy runtime removal before all
applicable parity and assembled-tree gates pass.

## Reviewable layers

1. Audit live trees, record conflict inventory and a capability parity manifest.
2. Define model-owned inference declarations, typed runtime capabilities,
   evaluator requirements, data context and generic lifecycle orchestration.
3. Resolve the AV/OpenPI environment policy with pinned dependency evidence;
   restore package resources and cloud-free local imports.
4. Port retained HPT and PI recipes, multi-head/keypoint/language paths,
   initialization semantics, camera naming and prompt warnings.
5. Preserve data/visualization tools, generic video requirements and standalone
   evaluation without reopening training data or forcing one device.
6. Add recursive config, CPU functional, packaging and CI gates; run scheduled
   real-weight/data GPU validation after CPU preflight.
7. Assemble against EgoVerse, resolve conflicts with behavior evidence, verify
   parity and checkpoint policy, publish a reviewable PR stack. Cutover remains
   gated by that evidence; do not silently retire unsupported capabilities.

## Decisions and evidence

- The live remote heads match the dated audit. A read-only `git merge-tree`
  reports conflicts; green status in either separate tree is insufficient.
- Existing graph experiments remain supported while declarations are migrated.
  Missing declarations fail closed, with reasons, rather than model-family
  inference in shared Python.
- No capability is deprecated merely because its graph implementation is
  missing. Research-path disposition must be supported by source evidence or
  an explicit user decision.

## Current status

The first contracts layer implements model-owned declarations and stable typed
controls, explicit evaluator requirements, data-owned normalization and strict
standalone-eval restoration, immutable-source OpenPI installation, and package
resource coverage. It includes complete-episode preflight and bounded video
spooling/gathering. No destination merge, legacy removal, GPU launch, or final
parity completion has occurred.

The working integration tree has passed 1,347 CPU tests. A follow-up run passed
44 video/data-context/initialization tests, including an actual h264 round trip,
nonzero-rank rendering, and full-episode truncation rejection. The recursive
audit resolves 235/235 YAMLs; lint/format and lock checks pass. These receipts
refer to the working integration tree, not an assembled EgoVerse destination or
a real-weight GPU run. Each published layer is checked separately before push.

Retained HPT/PI recipe ports, keypoint paths, initialization scheduling and CI
work are being split into subsequent review layers. OT/DTW, latent diagnostics,
data tools, real-weight OSMO gates, fixed-fixture comparisons and destination
assembly remain open. The parity manifest stays blocked until evidence exists
for every retained capability.

The HPT layer now includes every retained root HPT recipe plus EgoBridge's
configured cross-source OT/DTW graph. CPU comparisons against pinned original
methods check loss and gradients, the BC-only detach boundary, independent
feature passes and checkpointed warm starts. Pooled/per-token language tests
run two optimizer steps and encode actual annotation-overlay videos with a
substituted text backend. Real-weight GPU and assembled-destination gates remain
open. Draft PR #150 contains the contracts base; subsequent layers build on it.

The isolated first contracts layer passed **1,350 tests** on tree
`7182a483fbf8d988c4496f316319e48999882d03`; see
`evidence/contracts-cpu-receipt.json`. Subsequent changes in that layer only add
this receipt and navigation/status documentation. The test run includes wheel
installation, typed controls, per-source PI width/time-grid decoding, full
episode checks, and video encoding. It does not depend on the uncommitted
retained-recipe ports or CI formatting layer.
