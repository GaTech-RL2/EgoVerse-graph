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

Published draft stack in EgoVerse-graph:

| Layer | PR | Head | Isolated CPU tests |
| --- | --- | --- | --- |
| Contracts | #150 | `ee94a8a3` | 1,350 |
| HPT recipe parity | #151 | `64251fce` | 1,369 |
| PI recipe/diagnostic parity | #152 | `daeaf924` | 1,388 |
| Recorded-data tools | #153 | `5cb26b9e` | 1,392 |
| Bound inference | #154 | `9bd2bed5` | 1,404 |
| Recursive configuration/CI gates | #155 | `09accb58` | 1,408 |
| Real-input OSMO harness | #156 | `37cb8db7` | 1,411 |
| Source fixtures and data compatibility | #157 | `69839b12` | 1,430 |

Each layer's tested tree, snapshot and log hash are recorded under `evidence/`.
The data-tool layer includes real HDF5 conversion, immutable output checks,
complete annotated video encoding and data preview without a model or fitting
normalization. The expanded working-tree constructor audit passes 254/254
shipped YAML contexts, including configured optimizers and schedulers.

The active branch is `codex/graph-integration-20260924/09-deployment-contracts`.
Layer 07 passed all 253 YAML/constructor contexts available at its revision,
including ready inference dependency plans. The gate caught a stale EVA wrist keymap and
Scale API access during construction; both are fixed. The layer also finishes
the shared diagnostic-prediction metric entrypoint and canonical Yam keymap,
with behavior tests. Ruff enforcement preserves the byte-pinned Yam files.
See `VALIDATION.md` for the exact scope and substitutions of the CPU gates.
The CI layer passed its isolated suite and is published. The OSMO layer adds
an explicit real-input matrix and a locally exercised five-update, validation,
checkpoint and no-hardware strict-reload harness. Source data is read-only;
public weights/tokenizer files are hash-pinned. GPU execution receipts remain
required; input metadata and CPU fixtures are not substitutes for them.

Layer 08 adds ten fixed-fixture old/new data, normalization and metric cases,
all-epoch language scheduler references, explicit preprocessing requirements
for restored root models, video corrections and optimizer constructor checks.
See `FIXED_FIXTURE_PARITY.md` for provenance and the limits of this evidence.
Its isolated suite passes all 1,430 tests; lock, Ruff and all 254 recursive
configuration contexts pass. The tested tree and log hash are recorded in
`evidence/parity-hardening-cpu-receipt.json`.

Layer 09 extends declared data requirements to all shipped ready models,
including nested Yam/ARC/PI/E1 and EgoBridge. Declarations now carry explicit
native/output frames; PI's model-owned frame setting also selects the deployment
adapter frame. E1 experiments declare their policy domain and bind the input
identity. All 1,452 isolated CPU tests and 254 configuration/constructor contexts
pass; `evidence/deployment-contracts-cpu-receipt.json` records the tested revision.
`MIGRATION.md` ties installation,
retained commands, data tools and checkpoint rules together.

OSMO currently returns HTTP 403 at both the service and login endpoints, even
for unauthenticated requests. No workflow has been submitted. Network access
must recover before the source-pinned real-input matrix can run.

Remaining gates: real-weight/data OSMO L40/L40S
training and distributed smokes,
refreshed destination conflict inventory,
assembled-tree validation and gated cutover. Public PI weight metadata matches
the pinned OpenPI architecture, and read-only R2 audits located real bimanual
EVA/Aria/Mecka/Scale/Yam episodes; neither observation is a real training result.
No destination merge, legacy runtime removal, or integration GPU launch has
occurred. DQC remains stopped.

The checkpoint policy decision is recorded in `CHECKPOINTS.md`: new high-level
loaders reject unbound historical checkpoints rather than silently pairing them
with current declarations or statistics. Original source, environments, branches
and data remain available; no automatic old-checkpoint converter is claimed.
