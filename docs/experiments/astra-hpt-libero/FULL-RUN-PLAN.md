# Full learner / adaptive-versus-uniform run

Authorized 2026-09-28 after corrected commissioning. Preserve the completed
commissioning, original specifications, all failed attempts and existing PRs.
Use the native Codex Astra generator transport already authorized by the user;
do not retry the exhausted gateway. One OSMO L40/L40S, one simulator worker.

1. Freeze independently authored control, development and sealed definitions,
   language realizations, scene-family firewall and exact simulator states
   before the first production update. Reuse the corrected commissioning
   normalizer and audited random HPT initialization.
2. Implement complete atomic checkpoints, optimizer/scheduler/RNG/replay resume,
   simulator-policy transport, measured control reports and bounded acquisition.
3. Collect 120 S1 demonstrations from the existing Astra-authored commissioned
   S1 templates using new episode/reset seeds. Commissioning trajectories stay
   excluded. Train 1,000 common updates and run 36 control probes.
4. Fork the same complete warm-start snapshot. For each arm, run four rounds
   of exactly 75 accepted demonstrations / 500 updates. Uniform gets 25 per
   stage with no learner feedback; fresh adaptive Astra calls use only frozen
   control reports. Preserve immutable decisions, at least two novel templates
   per round, all incurred attempts and actual per-phase compute.
5. Evaluate the shared warm-start and both final models on identical frozen
   development and sealed banks. Report all 708 scheduled policy rollouts,
   paired raw counts, 16-template clustered uncertainty, videos, losses,
   allocation history and cost. Do not tune after opening sealed outcomes.
6. Complete the specified training-only gradient diagnostic after policy
   evaluation; its 240 updates are discarded and never replace final models.

Checkpoint the coordinator and archive every phase before proceeding. A failed
acquisition quota or numerical check is an explicit incomplete run, never a
silent curriculum substitution. The four earlier Astra calls and 62 earlier
commissioning attempts remain separately accounted. No full-run results exist
at creation of this plan.

## Implemented launch path

The `campaign` coordinator now freezes the 36 control / 32 development / 96
sealed cases before training, acquires bounded demonstrations with the existing
single-writer ledger, trains the exact random HPT recipe, and restores the
complete warm-start state for each arm. Checkpoints include optimizer,
scheduler, RNG, replay state, data/normalizer provenance and a complete SQLite
ledger snapshot. A committed checkpoint can recover its loss log and ledger
completion after an interruption without repeating its updates. Unpublished
compute blocks remain charged as discarded work with a recorded upper bound.

The simulator and learner communicate through a bounded local socket containing
only raw images, proprioception and language plus transport metadata. Inference
executes the first of ten predicted native commands. Counterfactual pairs share
the exact full simulator/controller snapshot and policy-noise stream. Uniform
generation requests contain only its own generation archive, common approved
examples and validity feedback. Adaptive requests include a frozen measured
control report and six deterministically selected control frame sequences.

Each native Astra call has fresh context and requests `gpt-6-astra`; reasoning
effort inherits the current session rather than setting an override. The
requested effort is recorded as unavailable/inherited. This differs from the
earlier commissioning calls' explicitly recorded low-effort requests and is
the same invocation policy for both full-run arms. Provider snapshot, billing
and hard token-cap enforcement remain unavailable as previously documented.

Development wording reverses the two canonical clauses. Sealed wording uses a
semicolon and lowercase second clause. These are narrow surface-realization
holdouts within the bounded grammar. BDDL treats semicolons as comments, so its
compiler receives the canonical semantic wording; the student and independent
evaluator still receive the actual held-out TaskSpec text. No geometry, asset,
teacher-control or model settings changed in that compiler fix. The failed
local compiler fixture remains retained with the successful replacement.

Local checks passed: 36 combined CPU tests before the final checkpoint-recovery
fixture; then 25 focused checks including that additional fixture and the
updated subagent contract. Two held-out engineering scenes, one S2 and one S3,
compiled, rendered and restored with unchanged observations; all four paired
TaskSpecs resolved valid complementary referents. Repository Ruff and shell
syntax checks passed. Full-run compute and learner outcomes are still pending
at this source revision.

Launch with the activated project environment:

```bash
python -m scripts.astra_push.build_workflow --source-commit FULL_SHA \
  --name astra-hpt-full-UNIQUE_ID --skip-provider --full-experiment \
  --output /tmp/astra-full.yaml
osmo workflow submit /tmp/astra-full.yaml --pool groot-l40-01
```

Before the first production update, the job also requires an immutable
`full/source-validation.json` containing the launched source commit, successful
GitHub CI conclusion and run URL. Upload that receipt with the artifact bridge
after CI finishes; bank freezing and demonstration collection can run meanwhile.

The job emits immutable `full/generation/PHASE/request.json` files and waits for
native Codex Astra exchanges. The root session uses `SubagentExchange` to
reserve calls before spawning a fresh Astra child, validates the exact response
and hidden split firewall, then uploads the complete exchange and a separate
atomic READY marker through `scripts.astra_push.bridge`. No process pretends to
call a Codex subagent as an HTTP API. All invalid/repair calls must be uploaded
with their original archives and remain counted. The job independently checks
the eight-call phase cap and 80-call total including the four earlier calls.

The final report includes joint pair counts, the paired 16-template cluster
bootstrap, stage/control curves, training losses, allocations and selected
paired videos in self-contained HTML/ZIP. The twelve discarded diagnostic
branches restore the same complete final snapshot and measure fixed training-
only transfer windows after policy evaluation. No adaptation follows sealed
test outcomes.
