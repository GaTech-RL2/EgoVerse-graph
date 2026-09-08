# Pipeline and cluster operations

Use `GaTech-RL2/EgoVerse-graph` for model, graph, configuration, and tooling
source. The generic runner is `egomimic/pipeline/core.py`; its training adapter
is `egomimic/pipeline/algo.py`. Dataset and model semantics belong in stages,
not in the runner.

## Resolve the source before the task

The Pipeline/standard-DP, Planar/Paper-DP, and UNITE base stacks are merged into
`main`. The 2026-09-07 audit verified all three heads below are ancestors of
remote `main` at `75525eb04e0b2965843e21b3916f98ab27597e94`:

| Stack | Reviewed head |
| --- | --- |
| Pipeline + standard DP | `ae37d0760ddc16282faf5c74e8b60c08e2a9a4d4` |
| Planar V2 Paper DP | `6a1e780fbed7f7f3c259712dc0e37ba9057bc798` |
| UNITE register sweep | `5af08e91530208d7d70203037e3c9cf201317148` |

Refresh remote state and verify ancestry before choosing a new source. Older
documents describing these PRs as unmerged are historical. Existing checkpoint
source commits remain authoritative for strict reload and resume.

The real-data Action Flow implementation is a separate review stack:
`codex/usocket-action-flow-bc-20260905` is the parent of
`codex/action-flow-recon10-warmup10k-20260907`
([PR #26](https://github.com/GaTech-RL2/EgoVerse-graph/pull/26)). Its real-data
launcher is `scripts/train/launch_action_flow_usocket.sbatch` on those branches.
It is absent from the audited remote `main` snapshot, but present in this
Action Flow operational-cleanup child branch. The child preserves the warmup
branch as its parent and carries shared runtime fixes without replacing its
model code. Do not treat the synthetic Action Flow scripts on `main` as the
real-data implementation or reset this stack to `main`.

The `codex/torus-winners-usocket-system-test-20260907` child adds three explicit
conditional candidate configs to that same launcher. Read
[their objective and sampler contracts](action-flow-usocket-candidates.md)
before selecting one. The likelihood arm is not ordinary latent FM; the exact
graph-section arm is a restricted diagnostic, not an unrestricted solution.

## Cluster entry points

| Host | Execution authority | Current-task discovery |
| --- | --- | --- |
| Skynet (`sky1`, documented `sky2` fallback) | Live family launchers in `/coc/flash7/paphiwetsa3/scripts/train/`; evaluation protocol in `scripts/eval/` | Recorded run provenance and exact worktree; `EgoVerse-graph-PIPELINE-INDEX.md` in the user project root |
| ICE (`Ice`) | Hash-verified task copy of the selected launcher, explicit source and runtime | `pace-whoami` scratch path and `EgoVerse-graph-PIPELINE-INDEX.md` |
| PACE Phoenix (`pacerh9`) | Phoenix-specific scheduler bindings with the same verified task source contract | `pace-whoami` scratch path and `EgoVerse-graph-PIPELINE-INDEX.md` |

The index is a dated inventory, not live job status or launch authorization.
Refresh the referenced job, source, and manifest before acting. Do not discover
tasks by recursively scanning project or scratch roots. ICE and Phoenix paths,
accounts, and quotas are not interchangeable.

For an approved one-GPU configuration that fits H100 and H200, make one
scheduler request accepting `H100|H200`. Existing task scripts may encode older
manual races; those are historical. A hardware-only swap can reuse a passing
smoke when the full scientific identity still matches.

## Maintain one implementation per capability

Task launch files bind source, runtime, output, scheduler parameters, and
artifact hashes. Reuse the maintained launcher for the actual work; do not
copy its implementation into another task wrapper. Stage reusable tools under
a content-addressed service directory, keep a source/hash receipt, and leave
historical source checkouts immutable. Updating service tools does not update
the model source or retroactively validate a training smoke.

For checkpoint mirrors, both the monitor and executable validator must use
the pinned environment. Export `ICE_MIRROR_PYTHON` and
`ICE_MIRROR_PYTHONPATH` for the exact model source. The maintained wrappers put
that environment on `PATH` and that source first on `PYTHONPATH`. Each validator
still owns its recorded checkpoint/config/source contract.

Mirror health requires a recent successful strict validation and remote hash
verification. A scheduler state of RUNNING or a zero transfer-error counter is
insufficient. Treat validator rejection as actionable and inspect its reason.

## W&B and automatic restart

The shared training entrypoint binds runner-owned W&B logging before data or
logger setup. A fresh run requires an explicit stable ID and uses
`resume=never`. An automatic Slurm restart requires that ID to match the
runner-validated checkpoint metadata and uses `resume=must`. An intentional
initial checkpoint at restart count zero keeps its configured new-experiment
semantics. Debug logging does not contact W&B.

Use `scripts/train/check_wandb_health.py ENTITY/PROJECT/RUN_ID` for read-only
status. Pass repeated `--metric` names and `--expect-config KEY=VALUE` values
from the run manifest; optional `--min-optimizer-step` and `--max-age-seconds`
expose stalled progress. The command reads a bounded recent history window,
reports `trainer/global_step` separately from W&B's history counter, and reads
sparse metrics independently. Summary-only values carry no invented step or
freshness. PASS means observed telemetry health, not completed training.

Slurm-restarted validation writes artifacts under a job/restart namespace.
This preserves earlier validation output at the same optimizer step and still
rejects a duplicate write within the same attempt. The checkpoint budget and
evaluation objective are unchanged. The requeue runner records raw child exit
status and signal names and never requeues a cancellation or ordinary child
failure. A failed upstream dependency automatically retires the dependent
request; it does not permit bypassing its smoke gate.

In this Action Flow branch, `egomimic/eval/artifact_paths.py` supplies the same
namespace rule to EnergyScore, UNITE diagnostics, and Action Flow diagnostics.
The Action Flow smoke verifier selects a matching active job/restart namespace
when present and checks its payload identity. A separate validation allocation
or offline verification requires one unambiguous recorded artifact; its own
scheduler ID is not mistaken for the training job. Update readers alongside
writers when changing artifact paths;
legacy typed/native metric schemas and sidecar hashes remain unchanged.

## Clean task files by dependency

Before any removal, record exact paths, file count/bytes, active-job consumers,
source commit or content hash, and the verified replacement or archive. Keep
shared environments, datasets, and normalizers even when stored inside an old
task directory. Keep source used by any validator or checkpoint. Preserve
uncommitted work and published branch ancestry.

Failed preflight outputs, duplicate staging bundles, canceled submit wrappers,
and superseded smoke checkpoint payloads can be listed as candidates. Preserve
their useful logs, manifests, normalization, and smoke evidence. A stopped run
is not necessarily complete; completion requires the run's validated terminal
record. Mark canceled run launch entries retired so another session does not
mistake them for a request to resume.

Run focused checks for changed shared tools locally or on a scheduled CPU
allocation. Do not run a new training smoke for documentation or mirror-only
maintenance. GPU training, evaluation, and full checkpoint loads belong on
scheduled compute nodes.
