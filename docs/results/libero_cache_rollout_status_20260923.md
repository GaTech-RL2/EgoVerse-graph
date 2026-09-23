# Cache rollout: ARC pilots advancing, OAT pilot started

Status checked September 23, 2026, 03:55–04:06 UTC. Training source remains
`d2c89b9314aa59e5c59bd41b29c2a99b3b0b46b9` for cache-enabled runs.

The two previously queued ARC checkpoint resumes acquired L40S allocations.
Both now advance from restored optimizer/EMA state, have caught up with their
paused predecessors, and have uploaded new SHA256/size-verified checkpoints.
Their cache manifests and all seven full-array hashes passed verification;
1,128 normalized samples and 35 batches per suite matched the original Zarr
loader exactly, including every episode's first and last sample.

| ARC pilot | Previous updates/sec | Cache updates/sec | Observed speedup | Sample duration |
| --- | ---: | ---: | ---: | ---: |
| STK1 Spatial | 0.648 | 4.252 | 6.56× | 635 seconds |
| DUR1 LIBERO-10 | 0.388 | 3.582 | 9.23× | 388 seconds |

These are observed optimizer/EMA update rates from uploaded training metrics,
including normal training/validation overhead within the sampled interval.
They are early measurements from two runs, not universal speedups or a complete
campaign completion estimate. Prior rates are the same configurations on two
L40Ss before the decoded cache. Batch size 1,024, the 5,001-epoch schedule,
R/D/M, model, normalization and data splits are unchanged.

All 28 desired benchmark workers are active: 26 retained original jobs plus
the two cache-enabled ARC resumes, using 56 L40Ss. An additional two-L40S OAT
Spatial replacement started at 03:59:55 UTC and reached its policy-training
stage; its optimizer/checkpoint and sustained-throughput checks are pending.
The old OAT Spatial worker remains running during this verification. The other
25 replacements have not yet been submitted. LIBERO-90 remains deferred.

ARC progress, using 03:55 UTC original metrics and 04:06 UTC resume metrics:

| Suite | Fraction of full optimizer-update budget |
| --- | ---: |
| Spatial | 11.60–14.53% |
| Object | 11.58–12.01% |
| Goal | 13.47–14.08% |
| LIBERO-10 | 3.25–4.12% |

All four OAT tokenizers are complete. Their policies are training; the
LIBERO-10 policy had 4,230 updates (0.70%) in the 03:55 snapshot. Full learned
policy rollout scores remain pending. These four suites exclude LIBERO-90.

A bounded rollout controller started at 04:06 UTC with a four-hour limit.
Before broader submission it requires all three pilots to have verified new
checkpoints, exact data parity and at least five minutes of measured throughput
better than their old rate. It retains each predecessor until the replacement
has advanced, caught up and uploaded a verified checkpoint. Submissions use
NORMAL priority in `groot-l40s-01`, respect reported available quota and allow
at most three simultaneous replacements awaiting handoff. It does not resume
LIBERO-90 or touch other workloads. Errors stop the controller while preserving
unverified predecessors; expiry is not deployment completion.

Controller state, immutable workflow/source manifest and individual receipts
are under `scratch/libero-cache-rollout-20260923/`. A `migration-complete.json`
is written only after all 28 successors and stopped predecessors are verified.
It does not exist at this snapshot. The accompanying JSON preserves progress,
parity, checkpoint lineage and timing evidence. The earlier
[quota-blocked report](libero_decoded_cache_deployment_20260923.md) remains a
historical record of the initial deployment attempts.
