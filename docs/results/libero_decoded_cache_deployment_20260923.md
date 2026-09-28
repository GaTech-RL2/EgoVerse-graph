# Decoded LIBERO replay cache: implementation verified, rollout awaiting capacity

Later update: see the [current rollout status](libero_cache_rollout_status_20260923.md)
for verified checkpoint handoffs, measured training rates, the expanded GPU
allocation and deadline estimates. The state below records the earlier
quota-blocked attempts.

Implementation source: `d2c89b9314aa59e5c59bd41b29c2a99b3b0b46b9`.

The loader now decodes replay arrays once into immutable NumPy files and shares
their read-only memory mappings across ranks and data workers. Images retain
uint8 storage. Episode selection, observation/action alignment, padding,
normalization, R/D/M settings, model weights, optimizer/EMA recovery, and the
5001-epoch/global-1024 training budget are preserved. Source changes invalidate
the cache; concurrent preparation is locked and published atomically.

This addresses the repeated Zarr indexing/decompression measured in the
[loader profile](libero_loader_profile_20260922.md). No end-to-end training
speedup or revised completion estimate is established yet.

## Validation

- All 103 focused local checks passed across decoded replay, performance/resume,
  benchmark and cluster tests. Coverage includes exact frame/batch/normalization
  parity, episode boundaries, concurrent preparation, spawned workers, edited
  sources, truncated caches and optimizer/EMA continuity. Ruff and diff checks
  passed.
- In an isolated checkout inside the Linux cluster environment, 102 checks
  passed on the first invocation. The existing tiny-file checkpoint-upload test
  failed once on a rapid same-size rewrite and passed when rerun alone; this
  intermittent failure was not changed or represented as a clean initial pass.
  All new cache and resume checks passed.
- Real Spatial and LIBERO-10 corpora each passed SHA256 verification of all
  seven decoded arrays. For each suite, 1128 normalized samples—including the
  first and last frame of all 500 episodes plus 128 random windows—and 35
  batches of 32 matched the original Zarr loader exactly. These were CPU-only
  checks in isolated new-source checkouts; existing training retained its old
  loader.

The accompanying JSON retains parity results, checkpoint lineage and the
scheduler snapshot. Neither the fixture tests nor CPU data checks are new
learned-policy benchmark scores.

## Deployment state

As of September 23 at approximately 00:26 UTC:

- The three cache pilots did not reach verified optimizer updates. Two began
  initialization in `groot-l40s-03`; the third was queued. The cluster's
  `groot-idle-job-shutdown` service canceled all three at 00:07 UTC to reclaim
  lower-priority shared GPU capacity for a higher-priority workload. This was
  an operator quota action, not a model or cache failure.
- Twenty-six original two-L40S jobs remain running. Two original ARC jobs
  (STK1 Spatial and DUR1 LIBERO-10) were paused during rolling replacement after
  their durable checkpoints passed fresh SHA256/size verification. Those
  checkpoints remain available.
- Fresh checkpoint resumes for these two ARC jobs are queued at NORMAL
  priority in the original `groot-l40s-01` pool, with two GPUs, global batch
  1024, 5001 epochs and the cache enabled. Their run IDs are
  `arc-cache-resume-20260923-stk1-spatial` and
  `arc-cache-resume-20260923-dur1-10`. They reference the still-active original
  OAT pipelines for their respective suites.
- The other 25 planned replacements have not been submitted. Broad rollout
  remains gated on actual resumed training, new verified checkpoints and
  measured full-step throughput. All seven deferred LIBERO-90 jobs remain
  stopped. No other agents' worktrees or workflows were modified.

The first pool had no guaranteed quota available. The second pool had only
single-GPU gaps, so its attempted pilots used one GPU and accumulation four
while preserving global batch and update count. These attempts were canceled
by quota enforcement before training verification. The new queued resumes
return to the planned two-GPU layout. Available L40 hardware is an alternative
pending the user's hardware preference; no L40 job was submitted.

There is no active automatic deployment controller. Saved evidence and helpers
are under `scratch/libero-cache-20260922/`. Do not rerun old pilot specifications:
some prefixes contain initialization artifacts, and canceled workflow versions
are retained as history. Read actual names/resources from submission receipts.
Future rollout must first refresh scheduler state and verify resumed counters
and checkpoints rather than treating submission or allocation as completion.
