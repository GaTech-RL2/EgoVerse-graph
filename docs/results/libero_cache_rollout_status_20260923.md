# Decoded replay rollout and training deadline

Verified September 23, 2026 at 10:21 AM Pacific (2026-09-23T17:21:02.971507+00:00).
Loader/model source: `d2c89b9314aa59e5c59bd41b29c2a99b3b0b46b9`. The H100 launch option adds allocation validation and scheduling support; its source pin is recorded separately below.

24/24 ARC jobs have completed checkpoint-preserving migration to the new loader; 24/24 have verified advancing optimizer/EMA counters and new SHA256/size-verified checkpoints. 24/24 replacement workflows are RUNNING. The ARC target allocation is 64 L40S GPUs: 18 two-GPU jobs, five four-GPU LIBERO-10 jobs, and one eight-GPU STK1 LIBERO-10 recovery.

The full 5,001-epoch schedule, global batch 1,024, R/D/M, normalization and data splits remain unchanged. Two GPUs use microbatch 256 with accumulation two; four use 256 with accumulation one; eight use 128 with accumulation one. LIBERO-90 remains deferred with its seven checkpoints retained.

| Suite | Optimizer budget completed | Projected training finish (Pacific) | Runs with measured rates |
| --- | ---: | --- | ---: |
| Spatial | 78.6–85.6% | Sep 23, 12:55 PM–Sep 23, 2:17 PM | 6/6 |
| Object | 68.6–70.0% | Sep 23, 4:42 PM–Sep 23, 5:03 PM | 6/6 |
| Goal | 77.7–80.6% | Sep 23, 1:53 PM–Sep 23, 2:30 PM | 6/6 |
| LIBERO-10 | 10.8–59.8% | Sep 23, 7:00 PM–Sep 24, 6:19 AM | 6/6 |

These are extrapolations from actual optimizer/EMA advances. Where available, they use a recent 10–20 minute interval, including its validation/checkpoint overhead; averages including initial cache warmup are retained separately in the JSON. They exclude learned-policy rollout evaluation and future interruptions. The provisional cutoff is 11:59 p.m. Pacific on September 23; the user has not specified a more precise cutoff.

The STK1 LIBERO-10 four-GPU replacement was canceled by the cluster quota controller before training. Its original kept running. Newly available quota allowed an eight-L40S checkpoint recovery at NORMAL priority in `groot-l40s-03`; the other 23 cached jobs use `groot-l40s-01`.

The eight-GPU recovery has reached 65,600 optimizer updates and measured 7.51 updates/sec over 56.1 minutes. Its projected training finish is September 24, 6:19 AM Pacific. This misses the provisional cutoff; the estimate remains subject to throughput and scheduling changes.

H100 recovery: `arc-h1008b-20260923-stk1-10-1`, source `c609eb366b28a7aa1acb398b81273b4822c62743`, eight H100s, 80 CPU cores, global batch 1,024 and the same full budget. The first 96-core request was rejected because this pool exposes 94 allocatable cores; that attempt created no workflow. The corrected request uses NORMAL priority in `groot-h100-02`. Runtime differences from the cached L40S source are confined to hardware selection, validation and resource requests.

The H100 candidate is additional to the ARC target allocation while verification is pending. Its L40S predecessor continues running. A bounded controller requires caught-up optimizer/EMA counters, a new SHA/size-verified checkpoint and at least five minutes of measured improvement before retiring that predecessor.

The H100 candidate's recent 10.9-minute interval measured 10.88 updates/sec, versus 11.05 needed for the cutoff. This projects Sep 24, 12:11 AM Pacific, excluding future interruptions and rollouts. Its startup-inclusive measurement is preserved separately. This estimate does not establish a completed handoff or guarantee the deadline.

At 10:18 AM Pacific, 4/4 OAT baseline policies have verified advancing optimizer/EMA counters and new SHA/size-verified checkpoints on the decoded loader.

The migration controller stopped overnight after scheduler read failures and then one canceled successor. The 23 replacements kept training; old duplicates remained active until fresh checkpoint verification and retirement around 9 a.m. Pacific. The helper now retries scheduler reads and continues unaffected handoffs while preserving a failed successor’s original. No other agents’ jobs or worktrees were changed.

OAT Spatial continued on the cached loader. Object, Goal and LIBERO-10 OAT policies were checkpointed and paused to prioritize ARC allocations, then cache-enabled resumes were submitted at 9:06 a.m. Pacific after all ARC allocations were running. The primary pool filled again; those three never-started queued workflows were canceled and replaced by fresh NORMAL-priority resumes in `groot-l40s-03` around 9:17 a.m. Pacific. Object and Goal failed their pool-local demonstration-cache check before training. Their corrected resumes download the same pinned official files and verify source hashes, leaving existing caches untouched. Their scheduler/checkpoint states and original-to-resumed run aliases are recorded separately in the accompanying JSON. All four tokenizers were already complete. Full learned-policy scores remain pending; original OAT pipelines also contain later ARC stages, so these policy estimates are not whole-pipeline completion estimates.

Validation before rollout: 103 focused local checks passed. On the cluster, 102 initially passed and one existing rapid-file-rewrite checkpoint test passed on retry. Real Spatial and LIBERO-10 data each passed all seven decoded-array hashes, 1,128 normalized samples spanning every episode boundary, and 35 batches of 32 with bitwise equality to Zarr. Successful cache pilots also showed sustained optimizer-rate improvement before broad deployment.

Canonical deployment evidence is in `scratch/libero-arc-deadline-20260923/`, with the original immutable manifest plus the sealed eight-GPU replacement overlay. Individual retirement receipts retain checkpoint lineage. `migration-complete.json` is written only after all 24 successor checkpoints and predecessor terminal states are verified. The accompanying JSON freezes source pins, layouts, scheduler states, rate endpoints and checkpoint evidence. The [earlier deployment report](libero_decoded_cache_deployment_20260923.md) is historical.
