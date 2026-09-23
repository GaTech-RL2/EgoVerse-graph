# Decoded replay rollout and training deadline

Verified September 23, 2026 at 1:50 PM Pacific (2026-09-23T20:50:18.875297+00:00).
Loader/model source: `d2c89b9314aa59e5c59bd41b29c2a99b3b0b46b9`. The H100 launch option adds allocation validation and scheduling support; its source pin is recorded separately below.

24/24 ARC jobs have completed checkpoint-preserving migration to the new loader; 24/24 have verified optimizer/EMA progress and new SHA256/size-verified checkpoints. 24/24 replacement workflows are RUNNING and 0/24 have completed their workflow. 1/24 have finished training and advanced to rollouts or completed evaluation. The ARC target allocation is 8 H100 GPUs, 56 L40S GPUs: 18 two-GPU jobs, five four-GPU LIBERO-10 jobs, and one eight-GPU STK1 LIBERO-10 recovery.

The full 5,001-epoch schedule, global batch 1,024, R/D/M, normalization and data splits remain unchanged. Two GPUs use microbatch 256 with accumulation two; four use 256 with accumulation one; eight use 128 with accumulation one. LIBERO-90 remains deferred with its seven checkpoints retained.

| Suite | Optimizer budget completed | Remaining training finish (Pacific) | Finished training |
| --- | ---: | --- | ---: |
| Spatial | 97.6–100.0% | Sep 23, 1:57 PM–Sep 23, 2:16 PM | 1/6 |
| Object | 84.8–86.4% | Sep 23, 4:39 PM–Sep 23, 5:05 PM | 0/6 |
| Goal | 96.6–99.8% | Sep 23, 1:52 PM–Sep 23, 2:27 PM | 0/6 |
| LIBERO-10 | 32.8–75.9% | Sep 23, 7:04 PM–Sep 24, 12:16 AM | 0/6 |

These are extrapolations from actual optimizer/EMA advances. Where available, they use a recent 10–20 minute interval, including its validation/checkpoint overhead; averages including initial cache warmup are retained separately in the JSON. They exclude learned-policy rollout evaluation and future interruptions. The provisional cutoff is 11:59 p.m. Pacific on September 23; the user has not specified a more precise cutoff.

The STK1 LIBERO-10 four-GPU replacement was canceled by the cluster quota controller before training. Its original kept running. Newly available quota allowed an eight-L40S checkpoint recovery at NORMAL priority in `groot-l40s-03`; the other 23 cached jobs use `groot-l40s-01`.

The eight-GPU recovery has reached 198,440 optimizer updates and measured 10.83 updates/sec over 14.1 minutes. Its projected training finish is September 24, 12:16 AM Pacific. This misses the provisional cutoff; the estimate remains subject to throughput and scheduling changes.

H100 recovery: `arc-h1008b-20260923-stk1-10-1`, source `c609eb366b28a7aa1acb398b81273b4822c62743`, eight H100s, 80 CPU cores, global batch 1,024 and the same full budget. The first 96-core request was rejected because this pool exposes 94 allocatable cores; that attempt created no workflow. The corrected request uses NORMAL priority in `groot-h100-02`. Runtime differences from the cached L40S source are confined to hardware selection, validation and resource requests.

The H100 handoff is verified: the replacement caught up, uploaded a new checkpoint and measured faster than its L40S predecessor over at least five minutes. The exact L40S predecessor is now stopped.

The H100 recovery's recent 14.1-minute interval measured 10.83 updates/sec, versus 11.12 needed for the cutoff. This projects Sep 24, 12:15 AM Pacific, excluding future interruptions and rollouts. Its startup-inclusive measurement is preserved separately. A finish estimate is not a deadline guarantee.

At 1:55 PM Pacific, 4/4 OAT baseline policies have a running workflow, fresh optimizer/EMA counters and a new SHA/size-verified checkpoint on the decoded loader.

The cluster quota controller canceled the Object, Goal and LIBERO-10 OAT resumes around 11:32 a.m. Pacific. Their final saved metrics are historical checkpoint evidence, not evidence of current training. Fresh checkpoint resumes were submitted on eight H100s each in `groot-h100-02`, preserving the complete budget and tokenizer/policy lineage. Current scheduler states: libero_object: RUNNING, libero_goal: RUNNING, libero_10: RUNNING. Allocation or initialization alone is not counted as verified training progress.

At 1:55 PM Pacific, STK1 Spatial evaluation has recorded 252/2,500 episodes from the verified final EMA checkpoint. All 24 standalone ARC runs invoke the same evaluation immediately after training. Partial episode counts are progress, not a complete suite score.

A bounded evaluation dispatcher is active for the four OAT policies, pinned to `6d0a1aa1826f11a778367ab6f7e3626b20ac7796`. It checks for the complete 5,001 epochs and exact optimizer/EMA budget every 45 seconds plus polling overhead, then submits a separate one-GPU native evaluation from the immutable final checkpoint. This avoids waiting for later ARC stages in the original combined pipelines. Current dispatch count: 0/4; remaining policies are still training. The selected authorized pool must have quota; physical GPU availability can add queue time. Each evaluation retains 50 trials per task, five repetitions, seed 1000 and the 550-step horizon. The dispatcher is bounded to 24 hours and prevents duplicate submissions. Its source and runtime records are retained in the JSON.

The independent evaluation path passed 92 focused CPU checks, OSMO dry-runs on L40S and H100, and validation against the real completed ARC checkpoint. Separate dispatch fixtures verified duplicate and ambiguous-submit handling. No new OAT evaluation is reported as running before its workflow reaches rollouts.

Measured model sizes: ARC adds a 40,345,612-parameter action denoiser to the 22,394,248-parameter observation encoder, for 62,739,860 total (62,739,788 trainable). ARC codec operations add no learned parameters. OAT has a 5,022,976-parameter action predictor plus a 5,812,497-parameter tokenizer frozen during policy training, for 33,229,721 total including the same encoder (27,417,152 trainable). These recipes are not parameter matched.

The migration controller stopped overnight after scheduler read failures and then one canceled successor. The 23 replacements kept training; old duplicates remained active until fresh checkpoint verification and retirement around 9 a.m. Pacific. The helper now retries scheduler reads and continues unaffected handoffs while preserving a failed successor’s original. No other agents’ jobs or worktrees were changed.

OAT Spatial continued on the cached loader. Object, Goal and LIBERO-10 OAT policies were checkpointed and paused to prioritize ARC allocations, then cache-enabled resumes were submitted at 9:06 a.m. Pacific after all ARC allocations were running. The primary pool filled again; those three never-started queued workflows were canceled and replaced by fresh NORMAL-priority resumes in `groot-l40s-03` around 9:17 a.m. Pacific. Object and Goal failed their pool-local demonstration-cache check before training. Their corrected resumes download the same pinned official files and verify source hashes, leaving existing caches untouched. Their scheduler/checkpoint states and original-to-resumed run aliases are recorded separately in the accompanying JSON. All four tokenizers were already complete. Full learned-policy scores remain pending; original OAT pipelines also contain later ARC stages, so these policy estimates are not whole-pipeline completion estimates.

Validation before rollout: 103 focused local checks passed. On the cluster, 102 initially passed and one existing rapid-file-rewrite checkpoint test passed on retry. Real Spatial and LIBERO-10 data each passed all seven decoded-array hashes, 1,128 normalized samples spanning every episode boundary, and 35 batches of 32 with bitwise equality to Zarr. Successful cache pilots also showed sustained optimizer-rate improvement before broad deployment.

Canonical deployment evidence is in `scratch/libero-arc-deadline-20260923/`, with the original immutable manifest plus the sealed eight-GPU replacement overlay. Individual retirement receipts retain checkpoint lineage. `migration-complete.json` is written only after all 24 successor checkpoints and predecessor terminal states are verified. The accompanying JSON freezes source pins, layouts, scheduler states, rate endpoints and checkpoint evidence. The [earlier deployment report](libero_decoded_cache_deployment_20260923.md) is historical.
