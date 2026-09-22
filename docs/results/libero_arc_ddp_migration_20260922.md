# LIBERO two-GPU migration

Scheduler snapshot: 2026-09-22T18:18:15.334399+00:00. Training evidence: 2026-09-22T18:17:52.347704+00:00.

LIBERO-90 is deferred at the user's request. All six ARC jobs and its original OAT pipeline are stopped; their verified saved checkpoints and replay evidence remain available. This four-suite subset is not the complete 130-task benchmark.

All **28 replacement workflows are submitted**, requesting **56 L40Ss**: 24 ARC policies plus four original OAT pipelines, two GPUs each. Scheduler state: {'RUNNING': 26, 'PENDING': 2}. 6 workers have advanced optimizer/EMA counters after resume; 2 predecessors are verified stopped after a replacement uploaded a new verified checkpoint.

The checkpoint handoffs are still in progress. A bounded controller checks the submitted replacements every 45 seconds and retires only predecessors with verified advancing replacement checkpoints. Unverified predecessors remain running; scheduler RUNNING includes data/model initialization.

The five R/D/M triples remain unchanged. Training retains 5,001 epochs and global batch 1,024: microbatch 256 per GPU with accumulation 2. Each worker requests 24 CPU cores and 128 GiB memory. ARC resumes revalidate and reuse the completed mode-specific replay checks. Frozen tokenizers are reused when complete; partial optimizer and EMA state are restored.

ARC now caches exact native action-window encodings, and data workers reuse Zarr array handles. These changes preserve targets and codec settings. Changing world size can change floating-point reductions and sample/RNG ordering.

The first ARC pilots stopped before training because the resume check expected a top-level benchmark section in checkpoints. The fix reads the actual saved graph and validates matching encoder/decoder settings and the control protocol. Their original workers were preserved. The OAT Spatial pilot is retained on 8a0b0b6e; the other workers pin 18f50419.

Validation: 144 focused tests across targeted invocations, including a real one-process-to-two-process CPU resume, optimizer/EMA continuity, exact cache parity and three regressions using the actual trainHydra checkpoint configuration. All 70 cluster checks passed again after the compatibility fix. Full learned-policy scores are still pending.

| Run | Suite | GPUs | Scheduler | Restored update | Latest EMA update | Predecessor stopped |
| --- | --- | ---: | --- | ---: | ---: | --- |
| [stk_1 / stk](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-stk1-libero-spatial-1) | libero_spatial | 2 | RUNNING | 14580 | 14950.0 | False |
| [stk_1 / stk](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-stk1-libero-object-1) | libero_object | 2 | RUNNING | pending | pending | False |
| [stk_1 / stk](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-stk1-libero-goal-1) | libero_goal | 2 | RUNNING | pending | pending | False |
| [stk_1 / stk](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-stk1-libero-10-1) | libero_10 | 2 | RUNNING | pending | pending | False |
| [stk_2 / stk](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-stk2-libero-spatial-1) | libero_spatial | 2 | RUNNING | pending | pending | False |
| [stk_2 / stk](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-stk2-libero-object-1) | libero_object | 2 | RUNNING | pending | pending | False |
| [stk_2 / stk](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-stk2-libero-goal-1) | libero_goal | 2 | RUNNING | pending | pending | False |
| [stk_2 / stk](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-stk2-libero-10-1) | libero_10 | 2 | RUNNING | pending | pending | False |
| [dur_1 / dur](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-dur1-libero-spatial-1) | libero_spatial | 2 | RUNNING | pending | pending | False |
| [dur_1 / dur](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-dur1-libero-object-1) | libero_object | 2 | RUNNING | pending | pending | False |
| [dur_1 / dur](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-dur1-libero-goal-1) | libero_goal | 2 | RUNNING | pending | pending | False |
| [dur_1 / dur](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-dur1-libero-10-1) | libero_10 | 2 | RUNNING | 10890 | 11050.0 | False |
| [dur_2 / dur](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-dur2-libero-spatial-1) | libero_spatial | 2 | RUNNING | pending | pending | False |
| [dur_2 / dur](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-dur2-libero-object-1) | libero_object | 2 | RUNNING | pending | pending | False |
| [dur_2 / dur](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-dur2-libero-goal-1) | libero_goal | 2 | RUNNING | pending | pending | False |
| [dur_2 / dur](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-dur2-libero-10-1) | libero_10 | 2 | RUNNING | pending | pending | False |
| [shared / stk](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-shared-stk-libero-spatial-1) | libero_spatial | 2 | RUNNING | pending | pending | False |
| [shared / stk](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-shared-stk-libero-object-1) | libero_object | 2 | RUNNING | pending | pending | False |
| [shared / stk](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-shared-stk-libero-goal-1) | libero_goal | 2 | RUNNING | pending | pending | False |
| [shared / stk](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-shared-stk-libero-10-1) | libero_10 | 2 | RUNNING | pending | pending | False |
| [shared / dur](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-shared-dur-libero-spatial-1) | libero_spatial | 2 | RUNNING | pending | pending | False |
| [shared / dur](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-shared-dur-libero-object-1) | libero_object | 2 | RUNNING | pending | pending | False |
| [shared / dur](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-shared-dur-libero-goal-1) | libero_goal | 2 | PENDING | pending | pending | False |
| [shared / dur](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-shared-dur-libero-10-1) | libero_10 | 2 | PENDING | pending | pending | False |
| [original OAT](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-oat-ddp-20260922-libero-spatial-1) | libero_spatial | 2 | RUNNING | 12420 | 13440.0 | True |
| [original OAT](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-oat-ddp2-20260922-libero-object-1) | libero_object | 2 | RUNNING | 7800 | 8060.0 | False |
| [original OAT](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-oat-ddp2-20260922-libero-goal-1) | libero_goal | 2 | RUNNING | 12320 | 12630.0 | False |
| [original OAT](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-oat-ddp2-20260922-libero-10-1) | libero_10 | 2 | RUNNING | 401720 | 409790.0 | True |

## Observed throughput

Short simultaneous samples from uploaded optimizer/EMA counters, including warmup; different nodes and shared load, not a controlled scaling study or completion-time guarantee.

| Run | Interval (min) | Old updates/s | New updates/s | Ratio |
| --- | ---: | ---: | ---: | ---: |
| arc-five-ddp2-20260922-stk1-libero-spatial | 8.6 | 0.216 | 0.661 | 3.06× |
| arc-five-ddp2-20260922-dur1-libero-10 | 5.1 | 0.166 | 0.429 | 2.58× |
| arc-oat-ddp-20260922-libero-spatial | 7.6 | 0.221 | 0.641 | 2.90× |
| arc-oat-ddp2-20260922-libero-object | 6.0 | 0.210 | 0.661 | 3.15× |
| arc-oat-ddp2-20260922-libero-goal | 7.7 | 0.205 | 0.661 | 3.22× |

Checkpoint receipts, source revisions, parent mapping and measurement endpoints are preserved in the [machine-readable migration report](libero_arc_ddp_migration_20260922.json).
