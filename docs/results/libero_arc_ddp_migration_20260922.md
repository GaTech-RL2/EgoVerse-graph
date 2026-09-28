# LIBERO two-GPU migration

Scheduler snapshot: 2026-09-22T22:21:08.162441+00:00. Training evidence: 2026-09-22T22:20:16.669394+00:00.

LIBERO-90 is deferred at the user's request. All six ARC jobs and its original OAT pipeline are stopped; their verified saved checkpoints and replay evidence remain available. This four-suite subset is not the complete 130-task benchmark.

All **28 replacement workflows are submitted**, requesting **56 L40Ss**: 24 ARC policies plus four original OAT pipelines, two GPUs each. Scheduler state: {'RUNNING': 28}. 28 workers have advanced optimizer/EMA counters after resume; 28 predecessors are verified stopped after a replacement uploaded a new verified checkpoint.

All 28 handoffs are complete: every replacement advanced and uploaded a new verified checkpoint before its predecessor stopped. Scheduler checks confirm all 28 originals are terminal. The monitor initially stopped after 22 handoffs on a truncated response; compressed, checksummed response chunks recovered the final six without changing training workers.

The five R/D/M triples remain unchanged. Training retains 5,001 epochs and global batch 1,024: microbatch 256 per GPU with accumulation 2. Each worker requests 24 CPU cores and 128 GiB memory. ARC resumes revalidate and reuse the completed mode-specific replay checks. Frozen tokenizers are reused when complete; partial optimizer and EMA state are restored.

ARC now caches exact native action-window encodings, and data workers reuse Zarr array handles. These changes preserve targets and codec settings. Changing world size can change floating-point reductions and sample/RNG ordering.

The first ARC pilots stopped before training because the resume check expected a top-level benchmark section in checkpoints. The fix reads the actual saved graph and validates matching encoder/decoder settings and the control protocol. Their original workers were preserved. The OAT Spatial pilot is retained on 8a0b0b6e; the other workers pin 18f50419.

Validation: 144 focused tests across targeted invocations, including a real one-process-to-two-process CPU resume, optimizer/EMA continuity, exact cache parity and three regressions using the actual trainHydra checkpoint configuration. All 70 cluster checks passed again after the compatibility fix. Full learned-policy scores are still pending.

| Run | Suite | GPUs | Scheduler | Restored update | Latest EMA update | Predecessor stopped |
| --- | --- | ---: | --- | ---: | ---: | --- |
| [stk_1 / stk](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-stk1-libero-spatial-1) | libero_spatial | 2 | RUNNING | 14580 | 24420.0 | True |
| [stk_1 / stk](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-stk1-libero-object-1) | libero_object | 2 | RUNNING | 16250 | 25470.0 | True |
| [stk_1 / stk](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-stk1-libero-goal-1) | libero_goal | 2 | RUNNING | 16240 | 25400.0 | True |
| [stk_1 / stk](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-stk1-libero-10-1) | libero_10 | 2 | RUNNING | 9680 | 15390.0 | True |
| [stk_2 / stk](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-stk2-libero-spatial-1) | libero_spatial | 2 | RUNNING | 15660 | 25060.0 | True |
| [stk_2 / stk](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-stk2-libero-object-1) | libero_object | 2 | RUNNING | 16250 | 25630.0 | True |
| [stk_2 / stk](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-stk2-libero-goal-1) | libero_goal | 2 | RUNNING | 15680 | 25100.0 | True |
| [stk_2 / stk](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-stk2-libero-10-1) | libero_10 | 2 | RUNNING | 10890 | 16270.0 | True |
| [dur_1 / dur](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-dur1-libero-spatial-1) | libero_spatial | 2 | RUNNING | 15660 | 24660.0 | True |
| [dur_1 / dur](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-dur1-libero-object-1) | libero_object | 2 | RUNNING | 15600 | 24640.0 | True |
| [dur_1 / dur](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-dur1-libero-goal-1) | libero_goal | 2 | RUNNING | 16800 | 26000.0 | True |
| [dur_1 / dur](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-dur1-libero-10-1) | libero_10 | 2 | RUNNING | 10890 | 16760.0 | True |
| [dur_2 / dur](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-dur2-libero-spatial-1) | libero_spatial | 2 | RUNNING | 16200 | 25430.0 | True |
| [dur_2 / dur](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-dur2-libero-object-1) | libero_object | 2 | RUNNING | 15600 | 24520.0 | True |
| [dur_2 / dur](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-dur2-libero-goal-1) | libero_goal | 2 | RUNNING | 15680 | 25150.0 | True |
| [dur_2 / dur](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-dur2-libero-10-1) | libero_10 | 2 | RUNNING | 10890 | 16240.0 | True |
| [shared / stk](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-shared-stk-libero-spatial-1) | libero_spatial | 2 | RUNNING | 16740 | 26130.0 | True |
| [shared / stk](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-shared-stk-libero-object-1) | libero_object | 2 | RUNNING | 16250 | 25090.0 | True |
| [shared / stk](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-shared-stk-libero-goal-1) | libero_goal | 2 | RUNNING | 15680 | 24960.0 | True |
| [shared / stk](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-shared-stk-libero-10-1) | libero_10 | 2 | RUNNING | 10890 | 16240.0 | True |
| [shared / dur](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-shared-dur-libero-spatial-1) | libero_spatial | 2 | RUNNING | 16200 | 25120.0 | True |
| [shared / dur](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-shared-dur-libero-object-1) | libero_object | 2 | RUNNING | 15600 | 24560.0 | True |
| [shared / dur](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-shared-dur-libero-goal-1) | libero_goal | 2 | RUNNING | 15680 | 24630.0 | True |
| [shared / dur](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-ddp2-20260922-shared-dur-libero-10-1) | libero_10 | 2 | RUNNING | 10890 | 16510.0 | True |
| [original OAT](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-oat-ddp-20260922-libero-spatial-1) | libero_spatial | 2 | RUNNING | 12420 | 22600.0 | True |
| [original OAT](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-oat-ddp2-20260922-libero-object-1) | libero_object | 2 | RUNNING | 7800 | 17550.0 | True |
| [original OAT](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-oat-ddp2-20260922-libero-goal-1) | libero_goal | 2 | RUNNING | 12320 | 22100.0 | True |
| [original OAT](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-oat-ddp2-20260922-libero-10-1) | libero_10 | 2 | RUNNING | 401720 | 537250.0 | True |

## Current training progress

The following estimates use observed update rates since 18:55 UTC (about 3.4 hours). They cover only the current model's remaining training, exclude evaluation and later models, and assume unchanged throughput. Full learned-policy scores are pending.

| Model | Suite | Budget complete | Remaining training (days) |
| --- | --- | ---: | ---: |
| ARC (six policies) | libero_spatial | 9.0–9.7% | 4.15–4.40 |
| ARC (six policies) | libero_object | 7.5–7.9% | 5.16–5.39 |
| ARC (six policies) | libero_goal | 8.8–9.3% | 4.40–4.53 |
| ARC (six policies) | libero_10 | 2.5–2.8% | 16.30–17.87 |
| OAT policy | libero_spatial | 8.4% | 4.57 |
| OAT policy | libero_object | 5.4% | 5.44 |
| OAT policy | libero_goal | 7.9% | 4.55 |
| OAT tokenizer | libero_10 | 88.8% | 0.09 |

LIBERO-10's OAT tokenizer is 88.8% complete, with about 2.2 hours of tokenizer training left at the measured rate; its policy training follows. The other three OAT tokenizers are already complete.

## Observed throughput

Simultaneous samples from uploaded optimizer/EMA counters before each cutover, including warmup; different nodes and shared load, not a controlled scaling study or completion-time guarantee.

| Run | Interval (min) | Old updates/s | New updates/s | Ratio |
| --- | ---: | ---: | ---: | ---: |
| arc-five-ddp2-20260922-stk1-libero-spatial | 14.9 | 0.211 | 0.651 | 3.08× |
| arc-five-ddp2-20260922-stk1-libero-object | 16.5 | 0.228 | 0.665 | 2.92× |
| arc-five-ddp2-20260922-stk1-libero-goal | 14.6 | 0.227 | 0.661 | 2.91× |
| arc-five-ddp2-20260922-stk1-libero-10 | 231.6 | 0.145 | 0.410 | 2.83× |
| arc-five-ddp2-20260922-stk2-libero-spatial | 13.6 | 0.229 | 0.663 | 2.90× |
| arc-five-ddp2-20260922-stk2-libero-object | 17.6 | 0.227 | 0.639 | 2.82× |
| arc-five-ddp2-20260922-stk2-libero-goal | 15.4 | 0.224 | 0.673 | 3.00× |
| arc-five-ddp2-20260922-stk2-libero-10 | 233.4 | 0.157 | 0.383 | 2.45× |
| arc-five-ddp2-20260922-dur1-libero-spatial | 14.7 | 0.229 | 0.642 | 2.81× |
| arc-five-ddp2-20260922-dur1-libero-object | 16.9 | 0.231 | 0.654 | 2.83× |
| arc-five-ddp2-20260922-dur1-libero-goal | 14.7 | 0.216 | 0.664 | 3.07× |
| arc-five-ddp2-20260922-dur1-libero-10 | 247.6 | 0.161 | 0.393 | 2.44× |
| arc-five-ddp2-20260922-dur2-libero-spatial | 14.5 | 0.260 | 0.699 | 2.69× |
| arc-five-ddp2-20260922-dur2-libero-object | 16.9 | 0.221 | 0.653 | 2.96× |
| arc-five-ddp2-20260922-dur2-libero-goal | 15.4 | 0.243 | 0.649 | 2.68× |
| arc-five-ddp2-20260922-dur2-libero-10 | 226.1 | 0.157 | 0.394 | 2.51× |
| arc-five-ddp2-20260922-shared-stk-libero-spatial | 14.7 | 0.228 | 0.632 | 2.77× |
| arc-five-ddp2-20260922-shared-stk-libero-object | 17.9 | 0.230 | 0.628 | 2.74× |
| arc-five-ddp2-20260922-shared-stk-libero-goal | 13.7 | 0.236 | 0.710 | 3.00× |
| arc-five-ddp2-20260922-shared-stk-libero-10 | 227.0 | 0.168 | 0.392 | 2.33× |
| arc-five-ddp2-20260922-shared-dur-libero-spatial | 14.8 | 0.239 | 0.648 | 2.71× |
| arc-five-ddp2-20260922-shared-dur-libero-object | 17.9 | 0.230 | 0.645 | 2.81× |
| arc-five-ddp2-20260922-shared-dur-libero-goal | 15.0 | 0.217 | 0.653 | 3.02× |
| arc-five-ddp2-20260922-shared-dur-libero-10 | 223.4 | 0.164 | 0.417 | 2.54× |
| arc-oat-ddp-20260922-libero-spatial | 7.6 | 0.221 | 0.641 | 2.90× |
| arc-oat-ddp2-20260922-libero-object | 17.8 | 0.205 | 0.638 | 3.12× |
| arc-oat-ddp2-20260922-libero-goal | 15.8 | 0.213 | 0.630 | 2.95× |

Checkpoint receipts, source revisions, parent mapping and measurement endpoints are preserved in the [machine-readable migration report](libero_arc_ddp_migration_20260922.json).
