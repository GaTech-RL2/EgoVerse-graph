# LIBERO completion campaign, September 25, 2026

Operational snapshot: 2026-09-25T06:55:49.462799+00:00. Training source `6590ab8a01dd2347b3659356d03c821fee318a76`; new evaluation migrations and Goal recovery use `cff54402b3a424075074b536443c7ac6c28c5c1c`. [Current numeric tables](arc_vs_oat_libero_20260925.md).

**Every missing result has a submitted job, but the complete table will miss the September 25, 09:00 Pacific target at the measured rates.** The two LIBERO-10 controls need roughly 14–17 additional hours of training alone. Evaluation and future queueing add time. All jobs retain 5001 epochs, global batch 1024 and 2500 evaluation episodes.

Five plain-DP runs have advancing full-training counters and finite losses. Both Spatial jobs remain queued. The OAT-DP Goal run passed GPU preflight and reached approximately 2100 updates before rank 5 reported a CUDA unspecified launch failure through the NCCL watchdog; the underlying cause is unconfirmed. Its durable checkpoint was downloaded, SHA-256 checked and verified at 1680 optimizer/EMA updates. A new eight-L40S job resumes from that checkpoint and is queued. The original failed artifacts are preserved.

All eleven original ARC paired reevaluations now have L40S allocations and actual rollout records. Only unstarted, owned jobs were canceled during the move from saturated pool01 to available single-GPU capacity in pool03. Checkpoints and evaluation coverage are unchanged. The ARC/Transformer recovery campaign has one active evaluation-only job and 17 queued training jobs.

## Plain DP controls

Each training task requests eight L40S GPUs with 128 examples per rank. Its native train→evaluate dependency releases those GPUs after full training, then requests one L40S with five independent repetition workers. All six original allocated runs passed the separate real-GPU preflight; those two preflight updates never enter full training.

| Suite | Backbone | Current workflow | Snapshot | Full optimizer updates |
| --- | --- | --- | --- | ---: |
| libero_10 | unet | [dp-plain-20260925-unet-10-1](https://us-west-2-aws.osmo.nvidia.com/workflows/dp-plain-20260925-unet-10-1) | RUNNING | 9230 / 605121 |
| libero_10 | oat_dp | [dp-plain-20260925-oat-10-1](https://us-west-2-aws.osmo.nvidia.com/workflows/dp-plain-20260925-oat-10-1) | RUNNING | 9500 / 605121 |
| libero_object | unet | [dp-plain-20260925-unet-object-1](https://us-west-2-aws.osmo.nvidia.com/workflows/dp-plain-20260925-unet-object-1) | RUNNING | 8030 / 325065 |
| libero_object | oat_dp | [dp-plain-20260925-oat-object-1](https://us-west-2-aws.osmo.nvidia.com/workflows/dp-plain-20260925-oat-object-1) | RUNNING | 15930 / 325065 |
| libero_goal | unet | [dp-plain-20260925-unet-goal-1](https://us-west-2-aws.osmo.nvidia.com/workflows/dp-plain-20260925-unet-goal-1) | RUNNING | 9450 / 280056 |
| libero_goal | oat_dp | [dp-plain-r1-20260925-oat-goal-1](https://us-west-2-aws.osmo.nvidia.com/workflows/dp-plain-r1-20260925-oat-goal-1) | PENDING recovery | 1680 in saved checkpoint / 280056 |
| libero_spatial | unet | [dp-plain-20260925-unet-spatial-1](https://us-west-2-aws.osmo.nvidia.com/workflows/dp-plain-20260925-unet-spatial-1) | PENDING | Not started |
| libero_spatial | oat_dp | [dp-plain-20260925-oat-spatial-1](https://us-west-2-aws.osmo.nvidia.com/workflows/dp-plain-20260925-oat-spatial-1) | PENDING | Not started |

Observed throughput estimates use 5–20 minute windows of uploaded optimizer counters. These are training-only projections, not score delivery promises.

| Running baseline | Updates/s | Training hours remaining | Projected training finish, Pacific |
| --- | ---: | ---: | --- |
| unet-10 | 9.70 | 17.1 | Sep 25 16:54 |
| oat-10 | 12.25 | 13.5 | Sep 25 13:20 |
| unet-object | 9.11 | 9.7 | Sep 25 09:30 |
| oat-object | 12.07 | 7.1 | Sep 25 06:57 |
| unet-goal | 9.17 | 8.2 | Sep 25 08:02 |

## Missing ARC/Transformer results

Fifteen jobs resume independently verified optimizer/EMA checkpoints. Two shared LIBERO-10 variants start fresh because no checkpoint existed. STK2/Object has a complete 325065-update checkpoint and only needs evaluation; its old partial rollout score is excluded. Training jobs request eight L40S GPUs and retain that allocation through the existing inline evaluation.

| Suite | Selection | Recovery | Workflow | Snapshot |
| --- | --- | --- | --- | --- |
| libero_object | stk_2 / stk | evaluation | [arc-dpr-20260925-stk2-object-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dpr-20260925-stk2-object-1) | RUNNING |
| libero_10 | dur_2 / dur | resume_training | [arc-dpr-20260925-dur2-10-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dpr-20260925-dur2-10-1) | PENDING |
| libero_10 | shared / stk | fresh_training | [arc-dpr-20260925-shared-stk-10-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dpr-20260925-shared-stk-10-1) | PENDING |
| libero_10 | shared / dur | fresh_training | [arc-dpr-20260925-shared-dur-10-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dpr-20260925-shared-dur-10-1) | PENDING |
| libero_spatial | stk_1 / stk | resume_training | [arc-dpr-20260925-stk1-spatial-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dpr-20260925-stk1-spatial-1) | PENDING |
| libero_goal | dur_2 / dur | resume_training | [arc-dpr-20260925-dur2-goal-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dpr-20260925-dur2-goal-1) | PENDING |
| libero_goal | dur_1 / dur | resume_training | [arc-dpr-20260925-dur1-goal-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dpr-20260925-dur1-goal-1) | PENDING |
| libero_spatial | stk_2 / stk | resume_training | [arc-dpr-20260925-stk2-spatial-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dpr-20260925-stk2-spatial-1) | PENDING |
| libero_spatial | shared / stk | resume_training | [arc-dpr-20260925-shared-stk-spatial-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dpr-20260925-shared-stk-spatial-1) | PENDING |
| libero_object | dur_1 / dur | resume_training | [arc-dpr-20260925-dur1-object-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dpr-20260925-dur1-object-1) | PENDING |
| libero_object | dur_2 / dur | resume_training | [arc-dpr-20260925-dur2-object-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dpr-20260925-dur2-object-1) | PENDING |
| libero_goal | shared / dur | resume_training | [arc-dpr-20260925-shared-dur-goal-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dpr-20260925-shared-dur-goal-1) | PENDING |
| libero_spatial | dur_2 / dur | resume_training | [arc-dpr-20260925-dur2-spatial-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dpr-20260925-dur2-spatial-1) | PENDING |
| libero_spatial | shared / dur | resume_training | [arc-dpr-20260925-shared-dur-spatial-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dpr-20260925-shared-dur-spatial-1) | PENDING |
| libero_object | shared / dur | resume_training | [arc-dpr-20260925-shared-dur-object-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dpr-20260925-shared-dur-object-1) | PENDING |
| libero_goal | shared / stk | resume_training | [arc-dpr-20260925-shared-stk-goal-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dpr-20260925-shared-stk-goal-1) | PENDING |
| libero_object | stk_1 / stk | resume_training | [arc-dpr-20260925-stk1-object-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dpr-20260925-stk1-object-1) | PENDING |
| libero_object | shared / stk | resume_training | [arc-dpr-20260925-shared-stk-object-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dpr-20260925-shared-stk-object-1) | PENDING |

## Paired reevaluations and result publication

All eleven `arc-paired-s03-20260925-*` workflows are RUNNING in groot-l40s-03. The first live audit verified 14–26 episode records per job, plus 165 episodes for the separate ARC/Transformer STK2/Object evaluation. These are progress counts, not final success rates.

The CPU-only [libero-table-live-20260925-1](https://us-west-2-aws.osmo.nvidia.com/workflows/libero-table-live-20260925-1) polls every five minutes and publishes under `s3://rldb/experiments/arc-oat-20260919/campaigns/libero-table-live-20260925/`. Its verified first snapshot has 34 scored cells out of 60: 23 preferred evaluations complete and eleven older scores visibly marked as awaiting paired reevaluation. It follows all eleven new paired run IDs and the recovered Goal baseline. The superseded CPU collector was stopped only after this report was verified.

The collector validates full task/seed coverage, checkpoint identity and actual full-budget training counters before publishing new scores. Paired comparisons additionally require matching starting-state hashes. Partial results remain Pending. It writes `latest.md` and `latest.json` to R2 and workflow outputs; it does not push GitHub. The checked-in numeric table is a dated snapshot.

All actual jobs remain on L40S at NORMAL priority. A concrete L40 fallback was prepared and passed dry-run checks, but no L40 jobs were submitted: that distinct device family still requires the user’s exception to the earlier L40S-only request. Access denial from another L40S pool was respected. Other agents’ jobs and worktrees remain untouched.

OAT Object 0.36% remains an unresolved anomaly. Results use one trained checkpoint and five evaluation repetitions, not five training seeds. LIBERO-90 and ARC+OAT hybrid remain deferred.

Validation: 115 focused CPU tests passed for the baseline implementation and regression checks; all six allocated raw-DP GPU preflights passed. The later distinct L40/L40S validation change passed 80 cluster tests. [Machine-readable launch, progress and recovery evidence](libero_completion_launch_20260925.json).
