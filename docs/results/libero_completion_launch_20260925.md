# LIBERO completion campaign, September 25, 2026

Operational snapshot: 2026-09-25T06:28:20.992821+00:00. Source: `6590ab8a01dd2347b3659356d03c821fee318a76`. [Current numeric tables](arc_vs_oat_libero_20260925.md).

**All eight plain-DP controls and all 18 missing ARC/Transformer recoveries are submitted.** The requested target is September 25, 09:00 Pacific (16:00 UTC), but it is at risk because of L40S capacity and remaining training. No budget or evaluation coverage was shortened to meet the target.

The latest baseline scheduler snapshot has six RUNNING workflows (48 L40S GPUs allocated) and two PENDING. GPU preflight passed for current-U-Net/LIBERO-10 and OAT-DP/Object, including eight-rank bf16 optimizer steps, EMA reload, and short real simulator episodes on all ten tasks. Their full-training processes have started; this snapshot does not yet contain full-run optimizer counters. Scheduler RUNNING alone is not evidence of advancing training.

The ARC recovery snapshot has one RUNNING evaluation-only workflow and 17 PENDING training workflows. Fifteen restore verified optimizer/EMA checkpoints; two shared LIBERO-10 variants had never produced checkpoints and start fresh. Most partial sources had completed only about 5–7% of training; STK1/Spatial had reached 230580/270054 updates. Source files were independently downloaded, SHA-256 checked, and actual checkpoint counters inspected.

## Plain DP controls

Each training task requests eight L40S GPUs, global batch 1024, and 5001 epochs. Each has a native train→evaluate dependency that releases its training GPUs before one L40S evaluates 2500 episodes. The GPU preflight has separate initialization and outputs; its two updates are not part of full training.

| Suite | Backbone | Workflow | Snapshot status |
| --- | --- | --- | --- |
| libero_10 | unet | [dp-plain-20260925-unet-10-1](https://us-west-2-aws.osmo.nvidia.com/workflows/dp-plain-20260925-unet-10-1) | RUNNING |
| libero_10 | oat_dp | [dp-plain-20260925-oat-10-1](https://us-west-2-aws.osmo.nvidia.com/workflows/dp-plain-20260925-oat-10-1) | RUNNING |
| libero_object | unet | [dp-plain-20260925-unet-object-1](https://us-west-2-aws.osmo.nvidia.com/workflows/dp-plain-20260925-unet-object-1) | RUNNING |
| libero_object | oat_dp | [dp-plain-20260925-oat-object-1](https://us-west-2-aws.osmo.nvidia.com/workflows/dp-plain-20260925-oat-object-1) | RUNNING |
| libero_goal | unet | [dp-plain-20260925-unet-goal-1](https://us-west-2-aws.osmo.nvidia.com/workflows/dp-plain-20260925-unet-goal-1) | RUNNING |
| libero_goal | oat_dp | [dp-plain-20260925-oat-goal-1](https://us-west-2-aws.osmo.nvidia.com/workflows/dp-plain-20260925-oat-goal-1) | RUNNING |
| libero_spatial | unet | [dp-plain-20260925-unet-spatial-1](https://us-west-2-aws.osmo.nvidia.com/workflows/dp-plain-20260925-unet-spatial-1) | PENDING |
| libero_spatial | oat_dp | [dp-plain-20260925-oat-spatial-1](https://us-west-2-aws.osmo.nvidia.com/workflows/dp-plain-20260925-oat-spatial-1) | PENDING |

## Missing ARC/Transformer results

| Suite | Selection | Recovery | Workflow | Snapshot status |
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

## Automatic result publication

The CPU-only [table collector](https://us-west-2-aws.osmo.nvidia.com/workflows/libero-table-20260925-2) is running in groot-l40s-01. Its first verified report contains 34 scored cells out of 60; 23 already use the final requested evaluation source, while 11 original results remain marked as awaiting paired reevaluation. It polls every five minutes, retains partial runs as Pending, validates full rollout protocol and new checkpoint budgets, and compares starting-state identities. It writes `latest.md` and `latest.json` under `s3://rldb/experiments/arc-oat-20260919/campaigns/libero-table-20260925/`, and exports final workflow outputs. The GitHub snapshot is historical; the collector updates R2, not this branch. No laptop watcher is needed.

The first CPU collector submission was rejected because groot-l40s-03 has no CPU-platform resources; no workflow was created there. Reconciliation confirmed this before submission to pool01, whose accepted workflow is suffixed `-2`. An additional L40S pool denied access; that boundary was respected. All GPU jobs use the authorized Groot L40S pools at NORMAL priority.

OAT Object 0.36% remains an unresolved anomaly. All complete scores use one trained checkpoint and five evaluation repetitions. The 11 original Spatial/LIBERO-10 reevaluations remain separately queued under `arc-paired-20260925-*`; old scores stay visibly flagged until replacement results pass pairing. LIBERO-90 and ARC+OAT hybrid remain deferred.

Validation: 115 focused CPU tests passed; both raw-action backbone GPU preflights above passed. [Full operational manifest and evidence](libero_completion_launch_20260925.json).
