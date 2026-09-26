# ARC with the released OAT DP configuration: launch evidence

Scheduler checked 2026-09-24T07:23:42.813177+00:00; artifacts checked 2026-09-24T07:23:46.866320+00:00.

All **24 fresh ARC DP training workflows are submitted**: 20 allocated/running and 4 pending at this snapshot. Scheduler allocation includes environment/data setup; 0 runs have uploaded optimizer progress with finite loss. No new policy scores are available yet.

The five fixed R/D/M triples produce six representation variants: STK1, STK2, DUR1, DUR2, shared STK and shared DUR. Each runs on Spatial, Object, Goal and LIBERO-10. LIBERO-90 remains deferred.

These use the released 4-layer, width-256, 4-head diffusion Transformer, 10 DDIM sampling steps, the released optimizer/EMA settings, 5001 epochs and global batch 1024. The required ARC adaptations are 12 channels and M supports. The raw-action released DP has 4,788,231 action parameters and 27,182,479 total; ARC DP totals range from 27,182,996 to 27,186,068. [Architecture and verification](../LIBERO_ARC_OAT_DP.md).

Each Spatial/Object/Goal job requests 4 L40S GPUs (microbatch 256 per rank). Each LIBERO-10 job requests 8 H100s (microbatch 128 per rank). All use accumulation 1 and the decoded replay loader. These are requested allocations; queued GPUs are not counted as active.

The audited demonstration replay selections are reused after checking their suite, parameters and codec hashes. All models start from fresh weights. Full 2500-episode evaluations start automatically after training with five repetition workers. Final ARC/OAT comparison still requires matching initial-state hashes.

Runtime source: `1e2a3b7a906a42fe7e7256b4148e737fc1f40cb5`. The separate U-Net Spatial scores (including 69.24% DUR2) do not belong to these new Transformer runs. Validation: 154 focused tests passed, including upstream numerical parity and STK/DUR training/EMA reload/resume.

A separate small GPU diagnostic passed on NVIDIA L40S with Torch 2.7.1+cu126: real camera encoders plus the released diffusion Transformer completed bf16 forward/backward, finite gradients and an optimizer step, followed by all 10 DDIM sampling steps for M24/M32/M36. This synthetic diagnostic is separate from full-dataset optimizer progress reported above.

| Workflow | R / D / M | GPUs | Scheduler | Phase |
| --- | --- | --- | --- | --- |
| [arc-dp-20260924-stk1-spatial-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dp-20260924-stk1-spatial-1) | 192 / 1.6 / 36 | 4 L40S | RUNNING | TRAINING |
| [arc-dp-20260924-stk1-object-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dp-20260924-stk1-object-1) | 192 / 1.6 / 36 | 4 L40S | RUNNING | STAGING_DATA |
| [arc-dp-20260924-stk1-goal-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dp-20260924-stk1-goal-1) | 192 / 1.6 / 36 | 4 L40S | RUNNING | TRAINING |
| [arc-dp-20260924-stk1-10-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dp-20260924-stk1-10-1) | 192 / 1.6 / 36 | 8 H100 | RUNNING | TRAINING |
| [arc-dp-20260924-stk2-spatial-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dp-20260924-stk2-spatial-1) | 192 / 0.8 / 32 | 4 L40S | RUNNING | PREPARING_REPLAY_CACHE |
| [arc-dp-20260924-stk2-object-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dp-20260924-stk2-object-1) | 192 / 0.8 / 32 | 4 L40S | RUNNING | STAGING_DATA |
| [arc-dp-20260924-stk2-goal-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dp-20260924-stk2-goal-1) | 192 / 0.8 / 32 | 4 L40S | RUNNING | TRAINING |
| [arc-dp-20260924-stk2-10-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dp-20260924-stk2-10-1) | 192 / 0.8 / 32 | 8 H100 | RUNNING | TRAINING |
| [arc-dp-20260924-dur1-spatial-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dp-20260924-dur1-spatial-1) | 192 / 1.6 / 24 | 4 L40S | RUNNING | TRAINING |
| [arc-dp-20260924-dur1-object-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dp-20260924-dur1-object-1) | 192 / 1.6 / 24 | 4 L40S | RUNNING | STAGING_DATA |
| [arc-dp-20260924-dur1-goal-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dp-20260924-dur1-goal-1) | 192 / 1.6 / 24 | 4 L40S | RUNNING | PREPARING_REPLAY_CACHE |
| [arc-dp-20260924-dur1-10-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dp-20260924-dur1-10-1) | 192 / 1.6 / 24 | 8 H100 | PENDING | PENDING |
| [arc-dp-20260924-dur2-spatial-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dp-20260924-dur2-spatial-1) | 384 / 0.8 / 32 | 4 L40S | RUNNING | BOOTSTRAP |
| [arc-dp-20260924-dur2-object-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dp-20260924-dur2-object-1) | 384 / 0.8 / 32 | 4 L40S | RUNNING | STAGING_DATA |
| [arc-dp-20260924-dur2-goal-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dp-20260924-dur2-goal-1) | 384 / 0.8 / 32 | 4 L40S | RUNNING | TRAINING |
| [arc-dp-20260924-dur2-10-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dp-20260924-dur2-10-1) | 384 / 0.8 / 32 | 8 H100 | PENDING | PENDING |
| [arc-dp-20260924-shared-stk-spatial-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dp-20260924-shared-stk-spatial-1) | 384 / 1.6 / 36 | 4 L40S | RUNNING | STAGING_DATA |
| [arc-dp-20260924-shared-stk-object-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dp-20260924-shared-stk-object-1) | 384 / 1.6 / 36 | 4 L40S | RUNNING | BOOTSTRAP |
| [arc-dp-20260924-shared-stk-goal-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dp-20260924-shared-stk-goal-1) | 384 / 1.6 / 36 | 4 L40S | RUNNING | BOOTSTRAP |
| [arc-dp-20260924-shared-stk-10-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dp-20260924-shared-stk-10-1) | 384 / 1.6 / 36 | 8 H100 | PENDING | PENDING |
| [arc-dp-20260924-shared-dur-spatial-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dp-20260924-shared-dur-spatial-1) | 384 / 1.6 / 36 | 4 L40S | RUNNING | STAGING_DATA |
| [arc-dp-20260924-shared-dur-object-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dp-20260924-shared-dur-object-1) | 384 / 1.6 / 36 | 4 L40S | RUNNING | STAGING_DATA |
| [arc-dp-20260924-shared-dur-goal-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dp-20260924-shared-dur-goal-1) | 384 / 1.6 / 36 | 4 L40S | RUNNING | STAGING_DATA |
| [arc-dp-20260924-shared-dur-10-1](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-dp-20260924-shared-dur-10-1) | 384 / 1.6 / 36 | 8 H100 | PENDING | PENDING |
