# FAST LIBERO launch — September 26, 2026

Artifact/scheduler audit: 2026-09-26T18:54:28.544556+00:00. This is a launch snapshot, not a success-rate result.

The native OAT-codebase FAST baseline is submitted for all four active suites.
Spatial and Object have verified NVIDIA L40S runtimes and are staging data;
Goal and LIBERO-10 are queued. GPU preflight and optimizer progress are not yet verified.

| Suite | Workflow | Phase | Full SR |
| --- | --- | --- | ---: |
| Spatial | [fast-libero-20260926-spatial-2](https://us-west-2-aws.osmo.nvidia.com/workflows/fast-libero-20260926-spatial-2) | STAGING_DATA | Pending |
| Object | [fast-libero-20260926-object-1](https://us-west-2-aws.osmo.nvidia.com/workflows/fast-libero-20260926-object-1) | STAGING_DATA | Pending |
| Goal | [fast-libero-20260926-goal-1](https://us-west-2-aws.osmo.nvidia.com/workflows/fast-libero-20260926-goal-1) | QUEUED | Pending |
| LIBERO-10 | [fast-libero-20260926-10-1](https://us-west-2-aws.osmo.nvidia.com/workflows/fast-libero-20260926-10-1) | QUEUED | Pending |

Training source: `d756b1ff38f54378593d1083218f3773d88b4074`. Every workflow requests four L40S GPUs at NORMAL priority in `groot-l40s-03`. A dependent evaluation releases the training allocation and uses one L40S with five workers.

FAST settings: DCT scale 10, vocabulary 1024, maximum sequence 128; autoregressive policy with four layers, width 256 and four heads. Training uses the decoded replay loader, seed 42, 5001 epochs, global batch 1024 and EMA. Evaluation uses 2500 trials per suite, the same starting-state seeds and 550-step cap as ARC/OAT, predicting 32 actions and executing 16 before replanning.

Each workflow first fits its training-split BPE, audits held-out reconstruction and token lengths, and runs native DDP/bf16/checkpoint plus real simulator preflight checks. Only a complete final checkpoint can trigger full evaluation. No FAST success rate is available yet.

CPU validation: 166 tests passed (8 FAST, 13 OAT parity, 145 shared regressions). These are distinct from GPU preflight and full training.

The first Spatial submission was rejected before execution because 128 GiB exceeded the pool's per-GPU memory allowance. The accepted workflow uses 120 GiB for evaluation, within the allocation limit. That rejected submission consumed the `-1` workflow suffix; the accepted Spatial workflow is `-2`. No duplicate Spatial training was started.

[Source evidence (JSON)](fast_libero_20260926.json) · [FAST implementation/protocol](../LIBERO_FAST.md) · [Existing ARC/OAT/raw-DP result snapshot](arc_vs_oat_libero_20260925.md)
