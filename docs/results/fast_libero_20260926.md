# FAST LIBERO launch — September 26, 2026

Artifact/scheduler audit: 2026-09-26T19:04:16.704198+00:00. This is a launch snapshot, not a success-rate result.

The native OAT-codebase FAST baseline is submitted for all four active suites.
Spatial and Object completed tokenizer fitting/reconstruction and entered the full training phase. Spatial has a verified four-GPU bf16/checkpoint and real-simulator preflight receipt. Goal and LIBERO-10 remain queued. Full-run optimizer progress is not yet reported.

| Suite | Workflow | Phase | Full SR |
| --- | --- | --- | ---: |
| Spatial | [fast-libero-20260926-spatial-2](https://us-west-2-aws.osmo.nvidia.com/workflows/fast-libero-20260926-spatial-2) | TRAINING | Pending |
| Object | [fast-libero-20260926-object-1](https://us-west-2-aws.osmo.nvidia.com/workflows/fast-libero-20260926-object-1) | TRAINING | Pending |
| Goal | [fast-libero-20260926-goal-1](https://us-west-2-aws.osmo.nvidia.com/workflows/fast-libero-20260926-goal-1) | QUEUED | Pending |
| LIBERO-10 | [fast-libero-20260926-10-1](https://us-west-2-aws.osmo.nvidia.com/workflows/fast-libero-20260926-10-1) | QUEUED | Pending |

Training source: `d756b1ff38f54378593d1083218f3773d88b4074`. Every workflow requests four L40S GPUs at NORMAL priority in `groot-l40s-03`. A dependent evaluation releases the training allocation and uses one L40S with five workers.

FAST settings: DCT scale 10, vocabulary 1024, maximum sequence 128; autoregressive policy with four layers, width 256 and four heads. Training uses the decoded replay loader, seed 42, 5001 epochs, global batch 1024 and EMA. Evaluation uses 2500 trials per suite, the same starting-state seeds and 550-step cap as ARC/OAT, predicting 32 actions and executing 16 before replanning.

Each workflow first fits its training-split BPE, audits held-out reconstruction and token lengths, and runs native DDP/bf16/checkpoint plus real simulator preflight checks. Only a complete final checkpoint can trigger full evaluation. No FAST success rate is available yet.

CPU validation: 166 tests passed (8 FAST, 13 OAT parity, 145 shared regressions). These are distinct from GPU preflight and full training.

The first Spatial submission was rejected before execution because 128 GiB exceeded the pool's per-GPU memory allowance. The accepted workflow uses 120 GiB for evaluation, within the allocation limit. That rejected submission consumed the `-1` workflow suffix; the accepted Spatial workflow is `-2`. No duplicate Spatial training was started.

[Source evidence (JSON)](fast_libero_20260926.json) · [FAST implementation/protocol](../LIBERO_FAST.md) · [Existing ARC/OAT/raw-DP result snapshot](arc_vs_oat_libero_20260925.md)

## Held-out reconstruction

These are tokenizer replay metrics, not policy success rates. BPE uses only training episodes; the action normalizer follows the OAT release's all-frame limits.

| Suite | Validation chunks | Action MSE | Median / maximum tokens | Truncated | Failed decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| libero_spatial | 6389 | 0.00027218 | 60 / 111 | 0.00% | 0/6389 |
| libero_object | 7567 | 0.00035636 | 44 / 96 | 0.00% | 4/7567 |

Object includes four malformed round trips (0.0529%). Their zero-action fallback is included in the MSE, preserving the released decoder behavior. No validation chunk in either suite exceeds the 127-action-token supervised limit.

Spatial and Object dataset/split/action-normalizer fingerprints match the existing OAT evaluation artifacts exactly. These data checks are separate from rollout initial-state pairing, which will be checked when full FAST evaluations exist.
