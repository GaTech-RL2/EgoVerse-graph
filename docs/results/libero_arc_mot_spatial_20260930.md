# Completed LIBERO Spatial ARC MoT comparison

All three MoT policies completed 5,001 epochs and 270,054 optimizer/EMA updates. Each final EMA policy completed 2,500 episodes: ten tasks, 50 trials per task, five evaluation repetitions. The final workflow finished September 30, 2026 at 17:56 PDT (October 1, 00:56 UTC).

| Decoder | Spatial SR | Delta vs original ARC |
| --- | ---: | ---: |
| Original ARC (joint tokens) | 76.16% | +0.00 pp |
| Shared decoder (split shape/velocity tokens) | 70.40% | -5.76 pp |
| Separate shape/velocity decoders | 76.04% | -0.12 pp |
| Shared decoder + shape-to-velocity mask | 77.12% | +0.96 pp |
| MoT xyz / rotation / gripper | 64.60% | -11.56 pp |
| MoT shape / velocity | 60.44% | -15.72 pp |
| MoT shape / velocity + mask | 72.48% | -3.68 pp |

Masked shape/velocity MoT is the strongest new arm at 72.48%, improving 12.04 percentage points over unmasked shape/velocity MoT. It remains 3.68 points below original ARC and 4.64 points below the previous masked shared decoder. These are single-training-seed Spatial results; the five repetitions vary evaluation trials, not training seeds.

## Protocol and validation

All models use frozen STK2 (R=192 degrees, D=0.8 m, M=32), global batch 1,024, horizon 32, two observation frames, ten DDIM steps, and 16 executed actions before replanning. The MoT action networks match the prior shared decoder parameter budget within 0.1%; per-modality widths differ as documented in [the architecture description](../LIBERO_ARC_MOT.md).

Training resumed from verified optimizer/EMA checkpoints after preemption. The initial allocation was eight L40S GPUs at microbatch 128; recovery used four L40S GPUs at microbatch 256. The total batch, optimizer/EMA budget, learning rates, source, and evaluation protocol are unchanged. The hardware/layout change does not imply bitwise-identical stochastic training.

The final audit independently checked artifact SHA-256 hashes, source/model identity, completed training budgets, all 7,500 unique planned candidate episodes, ten complete tasks per arm, and exact pairing to the original ARC initial-state hashes. There were zero initial-state mismatches. Scores were recomputed from the episode records and matched the published collector results.

## Per-task success rates

Every row has 250 trials per model. All tasks pick up the black bowl and place it on the plate; the first column describes its initial location.

| Initial bowl location | MoT xyz/rotation/gripper | MoT shape/velocity | MoT masked |
| --- | ---: | ---: | ---: |
| between the plate and the ramekin | 62.00% | 64.00% | 78.80% |
| from table center | 84.40% | 72.40% | 91.60% |
| in the top drawer of the wooden cabinet | 85.20% | 84.00% | 88.00% |
| next to the cookie box | 79.20% | 62.80% | 86.00% |
| next to the plate | 48.80% | 44.80% | 68.00% |
| next to the ramekin | 68.00% | 64.80% | 87.20% |
| on the cookie box | 64.80% | 64.80% | 58.40% |
| on the ramekin | 9.60% | 14.00% | 13.20% |
| on the stove | 88.00% | 91.20% | 81.20% |
| on the wooden cabinet | 56.00% | 41.60% | 72.40% |

## Artifacts

- [Verified final audit and artifact receipts](libero_arc_mot_spatial_20260930.json)
- [Overall CSV](libero_arc_mot_spatial_20260930.csv)
- [Per-task CSV](libero_arc_mot_spatial_20260930_per_task.csv)
- [Source, checkpoint and workflow manifest](../arc_mot_runs_20260930.json)
- [PR #195](https://github.com/GaTech-RL2/EgoVerse-graph/pull/195)

The checked-in audit includes the immutable R2 receipt for each source artifact. The collector result is `s3://rldb/experiments/arc-oat-20260919/campaigns/arc-mot-20260930-spatial/results.json`.
