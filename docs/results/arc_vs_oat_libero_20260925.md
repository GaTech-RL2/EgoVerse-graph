# ARC versus OAT: LIBERO results, September 25, 2026

Artifact audit: **2026-09-25T05:43:47.619118+00:00**. All complete entries use 10 tasks × 50 trials × five evaluation repetitions (2500 episodes), from one trained checkpoint. These are not five independent training seeds.

The same native OAT reference is repeated in both tables. The original ARC head is the conditional U-Net; the new ARC head uses the released OAT diffusion Transformer. Native OAT itself remains the learned tokenizer plus autoregressive policy.

## Original ARC: current U-Net DP

| Method | Spatial | Object | Goal | LIBERO-10 |
| --- | ---: | ---: | ---: | ---: |
| Native OAT | 61.68% | 0.36%* | 75.52% | 59.44% |
| ARC STK1 | 68.00%† | 52.80% | 82.20% | 51.88% |
| ARC STK2 | 68.44%† | 52.36% | 84.64% | 52.24%† |
| ARC DUR1 | 68.32%† | 53.64% | 89.56% | 54.76%† |
| ARC DUR2 | 69.24%† | 63.72% | 87.28% | 57.44%† |
| ARC shared STK | 65.92%† | 58.88% | 84.84% | 55.44%† |
| ARC shared DUR | 67.72%† | 59.64% | 88.96% | 56.20%† |
| Plain DP, no tokenizer | Pending | Pending | Pending | Pending |

## ARC: released OAT DP Transformer

| Method | Spatial | Object | Goal | LIBERO-10 |
| --- | ---: | ---: | ---: | ---: |
| Native OAT | 61.68% | 0.36%* | 75.52% | 59.44% |
| ARC STK1 | — | — | 87.00% | 67.88% |
| ARC STK2 | — | — | 81.32% | 70.12% |
| ARC DUR1 | 73.76% | — | — | 68.76% |
| ARC DUR2 | — | — | — | — |
| ARC shared STK | — | — | — | — |
| ARC shared DUR | — | — | — | — |
| Plain DP, no tokenizer | Pending | Pending | Pending | Pending |

## Interpretation and unresolved entries

All six completed Transformer ARC comparisons pass native full-protocol validation, including zero initial-state mismatches against OAT. Their final checkpoints were downloaded and independently verified: all 5001 epochs, exact optimizer/EMA budgets, complete optimizer and normalization states. The three LIBERO-10 L40S migrations have finished, superseding the September 24 pending-migration snapshot.

The original STK1/LIBERO-10 reevaluation is complete at 51.88% (1297/2500), versus OAT 59.44%, with matching initial states. All 24 original ARC individual scores are complete.

† Eleven original comparisons have different initial states: all six Spatial runs (200 episodes each), and five LIBERO-10 runs (600 episodes each). Full 2500 episode reevaluations on the original immutable checkpoints were accepted on L40S under `arc-paired-20260925-*`; all 11 were PENDING at the scheduler snapshot stored in the accompanying JSON. Old scores are preserved here until replacement evaluations finish. Do not interpret these entries as fully paired estimates.

* OAT Object is 9/2500=0.36%, an unresolved anomaly. The correct full-budget checkpoint and its identity were verified; this alone does not establish the cause. Avoid a settled ARC-versus-OAT conclusion on Object.

Only 6 of 24 Transformer ARC selections have complete scores. The other 18 workflows were interrupted (17 canceled, one backend failure); their saved artifacts are not live progress. STK2/Object was canceled by quota reclamation at 1559/2500 episodes, which is not a suite score.

Plain DP with no ARC/OAT tokenizer is newly requested for both architectures on Spatial/Object/Goal/LIBERO-10. No verified baseline score exists yet; pending cells must not be read as zero. Full training remains 5001 epochs/global 1024 and full evaluation remains 2500 episodes. LIBERO-90 and the ARC+OAT hybrid remain deferred.

## Configuration and evidence

| Selection | R | D | M |
| --- | ---: | ---: | ---: |
| STK1 | 192 | 1.6 | 36 |
| STK2 | 192 | 0.8 | 32 |
| DUR1 | 192 | 1.6 | 24 |
| DUR2 | 384 | 0.8 | 32 |
| shared STK / DUR | 384 | 1.6 | 36 |

All policies use the same two-frame observations, dense 32 step horizon,16 executed actions per replan, and550 step episode cap. Resets use seeded simulation with settling and fresh post-settle observations; they do not use the official saved initial-state files.

The [machine-readable snapshot](arc_vs_oat_libero_20260925.json) includes per-task scores, exact run identities, checkpoint/artifact hashes, protocol checks, pairing failures, and new checkpoint audits. [ARC/OAT DP architecture](../LIBERO_ARC_OAT_DP.md).
