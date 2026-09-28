# LIBERO training progress: September 26 night

Measured at 2026-09-27T04:29:42.706244+00:00. Original comparison: 57/60 policies have finished training and 51/60 have full evaluations. One completed result needs its paired starting-state check repaired. Including FAST, there are 64 requested cells and 51 full scores.

| Run | Optimizer updates / target | Complete | Estimated training remaining |
| --- | ---: | ---: | ---: |
| arc-dpr3-20260926-stk1-object | 306,360 / 325,065 | 94.2% | 0.7 hours |
| arc-dpr3-20260926-shared-stk-object | 308,110 / 325,065 | 94.8% | 0.7 hours |
| arc-dpr3-20260926-shared-dur-object | 306,720 / 325,065 | 94.4% | 0.7 hours |
| fast-libero-20260926-goal | 190,720 / 280,056 | 68.1% | 4.0 hours |
| fast-libero-20260926-10 | 201,400 / 605,121 | 33.3% | 17.3 hours |

Estimates use actual advancing optimizer/EMA logs over a recent interval, and exclude rollout evaluation, queueing and future interruptions. All policy jobs retain 5001 epochs, global batch 1024 and seed 42. Full scores require 2500 episodes.

Six additional ARC OAT-DP models have finished training and are evaluating:

| Config / suite | Completed trials at score snapshot |
| --- | ---: |
| dur_1 dur / libero_object | 2119 / 2500 |
| dur_2 dur / libero_spatial | 963 / 2500 |
| dur_2 dur / libero_object | 1293 / 2500 |
| shared stk / libero_goal | 275 / 2500 |
| shared dur / libero_spatial | 359 / 2500 |
| shared dur / libero_goal | 1563 / 2500 |

Recovery allocations (NORMAL priority, L40S; running setup at verification):

- `fast-libero-r1-20260927-spatial-1`: resume training.
- `fast-libero-r1-20260927-object-1`: fresh training after cuda initialization failure.
- `arc-eval-paired-20260927-shared-dur-10-1`: fresh evaluation for starting state validation.

FAST Spatial stopped after 880 logged updates with an unspecified CUDA launch failure; its SHA-verified durable checkpoint restores 540 optimizer/EMA updates and the embedded BPE. FAST Object passed its two-update GPU/simulator preflight, then failed CUDA initialization before its full policy training started. Its replacement starts full training afresh. The two replacements use the same immutable implementation and full budgets; Goal and LIBERO-10 continue untouched.

The completed shared DUR / OAT-DP / LIBERO-10 evaluation has 29 initial-state mismatches, all in the cabinet task. The 68.56% score is flagged; the fresh evaluation uses the same full checkpoint and reruns all 2500 trials without selecting outcomes. The original score and records remain preserved.

[Full result table](arc_vs_oat_libero_20260926_night.md) · [Progress and launch evidence](libero_progress_20260926_night.json)
