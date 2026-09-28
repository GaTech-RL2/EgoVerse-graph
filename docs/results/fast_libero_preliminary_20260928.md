# Preliminary FAST LIBERO Goal results — September 28, 09:52 Pacific

This is an interim task subset, not a complete LIBERO Goal suite result. FAST uses its fully trained 280056-update EMA checkpoint. Artifacts were read September 28 at 16:52 UTC. Five tasks have all 250 planned episodes each; a sixth task is partly evaluated. The remaining four tasks have not yet been evaluated.

On the five fully evaluated tasks, FAST achieved **20/1250 successes (1.60%)**. Including the partly evaluated sixth task, the running tally is 20/1429 (1.40%). No final FAST suite score is published. The cause of the poor initial performance is unresolved.

All comparisons below use the exact same 1250 trial identities and initial-state hashes, and the same observation/action/evaluation protocol. All model checkpoints completed their training budgets. These results come from one training seed.

| Method | Successes / 1250 | Success rate (%) |
| --- | ---: | ---: |
| Native FAST | 20 | 1.60 |
| ARC / U-Net / STK1 | 1090 | 87.20 |
| ARC / U-Net / STK2 | 1101 | 88.08 |
| ARC / U-Net / DUR1 | 1145 | 91.60 |
| ARC / U-Net / DUR2 | 1141 | 91.28 |
| ARC / U-Net / shared STK | 1093 | 87.44 |
| ARC / U-Net / shared DUR | 1146 | 91.68 |
| ARC / OAT-DP / STK1 | 1116 | 89.28 |
| ARC / OAT-DP / STK2 | 1069 | 85.52 |
| ARC / OAT-DP / DUR1 | 1039 | 83.12 |
| ARC / OAT-DP / DUR2 | 1060 | 84.80 |
| ARC / OAT-DP / shared STK | 1092 | 87.36 |
| ARC / OAT-DP / shared DUR | 1072 | 85.76 |
| Native OAT | 954 | 76.32 |
| Plain DP / U-Net | 1079 | 86.32 |
| Plain DP / OAT-DP | 1046 | 83.68 |

FAST completed-task breakdown:

| Task | Successes / 250 | Success rate (%) |
| --- | ---: | ---: |
| open the middle drawer of the cabinet | 0 | 0.00 |
| put the bowl on the stove | 5 | 2.00 |
| put the wine bottle on top of the cabinet | 10 | 4.00 |
| open the top drawer and put the bowl inside | 0 | 0.00 |
| put the bowl on top of the cabinet | 5 | 2.00 |

Other FAST suites continue training on 4 L40S each; the Goal evaluation runs on 1 L40S. Verified optimizer progress around 09:50 Pacific:

| Suite | Optimizer updates | Training complete (%) |
| --- | ---: | ---: |
| libero_spatial | 162530 / 270054 | 60.2 |
| libero_object | 164070 / 325065 | 50.5 |
| libero_10 | 385970 / 605121 | 63.8 |

The full-score collector has reached 60/64 completed cells: all original ARC/OAT/plain-DP results are complete; all four FAST suite scores remain pending.

[Source audit](fast_libero_preliminary_20260928.json) · [Matched subset CSV](fast_libero_preliminary_20260928.csv)
