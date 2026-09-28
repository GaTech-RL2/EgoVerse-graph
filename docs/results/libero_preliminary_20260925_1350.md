# Preliminary LIBERO evaluations: September 25, 13:50 Pacific

Snapshot: 2026-09-25T20:50:19.290970+00:00. These are observed successes among episodes completed so far, not full-suite estimates. Every run targets 2500 episodes. Task-ordered execution creates unequal task coverage.

| Suite | Method | Successes / completed trials | Preliminary SR | OAT on the same trials | Fully evaluated tasks / 10 |
| --- | --- | ---: | ---: | ---: | ---: |
| Spatial | ARC / OAT-DP STK1 | 1532 / 1969 | 77.81% | 64.14% | 7 / 10 |
| Spatial | Plain DP / OAT-release | 1411 / 2033 | 69.40% | 64.24% | 7 / 10 |
| Spatial | Plain DP / U-Net | 23 / 130 | 17.69% | 63.08% | 0 / 10 |
| Object | Plain DP / U-Net | 787 / 1632 | 48.22% | 0.37% | 6 / 10 |
| Object | Plain DP / OAT-release | 0 / 723 | 0.00% | 0.69% | 2 / 10 |
| Goal | Plain DP / U-Net | 2018 / 2350 | 85.87% | 75.23% | 9 / 10 |
| LIBERO-10 | Plain DP / OAT-release | 43 / 120 | 35.83% | 59.17% | 0 / 10 |

On the exact same 1969 Spatial trials and starting-state hashes, ARC/OAT-DP STK1 scores 77.81% (1532/1969) and plain OAT-release DP scores 69.53% (1369/1969), a preliminary difference of 8.28 percentage points. Native OAT scores 64.14% on that same subset.

The OAT comparator in each row is recomputed over that row's exact completed trial identities, with matching starting-state hashes and control protocol. Rows cover different subsets, so their OAT values differ. Do not compare aggregate partial scores across rows as if task coverage were equal.

Plain U-Net Spatial and plain OAT-release DP LIBERO-10 have only reached their first task. Goal U-Net has completed nine tasks and begun the tenth. Object remains unresolved: plain OAT-release DP has zero successes across 723 trials, including two fully evaluated tasks; native OAT and ARC/OAT-DP STK2 also had anomalously low completed Object scores. This does not establish a root cause.

Each result uses one trained checkpoint. The five evaluation repetitions are not five independent training seeds. These numbers do not fill the final-result table; its completed count remains 35/60 at this check.

[CSV](libero_preliminary_20260925_1350.csv) · [Artifact hashes, per-task counts and matched comparisons](libero_preliminary_20260925_1350.json) · [Completed results](arc_vs_oat_libero_20260925.md)
