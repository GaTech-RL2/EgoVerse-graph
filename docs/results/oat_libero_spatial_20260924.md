# OAT LIBERO Spatial: full evaluation

OAT achieved **61.68% success (1,542/2,500 episodes)** across all 10 LIBERO Spatial tasks. The evaluation completed September 24, 2026 at approximately 06:10 UTC (September 23, 11:10 p.m. Pacific).

The protocol uses 50 trials per task across five evaluation repetitions: 250 trials per task. Each repetition covers every task, with success rates of 60.8%, 63.4%, 61.4%, 62.0%, and 60.8%. The sample standard deviation across repetitions is 1.08 percentage points; these are evaluation repetitions of one trained policy, not training seeds.

Each task places the black bowl onto the plate:

| Bowl starting location | Successes | Success rate |
|---|---:|---:|
| between the plate and the ramekin | 150/250 | 60.0% |
| next to the ramekin | 141/250 | 56.4% |
| from table center | 224/250 | 89.6% |
| on the cookie box | 181/250 | 72.4% |
| in the top drawer of the wooden cabinet | 158/250 | 63.2% |
| on the ramekin | 31/250 | 12.4% |
| next to the cookie box | 197/250 | 78.8% |
| on the stove | 211/250 | 84.4% |
| next to the plate | 118/250 | 47.2% |
| on the wooden cabinet | 131/250 | 52.4% |

The policy completed 5,001 epochs and 270,054 optimizer/EMA updates at global batch size 1,024. Evaluation uses the EMA weights, two adjacent observations, a 32-action prediction horizon, execution of the first 16 actions before replanning, and a maximum of 550 steps per episode. The native OAT tokenizer and predictor are restored from the self-contained final checkpoint.

Five evaluation processes ran the five repetitions concurrently on one L40S. The full merged records were independently checked for missing, unexpected, and duplicate episodes, exact task/seed coverage, initial-state identities, and the released control protocol. Scores were recomputed from the merged records and matched the exported scores exactly. The 49 episodes from the earlier preempted evaluation are excluded.

Observed inference latency was 23.39 ms mean and 27.72 ms p95 under five concurrent evaluation workers. These timings describe this concurrent execution setup and should be reported separately from single-worker latency measurements.

Provenance:

- Evaluation workflow: `arc-oat-eval-p01-20260923-spatial-1`.
- Evaluation source: `c7858777a5472aaa341e66baed6dd3b2dc08c58c`.
- Training run: `arc-cache-oat-20260923-spatial`.
- Training source: `d2c89b9314aa59e5c59bd41b29c2a99b3b0b46b9`.
- Final checkpoint SHA-256: `7b9346b2fd1a5b7162206c475a78f9fba865ae444b94801d44f94d69d8f222f5`.
- [Machine-readable scores and artifact hashes](oat_libero_spatial_20260924.json).

This report covers Spatial. It does not establish the outcome of the full ARC-versus-OAT comparison or the remaining LIBERO suites.
