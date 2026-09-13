# UNITE cotrain campaign helpers (ICE)

Everything used to launch, score and summarise the UNITE-vs-Paper-DP cotrain
sweeps on PushShapes sim_v2 (U-Socket + ChainGripper), September 2026. The
reproduction guide with row tables, environment identity, protocol and results
is `docs/experiments/unite_cotrain_handoff_2026-09-10.md`.

| file | role |
|---|---|
| `launch_ct2.sh` | row table → `scripts/ice/launch_unite_cotrain.sbatch` (2 × H200, cedar output, requeue-safe) |
| `rollout_ct3.sh` | protocol rollout submitter (rev-3 budgets, full horizon, EMA, replan 8, chunk start 0; level 0 seed blocks or OEC-56 level ranges) |
| `monitor_ct3.py`, `monitor_ct3_rows_s3.json`, `monitor_ct3.sbatch` | self-chaining checkpoint scorer; rewrites `summary.tsv` |
| `cotrain_predicate_rev3.py` | success predicate (one UNITE-cotrain checkpoint above best DP BC and best DP cotrain on both embodiments) |
| `export_peaks.py`, `plot_protocol.py` | per-episode peaks → four-curve protocol figure |
| `oec_summary.py`, `ttfc.py`, `ln_gain.py`, `wb_*.py`, `bench_unite.py` | obstacle-level summary, time-to-first-contact, LayerNorm-gain drift, W&B history pulls, step-time micro-benchmark |
| `budgets/`, `seeds/`, `protocol/` | rev-3 horizon budgets, OEC-56 seed bank + ICE verification report, Elmo's protocol text and budget scripts |
| `bad_gpu_nodes.txt` | Slurm exclude list |

Paths are the ICE account's (`/home/hice1/agao81/scratch`, cedar
`/storage/cedar/cedar0/cedarp-dxu345-0/agao81/runs`); grep for `hice1/agao81`
when porting.
