# Pusher: absolute-duration ARC versus fixed-time DP16

Completed November 17, 2026 UTC. **SR@80 is the primary success metric.**
All 372 planned level evaluations in this comparison are complete: two models,
two training regimes, three training seeds, and levels L0 through L30. This is
18,600 rollouts; no missing levels or partial rollout subsets are averaged in.

## Downloads

- [Every level, side-by-side comparison](per_level_comparison.csv): 62 rows,
  one per training regime and level, with both models and ARC-minus-DP differences.
- [Every level and training seed](per_level_by_seed.csv): 372 rows, including
  success counts, coverage, run IDs, and exact R2 source object URIs.
- [Overall summary, L0 separated from L1–30](summary.csv): four model/regime rows.

All columns ending in `_pct` are percentages on a 0–100 scale, and differences
ending in `_pp` are percentage points. Source JSON metrics remain fractions.
`sr80_seed_sd_pp` is the sample standard deviation across the three training
seeds, not a confidence interval or episode-level standard deviation.

## Coverage heatmaps with level designs

- [All 31 levels (PDF)](figures/coverage-by-level-and-design.pdf)
- [All 31 levels (PNG)](figures/coverage-by-level-and-design.png)
- Larger panels: [L0–L14](figures/coverage-by-level-and-design-L00-L14.pdf) and
  [L15–L30](figures/coverage-by-level-and-design-L15-L30.pdf).

These charts show mean peak coverage, as requested, rather than the primary
SR@80 success metric. Both training regimes use the same 0–100% color scale;
ARC-minus-DP panels use a shared −30 to +30 percentage-point scale. Each cell
contains all 150 rollouts for that level, regime, and model.

The obstacle schematics use exact segment endpoints from the frozen triangle
seed bank. All 31 layouts were checked against the pinned simulator obstacle
source (SHA256 in `figures/level-designs.json`). Object, goal, and pusher poses
are omitted because they vary across resets; the diagrams represent level
geometry, not one rollout screenshot. The arena is 512×512 with wall radius 4.

To rebuild, install `figures/plot-requirements.txt` in an isolated plotting
environment and run `figures/render_heatmap.py`. The figure provenance file
records plot versions and input hashes. Refresh `build_results.py` afterward
to update the report file manifest.

## Completed results

| Training regime | Model | L0 SR@80 | L1–30 SR@80 | L0 peak coverage | L1–30 peak coverage |
| --- | --- | ---: | ---: | ---: | ---: |
| Clean only | DP16 | 15.33% | 0.44% | 52.41% | 9.50% |
| Clean only | ARC absolute duration | 32.00% | 0.13% | 69.82% | 6.93% |
| Obstacle co-training | DP16 | 10.00% | 2.98% | 53.97% | 23.59% |
| Obstacle co-training | ARC absolute duration | 22.00% | 7.87% | 65.14% | 27.97% |

Each model/regime row has 150 L0 rollouts and 4,500 L1–30 rollouts. ARC improves
coverage and SR@80 with obstacle co-training. With clean-only training, ARC
improves L0 but does not improve obstacle coverage or SR@80. Even with
co-training, absolute obstacle success remains low.

## Scope and metric definitions

- Models: fixed-time raw-action DP16 and ARC absolute duration (`xy_absdur_m56`).
- Target embodiment: triangle. Triangle obstacle demonstrations are held out.
- `clean`: clean-only training demonstrations from the selected pushers.
- `cotrain`: the same clean demonstrations plus obstacle demonstrations from
  the other training pushers; triangle obstacle data stays excluded.
- Training seeds: 42, 43, and 44. Evaluation: the same 50 frozen resets per
  level for each model and training seed, with L0 reported separately.
- SR@80: the fraction of rollouts whose peak target coverage is at least 0.80.
- Peak coverage: mean of each rollout's maximum target coverage.
- Final coverage: mean target coverage at the end of each rollout.
- Per-level comparison values pool all 150 episodes. Because each seed has
  the same 50 episodes, pooling equals the mean across training seeds.
- L1–30 aggregates pool all 4,500 episodes. Equal episode counts per level
  make this equal to the unweighted average of the 30 level means.

These files contain only this completed pusher comparison. Other ARC timings
and LIBERO results are outside this report.

## Pinned experiment and provenance

Training source: `505c3eed6a6e62ad3fc985ab4bdee6ebe830f0d8`, runtime branch
`codex/pusher-main-arrival-20261105`. ARC timing follows main
`c7e110461650d09a88b2c2dff8f02db54b1deace`, with absolute arrival times,
measured actuator-state anchoring, native units, and no speed-bound/projection
adjustment. Geometry is D80/R26/M56. Training used final-240k checkpoints,
global batch 128, and four L40S GPUs per run. Evaluation used EMA, 100 sampler
steps, fixed-eight-control replanning, and one L40S per level job.

The original submission records are preserved in the adjacent
[benchmark launch report](../benchmark-rerun-20261113/README.md). Recovery jobs
reused the same run IDs/checkpoints and frozen evaluation contracts. A failed
attempt is not an additional sample: exports deduplicate each
(regime, model, training seed, level) and require all expected cells.

`source-level-summaries.json` preserves the collected level summary fields used
by this report, with the previous SR95 field omitted. `provenance.json` records
the experiment contract. `manifest.json` records file hashes and coverage checks.
The exact source object for each row is listed in `per_level_by_seed.csv`.
No checkpoint files, episode traces, datasets, or credentials are included.

## Rebuild and validate

After activating the project Python environment, run from repository root:

```sh
python validation/pusher-main-arrival-results-20261117/build_results.py
```

The script uses only the Python standard library. It checks the exact matrix,
unique keys, 50 episodes per seed-level, integer SR@80 success counts, finite
bounded metrics, and agreement between pooled success counts and reported SR.
It regenerates the three CSVs and file manifest deterministically.

This results branch extends the separate validation branch; runtime source and
main are not modified.
