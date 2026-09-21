# Five ARC settings shared across all LIBERO suites

These are **calibration-ranked sweep configurations**, using the same R/D/M on
LIBERO Spatial, Object, Goal, 10 and 90. There are two STK configurations, two
DUR configurations, and one common configuration evaluated in both modes.
The five distinct parameter triples expand to six mode/config combinations,
or **30 ARC policy runs** across five suites. This sweep has not been launched.

| Configuration | Mode | R (degrees) | D (metres) | M | Equal-suite calibration success |
| --- | --- | ---: | ---: | ---: | ---: |
| STK 1 | stk | 192 | 1.6 | 36 | 87.22% |
| STK 2 | stk | 192 | 0.8 | 32 | 86.11% |
| DUR 1 | dur | 192 | 1.6 | 24 | 88.33% |
| DUR 2 | dur | 384 | 0.8 | 32 | 84.67% |
| Shared | stk and dur | 384 | 1.6 | 36 | STK 87.22%; DUR 85.00% |

R and D cap accumulated rotation and translation in the 32-command lookahead.
M counts support rows including the initial anchor. All configurations retain
the original 16-command execution cadence, dt=0.05 seconds, float32 actions,
and separate translation/rotation clocks. These M values prioritize replay
fidelity: 24–36 supports carry 288–432 scalars versus 224 raw action scalars.
They are not strong compression settings.

## Evidence and selection

Only **calibration demonstrations 0 and 1 per task** determine these choices:
20 episodes for each small suite and 180 for LIBERO-90. Every table entry was
physically replayed on every suite. No final-test outcome is used. The raw
calibration baseline is 93.33% when suites receive equal weight.

| Configuration / mode | Spatial /20 | Object /20 | Goal /20 | 10 /20 | 90 /180 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw | 19 | 19 | 20 | 17 | 165 |
| STK 1 / stk | 16 | 17 | 20 | 17 | 155 |
| STK 2 / stk | 16 | 18 | 20 | 15 | 154 |
| DUR 1 / dur | 16 | 19 | 20 | 16 | 156 |
| DUR 2 / dur | 15 | 17 | 20 | 15 | 159 |
| Shared / stk | 16 | 17 | 20 | 17 | 155 |
| Shared / dur | 16 | 16 | 19 | 16 | 162 |

The original grid has 336 settings per mode/suite. There are **54 settings per
mode** passing the >=99% calibration execution-coverage screen in all five
suites, including uncapped budgets; 24 have finite R and D. Each suite has
weight 1/5, so LIBERO-90 does not dominate the choice. All selected settings
have at least 99.29% aggregate command coverage separately in each suite.

The shared configuration maximizes the lower of STK and DUR's equal-suite
success rates, then their mean, then prefers fewer supports and lower MSE.
This chooses R=384, D=1.6, M=36: the weaker mode scores 85.00%. Its worst suite
gap to raw is 15 percentage points. The best mean at M=24 ties its 86.11%
combined mean, but STK scores 83.89% and its worst suite gap is 25 points.

The four additional configurations jointly maximize their equal-suite success
rates, requiring five distinct finite R/D/M triples overall and two different
M values per mode. Ties prefer fewer total supports, then lower MSE. This puts
DUR's M=36 option in the shared comparison, with M=24 and M=32 as its two
additional settings. STK 1 and shared STK have identical aggregate calibration
successes and very similar reconstruction error; their small R effect should
not be interpreted as a reliable performance difference.

The [machine-readable evidence](libero_arc_common_configs_20260921.json)
contains all universally eligible calibration candidates, per-suite outcomes,
coverage, paired losses/gains, source run IDs and SHA-256 artifact receipts.
All 40 input artifacts were rehashed against the earlier independent audit.
Calibration controls pass exact raw-state repeatability and dense command
MSE <=1e-12. The replay source is
`7f28d4d99b69be775e281122b50683f31420dfa6`.

These are demonstration replay scores, not trained-policy scores, statistical
equivalence to raw, or global optima. The calibration samples are small.
The larger selection split (demos 2–9) previously tested STK 1 and DUR 1 on
Goal only: **28 of the 30 mode/config/suite combinations still need that
validation**. Existing per-suite final-test results do not establish these
shared configurations' performance. Keep the choices frozen for any subsequent
confirmation; do not retune them using the earlier final outcomes.

## Ready-to-compose recipes

| Configuration | Hydra experiment |
| --- | --- |
| STK 1 | `oat/libero_arc_stk_1_r192_d1p6_m36` |
| STK 2 | `oat/libero_arc_stk_2_r192_d0p8_m32` |
| DUR 1 | `oat/libero_arc_dur_1_r192_d1p6_m24` |
| DUR 2 | `oat/libero_arc_dur_2_r384_d0p8_m32` |
| Shared | `oat/libero_arc_shared_r384_d1p6_m36` |

Every recipe takes the same `benchmark.suite` and `benchmark.dataset` overrides
as the existing ARC policy recipes. The shared recipe requires an explicit
`benchmark.arc_mode=stk` or `benchmark.arc_mode=dur`; it has no implicit mode.
All inherit 5,001 epochs, global batch 1,024, 12-channel timed ARC and the
correct sqrt(3) STK velocity normalization bound.

Validation passed for all 30 mode/config/suite compositions, both ARC stage
constructors, inherited training budgets and the velocity bound. Constructing
the shared ARC stage without an explicit mode is rejected.

For example, compose shared STK for LIBERO-10 without starting training:

```bash
source emimic/bin/activate
python -m egomimic.trainHydra --cfg job \
  +experiment=oat/libero_arc_shared_r384_d1p6_m36 \
  benchmark.arc_mode=stk benchmark.suite=libero_10 \
  benchmark.dataset=/path/to/libero_10.zarr
```

Existing running full campaigns still use their earlier per-suite replay
selections. Adding these recipes does not alter their source, checkpoint
meaning, training budget, or selection evidence. Full training's replay gate
continues to require the matching completed replay result for each policy.
