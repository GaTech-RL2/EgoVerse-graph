# Results

All numbers are **NON-PROTOCOL** (see `docs/03_EVAL.md`). Peak coverage, 40
episodes per cell, 3,000-episode budgets, level 0, seeds 0-39.

Raw table: `/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/results/artic_rollout_45cells.csv`

## Articulated co-train sweep — 45/45 cells, COMPLETE

| embodiment | dur_M56 | stk_M56 | dur_M16 | **stk_M16** | DP | role |
|---|---|---|---|---|---|---|
| u_socket | 0.103 | 0.116 | 0.106 | 0.093 | 0.078 | in-domain |
| gripper | 0.044 | 0.045 | 0.036 | 0.088 | 0.023 | in-domain |
| chain_gripper | 0.089 | 0.062 | 0.058 | 0.078 | 0.057 | in-domain |
| suction | 0.359 | **0.382** | 0.221 | 0.255 | 0.279 | in-domain |
| triangle | 0.557 | 0.693 | 0.610 | **0.710** | 0.601 | in-domain |
| flipper | 0.467 | 0.532 | 0.519 | **0.588** | 0.437 | in-domain |
| spring | 0.545 | 0.587 | 0.572 | **0.602** | 0.483 | in-domain |
| umi | 0.001 | **0.058** | 0.001 | 0.039 | 0.031 | ZERO-SHOT |
| scoop | 0.049 | 0.028 | 0.050 | **0.052** | 0.025 | ZERO-SHOT |
| **in-domain mean** | 0.309 | 0.345 | 0.303 | **0.345** | 0.280 | |
| **zero-shot mean** | 0.025 | 0.043 | 0.025 | **0.045** | 0.028 | |

### ARC vs DP, paired on the same 40 seeds

Every arm scores identical seeds, so this is a per-episode paired difference;
CIs are a 20k-sample bootstrap.

| scope | ARC − DP | 95% CI |
|---|---|---|
| all in-domain (7 tools, n=280) | **+0.046** | [+0.018, +0.074] |
| pushers only (3 tools, n=120) | **+0.075** | [+0.036, +0.116] |
| flipper | +0.090 | [+0.030, +0.148] |
| spring | +0.094 | [+0.019, +0.171] |
| scoop (zero-shot) | +0.020 | [+0.002, +0.040] |
| every prehensile tool | +0.015 … +0.031 | all include 0 |

**Every significant gain is in the pusher group.** No prehensile tool separates
from DP — consistent with none of them working well enough for the action
representation to matter.

### Arms disagree — do not average them

`stk_M16` leads all three pushers but is nearly the **worst** arm on suction
(0.255 vs `stk_M56`'s 0.382). On `umi` the two duration arms collapse to
**~0.001** — total failure — while `stk_M56` reaches 0.058. Averaging the four
produced a misleading 0.024 and hid the sharpest result in the study:

> **Stacked transfers to unseen tools; duration does not.**

## Single-embodiment BC baselines — 6/7 cells

Same tokenizer, model, optimiser, 240k steps; trained on one embodiment each.
Exposure is matched by construction (see `docs/02_TRAINING.md`).

| embodiment | BC | co-train (mean of 5 arms) | BC − co-train | CI |
|---|---|---|---|---|
| flipper | 0.128 | 0.507 | **−0.380** | excludes 0 |
| suction | 0.022 | 0.299 | **−0.277** | excludes 0 |
| chain_gripper | 0.006 | 0.069 | **−0.063** | excludes 0 |
| u_socket | 0.047 | 0.099 | **−0.052** | excludes 0 |
| gripper | 0.009 | 0.047 | **−0.038** | excludes 0 |
| triangle | 0.575 | 0.634 | −0.059 | **includes 0** |
| spring | — | 0.558 | — | **cell never ran** |

Pooled over the six:

| comparison | value | 95% CI |
|---|---|---|
| co-train DP − BC | **+0.115** | [+0.081, +0.150] |
| ARC − co-train DP | **+0.038** | [+0.008, +0.067] |

**The multi-embodiment mixture is worth about three times what the codec is
worth on top of it.** Sharing 262.78M parameters across seven tools *buys*
control rather than diluting it — the opposite of the capacity-dilution
hypothesis the baseline was built to test.

`triangle` is the exception: BC holds its own. It is the simplest pusher (3-DOF,
no grip), which is a plausible reason it needs no help.

## THE CONFOUND — never report the grouping without it

Grouped by contact mechanism, coverage looks like a clean story: prehensile
0.02-0.12, adhesion 0.22-0.38, pushers 0.44-0.71. It is not clean.

**Across the seven trained tools, peak coverage vs p99 frame budget gives
r = +0.984.**

| group | budget | coverage |
|---|---|---|
| prehensile (u_socket, gripper, chain_gripper, umi) | 626-752 | 0.02-0.12 |
| adhesion (suction) | 2,218 | 0.22-0.38 |
| pushers (triangle, flipper, spring, scoop) | 4,826-5,365 | 0.44-0.71 |

Peak coverage is a **maximum over time**, so a ~7x longer budget mechanically
buys more of it. The data **cannot** separate "grasping is hard" from "these
episodes are short". Both figures state this on their face.

The two zero-shot tools are the exceptions that break the fit — `scoop` has the
**longest budget of all** (5,365) and still scores 0.04, which is a
generalization failure, not a budget effect.

## Why coverage is low at all

Not underfitting. Evidence:

| signal | value |
|---|---|
| train loss | 5-10x reduction, flat over final 20% |
| `sqrt(MSE)/target_rms` | 5-12% |
| val EnergyScore | improves monotonically, no overfitting |
| DP arm | **best fit** (train 0.00194) + **worst coverage** (0.053 at the time) |
| corr(train loss, coverage) | **+0.49** — better fit, worse control |
| corr(val distance, coverage) | **+0.52** |
| u_socket **replay** vs rollout | **96.6% vs 22.5%** |

That last row is decisive: feeding ground-truth action chunks through the same
tokenizer and decoder reaches 96.6%. The codec can express the task. The entire
gap is closed-loop — compounding error, the classic BC failure. More gradient
steps do not touch it, and the positive loss-coverage correlation suggests they
may hurt.

Curves: `/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/results/figures/artic_curves.png`

## Figures

| figure | absolute path |
|---|---|
| every condition + zero-shot panel | `/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/results/figures/conditions_full.png` |
| BC vs co-train | `/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/results/figures/bc_vs_cotrain.png` |
| BC vs DP vs ARC | `/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/results/figures/bc_vs_dp_vs_arc.png` |
| grouped by mechanism | `/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/results/figures/artic_by_mechanism.png` |
| **the budget confound** | `/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/results/figures/artic_budget_confound.png` |
| ARC−DP paired deltas | `/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/results/figures/arc_vs_dp_delta.png` |
| pushers isolated | `/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/results/figures/arc_vs_dp_pushers.png` |
| loss / val curves | `/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/results/figures/artic_curves.png` |

Each has a `_dark` variant. Copies in
`/Users/rpunamiya/Desktop/GEAR/handover_figures/`. Regenerate with the
`plot_*.py` scripts in `/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/tools/`.

## The 260M comparison is CONFOUNDED

`HANDOVER.md` §3a reports `arc_dur_paper` 0.590 / `arc_stk_paper` 0.568 /
`dp_paper` 0.581 and claims "only the action representation differs". **That
claim is false for those runs.** An action-target off-by-one meant both ARC arms
trained on `a_t..a_t+80` (81 frames) while the DP baseline trained on
`a_t+1..a_t+16` — one frame earlier and one longer, mildly favouring ARC.

Token-level effect at D=80/M=56: mean 0.99 px, max 3.5 px, ~1.2% of the arc
budget. Small but systematic, and the reported gaps were already insignificant
(duration − stacked p=0.78).

Code fixed (`61bf2bf`); **the runs were trained before the fix and need
retraining before that table can be quoted.** The user deferred this
deliberately. The articulated work is unaffected — all `artic_*` configs always
sliced correctly.

## Superseded claims — do not propagate

- "All five arms fail the task" — drawn from the 21 cheapest cells that finished
  first, which are exactly the low-coverage ones. Wrong.
- "Coverage is low everywhere" — false; pushers reach 0.44-0.71.
- "`save_top_k=0` keeps only the newest checkpoint" — false; it retains periodic
  copies. Explicit pruning is what works.
