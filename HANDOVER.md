# Handover — U-Socket ARC action-codec study (+ PushShapes data generation)

**Date:** 2026-09-10 · **Author:** previous agent session · **Audience:** the agent picking this up

Two workstreams are live and they are coupled: the codec study is bottlenecked on
data quality, and the data-generation fix is what unblocks it.

---

## 0. TL;DR — what you need to know in 60 seconds

1. **The codec question is answered, and the answer is "it doesn't matter."** At the
   260M paper-DP scale, duration-timed ARC, stacked-velocity ARC, and plain Diffusion
   Policy are **statistically indistinguishable** (all pairwise paired-t `p > 0.78` on
   the same 40 seeds). See §3.
2. **The bottleneck is the data, not the representation.** Replay of the ARC codec
   reconstructs held-out trajectories at 96.6% coverage; the trained policy rolls out at
   ~22–32%. A ~70-point replay-to-rollout gap cannot be closed by changing the token.
3. **The training corpus is unusable for 9 of the embodiments.** 27,045 of 33,594
   episodes were collected with the engage channel never written and orientation never
   commanded — every "grasp" is a shove. Documented in
   `Tsimulation/sim_v2/collect/README_DATA_GENERATION.md` (on `origin/sim/data-gen-readme`).
4. **Everything is committed and pushed.** Nothing of value lives only on this laptop.
5. **The AWS access key used in this session needs rotating** (the user said they would).
   It is `AKIA…IUO7` under the osmo credential `egoverse-aws`. Do not re-paste it into
   files.

**If you do one thing next:** fix data generation (§6.1). Do not run more codec cells.

---

## 1. Where the code is

| What | Where |
|---|---|
| **Active codec worktree** | `/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec` |
| Branch | `codec-replay-rotfix` @ `25fbdf1`, **pushed** to `origin/codec-replay-rotfix` |
| Duration-codec-only branch (user asked for this separately) | `origin/codec-duration` |
| Parent repo of the worktree | `/Users/rpunamiya/Desktop/GEAR/EgoVerse-graph` (checked out on an *older* branch — don't work there) |
| Data-generation repo | `/Users/rpunamiya/Desktop/GEAR/EgoVerse` on `sim/dedup-gen-pipeline` |
| Data-gen README | pushed to `origin/sim/data-gen-readme` @ `f952ca0d` |
| WIP engage controller (**broken, 0/10**) | `/Users/rpunamiya/Desktop/GEAR/sim_run/engage/engage_controller.py` |
| Bar charts from the earlier sweep | `/Users/rpunamiya/Desktop/GEAR/arc_eval_out/arc_{variants_overview,axis_sweeps,outcome_breakdown}.png` |

`git worktree list` from `EgoVerse-graph` shows the wt_codec worktree. The codec code
is **not** in `~/Desktop/GEAR/EgoVerse` — that repo owns Tsimulation, not the policy.
The eval jobs clone **both** repos because neither owns both halves.

### Key source files (all in `wt_codec`)

| File | Role |
|---|---|
| `egomimic/rldb/zarr/planar_arc.py` | Tokenizers: `TokenizeUSocketArcVelocityStacked`, `…Carry`, `…Duration`. `PLANAR_ARC_STACKED_DIM = 6`. |
| `egomimic/pipeline/stages_arc.py` | Detokenizers: `ArcDetokenizeStackedStage`, `ArcDetokenizeDurationStage`. |
| `egomimic/rldb/embodiment/usocket_arc_velocity.py` | `get_*_transform_list` factories + native decoders used at eval. |
| `egomimic/models/denoising_nets.py` | `ConditionalUnet1D` (standard, 40.3M) and `PaperConditionalUnet1D` (paper, 247.8M). Time-axis padding fix + `input_groups` live here. |
| `egomimic/eval/sim_rollout_planar_eval.py` | **NON-PROTOCOL** closed-loop rollout harness. Read §5.6 before touching. |
| `osmo/usocket_codec_sweep.yaml` | Training launcher. |
| `osmo/usocket_codec_rollout.yaml` | Eval launcher. |
| `osmo/usocket_codec_grid.yaml` | 896-cell replay grid (no training). |
| `osmo/r2_inventory.yaml` | Counts collected episodes per embodiment in R2. |
| `tests/test_usocket_arc_velocity_stacked.py`, `tests/test_wandb_run_id_length.py`, `tests/test_semantic_blocks_partition.py` | Regression guards for three bugs that each cost a full job. |

---

## 2. What the ARC codec actually is

An action chunk of 80 raw frames is re-expressed as a fixed-length token whose rows are
spaced by **arc length**, not time — so the token says "where the path goes," and a
separate channel says "how fast."

Three layouts were tried:

| Layout | Shape | Row contents |
|---|---|---|
| **row-split** (original) | `[2M, 5]` | rows `[:M]` = `[x, y, 0, 0, v_xy]`; rows `[M:]` = `[0, 0, cos, sin, ω]` |
| **stacked** (user-directed) | `[M, 6]` | `[x, y, v_xy, cos, sin, ω]` |
| **duration** | `[M, 6]` | `[x, y, dt_translation, cos, sin, dt_rotation]` |

Translation and rotation are resampled on **independent clocks**:
`translation_arc = cumsum(|Δxy|)` capped at `D`, `angle_arc = cumsum(|Δθ|)` capped at `R`.
Both produce `M` samples, so they concatenate on the action dim — this is why the
row-split layout was unnecessary, and the user was right to reject it.

**Budget parameters:** `D` = translation budget (px), `M` = waypoints,
`R` = angular budget in **radians** (`angle_unit() == math.radians`; the config value
`0.4537856055185257` is 26°).

The duration variant came from `~/Downloads/shape_clock_action_codec.html` — same
progress/distance parameterization, but each row carries the *time to traverse* that
segment instead of the velocity at it.

---

## 3. Results

All numbers are **NON-PROTOCOL**: a repo-local harness following the canonical
protocol's substance (p99 frame budget = 318, PEAK coverage, 40 level-0 rollouts on
seeds 0–39, sim_v2, `replan_every=8`, `chunk_start=0`, 100 sampler steps). The canonical
`bf_eval_par.sbatch` lives on Skynet and there is no registered inference contract for
the ARC family. **Never pool these with canonical results.** The harness stamps every
summary line `NON-PROTOCOL_REPO_LOCAL_ROLLOUT_NOT_COMPARABLE` on purpose.

### 3a. The 260M "paper-DP" comparison — the headline result

Identical training contract (obs horizon 2, `action_target_offset 1`, DDPM-100, EMA
0.9999, batch 32, `PaperConditionalUnet1D` down_dims `[512,1024,2048]` kernel 5,
247.8M denoiser params ≈ **251.6M total**), 240k steps. **Only the action representation differs.**

| Run | mean cov | median cov | SR@0.80 | SR@0.95 | total-failure seeds |
|---|---|---|---|---|---|
| `planar_v2_usocket_arc_dur_paper_D80_M56_R26deg` | **0.590** | 0.656 | 27.5% | 12.5% | 7/40 |
| `planar_v2_usocket_arc_stk_paper_D80_M56_R26deg` | 0.568 | 0.677 | **32.5%** | **22.5%** | 10/40 |
| `planar_v2_usocket_dp_paper` (baseline, no ARC) | 0.581 | 0.678 | 30.0% | **22.5%** | 8/40 |

Paired t-test on peak coverage, same seeds:

| Pair | Δ mean | p |
|---|---|---|
| duration − stacked | +0.022 | 0.78 |
| duration − DP | +0.009 | 0.89 |
| stacked − DP | −0.012 | 0.87 |

**Read this as a null result.** At 260M, arc-length tokenization neither helps nor hurts.
Duration edges out on mean coverage; stacked edges out on strict success; neither margin
is anywhere near significance at n=40. Workflow: `usevalpaper4-1` (COMPLETED).

### 3b. Duration × M sweep at the standard recipe

Workflow `usevaldurM-1` (COMPLETED). This tested the user's hypothesis that
*"duration with stacked M16 config may work even better."*

| Run | mean cov | SR@0.80 | SR@0.95 |
|---|---|---|---|
| `…arc_duration_D80_M16_R26deg` | 0.476 | 12.5% | 5.0% |
| `…arc_duration_D80_M36_R26deg` | 0.498 | 20.0% | 10.0% |
| `…arc_duration_D80_M56_R26deg` (earlier run) | 0.469 | 17.5% | — |

**Hypothesis not supported.** M16 helped the *velocity* codec (0.519 vs 0.408) but not
the *duration* codec — M16 is the worst duration cell. The M-benefit does not transfer.
M16 − M36 is `Δ = −0.022, p = 0.75`.

### 3c. Earlier standard-recipe sweep (for context)

Standard recipe = `ConditionalUnet1D` `[256,512,1024]` kernel 3, DDIM-100, obs horizon 1,
**40.3M denoiser params** — roughly 6× smaller than the paper recipe. SR column is SR@0.80.

| Variant | mean cov | SR |
|---|---|---|
| DP standard (no ARC) | 0.557 | 15.0% |
| stacked **M16** D80 R26 | 0.519 | 27.5% |
| duration M56 D80 R26 | 0.469 | 17.5% |
| carry (shared clock) | 0.415 | 25.0% |
| **row-split baseline** M56 D80 R26 | 0.408 | 22.5% |
| stacked, grouped conv (`stackg2`) | 0.288 | — |

Only M16-vs-M56 cleared `p < 0.05` (+0.190, p=0.026). Duration-vs-its-own-cell was
+0.140 (p=0.098) — suggestive, and it is exactly this that the 260M run failed to confirm.

### 3d. Things that were tested and settled

- **Replay grid (896 cells).** `GRID_RANKING.csv`, sha256 `f456cda…`. **6 cells PASS**, not
  11 as an earlier buggy run reported. Rank 1 = D80/M56/R26 — which is why that is the
  baseline cell everywhere above. Best replay coverage 96.6%.
- **The ~70-point replay-to-rollout gap.** 96.6% replay vs 22.5% rollout on the same
  cell. The codec can represent the trajectories; the policy cannot produce them. This
  is the single most important number in the study.
- **Replan cadence is an inverted U peaking at 8.** `replan_every` ∈ {1,2,4,8,16,0}.
  Both extremes are worse. Explanation: `decoded[0]` is *exactly the current pose*, so at
  `replan_every=1` the policy re-issues a zero-displacement first step every frame and
  stalls. `replan_every=0` (full open-loop chunk) drifts.
- **`chunk_start=1` hypothesis — disproved.** Skipping the redundant first waypoint made
  things *worse* (0.332 / 17.5% vs 0.408 / 22.5%). Baseline stays `chunk_start=0`.
- **Grouped input convolution (`input_groups`) hurt badly** (0.288). The idea was to stop
  layer 1 blending the translation and rotation streams. It does not pay for itself.
- **Episode sets are byte-identical across the four compared configs** (2970 train / 29 val,
  seed 42, matching sha256 guards). The apples-to-apples check the user asked for passed.

---

## 4. Infrastructure runbook

### 4.1 Launching an eval

```bash
cd /Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec
osmo workflow submit osmo/usocket_codec_rollout.yaml \
  --set job_name=<unique-name> \
  --set branch=codec-replay-rotfix \
  --set sim_commit=<EgoVerse sha> \
  --set experiments="<exp1> <exp2>" \
  --set ckpt_jobs="<training job names>" \
  --set n_episodes=40 --set replan_every=8 --set chunk_start=0
```

The job runs a **1-episode smoke first**, then the 40-episode rollout. The smoke exists
because three separate config bugs only surfaced at model-construction time, hours in.
**Never remove it.**

### 4.2 Reading results

```bash
osmo workflow query <job>                          # Status
osmo workflow logs <job> -t rollout > /tmp/x.log   # full log
grep "SUMMARY" /tmp/x.log                          # the JSON result lines
```

Each `[sim] SUMMARY` line is JSON with `peak_coverage_mean`, `SR@0.80`, `SR@0.95`, and the
per-episode `ep_coverages` array. **Keep the arrays** — they let you run paired tests later
(same seeds), which is far more powerful than comparing means.

Logs are only readable **after** the workflow terminates.

### 4.3 Paths

| Thing | Path |
|---|---|
| Training corpus (u_socket, 2999 clean episodes) | `s3://rldb/processed_v3/pushshapes_sim/u_socket_3000_v2/` |
| Training checkpoints | `s3://rldb/staged/usocket_codec/<ckpt_job>/<exp_name>/checkpoints/last.ckpt` |
| Norm stats (must match the ckpt) | `s3://rldb/staged/usocket_codec/<ckpt_job>/<exp_name>/norm_stats/norm_stats.json` |
| Rollout outputs | `s3://rldb/staged/usocket_codec_rollout/<job_name>/` |
| Frame budget | `/workspace/budgets/u_socket_3000_v2_clean_p99.json` (budget = 318) |

The corpus is the raw 3000 minus `episode_T_u_socket_obs0_000270`. It is **not** under
`staged/Tsim_v2/` — an early job wasted hours on that wrong prefix.

`s3://rldb` is **Cloudflare R2**, endpoint
`https://1beb594fb475d71c4420f7b693524e19.r2.cloudflarestorage.com`. The R2 keys are
32 chars and are **different** from the AWS keys; if you try to list R2 with the AWS key
you get `Credential access key has length 20, should be 32`. Both come from the osmo
credential store (`egoverse-aws`), so this only bites you when running locally.

The three 260M checkpoints are all at **epoch 2399 / global_step 240000**, verified by sha256
in the eval log.

### 4.4 OSMO pool notes

- GPU counts must be **powers of two**. A request for 6 is rejected.
- `groot-h100-01` demands `num_gpu=8`; `groot-l40s-01` rejects it.
- Whole-node asks queue indefinitely (the scheduler reported 227/230 nodes short).
- `--dry-run` skips resource assertions, so it will not catch a bad size.
- The log rate limiter **drops lines**. `s5cmd` staging output floods it and eats the real
  error. Commit `b2c1040` redirects s5cmd to a file for exactly this reason.
- Preemption and dead nodes are routine: `l40s-03` preempted three jobs at once
  (`Killing: Stopping container`), `l40s-01` went `NodeNotReady → TaintManagerEviction`.
  **Split independent jobs across pools** — that is why `usevalpaper4-1` and `usevaldurM-1`
  ran on different pools and both survived.
- Session-wide `403 Forbidden` on *every* endpoint (including `pool list`) means the osmo
  token expired. Fix: user runs `osmo login`. It is not a job-specific failure.

---

## 5. Landmines — read before changing anything

These each cost hours. They are listed in descending order of how quietly they fail.

### 5.1 Hydra validates on construction, and construction is late
A malformed config does not fail at submit. It fails after cloning, installing, staging
40GB, and building the model. Always instantiate locally before submitting.

### 5.2 `**_kwargs` silently ate `rotation_distance_unit`
`get_usocket_arc_velocity_transform_list` absorbed misrouted kwargs through `**_kwargs`,
leaving `rotation_distance_unit=None`, which **uncaps the angular budget**. Every R cell
tokenized identically and the whole angular sweep was a null experiment. Fixed by wiring
it explicitly in each experiment config:

```yaml
data:
  train_datasets:
    pushshapes_sim_u_socket:
      resolver:
        transform_list:
          dt: ${planar.arc_dt}
          rotation_distance_unit: ${planar.arc_rotation_distance}
  valid_datasets:   # same block — forgetting this silently skews validation
    ...
```

Verified after the fix: R26 vs R18 tokens differ by 1.54. **If you add a new ARC config,
copy this block or your R value does nothing.**

### 5.3 `PipelineAlgo` exposes the graph as `.pipeline.stages`, not `.stages`
`getattr(model, "stages", None)` returns `None`. The eval harness silently kept
`n_obs = 1` and was about to score an obs-horizon-2 checkpoint as single-frame. It now
**raises** rather than defaulting — a wrong observation horizon yields a plausible wrong
number, not a crash, which is the worst failure mode there is.

### 5.4 `FusedObsEncoder` is asymmetric in the obs axis
At `n_obs == 1` it does **not** collapse the obs axis; at `n_obs > 1` it reshapes
`(B,T,…) → (B*T,…)`. So the caller must omit the axis entirely when `n_obs == 1` and stack
it when `n_obs > 1`. My first local repro hand-performed the reshape the real code skips and
therefore falsely "passed" — reproduce against the real call path, not your model of it.

### 5.5 Norm stats are per-`(row, channel)` and horizon-shaped
Shape `[112,5]` for row-split, `[56,6]` for stacked/duration, and `(2,6)` when
`observation_horizon=2`. Mismatches surface as
`RuntimeError: shape '[1, 6]' is invalid for input of size 12`.
Also: `norm_mode` is **quantile**, not mean/std. Assuming mean/std produced four confident
wrong verdicts earlier in this project.

### 5.6 `semantic_blocks` is hardcoded width-5
`eval_planar_v2` assumes `[x,y][cos,sin][grip]`. Width-6 tokens need
`[[0,2],[2,3],[3,5],[5,6]]` in the config, and the EnergyScore validator requires the
blocks to partition **every** channel exactly once. Then the *rollout* evaluator rejected
the same key with `unexpected keyword argument` — fixed by accepting-and-discarding it
explicitly (**not** via `**kwargs`, which would re-open 5.2).

### 5.7 wandb `Name` has a 128-char limit
Generated config names hit 130–136 chars and the run died at logger init.
`tests/test_wandb_run_id_length.py` guards this.

### 5.8 Flat S3 wildcard listings truncate at 168,096 objects
This bit **three times** and produced a bogus uniform "125 episodes per embodiment."
Only a hierarchical walk gives true counts. `osmo/r2_inventory.yaml` does it correctly.

### 5.9 `find | head` under `set -o pipefail` exits 141
SIGPIPE kills the launcher when there are more matches than `head` wants (143 checkpoints).
Write to a file instead.

### 5.10 Checkpoints are named `epoch_epoch=2399.ckpt`
Note the doubled `epoch`. Any epoch-sorting regex written against `epoch_2399.ckpt` will
match nothing. **This is the prime suspect for the undiagnosed `arcvid4-1` failure** (§6.3).

### 5.11 Checkpoint discovery is a fuzzy glob
The launcher resolves `find "$CK" -path "*${EXP}*" -name last.ckpt | head -1`. Note that
`planar_v2_usocket_dp_paper` matched the directory `planar_v2_usocket_dp_paper_h16`. That
was correct here, but a substring collision would silently evaluate the wrong model.

### 5.12 Every `Eval` object must supply `override_dict`
Omitting it fails at construction. Caught by the 1-episode smoke.

---

## 6. Open threads, in priority order

### 6.1 Fix data generation ← **do this one**

`Tsimulation/sim_v2/collect/README_DATA_GENERATION.md` (on `origin/sim/data-gen-readme`)
is the full brief, written for exactly this handover. Summary of the three defects, all
measured against a live env rather than inferred:

1. **The engage channel is never used.** `pose_collect` never writes `a[3]`, so all six
   grip-capable agents were collected with jaws permanently open — measured grip
   `min = max = 0.000` over 400 steps. Every "grasp" is a shove.
2. **Orientation is never commanded.** So no episode demonstrates using the u_socket latch
   (pivot + gear constraints, `agent.socket_latched`) or the gripper as a gripper.
3. **The motion is jerky.** The user's words: *"the motion should look much smoother, rn it
   is so jerky."*

The user's framing: *"for usocket or gripper, u can't cheat by using the gripper to push,
you need to use the gripper to engage the T."*

**Scope:** 27,045 of 33,594 episodes across the 9 embodiments with ≥3 DOF, plus 5
defective `gap-us-*` control-gap collections.

**Where it stalled:** `/Users/rpunamiya/Desktop/GEAR/sim_run/engage/engage_controller.py`
fixes smoothness and aiming but **engages 0/10**. It is not collection-ready.
**Recommended path (untried):** reuse `pose_controller.py`'s proven routing to reach the
pre-engage pose, then hand off to aim-and-insert for the final ~60 units. The existing
controller tries to do both with one policy and does neither.

The README has a pre-collection validation table — run it before collecting anything, or
you will generate another 27k unusable episodes.

### 6.2 Decide whether the codec study continues

Given §3a is a clean null at 260M, the honest read is: **stop adding codec cells.** The
representation is not what is limiting the policy; the ~70-point replay-to-rollout gap is.
If the user wants to keep going, the two defensible moves are:

- **Re-run the top variants at higher n.** Several comparisons sit at `p ≈ 0.1`. n=40 cannot
  resolve them. n=200 could. Cheap, and it either kills or confirms the duration effect.
- **Retrain on fixed data** once §6.1 lands, and re-run the same three-way 260M comparison.
  That is the experiment that would actually mean something.

### 6.3 `arcvid4-1` — failed 3×, undiagnosed

Uploads validation videos for the last 5 checkpoints × 6 runs (abc/fold-clothes/cotrain/hpt).
Known suspect: the `epoch_epoch=` naming in §5.10. Launcher is `osmo/abc_arc_videos.yaml`.
Low priority.

### 6.4 Add checkpoint-resume to the training launcher

Checkpoints already sync to R2 every ~7 min, but the launcher cannot resume from them, so
`uscodecdurM3` lost ~4h to a preemption it could have survived. Small, high-value.

---

## 7. Conventions the user has established

- **Report results, not status.** The user asked for status many times because results were
  slow; the standing instruction is *"let me know the updated results whenever the training
  and evals finish."* Lead with numbers.
- **Match episode counts across arms** so comparisons are apples-to-apples. Verify with
  sha256 over the resolved episode list, not by trusting the config.
- **Number of demonstrations and their distribution matter more than horizon.** The user cut
  off a horizon/padding investigation with exactly this.
- **Tokens are `[M, D]`, always.** The user rejected the row-split layout outright:
  *"they should be stacked M, D always… yes rotation and translation get interpolated
  separately but they both produce M length vectors so they can concat along action dim."*
  Do not reintroduce row-splitting.
- Non-protocol results are labeled as such in the artifact itself, not just in prose.

---

## 8. Housekeeping

- **Rotate the AWS access key** `AKIA…IUO7` (osmo credential `egoverse-aws`). The user
  authorized in-session use and said they would rotate it. Also rotate the wandb key that
  was pasted in-session.
- Branches `algo/causal-articulated-20260910` and `sim/articulated-demos-20260909`, and the
  `causal-artic-*` / `articulated-2026*` workflows, are **a different workstream** — not
  part of this handover. Don't assume they're yours.
