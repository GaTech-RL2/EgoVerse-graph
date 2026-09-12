# PushShapes eval protocol — sim_v2

**The eval protocol for models trained on `*_v2` (sim_v2) datasets.**
Supersedes `DP_SMALL_CIRCLE_SIM_V2_EVAL_PROTOCOL.md`.

**Horizon revision 3 (2026-08-12): the budget statistic is `p99`, not `max`.**
The horizon is still data-derived (revision 2), but `max` proved outlier-fragile:
ONE anomalous episode in `small_circle_3000_v2sub` (2255 frames vs a
second-longest of 442) inflated that embodiment's entire budget table ~5x and
silently invalidated every circle-vs-small-circle comparison. Numbers are **not
comparable across revisions** — see [Horizon revisions](#6-horizon-revisions).

**Obstacle target-side revision 4 (2026-08-27):** levels 1-30 use the pinned
five-seed-per-level bank and reviewed simulator identity in §7.

> Naming: "v1 / v2 / v3" in this project refer to **simulator eras** (geometry +
> physics), not to protocol versions. This document is the protocol *for sim_v2*;
> changes to it are numbered as **horizon revisions** so the two never collide.

---

## 1. What changed, and why

Revision 1 gave every rollout a flat `--max-steps 1800` (60 s at 30 Hz).
Revision 2 derives the budget from the data the model was trained on:

```
budget(dataset, level) = ceil(1.1 * p99(total_frames | that level's episodes))
```

The budget is set by the **99th-percentile demonstration at that level**, plus
10% headroom. (Revision 2 used `max`; see revision 3 below for why that broke.) "Success" therefore means *solved it in roughly the time the
demonstrations took* rather than *solved it within an arbitrary horizon*.

Two concrete reasons this replaces the constant:

1. **1800 was 3–4× too generous.** On `circle_4500_plus_gen_v2` no level needs
   more than 668 frames (range 405–668). Most of every eval was spent on
   episodes that had already peaked.
2. **A constant cannot serve embodiment speed variants.** The 0.25× pusher
   dataset has ~4× longer episodes; its level-0 budget is ~2006, i.e. **above
   1800**. Under revision 1 a 0.25×-trained policy is truncated before it finishes and
   scores artificially low. The budget must travel with the dataset.

---

## 2. Computing the budget

Budgets are a **recorded artifact**, never computed inline:

```bash
python /coc/flash7/paphiwetsa3/scripts/eval/episode_budget.py \
    /coc/flash7/paphiwetsa3/datasets/Tsim_v2/<dataset>
# -> /coc/flash7/paphiwetsa3/scripts/eval/budgets/<dataset>.json
```

```json
{ "dataset": "...", "statistic": "p99", "multiplier": 1.1,
  "formula": "ceil(multiplier * p99(total_frames | level))",
  "by_level": { "0": { "n": 4286, "max": 456, "p95": 306,
                       "budget": 502, "max_over_p95": 1.49 } },
  "content_sha256": "..." }
```

Rules:

- **Use the model's OWN training dataset.** That is what makes speed variants
  evaluable. It also means two models trained on different data are on different
  clocks — so the budget must be reported with every number (§5).
- **`total_frames`, not array length.** Episode arrays are chunk-padded (a
  580-frame episode lives in a 600-row array); the attr is the true length.
- **Regenerate when the dataset changes.** The JSON records a `content_sha256`;
  cite it in the ledger row.
- **`p99`, not `max` — this is revision 3 and it is mandatory.** `max` lets a
  single pathological demo set the horizon for a whole dataset. It did exactly
  that: `small_circle_3000_v2sub` episode `..._obs0_000274.zarr` runs 2255 frames
  where the second-longest is 442 and the median is 205 — 1 episode in 3000, and
  it alone produced a level-0 budget of 2481 (vs 372 under p99) and a derive
  ratio of 4.9452 (vs 0.9388). Every small-circle number measured under `max` got
  ~8.5x headroom over its own p95 while circle got 1.64x.
- **`max_over_p95` is still the health flag.** A ratio above ~1.5 means one
  episode is driving that level. Under p99 this is diagnostic rather than
  load-bearing, but a very high value still signals bad data worth inspecting.
- **Never mix statistics within one table**, and record the statistic in every
  ledger row (§5).

Reference budgets, `circle_4500_plus_gen_v2` (statistic=p99, multiplier=1.1):

| level | budget | | level | budget |
|---|---|---|---|---|
| 0 (in-domain) | 502 | | 10 (longest) | 668 |
| 1–6 | 446–615 | | 23 (shortest) | 405 |

---

## 2b. Levels the embodiment has no data for (DERIVED budgets)

Some embodiments have no episodes at some levels — `u_socket_3000_v2_clean` is
**level 0 only** (2999 episodes, all obstacle-free), so there is no U-Socket
level-L statistic to use. Do **not** fall back to the other embodiment's budget
outright: that hands it ~20% more time than its own demonstrations ever needed,
a free pass on the exact quantity being measured.

Scale instead by a per-embodiment speed factor measured where both DO have data:

```
ratio        = stat(A | obs0) / stat(B | obs0)
budget(A, L) = ceil( multiplier * stat(B | L) * ratio )
```

A = target embodiment (missing level L), B = source embodiment that has it.

```bash
python /coc/flash7/paphiwetsa3/scripts/eval/derive_budget.py \
    --target <datasets>/u_socket_3000_v2_clean \
    --source <datasets>/circle_4500_plus_gen_v2
```

Worked, current values:

```
ratio = p99(u_socket|obs0) / p99(circle|obs0)     # was max/max in revision 2
budget(u_socket, 10) = ceil(1.1 * 607 * 0.8311) = 555      # circle's is 668
```

Rules:

- **Derived rows are marked `"derived": true`** in the JSON, with
  `source_embodiment`, `source_stat`, `ratio` and `ratio_basis`. A derived budget
  is an ESTIMATE and must never be reported as measured; say so in the ledger row.
- **The ratio is computed at level 0**, the only level both embodiments share.
- **Source precedence:** use the target embodiment's real data when it exists.
  Otherwise use a source embodiment with real data for that level from the same
  experiment family. For Flow Transfer, derive missing U-Socket obstacle rows
  from the paired ChainGripper obstacle data, with the ratio anchored on their
  clean level-0 data. Historical circle-derived budgets remain valid only for
  the historical circle/U-Socket comparison that created them.
- **Regenerate when either dataset changes** — the ratio depends on both.

### Assumption, and when it breaks

The premise is that one in-domain ratio transfers to every obstacle level, i.e.
the speed factor is a property of the embodiment and not of the level. This is
plausible but **untested**, and obstacles are where it is most likely to fail:
u_socket is 3-DOF with an orientation to manage, so a narrow gate may cost it
disproportionately more than it costs a circle. If so 0.8311 underestimates, the
budget is too tight, and the result is false failures.

Treat an unexpectedly low derived-level score as a possible budget artifact
before treating it as a policy result — re-run that level with the source
embodiment's unscaled budget and see if the number moves.

The `max`-based ratio is also outlier-on-outlier: both terms are single episodes.
The p95-based ratio is 0.7977 (~4% tighter); it is recorded in the JSON as
`ratio_p95_alternative`. Do not mix statistics within one table.

---

## 2c. Action space — cursor vs pusher

`actions` do **not** mean the same thing in every dataset, and the env accepts
both, so a mismatch is silent rather than an error.

| convention | `actions[t]` is | typical distance from the pusher |
|---|---|---|
| `cursor` | the commanded target the human's cursor was at | often 100+ world units ahead |
| `pusher` | the pusher pose actually ACHIEVED after the step | ≤ one step's travel (e.g. 1.67 units at 0.25x) |

A policy trained on one convention has learned a different output distribution
from a policy trained on the other. **They are not comparable**, exactly as
across horizon revisions.

Current datasets:

| dataset | action_space |
|---|---|
| `circle_4500_plus_gen_v2` | `cursor` |
| `u_socket_3000_v2_clean` | `cursor` |
| `circle4500gen_v2_pusher0.25x` | **`pusher`** |

Rules:

- The budget JSON records `action_space`, detected from the dataset's episode
  attrs (absent ⇒ `cursor`, since the field postdates those collections).
- The launcher echoes `[protocol] action_space=...` into the log and appends
  `_as<space>` to its completion line.
- **Assert it when you know what the model expects:**
  `EXPECT_ACTION_SPACE=pusher sbatch ... bf_eval_par.sbatch ...` — the launcher
  exits 3 on mismatch rather than producing a plausible wrong number.
- Report `action_space` in the ledger row alongside the budget.

⚠️ The policy eval itself (`ckpt_loading` / `eval_sim.py`) has **no** action-space
concept — it feeds predicted actions straight to `env.step`. The guard above is
provenance and a caller-supplied assertion, not enforcement inside the eval. A
stricter version would have the model declare its convention and the eval check
it; not built.

---

## 3. Running an eval

Reuse the launcher — the protocol lives in it, not in any config
(every repo default is a smoke value):

The canonical entry point is:

```text
/coc/flash7/paphiwetsa3/scripts/eval/bf_eval_par.sbatch
```

For PACE execution, transfer this exact launcher, protocol, and every hash-pinned
auxiliary contract they reference as one audited mirror, verify every source
SHA-256 value, and point `SCRIPT_DIR` at that mirror.
`PY`, `RUNTIME_CWD`, `EVAL_SCRATCH_ROOT`, and `OUT_ROOT` may be rebound to
PACE-local absolute paths. Scheduler directives may be overridden only at
submission for an authorized PACE account/partition/QoS/GPU. This is the same
canonical launcher, not permission to fork its evaluation logic; the launch log
must record the mirrored launcher/protocol hashes and all PACE-local artifact
paths. Dataset/config path-only rebinding must preserve and record the original
artifact hashes and semantic dataset content identity.
The Paper262M post-step registry currently binds exact Skynet config,
checkpoint, dataset, generator, and evidence paths. Those two rows are not PACE
portable until a reviewed contract revision records the original and portable
identities explicitly; do not weaken or bypass the match in a copied launcher.
If the pinned PACE environment lacks a simulator dependency, install it into an
immutable task-local directory and expose only that directory through
`EXTRA_PYTHONPATH`; never mutate a shared training environment. Record the
directory, package versions, and installation manifest in task provenance.

Its positional arguments are `NAME SNAPSHOT_CKPT CFG_A BUDGET_A [BUDGET_B]
[LEVEL]`. `LEVEL` defaults to 0. Pair mode requires explicit `CFG_B` and
`BUDGET_B`; neither silently defaults. The launcher also requires immutable
checkpoint SHA, exact clean policy and simulator Git identities, action space,
and coherent embodiment tuples through environment variables documented in its
header.

Before every real launch, probe every live authorized compatible allocation
with Slurm test-only using the exact launcher, task shape, exclusions, GPU,
CPU, memory, and wall time. The launch environment must bind the probe time,
candidate results, estimated start, selection reason, and selected
account/partition/QOS/GPU/count/CPU/memory/wall-time fields. The canonical
launcher fails closed when these are absent on a real job, verifies the Slurm
account, partition, QOS, and GPU count exposed to the job, and records all of
them in `launch.log`.

Array shape is protocol state:

- level 0 single embodiment: tasks `0-3`, 10 seeds each = 40 rollouts;
- level 0 paired embodiments: tasks `0-7`, 40 rollouts per embodiment;
- obstacle level 1-30 single embodiment: task `0`, five explicit seeds;
- obstacle level 1-30 paired embodiments: tasks `0,4`, five explicit seeds per
  embodiment.

The canonical launcher accepts `SAMPLER_INFERENCE_STEPS=<positive integer>`.
It applies the override after the checkpoint-pinned model and all tensors load
strictly, records the checkpoint default and effective value in the launch log,
and requires exactly one sampler stage exposing `num_inference_steps`.

### Paper DP H16 post-step compatibility — action-alignment revision 1 (2026-09-02)

The registered frozen Paper262M Paper-DP run bundles were trained from source episodes
whose stored rows are post-step: the generator calls `env.step(action[t])` and
then persists the returned observation/state beside that same `action[t]`.
With observation horizon 2 and the configured training target offset 1, decoded
token 0 is therefore retrospective relative to the latest observation. The
first deployable action is token 1.

The only approved compatibility identities are enumerated in:

```text
/coc/flash7/paphiwetsa3/scripts/eval/contracts/paper_dp_poststep_exec1_rev2.json
SHA-256 e84e3a5ab36da6440474c4ac7ad6314fee991ff8363f5b147a3e8158941983c3
```

The registry contains the final ChainGripper and U-Socket runs from both the
20260901 source bundle and the audited `1b4d2ee7` 20260902 source bundle. Each
row is bound to its exact checkpoint, resolved config, run-bundle manifest,
dataset inventory, metadata, generator, and simulator identities. For every
row the canonical invocation must set:

```text
ACTION_CHUNK_START_INDEX=1
REPLAN_EVERY=8
SAMPLER_INFERENCE_STEPS=100
USE_EMA=1
```

The model still predicts H16, but the evaluator executes decoded indices
`[1,9)` (tokens 1 through 8), then replans. The launcher must record action
alignment revision 1, start 1, exclusive stop 9, execution horizon 8, and the
contract row/hash. It fails closed if the explicit start is absent, if any
identity differs, or if a non-registered model requests a nonzero start. Do not
infer this setting from Paper-DP architecture alone. Flow Transfer Latent H16
and every unregistered model retain start index 0.

Complete level-0 seeds 0-39 from these exact registry rows use the distinct
comparability label `CANONICAL_PROTOCOL_ACTION_ALIGNMENT_REV1`; they are
canonical only within action-alignment revision 1 and must not be silently
pooled with other revisions. One-episode engineering smokes remain
`NON-PROTOCOL_DIAGNOSTIC_NOT_COMPARABLE`. The earlier 40+40 Paper262M outputs
that executed `[0,8)` are **INVALID/SUPERSEDED** by this contract and must not be
pooled, cited as canonical, or compared as if only model quality changed. The
post-step training-data caveat remains part of every report; a future clean
training run should repair row timing rather than depend on evaluation
compensation.

### Flow Transfer latent H16 inference contract (updated 2026-09-01)

For direct Flow Transfer Latent Dense and UNITE-style latent policies with
`action_horizon=16`, the canonical setting is:

```text
SAMPLER_INFERENCE_STEPS=8
REPLAN_EVERY=8
ACTION_CHUNK_START_INDEX=0
```

The policy predicts all 16 actions, executes actions 0-7, then replans from the
new observation. The sampler count and execution cadence are separate knobs,
but both must resolve to 8 for canonical results. The launcher detects this
model family from the resolved model tree and fails closed if either value is
missing or different. Older Latent Dense checkpoints may embed
`num_inference_steps=16`; that value is retained as training provenance and is
not used as this family's canonical evaluation value. UNITE checkpoints
currently embed the canonical value 8, but the launcher still resolves,
verifies, and records the effective override.

### Action Flow H16 inference contract (added 2026-09-06)

For `ActionFlowModelWrapper` U-Socket policies whose exact resolved model uses
`action_horizon=16` and one `ConditionalVelocityStage` with a checkpoint-pinned
16-step reverse-Euler sampler, the canonical setting is:

```text
SAMPLER_INFERENCE_STEPS=16
REPLAN_EVERY=8
ACTION_CHUNK_START_INDEX=0
```

The evaluator must load the exact checkpoint model subtree strictly and may
only adapt simulator observations and decoded native actions at the boundary;
it must not alter learned parameters. The policy predicts all 16 actions,
executes actions 0-7, then replans. The launcher detects this family from the
resolved wrapper and model graph, preserves the checkpoint sampler count, and
fails closed if the execution cadence or chunk start differs.

The canonical teacher-forced overlay launcher permits a paired sampler-step
ablation only when `DIAGNOSTIC_SAMPLER_STEP_PROBE=1` and an explicit positive
non-8 `SAMPLER_INFERENCE_STEPS` is supplied. It keeps the checkpoint, clean
validation split and episode order, normalization, RNG seed, precision,
prediction horizon, and all other inference state fixed. Such output must be
labeled `NON-PROTOCOL_SAMPLER_STEP_PROBE_NOT_COMPARABLE`; it is an offline
engineering diagnostic and not a canonical rollout or policy score. This
overlay exception does not by itself relax the rollout launcher's J=8
requirement.

For a UNITE checkpoint, the same canonical teacher-forced launcher exposes the
reusable offline diagnostic bundle with `UNITE_DIAGNOSTICS=1`. It is valid only
for a resolved `UniteLatentPolicy` and records the clean-latent reconstruction,
the generated action, clean and generated latent tokens, every intermediate
latent state, the decoded action after every sampler step, and a
fixed-observation noise-diversity batch
(`UNITE_DIVERSITY_SAMPLES=32` by default). The diagnostic RNG seed, sample cap,
UMAP setting, J, checkpoint/config/source/split/normalization identity, action
contract, precision, and horizon are part of the immutable request key. The
prediction artifact and UNITE diagnostic artifact are a paired cache: a partial
pair is a hard error. Reconstruction/generation overlays, separate
decoded-action and true-latent denoising animations, noise diversity, and the
shared-latent PCA/UMAP maps must be rendered from this artifact without another
model inference pass. The latent animation fits one PCA basis over the clean
reference plus the complete trace and keeps its PCA axes and residual-heatmap
scale fixed across every frame; never label decoded action as latent. PCA/UMAP
reuse the existing latent-visualization reduction backend and color U-Socket
versus ChainGripper while marking clean versus generated tokens separately.
The aggregate shared-latent map must also render a moving-point PCA/UMAP video
with every selected token from both embodiments. Fit each reducer once over the
clean references plus all stored denoising states, keep axes and camera fixed,
keep clean points stationary, and preserve each moving token's identity through
one exact frame per sampler step. Never refit PCA or UMAP per frame. These are
offline diagnostics, not simulator rollouts or canonical policy scores.

The canonical teacher-forced launcher also exposes a separate visual-encoder
probe with `VISUAL_KEYPOINT_DIAGNOSTICS=1`. It requires exactly one configured
`VisualCore` using `pool_type=spatial_softmax`. Hooks record the exact cropped
image entering the ResNet backbone, every SpatialSoftmax probability map, and
the expected `(x,y)` coordinate of every keypoint without replaying or replacing
model computation. The immutable artifact is paired with the prediction request
and must validate before offline rendering. Render moving-point videos with
fixed keypoint identity and attention scale plus representative all-channel
heatmap sheets and concentration/redundancy metrics. Cap stored frames with the
explicit positive `VISUAL_KEYPOINT_MAX_SAMPLES`. This diagnostic may be run
without `UNITE_DIAGNOSTICS`; it is not a simulator rollout or policy score and
must be labeled
`NON-PROTOCOL_VISUAL_KEYPOINT_DIAGNOSTIC_NOT_COMPARABLE`.

The canonical rollout launcher separately permits a non-8 sampler-step probe
only with both `DIAGNOSTIC_SAMPLER_STEP_ROLLOUT=1` and
`DIAGNOSTIC_SINGLE_EPISODE=1`. It is restricted to one embodiment, level 0,
`EVAL_SCOPE=single`, and exactly one rollout (`NEPS=1`). The policy still
predicts 16 actions and `REPLAN_EVERY=8` remains fixed, so the explicit positive
non-8 `SAMPLER_INFERENCE_STEPS` is the only inference-contract change. Chunk
stitching, shifted noise, temporal ensembling, RTC, attention export, and the
replan-cadence probe must all be disabled. Every launch must record
`NON-PROTOCOL_SAMPLER_STEP_ROLLOUT_PROBE_NOT_COMPARABLE`; it is a visual
engineering diagnostic, not a canonical policy score, and it must never be
pooled with or substituted for the canonical J=8 result.

The launcher also exposes an engineering-only seam probe through
`CHUNK_STITCH_WEIGHTS`. It blends the unused tail of the previous chunk into
the head of the newly predicted chunk in decoded native action space; theta is
interpolated on the circle. This probe is currently restricted to
`DIAGNOSTIC_SINGLE_EPISODE=1`, `EVAL_SCOPE=single`, level 0, and exactly one
rollout. The launch log must say
`NON-PROTOCOL_DIAGNOSTIC_NOT_COMPARABLE`. It is for paired visual/jitter
diagnosis only and must never be reported as a canonical policy result.

`CHUNK_SEAM_EXPORT=1` captures the unmodified evidence behind that question at
every replan: the previous unused native-action tail, the raw newly predicted
head, and the head after any optional chunk stitching. It derives position,
circular rotation, grip, velocity, and jerk plots offline from an immutable
artifact. Capture requires the same one-episode level-0 gate plus an explicit
`REPLAN_EVERY`; it cannot be combined with shifted noise, temporal ensembling,
RTC, sampler-step probing, or attention export. It may be paired with
`CHUNK_STITCH_WEIGHTS` because both the raw and stitched heads are retained.
Every launch must record
`NON-PROTOCOL_CHUNK_SEAM_DIAGNOSTIC_NOT_COMPARABLE`; no seam plot is a policy
score.

The launcher separately exposes an engineering-only latent-noise continuity
probe through `ROLLOUT_NOISE_SHIFT_TOKENS`. For H16/replan-8 it must be exactly
8: the next replan's noise prefix is the previous Gaussian-noise tail, while
the remaining eight tokens stay freshly sampled. A complete fresh Gaussian
tensor is always drawn before the prefix replacement so paired control and
probe runs consume identical RNG streams. This probe is inference-only, resets
at every episode boundary, cannot be combined with `CHUNK_STITCH_WEIGHTS`, and
has the same single-episode, level-0, non-protocol restrictions as the seam
probe. It must never be reported as a canonical policy result.

The launcher also exposes proper online temporal ensembling through
`TEMPORAL_ENSEMBLE_DECAY`. Unlike `CHUNK_STITCH_WEIGHTS`, this does not blend a
hand-selected prefix of the previous tail. Every decoded chunk is retained with
its absolute simulator start timestep. At each executed timestep, all still-live
predictions for that exact absolute action are combined using normalized
recency weights

```text
weight(chunk at time t) proportional to
    exp(-TEMPORAL_ENSEMBLE_DECAY * (t - chunk_start_t)).
```

Thus newer closed-loop predictions always receive more weight. XY and optional
grip are averaged linearly; native theta is averaged on the unit circle. State
is reset at every episode boundary. The option requires `REPLAN_EVERY`, cannot
be combined with chunk stitching or shifted noise, and is currently restricted
to the same one-episode level-0 diagnostic gate. It is
`NON-PROTOCOL_DIAGNOSTIC_NOT_COMPARABLE` and must be compared only against an
otherwise identical control.

The launcher exposes decoder-attention inspection through
`DECODER_ATTENTION_EXPORT=1`. It is restricted to
`DIAGNOSTIC_SINGLE_EPISODE=1`, `EVAL_SCOPE=single`, level 0, and exactly one
rollout, and it cannot be combined with chunk stitching, shifted noise,
temporal ensembling, or the replan-cadence probe. The selected model must
contain a `TransformerActionDecoder`. `DECODER_ATTENTION_MAX_CALLS` is a
positive integer and limits capture to the first N decoder calls (normally 1).
Each call stores lossless per-layer/per-head tensors plus self- and
cross-attention PNG grids. Self-attention is action-query index by action-key
index; cross-attention is action-query index by latent-token index. These are
temporal token maps, **not spatial image attention maps**. The implementation
replays only the small attention operators with weight reporting enabled while
returning the unchanged normal decoder output. Artifacts and their hashes are
validated after rollout, and every launch is labeled
`NON-PROTOCOL_DECODER_ATTENTION_NOT_COMPARABLE`; it must never be reported as a
canonical policy score.

The launcher separately exposes denoiser-attention inspection through
`DENOISER_ATTENTION_EXPORT=1`, under the same single-episode, single-embodiment,
level-0 restrictions. It cannot be combined with decoder-attention export,
chunk stitching, shifted noise, temporal ensembling, RTC, or the replan-cadence
probe. The selected model must contain exactly one `CrossTransformer`, and
`DENOISER_ATTENTION_MAX_CALLS` limits capture to the first N denoiser calls
(normally 1). Each call records the flow timestep and stores lossless
per-block/per-head tensors plus rendered grids. Self-attention is noisy
latent-action query index by noisy latent-action key index. Cross-attention is
noisy latent-action query index by conditioning-token index; when the sampler
supplies one conditioning token, this distribution is necessarily a single
column of probability 1 and is not evidence that conditioning is unused. These
are temporal token maps, **not spatial image attention maps**. The diagnostic
replays only the small attention operators to report weights while returning
the unchanged normal denoiser output. Artifacts and hashes are validated after
rollout, and every launch is labeled
`NON-PROTOCOL_DENOISER_ATTENTION_NOT_COMPARABLE`; it must never be reported as a
canonical policy score.

The launcher additionally exposes **Real-Time Chunking (RTC)** as an
engineering-only diagnostic through the all-or-none settings
`RTC_INFERENCE_DELAY`, `RTC_PREFIX_ATTENTION_SCHEDULE`, and
`RTC_MAX_GUIDANCE_WEIGHT`. For the current synchronous PushShapes simulator,
the only reviewed setting is zero simulated inference delay, exponential prefix
attention, and a guidance cap of 5:

```text
RTC_INFERENCE_DELAY=0
RTC_PREFIX_ATTENTION_SCHEDULE=exp
RTC_MAX_GUIDANCE_WEIGHT=5
```

At every H16/replan-8 seam, the unused eight-action tail of the previous chunk
is aligned with the new chunk's first eight actions. Because this policy
denoises latent tokens rather than actions directly, the endpoint estimate is
decoded into the model's normalized action representation; the weighted prefix
error is then propagated to the latent flow by a vector-Jacobian product. This
preserves RTC's flow-guidance schedule, but it is a latent-policy adaptation,
not a claim of exact equivalence to direct-action RTC. It must be launched only
with `DIAGNOSTIC_SINGLE_EPISODE=1`, `EVAL_SCOPE=single`, level 0, `NEPS=1`,
`REPLAN_EVERY=8`, and `SAMPLER_INFERENCE_STEPS=8`. It cannot be combined with
chunk stitching, shifted noise, temporal ensembling, decoder-attention export,
denoiser-attention export, or the replan-cadence probe. Every launch must say
`NON-PROTOCOL_RTC_DIAGNOSTIC_NOT_COMPARABLE` and use an otherwise identical,
RNG-paired control for visual comparison. Guidance events must be retained in
`rtc_diagnostics.json`.

The launcher permits a separate replan-cadence probe through
`DIAGNOSTIC_REPLAN_PROBE=1`. It is restricted to level 0, one embodiment, and
`SAMPLER_INFERENCE_STEPS=8`. A reduced probe uses exactly three consecutive
seeds from one canonical level-0 array shard. A larger probe must use the full
standard level-0 set: four array shards `0-3`, ten seeds per shard, giving exact
seeds `0-39` with no omissions or overlap. Direct Flow Transfer Latent Dense
H16 may use only
`REPLAN_EVERY=8` (matched control), `REPLAN_EVERY=4` (short-cadence probe), or
`REPLAN_EVERY=16` (full predicted-chunk probe). Chunk stitching, shifted noise,
temporal ensembling, decoder-attention export, denoiser-attention export, and
RTC must all be disabled so replan cadence is the only
experimental variable. Every row is
`NON-PROTOCOL_REPLAN_PROBE_NOT_COMPARABLE`; they answer an engineering question
and must never be pooled with the canonical 40-seed result.

For other families, a non-default sampler-step setting remains a **diagnostic
probe** unless that family's section explicitly declares it canonical. Compare
diagnostic variants only with identical checkpoint, seeds, replan cadence,
budgets, simulator, and all other protocol state.

The launcher reads `budget[LEVEL]` from each JSON and fails if the level is
absent. It never falls back to a default. Use `PREFLIGHT_ONLY=1` before a real
submission and retain the emitted provenance log.

Knobs, all pinned explicitly:

| knob | value | why |
|---|---|---|
| `--max-steps` | `budget[level]` | **rev-2 change** — was flat 1800 |
| `--obstacle-level` | `<LEVEL>` | exact evaluated level, logged explicitly |
| `--coverage-threshold 1.01` | > 1.0 | disables the eval's early stop so peak is unbiased |
| `--full-horizon` | on | also ignores the env's own `terminated` at coverage ≥ 0.95 |
| `--max-coverage` | on | record peak, not final |
| `--init-mode seeds` | — | env-generated inits, not dataset replay |
| initialization | level 0: bases 0/10/20/30; obstacle: §7's five explicit seeds | deterministic and logged |
| `--rng-pairing` | on | same inits across embodiments |
| `--action-chunk-start-index` | default 0; exact registered Paper262M rows: 1 | explicit temporal alignment; never family-inferred |

The rollout is bounded by **`max_steps` alone** in this codebase — `eval_sim.py`
runs `for t in range(self.max_steps)` and passes `T_max=self.max_steps` into
`inference_step`. There is **no `action_horizon`** on this path; that knob (and
the `T_eff = min(max_steps, action_horizon)` cap) belongs to the older
EgoVerse-gmm eval and does not apply here. Verified 2026-08-12 — `grep -rn
action_horizon egomimic/eval/` returns nothing.

---

## 4. Metric

- **Coverage** = *peak* coverage over the episode (`max`, not final).
- **SR@X** = fraction of per-episode peaks ≥ X, recomputed from the raw peaks.
  Report **SR@0.80** by default; also report **SR@0.95** when the source data was
  collected with early-stop at 0.95, because every source episode then sits in a
  narrow band just above 0.95 and SR@0.80 flatters any degraded variant.
- Do **not** use the `success_rate` the tool prints — with
  `--coverage-threshold 1.01` that bar is ≈ unreachable by construction.
- Parse per-episode peaks from `[sim] emb<N> ep_coverages: ...` (4 dp).
  Handle scientific notation (`8.06e-05`) — a decimal-only regex reads that as
  `8.06`.

Two lines that record **0 coverage** rather than failing, so grep for them
before trusting an aggregate:

```
[sim] WARNING: non-finite action ... abort 0-cov.
[sim] WATCHDOG: rollout ... exceeded Ns; logging 0-coverage and continuing.
```

---

## 5. Provenance (mandatory)

Every Results Ledger row records, in addition to the numbers:

- **budget used** + `dataset_name` + `statistic` + `multiplier` + `content_sha256`
- **`action_space`** (`cursor` / `pusher`) — see §2c; not comparable across values
- checkpoint path and the **true epoch read from `ckpt["epoch"]`** — snap the
  ckpt first; never label an eval by `last.ckpt`
- simulator name + commit/hash + clean/dirty
- unique output and log path, tagged with model, true checkpoint epoch, run
  timestamp, embodiment, seed base, and level; retain the full launcher preamble
- action alignment revision, contract path/SHA/row when applicable, decoded
  execution start/stop/horizon, replan cadence, and checkpoint/effective sampler
  counts

---

## 6. Horizon revisions

Two axes, kept separate:

**Simulator era** — which sim produced the data and runs the rollout:

| sim era | notes |
|---|---|
| sim_v1 | old obstacle geometry, non-solid pusher |
| **sim_v2** | v2 geometry + pocket-bottom-only socket friction, `solid_pusher=true` — **this document** |

**Horizon revision** — within the sim_v2 protocol:

- **rev 3 (2026-08-12)** — statistic `max` -> `p99`. Triggered by a single
  2255-frame outlier in `small_circle_3000_v2sub` (2nd longest 442) that inflated
  every small-circle budget ~5x. Effect: circle barely moves (405-668 -> 397-655,
  because n=24 per obstacle level makes p99 ~= max there) but circle's level-0
  budget drops 502 -> 397, and small-circle drops 2002-3302 -> 372-615. Any
  cross-embodiment comparison made under rev 2 is confounded and must be re-run.
  Current p99 budget files carry a `_p99` suffix; the rev-2 `max` files are
  retained ONLY to reproduce runs dated 2026-08-12 or earlier.

| rev | horizon | valid for |
|---|---|---|
| rev 1 | flat 1800 steps | 2026-06 → 2026-08-12; historical only |
| rev 2 | per-level data-derived `max` budget | historical 2026-08-12 runs only |
| **rev 3** | **per-level data-derived `p99` budget** | **current** |

A shorter budget can only lower coverage, never raise it, so rev-3 numbers are
no greater than rev-1 numbers for the same model. **Never compare across revisions or sim
eras.** Record both in the ledger row; re-baseline before comparing.

The pre-sim_v2 protocol (400/1200 steps, seeds 0–4, non-solid pusher) is a
different simulator era entirely — those numbers are not comparable to anything
here.

---

## 7. Obstacle target-side protocol (revision 4, 2026-08-27)

### Canonical call name

```text
Protocol call name: PushShapes OEC-56
Short alias: OEC-56
Machine ID: pushshapes_oec56_30x5_rev4
Seed-bank name: OEC-56 30x5 seeds
```

`OEC` means **Obstacle Equal Clearance** and `56` is the shared 56 px
full-target-silhouette clearance from both the arena boundary and physical
obstacle walls. When the user says **"use OEC-56"**, **"run OEC-56"**, or
**"use the OEC-56 seeds"**, interpret that as this exact section and require:

- levels 1-30 with exactly five canonical seeds per level;
- the seed path and SHA-256 immediately below;
- 56 px arena-edge and 56 px physical-obstacle target clearance;
- at least 90 px target depth beyond portals on gate levels;
- the pinned Sim V2 identity and fixed physics in this document;
- the canonical `bf_eval_par.sbatch` launcher; and
- horizon revision 3 with the model dataset's recorded `p99` budget.

Do not silently map `OEC-56` to another seed bank, simulator, launcher, seed
count, clearance, or horizon rule. If any component changes, give the revised
protocol a new call name and retain `OEC-56` for exact reproduction.

Canonical seed bank:

```text
/coc/flash7/paphiwetsa3/scripts/eval/seeds/eval_side_access_seeds_newgeom_150.json
SHA-256: 499560f289a829936afa7f16819f7bd9cb22712e5d366e4fdc6cd39603b2d89d
```

This bank contains exactly **30 levels x 5 seeds = 150 rollouts**. It replaces
the 2026-08-12 top-detour bank for new obstacle evaluations. The canonical
launcher rejects the old filename, a changed file hash, a noncanonical seed
range, a mismatched simulator, and any level that does not resolve to exactly
five entries.

### Target-side clearance rule

For every selected reset:

```text
distance(full rotated target-T silhouette, arena boundary) >= 56 px
distance(full rotated target-T silhouette, physical obstacle walls) >= 56 px
```

This is not a target-center test. The selector builds the exact union of the
two rotated T rectangles and measures its closest arena-edge distance.
Revision 4 uses one equal **56 px full-silhouette clearance** for both kinds of
wall. The arena containment square therefore grows from 384x384 px to
**400x400 px**, allowing targets slightly closer to the arena edge, while the
new internal-wall rule prevents targets from being tucked against an obstacle.
Distance is measured against the physical obstacle silhouette, including its
wall radius, not merely against the obstacle centerline. This bank is shared by
ChainGripper and U-Socket.

Gate levels have a second mandatory rule:

```text
perpendicular_distance(target center, intended portal line) >= 90 px
```

This applies to wide gates 11-14, diagonal gates 23-24, and corner-square gates
25-26. It prevents a technically valid crossing from placing the target near
the portal/arena middle. The target must be visibly established on the far side
while its full T silhouette still satisfies both 56 px wall-clearance rules.
Across the final 40 gate entries, observed target depth is
**90.622-183.783 px**.

### Route validity and selection

Seeds come from the unseen range `[10000, 40000)`, disjoint from the collection
bank (`0-9999`) and the previous evaluation bank (`2000-2999`). For each level,
the selector accumulates 20 candidates that pass both equal 56 px target
clearance rules, the reviewed obstacle-init validator, and—on gate levels—the
90 px portal-depth rule. It then chooses five by balanced route group and
deterministic farthest-point pose sampling.

The reviewed validator requires physically valid T, goal, and open
ChainGripper poses; at least 180 px start-to-goal distance; agent-to-object
line of sight and at most 220 px distance; at least 10 px object/goal obstacle
clearance; and valid level-specific route semantics:

- wall levels require a physical swept-T obstacle crossing;
- finite-gate levels 11-14 and 23-26 require crossing the intended portal;
- levels 25-26 reject the sealed arena-corner pockets explicitly.

The last point matters: a naive `straight_blocked` filter selects impossible
resets on levels 25-26 by placing the object inside a sealed pocket. Those
resets are not difficult tasks; they are unsolvable and are forbidden here.

### Pinned simulator and verification

Obstacle evaluation is pinned to this clean simulator identity:

```text
worktree: /coc/flash7/paphiwetsa3/worktrees/flow-transfer-obstacle-init-20260826
Git HEAD: 7e75d4236baee59c88984f9f6a458790db2dce46
obstacles.py SHA-256: 12b64ad2dcf536306b680154fc3cc4aa894e087541dde5dcca432c93d8a2c4a8
```

The final bank passed:

- 150/150 exact ChainGripper reset replay;
- 150/150 reviewed route-policy revalidation;
- 150/150 U-Socket object/goal pose matches for the same level+seed;
- 150/150 full target silhouettes at or above 56 px arena-edge clearance;
- 150/150 full target silhouettes at or above 56 px physical-obstacle
  clearance;
- observed arena-edge clearance of 56.143-115.475 px;
- observed physical-obstacle clearance of 56.051-128.255 px;
- 40/40 wide, diagonal, and corner-square gate targets at least 90 px beyond
  their intended portal line;
- fixed Sim V2 physics: solid pusher, solid contact guard, and
  socket-inside friction all enabled.

Generator, verifier, verification JSON, and overview PNG are retained in
`/coc/flash7/paphiwetsa3/experiments/obstacle_eval_protocol_20260827/` and the
generator/verifier are installed beside this protocol.

### Launch shape

Use the canonical `bf_eval_par.sbatch`; never make a near-duplicate launcher.
For a single embodiment, one level is one array task (`--array=0`). For a paired
U-Socket/ChainGripper evaluation, use `--array=0,4`. Each task runs the level's
five explicit seeds. Levels remain separate in reporting; do not pool all 30
families into one score.

Budgets still follow horizon revision 3: explicit `p99`, multiplier 1.1, and a
separately recorded budget for every dataset and level. A model without real
obstacle data uses a clearly marked derived budget under §2b.

All prior seed banks, old-geometry copies, three-byte worktree stubs, and
repo-local legacy launchers are historical only and are non-protocol.

---

## 8. Physics flags -- all three ON, always

```
solid_pusher = solid_contact_guard = socket_inside_friction_only = TRUE
```

**In Sim V2 these are CLASS CONSTANTS, not arguments** -- `SOLID_PUSHER`,
`SOLID_CONTACT_GUARD`, `SOCKET_INSIDE_FRICTION_ONLY`, all `True`. The constructor
keeps them out of its public signature on purpose and swallows legacy kwargs:

> "Accept old call sites without letting their flags alter Sim V2. Keeping these
> out of the public signature prevents new code from treating fixed collision
> physics as an episode-level option."

Verified 2026-08-12: a legacy call passing `solid_pusher=False` still yields
`True / True / True`. So on Sim V2 this rule is satisfied **structurally** -- there
is nothing to pass and nothing to get wrong, and `--solid-pusher` on the DP CLI is
now a harmless no-op. The env still reports all three in its state dict, so they
stay visible in provenance.

⚠️ This is NOT true of the OLD sim, where `solid_pusher` is a constructor arg
**defaulting to False** and `solid_contact_guard` does not exist at all. That is
the fault that invalidated the earlier `dp_sc_v2_*` results. Another reason every
eval must run on Sim V2.

Never infer these from training metadata: 98-100% of obs0 episodes record no flags
(the fields postdate them), while 100% of obstacle episodes record solid+guard.

## 9. Previously-known obstacle block (resolved 2026-08-27)

The 2026-08-12 obstacle sweep was blocked because its seed copies were stale,
empty, or bound to pre-final geometry. Do not restore or use those artifacts.

The block is resolved only through §7's target-side seed bank, exact simulator
commit, and canonical launcher gates. Obstacle evaluation is runnable for levels
1-30 when all of those gates pass. A launch using another seed file, another
simulator commit, or a legacy repo-local launcher remains non-protocol and must
not be reported as comparable.
