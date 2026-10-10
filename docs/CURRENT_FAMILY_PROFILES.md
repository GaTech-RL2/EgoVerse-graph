# Current family profiles (2026-10-10)

PR228 integrated the requested historical source lines. It did not establish
that every later task-local recipe works on that source. The three owner audits
found missing multiplier routes and additive stationary Transformer/evaluator
options. This follow-up restores those interfaces; existing runs remain pinned
to their recorded sources. CPU tests are not full-size GPU migration approval.

## LIBERO: preserve the working recipe

The four `libero/action_flow_<suite>_oat_dp_matched_s42` profiles retain original
OAT observations, H32/obs2/execute16, latent16x16, H240, Euler50/CFG4,
ActionVelocity1, batch32/accum1, clip3, fixed EMA0.9978 without warmup, seed42
450/50 train/validation split and the released composite optimizer/schedule.
Keep the native normalizer (`norm_stats.precomputed_norm_path=null`). They are
the active 120k recipes, with `max_steps=120000`,
`check_val_every_n_epoch=null` and `val_check_interval=20000`. The active
`libero` directory contains eight recipes: four AF and four batch16 DP.
Thirteen superseded recipes live under `libero_historical`, including the
checkpoint-bound original80k and older H16/obs1 selections.

The working 120k continuations additionally bind the original full-state80k
checkpoint, strict RAW/EMA/optimizer/normalizer restoration, the original W&B ID
with resume=must. These continuation bindings are separate from selecting the
active120k recipe and must come from the exact continuation
receipt, never from a guessed checkpoint filename. Duplicate final validation
remains an operational lifecycle issue: do not repeat inference or overwrite
existing artifacts to bypass it.

The four new `libero/dp_<suite>_oat_batch16_keep_steps_s42` profiles retain the
latest requested DP batch16/accum1 and original update targets (605121,270054,
325065,280056 for10/Spatial/Object/Goal). OAT DP keeps its own optimizer, schedule
and EMA. This is distinct from stationary/PushShapes DP's EMA-disabled recipe.
Spatial retains its15000-update checkpoint cadence. The old global1024 DP
configurations are retained explicitly under `libero_historical` for historical
reproducibility; they are not active batch16 alternatives.

Owner authority: `libero-four-suite-oat-dp-matched-20261008`,
`LIBERO_COMMON_CODE_COMPATIBILITY_20261010.json`, original source7c253e5 and
batch16 verifier sourceb7c2aa64. Preserve original result/proof hashes.

## Stationary: additive candidates, no automatic selection

The matched DP builder defaults remain private-native U-Net14/138. Optional
`dp_denoiser=transformer` supports private-native14/138 or
`dp_action_contract=shared_cartesian14` with one shared Transformer and shared
input/output projections. The latest candidate uses width712/depth11/heads8,
100x14 outputs for robot and human, human XYZ/Euler/zero-gripper targets,
observation-only conditioning and no augmentation. Old138 normalization must
not be reused; train-only Cartesian14 statistics and window eligibility are
pending. The AF candidate remains human138, latent64x16, human multipliers
0.2/0.4/0.6/0.8/1.0 and a single robot1x view. Do not change AF to14 by analogy.

The optional open-loop evaluators reuse already-generated predictions and
require ordered full validation and robot identity retiming. They score eligible
validation-window starts, not complete raw recording coverage.

Owner authority: `stationary-dp-transformer-shared14-180m-20261009`,
`COMMON_CODE_COMPATIBILITY_AUDIT.json`; sources26561551/3a95061a. The exact
selected emitted configs remain separate immutable task artifacts.

## PushShapes: explicit multiplier versus historical physical speed

AF selects `pusht/action_flow_cotrain_uc_multiplier_interpolation`; its typed
route requires fresh initialization, scalar raw dimensionless multiplier, no
physical-speed reference, Euler50, batches32+32 and fixed EMA0.9978. The latest
explicit PushT preference uses normal native per-source image processing, not
grouped64. Native DiT-half is configured independently on the encoder/velocity
backbones; it does not wrap image execution. Full-size activation remains required.
The historical speed recipe is retained separately, never reinterpreted as a
multiplier checkpoint. Numerical differences accepted for the original fresh
recipe do not grant a waiver to other configurations.

DP selects `DP_COTRAIN_STANDARD=usocket_chain_manual4919_af_obs_multiplier_261m_v1`
through the maintained launcher. It preserves the261M AF-matched observation
recipe, AdamW/cosine/500warmup and EMA off. Explicit unrecognized or conflicting
profile selectors fail; they never fall through to Chain-only or clean3K.

Owner authority: `dp-af-multiplier-training-pair-20261008`,
`COMMON_CODE_PUSHSHAPES_COMPATIBILITY_V1.json`; original DPf68ba039 and
AFaf2dc9e smokes. Those smokes do not certify a new integrated source. Neither
full80k run had been accepted at this audit. Quota and source-safe PID/requeue
guards remain independent launch gates.

## Historical preservation and launch boundary

An obsolete-for-new-selection recipe is not evidence that no checkpoint used it.
Historical recipes, checkpoint-bound sources and artifacts are preserved.
The proposed grouped-prefix restoration was withdrawn before publication to honor
`IMAGE_BATCH_PREFERENCE_V1.json`; LIBERO and stationary execution is unchanged.
No live launcher deployment, source allowlist change, new training, or running
source migration is authorized by these compatibility changes.
