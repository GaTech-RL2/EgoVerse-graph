# Continuous-time MoT world model

This is an independent research model, not an Action Flow variant. The implementation
is on `codex/continuous-time-mot-world-model`, based on the verified common release
7043c025. It uses the existing PipelineAlgo, ModelWrapper, DataContext and Eval
interfaces. No existing model, launcher or run is changed.

## Confirmed design

- Human, YAM, LIBERO Sharpa and LIBERO gripper co-training.
- Three modality experts: DINO image features, XYZ tracks and proprioception.
  Each expert has its own Q/K/V/output projections, normalization and FFN.
  All valid tokens within one example participate in joint attention. Experts
  are shared across embodiments; pose input/output adapters are private.
  After each block, image tokens also pass through the selected embodiment's
  private residual adapter. Each block has separate adapters for each embodiment;
  the image output head remains shared.
- Frozen DINO targets; denoising objectives on both image features and tracks.
- Pose readout sees observed context and predicted visual/track features. Pose
  gradients propagate through those predictions to the visual generator.
- Visual/track warm-up followed by a pose-loss ramp.
- Poses and tracks expressed in one fixed reference frame per sequence.
- Regular image-grid tracking, persistent point IDs and confidence/visibility.
  Do not substitute interpolated endpoints for actual tracks.
- Source sampling proportional to dataset size. This implementation defines
  size as eligible temporal anchors/windows, and returns homogeneous source
  batches. PipelineAlgo sees one source per update; it must not apply a second
  source-balancing factor. Exact source probabilities depend on the corpus and
  temporal mode and must be reported for a comparison.
- Continuous: 2–16 random recorded timestamps (user-selected maximum), each within
  -3 to +3 physical seconds of the current/anchor frame (user-selected range).
  Bounds clip to episode edges; the anchor is included. Draw without replacement
  from eligible recorded times, not uniformly over continuous seconds. Anchors
  cover the episode; individual examples span at most six seconds. Isolated
  anchors with fewer than two eligible times are excluded, never replaced with
  out-of-range frames. This replaces the earlier unbounded frame-index heuristic
  in the configured continuous recipe. Direct calls without time bounds retain
  the historical sampler for checkpoint-bound compatibility.
  Discrete: four fixed-stride frames (provisional), integer temporal positions.
  This compares complete recipes, not time encoding alone.

## Proposed engineering defaults, not measured training settings

Width384, depth8, heads6, FFN expansion4. There are 43,767,555 shared trainable
parameters excluding embodiment pose input/output adapters and the frozen DINO
backbone. Pose adapter count is sum_m(769*D_m + 384). Pose widths deliberately remain
required config values until measured-pose schemas are verified; historical
7D gripper or 69D Sharpa policy command widths are not measured-pose schemas.

The image residual adapters add 1,218,048 trainable parameters for four
embodiments: 304,512 per embodiment across eight blocks. The configurable
bottleneck is 48, giving 384 → 48 → 384 per adapter:

    image_next = image_shared + Up_e(GELU(Down_e(LayerNorm_e(image_shared))))

Only the current example's embodiment adapter is used. The up projection starts
at zero, so initialization exactly preserves the shared model's forward output.
On the first update the up projection learns; the down projection and adapter
normalization begin receiving nonzero gradients once the up weights change.
Adapters are reused in both visual denoising and pose-readout passes. Padding is
masked again after the adapter. The total excluding pose adapters and frozen
DINO is now 44,985,603 parameters; 43,767,555 of those remain shared.

Set image_adapter_dim=0 explicitly for an adapter-free ablation or the original
adapter-free checkpoint architecture. Strict loading rejects a missing adapter
state; enabling adapters is an architecture change, not transparent resume.

The maintained upstream feature extractor is
`scripts/embedding_process/dinov3_embedding.py`, with a DINOv3 ViT-B/16 default.
It supports pinned model/processor revisions and explicit full-FOV letterboxing;
patch counts are checked against the actual processed pixel grid and model output.
Preprocessing geometry, normalization, precision and revision are saved with each
output. The correspondence diagnostic in `dinov3_correspondence.py` uses hashed
native frames and independent visible-point labels, excludes padding, and saves
feature caches. Its tests and a tiny untrained Transformers DINOv3 runtime check
verify geometry, not pretrained accuracy. Official pretrained access has now
been verified in the task's FP32 DINO pilot. Whole-hand correspondence uses
estimated projected Aria references with unknown occlusion; it does not measure
independently verified visible-joint accuracy or establish a final resolution.
The proposal explicitly favors high-resolution
features for small gripper detail. The earlier224x224/196-patch estimate is only
a debugging budget, NOT an approved scientific preprocessing choice. The actual
processor/checkpoint revision, resolution and produced grid must be pinned and
checked during preprocessing. The prepared loader rejects declared patch grids
that disagree with the cache.

The isolated `scripts/embedding_process/dinov3_pose_probe.py` diagnostic reuses
SHA-verified image features and trains only a linear patch-to-landmark heatmap
head. The world-model pose readout cannot express this control because it uses
MoT context and track predictions. The probe belongs with the maintained DINO
diagnostics, accepts paths and settings as arguments, and leaves entire episodes
out. It compares training and held-out image-only predictions with a training-only
mean-pose baseline and predictions from shuffled held-out images. It never selects
a checkpoint on the test set. Projected references retain their uncertainty;
this is current-image 2D landmark recovery, not future 3D pose forecasting.

A proposed1120-point28x40 grid is selected in ONE reference camera, with its
3D positions expressed in the common reference frame. It is not1120 tracks
per camera. Four timestamps, three cameras and196 image patches produce
4*(3*196 +1120 +1)=6836 tokens. The initial hard limit is8192 tokens. This is a
budget guard, not evidence the full-size model fits a particular GPU. Joint
attention scales quadratically in this token count; memory/time need a scheduled
GPU measurement before training. No silent point subsampling or pooling occurs.
For comparison,448x448/784 patches at four timestamps gives13892 tokens and
exceeds this guard; two timestamps gives6946. Resolution and temporal/track
coverage must be budgeted together, without silently reducing the requested
image detail. Neither example is an approved full training resolution.

The requested continuous maximum is now16 timestamps, including observed and
target times together. With640x640 DINO inputs, three YAM cameras,1120 tracks
and one pose token per time,16 times require94736 tokens. The8192-token guard
remains active: this configuration is not ready for a full-resolution launch.
Sixteen versus four full times gives4x as many tokens and16x the dense attention
pair count. A materialized FP32 six-head attention-score tensor alone would
occupy about200.6GiB per layer at batch1; that is a backend-dependent warning,
not a measured peak or a claim that memory-efficient attention materializes it.
Do not silently raise the guard, reduce resolution, pool features, subsample
tracks, or switch precision to make it fit. GPU feasibility and any redesigned
token budget remain separate work. Historical direct sampler defaults stay4.

Memory controls are available without changing parameters or token coverage:
\`attention_query_chunk_size\` splits attention queries while retaining all keys
and values; non-reentrant checkpointing recomputes each query chunk in backward.
\`activation_checkpointing\` recomputes each MoT block instead of retaining its
internal activations. Both default off for historical behavior. FP32 fixture
tests compare outputs, losses, all gradients and one AdamW update, including
padding and randomly placed observations. These controls change execution and
roundoff, not the attention graph; they do not remove quadratic compute cost.
The maintained \`scripts/benchmarks/profile_world_model.py\` measures synthetic
full-size FP32 forward/backward peak memory on one allocated GPU. Its explicit
diagnostic pose widths and token guard override are not training schema or
launch approvals. It performs no optimizer updates; real-data optimizer,
validation and reload checks remain required separately.

Warm-up10000/ramp10000, AdamW1e-4/weight-decay0.01, loss weights1/1/1 and20 sampler
steps are provisional configurable defaults. No CFG, Action Flow Jacobian loss,
velocity augmentation or historical policy-specific optimizer is inherited.
Actual training duration, split/data identities and physical temporal span are
not selected by the config. `max_steps`, `manifest_path` and `pose_dims` are
mandatory before use. Both variants share the same parent recipe.

## Training computation

Physical time tau and diffusion noise s are separate. Temporal RoPE uses actual
relative seconds in continuous mode, grid indices in discrete mode. A common
random time-origin shift preserves all temporal differences. Image tokens also
carry patch XY and camera ID; tracks carry reference XY, camera and point ID.
Observed/query status and s have their own embedding.

For each withheld visual frame and each modality, independently sample s in
(0,1) and Gaussian epsilon. All image patches/cameras at that timestamp use the
same s. With alpha=cos(pi*s/2), sigma=sin(pi*s/2):

    x_s = alpha*x_0 + sigma*epsilon
    v_target = alpha*epsilon - sigma*x_0
    x0_estimate = alpha*x_s - sigma*v_prediction

This is the default `image_prediction_type: v`. The opt-in `x0` setting instead
interprets the image head as a clean-feature estimate and applies image MSE
directly against x_0. Track targets remain v in both modes. The pose readout
receives the corresponding differentiable clean estimate. The setting is part
of checkpoint/config identity: matching weight shapes do not make v and x0
checkpoints interchangeable.

A linear image output head narrower than the DINO feature dimension cannot
represent arbitrary full-dimensional noise in v. The x0 alternative lets the
head model structured clean features while the sampler carries noise outside
the head analytically. Its DDIM update uses
`alpha_next*x0 + (sigma_next/sigma)*(x_s-alpha*x0)` with zero-level anchors
handled separately; this avoids overflow at tiny positive refinement strengths.
Clean-feature MSE corresponds to sigma-squared-weighted v MSE, so this is an
explicit objective ablation, not a numerically equivalent optimization. It
preserves all image channels but does not guarantee improved generalization.

Observed tokens remain clean. The image and track losses are weighted mean MSE
on supervised withheld tokens. Track confidence is a loss weight; unknown-future
visibility is NOT an attention mask or input signal. Padding is separately
masked. Inputs never contain withheld clean poses.

Half of sampled masks are past-only forecasting by default. The other half can
condition/query before, between or after observations, with at least one observed
and one withheld image frame. Neither endpoint is forced in this random branch.
The50/50 mixture is provisional. Fractional confidence weights are normalized
by their actual sum, including sums below one; empty objectives return zero.

### Missing-track policy

Missing coordinates must never become clean zero coordinates before partial
noising. A track is unavailable when its target-valid mask is false or confidence
is zero. The default `missing_track_policy: error` rejects an incomplete training
batch with a clear error. It also rejects invalid/zero-confidence observed visual
values and unsupported missing image features. Padding is not a missing target.
Geometric target availability is distinct from camera visibility: an occluded
simulator point can still have an exact measured XYZ target. Do not automatically
set its confidence to zero solely because it is occluded; confidence describes
coordinate reliability. Future visibility must not become an input feature.

For explicitly approved incomplete-data experiments, the existing stage accepts
`missing_track_policy: conditional_sample` and `imputation_steps` (default20).
This is approximate conditional data completion, not measured geometry:

1. Build a private auxiliary batch with available recorded image/track targets
   clamped as clean anchors. Keep only the originally observed poses; sanitize
   all withheld poses and missing coordinate placeholders.
2. Use the existing sampler under `no_grad` and detach its missing-track outputs.
3. Insert only those missing coordinates before ordinary per-frame forward
   noising. Their direct track-loss weights remain zero. Do not promote them to
   supervised or observed values in the main pass.

Target availability selects anchors in the private auxiliary sampler; it is
not added as a flag, attention mask or noise-level change in the main denoising
pass. The main pose loss still differentiates through the denoised predictions,
but never through the auxiliary completion sampler. The current model supplies
the completions, so early/untrained completions can be inaccurate. This is not
an exact marginal-likelihood objective or proof of correct occluded geometry.

Complete-data batches preserve their previous losses, gradients and RNG state
bit-exact in tested CPU FP32/BF16 cases and incur no extra sampling. An incomplete
batch with20 completion steps costs21 extra backbone forwards (the shared sampler
also computes a final pose readout). `world_imputed_track_tokens` and
`world_imputation_forwards` report this cost. The default remains strict to avoid
silently introducing that compute and approximation into a real run. Choose and
measure a completion policy on actual data before full training.

Regression tests cover strict rejection, zero-confidence finite placeholders,
NaN/large-value placeholder invariance, withheld-pose isolation, no batch mutation,
detached completion, unchanged main frame-noise levels, finite CPU BF16 gradients,
pipeline controls/cost logging and complete-data compatibility.

After warm-up a second forward through the SAME backbone uses differentiable
same-trajectory x0 estimates plus empty pose queries. Its regression loss updates
both visual output heads and their upstream shared parameters. It must never
pair an independently sampled alternative future with the recorded pose target.
There are two network forwards per training batch after warm-up, one before;
there is no unrolled differentiable sampler or higher-order/Jacobian objective.

Inference uses deterministic DDIM updates from Gaussian initialization, clamps
observed visual values at every step, then runs one pose readout. All unknown
input values can be NaN placeholders without affecting output. The sampler never
reads training confidence/target-valid fields. Image outputs are DINO FEATURES,
not RGB video; no image decoder is included.

### Selective prediction refinement

The same sampler optionally accepts `world["refinement"]`. Each image/track
entry has `strength[B,N]` and, when retaining or revising existing predictions,
`prediction[B,N,C]`. The ordinary pipeline stage passes these controls through
inside its existing world input; there is no separate inference launcher.

| Strength | Behavior |
| --- | --- |
| 0 | Keep the supplied clean prediction bit-exact as an anchor throughout sampling |
| Between 0 and 1 | Add the corresponding noise to a supplied clean prediction, then refine |
| 1 | Generate from fresh Gaussian noise, ignoring any supplied previous value |

For a clean previous prediction y and strength s, initialization is
`cos(pi*s/2)*y + sin(pi*s/2)*epsilon`. Each query then descends from its own s
to zero over the configured number of DDIM steps. Strength is a noise level,
not a percentage change or a guarantee of geometric closeness. All image patches
and cameras at one unknown timestamp must agree on strength; track points at
one timestamp must also agree, independently of image strength.

Example for a caller that has already aligned previous outputs to the requested
token layout:

```python
world["refinement"] = {
    "image": {"prediction": previous["image"], "strength": image_levels},
    "track": {"prediction": previous["track"], "strength": track_levels},
}
# e.g. frame levels: [0.0, 0.0, 0.4, 1.0], repeated over each frame's tokens
result = model.sample(world, embodiment, steps=20, generator=generator)
```

Actual observations must have strength zero and always use their measured
values, even if the supplied previous prediction disagrees. An unknown frame
at strength below one requires an explicit finite prediction; the sampler never
falls back to withheld values in the batch. Fresh-frame prediction placeholders
may be NaN. Prediction arrays must match the value shape/device/dtype, normalized
feature/track representation, timestamps, camera IDs and point ordering.
The caller owns this alignment; shape checks cannot prove semantic alignment.
Omitted modality controls retain ordinary fresh generation for that modality.

Fixed generated anchors are marked clean only in a private sampling copy;
caller inputs and observation masks are unchanged. Pose queries are regressed
again from the refined visuals; pose diffusion/refinement is not introduced.
Keeping a previously generated pose fixed requires explicitly treating it as
conditioning rather than passing it to the visual refinement controls.

Tests cover mixed keep/refine/generate requests in continuous and discrete time,
CPU BF16 mixed, exact fixed anchors at every denoising call, repeated calls with
changed query times, ordinary pipeline inference, and rejection before RNG use
of invalid strengths, missing predictions or attempts to alter observations.
Automatic timestamp insertion/reordering and observation-update orchestration
remain separate future work. Learned refinement quality needs real-data tests.

Open scientific issues: the one-pass training x0 distribution differs from
multi-step sampled features, direct context can provide a readout shortcut,
frozen features may omit contact/gripper detail, track estimates may be biased,
and deterministic pose regression can average ambiguous motions. Parameter
sharing does not establish transfer. Evaluate forecasting and interpolation
separately and compare matching physical query times. Do not label normalized
feature/pose MSE as robot task success or action-distribution fidelity.

## Prepared data interface

`rldb/world_model.py` owns the typed adapter and timestamp sampling. It consumes
an immutable JSON manifest plus per-episode NPZ or HDF5 arrays; it does not modify source
Zarr datasets. The existing DINO extraction and raw dataset readers remain the
upstream authorities. This new cache contract is necessary because the old policy
loaders do not supply synchronized irregular image/track/pose targets.

Manifest schema `mot-world-prepared-v1` requires:
- `dino_identity`: checkpoint/revision, feature layer, preprocessing/grid identity.
- `normalization`: train-only provenance and statistics SHA; arrays already normalized.
- `pose_schemas`: per-embodiment dimension and coordinate/rotation/field semantics.
- `track_provenance`: tracker/simulator identity, calibration and frame binding;
  `interpolated_endpoints` must be false.
- `episodes`: embodiment, episode_id, relative path, SHA256, frames, explicit
  train/val assignment, frame=`fixed_reference`, pose_kind=`measured_pose`.
  Duplicate IDs and resolved physical paths across splits are rejected.

Per-episode arrays:

| Array | Shape / semantics |
| --- | --- |
| time | [T], increasing physical seconds; subtract origin before casting float32 |
| image | [T,V,P,D], frozen normalized DINO patch features |
| image_xy | [P,2], patch positions |
| track | [T,Q,3], normalized common-reference positions of persistent points |
| track_xy | [Q,2], regular-grid reference image positions |
| track_camera | [Q], integer reference camera identifiers |
| track_confidence | [T,Q], [0,1], unavailable=0 |
| pose | [T,Dm], normalized measured poses in same fixed reference frame |
| pose_valid | [T], measured target availability |

The current adapter assumes synchronized arrays on a verified physical timeline.
Do not fabricate time from arbitrary indices or interpolate unseen trajectories.
Do not copy policy action arrays into pose. Use original measured state and
calibrated transforms. `points_to_reference` validates proper rigid transforms
for XYZ; orientation conversions should use existing pose utilities. Sampling
and padded collation preserve actual times and masks. Different query schedules
per modality are supported by the model token contract; the initial prepared
loader samples the same timestamp set for all three modalities.

NPZ remains a correctness/prototype interface: selecting a few frames reads
the entire episode field before slicing. Dense caches should use the supported
HDF5 backend with one-frame image chunks; the reader selects frame indices on
disk before materializing image/track/pose arrays. Both backends retain the same
SHA-bound manifest, sampling, normalization and masks. Regression tests require
bit-exact samples in both temporal modes and intercept image reads to reject
whole-episode access. The finalizer hashes each new immutable payload and embeds
its SHA-bound filesystem fingerprint in the trusted prepared manifest. Startup
reuses that verification only when the resolved path, device, inode, byte size,
mtime and ctime all match exactly. Copies, relocations, mutations, malformed
fingerprints and historical manifests fall back to a full content hash. This
proof is authored only if ctime was at least one second old when hashing began;
recent files are hashed without creating a reusable proof, guarding against
same-clock-tick rewrites. Aging later cannot revive a recent hash's proof. This
fast path relies on a trusted immutable manifest and filesystem metadata; it is
not a substitute for hashing new artifacts or an independent silent-bit-rot
check. Startup also reads the small timestamp vectors. Metadata assertions are
not proof of actual calibration or leakage-free normalization; preprocessing
receipts must establish those properties. Exact data-RNG continuation across
checkpoint resume is not implemented or claimed.

Single-device loading only: distributed training fails explicitly until source
sampling and data sharding are implemented and tested. Training source choice
is proportional to eligible windows, with replacement; validation enumerates
anchors once with deterministic timestamp/mask choices. This is not a claim of
exhaustive evaluation over every possible temporal subset.

## What has been verified

Tests cover real tiny optimizer updates for both embodiment heads, strict model
and optimizer reload, shared Lightning fit/validation/resume, both Hydra recipes,
BF16 mixed finite gradients on CPU, pose gradients into both visual heads,
no withheld-target leakage at inference, clean observation clamping, padding
invariance, temporal shift invariance, discrete indices, irregular sampling,
proper reference-frame transforms and cache split/provenance rejection.
Adapter tests cover all four embodiment routes, pose-loss gradients, unchanged
unused adapters, zero-initialization equivalence, embodiment-specific effects,
invalid bottleneck rejection and strict checkpoint restoration.

These are synthetic fixtures. They do not establish real-data performance,
CUDA numerical equivalence, full-size memory fit or GPU training readiness.
Regression tests also cover backward/interpolated image queries, fractional
confidence normalization, empty objectives and declared-versus-actual patch grids.
Ordered hierarchical plan tokens are intentionally deferred by the original
proposal's first-version decision, rather than an accidentally missing component.

## Data inspection and remaining work

Training uses Lightning `32-true` (FP32), explicitly selected by the user.
The discrete recipe inherits this setting from the continuous parent. Historical
BF16 fixture tests remain compatibility checks, not the selected training mode.

Live shell inspection confirmed both500-episode LIBERO Spatial Zarr roots exist.
Sharpa data has61535 rows, EEF position3/quaternion, hand state22, two RGB cameras,
and action/observation host-nanosecond timestamps. Original gripper data has62250
rows, EEF position3/quaternion, gripper state2, joint state and two RGB cameras.
Neither listed processed replay contains scene-wide tracks. A bounded audit
recovered original HDF5 simulator states and XML for one episode of each LIBERO
embodiment; both physical clocks advance in 0.05-second steps. Exact state/image
alignment, asset restoration and persistent-point extraction remain unverified.
Use simulator time, not Sharpa host wall-clock timestamps.

The human sample has real device timestamps, including dropped-frame gaps, and
moving camera poses. Its 21 hand landmarks follow raw Aria-supported ordering:
wrist index5 and fingertip indices0–4, not canonical MANO. Aria lacks MANO's thumb
CMC joint, so conversion is not a full permutation. Prepared pose schemas must
bind the original producer, input/output topology and any fitted conversion;
63 coordinates alone are not topology evidence. Preserve original data.

The inspected YAM sample has measured EEF/gripper/joint states, all three camera
intrinsics and front-camera/inter-arm calibration, but no acquisition timestamps.
Nominal30fps is not a verified physical clock. Wrist-camera metric extrinsics are
needed if using those views for 3D geometry; front-reference tracks do not require
them for image-only wrist-camera feature tokens. No inspected embodiment has a
verified scene-grid track cache. Occlusion and coordinate availability differ:
an occluded simulated point can still have valid XYZ. Keep visibility separate
from coordinate confidence and never expose unknown future visibility as input.

Before any real training: resolve human/YAM camera poses and calibration, exact
DINO runtime/weights, source simulator assets/state replay, 3D track extraction
for each domain, synchronized physical clocks, measured-pose schemas, train-only
normalization, dataset splits and complete preprocessing receipts. Then verify
an actual batch and a scheduled full-size GPU optimizer/validation/reload check.
Do not treat this code milestone as a prepared four-domain training dataset.


## Selectable attention execution

New continuous and discrete MoT experiments default to the combined Flash4 BF16
attention profile. No attention override is needed: select the experiment normally.
Weights, FFNs and losses remain FP32; TF32 stays disabled. Whole-block activation
checkpointing is off and queries are unchunked.

Explicit alternatives remain available through the already-selected config group:

```text
world_model_attention=unchunked_fp32
world_model_attention=native
```

The shared continuous recipe selects the default and the discrete recipe inherits
it. Historical checkpoint-bound sources/resolved configs and direct Python model
constructor defaults remain unchanged. Inconsistent explicit strict-FP32/BF16
declarations are rejected. Existing explicit per-model overrides still take
precedence and must be inspected during a future launch preflight.

Both optimized profiles disable whole-block activation checkpointing and set
query chunk size zero. Every query still attends to every valid key across
all modalities and timestamps. Image patchification, token counts, parameter
counts, data sampling, normalization and loss weights are unchanged. The FP32
profile requires fused efficient CUDA SDPA and rejects a math fallback.

The Flash4 profile explicitly casts only attention Q/K/V to BF16 and converts
its output back to the incoming dtype. With the existing `trainer.precision=32-true`,
parameters, the remaining model and losses stay FP32. The Flash4 profile explicitly
sets `runtime.strict_fp32=false` and `runtime.disable_tf32=true`: attention is
intentionally BF16 while FP32 backend TF32 shortcuts remain disabled. It currently requires
batch one and Hopper-or-newer CUDA hardware. Boolean padding (including holes)
is handled by gathering all valid keys/values; all queries are retained.
No implicit fallback, reduced attention window or token subsampling is used.

The tested optional runtime is Torch2.9.1+cu126, torchvision0.24.1+cu126,
FlashAttention4.0.0b35 (beta), CuTeDSL4.8.1, quack0.6.5, apache-tvm-ffi0.1.14.post1
and cuda-bindings12.9.0. The ordinary project environment remains Torch2.7.1;
**do not install Flash4 into it or replace its shared dependencies.** The
optional backend is lazy imported only when selected and requires the separately
verified runtime. The base package's Torch pins have not been relaxed; this is
an explicit source/runtime execution profile rather than a claim that the
standard dependency resolver installs that optional environment.

The external beta API is checked for output container, shape, dtype and device.
Numerical BF16 differences are intentional; finite gradients and short timings
do not prove unchanged convergence or final task quality. A requested future
full run must use its own resolved-source/config/runtime optimizer, validation
and checkpoint compatibility preflight. No existing canceled run is resumed
by selecting or adding these profiles.

## Pixel-only distributed loader execution

The optional `+pixel_execution=ddp4_prefetch` profile applies to the new
`PixelProprioDataModule` and pixel video recipes. It inherits four-rank DDP,
enables two loader workers per rank, pinned host memory, persistent workers,
and two prefetched batches per worker. It explicitly fixes
`data.validation_batch_size=1`. Training batch size remains one until an
allocated GPU benchmark selects a larger `data.batch_size`; the global training
batch is that value times four. Attention backend and compilation choices remain
separate execution settings. This profile is not a launch or performance claim.

For the frozen 450/50 gripper split, validation still covers two deterministic
anchors in each of 50 episodes: 100 global batches and 25 contiguous batches per
rank, with no dropped or duplicated examples. Global validation IDs preserve
the same frame choices, observation masks, and sampling-noise seeds as the
unsharded evaluation. Training batch changes do not alter this validation grid.
The loader rejects a validation batch count that cannot divide evenly across
the configured ranks. Unattached inspection and strict reload see all 100
validation batches.

Checkpointed training draws use an immutable `(anchor, consumed cursor, global
draw slot)` key. Workers derive frame selection and conditioning masks from
that key, so prefetched but unconsumed batches are discarded safely on restart.
Only acknowledged batches advance the saved cursor. Worker count, pinning,
persistence, and prefetch depth are operational settings recorded in the
resolved config; they do not change source draw identity. Explicit train/val
batch sizes and the optional work-assignment strategy are checkpoint-bound.
Changing those identities is not an exact resume. Historical configs omit the
new optional identity fields and retain their previous data-context bindings.

`data.balance_training_work=true` enables deterministic work redistribution
inside each global training batch. It previews the exact sampled frame count
from the same RNG helper used by the loader, without reading image data, and
estimates attention work as squared image-token count. Longest-first assignment
keeps exactly the configured number of samples on each rank and is accepted
only when it lowers the largest rank's estimated total. It never changes the
global batch's selected clips, frame indices, conditioning masks, or draw keys,
and never moves a clip across optimizer updates. Batch-one and equal-length
discrete batches retain their original assignment. The estimate is not measured
GPU time: FFNs, padding, transfers, and synchronization can affect throughput.
Reassigned clips receive different rank-seeded diffusion-noise realizations;
their noise distribution is unchanged, but per-clip noise equality with an
unbalanced run is not claimed.

CPU tests exercise two workers with training batch two, discard prefetched
queues, and compare resumed versus uninterrupted frames, values, timestamps,
and masks exactly, with both balancing settings and changed worker counts.
They also verify full 100-anchor coverage across four rank shards and reject
incompatible batching/work-assignment checkpoint identities. Lightning's default
batch transfer already requests nonblocking CUDA copies for these nested tensor
containers; pinned allocation and useful transfer overlap require CUDA proof.
GPU throughput, memory, the selected larger batch, and an actual optimized
train/validation/checkpoint smoke remain pending measurements. Existing
zero-worker and four-rank baseline receipts do not certify this new profile.

## Modular pixel state/command readout

The deferred `pixel_proprio/readout_continuous` and `readout_discrete` recipes
keep one registered model at pipeline stage index 1. `PixelReadoutModelStage`
owns its parameters, applies the explicit `frozen_image` or `joint` policy, and
publishes the transient `pixel_video/denoiser` reference. The stable
`pixel_denoiser` and `image_proprio` aliases both select this owner; strict
initialization and tensor checkpoints retain `stages.1.model.*` names. The
legacy `PixelReadoutDenoiserStage` still implements its previous combined
ownership/prediction behavior for older serialized configurations.

The high-noise image objective is an independent branch: `PixelVideoNoisingStage`
followed by `PixelReadoutVideoDenoiserStage` and `PixelReadoutImageLossStage`.
Removing that branch does not remove the shared model or readout gradients.
When editing the stage list, preserve the owner's index 1 if reusing its exact
checkpoint namespace; for example, the retained low-noise noising node can
occupy index 0. The default recipe still executes the high-noise branch, keeping
the existing losses and random-draw order. Under `frozen_image` its image MSE
contributes to reporting but has no trainable dependency.

Both training and inference now supply the tensor port
`pixel_readout/image_prediction` to `PixelStateActionReadoutStage(image_key=...)`.
The default training producer remains `PixelReadoutImageDenoiserStage`, fed by
the separate corruption node with maximum strength 0.2. The inference producer
is `PixelReadoutSampledImageStage`, using the configured 20-step sampler and
`modes: [inference]`. The readout itself does not choose how images are generated.

For a future experiment, a sampled training producer can replace the low-noise
noising/prediction branch using the same port:

```yaml
_target_: egomimic.pipeline.stages_pixel_readout.PixelReadoutSampledImageStage
modes: [train]
prediction_key: pixel_readout/image_prediction
gradient_policy: stop
sampler:
  _target_: egomimic.models.pixel_diffusion.LinearDDIMImageSampler
  steps: 20
```

This is an available extension point, not a selected training algorithm. The
built-in sampler and this provider stop gradients through generated images.
Frozen-image readout training is supported. A future joint-training recipe
would receive image gradients only through separate image objectives or Phase B,
not through this sampled producer. The low-noise producer retains its original
differentiable behavior. Image, measured-state and controller-command losses
remain separate configured nodes.

The current image architecture uses AdaLN-Zero and spatial-temporal RoPE with
time/x/y dimensions `[64,32,32]` per 128-channel head and a `[10,10]` patch grid.
There is no learned XY projection or phase embedding; camera embeddings remain.
Discrete time uses sampled-position indices and continuous time uses recorded
seconds, an intentional comparison difference. The image expert has 88,178,688
parameters. Stage-two expansion requires identical noise/position settings and
geometry, and freezes all copied image parameters, including AdaLN modules.

Frozen image parameters preserve image-only Phase A and its sampler. Joint
attention during Phase B can still change image activations through state/query
tokens; the exposed image prediction is the Phase-A result. The default readout
training images come from a one-pass estimate of low-noise ground-truth future
images, whereas inference uses full sampled images. That distribution gap and
the eventual stage-two training strategy remain unresolved. No stage-two GPU
launch or certification is implied by these extension points.

CPU tests verify default output, gradient and RNG parity; strict legacy/new
tensor checkpoint compatibility; removal of the high-noise branch; sampled
provider substitution with unchanged model weights; explicit stopped gradients;
and frozen readout initialization under both clock and position variants.
