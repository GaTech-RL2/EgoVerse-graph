# ImageNet native image-generation admission

`kind: native-imagenet-world-model-v1` selects the maintained ImageNet branch in
`scripts.train.joint.run`. Other request kinds continue through the existing
Elmo/LIBERO admission, including their unchanged temporal validation contracts.
This branch is for fresh training. It does not accept a continuation, an
interruption request, an intermediate allocation boundary or an archive/pruning
policy.

The constructed model retains all **314,227,000** trainable parameters, with the
same QK-normalized attention, SwiGLU and RMSNorm architecture as the paired Elmo
model. The `imagenet` embodiment executes the image expert and shared class-text
projection (**159,006,592** parameters). Robot/human state, action, camera and
unused language-fusion banks remain present and unused; they receive no fake
vectors, gradients, Adam moments or updates. Class names use the existing
Qwen-1024 to 128 projection. There is no new class-ID embedding.

Science is exact: native RGB160/P16, one frame/camera at time/index zero,
independent uniform corruption, plain FP32 image x0 MSE, AdamW peak 5e-6 with
3,000 optimizer-update linear warmup and then constant LR, weight decay .01,
seed42, four dedicated H100 GPUs, rank8/global32 and BF16 mixed precision with
FP32 stored weights and Adam moments. Flash4 packed attention and default dynamic
compilation remain enabled.

## Required real smoke

The maintained shared Lightning trainer must optimize **50 actual updates**,
then run scheduled image-generation validation on **all 1,000 frozen anchors**
(250 per rank). The anchors are exactly one deterministically selected official
validation image per class. The complete 50,000-image validation inventory is
bound to the context and excluded from training. The production and smoke data
contexts are identical; a 100-anchor substitute is rejected.

Before fitting, the allocated four-rank capacity path executes native rank8,
T1/C1/P100 compiled BF16 forward and finite FP32 backward, reserving the actual
Adam-memory estimate without an optimizer update. Every rank must identify a
distinct actual H100 and the same source, config and data context.

The smoke callback records every update's LR, finite FP32 loss/gradients,
conditional parameter usage and unused-bank weight identity. It validates all
1,000 generated images, saves one diagnostic full-state checkpoint at update50,
and preserves all four rank RNG payloads and the consumed data cursor. Allocated
postflight strictly reloads the complete wrapper, conditional AdamW state,
scheduler and sampler, and verifies that each saved rank RNG payload can be
restored. This is payload restoration, not a claim of dynamic distributed
continuation.

A separate native20-step sample has every target RGB value replaced by NaN while
retaining the genuine finite class language. All generated pixels must be finite
FP32, and exactly20 descending denoiser calls must be observed. The sample must
match the first actual frozen anchor and its exact class feature binding except
for the intentionally hidden RGB values.

Measured warm timing covers updates11–50. The separately reported validation
interval includes validation, diagnostic checkpoint creation and train-end
checks; it must not be presented as validation-only timing.

Validation MSE is a paired class-conditioned diagnostic, not FID or an official
perceptual-quality benchmark. The evaluator saves only the fixed32-image preview
subset and aggregate artifacts.

## Fresh full admission

The full request binds the actual smoke request, preflight, resolved config,
capacity, completion, host exit and successful native terminal record. The
source, architecture, optimizer, data and 1,000-anchor context must remain exact.
The first production allocation uses fresh step0 and W&B `resume=never` for its
new identity. The smoke checkpoint never initializes production.

The total budget is300,000 updates. Scheduled validation and immutable
`epoch_epoch=####-step_step=#########.ckpt` files occur every20,000 updates; all15
remain local, with `last.ckpt` a symlink to300k. No epoch, wall-clock or terminal
production save is introduced. Terminal postflight strictly reloads300k,
rejects duplicated/nonfinite validation history and checks all15 retained
milestone files. Its `full-completion.json` still requires separate actual host0
and native `COMPLETED 0:0` certification.

## Data proof and execution boundary

Gateway interpretation is pure standard library and bounds each proof metadata
file to4MiB. It checks genuine accepted held parent/afterany observer claims,
submitted argv, request and wrapper manifests, actual RUNNING identities,
successful terminal queries and absence of remaining parent steps. A missing or
unknown submission outcome fails closed. The observer's own purged terminal
record is not promoted to a new scientific gate.

The explicitly reviewed Skynet source-CPU v5 route uses CR_CPU scheduling:24GiB
is requested, but is not consumably reserved or claimed as enforced RAM. Its
actual producer and observer records must preserve MemAvailable, the observed
controller/cgroup configuration and every inherited memory-limit observation.
The destination Lambda route separately requires its actual allocated memory
TRES; the Skynet exception never applies to Lambda.

Large raw inventories, Qwen feature metadata, source archives and data hashes are
not loaded or hashed by the gateway reader. The actual allocated driver repeats
the full bound source/runtime/data checks, validates the complete raw inventory,
recomputes the raw-to-language manifest projection and constructs the actual
DataModule context before any optimization. Later operational commits may reuse
a genuine source-data producer only when all five exact preparation components
remain byte-identical.

Transfer extraction is non-admitting. The actual target payload must freshly
hash and decode the entire destination corpus, compare its complete byte/RGB
identity with the genuine source inventory and prove file/directory fsync. Its
`imagenet-cache-readback-v2` result remains
`status: EXTERNAL_COMPLETION_REQUIRED`, `data_admitted: false`. Admission also
requires its separately reviewed held/afterany/release, host0 and successful
native completion closure. A JSON `PASS`, directory listing or source decode
alone cannot admit the destination.


## Published source and decoder boundaries

The raw/Qwen CPU producer is pinned to published source I,
`e9bff72c82618e74175886d16983f3cf90780fb0`. GPU smoke and full training must use
this separately published admission child C, with its actual revision, source
manifest, archive and upstream readback. I lacks the ImageNet smoke callback;
a local instrumentation overlay is not a substitute for published C. C keeps
the model, noiser, loss, sampler, dataset and preparation implementations exact
I. Its experiment config adds only the maintained conditional-usage callback
and disables the unrelated profiling callback.

A genuine source-I data producer is reused by exact component identity, not by
claiming I and C have the same whole revision. Admission compares all five
consumed dataset/preparation/language files against the source producer's
actual manifest and current C bytes. Source-v5 explicitly binds the configured
interpreter alias, its literal target and the measured physical CPython3.11.14
executable. The pure offline reader never resolves a remote alias locally.

The Lambda target-v2 producer uses the exact transfer-v3/source-v5 protocol:
allocated-host native/source authentication and full byte hashing precede ML;
inside the actual container, an isolated native Slurm capsule must authenticate
the running owner/request before decoder import. The official Pillow13 CPython
3.12 wheel is extracted with the shared maintained helper into an exclusive
private overlay. Interpreter, Pillow/native origin, installed bytes and five
namespace identities are bound before the unchanged Torch bootstrap and again
at decode. Source uses the separately pinned official CPython3.11 wheel; both
profiles retain Pillow13's exact decoder implementation contract.

The decoder export is an explicit published C subset: a flat
`SERVICE_FILES.sha256` and byte-identical `imagenet_wheel_overlay.py`, with an
actual publication/readback receipt. This subset manifest is not relabeled as
the whole training-source manifest. The reader executes no externally supplied
ops module. It invokes only its own pure standard-library helper after exact
manifest and publication binding.

Target admission replays the source-v5 proof chain, target-v2 accepted
held/afterany/release and actual host/native completion, source projection,
fresh complete physical runtime inventory and same-container native/decoder
proofs. The prior completed Lambda Elmo CPU-v4 proof is runtime-only scope. The
allocated target's manifest-pinned reader reopens that historical chain before
work; the maintained consumer binds its result and actual fresh target
runtime/source/native proofs without inventing ImageNet authority for it.

All raw-data, target-cache and actual GPU bindings remain external. Local
synthetic fixtures, real local archive inspection, tiny CPU Lightning checks
and source readback do not certify any actual cluster data, native execution,
GPU capacity, throughput, smoke or full launch.
