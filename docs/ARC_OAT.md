# Native ARC versus OAT

This port targets **EgoVerse-graph**, using `PipelineAlgo`, `ModelWrapper`,
`MultiDataModuleWrapper`, and `trainHydra` for both methods. OAT is the actual
released implementation of Chaoqi Liu's Ordered Action Tokenization. The
archived repository now links to Praxis; Praxis is not used in this port.

## GPU launch through OSMO

The launch setup follows the other EgoVerse agent's working OSMO recipe:
`groot-l40s-03`, the NVIDIA PyTorch 25.06 image, immutable Git source,
`egoverse-github` credentials, and R2 checkpoints using the existing
`grabber-arc-r2-20260916` credential. It does not use the Slurm launcher.
L40S is the default. `--gpu-type H100` selects the `dgx-h100` platform;
submit that workflow to an authorized H100 pool. Before staging data, the
container verifies every allocated GPU against the requested family and count.
The selected family is recorded in `runtime.json`.

From a clean, **pushed** source revision, render and inspect a workflow:

```sh
source emimic/bin/activate
python -m scripts.benchmarks.launch_libero_osmo \
  --commit "$(git rev-parse HEAD)" \
  --run-id arc-oat-libero10-smoke-unique \
  --suite libero_10 --mode smoke --output /tmp/libero-osmo.yaml
osmo workflow submit /tmp/libero-osmo.yaml --pool groot-l40s-03 --dry-run
osmo workflow submit /tmp/libero-osmo.yaml --pool groot-l40s-03
```

Use `--gpus 2`, `4`, or `8` to distribute the same full global batch of 1,024.
Eight GPUs use microbatch 128 with one accumulation step. Accelerator selection
does not change the model, data, precision, optimizer budget or replay settings.
Checkpoint resumes retain optimizer and EMA state across these layouts; verify
advancing counters and a newly uploaded checkpoint before retiring a predecessor.

Use a lowercase, unique run ID. The workflow refuses existing R2 output at
`s3://rldb/experiments/arc-oat-20260919/<run-id>/`. It requests one GPU, 12 CPU
cores, 64 GiB RAM and 240 GiB disk. The smoke run trains the actual default
tokenizer, OAT policy, STK ARC policy and DUR ARC policy for two batches each, reloads their EMA
checkpoints, performs one four-step rollout on every task in the selected
suite, checks paired initial-state hashes, and evaluates 16 reconstruction
windows. These checks are not benchmark performance measurements.

`--mode full` runs the standard 5001-epoch recipe for the OAT tokenizer, OAT
policy, DUR ARC policy, and STK ARC policy,
all held-out reconstruction windows, and the complete 50-trial × 5-repetition
rollout protocol. `--epochs` can set a smaller pilot budget; such a run is not
the full released training budget. Render a separate workflow per suite from
the inventory below to cover the complete comparison. Runs initialize from
scratch unless `--resume-from-run` is supplied; smoke checkpoints never
initialize full training. Final checkpoints
include the last epoch even when it falls outside the periodic save cadence.

The released OAT Slurm recipes launch **four processes with batch 256 each**.
The reference global optimizer batch is therefore **1024**, not 256. These
single-L40S runs accumulate four microbatches of 256. `OATBatchBudgetCallback`
drops incomplete global batches at epoch boundaries, matching Accelerate's
four-rank `drop_last` behavior. For `N` training windows, each model performs
`floor(N / 1024)` optimizer updates per epoch and `5001 * floor(N / 1024)` in
total. EMA advances only on optimizer updates. Each model writes its measured
window count and exact update budget to `training-budget.json` and checkpoints.
For four-GPU training, use batch 256 and accumulation 1; numerical sample/RNG
ordering can still differ from the upstream Accelerate runtime.

The launcher accepts `--gpus 1`, `2`, `4`, or `8`. It requests all GPUs in one
worker and uses Lightning DDP, with CPU and memory reservations scaled by the
GPU count. Full training keeps global batch 1024: two GPUs use microbatch 256
and accumulation 2; four use microbatch 256 and accumulation 1; eight use
microbatch 128 and accumulation 1. Runtime and checkpoint receipts record the
actual layout. Switching GPU counts can change floating-point reduction and
sample/RNG ordering, while preserving the optimizer, EMA and update budget.

ARC training has a bounded cache of 262144 exact action-window encodings per
rank. Entries use native float32 action bytes, retain the original codec output,
and are invalidated if codec settings change. Cached values are derived data
and are not included in checkpoints. Zarr array handles are also reused within
data workers. These optimizations preserve target values, normalization,
episode boundaries and R/D/M settings.

LIBERO recipes also enable `benchmark.decoded_replay_cache=true`. The replay is
decoded once into immutable NumPy files alongside the source Zarr directory,
under `<dataset>.decoded/`. Both ranks and all data workers use read-only
memory mappings, so resident image pages are shared rather than copied into
each worker. Images retain their original uint8 values until sample conversion;
episode padding, splits, float conversion, normalization and ARC targets are
unchanged. The cluster records array shapes, dtypes and decoded SHA256 hashes
in `decoded-replay-cache.json`. Preparation uses a process lock and atomic
publication, and source-file changes select a new cache. This needs additional
local storage approximately equal to the uncompressed replay. Set the benchmark
option to false when comparing the original per-sample Zarr path directly.
The [loader profile](results/libero_loader_profile_20260922.md) separates data
access measurements from end-to-end training throughput.

To resume a frozen ARC profile on more GPUs, provide `--resume-from-run` and
its existing `--arc-replay-runs-file` together with the original profile/mode.
The completed replay is revalidated against the frozen profile and current
codec sources; it is not rerun. Resume compatibility is checked against the
resolved encoder and decoder settings saved in the graph checkpoint, including
the control protocol; it does not require a top-level Hydra benchmark section.
A replacement must advance from the recovered optimizer/EMA state and upload
a verified new checkpoint before its migration is marked complete. With spare
GPU quota, keep its predecessor running until that verification. At exhausted
quota, verify and retain the predecessor's durable checkpoint before releasing
its allocation for a rolling replacement. LIBERO-90 is deferred at the user's request as of September 22;
its seven paused jobs retain their R2 checkpoints and replay evidence. Results
for the other four suites must not be labeled the complete 130-task suite.

The initial `arc-oat-full-20260920-v1` submissions used effective batch 256 and
were cancelled when this mismatch was identified. They must not be
reported as the released training setup. Corrected submissions use the budget
above. The paper documents architecture/optimizer settings; the exact 5001-epoch
budget and four-process execution come from the released code and launchers.

Full runs write CSV training metrics under each model's `metrics/version_0/`
directory and automatically produce a paired `comparison.json` per suite.
To aggregate all five suites automatically, use the same `--campaign-id NAME`
on each workflow and set `--run-id NAME-libero-10`, `NAME-libero-90`,
`NAME-libero-spatial`, `NAME-libero-object`, and `NAME-libero-goal` respectively.
After uploading its completed results, each workflow checks all five receipts;
the last suite publishes `campaigns/NAME/comparison.json` under the R2 experiment
prefix. It rejects mixed source revisions, training budgets and incomplete
protocols. Full workflows allow up to two days in queue and 60 days execution;
this is a timeout, not a runtime estimate. Use a pool with available NORMAL
quota (for example `groot-l40s-01` or `groot-l40s-03`).

The released, SHA-256-verified LIBERO-10 Zarr archive is pinned to Hugging Face
revision `685b2b764e525ad33ab36d7315adbcab07494251`. The other suites download
official demonstrations at `yifengzhu-hf/LIBERO-datasets` revision
`f13aa24a3da8c43c7225569f28c562979fa0e35a`, verify every file's LFS SHA-256,
and use the shared converter. Both methods see the same replay and split.

Checkpoints upload periodically under content-addressed R2 keys. The run's
`checkpoint-receipts.json` maps local checkpoint names to their hashes and
durable URIs; `status.json`, logs, resolved Hydra configs, environment versions
and data receipts also upload. Monitor with `osmo workflow query <id>` and
`osmo workflow logs <id> -n 80`. LOW priority may use spare physical capacity
but is preemptible. To resume a stopped or superseded full run, supply
`--resume-from-run <run-id>` to a new workflow. Partial checkpoints retain their
optimizer, EMA, normalizer and epoch/update counters; completed stages are
skipped. The runner verifies source receipts, hashes, suite and training budget
before resuming through the shared `ckpt_path` training option. The default
NORMAL-priority smoke only needs one free GPU.

An isolated suite recovery can remain in the same campaign using
`--campaign-runs-file manifest.json` when rendering its workflow. The manifest
maps every suite to an explicit `run_id` and immutable `source_commit`, including
the replacement's own source. The publisher verifies every declared source,
budget, suite and completed evaluation; aggregate reports retain the per-suite
source revisions. This lets a recovered suite use its own artifact prefix while
the other suites continue from their existing checkpoints. `--raw-cache` also
supports full training on the converted suites: verified HDF5 files are read
without modification and the normal converter writes a fresh local Zarr.

Full training defaults to **both `dur` and `stk`**. Supply
`--arc-replay-runs-file replays.json` with `{"dur": "dur-replay-run", "stk": "stk-replay-run"}`
pointing to completed [R/D/M replay calibrations](LIBERO_ARC_REPLAY.md) for that
suite. Each mode trains its own policy from its own measured R/D/M settings;
the tokenizer and OAT policy are trained once and shared in both comparisons.
Both graph stages load the matching mode and measured choice from its receipt.
The gate checks all codec source dependencies. Missing or invalid replay evidence
preserves the OAT checkpoints and stops before ARC starts. Replay outcomes
measure demonstration reconstruction, separately from trained-policy scores.
The earlier shared-clock baseline remains available explicitly as
`--arc-modes joint_dur --arc-replay-run <legacy-run-id>`; its receipts cannot
authorize either independent-clock variant. Aggregate publication requires
both requested variants in every suite, with paired initial states and seeds.

To rerun evaluation from completed training after a simulator or inference
failure, render a workflow with a **new** run ID and
`--evaluate-from-run <original-run-id>`. Use the original R2 run ID, which may
differ from OSMO's suffixed workflow name. This restores all requested recorded
final checkpoints, verifies their hashes, completed epoch counts, suite and
training budget, and starts fresh paired rollouts/reconstruction. It does not
repeat training or reuse partially written rollout results. Recovery provenance
records both the training source revision and current evaluation revision.

Native factory hooks preserve OAT normalizers' device when loading CPU-mapped
checkpoints into an accelerator model, including Lightning resume. Upstream's
normalizer load method replaces its parameter dictionaries, so moving the
networks before loading weights alone is insufficient.

Source pins:

- [OAT](https://github.com/Chaoqi-LIU/oat/tree/1da92695ef12c23b7000a0b1a76cab0aef4750e6)
- [Its LIBERO submodule](https://github.com/Chaoqi-LIU/LIBERO/tree/6090ff21837566fed47b7c9061c9899f4749d36b)
- ARC base: EgoVerse-graph `c13f166`, including the duration-timed ARC stack.

## Benchmark inventory

The pinned OAT repository contains one environment family, LIBERO. The
tokenizer/policy YAMLs and Slurm recipes explicitly train **LIBERO-10**.
`oat/env/libero/factory.py` additionally exposes LIBERO-90 as a multitask
benchmark and accepts every individual task in the pinned task catalog.
The README downloads spatial, object, goal and LIBERO-100 demonstrations;
LIBERO-100 consists of the disjoint 90/10 subsets. It is not another 100 unique
tasks. The other tokenizer methods in that repository are baselines, not
additional environments.

The port covers the whole task catalog, with suite-level recipes also provided
for the task groups that upstream only exposes through individual task names:

| Suite | Tasks | Status in the OAT source |
|---|---:|---|
| `libero_10` / `libero10` | 10 | Released training and evaluation recipe |
| `libero_90` / `libero90` | 90 | Multitask factory and data composition |
| `libero_spatial` | 10 | Individual-task environments and conversion |
| `libero_object` | 10 | Individual-task environments and conversion |
| `libero_goal` | 10 | Individual-task environments and conversion |

`egomimic/benchmarks/libero/tasks.json` preserves the submodule's order and all
130 unique task names. This order defines the numeric `task_uid` observation.
Do not alphabetize or renumber it. Suite coverage is checked on dataset load.

The shared closed-loop protocol uses 50 trials per task, five evaluation
repetitions, 550 control steps, two 128×128 cameras, two observation frames,
a 32-action prediction horizon, and execution of the first 16 actions before
replanning. This gives 500 trials/repetition on LIBERO-10, as in its recipe.
The five-suite campaign contains 32,500 trials per method. The 550-step budget
is the OAT budget; this should not be labeled a reproduction of other LIBERO
leaderboards' differing suite-specific horizons.

## What is actually OAT

`egomimic/models/oat/` contains native copies of the source networks, with
source hashes and original notices. Runtime imports never need the `oat`
package or its Workspace classes.

- Encoder: 32×7 normalized actions, two 256-wide transformer layers, eight
  learnable registers. Actions attend to actions; register attention is causal.
- FSQ: levels `[8,5,5,5]`, a 1,000-entry implicit codebook and straight-through
  rounding. This is not VQ, scalar action binning, FAST, or a reconstruction proxy.
- Decoder: four transformer-decoder layers, learned queries, causal output
  attention and learned suffix masks. Training retains a random prefix from
  `{1,2,4,8}`. All prefixes reconstruct the full action horizon.
- Policy: the released two-camera robomimic ResNet18/SpatialSoftmax encoders,
  GroupNorm, 76×76 crop, raw state projection, and four-layer 256-wide
  autoregressive transformer with tied input/output embeddings and KV caching.
  Teacher forcing predicts FSQ codes; inference uses temperature 1 and top-k 10.
- Training: tokenizer MSE first, then token cross entropy with the tokenizer
  frozen; AdamW rates 5e-5 for tokenizer/policy and 1e-5 for observation encoders,
  betas `(0.9,0.95)`, zero weight decay, gradient clipping 1, microbatch 256,
  effective batch 1024, 5,001 epochs and the source EMA schedule. The source's `constant` scheduler
  ignores its warmup setting; the native recipe likewise keeps LR constant.

The source architecture is preserved. One explicit training correction keeps
the frozen tokenizer in eval mode even when its parent enters train mode;
`requires_grad=False` alone leaves nested dropout active. Reference policy-loss
tests put the upstream frozen tokenizer in eval mode for this comparison.
The default architecture/config is tested in the repository's Torch 2.7.1
runtime; upstream's lockfile uses Torch 2.10. GPU numerical parity across those
runtime versions has not been established by the CPU source-parity tests.

## ARC representation and test cases

ARC separates **geometric path** from **time along that path**. Its fixed
spatial supports can represent a path traversed at different speeds; timing is
needed to reconstruct the controller's regularly sampled action sequence.
A single mean velocity loses within-chunk timing, and a chord speed can run
too slowly around a curved path. Duration clocks avoid division by zero at a
hold. The repository also has legacy bimanual mean/profile/log-duration codecs
and the newer planar duration/adaptive curve codec; their layouts differ.

LIBERO is single-arm 6-DoF delta OSC control, so neither the planar tokenizer
nor a reduced xyz/gripper bimanual adapter is a valid drop-in codec.
`LiberoArcTimedCodec` ports the independently timed native planar STK and DUR
contracts to this control space. Each has M×12 floats: xyz, translation timing,
rotation6d, rotation timing, and gripper. STK stores local m/s and rad/s;
DUR stores interval seconds. Translation and SO(3) rotation have separate
uniform arc supports and separate D/R budgets. DUR reserves dwell boundaries;
STK cannot encode a stationary pause through zero velocity alone. Decoding
uses linear translation, SO(3) SLERP, and held gripper commands following the
translation clock. The two modes are independently calibrated before training.

The earlier shared-clock **`joint_dur`**
`LiberoArcCodec` is an explicit SE(3) duration-ARC extension for this control
space. It integrates controller increments (0.05 m and 0.5 rad scales) into an
anchored command path, uses the native ARC chordal rotation metric with xyz
arc length, and stores xyz, continuous 6D rotation, gripper, and interval
duration in M×11 floats. A gripper progress term and retained dwell/transition
boundaries cover motion with no translation. Rotation uses SLERP, position
uses a natural cubic curve in progress, and the duration clock maps progress
back to 20 Hz before differencing into the original delta commands.

These are documented control-space extensions, not unchanged legacy
bimanual codecs or the planar curvature-adaptive recipe. The maximum lookahead
is 32 commands, capped by the R/D choices measured in replay. At a limited support budget
it is lossy; when there are more dwell/gripper boundaries than supports, some
must be omitted. The reconstruction sweep measures that error. The command
path is not a claim about the robot's physically realized future pose.

M=16 and uncapped R/D are recipe placeholders, overridden by each mode's
completed replay evidence in full training. R is accumulated geodesic degrees;
D is accumulated translation metres. The shared-clock baseline's 0.05 m
rotation radius weights its progress metric and is distinct from R and D.

Regression cases cover straight/curved/closed paths; full holds; leading,
internal and trailing pauses; pure rotation crossing π; simultaneous
noncommuting rotations; gripper-only transitions; all seven action channels;
degenerate predicted 6D rotations; non-positive predicted durations; finite
input validation; normalization exactly once; dense-support round trips;
time-scale invariance of geometry; bfloat16 predictions; episode boundaries;
and observation/action alignment. At control-grid grip
transitions, float32 clock tolerance prevents a one-frame delay.

Tokenizer-only reconstruction compares OAT prefixes `{1,2,4,8}` against both ARC variants at
support budgets `{2,4,8,16,33}` with uncapped R/D on identical held-out chunks.
This rate/error sweep is separate from the replay-selected settings used by
the trained ARC policies. It reports raw
action MSE (the source benchmark), MAE, per-channel-family MSE, gripper sign
accuracy and integrated translation-command endpoint error. Representation
sizes are recorded: an OAT token has a 1,000-value vocabulary; an ARC support
has 12 floats (11 for the legacy shared-clock baseline). Equal row counts are **not** equal bit rates.

Closed-loop comparison uses the actual OAT autoregressive policy and the
existing continuous EgoVerse diffusion graph for ARC, with identical
observation encoders and simulator protocol. This is a method comparison:
the prediction heads and parameter/compute budgets differ, so a success-rate
gap cannot be attributed solely to the tokenizer. Reconstruction is the
separate codec comparison.

## Data and evaluation contract

The converter reads all tasks in a selected suite, flips both image streams
vertically as the OAT converter does, converts axis-angle proprio to xyzw
quaternions, and preserves all seven delta action channels. Conversion/merge
order is seeded and the manifest records source HDF5 checksums and episode
identities. It refuses to overwrite output. OAT-format replay Zarr files can
also be used directly.

The native resolver maps replay episodes into the shared MultiDataset. Splits
match OAT's `default_rng(seed).choice` episode mask. Observation history ends
at t and the first target is a_t. Start/end padding repeats within an episode;
validation never substitutes a failed sample. OAT's min/max channel fit uses
all replay frames, including validation, and maps constant channels to zero.
Both methods deliberately use this same source normalization scope. Graph
models receive normalized values; decoding unnormalizes exactly once.

The runner fixes two upstream protocol problems for **both** methods:

1. It settles objects for ten zero-motion/open-gripper steps, then fetches fresh
   reset observations, as the upstream README requires.
2. It reapplies each recorded seed at reset and allocates equal trial counts
   to every task, avoiding reuse of a worker's old seed or truncation of later
   task groups. Initial simulator-state hashes verify paired starts.

Success terminates inside an action chunk. Episode records include task,
trial, repeat, seed, initial-state hash, steps and inference latency. Videos
use the policy camera at 20 FPS. Execution groups trials by task to reuse
simulator assets; it retains the complete paired seed plan and reseeds every
reset. Reporting rejects missing/duplicate episodes,
protocol mismatches and different initial states. `compare` requires all five
suites and all planned trials; it does not turn a smoke test into a full score.
Statistics include per-task/per-repeat success, sample standard deviation,
standard error across the five repeats, and the paired ARC−OAT difference.
These five repeats concern evaluation randomness, not five independently
trained policies. The campaign preselects the final EMA checkpoints.

## Running

Use an isolated Linux GPU environment for the complete study. Source
`emimic/bin/activate` before Python commands. Install the repository with its
`oat` and `libero` extras:

```sh
source emimic/bin/activate
uv pip install --python emimic/bin/python -e '.[oat,libero]'
```

This selects only the benchmark extras. The existing whole-project `uv.lock`
predates optional PI05/Yam dependencies; regenerating it currently encounters
the unrelated PI05/LeRobot `av>=14.2` versus project `av==12.0.0` conflict.
The selected benchmark environment is resolved independently. OAT's Workspace
`dill>=0.4.1` dependency is unnecessary for native graph checkpoints and is
omitted, preserving the graph's existing dataset dependency constraints.
The selected extras were successfully resolved for Linux x86-64 / Python 3.11
with `uv pip compile pyproject.toml --extra oat --extra libero`.

LIBERO is pinned to OAT's submodule revision; rollout verifies the installed
Git revision and rejects a modified editable checkout. Configure
LIBERO's asset paths in `LIBERO_CONFIG_PATH/config.yaml` before a noninteractive
job (its upstream first-import setup otherwise prompts). On Slurm, acquire a
GPU before training; on sky1/sky2 the repository allocation is:

```sh
salloc -p rl2-lab -A rl2-lab --gres=gpu:a40:1 -c 12 --mem=30G
source emimic/bin/activate
```

Convert each suite once; the raw root can contain the four downloaded LIBERO
collections recursively. Run for each of the five canonical names above:

```sh
python -m egomimic.benchmarks.libero.convert \
  --input-root /datasets/libero_raw --suite libero_10 \
  --output /datasets/arc_oat/libero_10.zarr --seed 42
```

Generate the full dependency-ordered command plan (JSON argv lists):

```sh
python -m egomimic.benchmarks.libero.campaign \
  --data-root /datasets/arc_oat --output-root /runs/arc_oat --output campaign.json
```

The manual plan contains ARC placeholders; apply each suite/mode's frozen
replay settings before executing its ARC training command. The OSMO runner
above performs this verification and substitution automatically.

For one suite, the training stages are ordinary graph recipes. These ARC
examples use the measured LIBERO-10 selections in the
[dated replay report](results/libero_arc_timed_replay_20260921.md):

```sh
python -m egomimic.trainHydra +experiment=oat/libero_oattok \
  benchmark.suite=libero_10 benchmark.dataset=/datasets/arc_oat/libero_10.zarr \
  hydra.run.dir=/runs/tokenizer
python -m egomimic.trainHydra +experiment=oat/libero_oatpolicy \
  benchmark.suite=libero_10 benchmark.dataset=/datasets/arc_oat/libero_10.zarr \
  benchmark.tokenizer_checkpoint=/runs/tokenizer/checkpoints/last.ckpt \
  hydra.run.dir=/runs/oat
python -m egomimic.trainHydra +experiment=oat/libero_arc_dur_policy \
  benchmark.suite=libero_10 benchmark.dataset=/datasets/arc_oat/libero_10.zarr \
  benchmark.arc_waypoints=32 benchmark.arc_max_translation=0.8 \
  benchmark.arc_max_rotation_degrees=192 hydra.run.dir=/runs/arc_dur
python -m egomimic.trainHydra +experiment=oat/libero_arc_stk_policy \
  benchmark.suite=libero_10 benchmark.dataset=/datasets/arc_oat/libero_10.zarr \
  benchmark.arc_waypoints=32 benchmark.arc_max_translation=1.6 \
  benchmark.arc_max_rotation_degrees=128 hydra.run.dir=/runs/arc_stk
```

Checkpoint loading selects EMA explicitly and validates exact state keys.
Policy checkpoints include tokenizer architecture, frozen weights and data
normalization, so evaluation does not require the original tokenizer file.
Tokenizers and policies from different datasets/splits/normalizers are rejected.
Resume and standalone evaluation also reject changed camera/state observations.
Resume training with the standard `ckpt_path=...` option.

```sh
python -m egomimic.benchmarks.libero.cli reconstruct \
  --suite libero_10 --dataset /datasets/arc_oat/libero_10.zarr \
  --checkpoint /runs/tokenizer/checkpoints/last.ckpt --output reconstruction.json
python -m egomimic.benchmarks.libero.cli rollout \
  --checkpoint /runs/oat/checkpoints/last.ckpt --output /results/oat/libero_10
python -m egomimic.benchmarks.libero.cli rollout \
  --checkpoint /runs/arc_dur/checkpoints/last.ckpt --output /results/arc_dur/libero_10
python -m egomimic.benchmarks.libero.cli rollout \
  --checkpoint /runs/arc_stk/checkpoints/last.ckpt --output /results/arc_stk/libero_10
# After completing every suite for all three policies:
python -m egomimic.benchmarks.libero.cli compare \
  --arc-root /results/arc_dur --oat-root /results/oat --output comparison_dur.json
python -m egomimic.benchmarks.libero.cli compare \
  --arc-root /results/arc_stk --oat-root /results/oat --output comparison_stk.json
```

Use `--tokens K` for OAT closed-loop prefix ablations, in separate result roots;
ARC support-count policy ablations require training with the selected
`benchmark.arc_waypoints` (multiples of four for this UNet). Limited trials or
`reconstruct --limit N` are explicitly partial checks.

## Validation and remaining experiment work

On 2026-09-21, the corrected STK/DUR implementation passed **178 focused tests**
across targeted invocations, including the pinned upstream parity tests.
The [L40S smoke receipt](results/libero_arc_timed_gpu_smoke_20260921.json)
records source `df3f26f2`, all four real training stages, EMA reload,
ten paired short rollouts per policy, and 16 reconstruction windows. Its STK
configuration includes the three-axis velocity normalization bound.
These checks establish the training/evaluation path, not benchmark performance.
The [replay report](results/libero_arc_timed_replay_20260921.md) records the
separate R/D/M searches and their measured gaps to raw commands.

Local validation on 2026-09-19: **357 tests passed** across the four new test
modules and existing pipeline, planar/bimanual ARC, duration bridge, pose,
checkpoint and UNITE regressions. Source parity tests ran against the pinned
upstream checkout, with no parity-test skips. Ruff checks and Linux Python 3.11
dependency resolution for the selected extras passed.

`tests/test_oat_native.py` checks exact upstream source hashes and compares
tokenizer losses, gradients, codes, every requested prefix, register causality,
FSQ's complete codebook, the real vision encoder, policy cross entropy,
autoregressive logits/cached sampling, and EMA decay. Set `OAT_REFERENCE_ROOT`
to the pinned checkout to run the source comparisons; without it those tests
are explicitly skipped rather than replaced by a proxy.

Run the focused new tests with the OAT checkout and its pinned LIBERO submodule
available locally:

```sh
source emimic/bin/activate
OAT_REFERENCE_ROOT=/path/to/pinned/oat OMP_NUM_THREADS=1 python -m pytest \
  tests/test_oat_native.py tests/test_libero_arc.py \
  tests/test_libero_arc_timed.py tests/test_libero_replay.py \
  tests/test_libero_cluster.py tests/test_libero_benchmark.py \
  tests/test_oat_training.py -q
```

`tests/test_oat_training.py` runs tokenizer training, frozen-token policy
training, resume, EMA checkpoint reload and ARC policy training through the
actual shared `trainHydra` entrypoint on a tiny generated replay, in float32
and bfloat16 mixed precision on CPU. Data,
conversion, all-suite configuration, closed-loop control flow and strict report
validation have separate regression tests.

Real simulator integration was also checked with the pinned LIBERO source,
Robosuite 1.4.0 and MuJoCo 3.4.0: both trained smoke-test policies executed two
four-control-step rollouts with both cameras on the first LIBERO-10 task,
reusing the environment across trials. Their paired initial-state hashes
matched, and resetting seed 1000 twice reproduced the same simulator state.
These short, untrained-on-demonstrations rollouts
establish interface compatibility only.

The generated replay, mock-environment tests and short real simulator checks
are integration tests, **not benchmark results**. Full demonstration training,
CUDA/distributed parity and
the complete simulator score campaign require the GPU/data run environment.
No ARC/OAT success-rate or learned reconstruction claims are supplied before
those jobs run.
