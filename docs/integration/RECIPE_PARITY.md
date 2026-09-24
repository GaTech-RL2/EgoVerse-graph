# Retained recipe decisions

Owner: graph integration. Source: EgoVerse
`ec5c903c067bf1b29bbff781bc414c6df2bcf1f2`; resolved source recipes are recorded
in `evidence/legacy-model-recipes.json`. These decisions describe the port;
they do not assert that the final assembled-tree/GPU cutover gates have passed.

## HPT

The root `hpt_*` recipes retain the source's camera sets, separate image
encoders and MLP stems, sinusoidal positions before the stems, domain-before-
shared token order, optimizer values, and 64 conditioning tokens. Current graph
campaigns retain their existing one-token default. The old `use_domain_embedding`
flag was assigned but never consumed in `HPTPolicy`; parity recipes therefore
do not introduce a new learned embedding. Flow heads retain their original
concatenated time conditioning and beta-distributed training times.

The shared-head recipe explicitly inserts human gripper channels and removes
them from human predictions. The separate-head recipe routes homogeneous
selectors through configured subgraphs, yielding 14D robot and 12D human
outputs. Routing has no embodiment registry in shared Python.

Two stale source settings are corrected explicitly:

- Several Cartesian cotrain models referred to `state_joint_positions` and
  `actions_joints`, while their shipped keymaps/transforms only produced
  Cartesian state/actions. The retained Cartesian recipes bind those actual
  keys. This is not a joint-action checkpoint converter.
- `hpt_cotrain_enc_dec_base` declared eight decoder context tokens against
  a 64-token trunk. Its deterministic decoder now declares 64 context tokens;
  its smooth-L1 loss retains beta 0.05. It rejects mismatched target shapes.

The keypoint model retains two 69D blocks: wrist pose(6) followed by MANO
keypoints(63). The restored head-frame and wrist-frame entry points select
their data transforms and shared graph evaluator explicitly. Empty camera
reversion lists declare identity; an absent conversion remains unsupported.
The source keypoint flow head intentionally projected 138D noisy actions into
64D before concatenating time features. The port explicitly opts into that
input bottleneck to preserve the architecture; it does not claim full-rank
reconstruction. All other recipes retain the denoiser's default rank check.

EgoBridge retains supervised/unsupervised Sinkhorn, action-MSE/soft-DTW matching,
block 13 representation capture, independent BC/alignment feature passes, and
independent BC/OT warm starts. Source groups and feature/action keys are explicit
in its loss graph. The freeze boundary detaches only BC conditioning; alignment
still reaches the earlier blocks. Warm starts count training batches starting
at one, as in the source, and now survive strict checkpoint resume. Install the
locked `alignment` extra for this optional path.

The fixed-fixture comparison exposed a source quirk: GeomLoss passes a leading
singleton batch axis into the supervised cost, but the old `unsqueeze(1)` /
`unsqueeze(0)` broadcasts aligned sample costs rather than a full pairwise
distance matrix. The retained recipe explicitly declares `legacy_broadcast`,
including the source's mask on debiasing self-cost terms, to reproduce its loss
and gradients. The general stage defaults to `pairwise`; changing an experiment
to that objective is an explicit YAML choice, not a silent parity claim. The
source YAML's `lambd` was unused (Python read `lambda`); its effective value was
0.5, which the port preserves.

`tests/fixtures/legacy_alignment_reference.py` contains the original methods
from the pinned source. Numerical loss/gradient comparisons cover unsupervised,
supervised-MSE and supervised-DTW modes. The configured two-source optimizer
test checks separate feature passes, reduction weights and strict resume.

Pooled and per-token language recipes exercise the actual Qwen stem, local
episode loader, normalizer, optimizer and shared video evaluator. Only the
downloaded text backend is replaced in CPU tests. Prompt fallbacks, frozen
encoder/trainable projection behavior and painted language videos are checked;
these are not real-weight Qwen GPU receipts.

## PI

The restored root PI recipes preserve the legacy normalized-YPR-to-6D action
encoding, rather than silently changing saved behavior to current campaign 6D
targets. Existing nested campaign recipes retain their explicit encodings.
Root PI recipes preserve the source cosine schedules. Canonical dataset camera
names map to OpenPI slots within the PI adapter, including missing-camera masks.
Empty annotations/default prompts produce the restored once-per-instance warning.

The eight-source CPU gate uses the real data adapter, normalization, graph and
optimizer, with a tiny substituted OpenPI backend. It is not evidence for a
real-weight PI GPU training run. A separate two-source optimizer/overlay test
checks EVA 14D and human 12D normalization together, both source losses, the
sampled prompt, and actual encoded videos. Resolved optimizer/scheduler fields
are compared with the pinned source snapshot for every root PI recipe.

The latent path uses an explicitly configured PI attention provider and the
model-independent `TokenDiagnosticsEval`. It captures first-call prefix and
action-expert attention keys, scopes hooks to one inference, and restores the
compiled sampler afterwards. PCA, CPU UMAP, PCA→UMAP, 2D/3D t-SNE, CSV/raw-key
archives, plots, and standalone rebuilds are retained. Each rank emits an
explicitly rank-local archive; projections are not advertised as a pooled
distributed reduction. CPU UMAP replaces the old automatic cuML selection to
keep this optional environment reproducible; projection coordinates across
backends are not claimed to be identical.

Latent selections are now `data.selection.mode=random|pairs|custom`, with
`pair_hashes`, `custom_hashes`, `frames_per_episode`, and `stride` in the data
YAML. Recorded pair hashes remain unchanged. Exact selections reject missing
episodes instead of silently returning a subset. The default batch is 16 and
the retained-token cap is 4,096 per source per layer, replacing the old batch
of 500 and unbounded retention. Both are explicit configuration knobs. Actual
episode/frame/token identities are preserved; repeated samples are deduplicated.
The diagnostic data selection remains a reproduction/visualization dataset,
not a held-out score split.

Multi-source tests exposed a schema bug: shape inference previously rewrote
every source with the currently inspected sample's dimensions. It now updates
only the declared sample identity and rejects conflicting schemas. Subsampled
datasets delegate normalization to their underlying data instead of adopting
`MultiDataset`'s class identity without its initialization.

## Data and time

The current Scale converter `external/scale/scripts/sfs_to_egoverse_zarr.py`
stores world-frame poses and `obs_head_pose`. The graph Scale keymap incorrectly
disabled head poses while selecting the world-to-camera transform. It now
loads them, consistent with the converter and retained source tests. Missing
head poses are not replaced with an invented identity frame.

The old Cartesian models predict 100 interpolated rows, not 100 control-rate
steps. EVA's 45-row window covers source indices 0..44. Human windows have 30
rows; stride three retains indices 0..27, while stride one retains 0..29.
The restored model declarations describe those time grids and use an explicit
uniform-grid decoder to return 45, 28 or 30 control-rate rows at 30 Hz. Target
construction and interpolation during training are unchanged. This distinction
also bounds the declared replanning control. Consumers must select the model
declaration matching their data stride and coordinate frame.

Old-only data YAML retains its recorded selections and train/validation groups.
Ports change the configured DataModule and transform argument API, not episode
contents. Some historical recipes intentionally share train and validation
data; those are visualization/reproduction recipes, not held-out score claims.

## Evidence

- `evidence/retained-recipe-steps.log`: eight HPT/stubbed-PI source cases with
  two optimizer steps, finite gradients, inference, strict checkpoint loading,
  and disjoint synthetic train/validation episodes; 28 tests including planar
  config regressions passed.
- `evidence/eval-contract-tests.log`: 121 evaluator/data-context tests passed.
- `evidence/wheel-tests.log`: wheel contents and fresh outside-checkout install
  passed, including EVA model XML and referenced meshes.
- `evidence/full-foundation-tests.log`: initial broad run, including failures
  subsequently repaired; do not read it as the final assembled-tree receipt.

Real-weight GPU checks, fixed-fixture old/new
comparisons, and destination conflict resolution remain mandatory gates.

## Initialization

`PipelineAlgo.initialization` selects a stable stage and module path, an exact
source namespace and a required source SHA-256. All keys, shapes, dtypes and
finite values are preflighted before any target changes. Hugging Face sources
require an immutable commit. `TrainabilitySchedule` applies declared freeze or
unfreeze steps and replays them at checkpoint restore; optimizer membership
retains parameters that will unfreeze later. This is weights initialization,
not a converter for arbitrary legacy checkpoints.
