# Using the graph integration

These commands describe the integration branch. The published PR stack and
remaining cutover gates are tracked in [PLAN.md](PLAN.md); they are not a claim
that EgoVerse main has already switched runtimes. Preserve old checkouts and
checkpoints while reviewing this stack.

## Install a reproducible environment

Use Python 3.11 and the checked-in lock:

```bash
uv venv emimic --python 3.11
source emimic/bin/activate
UV_PROJECT_ENVIRONMENT=emimic uv sync --locked
```

For PI, add `--extra pi05` to the sync command, then run
`python scripts/install_pi05_source.py`. Add `--extra alignment` for EgoBridge
or `--extra diagnostics` for latent reductions before installing OpenPI source.
The source installer is required after any synchronization that removes OpenPI.
The [dependency policy](DEPENDENCIES.md) explains the pinned source, patch
verification and separate Yam hardware environment.

## Select a complete recipe

The general `train_zarr_cartesian` entry point requires explicit model, data and
evaluator choices. For example, after configuring credentials and dataset paths:

```bash
python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  model=hpt_bc_flow_eva data=eva evaluator=eval_hpt
```

The PI convenience entry point retains its concrete ABC/6D/wrist-frame default
and requires an explicit pretrained weight path. Root retained recipes can be
selected on the general entry point too:

```bash
export EGOVERSE_PI05_WEIGHTS=/absolute/path/to/pi05_base_pytorch
python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  model=pi0.5_bc_eva data=eva_pi evaluator=eval_pi
```

PI weights/tokenizers must be available at the selected model's configured
paths. [OSMO_VALIDATION.md](OSMO_VALIDATION.md) describes the scheduled real-input
gate and hash-pinned artifacts; ordinary examples do not submit a GPU job.

`model.data_requirements` is the model's expected preprocessing contract.
Changing only the data's frame, stride, frame rate or tokenizer must fail when
it contradicts that contract. The nested PI recipes expose
`model.coordinate_frame=camframe|eef_frame`; it must match the data selection
and the evaluator's frame reversion. The model declaration also chooses the
deployment frame, so a station cannot silently reinterpret wrist-relative
actions as camera-frame actions. E1's `e1.policy_domain` is explicit experiment
metadata and binds its source identity and data contract.

The keypoint wrist entry point shares the retained keypoint architecture but
selects a wrist-frame model contract. Nested Yam/ARC models declare padded
human channels, tokenizer timing and sampling separately from episode filters.
Every shipped model advertising ready inference is covered by a declaration
check; a constructor check is not a real-weight training result.

## Evaluate or resume a bound checkpoint

Keep the run's resolved configuration, checkpoint, `data-context.json` and
`inference-config.yaml` together. Resume uses `mode=train ckpt_path=...` with
the original model/data semantics. Standalone evaluation uses the same saved
configuration with `mode=eval ckpt_path=...`; it restores normalization from the
checkpoint and opens only the configured validation data. Repointing validation
episode locations is allowed; changing their preprocessing is rejected.

No-hardware action inference and robot rollout share `load_bound_graph`.
The sequence API consumes already-assembled model-facing observations; the
robot adapter owns camera routing, calibration and history buffering. Sampling
and execution-prefix controls come from the saved model declaration.

Unbound historical checkpoints require their exact original source/environment.
There is no automatic general converter. See [CHECKPOINTS.md](CHECKPOINTS.md)
for strict loading, explicit weights-only initialization and supported aliases.

## Inspect or convert data

Use [DATA_TOOLS.md](DATA_TOOLS.md) for recorded video/PNG/array exports,
annotation overlays, HDF5 conversion and field audits. These tools do not need
a model or a fitted training normalizer. Full-episode video requests reject
partial evaluation loaders and preserve source timing, including sparse frames.

Before cutover, repeat [VALIDATION.md](VALIDATION.md) on the assembled EgoVerse
revision, including the scheduled GPU matrix. Individual graph PR checks do
not replace that combined-tree evidence.
