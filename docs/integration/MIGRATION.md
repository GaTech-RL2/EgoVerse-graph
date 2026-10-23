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

Prefer a complete task entry point; its defaults explicitly select model, data,
evaluator, trainer, callbacks and logging. Inspect without launching:

```bash
python egomimic/trainHydra.py \
  +experiment=arc_real/stationary_velocity --cfg job --resolve
```

See [the configuration guide](../CONFIGURATION.md) for the arc_real/arc_sim menus,
family-organized model configs and behavior-named evaluators. The general
`train_zarr_cartesian` entry point also supports explicit component selection.
For example, after configuring credentials and dataset paths:

```bash
python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  model=hpt/bc_flow_eva data=eva evaluator=eval_cartesian_camera
```

The PI convenience entry point retains its concrete ABC/6D/wrist-frame default
and requires an explicit pretrained weight path. Retained legacy recipes can be
selected on the general entry point too:

```bash
export EGOVERSE_PI05_WEIGHTS=/absolute/path/to/pi05_base_pytorch
python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  model=pi/bc_eva_legacy data=eva_pi evaluator=eval_cartesian_camera_prompt
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
actions as camera-frame actions. Contracts bind the selected model and data
semantics without inferring them from folder names. Removed E1 recipes must
use their original source; they are not current launch choices.

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
