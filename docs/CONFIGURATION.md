# Configuration guide

Start with a task experiment, then follow its Hydra `defaults` links. Folder
names organize recipes; Python `_target_` declarations identify implementations.
The generic pipeline does not choose behavior from a model or task name.

```text
egomimic/hydra_configs/
├── experiment/
│   ├── arc_real/
│   │   ├── stationary_{time,duration,velocity}.yaml
│   │   ├── towels_{time,duration,velocity}.yaml
│   │   └── earbuds_{time,duration,velocity}.yaml
│   ├── abc_arc/robot_bc/     retained ABC-lab recipes, including multitask
│   └── arc_sim/              PushT, LIBERO, OAT and synthetic recipes
├── data/                     episode selection, transforms and loaders
├── model/
│   ├── base/                 shared model definitions and settings
│   ├── hpt/                  HPT recipe leaves
│   ├── pi/                   PI recipe leaves
│   └── …                     distinct non-HPT/PI model families
├── model_contract/           shared deployment and data contracts
├── evaluator/
│   ├── eval_open_loop_sim.yaml
│   ├── eval_cartesian_*.yaml  behavior/frame-specific metrics and videos
│   ├── _base/                inherited evaluator defaults
│   └── viz/                  reusable renderers
└── hydra/launcher/            cluster allocation, separate from experiment
```

## Select and inspect one experiment

The three real tasks each expose `_time`, `_velocity` and `_duration`:
nine standard entry points, preserving the current task pools and protocols.
Every leaf directly links model, data,
evaluator, trainer, callbacks and logger. The model owns the optimizer and
scheduler. Real recipes do not inherit other experiments: each contains its
task/run overrides and imports reusable definitions from the component groups.

Activate the project environment, configure required data-cache/weight paths,
then inspect without running training:

```bash
python egomimic/trainHydra.py \
  +experiment=arc_real/stationary_velocity --cfg job --resolve
```

The [experiment menu](../egomimic/hydra_configs/experiment/README.md) records
task pools and numeric defaults. The top-level experiment directories `abc/`
and `abc_arc/human_bc/`, and `arc_real/arcdur_abstime/`, are removed. Their old
commands are historical, not aliases for the retained task recipes. The
removed absolute-time recipe menu does not remove duration mode:
`arc_real/<task>_duration` uses absolute waypoint seconds.

For an authorized cluster launch, use that same experiment with `-m` and an
explicit approved `hydra/launcher=...` selection. Verify the resolved resource
request, effective global batch, pinned data and normalization before launch.
See [training.md](../training.md) and [cluster operations](cluster-pipeline.md).

## What belongs in each layer

| Layer | Owns | Does not imply |
| --- | --- | --- |
| `experiment/arc_real/<task>_<mode>` | Complete component selection, task values and run schedule | Another task-specific Python runtime |
| `data/` | SQL filters or pinned episode lists, splits, transforms, loader behavior | The model architecture |
| `model/hpt/`, `model/pi/` | Model graph and family-specific recipe overrides | A model-specific evaluator implementation |
| `model/base/` | Identical inherited graphs/settings factored out of leaves | An experiment with data and evaluation selected |
| `model_contract/` | Reusable inference/deployment and preprocessing declarations | Permission to reinterpret a checkpoint |
| `evaluator/` | Metrics, frame reversion, execution protocol and artifacts | Training-time tokenization choices inferred from a model name |

`model/base/` contains shared HPT flow, cotrain, keypoint and Qwen definitions,
PI defaults, and YAM HPT/diffusion settings. Distinct architectures and
contracts remain distinct; only equivalent definitions should be shared.
The Python networks stay in `egomimic/models/`, and configured graph stages
stay in `egomimic/pipeline/`. Moving YAML does not move or resize a network.

### Direct component links; no shared experiment parents

`experiment/_shared/` and the real experiment-parent chain are removed. Every
one of the nine `arc_real/` recipes selects its six components directly. Model
dimensions live in `model/base/real_*_settings`, training defaults in
`trainer/real_*`, and the checkpoint schedule in
`callbacks/real_yam`. Task identity, dataset selection, caps and run
overrides remain visible in the experiment itself.

Canonical clock dimensions/provenance live in
`model_contract/arc_clock_settings`. YAM ARC experiments explicitly
select `model/hpt/yam_300m_arc` and the matching `data/abc_arc/*_arc`
adapter; the data adapter retains the original pinned pool and binds the
model-declared transforms through `data/_base/yam_arc`.

Component inheritance still deduplicates model graphs and dataset lists.
There is no earbuds-to-towels-to-stationery experiment inheritance, and no
extra experiment file is needed to launch any real recipe. Simulation recipes
retain their existing parent structure under the renamed `arc_sim/` folder.

## Evaluator selection

| Selection | Purpose |
| --- | --- |
| `eval_open_loop_sim` | Recorded-episode execution simulation for real baseline and ARC recipes; metrics, DTW and executed-chunk videos |
| `eval_bimanual_cartesian` | Time-indexed chunk errors and overlays, without sequential execution/replanning |
| `eval_cartesian_chunk_metrics` | Metrics-only chunk comparison using the configured YAM batch budget; no renderer |
| `eval_cartesian_camera`, `eval_cartesian_wrist` | Cartesian pose metrics with the chosen frame reversion |
| `eval_cartesian_*_prompt` | Corresponding metrics with sampled-prompt overlays; `6d` variants use continuous-6D rotations |

For example, the old `eval_hpt` selector becomes `eval_cartesian_camera`, and
the old root `eval_pi` becomes `eval_cartesian_camera_prompt`. A model name is
not sufficient to choose the right evaluator: frame, rotation encoding,
ground-truth key and execution protocol must match the data/model contract.
Renderer-only defaults live in `evaluator/viz/`; common evaluator definitions
live in `evaluator/_base/`.

`eval_open_loop_sim.yaml` groups settings by execution, canonical codec,
complete-episode scoring and artifacts. Real baselines predict 100 actions,
execute the first 25 at 30 Hz, then replan against the next recorded
observation. ARC keeps a waypoint prefix and ends when the first moving stream
finishes. This is open-loop replay of recorded observations, not closed-loop
physics or robot evaluation. Ordered complete episodes and matching video/GT
coverage are part of its contract, not optional batch-sampling details.

The [Cartesian clock contract](ARC_CLOCK_CONTRACT.md) defines velocity versus
absolute-seconds duration, four stream layouts and measured-proprio-to-waypoint arrival semantics.
Planar and LIBERO codecs have their own contracts; the removed Cartesian
legacy options must not be reintroduced into the open-loop YAML.

## Provenance and migration

Use the new selectors for new compositions. Keep the original source commit,
resolved config, normalization and checkpoint for an existing run. Do not
resume a different action representation under the same run identity or infer
decoder compatibility from M×20/M×18/M×16 shape alone. YAML organization does
not migrate trained weights. See [checkpoint rules](integration/CHECKPOINTS.md)
and [current real-task recipes](STATTEMPO.md).

The exact old-to-new model, evaluator and data selector map is in
[`config_cleanup_migration.json`](config_cleanup_migration.json). Removed
experiment menus have no automatic replacement: choosing another task recipe
could change its data pool or training protocol.
