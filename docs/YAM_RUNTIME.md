# Yam rollouts and Quest teleoperation

The hardware runtime is [GaTech-RL2/yam-pipeline](https://github.com/GaTech-RL2/yam-pipeline/tree/1b1f9b12d300a41872b8a8c6c04f0c9f5f892f88/rl2_yam),
pinned to `1b1f9b12d300a41872b8a8c6c04f0c9f5f892f88`. EgoVerse uses its Quest
mapper, recording, cameras, homing and rollout guards directly. The launcher
checks the revision and runs its existing uv environment with `--no-sync`.
It never changes that checkout or installs hardware dependencies during launch.
Use the upstream setup instructions for its submodules, APK, CAN interfaces,
station YAML, cameras and calibration. Keep this hardware environment separate
from the graph training environment.

## Teleop and collection

From an activated graph environment, point at a prepared checkout of the pinned
Yam revision and a commissioned station configuration:

```bash
export YAM_PIPELINE_ROOT=/path/to/yam-pipeline
python -m egomimic.robot.yam teleop --config /path/to/quest_teleop.yaml
```

`python egomimic/robot/collect_demo.py` now launches this same Quest runtime.
Use `--record-dir /path/to/new/recordings`, `--no-dashboard`, or `--no-cameras`
as supported by upstream. `--print-command` prints the exact command without
opening devices. Paths are resolved before entering the hardware checkout.
Saved data uses the upstream synchronized Zarr schema; it is not silently
labelled as the older Eva HDF5 format.

The prior Eva collector remains at `egomimic/robot/eva/collect_demo_legacy.py`,
along with the existing Eva ROS and Oculus sources. These are retained for
existing stations; the default collection entry point uses Yam Quest teleop.

## Graph policy server

Install the `yam` extra in the graph environment. The gRPC schema is copied
unchanged from the pinned upstream runtime; its checked-in Python descriptor
has the same wire format. No robot driver is imported by the server.

Copy `egomimic/hydra_configs/robot/yam_server.yaml` to a deployment file. Supply
the saved **fully composed** training YAML, graph checkpoint, normalizer cache,
MuJoCo arm/gripper model, ordered arm joint names and per-arm calibration.
PI also needs its optional model dependencies and constructor artifacts. Loading
binds the normalizer before strictly restoring online/EMA graph weights.
Use matching training artifacts; the server does not instantiate datasets.

New normalizer caches contain `normalizer_state` with the original key types,
shapes, embodiment IDs and values. For an older cache, export to a new directory
with the matching training recipe and its existing statistics:

```bash
python egomimic/trainHydra.py --config-name=<training-recipe> \
  norm_stats_only=true norm_stats.precomputed_norm_path=/path/to/old/norm_stats \
  norm_stats.save_cache_dir=/path/to/new/deployment-cache
```

This loads data to recover its schema, without constructing/training a model.
The server refuses a values-only cache because it lacks the camera/proprio
schema needed by the policy. Check the export path before rerunning it.

Both HPT and PI use `PipelineAlgo.forward_eval` and `pred_action`. The adapter
uses the most recent observation from Yam's two-frame history, converts RGB
to CHW `[0,1]`, computes TCP proprio using FK, and normalizes once. It then
unnormalizes predictions once, decodes ARC when configured, recomposes wrist
poses against the observed anchor, and solves for 24 absolute joint targets.
Grippers use upstream's convention: 0 closed, 1 open. Failed IK, invalid frame
matrices, nonfinite values or incompatible layouts fail the RPC before any
chunk is sent to the controller.

Deployment settings must match the data YAML:

| Training representation | Deployment settings |
| --- | --- |
| E1/Yam wrist-relative Euler | `embodiment_id: 7`, `rotation_mode: euler`, `action_frame: eef_frame`; proprio calibration maps the station world into each arm base |
| PI Eva-encoded wrist-relative 6D | `embodiment_id: 6`, `rotation_mode: 6D`, `action_frame: eef_frame`; use each calibrated `base_T_camera` from the selected training calibration YAML |
| Absolute camera/world poses | `action_frame: model_frame`; `base_T_model` must describe that same frame |

Map only the cameras consumed during training. PI camera names are
`base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`; HPT names are in the
training `camera_keys`/stems. Protocol history/frequency/chunk size are fixed by
the upstream client: two observations, 30 Hz, 24 joint actions. This adapter
currently advertises RGB only.

For an E1 duration model, replace `decoder: null` with the exact trained codec:

```yaml
decoder:
  _target_: egomimic.robot.arc_decoder.BimanualArcDecoder
  token_layout: e1_dur
  min_distance_unit: 0.4
  resampled_vector_length: 100
  dt: 0.03333333333333333
  action_horizon: 100
```

Other supported layouts are `lab`, `e1_logdur`, and `e1_profile`; their parameters
must match training. A decoder produces Euler poses before IK. Ordinary
time-indexed models use `decoder: null` and need at least 24 predicted rows.

Start inference on the model host:

```bash
python -m egomimic.robot.yam.server --config /path/to/deployment.yaml
```

The server defaults to loopback. Use a local SSH tunnel when the hardware runs
on another host. Start upstream cameras, then request predictions on the robot:

```bash
python -m egomimic.robot.yam cameras --config /path/to/quest_teleop.yaml
python -m egomimic.robot.yam rollout --config /path/to/quest_teleop.yaml \
  --server-address 127.0.0.1:18080 --chunks 1
```

Upstream defaults to prediction-only policy operation; enabling commands uses
its explicit `--execute` flag. Hardware initialization still opens robot/camera
devices. Its joint, velocity, gripper, workspace, collision, freshness and
deadline checks remain in the control runtime. Do not run simultaneous teleop
and policy control on the same station.

Validation covers CPU normalization/frame/codec boundaries, local RPC and the
actual upstream client, plus offline MuJoCo FK/IK. Physical Quest operation,
station calibration and robot rollouts still require commissioning on hardware.
