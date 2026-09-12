# Shared Eva and Yam robot runtime

Collection and rollout live in `egomimic/robot/`, outside embodiment directories.
`interface.py` defines their common contract; the existing Eva `ARXInterface`
and the new `yam/interface.py::YamInterface` implement it. Eva remains supported,
including its ROS workspace, controllers, kinematics, home behavior and cameras.

Yam uses the local i2rt hardware driver. There is no inference server, RPC
protocol, upstream runtime launcher or dependency on an `rl2_yam` checkout.
The Quest world-frame convention follows
[`rl2_yam/agents/quest_mapper.py`](https://github.com/GaTech-RL2/yam-pipeline/blob/1b1f9b12d300a41872b8a8c6c04f0c9f5f892f88/rl2_yam/agents/quest_mapper.py).
The i2rt extra pins the SDK revision used by that upstream snapshot:
`2b851d7f4511b6bda442db85e98ec1122835221b`.

## Station setup

Activate the project environment before Python commands. On a Linux Yam
station, install `.[yam]`; Eva uses `.[robot]` alongside its existing ARX/Aria/
RealSense installation. ADB must be installed and the Quest authorized. i2rt's
`ruckig` build requires `scikit-build-core<0.10`; supply that as a build constraint
when installing from source, as in the pinned SDK's own `pyproject.toml`.

Copy the appropriate YAML from `egomimic/hydra_configs/robot/` and set the
station's CAN channels, gripper variant, home poses and camera serials. Yam's
examples read `YAM_FRONT_SERIAL`, `YAM_LEFT_WRIST_SERIAL` and
`YAM_RIGHT_WRIST_SERIAL`. Eva's `robot.config_path` points to its existing camera,
CAN and gripper calibration YAML; it can point to a station-specific copy.
Paths in the examples are relative to the repository root.

### Quest app

Build the modified bundled app using
[the existing Android build instructions](../egomimic/robot/oculus_reader/app_source/README.md).
Install that build on the Quest with `adb install -r /path/to/new/teleop-debug.apk`.
The checked-in prebuilt APK predates this change and has **not** been rebuilt.
Keep the prior APK when preparing the new build.

The modified source emits `wE9ryARXWorld` messages containing tracked controller
poses in the VR tracking-origin frame. It also keeps the original `wE9ryARX`
headset-relative stream for other reader clients. Collection explicitly selects
the world stream, drops stale input after `quest.max_age`, and reports a startup
timeout if only the old app is installed. This prevents interpreting a head
frame as a world frame.

Quest +X/right, +Y/up, -Z/forward map to arm-base -Y, +Z, +X. A clutch captures
the controller's world pose and the measured robot pose. Position changes and
left-multiplied rotation changes are mapped through this fixed basis. Moving
one's head therefore does not move a stationary hand target. `side_yaw` and
`headset_yaw_degrees` calibrate the station orientation; they are fixed YAML
values, not the live head orientation. Tracking loss or a pose jump drops the
clutch; the next valid sample anchors at the measured arm pose.

## Collection and live views

```bash
python -m egomimic.robot.collect_demo --config egomimic/hydra_configs/robot/yam_collect.yaml
python -m egomimic.robot.collect_demo --config egomimic/hydra_configs/robot/eva_collect.yaml
```

Hold a controller's **grip** to move that arm; its index trigger closes the
normalized gripper. Press its joystick to reanchor. Each arm clutches
independently. Translation/orientation gains, filtering, jump thresholds and
joint velocity limits are YAML settings. IK failures hold the previous command.

The configured front, left wrist and right wrist cameras display continuously,
including while idle. Set `preview.enabled: false` for a station without a GUI.
Disconnected cameras appear as waiting and do not contribute recording rows.
A still scene is valid: freshness uses frame arrival times, not image hashes.

Default controls (Quest buttons can be reassigned in YAML):

| Action | Quest | Preview keyboard |
| --- | --- | --- |
| Start recording / finish episode | B | b |
| Stop and preserve an interrupted episode | X | x |
| Preserve current episode, then home | Y | y |
| Quit, preserving current episode | A | q / Escape |

Recording ends automatically at `recording.episode_length`. A held button only
causes one transition. Starting another episode chooses the next unused number.
Recording streams to disk, avoiding an episode-sized camera buffer in RAM.

### Preserved HDF5 format

Files remain `demo_<id>.hdf5`, with `sim=False`. Each row uses left arm then
right arm; unselected arms occupy zero slots. Gripper opening is 0 closed,
1 open. Cartesian poses are xyz + intrinsic ZYX Euler angles in radians +
gripper, in each arm's base frame.

| Dataset | Shape / dtype |
| --- | --- |
| `observations/images/<configured camera name>` | `(T,H,W,3)` uint8 RGB |
| `observations/joints`, `observations/joint_positions` | `(T,14)` float32 measured joints |
| `observations/eepose` | `(T,14)` float32 measured Cartesian poses |
| `actions/joints`, `action` | `(T,14)` float32 commanded joints |
| `actions/eepose` | `(T,14)` float32 Cartesian pose of the commanded joints |

Image chunks remain one frame. Drivers provide BGR; the writer converts to RGB,
as the previous collector did. All images and vectors are validated before a
row is appended. Files open exclusively, so existing data is never overwritten.
The additional `complete` attribute is false for interrupted episodes, including
X, home, quit and exceptions. These files remain available for inspection.

## Local graph rollout

```bash
python -m egomimic.robot.rollout --config egomimic/hydra_configs/robot/yam_rollout.yaml
python -m egomimic.robot.rollout --config egomimic/hydra_configs/robot/eva_rollout.yaml
```

Copy and fill the deployment YAML first. Supply the saved **fully composed**
training `.hydra/config.yaml`, checkpoint, device, and full normalization cache.
`trainHydra.py norm_stats_only=true norm_stats.save_cache_dir=...` exports
`norm_stats/norm_stats.json` with the complete `normalizer_state`; use the same
training recipe and dataset normalization settings. Old numeric-only caches must
be exported again. Rollout never instantiates the training datasets.

Inference constructs `PipelineAlgo`, binds its normalizer, strictly restores its
weights (and EMA only if requested), then runs graph inference. HPT and PI use
this same path. No previous ModelWrapper policy dispatch remains. The CLI's only
inference mode is `graph`.

The current shared graph adapter accepts bimanual Cartesian policies with Euler
or 6D rotations. Match camera-to-graph key names, resize dimensions, embodiment
ID (Eva bimanual 6, Yam bimanual 7), prompt and rotation encoding to training.
`base_T_model` is the per-arm transform from the **training proprio frame** into
that arm's base. Supply real calibration; the example deliberately leaves it
unset. For Yam wrist-frame recipes the proprio frame is the training station
frame; for Eva recipes it is normally the camera frame. Identity is valid only
when those frames actually coincide.

`action_frame: eef_frame` anchors every predicted pose in a chunk to the measured
EEF pose at inference time. `model_frame` instead uses the configured calibration.
Predictions are unnormalized once before decoding/frame reversion. For ARC,
configure `policy.adapter.decoder`, for example:

```yaml
decoder:
  _target_: egomimic.robot.arc_decoder.BimanualArcDecoder
  token_layout: e1_dur
  min_distance_unit: 0.4
  resampled_vector_length: 100
  dt: 0.03333333333333333
  action_horizon: 100
```

Use the codec and numbers from training. Supported layouts are `lab`, `e1_dur`,
`e1_logdur` and `e1_profile`. Decoding precedes IK. A graph plan executes at most
`execute_steps` before replanning. Both arms' commands must pass the joint step
limit before either command is sent. Camera loss pauses commands and discards
the old plan. Quit with q/Escape or Ctrl-C.

## Zarr replay and upload

```bash
python -m egomimic.robot.rollout --config egomimic/hydra_configs/robot/yam_replay.yaml
python -m egomimic.scripts.data_upload.yam_uploader --list ./demos/yam
python -m egomimic.scripts.data_upload.yam_uploader
```

The replay YAML selects an existing Zarr episode and its commanded joint and
gripper keys. Replace its `robot` section with the Eva station config to replay
Eva. Replay reads the store in mode `r`, honors `total_frames` instead of chunk
padding, consumes every selected frame once, and stops at EOF. `start`/`stop`
select a range. For a store with a single `(T,14)` joint array, replace `keys`
with `action_key: data/action` (or that store's actual array name). Align the
robot to the selected starting joints before replay; excessive steps are refused.

The uploader uses the existing interactive `Uploader` metadata and S3 workflow
under `raw_v2/yam/`. It uploads the same HDF5 files without re-encoding or deleting
them. Inspect interrupted files (`complete=False`) before selecting an upload
directory. `--list` performs no upload and needs no AWS credentials.

## Validation scope

`tests/test_robot_runtime.py` checks collection, format compatibility, world-frame
clutch control, stale tracking/cameras, previews, Eva/Yam interfaces, cleanup and
Zarr replay. `tests/test_robot_graph_policy.py` checks strict checkpoint loading,
normalization, camera/frame conversion, all ARC layouts and real MuJoCo FK/IK on
a small offline model. These checks open no physical robot or camera. The Android
APK build and operation on a real Quest/Eva/Yam station require those devices and
the Android/Oculus toolchain; they are separate from CPU validation.
