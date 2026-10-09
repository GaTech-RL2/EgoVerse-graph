# Shared Eva and Yam robot runtime

Collection and rollout live in `egomimic/robot/`, outside embodiment directories.
`interface.py` defines their common contract; the existing Eva `ARXInterface`
and the new `yam/interface.py::YamInterface` implement it. Eva remains supported,
including its ROS workspace, controllers, kinematics, home behavior and cameras.

Yam uses the local i2rt hardware driver. There is no inference server, RPC
protocol, upstream runtime launcher or dependency on an `rl2_yam` checkout.
The exact Quest mapper and streaming IK implementation are copied into this
repository from
[`rl2_yam/agents/quest_mapper.py`](https://github.com/GaTech-RL2/yam-pipeline/blob/1b1f9b12d300a41872b8a8c6c04f0c9f5f892f88/rl2_yam/agents/quest_mapper.py)
and
[`rl2_yam/utils/ik.py`](https://github.com/GaTech-RL2/yam-pipeline/blob/1b1f9b12d300a41872b8a8c6c04f0c9f5f892f88/rl2_yam/utils/ik.py).
Their source blobs and byte hashes are recorded in
[`yam/REFERENCE.md`](../egomimic/robot/yam/REFERENCE.md).
The i2rt extra pins the SDK revision used by that upstream snapshot:
`2b851d7f4511b6bda442db85e98ec1122835221b`.

## Station setup

Activate the project environment before Python commands. On a Linux Yam
station, install `.[yam]`; that extra includes the pinned official Dynamixel SDK
for USB GELLO input. Aria capture and processing use `.[robot,aria]`.
Those extras deliberately cannot be installed together: Project Aria 2.0.0 pins
`rerun-sdk==0.22.1`, whereas the pinned i2rt SDK requires `rerun-sdk>=0.32.2`.
ADB must be installed and the Quest authorized. i2rt's `ruckig` build requires
`scikit-build-core<0.10`; supply that as a build constraint when installing from
source, as in the pinned SDK's own `pyproject.toml`. The copied streaming solver
requires the reference-compatible `mink==1.1.0`; startup checks that API before
opening either robot driver.

Keep USB ADB when the station cannot reach the Quest's Wi-Fi address. Do not run
`adb tcpip` merely to work around a flaky cable: restarting `adbd` can require a
new headset authorization, and Wi-Fi ADB still fails on a segmented network.

Copy the appropriate YAML from `egomimic/hydra_configs/robot/` and set the
station's CAN channels, gripper variant, home poses and camera serials. Yam's
examples read `YAM_FRONT_SERIAL`, `YAM_LEFT_WRIST_SERIAL` and
`YAM_RIGHT_WRIST_SERIAL`. Eva's `robot.config_path` points to its existing camera,
CAN and gripper calibration YAML; it can point to a station-specific copy.
Paths in the examples are relative to the repository root.

The RL2 station has checked-in `yam_rl2_collect.yaml` and
`yam_rl2_replay.yaml` profiles derived from `yam-pipeline` commit
`1b1f9b12d300a41872b8a8c6c04f0c9f5f892f88`. They select
`can_follower_l`/`can_follower_r`, the measured rest poses and the D405 serials
reported by librealsense. The top camera `230322272195` matches the supplied
agentview calibration. The exact upstream `agentview_extrinsics.yaml` and
`workspace.yaml` are checked in beside the station profile; the `rl2yam`
calibration registry carries the same per-arm matrices and calibrated camera K.
The camera preflight verifies that every configured
RealSense serial is present before any robot driver is initialized. Visually
confirm the top/left/right views before the first physical session.

### Quest app

The bundled `teleop-debug.apk` is the tested world-frame build imported from
[`rohan-bansal/rohan-gello`](https://github.com/rohan-bansal/rohan-gello) commit
`e23154aa625fea10d934ffcec6c72c31db95da29` (`fix apk to be world frame`). Its
SHA-256 is recorded in
[`APK/PROVENANCE.md`](../egomimic/robot/oculus_reader/oculus_reader/APK/PROVENANCE.md).
The matching C++ source is checked in under `app_source/`.

That RAIL/Yam build emits tracked controller poses in the VR tracking-origin
frame under the established `wE9ryARX` log tag. It does not provide a separate
headset-relative stream. The bundled Python reader therefore defaults to and
only accepts `pose_frame="world"`, drops stale input after `quest.max_age`, and
matches the complete `wE9ryARX: ` marker so similarly named tags cannot be
misread.

`OculusReader` preserves an already installed package. To guarantee that the
Quest has this exact build, install it explicitly with:

```bash
adb install -r egomimic/robot/oculus_reader/oculus_reader/APK/teleop-debug.apk
```

Quest +X/right, +Y/up, -Z/forward map to arm-base -Y, +Z, +X. A clutch captures
the controller's world pose and the measured robot pose. Position changes and
left-multiplied rotation changes are mapped through this fixed basis. Moving
one's head therefore does not move a stationary hand target. The RL2 station's
required room-to-robot calibration is `headset_yaw_degrees: 180.0`; it is a fixed
YAML value, not the live head orientation. `side_yaw` may override it only for a
separately calibrated station. Tracking loss or a pose jump drops the
clutch; the next valid sample anchors at the measured arm pose.

## Collection and live views

```bash
python -m egomimic.robot.collect_demo --config egomimic/hydra_configs/robot/yam_rl2_collect.yaml
python -m egomimic.robot.collect_demo --config egomimic/hydra_configs/robot/yam_collect.yaml
python -m egomimic.robot.collect_demo --config egomimic/hydra_configs/robot/eva_collect.yaml
```

Hold a controller's **grip** to move that arm; its index trigger closes the
normalized gripper. Press its joystick to reanchor. Each arm clutches
independently. RL2 runs the reference 60 Hz command loop and local, posture-
regularized streaming IK while recording synchronized rows at 30 Hz. This avoids
the elbow-branch jumps produced by a fully converged IK solve on YAM's parallel
joints. Translation/orientation gains, filtering, jump thresholds and joint
velocity limits are pinned in the station YAML. IK failures hold the previous
arm command.

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

### USB/Dynamixel GELLO collection

The GELLO path is self-contained in EgoVerse and does not require another GELLO
checkout or Dynamixel Wizard at runtime. Wizard may still be used separately to
provision servo IDs, baudrate, and operating mode. The checked-in profile is
intentionally marked `gello.calibrated: false`: before a physical run, fill each
leader's stable `/dev/serial/by-id` path, six joint signs/offsets, and measured
open/closed gripper radians, then set that flag true. Do not use transient
`/dev/ttyUSB*` enumeration to assign left and right.

These commands are safe before the GELLOs are connected:

```bash
python -m egomimic.robot.collect_gello \
  --config egomimic/hydra_configs/robot/yam_rl2_gello_collect.yaml \
  --check-config
python -m egomimic.robot.collect_gello \
  --config egomimic/hydra_configs/robot/yam_rl2_gello_collect.yaml \
  --list-ports
```

Once connected and calibrated, the read-only device preflight checks stable USB
paths, both follower SocketCAN interfaces, and configured cameras without
opening motors:

```bash
python -m egomimic.robot.collect_gello \
  --config egomimic/hydra_configs/robot/yam_rl2_gello_collect.yaml \
  --check-devices
```

The physical collection command is:

```bash
python -m egomimic.robot.collect_gello \
  --config egomimic/hydra_configs/robot/yam_rl2_gello_collect.yaml
```

#### Capturing GELLO calibration

The leader-only calibration tool does not create or command a Yam follower,
open CAN, or open cameras. It opens exactly one configured USB leader, disables
its torque through the same passive adapter, and reads its seven raw encoder
positions. Static validation remains device-free:

```bash
python -m egomimic.robot.calibrate_gello \
  --config egomimic/hydra_configs/robot/yam_rl2_gello_collect.yaml \
  --check-config
```

After setting a stable path plus verified servo IDs and bus settings for one
side, put that leader at the intended six-joint zero pose and run:

```bash
python -m egomimic.robot.calibrate_gello \
  --config egomimic/hydra_configs/robot/yam_rl2_gello_collect.yaml \
  --arm left \
  --output ./gello_left_calibration.yaml
```

The terminal shows seven raw radians. Press **z** to capture all six joint zero
offsets, **o** at gripper-open, **c** at gripper-closed, and **1** through **6**
to toggle a joint sign after a direction check. Press **p** to print a YAML
snippet or **w** to write the requested new output file; it refuses to overwrite
one. Press **q** to exit. Repeat for the other arm, paste both snippets into the
station profile, verify directions in teleop one arm at a time, and
only then set `gello.calibrated: true`.

The leader adapter disables torque on every Dynamixel and never sends leader
position targets. Its two serial reads run concurrently. Followers start
disarmed: press **g** or **b** to arm both, **b** also queues an episode,
**x** to disarm and preserve an interrupted episode, **y** to home, and **q**
to quit. Absolute alignment is the station default: GELLO calibrated zeros map
onto each follower's configured home. Arming only enables the open-then-squeeze
trigger gate; that gate starts follower motion to home plus the current GELLO
offset from those zeros, and **b** starts recording at the same moment. Relative
alignment remains available to clutch at the measured follower pose instead.
All joint commands are velocity-limited. A missing, malformed, or stale sample
disarms both followers and requires another explicit activation. Use
`--no-cameras` for teleop-only operation; it disables preview and recording and
reads the same single-key controls from the interactive terminal without
requiring Enter.

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

Every normal `trainHydra.py` training run also writes
`checkpoints/inference-config.yaml`. Keep it beside the checkpoint when moving a
bundle. Rollout discovers either that generic name or the immutable
`<run-prefix>.inference-config.yaml` form automatically. The artifact is bound to
the exact resolved `model.pipeline` and contains only model semantics; station
camera maps, calibration, prompt, and safety limits stay in the rollout YAML.
For an older checkpoint, generate the same artifact without opening hardware:

```bash
python -m egomimic.pipeline.inference_config \
  --training-config /path/to/resolved-config.yaml \
  --output /path/beside/checkpoint/inference-config.yaml
```

Inference constructs `PipelineAlgo`, binds its normalizer, strictly restores its
weights (and EMA only if requested), then runs graph inference. HPT and PI use
this same path. No previous ModelWrapper policy dispatch remains. The CLI's only
inference mode is `graph`.

The rollout loop has one model-independent boundary: it passes the current
camera/proprio observation to `policy.predict()` and receives a canonical
Cartesian action trajectory. The selected model's `inference-config.yaml` owns
the rest of inference: optional observation history, checkpoint-stage matching,
sampler settings, native output shape, ARC decoding, and the final output
contract. A model with history keeps and resets that history inside the policy;
the rollout loop does not branch on model family. Profiles fail closed unless
exactly one checkpoint-declared stage/variant match is found. This keeps Flow,
ARC-velocity, ARC-duration, and diffusion checkpoints behind the same robot
interface while preventing a Flow sampler override from being applied to a
diffusion stage. `policy.inference_graph` remains only as a legacy fallback for
checkpoint bundles created before the model-owned artifact.

Each matched profile may expose a bounded `overrides` mapping. Those declarations
are the sole source of runtime controls shown by the dashboard: the UI does not
hard-code Flow, diffusion, or replanning fields. Every control declares its type,
bounds, default, label, and mutation target. Stage attributes cover values such
as Euler or diffusion steps; the policy-owned `replan_every` target chooses the
executable prefix while retaining the full canonical prediction for the action
overlay. Changing a control discards the old queued prefix and requests a fresh
plan. Selecting another checkpoint replaces the controls with those from its
matched profile. Undeclared names, invalid bounds, and private attribute paths
are rejected.

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
Predictions are unnormalized once before decoding/frame reversion. A generated
ARC artifact contains a matching `inference_graph.profiles` entry, for example:

```yaml
kind: egomimic.graph-inference
schema_version: 1
status: ready
model_pipeline_sha256: <sha256>
inference_contract_sha256: <sha256>
inference_graph_sha256: <sha256>
inference_graph:
    input:
      history_length: 1
    output:
      representation: cartesian
      shape: [100, 14]
    profiles:
      flow_arcdur:
        match:
          stage_target: egomimic.pipeline.stages_flow.FlowDenoiserStage
          variant: arcdur
        native_shape: [100, 16]
        overrides:
          inference_steps:
            label: Euler integration steps
            description: Number of Flow solver steps used for each prediction.
            type: integer
            min: 1
            max: 100
            step: 1
            default: 10
            target:
              kind: stage_attribute
              attribute_path: num_inference_steps
          replan_every:
            label: Repredict every
            description: Execute this many actions before requesting a fresh prediction.
            type: integer
            min: 1
            max: 100
            step: 1
            default: 30
            target:
              kind: policy_attribute
              attribute_path: replan_every
        adapter:
          decoder:
            _target_: egomimic.robot.arc_decoder.BimanualArcDecoder
            token_layout: e1_dur
            min_distance_unit: 0.4
            resampled_vector_length: 100
            dt: 0.03333333333333333
            action_horizon: 100
```

Use the codec and numbers from training. Supported layouts are `lab`, `e1_dur`,
`e1_logdur` and `e1_profile`. Decoding precedes IK. The inference policy returns
the profile-selected executable prefix; the rollout loop has no model-specific
replanning parameter. Both arms' commands must pass the joint step limit before
either command is sent. Camera loss pauses commands and discards the old plan.
Quit with q/Escape or Ctrl-C.

## Zarr replay and upload

```bash
YAM_REPLAY_PATH=/absolute/path/to/episode.zarr \
  python -m egomimic.robot.rollout \
  --config egomimic/hydra_configs/robot/yam_rl2_replay.yaml
python -m egomimic.scripts.data_upload.yam_uploader --list ./demos/yam
python -m egomimic.scripts.data_upload.yam_uploader
```

The replay YAML selects an existing Zarr episode and its commanded joint and
gripper keys. The RL2 profile directly reads `rl2_yam.episode.v1` stores with
`actions/joint_position` shaped `(T,2,6)`, `actions/gripper` shaped `(T,2)`, and
the root `arm_order` attribute. It rejects `complete=False`, honors
`committed_samples` (or `total_frames`) instead of chunk padding, consumes every
selected frame once, and stops at EOF. `start`/`stop` select a range.

The generic per-arm key mode remains available. For a store with a single
`(T,14)` joint array, replace `keys` with `action_key: data/action` (or that
store's actual array name). Replace the `robot` section with the Eva station
config to replay Eva. Align the robot to the selected starting joints before
replay; excessive steps are refused before either arm receives a command.

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
