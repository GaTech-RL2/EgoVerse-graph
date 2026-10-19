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

GELLO recording has no automatic frame cap by default
(`recording.episode_length: null`). The dashboard's **Episode limit** checkbox
enables/disables automatic saving; enter a positive frame count and press
**Apply** to change it live. The applied limit and recorded-frame count appear
below the cameras. Lowering the limit to or below an active take's current
count saves that take. With the limit off, use the normal start/save control.
After deploying this feature into another checkout, finish the current take
and restart teleop normally to load the backend; source edits alone do not
change a running collector. The controls stay disabled until the collector
confirms that it supports episode limits. Recording still streams to disk,
and motion/input safety checks are unchanged.

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

### Virtual YAM with GELLO

Test the current calibration against a MuJoCo dual-YAM view without opening
physical followers, CAN, cameras, or Quest:

```bash
python -m egomimic.robot.virtual_gello_teleop \
  --input gello \
  --config egomimic/hydra_configs/robot/yam_rl2_gello_collect.yaml \
  --alignment absolute \
  --duration 600
```

Real-input mode opens both passive leaders with torque disabled and uses the
shared GELLO reader/controller. The combined profile supplies joint IDs,
offsets, signs, gripper endpoints, input smoothing, speed limits and follower
home. Absolute mode commands virtual home plus the calibrated leader angles;
relative mode (`--alignment relative`) clutches at the starting pose and can
therefore hide a zero-offset problem. The virtual runner activates both
simulated arms immediately. Close the MuJoCo viewer to release the leader ports
before starting physical teleop.

For a test with no connected devices or display, use generated inputs:

```bash
python -m egomimic.robot.virtual_gello_teleop --headless --duration 1
```

Optional `--initial-left` / `--initial-right` take six simulated joint angles;
they never read or move a physical arm. `--left-calibration` and
`--right-calibration` are optional legacy overlays; omit them when checking the
combined profile. The viewer performs kinematic updates, not a hardware or
collision-safety validation. Do not wrap full-turn calibration errors to make
the pose look correct; the first real-input angles are printed to the terminal.

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

Check the complete checkpoint load on CPU before opening station devices:

```bash
python -m egomimic.robot.rollout \
  --config egomimic/hydra_configs/robot/yam_rl2_towels_hpt300_rollout.yaml \
  --check-model
python -m egomimic.robot.rollout \
  --config egomimic/hydra_configs/robot/yam_rl2_earbuds_hpt300_rollout.yaml \
  --check-model
```

`--check-model` forces CPU in memory, restores the bound graph and strict
checkpoint weights, prints a JSON result, and exits before creating any robot,
CAN, camera, or dashboard. It leaves the YAML unchanged. `--check-config`
validates settings only; it does not test checkpoint compatibility.

The RL2 HPT300 profiles select the 180k-step towel and earbud bundles from
2026-10-12; the default `yam_rl2_hptflow_rollout.yaml` selects towels. Their
schema-2 inference sidecars require the matching bound-inference runtime.
The older tokenshapes/IK runtime fails constructing `FlowDenoiserStage` with
`unexpected keyword argument 'clip_sample_range'`. Use the matching runtime;
removing the keyword alone leaves further graph/schema incompatibilities.
Keep each bundle's config, data context, normalizer and inference sidecar
together. These profiles retain the existing IK controls: MuJoCo by default,
optional TRAC-IK, and pose relaxation disabled by default.

```bash
python -m egomimic.robot.rollout --config egomimic/hydra_configs/robot/yam_rollout.yaml
python -m egomimic.robot.rollout --config egomimic/hydra_configs/robot/eva_rollout.yaml
```

Copy and fill the deployment YAML first. Supply the saved **fully composed**
training `.hydra/config.yaml`, checkpoint, device, and the training run's
`data-context.json` as `policy.normalizer_path`. This full context includes
preprocessing and normalization; a statistics-only cache is insufficient for
bound deployment. Rollout never instantiates the training datasets.

Every normal `trainHydra.py` training run also writes
`checkpoints/inference-config.yaml`. Keep it beside the checkpoint when moving a
bundle. Rollout discovers either that generic name or the immutable
`<run-prefix>.inference-config.yaml` form automatically. The artifact is bound to
the exact resolved `model.pipeline`, inference declaration and data context;
station
camera maps, calibration, prompt, and safety limits stay in the rollout YAML.
Do not put `inference_graph`, inferred family profiles or sampler defaults in
the station policy. The loader rejects these obsolete settings rather than
silently ignoring them. The checkpoint picker prefers `data-context.json`
(including run-prefixed names); a historical `norm_stats.json` filename does
not exempt its contents from the full-context and checkpoint-binding checks.
The RL2 template requires explicit bundle paths through
`YAM_HPTFLOW_TRAINING_CONFIG`, `YAM_HPTFLOW_CHECKPOINT`, and
`YAM_HPTFLOW_NORMALIZER` (the full data-context file, despite the legacy variable
name). There is no fallback to an old unbound checkpoint.
To verify and re-export a sidecar for an already-bound checkpoint:

```bash
python -m egomimic.pipeline.inference_config \
  --training-config /path/to/resolved-config.yaml \
  --checkpoint /path/to/model.ckpt \
  --data-context /path/to/data-context.json \
  --output /path/beside/checkpoint/inference-config.yaml
```

Older unbound checkpoints require their immutable original runtime or an
explicitly verified migration. Adding a sidecar alone cannot establish their
preprocessing semantics. See [checkpoint policy](integration/CHECKPOINTS.md).

Inference constructs `PipelineAlgo`, binds its normalizer, strictly restores its
weights (and EMA only if requested), then runs graph inference. HPT and PI use
this same path. No previous ModelWrapper policy dispatch remains. The CLI's only
inference mode is `graph`.

The rollout loop has one model-independent boundary: it passes the current
camera/proprio observation to `policy.predict()` and receives a canonical
Cartesian action trajectory. The selected model's `inference-config.yaml` owns
the rest of inference: optional observation history, stable stage identifiers,
sampler settings, native output shape, ARC decoding, and the final output
contract. A model with history keeps and resets that history inside the policy;
the rollout loop does not branch on model family. Profiles fail closed unless
exactly one model-declared deployment profile resolves to its stable stage ID. This keeps Flow,
ARC-velocity, ARC-duration, and diffusion checkpoints behind the same robot
interface while preventing a Flow sampler override from being applied to a
diffusion stage. Old inferred profiles are not a deployment fallback; retain the
original runtime or perform a verified migration for unbound bundles.

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
ARC artifact contains the schema-2 model declaration and immutable binding
hashes; it is generated, not hand-written. The explicit model-owned example
[`station_arc_duration.yaml`](examples/station_arc_duration.yaml)
declares the decoder, per-interval timing, sampler budget, waypoint cap,
fastest-stream toggle, replay tempo, and hold tempo. Adapt its frame, observation
keys, stage identifier, native shape and codec to the exact recipe **before**
training/exporting; this example cannot migrate existing weights by itself.

A `decoder_attribute` control binds only to that explicitly configured decoder.
No ARC controls are injected by Python. The generic inference consumers use the
decoder's optional `execution_steps()` capability for a prefix; otherwise
`replan_every` remains authoritative. Setting changes are transactional even
when a property rejects a value. Keep percentage bounds compatible with the
waypoint count (e.g. step 2 for M=50).

The decoders support E1 duration/log-duration/profile, hybrid and tri clocks,
event, reference and projection layouts, the lab mean token and the multistream
interval tokens, all through the current training tokenizers. Duration is the
decoder default; legacy mean timing is not a new-training default.
Decoding precedes IK. The inference policy returns
the profile-selected executable prefix; the rollout loop has no model-specific
replanning parameter. Both arms' commands must pass the joint step limit before
either command is sent. Camera loss pauses commands and discards the old plan.
Quit with q/Escape or Ctrl-C.

### One codec: the training tokenizer

`BimanualArcDecoder` decodes every layout with the tokenizer that trains it, the
live `egomimic.rldb.zarr` codecs (`e1_arc_tokenizer`, `arc_projection_tokenizer`,
`arc_length_tokenizer`): E1 duration / log-duration / profile, hybrid and tri
clocks, event and reference layouts, projection layouts, absolute waypoint
timing, the hybrid rotation budget and speed floors. `FirstStreamArcDecoder`
executes the multistream interval tokens of `arc_length_tokenizer` (wide,
stacked and four-clock layouts) to their first moving stream, and
`BimanualIntervalArcDecoder` decodes them whole. No checkpoint-era tokenizer
copies are carried: a bundle trained by another revision of a tokenizer needs
that revision's runtime. Replay tempo and fastest-stream termination run on the
codec's own clocks through decode-only helpers (`detokenize(times=...)`,
`stream_clocks`, `stream_ends`); tokenize is untouched.

The shared E1 contracts (`model_contract/e1_wrists.yaml` and the projection
model leaves) declare the decoder with `first_stream: 1` and
`execute_percent: 50`, and expose the four `decoder_attribute` controls from `model.arc_decoder_controls`
(`multistream_fastest_stream`, `execute_waypoint_percent`, `arc_speed_percent`,
`arc_hold_speed_percent`) next to `inference_steps` and `replan_every`. The
`time` contract declares a `TimeChunkRetimer` so a time-indexed policy gets the
same replay / hold tempo controls; at 100 % it returns the chunk untouched.
Bundles exported before 2026-10-18 carry no decoder controls: they decode as
before and show only the sampler and replanning controls. ARC models
trained from the stack keep the recipe's own sampler default
(`model.sampling_steps`, 50 for the HPT-Flow E1 models); the station's former
20-step ARC default is a dashboard setting, not a contract value.

### Rollout capture and active time

The dashboard's **Record episode (d)** control captures the same top-level HDF5
observations/actions/cameras as GELLO collection. It also records policy plans,
per-row timestamps and inference settings under `rollout/`. Save a take as
success, failure or unlabeled; discard removes only that take. **Recorded
episodes** lists finalized takes. MP4 capture remains available independently.

The background writer uses a bounded queue. Slow disk, a full queue or low free
space ends capture as incomplete without adding robot commands or blocking the
control loop. Startup/save metadata still requires small filesystem operations;
continuous image writes and finalization run on the writer thread. The RL2
profile sets a 20 GB free-space floor. Plan records distinguish the selected
executable prefix from actual commands; per-row commands are the execution
evidence. Restart saves an active take as unlabeled; camera/model changes and
abnormal exit preserve it as incomplete. Never overwrite existing data.

**Rollout time** is active session time since Continue: explicit pause and
camera/model-load holds freeze it; browser reconnect preserves it; Restart
resets to zero. This is not a per-action motion-time measurement and does not
subtract inference latency or safety decision wait time.

After a failed checkpoint selection, execution remains locked until a valid
model loads. The old loaded model is never resumed implicitly.

### Reset guides

The **Reset guide** menu above the live cameras on both dashboards offers
**Off** (default) and saved placement templates. The RL2 earbuds template shows
adjacent left-earbud, case and right-earbud regions projected onto the table in
`front_img_1`. It is a visual annotation, not surveyed calibration or an
automatic reset: recheck placement if the camera or table moves. Guides are SVG
browser annotations only; raw camera pixels, recorded videos and episodes,
policy inputs and robot commands are unchanged. They hide when disconnected,
awaiting frames or on an incompatible image aspect ratio, track the image's
letterboxing, and survive a websocket reconnect until the page is refreshed.

Template geometry lives in `egomimic/hydra_configs/robot/reset_overlay_templates.yaml`.
To add or change a template run `python -m egomimic.scripts.build_reset_guides`
and commit both generated dashboard catalogs and the rollout renderer;
`--check` detects a stale catalog or renderer drift. The renderer is
task-agnostic (its canonical block lives in `teleop_dashboard_static/app.js`;
the builder synchronizes rollout's copy); keep template coordinates out of
JavaScript. Refresh the browser to load these static-only changes; no
collection or robot session restart is needed.

### Optional rollout IK pose relaxation

`ik_relaxation.enabled` defaults to `false`. On YAM Cartesian rollouts the
**Allow small IK pose errors** dashboard checkbox changes it on the control
thread, discards the queued chunk and requests a fresh plan. It is independent
of checkpoint inference settings and persists across model changes and Restart
within the process; a process restart restores the YAML value. The checkbox
shows the applied state and the configured bounds.

```yaml
ik_relaxation:
  enabled: false
  position_tolerance_m: 0.005
  rotation_tolerance_deg: 3.0
```

Strict endpoint IK is always tried first. Only after it fails does a bounded
retry use tolerance-scaled position and orientation errors. A candidate must
satisfy **both** FK error bounds relative to the original target; a low summed
cost alone is insufficient. The normal paired joint-step guard still applies.
The fallback does not change the target gripper opening, the pinned Quest
solver, or joint replay. Configuration is validated without devices, including
while disabled: position tolerance must be in (0, 0.02] meters and rotation in
(0, 15] degrees. These are absolute envelopes, not increments to strict limits.
The defaults are experimental and require physical validation on the task.
This relaxes all orientation axes equally; it does not plan around an
unreachable fold apex or guarantee recovery from a singularity.

### Differential IK recovery

`yam_rl2_hptflow_rollout.yaml` and the two HPT300 profiles enable
`ik_recovery`; other profiles default to disabled. Normal successful endpoint
IK is unchanged. When either arm's strict solve fails, both arms use local,
posture-regularized differential steps toward the same queued Cartesian row.
The pinned Quest `streaming_ik.py` and teleop settings are unchanged.

```yaml
ik_recovery:
  enabled: true
  max_joint_velocity: 1.0
  position_scale_m: 0.05
  rotation_scale_rad: 0.5
  damping: 0.01
  posture_weight: 0.05
  max_ticks_per_target: 30
```

Each solve is a six-joint bounded least-squares problem with posture continuity,
extra regularization on joints 2-4, scaled Cartesian residuals and stronger
damping near singularity. Its bounds intersect the model joint limits with
one tick of motion around BOTH measured joints and the last sent command.
The rate is the smaller of the recovery rate and the rollout rate, with the
actual configured rollout period. A bounded nonlinear backtracking check
requires Cartesian error reduction; solver failure, incompatible bounds or
no progress commands neither arm and discards the plan.

The row remains queued during partial progress. Both grippers retain their
last commanded opening until both commanded endpoints reach the strict IK
tolerances, then the original row's gripper targets apply. No later row can
execute during recovery. Pause, restart, camera loss and new-model events
still clear the queue. After `max_ticks_per_target` commanded recovery steps,
the plan is discarded and the policy replans; this is not a session-wide
stall timeout. A permanently infeasible target may still require operator
intervention. Recovery changes timing and may affect inter-hand geometry;
it does not enforce a cloth constraint or provide collision avoidance.
Checkpoint plan metadata records the recovery settings. Set
`ik_recovery.enabled: false` to restore the endpoint-only rollout behavior.

### Optional TRAC-IK endpoint solver

The dashboard **Endpoint IK** selector and the YAML choose the endpoint solver:

```yaml
ik_solver:
  backend: mujoco  # default; select pytracik to opt in
  timeout_s: 0.005  # per arm, (0, 0.02] seconds
  solver_type: Distance  # also Speed, Manip1, Manip2
```

Install the optional dependency in the station environment with
`python -m pip install pytracik==0.0.3` (or the `yam-tracik` project extra).
`--check-config` verifies that it imports when selected in YAML. A missing
library or failed chain check is reported explicitly; a failed dashboard switch
keeps the previous backend. Settings persist across Restart and checkpoint
changes in that process; a new process uses YAML again.

TRAC-IK replaces strict endpoint solving. The adapter generates a serial URDF
from the **loaded** MuJoCo model's joints, limits and TCP, then checks TRAC-IK FK
against MuJoCo on 13 configurations before selection. Every returned endpoint is
checked for finite joints, joint limits, and the same independent Cartesian
acceptance bounds (normally 2 mm and 0.02 rad). Both arms must still pass the
rollout's joint-step guard. MuJoCo remains responsible for FK, optional bounded
pose relaxation and bounded differential recovery. Backend changes run on the
rollout thread, prepare both arms before switching, discard queued waypoints,
and resume from a fresh observation. The selected solver settings are recorded
with each plan.

A reproducible CPU comparison, with no robot-driver imports:

```bash
python -m egomimic.robot.yam.benchmark_ik \
  --xml-path /path/to/the/station/yam_linear_4310_d405.xml \
  --count 100 --repeats 3 --seed 20261011 \
  --solver-type Distance --timeout-s 0.005 --max-step 0.4 \
  --output /tmp/yam-ik-comparison.json
```

Station CPU result (2026-10-11, pytracik 0.0.3, MuJoCo 3.4.0; valid endpoints /
endpoints within the 0.4-rad command step, out of 300 per group): TRAC-IK
solved 300 / 300 high-TCP targets against MuJoCo's 273 / 273 and rescued 72
failed reachable attempts while missing three on its time budget; typical
reachable-case medians were 5.5 ms per arm for TRAC-IK versus 0.6-1.8 ms for
MuJoCo. MuJoCo stays the default; the comparison is synthetic and does not
establish physical fold success.

### Station monitors and checkers

- `egomimic/robot/camera_align/` watches the front D405 for camera or desk
  motion against a saved reference (SIFT roll / tilt / pan, ArUco poses) and
  serves a loopback page on port 8090. `run.sh` opens the camera directly
  (`--source realsense`, so no rollout dashboard is needed; stop it before a
  rollout or `collect_gello` starts), `SOURCE=stream ./run.sh` reads the rollout
  dashboard's websocket instead. See its README.
- `python -m egomimic.robot.hdf5_tempo <episodes dir>` buckets recorded GELLO
  episodes by tempo and skips an unusable file instead of aborting the scan.
- `python -m egomimic.robot.rollout --config <profile> --check-model` strictly
  restores the selected bundle on CPU and exits before any device opens.

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
