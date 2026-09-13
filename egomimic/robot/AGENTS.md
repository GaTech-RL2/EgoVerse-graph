# Robot navigation

The root AGENTS.md applies here. Start with [YAM_RUNTIME.md](../../docs/YAM_RUNTIME.md)
for Eva/Yam setup, Quest app build requirements, data format and operator commands.

- `interface.py`: shared robot protocol, factory and arm/pose conventions.
- `collect_demo.py`: shared Quest collection loop and incremental HDF5 writer.
  Keep the existing camera names, RGB uint8 image storage, 14D left/right vectors,
  intrinsic ZYX Euler angles and joint/action aliases. Never overwrite a demo.
- `teleop.py`: world-frame clutch control, tracking-loss reanchoring and command
  velocity limits. The controller delta is measured in the fixed VR world basis.
- `cameras.py`: shared camera setup, freshness checks and live front/wrist views;
  drivers remain in `eva/eva_ws/src/eva/stream_{aria,d405}.py`.
- `eva/eva_ws/src/eva/robot_interface.py`: supported Eva ARX interface; retain its
  controller, gripper calibration and ROS workspace. `eva/eva_kinematics.py`
  implements its existing FK/IK solver.
- `yam/interface.py`: local i2rt Yam implementation of the same robot API.
  `yam/kinematics.py`: MuJoCo FK/IK using the configured Yam model and TCP site.
- `rollout.py`: shared rollout loop. Inference uses only the local graph path.
- `graph_policy.py`: strict PipelineAlgo checkpoint loading, full normalization
  state, camera/proprio mapping and explicit Cartesian action-frame conversion.
- `replay_policy.py`: read-only Zarr joint replay, bounded by total_frames and EOF.
- `arc_decoder.py`: decode native ARC predictions before frame conversion and IK.
  E1 timing channels follow all 14 pose channels; they are not interleaved.
- `oculus_reader/`: the pinned RAIL/Yam APK, matching source, and Python reader.
  The bundled app emits tracking-origin/world poses under the established
  `wE9ryARX` tag; the reader is world-only and rejects a head-frame mode. Keep the
  APK provenance hash and source guard current whenever the app changes.
- `../scripts/data_upload/yam_uploader.py`: existing RLDB upload workflow for Yam
  HDF5 demos; `--list` lists files without network access or credentials.

Station/task choices belong in `../hydra_configs/robot/` or a copied YAML:
CAN/camera IDs, home poses, teleop gains, prompts and graph frame/codec settings.
Use the same graph, data normalization and evaluation components for HPT and PI.
Do not introduce an embodiment-specific inference server or upstream runtime
launcher. Do not deprecate Eva when adding another robot implementation.

Run `tests/test_robot_runtime.py` and `tests/test_robot_graph_policy.py` for CPU
validation. `--help` and uploader `--list` open no devices. Collection/rollout
commands open hardware; operate a physical station only when the user requests
that operation. Preserve source branches, user worktrees and recorded data.
