# Robot navigation

The root AGENTS.md applies here. Start with [YAM_RUNTIME.md](../../docs/YAM_RUNTIME.md)
for setup, the upstream revision and operator commands.

- `yam/runtime.py`, `yam/__main__.py`: launch the pinned external `rl2_yam`
  Quest teleop, camera or rollout process in its existing uv environment.
- `collect_demo.py`: default collection entry point, forwarding to Yam Quest.
- `yam/server.py`, `yam/policy_inference.proto`: graph inference RPC compatible
  with the upstream `PolicyClient` (two observations, 24 joint targets at 30 Hz).
- `yam/policy.py`: restore the training graph/checkpoint/normalizer and apply
  normalization at the live boundary.
- `yam/adapter.py`: camera mapping, current TCP proprio, action-frame reversion
  and conversion into left/right joint targets.
- `yam/kinematics.py`: hardware-free FK/IK with a configured MuJoCo model.
- `arc_decoder.py`: decode native ARC predictions into time-indexed Cartesian
  poses before frame reversion and IK. E1's timing channels follow all 14 pose
  channels; they are not interleaved with the arms.
- `eva/collect_demo_legacy.py`, `eva/eva_ws/`, `oculus_reader/`: retained Eva/ROS
  and Oculus sources for existing stations.

Deployment choices belong in `../hydra_configs/robot/yam_server.yaml` or the
operator's copied YAML. Match training's action codec, rotation representation,
proprio frame, camera keys, embodiment ID and full normalization state.
Do not assume world-frame and camera-frame proprio are interchangeable.

Run `tests/test_yam_interface.py` for CPU/RPC/FK/IK validation. The
`--print-command` launcher path opens no devices. Live teleop/rollout commands
open hardware; run them only when the user explicitly requests operation of the
configured station. Preserve upstream command guards and the prior station
implementations when changing the integration.
