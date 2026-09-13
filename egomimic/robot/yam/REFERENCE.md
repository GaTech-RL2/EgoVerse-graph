# RL2 Yam reference import

The station-specific Quest mapping, streaming IK, workspace, and overhead-camera
calibration are pinned to `GaTech-RL2/yam-pipeline` commit
`1b1f9b12d300a41872b8a8c6c04f0c9f5f892f88`.

| Upstream path | EgoVerse path | Git blob | SHA-256 |
| --- | --- | --- | --- |
| `rl2_yam/agents/quest_mapper.py` | `egomimic/robot/yam/quest_mapper.py` | `dbad1482f0281b523e6148ae8efeccf33b83d553` | `a94d333cc3467da41670b49e77dd3e37f565f1f82579c28a6ad31f0f7d2a5b8d` |
| `rl2_yam/utils/ik.py` | `egomimic/robot/yam/streaming_ik.py` | `984a23453165d32fb6da2b952990244b73dc0f2c` | `488a453ab6623d05d3c690d3008970f83700a149240a8715c205f43ff46b8add` |
| `rl2_yam/config/agentview_extrinsics.yaml` | `egomimic/hydra_configs/robot/yam_rl2_agentview_extrinsics.yaml` | `4d0476a49efa5d2b88eb8dac8b3688c279202c2e` | `5795bd2480e3ef31a1ffcf993834613bd3bf267082e24c23c30094848fbd122c` |
| `rl2_yam/config/workspace.yaml` | `egomimic/hydra_configs/robot/yam_rl2_workspace.yaml` | `929cb529fb359d8d9b8b68f96a10d0409caa2ef5` | `bef2471d3a6d759728bcb7d85eb2943fd9f0377cf65d450d4712f2ad43f9a9b9` |

The station profile also copies the control values from
`rl2_yam/config/quest_teleop.yaml`, including the required fixed room-to-robot
`headset_yaw_degrees: 180.0`, 60 Hz control, and 30 Hz recording. Tests verify
the byte hashes and these values. Update the pin, copies, hashes, and parity tests
together; never hand-tune only one side of this import.

## Validation record

The hardware-free robot tests cover the 180-degree XYZ and roll/pitch/yaw basis,
command-seeded streaming IK, 60/30 Hz scheduling, and exact calibration parity.
The calibration parity test compares every copied intrinsic, distortion, and
per-arm extrinsic value; keep that test when updating calibration so a sign or
arm-label transcription cannot silently pass.

At the parent commit, two unrelated repository-wide checks are already blocked:
`uv lock --check` cannot re-resolve the Pi0.5 extra because its current OpenPI
dependency requires `av>=14.2` while the project pins `av==12`, and nine ABC
campaign tests fail Hydra Defaults List composition. Both failures reproduce from
a clean `git archive` of the parent. The installed YAM environment passes
`pip check`, and its Mink 1.1.0 lock stanza exactly matches this source revision.
