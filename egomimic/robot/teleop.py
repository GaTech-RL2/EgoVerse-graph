"""World-frame clutch mapping following rl2_yam's Quest frame convention.

Reference: GaTech-RL2/yam-pipeline, revision 1b1f9b12d300a41872b8a8c6c04f0c9f5f892f88,
rl2_yam/agents/quest_mapper.py. No upstream runtime is required.
"""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from egomimic.robot.interface import ARM_OFFSET, joint_vector, pose_matrix, pose_vector

# Quest world: right/up/back. Arm base: forward/left/up.
WORLD_TO_BASE = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def analog(buttons, key):
    value = np.asarray((buttons or {}).get(key, 0), dtype=float).reshape(-1)
    return (
        float(np.clip(value[0], 0, 1)) if value.size and np.isfinite(value[0]) else 0.0
    )


class ButtonEdge:
    def __init__(self, key):
        self.key, self.down = key, False

    def pressed(self, buttons):
        down = analog(buttons, self.key) > 0.5
        rising, self.down = down and not self.down, down
        return rising


def rigid_transform(value):
    matrix = np.asarray(value, dtype=float)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1])
        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-3)
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-3)
    ):
        raise ValueError("Expected a finite rigid 4x4 pose")
    return matrix.copy()


class WorldFrameTeleop:
    def __init__(
        self,
        arms,
        translation_gain,
        orientation_gain,
        target_lpf,
        max_controller_step_m,
        max_controller_step_deg,
        headset_yaw_degrees=0.0,
        orientation_rx_degrees=0.0,
        side_yaw=None,
    ):
        self.arms = tuple(arms)
        self.translation_gain, self.orientation_gain = (
            float(translation_gain),
            float(orientation_gain),
        )
        self.alpha = float(target_lpf)
        self.max_m, self.max_rad = (
            float(max_controller_step_m),
            np.deg2rad(max_controller_step_deg),
        )
        if (
            not 0 < self.alpha <= 1
            or min(
                self.translation_gain, self.orientation_gain, self.max_m, self.max_rad
            )
            <= 0
        ):
            raise ValueError(
                "Teleop gains/limits must be positive and target_lpf in (0, 1]"
            )
        self.axes = {
            arm: WORLD_TO_BASE
            @ Rotation.from_euler(
                "y", (side_yaw or {}).get(arm, headset_yaw_degrees), degrees=True
            ).as_matrix()
            for arm in arms
        }
        self.orientation_axes = {
            arm: basis
            @ Rotation.from_euler("x", orientation_rx_degrees, degrees=True).as_matrix()
            for arm, basis in self.axes.items()
        }
        self.reanchor = {
            arm: ButtonEdge("LJ" if arm == "left" else "RJ") for arm in arms
        }
        self.reset()

    def reset(self):
        self.anchors, self.previous, self.filtered = {}, {}, {}

    def update(self, poses, buttons, measured):
        targets = {}
        self.anchored = set()
        for arm in self.arms:
            side, grip, trigger = (
                ("l", "LG", "leftTrig") if arm == "left" else ("r", "RG", "rightTrig")
            )
            reclutch = self.reanchor[arm].pressed(buttons)
            try:
                controller = rigid_transform((poses or {}).get(side))
            except (TypeError, ValueError):
                controller = None
            held = (
                analog(buttons, grip) > 0.5
                or analog(buttons, "leftGrip" if arm == "left" else "rightGrip") > 0.5
            )
            previous = self.previous.get(arm)
            jumped = (
                controller is not None
                and previous is not None
                and (
                    np.linalg.norm(controller[:3, 3] - previous[:3, 3]) > self.max_m
                    or Rotation.from_matrix(
                        controller[:3, :3] @ previous[:3, :3].T
                    ).magnitude()
                    > self.max_rad
                )
            )
            if controller is None or not held or jumped or reclutch:
                self.anchors.pop(arm, None)
                self.filtered.pop(arm, None)
            self.previous[arm] = controller
            if controller is None or not held or jumped:
                continue
            if arm not in self.anchors:
                self.anchored.add(arm)
                self.anchors[arm] = (controller.copy(), rigid_transform(measured[arm]))
            initial, robot_initial = self.anchors[arm]
            basis, orientation = self.axes[arm], self.orientation_axes[arm]
            target = robot_initial.copy()
            target[:3, 3] += self.translation_gain * (
                basis @ (controller[:3, 3] - initial[:3, 3])
            )
            delta = (
                orientation @ (controller[:3, :3] @ initial[:3, :3].T) @ orientation.T
            )
            target[:3, :3] = (
                Rotation.from_rotvec(
                    self.orientation_gain * Rotation.from_matrix(delta).as_rotvec()
                ).as_matrix()
                @ robot_initial[:3, :3]
            )
            old = self.filtered.get(arm)
            if old is not None:
                target[:3, 3] = old[:3, 3] + self.alpha * (target[:3, 3] - old[:3, 3])
                target[:3, :3] = Slerp(
                    [0, 1],
                    Rotation.from_matrix(np.stack([old[:3, :3], target[:3, :3]])),
                )(self.alpha).as_matrix()
            self.filtered[arm] = target.copy()
            targets[arm] = np.r_[pose_vector(target), 1.0 - analog(buttons, trigger)]
        return targets


class TeleopControl:
    def __init__(self, robot, mapper, frequency, max_joint_velocity):
        self.robot, self.mapper = robot, mapper
        self.step_limit = float(max_joint_velocity) / float(frequency)
        if not np.isfinite(self.step_limit) or self.step_limit <= 0:
            raise ValueError("Joint velocity and frequency must be positive")
        self.last = None

    def reset(self):
        self.last = None
        self.mapper.reset()

    def step(self, poses, buttons, obs):
        if self.last is None:
            self.last = np.asarray(obs["joint_positions"]).copy()
        measured = {
            arm: pose_matrix(obs["ee_poses"][ARM_OFFSET[arm] : ARM_OFFSET[arm] + 6])
            for arm in self.robot.arms
        }
        targets = self.mapper.update(poses, buttons, measured)
        for arm in self.mapper.anchored:
            offset = ARM_OFFSET[arm]
            self.last[offset : offset + 7] = obs["joint_positions"][offset : offset + 7]
        commands, ee_commands = self.last.copy(), np.zeros(14)
        for arm in self.robot.arms:
            offset = ARM_OFFSET[arm]
            if arm in targets:
                target = targets[arm]
                try:
                    solved = self.robot.solve_ik(target[:6], arm)
                    command = joint_vector(np.r_[solved, target[6]])
                except (ValueError, RuntimeError):
                    command = commands[offset : offset + 7].copy()
                command[:6] = commands[offset : offset + 6] + np.clip(
                    command[:6] - commands[offset : offset + 6],
                    -self.step_limit,
                    self.step_limit,
                )
                commands[offset : offset + 7] = command
            # Hold the last accepted command on clutch release or stale tracking.
            command = joint_vector(commands[offset : offset + 7])
            self.robot.set_joints(command, arm)
            ee_commands[offset : offset + 7] = np.r_[
                self.robot.forward_kinematics(command[:6], arm), command[6]
            ]
        self.last = commands
        return commands.copy(), ee_commands
