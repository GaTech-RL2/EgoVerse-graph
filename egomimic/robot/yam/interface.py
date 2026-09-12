"""Local i2rt Yam driver with the same interface used by Eva collection/rollout."""

import time

import numpy as np
from scipy.spatial.transform import Rotation

from egomimic.robot.cameras import close_cameras, open_cameras
from egomimic.robot.interface import ARM_OFFSET, joint_vector, pose_matrix, pose_vector
from egomimic.robot.yam.kinematics import MujocoArmKinematics


class YamInterface:
    def __init__(
        self,
        arms,
        channels,
        cameras,
        kinematics,
        home,
        gripper_type="linear_4310",
        zero_gravity_mode=True,
        enable_auto_recovery=False,
        home_duration=3.0,
        frequency=30.0,
        driver_factory=None,
        solver_factory=None,
    ):
        self.arms = list(arms)
        if (
            not self.arms
            or len(set(self.arms)) != len(self.arms)
            or not set(self.arms) <= ARM_OFFSET.keys()
        ):
            raise ValueError("Yam arms must select left/right without duplicates")
        if not set(self.arms) <= channels.keys() or len(
            {channels[a] for a in self.arms}
        ) != len(self.arms):
            raise ValueError("Each selected Yam arm needs a distinct CAN channel")
        self.home = {a: joint_vector(home[a]) for a in self.arms}
        self.home_duration, self.frequency = float(home_duration), float(frequency)
        if min(self.home_duration, self.frequency) <= 0:
            raise ValueError("Home duration and frequency must be positive")
        self.controller, self.solvers, self.recorders, self.camera_res = {}, {}, {}, {}
        if driver_factory is None:
            from i2rt.robots.get_robot import get_yam_robot
            from i2rt.robots.utils import ArmType, GripperType

            def driver_factory(channel):
                return get_yam_robot(
                    channel=channel,
                    arm_type=ArmType.YAM,
                    gripper_type=GripperType.from_string_name(gripper_type),
                    zero_gravity_mode=zero_gravity_mode,
                    enable_auto_recovery=enable_auto_recovery,
                )

        solver_factory = solver_factory or MujocoArmKinematics
        try:
            for arm in self.arms:
                driver = self.controller[arm] = driver_factory(channels[arm])
                spec = dict(kinematics)
                spec.setdefault("xml_path", driver.xml_path)
                self.solvers[arm] = solver_factory(**spec)
                joint_vector(driver.get_joint_pos())
            self.recorders, self.camera_res = open_cameras(cameras)
        except BaseException:
            self.close()
            raise

    def get_joints(self, arm):
        return joint_vector(self.controller[arm].get_joint_pos())

    def forward_kinematics(self, joints, arm):
        return pose_vector(self.solvers[arm].fk(np.asarray(joints)[:6]))

    def get_pose(self, arm, se3=False):
        matrix = self.solvers[arm].fk(self.get_joints(arm)[:6])
        return matrix if se3 else (matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]))

    def get_obs(self):
        joints, poses = np.zeros(14), np.zeros(14)
        for arm in self.arms:
            offset = ARM_OFFSET[arm]
            q = self.get_joints(arm)
            joints[offset : offset + 7] = q
            poses[offset : offset + 7] = np.r_[
                pose_vector(self.solvers[arm].fk(q[:6])), q[6]
            ]
        return {
            "joint_positions": joints,
            "ee_poses": poses,
            **{name: camera.get_image() for name, camera in self.recorders.items()},
        }

    def solve_ik(self, ee_pose, arm):
        return self.solvers[arm].ik(pose_matrix(ee_pose), self.get_joints(arm)[:6])

    def set_joints(self, desired_position, arm):
        self.controller[arm].command_joint_pos(joint_vector(desired_position))

    def set_pose(self, pose, arm):
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (7,):
            raise ValueError("Expected xyz, yaw, pitch, roll, gripper")
        self.set_joints(np.r_[self.solve_ik(pose[:6], arm), pose[6]], arm)

    def set_home(self):
        start = {arm: self.get_joints(arm) for arm in self.arms}
        steps = max(1, round(self.home_duration * self.frequency))
        for step in range(1, steps + 1):
            weight = 0.5 - 0.5 * np.cos(np.pi * step / steps)
            for arm in self.arms:
                self.set_joints(
                    start[arm] + weight * (self.home[arm] - start[arm]), arm
                )
            time.sleep(1 / self.frequency)

    def close(self):
        errors = []
        try:
            close_cameras(self.recorders)
        except Exception as error:
            errors.append(error)
        for driver in self.controller.values():
            try:
                driver.close()
            except Exception as error:
                errors.append(error)
        self.recorders, self.controller = {}, {}
        if errors:
            raise ExceptionGroup("Yam cleanup failed", errors)
