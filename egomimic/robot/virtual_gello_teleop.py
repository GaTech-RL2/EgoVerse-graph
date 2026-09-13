"""Visual, hardware-free dual-YAM GELLO teleop smoke runner.

This module deliberately uses synthetic leader samples.  It never constructs a
YamInterface, DynamixelPositionBus, camera, CAN socket, or Quest reader.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
import yaml

from egomimic.robot.interface import ARM_OFFSET, joint_vector, pose_vector
from egomimic.robot.yam.gello import (
    BimanualGelloReader,
    GelloSample,
    GelloTeleopControl,
)


def default_model_path() -> Path:
    """Return i2rt's installed dual-arm RL2 station model without opening a driver."""
    import i2rt

    return (
        Path(i2rt.__file__).resolve().parent
        / "robot_models/station/yam_station_linear_4310_d405"
        / "yam_station_linear_4310_d405.xml"
    )


class VirtualYamFollower:
    """The small RobotInterface subset used by :class:`GelloTeleopControl`."""

    arms = ("left", "right")

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self._joint_ids = {
            arm: tuple(
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}_joint{i}"
                )
                for i in range(1, 9)
            )
            for arm in self.arms
        }
        self._body_ids = {
            arm: mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"{arm}_gripper"
            )
            for arm in self.arms
        }
        if any(
            value < 0 for values in self._joint_ids.values() for value in values
        ) or any(value < 0 for value in self._body_ids.values()):
            raise ValueError("Virtual model must be the dual-arm YAM station model")
        self.q = np.zeros(14, dtype=float)
        for arm in self.arms:
            ids = self._joint_ids[arm]
            offset = ARM_OFFSET[arm]
            for i, joint_id in enumerate(ids[:6]):
                lo, hi = self.model.jnt_range[joint_id]
                self.q[offset + i] = (lo + hi) / 2
            self.q[offset + 6] = 0.5
        self._sync_model()

    def _sync_model(self) -> None:
        for arm in self.arms:
            offset = ARM_OFFSET[arm]
            values = self.q[offset : offset + 7]
            for i, joint_id in enumerate(self._joint_ids[arm][:6]):
                self.data.qpos[self.model.jnt_qposadr[joint_id]] = values[i]
            # The model has two coupled finger slides.  Set both explicitly so
            # rendering is stable even when equality constraints are disabled.
            for joint_id in self._joint_ids[arm][6:]:
                lo, hi = self.model.jnt_range[joint_id]
                self.data.qpos[self.model.jnt_qposadr[joint_id]] = lo + values[6] * (
                    hi - lo
                )
        mujoco.mj_forward(self.model, self.data)

    def _pose(self, arm: str) -> np.ndarray:
        # Express gripper pose in the corresponding arm-base frame.
        base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{arm}_base")
        body_id = self._body_ids[arm]
        world_from_base = np.eye(4)
        world_from_base[:3, :3] = self.data.xmat[base_id].reshape(3, 3)
        world_from_base[:3, 3] = self.data.xpos[base_id]
        world_from_gripper = np.eye(4)
        world_from_gripper[:3, :3] = self.data.xmat[body_id].reshape(3, 3)
        world_from_gripper[:3, 3] = self.data.xpos[body_id]
        return pose_vector(np.linalg.solve(world_from_base, world_from_gripper))

    def get_obs(self) -> dict:
        poses = np.zeros(14, dtype=float)
        for arm in self.arms:
            offset = ARM_OFFSET[arm]
            poses[offset : offset + 7] = np.r_[self._pose(arm), self.q[offset + 6]]
        return {"joint_positions": self.q.copy(), "ee_poses": poses}

    def forward_kinematics(self, joints, arm: str) -> np.ndarray:
        values = joint_vector(
            np.r_[np.asarray(joints, dtype=float), self.q[ARM_OFFSET[arm] + 6]]
        )
        previous = self.q.copy()
        self.q[ARM_OFFSET[arm] : ARM_OFFSET[arm] + 7] = values
        self._sync_model()
        pose = self._pose(arm)
        self.q = previous
        self._sync_model()
        return pose

    def set_joints(self, desired_position, arm: str) -> None:
        self.q[ARM_OFFSET[arm] : ARM_OFFSET[arm] + 7] = joint_vector(desired_position)
        self._sync_model()


def synthetic_sample(t: float) -> GelloSample:
    """Smooth, bounded dual-leader motion used only by the virtual smoke."""
    joints = {}
    for arm, phase in (("left", 0.0), ("right", np.pi)):
        u = float(t) + phase
        joints[arm] = np.array(
            [
                0.22 * np.sin(0.65 * u),
                0.18 * np.sin(0.47 * u),
                0.16 * np.cos(0.59 * u),
                0.20 * np.sin(0.51 * u),
                0.12 * np.cos(0.73 * u),
                0.24 * np.sin(0.41 * u),
                0.5 + 0.35 * np.sin(0.35 * u),
            ],
            dtype=float,
        )
    return GelloSample(joints=joints, timestamps={"left": t, "right": t})


def real_leader_specs(config_path: Path, left_path: Path, right_path: Path) -> dict:
    """Load passive leader settings plus the two separately exported calibrations."""
    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    leaders = dict(config["gello"]["leaders"])
    for arm, path in (("left", left_path), ("right", right_path)):
        with path.open() as stream:
            exported = yaml.safe_load(stream)
        leaders[arm] = {**leaders[arm], **exported["gello"]["leaders"][arm]}
    bus = dict(config["gello"]["bus"])
    return {arm: {**bus, **dict(leaders[arm])} for arm in ("left", "right")}


def run_demo(robot: VirtualYamFollower, frequency: float, steps: int) -> np.ndarray:
    """Run the real relative GELLO controller against synthetic inputs."""
    if frequency <= 0 or steps <= 0:
        raise ValueError("frequency and steps must be positive")
    control = GelloTeleopControl(robot, frequency=frequency, alignment="relative")
    commands = None
    for step in range(steps):
        t = step / frequency
        result = control.step(
            synthetic_sample(t),
            robot.get_obs(),
            toggle_arms=("left", "right") if step == 0 else (),
            now=t,
        )
        commands = result.joints
    assert commands is not None
    return commands


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--frequency", type=float, default=60.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--initial-left", type=float, nargs=6, metavar="Q")
    parser.add_argument("--initial-right", type=float, nargs=6, metavar="Q")
    parser.add_argument(
        "--input",
        choices=("synthetic", "gello"),
        default="synthetic",
        help="Use generated samples, or read two passive real GELLO leaders.",
    )
    parser.add_argument(
        "--config", type=Path, help="Station GELLO profile for --input gello."
    )
    parser.add_argument(
        "--left-calibration", type=Path, help="Left export for --input gello."
    )
    parser.add_argument(
        "--right-calibration", type=Path, help="Right export for --input gello."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the same demo without a MuJoCo window.",
    )
    args = parser.parse_args(argv)
    if args.frequency <= 0 or args.duration <= 0:
        parser.error("--frequency and --duration must be positive")
    robot = VirtualYamFollower(args.model or default_model_path())
    for arm, values in (("left", args.initial_left), ("right", args.initial_right)):
        if values is not None:
            robot.q[ARM_OFFSET[arm] : ARM_OFFSET[arm] + 6] = np.asarray(
                values, dtype=float
            )
    robot._sync_model()
    steps = max(1, round(args.duration * args.frequency))
    if args.input == "gello" and not all(
        (args.config, args.left_calibration, args.right_calibration)
    ):
        parser.error(
            "--input gello requires --config, --left-calibration, and --right-calibration"
        )
    if args.headless:
        if args.input != "synthetic":
            parser.error("--headless is only supported with --input synthetic")
        command = run_demo(robot, args.frequency, steps)
        print(
            f"Virtual GELLO smoke passed: {steps} steps; max |q|={np.max(np.abs(command[:6])):.3f} rad"
        )
        return 0

    import mujoco.viewer

    reader = None
    if args.input == "gello":
        reader = BimanualGelloReader(
            ("left", "right"),
            real_leader_specs(
                args.config, args.left_calibration, args.right_calibration
            ),
        )
    print(
        f"Virtual GELLO teleop: {args.input} input active. Close the MuJoCo viewer to exit."
    )
    control = GelloTeleopControl(robot, frequency=args.frequency, alignment="relative")
    try:
        with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
            started = time.monotonic()
            step = 0
            while viewer.is_running() and step < steps:
                tick = time.monotonic()
                sample = reader.read() if reader is not None else synthetic_sample(tick)
                # USB reads complete after ``tick``; validate freshness against
                # the current clock so a freshly read sample is never negative-age.
                now = time.monotonic()
                control.step(
                    sample,
                    robot.get_obs(),
                    ("left", "right") if step == 0 else (),
                    now=now,
                )
                viewer.sync()
                step += 1
                time.sleep(max(0.0, 1.0 / args.frequency - (time.monotonic() - tick)))
            print(
                f"Virtual GELLO teleop finished after {time.monotonic() - started:.1f}s."
            )
    finally:
        if reader is not None:
            reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
