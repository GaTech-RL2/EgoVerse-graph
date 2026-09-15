"""Hardware-free checks for the YAM rollout browser view and its overlay."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from egomimic.robot.interface import ARM_OFFSET
from egomimic.robot.rollout import run_rollout
from egomimic.robot.rollout_dashboard import (
    load_action_overlay,
    validate_rollout_preview,
)

ROOT = Path(__file__).resolve().parents[1]


def calibration_file(tmp_path):
    path = tmp_path / "agentview.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "convention": "point_base = base_T_camera @ point_camera",
                "K": [[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]],
                "dist": [0, 0, 0, 0, 0],
                "channels": {
                    "can_l": {"base_T_camera": np.eye(4).tolist()},
                    "can_r": {"base_T_camera": np.eye(4).tolist()},
                },
            }
        )
    )
    return path


def overlay_config(path):
    return {
        "initial_enabled": False,
        "camera": "front_img_1",
        "calibration_path": str(path),
        "arm_channels": {"left": "can_l", "right": "can_r"},
    }


def test_overlay_uses_base_to_camera_calibration_without_mutating_frame(tmp_path):
    overlay = load_action_overlay(overlay_config(calibration_file(tmp_path)))
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    actions = np.zeros((2, 14), dtype=float)
    actions[:, 2] = 1.0
    actions[:, 9] = 1.0
    actions[:, 7] = 0.1

    output = overlay.draw(frame, actions)

    assert not np.any(frame)
    assert np.any(output)
    assert tuple(output[24, 32]) == (0, 190, 255)
    assert tuple(output[24, 42]) == (255, 185, 80)


def test_dashboard_validation_requires_loopback_and_pinned_calibration(tmp_path):
    preview = {
        "mode": "dashboard",
        "host": "127.0.0.1",
        "action_overlay": overlay_config(calibration_file(tmp_path)),
    }
    result, overlay = validate_rollout_preview(preview, {"front_img_1"})
    assert result["port"] == 8081
    assert overlay.camera == "front_img_1"

    preview["host"] = "0.0.0.0"
    with pytest.raises(ValueError, match="loopback"):
        validate_rollout_preview(preview, {"front_img_1"})


def test_hptflow_profile_derives_right_model_frame_from_pinned_calibration():
    profile = yaml.safe_load(
        (ROOT / "egomimic/hydra_configs/robot/yam_rl2_hptflow_rollout.yaml").read_text()
    )
    calibration = yaml.safe_load(
        (
            ROOT / "egomimic/hydra_configs/robot/yam_rl2_agentview_extrinsics.yaml"
        ).read_text()
    )
    expected_right_T_left = np.asarray(
        calibration["channels"]["can_follower_r"]["base_T_camera"], dtype=float
    ) @ np.linalg.inv(
        np.asarray(
            calibration["channels"]["can_follower_l"]["base_T_camera"], dtype=float
        )
    )
    adapter = profile["policy"]["adapter"]
    np.testing.assert_allclose(adapter["base_T_model"]["left"], np.eye(4))
    np.testing.assert_allclose(
        adapter["base_T_model"]["right"], expected_right_T_left, atol=1e-12
    )
    assert set(adapter["camera_keys"]) == {
        "front_img_1",
        "left_wrist_img",
        "right_wrist_img",
    }


class FakeRobot:
    def __init__(self):
        self.arms = ["left", "right"]
        self.camera_res = {"front_img_1": (2, 3)}
        self.q = np.zeros(14)
        self.q[[6, 13]] = 0.5
        self.commands = []

    def get_obs(self):
        return {
            "joint_positions": self.q.copy(),
            "ee_poses": self.q.copy(),
            "front_img_1": np.zeros((2, 3, 3), dtype=np.uint8),
        }

    def solve_ik(self, pose, _arm):
        return np.asarray(pose, dtype=float)

    def set_joints(self, command, arm):
        command = np.asarray(command, dtype=float).copy()
        self.commands.append((arm, command))
        offset = ARM_OFFSET[arm]
        self.q[offset : offset + 7] = command


class View:
    def __init__(self):
        self.updates = 0
        self.plans = []
        self.closed = False

    def update(self, _obs):
        self.updates += 1
        return None if self.updates == 1 else "q"

    def set_action_plan(self, plan, action_type):
        self.plans.append((np.asarray(plan).copy(), action_type))

    def close(self):
        self.closed = True


class GatedView(View):
    def __init__(self, controls):
        super().__init__()
        self.controls = iter(controls)
        self.statuses = []

    def update(self, _obs):
        self.updates += 1
        return next(self.controls)

    def set_status(self, status):
        self.statuses.append(status)

    def clear_action_plan(self):
        self.plans.clear()


def test_rollout_publishes_graph_plan_to_view_without_changing_command_path(
    monkeypatch,
):
    monkeypatch.setattr("egomimic.robot.rollout.time.sleep", lambda _: None)
    robot, view = FakeRobot(), View()
    target = np.zeros((1, 14), dtype=float)
    target[:, [6, 13]] = 0.5
    policy = SimpleNamespace(action_type="cartesian", predict=lambda _obs: target)

    steps = run_rollout(
        robot,
        policy,
        {
            "frequency": 30,
            "max_steps": 4,
            "execute_steps": 1,
            "max_joint_velocity": 1.0,
            "preview": {"enabled": False},
        },
        view=view,
    )

    assert steps == 1 and view.closed
    assert len(view.plans) == 1
    np.testing.assert_allclose(view.plans[0][0], target)
    assert view.plans[0][1] == "cartesian"
    assert {arm for arm, _ in robot.commands} == {"left", "right"}


def test_rollout_waits_for_c_and_restart_discards_the_existing_plan(monkeypatch):
    monkeypatch.setattr("egomimic.robot.rollout.time.sleep", lambda _: None)
    robot = FakeRobot()
    view = GatedView([None, "c", "r", "c", "q"])
    target = np.zeros((1, 14), dtype=float)
    target[:, [6, 13]] = 0.5
    calls = 0

    def predict(_obs):
        nonlocal calls
        calls += 1
        return target

    steps = run_rollout(
        robot,
        SimpleNamespace(action_type="cartesian", predict=predict),
        {
            "frequency": 30,
            "max_steps": 4,
            "execute_steps": 1,
            "max_joint_velocity": 1.0,
            "preview": {"enabled": False, "wait_for_start": True},
        },
        view=view,
    )

    assert calls == 2
    assert steps == 1  # The restart begins a new rollout step counter.
    assert len(robot.commands) == 4  # Two safe paired-arm commands total.
    assert view.closed
    assert any("Ready" in status for status in view.statuses)
    assert any("Restarted" in status for status in view.statuses)


def test_rollout_resamples_a_velocity_unsafe_plan_before_commanding(monkeypatch):
    monkeypatch.setattr("egomimic.robot.rollout.time.sleep", lambda _: None)
    robot = FakeRobot()
    view = GatedView([None, None, "q"])
    unsafe = np.zeros((1, 14), dtype=float)
    unsafe[:, [0, 7]] = 0.2
    unsafe[:, [6, 13]] = 0.5
    safe = np.zeros((1, 14), dtype=float)
    safe[:, [6, 13]] = 0.5
    plans = iter((unsafe, safe))

    steps = run_rollout(
        robot,
        SimpleNamespace(action_type="joints", predict=lambda _obs: next(plans)),
        {
            "frequency": 30,
            "max_steps": 4,
            "execute_steps": 1,
            "max_joint_velocity": 1.0,
            "max_velocity_replans": 1,
            "preview": {"enabled": False},
        },
        view=view,
    )

    assert steps == 1
    assert len(robot.commands) == 2  # Only the second, paired safe plan ran.
    assert any("Rejected velocity-unsafe" in status for status in view.statuses)
