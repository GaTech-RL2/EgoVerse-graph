"""Differential IK recovery checks with MuJoCo and fake followers only."""

from types import SimpleNamespace

import numpy as np
import pytest

from egomimic.robot.rollout import run_rollout, validate_rollout_config
from egomimic.robot.yam.kinematics import MujocoArmKinematics
from egomimic.robot.yam.rollout_ik import (
    RecoveryStep,
    differential_ik_step,
    validate_ik_recovery,
)
from tests.test_robot_runtime import FakeRobot, View


@pytest.fixture
def solver(tmp_path):
    pytest.importorskip("mujoco")
    bodies = "".join(
        f'<body><joint name="joint{i}" type="hinge" axis="0 0 1" '
        'limited="true" range="-1 1"/><geom type="sphere" size=".01" mass="1"/>'
        for i in range(1, 7)
    )
    xml = '<mujoco><compiler angle="radian"/><worldbody>' + bodies
    xml += '<site name="tcp_site" pos="1 0 0"/>' + "</body>" * 6
    xml += "</worldbody></mujoco>"
    path = tmp_path / "arm.xml"
    path.write_text(xml)
    return MujocoArmKinematics(path, [f"joint{i}" for i in range(1, 7)])


def test_singular_chain_makes_bounded_progress_until_reached(solver):
    q = np.zeros(6)
    target = solver.fk(np.full(6, 0.06)).copy()
    start_error = np.linalg.norm(target[:3, 3] - solver.fk(q)[:3, 3])
    errors = []
    for _ in range(50):
        step = differential_ik_step(solver, target, q, q, 0.01, {})
        assert np.max(np.abs(step.joints - q)) <= 0.01 + 1e-12
        assert np.max(np.abs(step.joints)) <= 1
        q = step.joints
        errors.append(step.position_error_m)
        if step.reached:
            break
    assert step.reached
    assert errors[-1] < start_error
    assert all(b <= a + 1e-12 for a, b in zip(errors, errors[1:]))


def test_tracking_lag_cannot_bypass_last_command_bound(solver):
    target = solver.fk(np.full(6, 0.1)).copy()
    with pytest.raises(ValueError, match="Tracking lag"):
        differential_ik_step(solver, target, np.zeros(6), np.full(6, 0.1), 0.01, {})
    step = differential_ik_step(
        solver, target, np.zeros(6), np.full(6, 0.008), 0.01, {}
    )
    assert np.max(np.abs(step.joints)) <= 0.01 + 1e-12
    assert np.max(np.abs(step.joints - 0.008)) <= 0.01 + 1e-12


def test_unreachable_radial_target_does_not_emit_a_motion(solver):
    target = solver.fk(np.zeros(6)).copy()
    target[0, 3] += 0.1
    with pytest.raises(ValueError, match="progress"):
        differential_ik_step(solver, target, np.zeros(6), np.zeros(6), 0.01, {})


def test_joint_limit_is_enforced(solver):
    q = np.ones(6)
    target = solver.fk(q + 0.1).copy()
    with pytest.raises(ValueError, match="progress"):
        differential_ik_step(solver, target, q, q, 0.01, {})


@pytest.mark.parametrize(
    "settings",
    [
        {"enabled": "true"},
        {"damping": float("nan")},
        {"posture_weight": 0},
        {"max_joint_velocity": float("inf")},
        {"max_ticks_per_target": 0},
        {"max_ticks_per_target": True},
        {"unknown": 1},
    ],
)
def test_invalid_options_fail_even_while_disabled(settings):
    with pytest.raises(ValueError):
        validate_ik_recovery(settings)


def test_wrong_embodiment_fails_before_hardware():
    with pytest.raises(ValueError, match="only for Yam"):
        validate_rollout_config(
            {"ik_recovery": {"enabled": True}, "robot": {"kind": "eva"}}
        )


class RecoveryRobot(FakeRobot):
    def __init__(self):
        super().__init__()
        self.strict_calls = 0
        self.recovery_calls = []
        self.fail_right = False
        self.excessive = False

    def solve_ik(self, pose, arm):
        self.strict_calls += 1
        raise ValueError("strict endpoint failure")

    def solve_ik_step(self, pose, arm, *, measured, last_command, max_step, settings):
        self.recovery_calls.append((arm, np.asarray(pose).copy(), max_step))
        if self.fail_right and arm == "right":
            raise ValueError("no safe step")
        q = measured + np.clip(pose - measured, -max_step, max_step)
        if self.excessive:
            q += 1
        return RecoveryStep(q, bool(np.allclose(q, pose)), 0.0, 0.0)


class StopView(View):
    def __init__(self, stop_at=10):
        super().__init__()
        self.tick = 0
        self.stop_at = stop_at

    def update(self, obs):
        self.tick += 1
        return "q" if self.tick >= self.stop_at else None


def run_fake(robot, settings=None, max_steps=4, view=None):
    plan = np.full((2, 14), 0.02)
    plan[0, [6, 13]] = 0.0
    plan[1, :] = 0.04
    plan[1, [6, 13]] = 1.0
    predictions = []

    def predict(obs):
        predictions.append(1)
        return plan.copy()

    run_rollout(
        robot,
        SimpleNamespace(action_type="cartesian", predict=predict),
        {
            "frequency": 30,
            "max_steps": max_steps,
            "max_joint_velocity": 12.0,
            "preview": {"enabled": False},
            "ik_recovery": {
                "enabled": True,
                "max_joint_velocity": 0.3,
                **(settings or {}),
            },
        },
        view=view or StopView(),
    )
    return predictions


def test_partial_steps_keep_the_row_and_grippers_until_both_reach(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    robot = RecoveryRobot()
    assert len(run_fake(robot)) == 1
    assert len(robot.commands) == 8
    left = [cmd for arm, cmd in robot.commands if arm == "left"]
    np.testing.assert_allclose([q[0] for q in left], [0.01, 0.02, 0.03, 0.04])
    np.testing.assert_allclose([q[6] for q in left], [0.5, 0.0, 0.0, 1.0])
    assert robot.strict_calls == 4  # Once per arm per new row, not each substep.


def test_recovery_is_default_off(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    robot = RecoveryRobot()
    run_fake(robot, {"enabled": False}, view=StopView(3))
    assert not robot.recovery_calls
    assert not robot.commands


@pytest.mark.parametrize("failure", ["fail_right", "excessive"])
def test_neither_arm_moves_when_one_recovery_fails_or_exceeds_rate(
    monkeypatch, failure
):
    monkeypatch.setattr("time.sleep", lambda _: None)
    robot = RecoveryRobot()
    setattr(robot, failure, True)
    run_fake(robot, view=StopView(3))
    assert not robot.commands


def test_recovery_budget_discards_target_and_replans(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    robot = RecoveryRobot()
    predictions = run_fake(robot, {"max_ticks_per_target": 1}, max_steps=2)
    assert len(predictions) == 2
    assert len(robot.commands) == 4
