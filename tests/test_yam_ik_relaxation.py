"""Bounded pose relaxation and its rollout/dashboard controls; no hardware."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from egomimic.robot.ik_options import validate_ik_relaxation
from egomimic.robot.rollout import run_rollout, validate_rollout_config
from egomimic.robot.rollout_dashboard import RolloutDashboard
from egomimic.robot.yam.kinematics import MujocoArmKinematics
from tests.test_robot_runtime import FakeRobot, View


@pytest.fixture
def solver(tmp_path):
    pytest.importorskip("mujoco")
    # A deliberately singular six-hinge chain: all axes coincide. Position
    # and yaw are coupled, giving provably incompatible pose requests.
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


def test_incompatible_rotation_accepted_only_inside_both_envelopes(solver):
    target = solver.fk(np.zeros(6)).copy()
    target[:3, :3] = Rotation.from_euler("z", 0.04).as_matrix()
    with pytest.raises(ValueError):
        solver.ik(target, np.zeros(6))
    q = solver.ik_relaxed(target, np.zeros(6), 0.005, np.deg2rad(3))
    pose = solver.fk(q)
    assert np.linalg.norm(target[:3, 3] - pose[:3, 3]) <= 0.005
    assert Rotation.from_matrix(
        target[:3, :3] @ pose[:3, :3].T
    ).magnitude() <= np.deg2rad(3)
    with pytest.raises(ValueError):
        solver.ik_relaxed(target, np.zeros(6), 0.005, np.deg2rad(1))


def test_relaxed_retry_actually_moves_to_a_nearby_feasible_pose(solver):
    target = solver.fk(np.full(6, 0.01 / 6)).copy()
    target[:3, :3] = Rotation.from_euler("z", 0.05).as_matrix()
    q = solver.ik_relaxed(target, np.zeros(6), 0.005, np.deg2rad(3))
    assert np.linalg.norm(q) > 0
    assert (np.abs(q) <= 1).all()
    pose = solver.fk(q)
    assert np.linalg.norm(target[:3, 3] - pose[:3, 3]) <= 0.005
    assert Rotation.from_matrix(
        target[:3, :3] @ pose[:3, :3].T
    ).magnitude() <= np.deg2rad(3)


def test_position_error_cannot_be_hidden_by_good_orientation(solver):
    target = solver.fk(np.zeros(6)).copy()
    target[0, 3] += 0.02
    with pytest.raises(ValueError):
        solver.ik_relaxed(target, np.zeros(6), 0.005, np.deg2rad(3))


@pytest.mark.parametrize(
    "config",
    [
        {"enabled": 1},
        {"position_tolerance_m": float("nan")},
        {"rotation_tolerance_deg": float("inf")},
        {"position_tolerance_m": 0},
        {"rotation_tolerance_deg": 16},
        {"position_tolerance_m": 0.021},
        {"position_tolerance_m": True},
        {"unknown": True},
        [],
    ],
)
def test_invalid_relaxation_config_is_rejected_even_when_disabled(config):
    with pytest.raises(ValueError):
        validate_ik_relaxation(config)


def test_relaxation_rejects_unsupported_embodiment_before_devices():
    with pytest.raises(ValueError, match="only for Yam"):
        validate_rollout_config(
            {"ik_relaxation": {"enabled": True}, "robot": {"kind": "eva"}}
        )


def test_dashboard_toggle_is_boolean_queued_and_reports_only_applied_state():
    dashboard = RolloutDashboard.__new__(RolloutDashboard)
    dashboard._lock = threading.RLock()
    dashboard._ik_relaxation = {"available": False, "enabled": False}
    dashboard._ik_relaxation_request = None
    assert not dashboard.request_ik_relaxation(True)
    dashboard.set_ik_relaxation({"available": True, **validate_ik_relaxation()})
    assert not dashboard.request_ik_relaxation(1)
    assert dashboard.request_ik_relaxation(True)
    assert not dashboard._ik_relaxation["enabled"]
    assert dashboard.take_ik_relaxation_request() is True
    assert dashboard.take_ik_relaxation_request() is None
    dashboard.request_ik_relaxation(False)
    assert dashboard.take_ik_relaxation_request() is False


class RelaxRobot(FakeRobot):
    def __init__(self):
        super().__init__()
        self.strict_calls = 0
        self.relaxed_calls = []
        self.relaxed_step = 0.01
        self.fail_right = False

    def solve_ik(self, pose, arm):
        self.strict_calls += 1
        if arm == "left" or self.fail_right:
            raise ValueError("Nonconverged")
        return super().solve_ik(pose, arm)

    def solve_ik_relaxed(self, pose, arm, **kwargs):
        self.relaxed_calls.append((arm, kwargs))
        if arm == "right" and self.fail_right:
            raise ValueError("Outside envelope")
        return np.asarray(pose) + self.relaxed_step


class ToggleView(View):
    def __init__(self, toggles=None, stop_at=6):
        super().__init__()
        self.ticks = 0
        self.toggles = toggles or {}
        self.stop_at = stop_at
        self.settings = []

    def update(self, obs):
        self.ticks += 1
        return "q" if self.ticks >= self.stop_at else None

    def take_ik_relaxation_request(self):
        return self.toggles.get(self.ticks)

    def set_ik_relaxation(self, settings):
        self.settings.append(settings.copy())


def run_fake(robot, view, enabled=False, **config):
    predictions = []

    def predict(obs):
        predictions.append(1)
        return np.tile(robot.q.copy(), (3, 1))

    run_rollout(
        robot,
        SimpleNamespace(action_type="cartesian", predict=predict),
        {
            "frequency": 30,
            "max_joint_velocity": 1.0,
            "max_velocity_replans": 8,
            "preview": {"enabled": False},
            "ik_relaxation": {"enabled": enabled},
            **config,
        },
        view=view,
    )
    return predictions


def test_default_off_and_toggle_on_then_off_clears_queued_plan(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    robot, view = RelaxRobot(), ToggleView({2: True, 3: False})
    predictions = run_fake(robot, view)
    # Tick 1 strict fails; tick 2 accepts fallback; subsequent ticks strict fail.
    assert len(robot.relaxed_calls) == 1
    assert [a for a, _ in robot.commands] == ["left", "right"]
    assert len(predictions) == 5
    assert [s["enabled"] for s in view.settings] == [False, True, False]


def test_relaxation_still_checks_both_arms_and_joint_step(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    robot = RelaxRobot()
    robot.fail_right = True
    run_fake(robot, ToggleView(stop_at=3), enabled=True)
    assert not robot.commands
    robot = RelaxRobot()
    robot.relaxed_step = 0.2  # Bigger than the 1/30 rad command bound.
    run_fake(robot, ToggleView(stop_at=3), enabled=True)
    assert not robot.commands


def test_enabled_relaxation_keeps_successful_strict_solutions(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    robot = RelaxRobot()
    robot.solve_ik = lambda pose, arm: np.asarray(pose).copy()
    run_fake(robot, ToggleView(stop_at=3), enabled=True)
    assert not robot.relaxed_calls
    assert len(robot.commands) == 4


def test_websocket_transports_toggle_and_applied_state(tmp_path):
    import asyncio

    from aiohttp import ClientSession

    from tests.test_rollout_dashboard import (
        available_loopback_port,
        calibration_file,
        overlay_config,
    )

    dashboard = RolloutDashboard(
        ("front_img_1",),
        port=available_loopback_port(),
        open_browser=False,
        action_overlay=overlay_config(calibration_file(tmp_path)),
    )
    settings = {"available": True, **validate_ik_relaxation()}
    dashboard.set_ik_relaxation(settings)

    async def exercise():
        async with ClientSession() as session:
            async with session.ws_connect(f"{dashboard.url}/ws") as ws:
                assert (await ws.receive_json())["ik_relaxation"] == settings
                await ws.send_json({"ik_relaxation": True})
                for _ in range(100):
                    request = dashboard.take_ik_relaxation_request()
                    if request is not None:
                        break
                    await asyncio.sleep(0.01)
                assert request is True
                assert dashboard._snapshot()["ik_relaxation"]["enabled"] is False
                dashboard.set_ik_relaxation({**settings, "enabled": True})
                for _ in range(20):
                    message = await asyncio.wait_for(ws.receive_json(), timeout=2)
                    if message.get("ik_relaxation", {}).get("enabled"):
                        break
                assert message["ik_relaxation"]["enabled"] is True
            async with session.ws_connect(f"{dashboard.url}/ws") as ws:
                assert (await ws.receive_json())["ik_relaxation"]["enabled"] is True

    try:
        asyncio.run(exercise())
    finally:
        dashboard.close()
