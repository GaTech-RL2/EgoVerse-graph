"""Optional endpoint solver parity, rejection and dashboard/runtime controls."""

from types import SimpleNamespace

import numpy as np
import pytest

from egomimic.robot.ik_options import validate_ik_solver
from egomimic.robot.rollout import validate_rollout_config
from egomimic.robot.yam.interface import YamInterface
from egomimic.robot.yam.kinematics import MujocoArmKinematics
from egomimic.robot.yam.tracik import TracIKEndpoint, chain_urdf
from tests.test_robot_runtime import FakeRobot
from tests.test_yam_ik_relaxation import ToggleView, run_fake


@pytest.fixture
def chain(tmp_path):
    pytest.importorskip("mujoco")
    axes = ("0 0 1", "0 1 0", "0 1 0", "1 0 0", "0 1 0", "0 0 1")
    xml = '<mujoco><compiler angle="radian"/><worldbody><body pos=".1 -.2 .3" euler=".1 .2 .3">'
    for i, axis in enumerate(axes):
        xml += f'<body pos=".1 .02 .03" euler=".03 .02 .01"><joint name="j{i}" axis="{axis}" pos=".01 0 .01" range="-2.12345 2.54321" ref=".1"/><geom size=".01" mass="1"/>'
    xml += '<body pos=".01 .02 .03"><site name="tcp_site" pos=".06 -.02 .07" euler=".2 .3 .4"/></body>'
    xml += "</body>" * 7 + "</worldbody></mujoco>"
    path = tmp_path / "arm.xml"
    path.write_text(xml)
    return MujocoArmKinematics(path, [f"j{i}" for i in range(6)])


def test_real_tracik_matches_nontrivial_frames_references_and_tcp(chain):
    pytest.importorskip("trac_ik")
    endpoint = TracIKEndpoint(chain)
    assert np.allclose(
        endpoint.solver.joint_limits,
        chain.model.jnt_range[chain.joints].T,
        rtol=0,
        atol=1e-12,
    )
    q = np.array([0.2, 0.4, -0.3, 0.1, -0.2, 0.3])
    target = chain.fk(q)
    result = endpoint.ik(target, q + 0.01)
    assert np.allclose(chain.fk(result), target, atol=1e-4)
    with pytest.raises(ValueError):
        endpoint.ik(
            np.diag([1.0, 1.0, 1.0, 1.0])
            + np.array([[0, 0, 0, 10], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
            q,
        )


def test_reordered_chain_rejected(chain):
    chain.joints = list(reversed(chain.joints))
    with pytest.raises(ValueError, match="ordered"):
        chain_urdf(chain)


@pytest.mark.parametrize(
    "output", [None, np.full(6, np.nan), np.zeros(7), np.full(6, 3), np.ones(6)]
)
def test_endpoint_independently_rejects_failed_invalid_or_wrong_solutions(
    chain, output
):
    endpoint = TracIKEndpoint.__new__(TracIKEndpoint)
    endpoint.kinematics = chain
    endpoint.lower, endpoint.upper = np.full(6, -2), np.full(6, 2)
    endpoint.solver = SimpleNamespace(ik=lambda *args: output)
    with pytest.raises(ValueError):
        endpoint.ik(chain.fk(np.zeros(6)), np.zeros(6))


def test_fk_parity_failure_refuses_backend(chain, monkeypatch):
    class WrongChain:
        dof = 6
        joint_limits = (np.full(6, -2), np.full(6, 2))

        def fk(self, q):
            return np.zeros(3), np.eye(3)

    monkeypatch.setattr(
        "egomimic.robot.yam.tracik.require_pytracik",
        lambda: lambda *a, **kw: WrongChain(),
    )
    with pytest.raises(ValueError, match="FK disagrees"):
        TracIKEndpoint(chain)


@pytest.mark.parametrize(
    "config",
    [
        [],
        {"backend": "unknown"},
        {"backend": []},
        {"timeout_s": True},
        {"timeout_s": 0},
        {"timeout_s": 0.021},
        {"timeout_s": float("nan")},
        {"solver_type": "unknown"},
        {"extra": 1},
    ],
)
def test_invalid_solver_config(config):
    with pytest.raises(ValueError):
        validate_ik_solver(config)


def test_unsupported_robot_rejected_before_import_or_devices():
    with pytest.raises(ValueError, match="only for Yam"):
        validate_rollout_config(
            {"ik_solver": {"backend": "pytracik"}, "robot": {"kind": "eva"}}
        )


def test_interface_switch_is_atomic_across_arms(monkeypatch):
    robot = YamInterface.__new__(YamInterface)
    robot.arms = ["left", "right"]
    robot.solvers = {"left": object(), "right": object()}
    robot.set_ik_backend("mujoco")

    def build(solver, **kw):
        if solver is robot.solvers["right"]:
            raise ValueError("bad right chain")
        return object()

    monkeypatch.setattr("egomimic.robot.yam.tracik.TracIKEndpoint", build)
    with pytest.raises(ValueError):
        robot.set_ik_backend("pytracik")
    assert robot._endpoint_solvers is robot.solvers
    assert not hasattr(robot, "_tracik_endpoints")


class BackendRobot(FakeRobot):
    def __init__(self, fail=False):
        super().__init__()
        self.backends = []
        self.fail = fail

    def set_ik_backend(self, **settings):
        if self.fail and settings["backend"] == "pytracik":
            raise RuntimeError("missing optional library")
        self.backends.append(settings["backend"])


class BackendView(ToggleView):
    def __init__(self):
        super().__init__(stop_at=7)
        self.solver_states = []

    def take_ik_solver_request(self):
        return {2: "pytracik", 4: "mujoco"}.get(self.ticks)

    def set_ik_solver(self, settings):
        self.solver_states.append(settings)


def test_switch_discards_old_waypoints_and_uses_fresh_observation(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    robot, view = BackendRobot(), BackendView()
    plans = run_fake(robot, view)
    assert robot.backends == ["mujoco", "pytracik", "mujoco"]
    assert len(plans) == 3
    assert len(robot.commands) == 8  # No command on either switching tick.
    assert [s["backend"] for s in view.solver_states] == robot.backends


def test_failed_switch_reports_error_keeps_backend_and_skips_tick(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    robot, view = BackendRobot(fail=True), BackendView()
    run_fake(robot, view)
    assert robot.backends == ["mujoco"]
    assert view.solver_states[1]["backend"] == "mujoco"
    assert "missing optional" in view.solver_states[1]["error"]
    assert len(robot.commands) == 10


def test_backend_websocket_reports_applied_state_and_reconnects(tmp_path):
    import asyncio

    from aiohttp import ClientSession

    from egomimic.robot.rollout_dashboard import RolloutDashboard
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
    assert not dashboard.request_ik_solver("pytracik")
    settings = {**validate_ik_solver(), "available": True, "error": None}
    dashboard.set_ik_solver(settings)
    assert not dashboard.request_ik_solver(True)
    assert not dashboard.request_ik_solver("unknown")

    async def exercise():
        async with ClientSession() as session:
            async with session.ws_connect(f"{dashboard.url}/ws") as ws:
                assert (await ws.receive_json())["ik_solver"]["backend"] == "mujoco"
                await ws.send_json({"ik_solver": "pytracik"})
                for _ in range(100):
                    request = dashboard.take_ik_solver_request()
                    if request is not None:
                        break
                    await asyncio.sleep(0.01)
                assert request == "pytracik"
                assert dashboard._snapshot()["ik_solver"]["backend"] == "mujoco"
                dashboard.set_ik_solver({**settings, "backend": "pytracik"})
                for _ in range(20):
                    message = await asyncio.wait_for(ws.receive_json(), timeout=2)
                    if message.get("ik_solver", {}).get("backend") == "pytracik":
                        break
                assert message["ik_solver"]["backend"] == "pytracik"
            async with session.ws_connect(f"{dashboard.url}/ws") as ws:
                assert (await ws.receive_json())["ik_solver"]["backend"] == "pytracik"

    try:
        asyncio.run(exercise())
    finally:
        dashboard.close()
