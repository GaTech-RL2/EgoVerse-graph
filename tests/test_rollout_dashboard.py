"""Hardware-free checks for the YAM rollout browser view and its overlay."""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from egomimic.robot.interface import ARM_OFFSET
from egomimic.robot.rollout import run_rollout
from egomimic.robot.rollout_dashboard import (
    SUPERSEDED_CLOSE_CODE,
    CheckpointBrowser,
    RolloutDashboard,
    _broadcast_dashboard_message,
    load_action_overlay,
    validate_rollout_preview,
)
from egomimic.robot.rollout_video import (
    RolloutVideoRecorder,
    list_rollout_videos,
    rollout_video_path,
    validate_video_recording,
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


def available_loopback_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


class DashboardSocket:
    def __init__(self, error=None, closed=False):
        self.error = error
        self.closed = closed
        self.messages = []

    async def send_json(self, message):
        if self.error is not None:
            raise self.error
        self.messages.append(message)


def test_dashboard_broadcast_discards_only_a_disconnected_browser():
    healthy = DashboardSocket()
    reset = DashboardSocket(error=ConnectionResetError())
    closed = DashboardSocket(closed=True)

    disconnected = asyncio.run(
        _broadcast_dashboard_message({healthy, reset, closed}, {"type": "frame"})
    )

    assert disconnected == {reset, closed}
    assert healthy.messages == [{"type": "frame"}]


def test_dashboard_start_command_reaches_rollout_start_gate(tmp_path):
    """A browser c/Start event becomes the rollout loop's c control."""
    dashboard = RolloutDashboard(
        ("front_img_1",),
        host="127.0.0.1",
        port=available_loopback_port(),
        open_browser=False,
        wait_for_start=True,
        action_overlay=overlay_config(calibration_file(tmp_path)),
    )

    async def request_start():
        from aiohttp import ClientSession

        async with ClientSession() as session:
            async with session.ws_connect(f"{dashboard.url}/ws") as ws:
                config = await ws.receive_json()
                assert config["type"] == "config"
                assert config["wait_for_start"] is True
                await ws.send_json({"start": True})

    try:
        asyncio.run(request_start())
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if dashboard.update({}) == "c":
                break
            time.sleep(0.01)
        else:
            pytest.fail("browser start command did not reach the rollout gate")
    finally:
        dashboard.close()


def test_dashboard_pause_command_toggles_only_after_start(tmp_path):
    dashboard = RolloutDashboard(
        ("front_img_1",),
        host="127.0.0.1",
        port=available_loopback_port(),
        open_browser=False,
        wait_for_start=True,
        action_overlay=overlay_config(calibration_file(tmp_path)),
    )

    async def toggle_pause():
        from aiohttp import ClientSession

        async with ClientSession() as session:
            async with session.ws_connect(f"{dashboard.url}/ws") as ws:
                await ws.receive_json()
                await ws.send_json({"paused": True})
                await asyncio.sleep(0.02)
                assert not dashboard.is_paused()
                await ws.send_json({"start": True})
                await ws.send_json({"paused": True})
                deadline = time.monotonic() + 1.0
                while not dashboard.is_paused():
                    if time.monotonic() >= deadline:
                        pytest.fail("pause command did not reach the dashboard")
                    await asyncio.sleep(0.01)
                await ws.send_json({"paused": False})
                while dashboard.is_paused():
                    if time.monotonic() >= deadline:
                        pytest.fail("resume command did not reach the dashboard")
                    await asyncio.sleep(0.01)

    try:
        asyncio.run(toggle_pause())
    finally:
        dashboard.close()


def test_dashboard_camera_reconnect_clears_the_start_gate(tmp_path):
    dashboard = RolloutDashboard(
        ("front_img_1",),
        host="127.0.0.1",
        port=available_loopback_port(),
        open_browser=False,
        wait_for_start=True,
        action_overlay=overlay_config(calibration_file(tmp_path)),
    )

    async def request_reconnect():
        from aiohttp import ClientSession

        async with ClientSession() as session:
            async with session.ws_connect(f"{dashboard.url}/ws") as ws:
                await ws.receive_json()
                await ws.send_json({"start": True})
                await ws.send_json({"reconnect_cameras": True})
                deadline = time.monotonic() + 1.0
                while not dashboard._camera_reconnect_requested.is_set():
                    if time.monotonic() >= deadline:
                        pytest.fail("camera reconnect did not reach dashboard")
                    await asyncio.sleep(0.01)

    try:
        asyncio.run(request_reconnect())
        assert dashboard.take_camera_reconnect_request()
        assert not dashboard._start_requested.is_set()
        assert not dashboard.is_paused()
    finally:
        dashboard.close()


def test_dashboard_camera_reconnect_interrupts_velocity_prompt(tmp_path):
    dashboard = RolloutDashboard(
        ("front_img_1",),
        host="127.0.0.1",
        port=available_loopback_port(),
        open_browser=False,
        action_overlay=overlay_config(calibration_file(tmp_path)),
    )
    decision = []

    def wait_for_velocity_choice():
        decision.append(
            dashboard.choose_velocity_action(
                {"arms": ["left"], "max_joint_step": 0.5, "limit": 0.4}
            )
        )

    try:
        thread = threading.Thread(target=wait_for_velocity_choice)
        thread.start()
        deadline = time.monotonic() + 1.0
        while dashboard._snapshot()["velocity_prompt"] is None:
            if time.monotonic() >= deadline:
                pytest.fail("velocity prompt was not published")
            time.sleep(0.01)
        dashboard.request_camera_reconnect()
        thread.join(timeout=1.0)
        assert not thread.is_alive()
        assert decision == ["reconnect"]
        assert dashboard.take_camera_reconnect_request()
    finally:
        dashboard.close()


def test_dashboard_resample_interval_updates_from_browser(tmp_path):
    dashboard = RolloutDashboard(
        ("front_img_1",),
        execute_steps=30,
        host="127.0.0.1",
        port=available_loopback_port(),
        open_browser=False,
        wait_for_start=True,
        action_overlay=overlay_config(calibration_file(tmp_path)),
    )

    async def set_interval():
        from aiohttp import ClientSession

        async with ClientSession() as session:
            async with session.ws_connect(f"{dashboard.url}/ws") as ws:
                config = await ws.receive_json()
                assert config["execute_steps"] == 30
                await ws.send_json({"execute_steps": 17})
                deadline = time.monotonic() + 1.0
                while dashboard.get_execute_steps() != 17:
                    if time.monotonic() >= deadline:
                        pytest.fail("resample interval did not reach dashboard")
                    await asyncio.sleep(0.01)

    try:
        asyncio.run(set_interval())
    finally:
        dashboard.close()


def test_dashboard_reports_rolling_inference_latency(tmp_path):
    dashboard = RolloutDashboard(
        ("front_img_1",),
        host="127.0.0.1",
        port=available_loopback_port(),
        open_browser=False,
        action_overlay=overlay_config(calibration_file(tmp_path)),
    )
    try:
        dashboard.record_inference(0.1)
        dashboard.record_inference(0.2)
        inference = dashboard._snapshot()["inference"]
        assert inference["samples"] == 2
        assert inference["last_ms"] == pytest.approx(200.0)
        assert inference["mean_ms"] == pytest.approx(150.0)
        assert inference["plans_per_second"] == pytest.approx(1000.0 / 150.0)
    finally:
        dashboard.close()


def test_dashboard_video_record_command_reaches_only_the_rollout_loop(tmp_path):
    dashboard = RolloutDashboard(
        ("front_img_1",),
        host="127.0.0.1",
        port=available_loopback_port(),
        open_browser=False,
        action_overlay=overlay_config(calibration_file(tmp_path)),
        video_recording={"enabled": True, "directory": str(tmp_path), "fps": 12},
    )

    async def request_video_recording():
        from aiohttp import ClientSession

        async with ClientSession() as session:
            async with session.ws_connect(f"{dashboard.url}/ws") as ws:
                config = await ws.receive_json()
                assert config["video_recording_enabled"] is True
                await ws.send_json({"record_video": True})
                deadline = time.monotonic() + 1.0
                while not dashboard._video_record_requested.is_set():
                    if time.monotonic() >= deadline:
                        pytest.fail("video record request did not reach dashboard")
                    await asyncio.sleep(0.01)

    try:
        asyncio.run(request_video_recording())
        assert dashboard.take_video_recording_request()
        dashboard.set_video_recording(True)
        assert dashboard._snapshot()["video_recording"] is True
    finally:
        dashboard.close()


def test_checkpoint_browser_lists_only_rooted_checkpoint_candidates(tmp_path):
    root = tmp_path / "models"
    run = root / "run_a"
    run.mkdir(parents=True)
    checkpoint = run / "model.ckpt"
    checkpoint.write_bytes(b"weights")
    (run / "resolved-config.yaml").write_text("model: {}\n")
    (run / "norm_stats.json").write_text("{}\n")
    (run / "notes.txt").write_text("not a checkpoint")
    outside = tmp_path / "outside.ckpt"
    outside.write_bytes(b"outside")
    (root / "outside-link.ckpt").symlink_to(outside)
    browser = CheckpointBrowser(root)

    assert browser.list_directory()["entries"] == [
        {"type": "directory", "name": "run_a", "path": "run_a"}
    ]
    assert browser.list_directory("run_a")["entries"] == [
        {"type": "checkpoint", "name": "model.ckpt", "path": "run_a/model.ckpt"}
    ]
    assert browser.resolve_bundle("run_a/model.ckpt").checkpoint == checkpoint.resolve()
    with pytest.raises(ValueError, match="escapes"):
        browser.resolve_bundle("../outside.ckpt")


def test_dashboard_model_swap_reaches_only_rollout_loop(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    checkpoint = root / "next.ckpt"
    checkpoint.write_bytes(b"weights")
    training_config = root / "resolved-config.yaml"
    training_config.write_text("model: {}\n")
    normalizer = root / "norm_stats.json"
    normalizer.write_text("{}\n")
    dashboard = RolloutDashboard(
        ("front_img_1",),
        host="127.0.0.1",
        port=available_loopback_port(),
        open_browser=False,
        wait_for_start=True,
        action_overlay=overlay_config(calibration_file(tmp_path)),
        model_browser={"enabled": True, "root": str(root)},
        policy={
            "checkpoint": str(checkpoint),
            "training_config": str(training_config),
            "normalizer_path": str(normalizer),
        },
    )

    async def request_model_swap():
        from aiohttp import ClientSession

        async with ClientSession() as session:
            async with session.ws_connect(f"{dashboard.url}/ws") as ws:
                config = await ws.receive_json()
                assert config["model_browser_enabled"] is True
                assert config["checkpoint"] == "next.ckpt"
                await ws.send_json({"swap_model": "next.ckpt"})
                deadline = time.monotonic() + 1.0
                while dashboard._model_swap_checkpoint is None:
                    if time.monotonic() >= deadline:
                        pytest.fail("model swap did not reach dashboard")
                    await asyncio.sleep(0.01)

    try:
        asyncio.run(request_model_swap())
        assert dashboard.take_model_swap_request().checkpoint == checkpoint.resolve()
        assert not dashboard._start_requested.is_set()
    finally:
        dashboard.close()


def test_rollout_video_recorder_publishes_completed_mosaic_and_manifest(tmp_path):
    config = {"enabled": True, "directory": str(tmp_path), "fps": 10}
    assert validate_video_recording(config) == config
    recorder = RolloutVideoRecorder(("front", "wrist"), config)
    recorder.start()
    frames = {
        "front": np.full((48, 64, 3), 40, dtype=np.uint8),
        "wrist": np.full((48, 64, 3), 180, dtype=np.uint8),
    }
    assert recorder.append(frames, now=10.0)
    assert not recorder.append(frames, now=10.01)
    assert recorder.append(frames, now=10.11)
    saved = recorder.close()

    assert saved is not None
    assert (tmp_path / saved["filename"]).is_file()
    assert (tmp_path / saved["filename"]).stat().st_size > 0
    videos = list_rollout_videos(tmp_path)
    assert len(videos) == 1
    assert videos[0]["id"] == saved["id"]
    assert videos[0]["filename"] == saved["filename"]
    assert videos[0]["frames"] == 2
    assert videos[0]["fps"] == 10
    assert videos[0]["duration_seconds"] == pytest.approx(0.2)
    assert rollout_video_path(tmp_path, saved["id"]) == tmp_path / saved["filename"]


def test_dashboard_restart_dismisses_velocity_prompt_and_returns_to_ready_gate(
    tmp_path,
):
    dashboard = RolloutDashboard(
        ("front_img_1",),
        host="127.0.0.1",
        port=available_loopback_port(),
        open_browser=False,
        wait_for_start=True,
        action_overlay=overlay_config(calibration_file(tmp_path)),
    )
    decision = []
    finished = threading.Event()

    def wait_for_velocity_choice():
        decision.append(
            dashboard.choose_velocity_action(
                {"arms": ["left"], "max_joint_step": 0.5, "limit": 0.4}
            )
        )
        finished.set()

    async def request_restart():
        from aiohttp import ClientSession

        async with ClientSession() as session:
            async with session.ws_connect(f"{dashboard.url}/ws") as ws:
                await ws.receive_json()
                deadline = time.monotonic() + 1.0
                while dashboard._snapshot()["velocity_prompt"] is None:
                    if time.monotonic() >= deadline:
                        pytest.fail("velocity prompt was not published")
                    await asyncio.sleep(0.01)
                await ws.send_json({"restart": True})

    try:
        thread = threading.Thread(target=wait_for_velocity_choice)
        thread.start()
        asyncio.run(request_restart())
        assert finished.wait(timeout=1.0)
        thread.join(timeout=1.0)
        assert decision == ["restart"]
        assert dashboard._snapshot()["velocity_prompt"] is None
    finally:
        dashboard.close()


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
    assert profile["max_joint_velocity"] / profile["frequency"] == 0.4
    assert profile["execute_steps"] == 30
    assert profile["policy"]["num_inference_steps"] == 10
    assert profile["reset_on_start"] is True
    assert profile["reset_home_on_restart"] is True
    assert profile["video_recording"] == {
        "enabled": True,
        "directory": "/home/rohan/rollouts/yam_hptflow",
        "fps": 12,
    }
    assert profile["model_browser"] == {
        "enabled": True,
        "root": "/home/rohan/checkpoints/EgoVerse",
    }
    assert set(adapter["camera_keys"]) == {
        "front_img_1",
        "left_wrist_img",
        "right_wrist_img",
    }


def test_dashboard_uses_space_for_pause_and_places_resample_below_cameras():
    static = ROOT / "egomimic/robot/rollout_dashboard_static"
    html = (static / "index.html").read_text()
    javascript = (static / "app.js").read_text()

    assert "Pause rollout <kbd>Space</kbd>" in html
    assert html.index('id="cameras"') < html.index('id="execute-steps"')
    assert "event.code === 'Space'" in javascript
    assert "event.key === 'v'" in javascript
    assert 'id="recording-indicator"' in html
    assert 'id="open-videos"' in html
    assert 'id="swap-model"' in html
    assert "/api/checkpoints" in javascript
    assert "event.key === 'p'" not in javascript


class FakeRobot:
    def __init__(self):
        self.arms = ["left", "right"]
        self.camera_res = {"front_img_1": (2, 3)}
        self.q = np.zeros(14)
        self.q[[6, 13]] = 0.5
        self.commands = []
        self.home_calls = 0

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

    def set_home(self):
        self.home_calls += 1


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


class VelocityChoiceView(GatedView):
    def __init__(self, controls, decision):
        super().__init__(controls)
        self.decision = decision
        self.velocity_details = []

    def choose_velocity_action(self, details):
        self.velocity_details.append(details)
        return self.decision


class PauseView(GatedView):
    def __init__(self, controls, pauses):
        super().__init__(controls)
        self.pauses = iter(pauses)
        self.paused = False

    def update(self, obs):
        self.paused = next(self.pauses)
        return super().update(obs)

    def is_paused(self):
        return self.paused


class ResampleIntervalView(GatedView):
    def __init__(self, controls, execute_steps):
        super().__init__(controls)
        self.execute_steps = execute_steps

    def get_execute_steps(self):
        return self.execute_steps


class InferenceView(View):
    def __init__(self):
        super().__init__()
        self.inference_seconds = []

    def record_inference(self, seconds):
        self.inference_seconds.append(seconds)


class CameraRecoveryView(GatedView):
    def __init__(self, controls):
        super().__init__(controls)
        self.reconnect_requests = iter((True, False, False))

    def take_camera_reconnect_request(self):
        return next(self.reconnect_requests)


class ModelSwapView(GatedView):
    def __init__(self, controls, checkpoint):
        super().__init__(controls)
        self.checkpoints = iter((checkpoint, None, None))
        self.loaded = []

    def take_model_swap_request(self):
        return next(self.checkpoints)

    def set_model_checkpoint(self, checkpoint):
        self.loaded.append(str(checkpoint.checkpoint))


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


def test_rollout_homes_on_startup_and_restart_when_enabled(monkeypatch):
    monkeypatch.setattr("egomimic.robot.rollout.time.sleep", lambda _: None)
    robot = FakeRobot()
    view = GatedView(["r", "q"])
    policy = SimpleNamespace(
        action_type="joints",
        predict=lambda _obs: pytest.fail("policy should wait for c"),
    )

    steps = run_rollout(
        robot,
        policy,
        {
            "frequency": 30,
            "max_steps": 4,
            "execute_steps": 1,
            "max_joint_velocity": 1.0,
            "reset_on_start": True,
            "reset_home_on_restart": True,
            "preview": {"enabled": False, "wait_for_start": True},
        },
        view=view,
    )

    assert steps == 0
    assert robot.home_calls == 2
    assert not robot.commands
    assert sum("Resetting YAM" in status for status in view.statuses) == 2


def test_rollout_camera_reconnect_preserves_process_and_requires_c(monkeypatch):
    monkeypatch.setattr("egomimic.robot.rollout.time.sleep", lambda _: None)
    robot = FakeRobot()
    robot.camera_reconnects = 0

    def reconnect_cameras():
        robot.camera_reconnects += 1

    robot.reconnect_cameras = reconnect_cameras
    view = CameraRecoveryView([None, "c", "q"])
    target = np.zeros((1, 14), dtype=float)
    target[:, [6, 13]] = 0.5

    steps = run_rollout(
        robot,
        SimpleNamespace(action_type="joints", predict=lambda _obs: target),
        {
            "frequency": 30,
            "max_steps": 4,
            "execute_steps": 1,
            "max_joint_velocity": 1.0,
            "preview": {"enabled": False, "wait_for_start": True},
        },
        view=view,
    )

    assert steps == 1
    assert robot.camera_reconnects == 1
    assert len(robot.commands) == 2
    assert any("Reconnecting RGB cameras" in status for status in view.statuses)


def test_rollout_model_swap_holds_and_requires_a_fresh_start(monkeypatch, tmp_path):
    monkeypatch.setattr("egomimic.robot.rollout.time.sleep", lambda _: None)
    checkpoint = tmp_path / "next.ckpt"
    checkpoint.write_bytes(b"weights")
    training_config = tmp_path / "resolved-config.yaml"
    training_config.write_text("model: {}\n")
    normalizer = tmp_path / "norm_stats.json"
    normalizer.write_text("{}\n")
    bundle = CheckpointBrowser(tmp_path).resolve_bundle("next.ckpt")
    robot = FakeRobot()
    view = ModelSwapView([None, "c", "q"], bundle)
    target = np.zeros((1, 14), dtype=float)
    target[:, [6, 13]] = 0.5
    replacement = SimpleNamespace(action_type="joints", predict=lambda _obs: target)
    loaded = []
    monkeypatch.setattr(
        "egomimic.robot.rollout.load_policy",
        lambda config: loaded.append(config) or replacement,
    )

    steps = run_rollout(
        robot,
        SimpleNamespace(
            action_type="joints", predict=lambda _obs: pytest.fail("old model ran")
        ),
        {
            "frequency": 30,
            "max_steps": 4,
            "execute_steps": 1,
            "max_joint_velocity": 1.0,
            "preview": {"enabled": False, "wait_for_start": True},
            "policy": {
                "kind": "graph",
                "checkpoint": "/old/model.ckpt",
                "training_config": "/old/resolved-config.yaml",
                "normalizer_path": "/old/norm_stats.json",
            },
        },
        view=view,
    )

    assert steps == 1
    assert loaded == [
        {
            "kind": "graph",
            "checkpoint": str(checkpoint),
            "training_config": str(training_config),
            "normalizer_path": str(normalizer),
        }
    ]
    assert view.loaded == [str(checkpoint)]
    assert len(robot.commands) == 4  # paired hold, then paired new-model command


def test_rollout_pause_holds_measured_joints_and_discards_policy_queue(monkeypatch):
    monkeypatch.setattr("egomimic.robot.rollout.time.sleep", lambda _: None)
    robot = FakeRobot()
    view = PauseView(controls=[None, None, None, "q"], pauses=[False, True, True, True])
    target = np.zeros((1, 14), dtype=float)
    target[:, [6, 13]] = 0.5
    calls = 0

    def predict(_obs):
        nonlocal calls
        calls += 1
        return target

    steps = run_rollout(
        robot,
        SimpleNamespace(action_type="joints", predict=predict),
        {
            "frequency": 30,
            "max_steps": 4,
            "execute_steps": 1,
            "max_joint_velocity": 1.0,
            "preview": {"enabled": False},
        },
        view=view,
    )

    assert steps == 1
    assert calls == 1
    assert len(robot.commands) == 4  # First paired policy target, then paired hold.
    assert any("Paused" in status for status in view.statuses)


def test_rollout_uses_dashboard_resample_interval_on_next_plan(monkeypatch):
    monkeypatch.setattr("egomimic.robot.rollout.time.sleep", lambda _: None)
    robot = FakeRobot()
    view = ResampleIntervalView([None, None, "q"], execute_steps=1)
    target = np.zeros((3, 14), dtype=float)
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
            "execute_steps": 3,
            "max_joint_velocity": 1.0,
            "preview": {"enabled": False},
        },
        view=view,
    )

    assert steps == 2
    assert calls == 2


def test_rollout_records_command_ready_inference_latency(monkeypatch):
    monkeypatch.setattr("egomimic.robot.rollout.time.sleep", lambda _: None)
    robot, view = FakeRobot(), InferenceView()
    target = np.zeros((1, 14), dtype=float)
    target[:, [6, 13]] = 0.5

    steps = run_rollout(
        robot,
        SimpleNamespace(action_type="cartesian", predict=lambda _obs: target),
        {
            "frequency": 30,
            "max_steps": 4,
            "execute_steps": 1,
            "max_joint_velocity": 1.0,
            "preview": {"enabled": False},
        },
        view=view,
    )

    assert steps == 1
    assert len(view.inference_seconds) == 1
    assert view.inference_seconds[0] >= 0


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


def test_rollout_executes_velocity_unsafe_pair_only_after_explicit_choice(monkeypatch):
    monkeypatch.setattr("egomimic.robot.rollout.time.sleep", lambda _: None)
    robot = FakeRobot()
    view = VelocityChoiceView([None, "q"], "execute")
    unsafe = np.zeros((1, 14), dtype=float)
    unsafe[:, [0, 7]] = 0.2
    unsafe[:, [6, 13]] = 0.5

    steps = run_rollout(
        robot,
        SimpleNamespace(action_type="joints", predict=lambda _obs: unsafe),
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
    assert len(robot.commands) == 2
    assert view.velocity_details == [
        {"arms": ["left", "right"], "max_joint_step": 0.2, "limit": 1 / 30}
    ]
    assert any("operator-approved" in status for status in view.statuses)


def test_a_new_browser_tab_supersedes_the_stale_dashboard_tab(tmp_path):
    """Tabs left open by finished runs must not keep pulling a live stream."""
    dashboard = RolloutDashboard(
        ("front_img_1",),
        host="127.0.0.1",
        port=available_loopback_port(),
        open_browser=False,
        action_overlay=overlay_config(calibration_file(tmp_path)),
    )

    async def two_tabs():
        from aiohttp import ClientSession, WSMsgType

        async with ClientSession() as session:
            async with session.ws_connect(f"{dashboard.url}/ws") as stale:
                assert (await stale.receive_json())["type"] == "config"
                async with session.ws_connect(f"{dashboard.url}/ws") as current:
                    assert (await current.receive_json())["type"] == "config"
                    deadline = time.monotonic() + 5.0
                    while time.monotonic() < deadline:
                        message = await stale.receive()
                        if message.type in (
                            WSMsgType.CLOSE,
                            WSMsgType.CLOSED,
                            WSMsgType.CLOSING,
                        ):
                            break
                    assert not current.closed
                    return stale.close_code

    try:
        assert asyncio.run(two_tabs()) == SUPERSEDED_CLOSE_CODE
    finally:
        dashboard.close()
