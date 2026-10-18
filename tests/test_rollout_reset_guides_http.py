"""Static guide deployment must not require restarting a robot session."""

import shutil
import socket
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import yaml

from egomimic.robot import rollout_dashboard

ROOT = Path(__file__).resolve().parents[1]


def test_reset_guides_are_served_live_without_robot_or_websocket(tmp_path, monkeypatch):
    static = tmp_path / "static"
    shutil.copytree(ROOT / "egomimic/robot/rollout_dashboard_static", static)
    monkeypatch.setattr(rollout_dashboard, "STATIC", static)
    calibration = tmp_path / "camera.yaml"
    calibration.write_text(
        yaml.safe_dump(
            {
                "convention": "point_base = base_T_camera @ point_camera",
                "K": [[100, 0, 32], [0, 100, 24], [0, 0, 1]],
                "dist": [0] * 5,
                "channels": {
                    side: {"base_T_camera": np.eye(4).tolist()}
                    for side in ("left", "right")
                },
            }
        )
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    dashboard = rollout_dashboard.RolloutDashboard(
        ("front_img_1", "left_wrist_img", "right_wrist_img"),
        port=port,
        open_browser=False,
        wait_for_start=True,
        action_overlay={
            "initial_enabled": False,
            "camera": "front_img_1",
            "calibration_path": str(calibration),
            "arm_channels": {"left": "left", "right": "right"},
        },
    )
    try:
        with urlopen(dashboard.url) as response:
            assert response.headers["Cache-Control"] == "no-store"
            assert 'id="reset-guide"' in response.read().decode()
        assert (
            "function setupResetGuides("
            in urlopen(dashboard.url + "/app.js").read().decode()
        )
        assert (
            "pointer-events: none"
            in urlopen(dashboard.url + "/style.css").read().decode()
        )
        with (static / "index.html").open("a") as stream:
            stream.write("<!-- static-refresh-fixture -->")
        assert "static-refresh-fixture" in urlopen(dashboard.url).read().decode()
        assert not dashboard._start_requested.is_set()
        assert not dashboard._restart_requested.is_set()
        assert not dashboard._camera_reconnect_requested.is_set()
        assert not dashboard._video_record_requested.is_set()
    finally:
        dashboard.close()
