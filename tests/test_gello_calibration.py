"""Hardware-free checks for the leader-only GELLO calibration utility."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from egomimic.robot import calibrate_gello

PROFILE = Path("egomimic/hydra_configs/robot/yam_rl2_gello_collect.yaml")


def profile():
    return yaml.safe_load(PROFILE.read_text())


def session(arm="left"):
    return calibrate_gello.GelloCalibration(arm, profile()["gello"]["leaders"][arm])


def test_calibration_captures_raw_offsets_endpoints_and_signs():
    calibration = session()
    raw = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    calibration.update(raw)
    calibration.capture_joint_offsets()
    calibration.capture_gripper_open()
    calibration.update(np.r_[raw[:6], -0.2])
    calibration.capture_gripper_closed()
    calibration.toggle_joint_sign(1)

    snippet = yaml.safe_load(calibration.yaml_text())
    leader = snippet["gello"]["leaders"]["left"]
    assert leader["joint_offsets_rad"] == pytest.approx(raw[:6])
    assert leader["gripper_open_rad"] == pytest.approx(0.7)
    assert leader["gripper_closed_rad"] == pytest.approx(-0.2)
    assert leader["joint_signs"] == [1, -1, 1, 1, 1, 1]


def test_calibration_refuses_incomplete_or_overwrite(tmp_path):
    calibration = session()
    calibration.update(np.arange(7, dtype=float))
    with pytest.raises(RuntimeError, match="Capture joint zero"):
        calibration.yaml_text()
    calibration.capture_joint_offsets()
    calibration.capture_gripper_open()
    calibration.update(np.r_[np.arange(6, dtype=float), 9.0])
    calibration.capture_gripper_closed()
    path = calibration.write_yaml(tmp_path / "left.yaml")
    assert path.exists()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        calibration.write_yaml(path)


def test_incomplete_export_reports_an_error_without_ending_session():
    calibration = session()
    messages = []
    assert not calibrate_gello._handle_key("p", calibration, None, messages.append)
    assert "Calibration not exported" in messages[-1]


def test_check_config_opens_no_leader_or_other_hardware(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("calibration config check touched a device")

    monkeypatch.setattr(calibrate_gello, "DynamixelPositionBus", forbidden)
    monkeypatch.setattr(calibrate_gello, "validate_serial_ports", forbidden)
    assert calibrate_gello.main(["--config", str(PROFILE), "--check-config"]) == 0
    assert "no devices were opened" in capsys.readouterr().out


def test_interactive_calibration_requires_tty_before_opening_leader(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("leader opened before interactive terminal validation")

    monkeypatch.setattr(calibrate_gello, "validate_serial_ports", lambda ports: ports)
    monkeypatch.setattr(calibrate_gello, "DynamixelPositionBus", forbidden)
    monkeypatch.setattr(
        calibrate_gello,
        "TerminalKeyView",
        lambda: (_ for _ in ()).throw(RuntimeError("interactive TTY required")),
    )
    with pytest.raises(RuntimeError, match="interactive TTY required"):
        calibrate_gello.main(["--config", str(PROFILE), "--arm", "left"])


def test_run_calibration_captures_and_writes_without_a_yam_robot(tmp_path):
    class Bus:
        def __init__(self):
            self.samples = iter(
                (
                    np.array([0, 1, 2, 3, 4, 5, 6], dtype=float),
                    np.array([0, 1, 2, 3, 4, 5, 7], dtype=float),
                    np.array([0, 1, 2, 3, 4, 5, 8], dtype=float),
                    np.array([0, 1, 2, 3, 4, 5, 8], dtype=float),
                )
            )

        def read_radians(self):
            return next(self.samples)

    class View:
        def __init__(self):
            self.keys = iter(("z", "o", "c", "w"))

        def update(self, obs, recording=False):
            return next(self.keys)

    output = tmp_path / "left.yaml"
    calibration = session()
    messages = []
    assert (
        calibrate_gello.run_calibration(
            Bus(),
            calibration,
            View(),
            rate_hz=100,
            output=output,
            max_steps=4,
            sleep_fn=lambda delay: None,
            time_fn=lambda: 1.0,
            emit=lambda *args, **kwargs: messages.append(args[0] if args else ""),
        )
        == 4
    )
    assert output.exists()
    assert "Wrote" in "\n".join(messages)
