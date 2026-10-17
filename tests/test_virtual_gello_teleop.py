from pathlib import Path

import numpy as np
import pytest
import yaml

from egomimic.robot.virtual_gello_teleop import (
    VirtualYamFollower,
    default_model_path,
    profile_home,
    real_leader_specs,
    run_demo,
)


def station_xml(tmp_path: Path) -> Path:
    """A minimal named dual-arm model keeps CPU checks independent of i2rt."""
    arms = []
    for arm, y in (("left", 0), ("right", 0.7)):
        links = [f'<body name="{arm}_base" pos="0 {y} 0">']
        for index in range(1, 7):
            links.append(
                f'<body pos="0 0 0.1"><joint name="{arm}_joint{index}" '
                'type="hinge" axis="0 0 1" range="-3.2 3.2"/>'
                '<geom type="sphere" size="0.02" mass="0.1"/>'
            )
        links.append(f'<body name="{arm}_gripper"><geom size="0.02" mass="0.1"/>')
        for index in (7, 8):
            links.append(
                f'<body><joint name="{arm}_joint{index}" type="slide" '
                'axis="0 1 0" range="0 0.02"/>'
                '<geom size="0.005" mass="0.01"/></body>'
            )
        links.append("</body>" * 8)
        arms.append("".join(links))
    path = tmp_path / "dual_arm.xml"
    path.write_text(
        '<mujoco><compiler angle="radian" autolimits="true"/><worldbody>'
        + "".join(arms)
        + "</worldbody></mujoco>"
    )
    return path


def test_virtual_gello_runs_shared_controller_against_dual_arm_model(tmp_path):
    robot = VirtualYamFollower(station_xml(tmp_path))
    start = robot.get_obs()["joint_positions"]
    command = run_demo(robot, frequency=60, steps=120)
    assert command.shape == (14,)
    assert np.isfinite(command).all()
    assert not np.allclose(command, start)
    assert np.all((robot.q[[6, 13]] >= 0) & (robot.q[[6, 13]] <= 1))


def test_real_leader_specs_overlay_exported_calibrations(tmp_path):
    config = {
        "gello": {
            "bus": {"baudrate": 57600},
            "input_filter_alpha": 0.9,
            "leaders": {"left": {"port": "L"}, "right": {"port": "R"}},
        }
    }
    left = {"gello": {"leaders": {"left": {"joint_offsets_rad": [1] * 6}}}}
    right = {"gello": {"leaders": {"right": {"joint_offsets_rad": [2] * 6}}}}
    paths = []
    for name, value in (("config", config), ("left", left), ("right", right)):
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(value))
        paths.append(path)
    specs = real_leader_specs(*paths)
    assert specs["left"]["joint_offsets_rad"] == [1] * 6
    assert specs["right"]["joint_offsets_rad"] == [2] * 6
    assert specs["left"]["filter_alpha"] == 0.9
    assert specs["right"]["filter_alpha"] == 0.9


def test_profile_home_uses_configured_follower_pose(tmp_path):
    path = tmp_path / "profile.yaml"
    path.write_text(
        yaml.safe_dump(
            {"robot": {"home": {"left": [1] * 6 + [1.0], "right": [2] * 6 + [1.0]}}}
        )
    )
    home = profile_home(path)
    assert np.array_equal(home["left"], np.ones(7))
    assert np.array_equal(home["right"], np.r_[np.full(6, 2.0), 1.0])


def test_virtual_absolute_mapping_preserves_home_and_full_turn_error(tmp_path):
    from egomimic.robot.yam.gello import GelloSample, GelloTeleopControl

    robot = VirtualYamFollower(station_xml(tmp_path))
    robot.home = {
        "left": np.r_[np.full(6, 0.01), 1.0],
        "right": np.r_[np.full(6, -0.02), 1.0],
    }
    leaders = {
        "left": np.r_[np.full(6, 0.1), 0.25],
        "right": np.r_[np.full(5, -0.1), -2 * np.pi + 0.04, 0.75],
    }
    controller = GelloTeleopControl(
        robot,
        frequency=60,
        alignment="absolute",
        max_joint_velocity=10000,
        max_alignment_velocity=10000,
    )
    command = controller.step(
        GelloSample(joints=leaders, timestamps={"left": 0, "right": 0}),
        robot.get_obs(),
        toggle_arms=("left", "right"),
        now=0,
    )
    for arm, offset in (("left", 0), ("right", 7)):
        np.testing.assert_allclose(
            command.joints[offset : offset + 6],
            robot.home[arm][:6] + leaders[arm][:6],
        )
        assert command.joints[offset + 6] == leaders[arm][6]
    # An erroneous full turn must remain visible in absolute calibration checks.
    assert command.joints[12] < -6


def test_installed_yam_model_when_optional_sdk_is_available():
    pytest.importorskip("i2rt", reason="Requires the optional Yam SDK model assets")
    robot = VirtualYamFollower(default_model_path())
    assert robot.get_obs()["joint_positions"].shape == (14,)
