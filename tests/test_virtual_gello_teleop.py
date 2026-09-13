from pathlib import Path

import numpy as np
import yaml

from egomimic.robot.virtual_gello_teleop import (
    VirtualYamFollower,
    real_leader_specs,
    run_demo,
)


def station_xml() -> Path:
    import i2rt

    return (
        Path(i2rt.__file__).resolve().parent
        / "robot_models/station/yam_station_linear_4310_d405/yam_station_linear_4310_d405.xml"
    )


def test_virtual_gello_runs_real_controller_against_dual_yam_model():
    robot = VirtualYamFollower(station_xml())
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
