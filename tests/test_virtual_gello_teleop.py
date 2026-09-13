from pathlib import Path

import numpy as np

from egomimic.robot.virtual_gello_teleop import VirtualYamFollower, run_demo


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
