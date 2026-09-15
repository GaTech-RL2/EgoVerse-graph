"""Offline behavior checks for the shared collection and rollout paths."""

import hashlib
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from egomimic.robot.collect_demo import EpisodeWriter, next_episode_path, run_collection
from egomimic.robot.interface import ARM_OFFSET, pose_matrix
from egomimic.robot.replay_policy import ZarrReplayPolicy
from egomimic.robot.rollout import load_policy, run_rollout
from egomimic.robot.teleop import TeleopControl, WorldFrameTeleop
from egomimic.robot.yam.interface import YamInterface


def mapper(arms=("left", "right"), **kwargs):
    return WorldFrameTeleop(
        arms,
        **dict(
            translation_gain=1.0,
            orientation_gain=1.0,
            target_lpf=1.0,
            max_controller_step_m=0.08,
            max_controller_step_deg=40.0,
            **kwargs,
        ),
    )


class FakeRobot:
    def __init__(self, arms=("left", "right")):
        self.arms = list(arms)
        self.q = np.zeros(14)
        self.q[[6, 13]] = 0.5
        self.camera_res = {
            "front_img_1": (2, 3),
            "left_wrist_img": (2, 3),
            "right_wrist_img": (2, 3),
        }
        self.recorders = {}
        self.commands = []

    def get_obs(self):
        return {
            "joint_positions": self.q.copy(),
            "ee_poses": self.q.copy(),
            **{
                key: np.tile(np.array([[[10, 20, 30]]], dtype=np.uint8), (2, 3, 1))
                for key in self.camera_res
            },
        }

    def set_joints(self, command, arm):
        self.commands.append((arm, command.copy()))
        self.q[ARM_OFFSET[arm] : ARM_OFFSET[arm] + 7] = command

    def solve_ik(self, pose, arm):
        return np.asarray(pose).copy()

    def forward_kinematics(self, q, arm):
        return np.asarray(q).copy()

    def set_home(self):
        self.q[:] = 0
        self.q[[6, 13]] = 1


class View:
    def __init__(self):
        self.closed = False
        self.frames = []

    def update(self, obs, recording=False):
        self.frames.append((obs, recording))

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)


def test_world_translation_does_not_follow_initial_controller_or_head_rotation():
    control = mapper(("right",))
    initial = np.eye(4)
    initial[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    robot_pose = np.eye(4)
    robot_pose[:3, :3] = Rotation.from_euler("x", 60, degrees=True).as_matrix()
    buttons = {"RG": True, "rightTrig": (0.25,)}
    control.update({"r": initial, "head": np.eye(4)}, buttons, {"right": robot_pose})
    moved = initial.copy()
    moved[2, 3] -= 0.02  # Quest forward becomes arm-base forward.
    changed_head = np.eye(4)
    changed_head[:3, 3] = 99
    target = control.update(
        {"r": moved, "head": changed_head}, buttons, {"right": robot_pose}
    )["right"]
    np.testing.assert_allclose(target[:3], [0.02, 0, 0], atol=1e-10)
    np.testing.assert_allclose(
        pose_matrix(target[:6])[:3, :3], robot_pose[:3, :3], atol=1e-10
    )
    assert target[6] == 0.75


@pytest.mark.parametrize(
    ("controller_delta", "robot_delta"),
    [
        ([0.01, 0.0, 0.0], [0.0, 0.01, 0.0]),
        ([0.0, 0.01, 0.0], [0.0, 0.0, 0.01]),
        ([0.0, 0.0, -0.01], [-0.01, 0.0, 0.0]),
    ],
)
def test_rl2_yaw_180_pins_xyz_basis(controller_delta, robot_delta):
    control = mapper(("right",), headset_yaw_degrees=180.0)
    initial = np.eye(4)
    buttons = {"RG": 1}
    control.update({"r": initial}, buttons, {"right": np.eye(4)})
    moved = initial.copy()
    moved[:3, 3] += controller_delta
    target = control.update({"r": moved}, buttons, {"right": np.eye(4)})["right"]
    np.testing.assert_allclose(target[:3], robot_delta, atol=1e-10)


@pytest.mark.parametrize(
    ("controller_axis", "robot_axis"),
    [
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
    ],
)
def test_rl2_yaw_180_pins_roll_pitch_yaw_basis(controller_axis, robot_axis):
    control = mapper(("right",), headset_yaw_degrees=180.0)
    initial = np.eye(4)
    buttons = {"RG": 1}
    control.update({"r": initial}, buttons, {"right": np.eye(4)})
    angle = np.deg2rad(5.0)
    moved = initial.copy()
    moved[:3, :3] = Rotation.from_rotvec(
        np.asarray(controller_axis) * angle
    ).as_matrix()
    target = control.update({"r": moved}, buttons, {"right": np.eye(4)})["right"]
    expected = Rotation.from_rotvec(np.asarray(robot_axis) * angle).as_matrix()
    np.testing.assert_allclose(pose_matrix(target[:6])[:3, :3], expected, atol=1e-10)


def test_world_rotation_uses_left_multiplication():
    control = mapper(("right",))
    initial = np.eye(4)
    initial[:3, :3] = Rotation.from_euler("x", 0.6).as_matrix()
    robot = np.eye(4)
    robot[:3, :3] = Rotation.from_euler("y", 0.4).as_matrix()
    control.update({"r": initial}, {"RG": 1}, {"right": robot})
    moved = initial.copy()
    moved[:3, :3] = Rotation.from_euler("y", 0.1).as_matrix() @ initial[:3, :3]
    target = control.update({"r": moved}, {"RG": 1}, {"right": robot})["right"]
    expected = Rotation.from_euler("z", 0.1).as_matrix() @ robot[:3, :3]
    np.testing.assert_allclose(pose_matrix(target[:6])[:3, :3], expected, atol=1e-10)


@pytest.mark.parametrize("loss", ["release", "missing", "jump", "invalid"])
def test_clutch_loss_reanchors_at_current_measured_pose(loss):
    control = mapper(("right",))
    first = np.eye(4)
    control.update({"r": first}, {"RG": 1}, {"right": first})
    moved = first.copy()
    moved[0, 3] = 0.04
    control.update({"r": moved}, {"RG": 1}, {"right": first})
    if loss == "release":
        assert not control.update({"r": moved}, {}, {"right": first})
    elif loss == "missing":
        assert not control.update({}, {"RG": 1}, {"right": first})
    elif loss == "jump":
        moved[0, 3] = 2
        assert not control.update({"r": moved}, {"RG": 1}, {"right": first})
    else:
        assert not control.update(
            {"r": np.full((4, 4), np.nan)}, {"RG": 1}, {"right": first}
        )
    current_robot = first.copy()
    current_robot[:3, 3] = [1, 2, 3]
    result = control.update({"r": moved}, {"RG": 1}, {"right": current_robot})["right"]
    np.testing.assert_allclose(result[:3], [1, 2, 3])


def test_arms_clutch_independently_and_joystick_reanchors():
    control = mapper()
    pose = np.eye(4)
    measured = {"left": pose.copy(), "right": pose.copy()}
    result = control.update({"l": pose, "r": pose}, {"LG": 1}, measured)
    assert set(result) == {"left"}
    measured["left"][0, 3] = 4
    result = control.update({"l": pose}, {"LG": 1, "LJ": 1}, measured)
    assert result["left"][0] == 4


def test_teleop_holds_on_ik_failure_and_on_lost_tracking():
    robot = FakeRobot(("right",))
    control = TeleopControl(robot, mapper(("right",)), 30, 1)
    control.step({"r": np.eye(4)}, {"RG": 1, "rightTrig": 0.5}, robot.get_obs())
    before = robot.q.copy()
    robot.solve_ik = lambda *args: None
    control.step({"r": np.eye(4)}, {"RG": 1, "rightTrig": 1}, robot.get_obs())
    control.step({}, {}, robot.get_obs())
    np.testing.assert_allclose(robot.q, before)


def test_hdf5_schema_rgb_alignment_and_exclusive_creation(tmp_path):
    robot = FakeRobot()
    obs = robot.get_obs()
    writer = EpisodeWriter(tmp_path / "demo_0.hdf5", robot.camera_res)
    writer.append(obs, np.ones(14), np.full(14, 2.0))
    invalid = {**obs, "left_wrist_img": None}
    with pytest.raises(ValueError, match="camera"):
        writer.append(invalid, np.ones(14), np.ones(14))
    writer.close()
    with h5py.File(writer.path) as data:
        assert not data.attrs["sim"] and data.attrs["complete"]
        for key in (
            "observations/joints",
            "observations/joint_positions",
            "observations/eepose",
            "actions/eepose",
            "actions/joints",
            "action",
        ):
            assert data[key].shape == (1, 14)
            assert data[key].dtype == np.float32
        for camera in robot.camera_res:
            frame = data[f"observations/images/{camera}"]
            assert frame.shape == (1, 2, 3, 3)
            np.testing.assert_array_equal(frame[0, 0, 0], [30, 20, 10])
        np.testing.assert_array_equal(data["action"], data["actions/joints"])
        np.testing.assert_array_equal(data["observations/eepose"][0], obs["ee_poses"])
    with pytest.raises(FileExistsError):
        EpisodeWriter(writer.path, robot.camera_res)
    assert next_episode_path(tmp_path).name == "demo_1.hdf5"


def test_collection_button_edges_and_interrupted_data_are_preserved(tmp_path):
    config = OmegaConf.to_container(
        OmegaConf.load("egomimic/hydra_configs/robot/eva_collect.yaml")
    )
    config["recording"]["directory"] = str(tmp_path)
    robot, view = FakeRobot(), View()
    buttons = iter([{"B": 1}, {"B": 1}, {}, {"X": 1}, {"B": 1}, {"A": 1}])
    reader = SimpleNamespace(
        get_transformations_and_buttons=lambda: (
            {"l": np.eye(4), "r": np.eye(4)},
            next(buttons),
        )
    )
    run_collection(robot, reader, config, view=view, max_steps=10)
    paths = sorted(tmp_path.glob("*.hdf5"))
    assert len(paths) == 2 and view.closed
    with h5py.File(paths[0]) as first, h5py.File(paths[1]) as second:
        assert first["action"].shape[0] == 3
        assert second["action"].shape[0] == 1
        assert not first.attrs["complete"] and not second.attrs["complete"]
    assert len(view.frames) == 6


def test_collection_can_record_30hz_while_teleop_runs_at_60hz(tmp_path):
    config = OmegaConf.to_container(
        OmegaConf.load("egomimic/hydra_configs/robot/eva_collect.yaml")
    )
    config["frequency"] = 60
    config["recording"].update(directory=str(tmp_path), rate_hz=30)
    robot, view = FakeRobot(), View()
    buttons = iter([{"B": 1}, {}, {}, {}, {}, {"A": 1}])
    reader = SimpleNamespace(
        get_transformations_and_buttons=lambda: (
            {"l": np.eye(4), "r": np.eye(4)},
            next(buttons),
        )
    )
    run_collection(robot, reader, config, view=view, max_steps=10)
    paths = list(tmp_path.glob("*.hdf5"))
    assert len(paths) == 1
    with h5py.File(paths[0]) as episode:
        assert episode["action"].shape[0] == 3
        assert not episode.attrs["complete"]


class Driver:
    xml_path = "test-model.xml"

    def __init__(self, channel):
        self.channel, self.q, self.closed = (
            channel,
            np.array([0, 0, 0, 0, 0, 0, 0.5]),
            False,
        )

    def get_joint_pos(self):
        return self.q.copy()

    def command_joint_pos(self, q):
        self.q = q.copy()

    def close(self):
        self.closed = True


class ReportingDriver(Driver):
    def get_robot_info(self):
        return {
            "kp": np.r_[np.zeros(6), 10.0],
            "kd": np.r_[np.zeros(6), 0.5],
            "gripper_index": 6,
            "limit_gripper_effort": 50.0,
        }


class Solver:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fk(self, joints):
        return pose_matrix(joints)

    def ik(self, pose, seed):
        from egomimic.robot.interface import pose_vector

        return pose_vector(pose)


class StreamingSolver:
    instances = []

    def __init__(self, xml_path, **kwargs):
        self.xml_path = xml_path
        self.dt = float(kwargs["dt"])
        self.max_joint_vel = float(kwargs["max_joint_vel"])
        self.n_arm = int(kwargs["n_arm"])
        self.calls = []
        self.instances.append(self)

    def fk(self, joints):
        return pose_matrix(np.asarray(joints)[:6])

    def ik(self, target, seed):
        from egomimic.robot.interface import pose_vector

        self.calls.append((np.asarray(target).copy(), np.asarray(seed).copy()))
        return True, pose_vector(target)


def yam_robot(arms=("left", "right"), driver_factory=Driver, streaming=False):
    teleop_kinematics = None
    streaming_solver_factory = None
    if streaming:
        teleop_kinematics = {
            "ee_site": "tcp_site",
            "n_arm": 6,
            "dt": 1 / 60,
            "steps": 4,
            "gain": 0.5,
            "damping": 0.01,
            "posture_cost": 0.08,
            "max_joint_vel": 2.5,
        }
        streaming_solver_factory = StreamingSolver
    return YamInterface(
        arms,
        {"left": "can0", "right": "can1"},
        {},
        {},
        {arm: [0, 0, 0, 0, 0, 0, 1] for arm in arms},
        teleop_kinematics=teleop_kinematics,
        driver_factory=driver_factory,
        solver_factory=Solver,
        streaming_solver_factory=streaming_solver_factory,
    )


def test_yam_interface_matches_shared_observation_order_and_cleanup():
    robot = yam_robot()
    drivers = list(robot.controller.values())
    robot.set_pose(np.array([0.1, 0.2, 0.3, 0, 0, 0, 0.8]), "right")
    obs = robot.get_obs()
    np.testing.assert_allclose(
        obs["joint_positions"][7:], [0.1, 0.2, 0.3, 0, 0, 0, 0.8]
    )
    np.testing.assert_allclose(obs["ee_poses"], obs["joint_positions"])
    assert drivers[0].channel == "can0" and drivers[1].channel == "can1"
    robot.close()
    robot.close()
    assert all(driver.closed for driver in drivers)


def test_yam_interface_verifies_reported_gripper_control(capsys):
    robot = YamInterface(
        ["left"],
        {"left": "can0"},
        {},
        {},
        {"left": [0, 0, 0, 0, 0, 0, 1]},
        gripper_kp=10,
        gripper_kd=0.5,
        gripper_force_limit=50,
        driver_factory=ReportingDriver,
        solver_factory=Solver,
    )
    assert "gripper Kp=10, Kd=0.5, force limit=50 N" in capsys.readouterr().out
    robot.close()


def test_yam_interface_rejects_driver_gripper_control_mismatch():
    with pytest.raises(RuntimeError, match="gripper control mismatch"):
        YamInterface(
            ["left"],
            {"left": "can0"},
            {},
            {},
            {"left": [0, 0, 0, 0, 0, 0, 1]},
            gripper_kp=8,
            gripper_kd=0.5,
            gripper_force_limit=50,
            driver_factory=ReportingDriver,
            solver_factory=Solver,
        )


def test_yam_teleop_uses_reference_streaming_ik_with_command_seed():
    StreamingSolver.instances.clear()
    robot = yam_robot(("right",), streaming=True)
    control = TeleopControl(robot, mapper(("right",)), 60, 2.5)
    initial = np.eye(4)
    buttons = {"RG": 1, "rightTrig": (0.25,)}
    control.step({"r": initial}, buttons, robot.get_obs())
    moved = initial.copy()
    moved[2, 3] -= 0.01
    commands, _ = control.step({"r": moved}, buttons, robot.get_obs())
    solver = StreamingSolver.instances[0]
    assert len(solver.calls) == 2
    np.testing.assert_allclose(solver.calls[-1][1], np.r_[np.zeros(6), 0.75])
    assert commands[7] == pytest.approx(0.01)
    assert commands[13] == pytest.approx(0.75)
    robot.close()


def test_yam_teleop_rejects_rate_drift_from_reference_solver():
    robot = yam_robot(("right",), streaming=True)
    with pytest.raises(ValueError, match="frequency"):
        TeleopControl(robot, mapper(("right",)), 30, 2.5)
    with pytest.raises(ValueError, match="joint velocity"):
        TeleopControl(robot, mapper(("right",)), 60, 1.0)
    robot.close()


def test_yam_failed_initialization_closes_already_opened_arm():
    first = Driver("can0")

    def factory(channel):
        if channel == "can1":
            raise RuntimeError("CAN unavailable")
        return first

    with pytest.raises(RuntimeError, match="CAN unavailable"):
        yam_robot(driver_factory=factory)
    assert first.closed


def test_camera_serial_preflight_rejects_bad_station_mapping():
    from egomimic.robot.cameras import validate_camera_devices

    cameras = {
        "front": {"enabled": True, "type": "realsense", "serial_number": "top"},
        "left": {"enabled": True, "type": "d405", "serial_number": "left"},
    }
    assert validate_camera_devices(cameras, ["left", "top"]) == ("top", "left")
    with pytest.raises(RuntimeError, match="front=top"):
        validate_camera_devices(cameras, ["left"])
    cameras["left"]["serial_number"] = "top"
    with pytest.raises(ValueError, match="distinct serials"):
        validate_camera_devices(cameras, ["top"])


def test_yam_camera_preflight_runs_before_driver_initialization():
    calls = []

    def reject(_config):
        raise RuntimeError("camera preflight failed")

    with pytest.raises(RuntimeError, match="camera preflight failed"):
        YamInterface(
            ["left"],
            {"left": "can0"},
            {"front": {"type": "realsense", "serial_number": "missing"}},
            {},
            {"left": [0, 0, 0, 0, 0, 0, 1]},
            driver_factory=lambda channel: calls.append(channel),
            solver_factory=Solver,
            camera_validator=reject,
        )
    assert not calls


def replay_store(tmp_path, padded=5, total=3):
    import zarr

    store = zarr.open_group(str(tmp_path / "demo.zarr"), mode="w")
    store.attrs["total_frames"] = total
    for side in ("left", "right"):
        store.create_array(f"{side}.cmd_joints", data=np.zeros((padded, 6)))
        store.create_array(f"{side}.cmd_gripper", data=np.full((padded, 1), 0.5))
    return store


def replay_keys():
    return {
        arm: {"joints": f"{arm}.cmd_joints", "gripper": f"{arm}.cmd_gripper"}
        for arm in ARM_OFFSET
    }


@pytest.mark.parametrize("robot_factory", [FakeRobot, yam_robot])
def test_rollout_replays_all_valid_zarr_frames_and_stops_at_eof(
    tmp_path, robot_factory
):
    replay_store(tmp_path)
    policy = ZarrReplayPolicy(tmp_path / "demo.zarr", keys=replay_keys(), chunk_size=2)
    config = dict(
        frequency=30,
        max_steps=50,
        execute_steps=1,
        max_joint_velocity=1,
        preview={"enabled": False},
    )
    view = View()
    assert run_rollout(robot_factory(), policy, config, view) == 3
    assert policy.cursor == 3 and view.closed


def test_rollout_validates_both_arms_before_commanding():
    robot = FakeRobot()
    bad = robot.q.copy()
    bad[7] = 100
    policy = SimpleNamespace(action_type="joints", predict=lambda obs: bad[None])
    with pytest.raises(ValueError, match="joint velocity"):
        run_rollout(
            robot,
            policy,
            dict(
                frequency=30,
                max_steps=2,
                execute_steps=1,
                max_joint_velocity=1,
                preview={"enabled": False},
            ),
        )
    assert not robot.commands


def test_replay_rejects_nan_and_ignores_chunk_padding(tmp_path):
    store = replay_store(tmp_path)
    store["right.cmd_joints"][3:] = np.nan
    policy = ZarrReplayPolicy(tmp_path / "demo.zarr", keys=replay_keys(), chunk_size=8)
    assert policy.predict(FakeRobot().get_obs()).shape == (3, 14)
    with pytest.raises(StopIteration):
        policy.predict({})
    store["right.cmd_joints"][1] = np.nan
    policy = ZarrReplayPolicy(tmp_path / "demo.zarr", keys=replay_keys(), chunk_size=8)
    with pytest.raises(ValueError, match="finite"):
        policy.predict(FakeRobot().get_obs())


def test_no_non_graph_inference_backend():
    with pytest.raises(ValueError, match="kind=graph"):
        load_policy({"kind": "legacy"})


def test_yam_uploader_lists_hdf5_without_touching_source_files(tmp_path):
    from egomimic.scripts.data_upload.yam_uploader import collect_files

    (tmp_path / "demo_1.hdf5").write_bytes(b"data")
    (tmp_path / "metadata.json").write_text("{}")
    assert collect_files(tmp_path) == [tmp_path / "demo_1.hdf5"]
    assert (tmp_path / "demo_1.hdf5").read_bytes() == b"data"


def test_world_reader_drops_stale_input_and_uses_rail_world_stream(monkeypatch):
    # Import the bundled reader, with only adb itself stubbed if unavailable.
    monkeypatch.syspath_prepend(str(Path("egomimic/robot/oculus_reader").resolve()))
    monkeypatch.setitem(sys.modules, "ppadb", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "ppadb.client", SimpleNamespace(Client=object))
    from oculus_reader.reader import OculusReader

    reader = OculusReader.__new__(OculusReader)
    reader.running = False
    reader._lock = threading.Lock()
    reader.tag = "wE9ryARX"
    reader.max_age, reader.last_update = 0.1, time.monotonic() - 1
    reader.last_transforms, reader.last_buttons = {"r": np.eye(4)}, {"RG": 1}
    assert reader.get_transformations_and_buttons() == ({}, {})
    reader.last_update = time.monotonic()
    assert "r" in reader.get_transformations_and_buttons()[0]
    assert reader.extract_data("wE9ryARXWorld: alternate tag") == ""
    assert reader.extract_data("wE9ryARX: RAIL world frame") == "RAIL world frame"

    monkeypatch.setattr(OculusReader, "get_device", lambda self: object())
    monkeypatch.setattr(OculusReader, "install", lambda self, **kwargs: None)
    configured = OculusReader(run=False)
    assert configured.pose_frame == "world"
    assert configured.tag == "wE9ryARX"
    with pytest.raises(ValueError, match="emits world poses only"):
        OculusReader(run=False, pose_frame="head")


def test_bundled_rail_world_apk_and_source_match_pinned_provenance():
    root = Path("egomimic/robot/oculus_reader")
    source = (root / "app_source/Src/OculusTeleop.cpp").read_text()
    assert "handPoseMatrixHeadCoord" not in source
    assert (
        "handPoseTransformations.push_back(std::make_pair(side, handPoseMatrix));"
        in source
    )
    assert '"wE9ryARX"' in source

    apk = (root / "oculus_reader/APK/teleop-debug.apk").read_bytes()
    expected = "cc990542fb539d541b0927a7dcce7ede4c3e6c3cb53990ad42c305bb42a05876"
    provenance = (root / "oculus_reader/APK/PROVENANCE.md").read_text()
    assert expected in provenance
    assert "`7496215` bytes" in provenance
    if apk.startswith(b"version https://git-lfs.github.com/spec/v1"):
        assert f"oid sha256:{expected}".encode() in apk
        assert b"size 7496215" in apk
    else:
        assert len(apk) == 7496215
        assert hashlib.sha256(apk).hexdigest() == expected


def test_camera_stream_pauses_disconnected_frames_and_resumes():
    from egomimic.robot.cameras import CameraStream

    image = np.zeros((2, 3, 3), dtype=np.uint8)
    recorder = SimpleNamespace(
        last_frame_time=time.monotonic() - 3, get_image=lambda: image
    )
    stream = CameraStream(recorder, max_age=1.0)
    assert stream.get_image() is None
    recorder.last_frame_time = time.monotonic()
    assert stream.get_image() is image


def test_live_view_displays_all_front_and_wrist_cameras(monkeypatch):
    import cv2

    from egomimic.robot.cameras import CameraView

    seen = []
    monkeypatch.setattr(
        cv2, "imshow", lambda name, image: seen.append((name, image.copy()))
    )
    monkeypatch.setattr(cv2, "waitKey", lambda _: ord("q"))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    robot = FakeRobot()
    view = CameraView(robot.camera_res)
    assert view.update(robot.get_obs(), recording=True) == "q"
    assert {name for name, _ in seen} == set(robot.camera_res)
    view.close()


def test_reclutch_control_anchors_to_measured_joints_after_manual_movement():
    robot = FakeRobot(("right",))
    control = TeleopControl(robot, mapper(("right",)), 30, 1)
    pose = np.eye(4)
    control.step({"r": pose}, {"RG": 1}, robot.get_obs())
    control.step({"r": pose}, {}, robot.get_obs())
    robot.q[7] = 1  # Operator moved the released arm.
    control.step({"r": pose}, {"RG": 1}, robot.get_obs())
    assert robot.q[7] == 1


def test_existing_eva_interface_still_records_and_accepts_normalized_gripper(
    monkeypatch, tmp_path
):
    from egomimic.robot.eva.eva_ws.src.eva import robot_interface as eva

    robot = eva.ARXInterface.__new__(eva.ARXInterface)
    robot.arms = ["right"]
    robot.recorders, robot.camera_res = {}, {}
    robot.gripper_close, robot.gripper_width = {"right": -0.01}, {"right": 0.1}
    robot.ts_offset = 0.2
    q = np.array([0.1, 0.2, 0.3, 0, 0, 0])
    received = []
    state = SimpleNamespace(timestamp=1.0, pos=lambda: q.copy(), gripper_pos=0.04)
    robot.controller = {
        "right": SimpleNamespace(
            get_joint_state=lambda: state, set_joint_cmd=received.append
        )
    }
    robot.kinematics_solver = SimpleNamespace(
        fk=lambda joints: (np.asarray(joints)[:3], Rotation.identity())
    )
    monkeypatch.setattr(eva, "ArxJointState", lambda *args: SimpleNamespace())
    obs = robot.get_obs()
    np.testing.assert_allclose(obs["joint_positions"][:7], 0)
    np.testing.assert_allclose(obs["ee_poses"][7:], [0.1, 0.2, 0.3, 0, 0, 0, 0.5])
    robot.set_joints(np.r_[q, 0.8], "right")
    assert received[0].gripper_pos == pytest.approx(0.07)
    writer = EpisodeWriter(tmp_path / "demo_0.hdf5", {})
    writer.append(obs, obs["joint_positions"], obs["ee_poses"])
    writer.close()
    with h5py.File(writer.path) as data:
        assert "observations/images" in data
    robot.close()


def test_yam_cleanup_attempts_both_arms_if_one_driver_fails():
    robot = yam_robot()
    second = robot.controller["right"]

    def fail():
        raise RuntimeError("driver shutdown")

    robot.controller["left"].close = fail
    with pytest.raises(ExceptionGroup, match="Yam cleanup"):
        robot.close()
    assert second.closed


def test_quest_stop_unblocks_a_pending_socket_read(monkeypatch):
    import socket

    monkeypatch.syspath_prepend(str(Path("egomimic/robot/oculus_reader").resolve()))
    monkeypatch.setitem(sys.modules, "ppadb", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "ppadb.client", SimpleNamespace(Client=object))
    from oculus_reader.reader import OculusReader

    local, remote = socket.socketpair()
    reader = OculusReader.__new__(OculusReader)
    reader.running, reader._lock = True, threading.Lock()
    reader.print_FPS = False
    reader.tag = "wE9ryARX"
    ready = threading.Event()

    def makefile():
        file = local.makefile()
        ready.set()
        return file

    connection = SimpleNamespace(
        socket=SimpleNamespace(makefile=makefile, shutdown=local.shutdown),
        close=local.close,
    )
    reader._connection = connection
    reader.thread = threading.Thread(
        target=reader.read_logcat_by_line, args=(connection,), daemon=True
    )
    reader.thread.start()
    assert ready.wait(timeout=1.0)
    reader.stop()
    remote.close()
    assert not reader.thread.is_alive()
