"""Hardware-free checks for the USB/Dynamixel GELLO teleop path."""

import copy
import time
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import yaml

from egomimic.robot import collect_gello
from egomimic.robot.interface import ARM_OFFSET
from egomimic.robot.teleop_dashboard import delete_demo, list_demos
from egomimic.robot.yam.gello import (
    BimanualGelloReader,
    DynamixelGelloLeader,
    DynamixelPositionBus,
    GelloInputError,
    GelloSample,
    GelloTeleopControl,
    list_serial_ports,
    validate_can_interfaces,
    validate_serial_ports,
)

PROFILE = Path("egomimic/hydra_configs/robot/yam_rl2_gello_collect.yaml")


class FakeFollower:
    def __init__(self, arms=("left", "right")):
        self.arms = list(arms)
        self.q = np.zeros(14)
        self.q[[6, 13]] = 0.5
        self.commands = []
        self.camera_res = {}
        self.homed = False
        self.home = {arm: np.zeros(7) for arm in self.arms}
        for arm in self.arms:
            self.home[arm][6] = 0.5

    def get_obs(self):
        return {"joint_positions": self.q.copy(), "ee_poses": self.q.copy()}

    def set_joints(self, command, arm):
        command = np.asarray(command).copy()
        self.commands.append((arm, command))
        offset = ARM_OFFSET[arm]
        self.q[offset : offset + 7] = command

    def forward_kinematics(self, joints, arm):
        return np.asarray(joints).copy()

    def set_home(self):
        self.homed = True


def leader_sample(*, left=None, right=None, timestamp=1.0):
    joints = {}
    if left is not None:
        joints["left"] = np.asarray(left, dtype=float)
    if right is not None:
        joints["right"] = np.asarray(right, dtype=float)
    return GelloSample(joints, {arm: timestamp for arm in joints})


def control(robot, **kwargs):
    return GelloTeleopControl(
        robot,
        frequency=10,
        max_input_age=0.25,
        max_joint_velocity=1.0,
        max_alignment_velocity=0.5,
        **kwargs,
    )


class FakePortHandler:
    def __init__(self, port):
        self.port = port
        self.baudrate = None
        self.closed = False

    def openPort(self):
        return True

    def setBaudRate(self, baudrate):
        self.baudrate = baudrate
        return True

    def closePort(self):
        self.closed = True


class FakePacketHandler:
    def __init__(self, protocol):
        self.protocol = protocol
        self.torque_writes = []

    def write1ByteTxRx(self, port, servo_id, address, value):
        self.torque_writes.append((servo_id, address, value))
        return 0, 0

    def getTxRxResult(self, result):
        return f"error {result}"


class FakeGroupRead:
    def __init__(self, port, packet, address, length, values):
        self.address, self.length, self.values = address, length, values
        self.ids = []

    def addParam(self, servo_id):
        self.ids.append(servo_id)
        return True

    def txRxPacket(self):
        return 0

    def isAvailable(self, servo_id, address, length):
        return servo_id in self.values and (address, length) == (
            self.address,
            self.length,
        )

    def getData(self, servo_id, address, length):
        return self.values[servo_id]


def fake_sdk(values):
    sdk = SimpleNamespace(COMM_SUCCESS=0)
    sdk.port = None
    sdk.packet = None
    sdk.group = None

    def port_handler(port):
        sdk.port = FakePortHandler(port)
        return sdk.port

    def packet_handler(protocol):
        sdk.packet = FakePacketHandler(protocol)
        return sdk.packet

    def group_read(port, packet, address, length):
        sdk.group = FakeGroupRead(port, packet, address, length, values)
        return sdk.group

    sdk.PortHandler = port_handler
    sdk.PacketHandler = packet_handler
    sdk.GroupSyncRead = group_read
    return sdk


def test_dynamixel_bus_uses_present_position_132_and_disables_all_torque():
    sdk = fake_sdk({1: 0, 2: 2048, 7: 0xFFFFFFFF})
    bus = DynamixelPositionBus("/dev/fake", (1, 2, 7), sdk=sdk)
    assert sdk.group.address == 132 and sdk.group.length == 4
    assert sdk.packet.torque_writes == [(1, 64, 0), (2, 64, 0), (7, 64, 0)]
    np.testing.assert_allclose(bus.read_radians(), [0, np.pi, -2 * np.pi / 4096])
    bus.close()
    bus.close()
    assert sdk.port.closed
    with pytest.raises(GelloInputError, match="closed"):
        bus.read_radians()


def test_dynamixel_leader_maps_sign_offset_and_opening_without_commands():
    requested = []
    bus = SimpleNamespace(
        read_radians=lambda: np.array([1, 2, 3, 4, 5, 6, 0.75]),
        close=lambda: None,
    )

    def factory(**kwargs):
        requested.append(kwargs)
        return bus

    leader = DynamixelGelloLeader(
        port="/dev/serial/by-id/test",
        joint_ids=[1, 2, 3, 4, 5, 6],
        gripper_id=7,
        joint_signs=[-1, 1, 1, 1, 1, 1],
        joint_offsets_rad=[0.5, 1, 1, 1, 1, 1],
        gripper_open_rad=1.0,
        gripper_closed_rad=0.0,
        bus_factory=factory,
        time_fn=lambda: 12.5,
    )
    joints, timestamp = leader.read()
    np.testing.assert_allclose(joints, [-0.5, 1, 2, 3, 4, 5, 0.75])
    assert timestamp == 12.5
    assert requested[0]["ids"] == (1, 2, 3, 4, 5, 6, 7)
    leader.close()
    with pytest.raises(GelloInputError, match="closed"):
        leader.read()


def test_bimanual_reader_closes_first_leader_if_second_open_fails():
    opened = []

    class Leader:
        def __init__(self, **kwargs):
            if kwargs["port"] == "right":
                raise RuntimeError("right leader unavailable")
            self.closed = False
            opened.append(self)

        def close(self):
            self.closed = True

    with pytest.raises(RuntimeError, match="right leader unavailable"):
        BimanualGelloReader(
            ("left", "right"),
            {"left": {"port": "left"}, "right": {"port": "right"}},
            leader_factory=Leader,
        )
    assert len(opened) == 1 and opened[0].closed


def test_bimanual_reader_reads_independent_serial_buses_concurrently():
    class Leader:
        def __init__(self, port, **kwargs):
            self.value = 1 if port == "left" else 2

        def read(self):
            return np.r_[np.full(6, self.value), 0.5], 12.5

        def close(self):
            pass

    reader = BimanualGelloReader(
        ("left", "right"),
        {"left": {"port": "left"}, "right": {"port": "right"}},
        leader_factory=Leader,
    )
    deadline = time.monotonic() + 1.0
    while True:
        try:
            sample = reader.read()
            break
        except GelloInputError as error:
            assert "Waiting for first GELLO sample" in str(error)
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.001)
    assert sample.timestamps == {"left": 12.5, "right": 12.5}
    np.testing.assert_allclose(sample.joints["right"][:6], 2)
    reader.close()


def test_bimanual_reader_returns_cached_samples_while_serial_reads_continue():
    release = Event()
    reading_again = {"left": Event(), "right": Event()}

    class Leader:
        def __init__(self, port, **kwargs):
            self.arm = port
            self.calls = 0

        def read(self):
            self.calls += 1
            if self.calls > 1:
                reading_again[self.arm].set()
                release.wait(1.0)
            value = 1 if self.arm == "left" else 2
            return np.r_[np.full(6, value), 0.5], 12.5

        def close(self):
            pass

    reader = BimanualGelloReader(
        ("left", "right"),
        {"left": {"port": "left"}, "right": {"port": "right"}},
        leader_factory=Leader,
    )
    try:
        assert all(event.wait(1.0) for event in reading_again.values())
        started = time.monotonic()
        sample = reader.read()
        assert time.monotonic() - started < 0.02
        np.testing.assert_allclose(sample.joints["left"][:6], 1)
        np.testing.assert_allclose(sample.joints["right"][:6], 2)
    finally:
        release.set()
        reader.close()


def test_serial_listing_and_preflight_open_no_devices(tmp_path):
    device_a, device_b = tmp_path / "ttyA", tmp_path / "ttyB"
    device_a.touch()
    device_b.touch()
    by_id = tmp_path / "by-id"
    by_id.mkdir()
    left, right = by_id / "left", by_id / "right"
    left.symlink_to(device_a)
    right.symlink_to(device_b)
    assert list_serial_ports(by_id) == (str(left), str(right))
    assert validate_serial_ports((str(left), str(right))) == (str(left), str(right))
    with pytest.raises(RuntimeError, match="unavailable"):
        validate_serial_ports((str(left), str(by_id / "missing")))
    alias = by_id / "alias"
    alias.symlink_to(device_a)
    with pytest.raises(ValueError, match="same device"):
        validate_serial_ports((str(left), str(alias)))


def test_can_preflight_is_read_only_and_checks_both_followers():
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        name = command[-1]
        return SimpleNamespace(
            returncode=0,
            stdout=f"7: {name}: <NOARP,UP,LOWER_UP> mtu 16\n    link/can",
            stderr="",
        )

    channels = ("can_follower_l", "can_follower_r")
    assert validate_can_interfaces(channels, runner=runner) == channels
    assert all(
        call[0][:5] == ["ip", "-details", "link", "show", "dev"] for call in calls
    )
    assert all(call[1]["check"] is False for call in calls)

    def down(command, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=f"7: {command[-1]}: <NOARP> mtu 16\n    link/can",
            stderr="",
        )

    with pytest.raises(RuntimeError, match=r"can_follower_l \(down\)"):
        validate_can_interfaces(("can_follower_l",), runner=down)


def test_relative_activation_offsets_arm_joints_but_maps_gripper_directly():
    robot = FakeFollower(("right",))
    robot.q[7:14] = [0.4, -0.3, 0.2, 0.1, 0, -0.2, 0.5]
    gello = control(robot)
    leader = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.2])

    result = gello.step(
        leader_sample(right=leader),
        robot.get_obs(),
        toggle_arms=("right",),
        now=1.0,
    )
    assert result.active_arms == {"right"}
    np.testing.assert_allclose(
        robot.commands[-1][1], [0.4, -0.3, 0.2, 0.1, 0, -0.2, 0.2]
    )

    moved = leader.copy()
    moved[0] += 0.5
    moved[6] = 0.0
    result = gello.step(leader_sample(right=moved), robot.get_obs(), now=1.0)
    assert result.joints[7] == pytest.approx(0.5)
    assert result.joints[13] == pytest.approx(0.0)


def test_arms_toggle_independently_and_start_disarmed():
    robot = FakeFollower()
    gello = control(robot)
    joints = np.r_[np.zeros(6), 0.5]
    result = gello.step(
        leader_sample(left=joints, right=joints), robot.get_obs(), now=1.0
    )
    assert not result.active_arms and not robot.commands
    result = gello.step(
        leader_sample(left=joints, right=joints),
        robot.get_obs(),
        toggle_arms=("left",),
        now=1.0,
    )
    assert result.active_arms == {"left"}
    assert [arm for arm, _ in robot.commands] == ["left"]
    before = len(robot.commands)
    result = gello.step(
        leader_sample(left=joints, right=joints),
        robot.get_obs(),
        toggle_arms=("left",),
        now=1.0,
    )
    assert not result.active_arms and len(robot.commands) == before


def test_bimanual_arming_requires_a_fresh_simultaneous_trigger_squeeze():
    robot = FakeFollower()
    gello = control(robot, activation_gripper_threshold=0.9)
    released = np.r_[np.zeros(6), 1.0]
    pressed = np.r_[np.zeros(6), 0.8]

    result = gello.step(
        leader_sample(left=released, right=released),
        robot.get_obs(),
        toggle_bimanual_arming=True,
        now=1.0,
    )
    assert result.phase == "armed: press both GELLO triggers"
    assert not result.active_arms and not robot.commands

    result = gello.step(
        leader_sample(left=pressed, right=released), robot.get_obs(), now=1.0
    )
    assert result.phase == "armed: press both GELLO triggers"
    assert not result.active_arms and not robot.commands

    result = gello.step(
        leader_sample(left=pressed, right=pressed), robot.get_obs(), now=1.0
    )
    assert result.phase == "active"
    assert result.active_arms == {"left", "right"}
    assert [arm for arm, _ in robot.commands] == ["left", "right"]


def test_stale_input_disarms_and_requires_explicit_reactivation():
    robot = FakeFollower(("right",))
    gello = control(robot)
    joints = np.r_[np.zeros(6), 0.5]
    gello.step(
        leader_sample(right=joints),
        robot.get_obs(),
        toggle_arms=("right",),
        now=1.0,
    )
    commands = len(robot.commands)
    with pytest.raises(GelloInputError, match="right sample age"):
        gello.step(
            leader_sample(right=joints, timestamp=1.0),
            robot.get_obs(),
            now=2.0,
        )
    assert not gello.active_arms and len(robot.commands) == commands
    result = gello.step(
        leader_sample(right=joints, timestamp=2.0), robot.get_obs(), now=2.0
    )
    assert not result.active_arms
    result = gello.step(
        leader_sample(right=joints, timestamp=2.0),
        robot.get_obs(),
        toggle_arms=("right",),
        now=2.0,
    )
    assert result.active_arms == {"right"}


def test_absolute_alignment_is_slow():
    robot = FakeFollower(("right",))
    gello = control(robot, alignment="absolute")
    joints = np.r_[np.ones(6), 0.5]
    result = gello.step(
        leader_sample(right=joints),
        robot.get_obs(),
        toggle_arms=("right",),
        now=1.0,
    )
    np.testing.assert_allclose(result.joints[7:13], np.full(6, 0.05))


def test_absolute_alignment_slews_by_gello_delta_from_home():
    robot = FakeFollower(("right",))
    robot.home = {"right": np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])}
    gello = control(robot, alignment="absolute")
    leader = np.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    result = gello.step(
        leader_sample(right=leader),
        robot.get_obs(),
        toggle_arms=("right",),
        now=1.0,
    )
    # Target is home + GELLO delta from calibrated zero (0.6); 0.5 rad/s at 10 Hz.
    np.testing.assert_allclose(result.joints[7], 0.05)
    np.testing.assert_allclose(result.joints[8:13], 0.0)


def test_absolute_arming_waits_for_the_trigger_gate_then_slews_from_home():
    robot = FakeFollower()
    robot.home = {
        arm: np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) for arm in robot.arms
    }
    gello = control(robot, alignment="absolute")
    released = np.r_[np.zeros(6), 1.0]
    pressed = np.r_[np.zeros(6), 0.8]
    result = gello.step(
        leader_sample(left=released, right=released),
        robot.get_obs(),
        toggle_bimanual_arming=True,
        now=1.0,
    )
    assert result.phase == "armed: press both GELLO triggers"
    assert not result.active_arms and not robot.commands

    result = gello.step(
        leader_sample(left=pressed, right=pressed), robot.get_obs(), now=1.0
    )
    assert result.phase == "active"
    assert result.active_arms == {"left", "right"}
    np.testing.assert_allclose(result.joints[0], 0.05)
    np.testing.assert_allclose(result.joints[7], 0.05)


def test_fk_failure_prevents_either_active_arm_command():
    robot = FakeFollower()

    def fail_on_right(joints, arm):
        if arm == "right":
            raise ValueError("bad FK")
        return np.asarray(joints)

    robot.forward_kinematics = fail_on_right
    gello = control(robot)
    joints = np.r_[np.zeros(6), 0.5]
    with pytest.raises(ValueError, match="bad FK"):
        gello.step(
            leader_sample(left=joints, right=joints),
            robot.get_obs(),
            toggle_arms=("left", "right"),
            now=1.0,
        )
    assert not robot.commands
    assert not gello.active_arms


def test_follower_write_failure_disarms_control():
    robot = FakeFollower(("right",))
    robot.set_joints = lambda command, arm: (_ for _ in ()).throw(
        RuntimeError("follower write failed")
    )
    gello = control(robot)
    joints = np.r_[np.zeros(6), 0.5]
    with pytest.raises(RuntimeError, match="follower write failed"):
        gello.step(
            leader_sample(right=joints),
            robot.get_obs(),
            toggle_arms=("right",),
            now=1.0,
        )
    assert not gello.active_arms


def test_rl2_profile_is_self_contained_and_requires_physical_calibration():
    profile = yaml.safe_load(PROFILE.read_text())
    assert profile["robot"]["channels"] == {
        "left": "can_follower_l",
        "right": "can_follower_r",
    }
    assert profile["gello"]["driver"] == "ROBOTIS Dynamixel SDK"
    assert profile["gello"]["sdk_version"] == "4.0.5"
    assert profile["gello"]["bus"]["present_position_address"] == 132
    assert profile["gello"]["alignment"] == "absolute"
    assert isinstance(profile["gello"]["calibrated"], bool)
    assert profile["robot"]["gripper_kp"] == 20.0
    assert profile["robot"]["gripper_force_limit"] == 50.0
    assert profile["gello"]["input_filter_alpha"] == 0.9
    assert profile["frequency"] == 60
    assert profile["recording"]["rate_hz"] == 30
    assert profile["keys"]["record"] == "b"
    assert "teleop_kinematics" not in profile["robot"]
    uncalibrated = copy.deepcopy(profile)
    uncalibrated["gello"]["calibrated"] = False
    collect_gello.validate_gello_config(uncalibrated)
    with pytest.raises(RuntimeError, match="calibration is incomplete"):
        collect_gello.validate_gello_config(uncalibrated, require_calibrated=True)


def test_check_config_opens_no_serial_can_camera_or_robot(
    monkeypatch, capsys, tmp_path
):
    def forbidden(*args, **kwargs):
        raise AssertionError("configuration check touched a device")

    monkeypatch.setattr(collect_gello, "validate_serial_ports", forbidden)
    monkeypatch.setattr(collect_gello, "validate_can_interfaces", forbidden)
    monkeypatch.setattr(collect_gello, "validate_camera_devices", forbidden)
    monkeypatch.setattr(collect_gello, "BimanualGelloReader", forbidden)
    monkeypatch.setattr(collect_gello, "create_robot", forbidden)
    profile = yaml.safe_load(PROFILE.read_text())
    profile["gello"]["calibrated"] = False
    config = tmp_path / "uncalibrated.yaml"
    config.write_text(yaml.safe_dump(profile))
    assert collect_gello.main(["--config", str(config), "--check-config"]) == 0
    output = capsys.readouterr().out
    assert "no devices were opened" in output and "calibration: required" in output


def test_terminal_key_view_reads_without_camera_window_and_restores_tty():
    class Stream:
        def isatty(self):
            return True

        def fileno(self):
            return 9

        def read(self, length):
            assert length == 1
            return "g"

    calls = []
    terminal = SimpleNamespace(
        TCSADRAIN=1,
        tcgetattr=lambda fd: calls.append(("get", fd)) or ["saved"],
        tcsetattr=lambda fd, when, settings: calls.append(
            ("restore", fd, when, settings)
        ),
    )
    tty = SimpleNamespace(setcbreak=lambda fd: calls.append(("cbreak", fd)))
    stream = Stream()
    view = collect_gello.TerminalKeyView(
        stream=stream,
        termios_module=terminal,
        tty_module=tty,
        select_fn=lambda read, write, error, timeout: (read, write, error),
    )
    assert view.update({}, recording=False) == "g"
    view.close()
    view.close()
    assert calls == [
        ("get", 9),
        ("cbreak", 9),
        ("restore", 9, 1, ["saved"]),
    ]


def test_normal_launch_rejects_incomplete_calibration_before_preflight(
    monkeypatch, tmp_path
):
    def forbidden(*args, **kwargs):
        raise AssertionError("incomplete config reached a device preflight")

    monkeypatch.setattr(collect_gello, "validate_serial_ports", forbidden)
    monkeypatch.setattr(collect_gello, "validate_can_interfaces", forbidden)
    monkeypatch.setattr(collect_gello, "validate_camera_devices", forbidden)
    profile = yaml.safe_load(PROFILE.read_text())
    profile["gello"]["calibrated"] = False
    config = tmp_path / "uncalibrated.yaml"
    config.write_text(yaml.safe_dump(profile))
    with pytest.raises(RuntimeError, match="calibration is incomplete"):
        collect_gello.main(["--config", str(config)])


class SequenceView:
    def __init__(self, keys):
        self.keys = iter(keys)
        self.closed = False
        self.recording = []
        self.statuses = []
        self.episodes = []

    def update(self, obs, recording=False):
        self.recording.append(recording)
        return next(self.keys)

    def set_status(self, status):
        self.statuses.append(status)

    def set_episode(self, episode_id, state="next"):
        self.episodes.append((episode_id, state))

    def close(self):
        self.closed = True


class SafeReader:
    def read(self):
        joints = np.r_[np.zeros(6), 0.5]
        return leader_sample(left=joints, right=joints, timestamp=time.monotonic())


class TriggerReader:
    """Open both grippers, then squeeze, matching the arming gate."""

    def __init__(self, joints=None, squeeze_after=2):
        self.reads = 0
        self.squeeze_after = int(squeeze_after)
        self.joints = np.zeros(6) if joints is None else np.asarray(joints, dtype=float)

    def read(self):
        opening = 1.0 if self.reads < self.squeeze_after else 0.8
        self.reads += 1
        pose = np.r_[self.joints, opening]
        return leader_sample(left=pose, right=pose, timestamp=time.monotonic())


def test_record_homes_then_starts_after_the_trigger_gate(tmp_path, monkeypatch):
    profile = yaml.safe_load(PROFILE.read_text())
    profile["robot"]["cameras"] = {}
    profile["recording"].update(directory=str(tmp_path), episode_length=10)
    profile["preview"]["enabled"] = False
    monkeypatch.setattr(time, "sleep", lambda delay: None)
    robot = FakeFollower()
    robot.home = {
        arm: np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) for arm in robot.arms
    }
    view = SequenceView(["b", None, None, None, None, "q"])
    steps = collect_gello.run_collection(
        robot,
        TriggerReader(joints=[0.4, 0, 0, 0, 0, 0]),
        profile,
        view=view,
        max_steps=6,
    )
    assert steps == 5 and view.closed and robot.homed
    assert view.recording[:3] == [False, False, False]
    assert view.recording[3:] == [True, True, True]
    assert any(status.startswith("armed") for status in view.statuses)
    assert any(status == "active" for status in view.statuses)
    assert (0, "queued") in view.episodes
    assert (0, "recording") in view.episodes
    paths = list(tmp_path.glob("*.hdf5"))
    assert len(paths) == 1
    with h5py.File(paths[0]) as episode:
        assert episode["action"].shape == (2, 14)
        assert not episode.attrs["complete"]
        # First recorded command has started the slow move toward home + GELLO delta.
        assert episode["action"][0, 0] > 0
        assert episode["action"][0, 7] > 0


def test_g_arms_without_starting_an_episode(tmp_path, monkeypatch):
    profile = yaml.safe_load(PROFILE.read_text())
    profile["robot"]["cameras"] = {}
    profile["recording"].update(directory=str(tmp_path), episode_length=10)
    profile["preview"]["enabled"] = False
    monkeypatch.setattr(time, "sleep", lambda delay: None)
    robot = FakeFollower()
    view = SequenceView(["g", None, None, "q"])
    collect_gello.run_collection(
        robot, TriggerReader(squeeze_after=1), profile, view=view, max_steps=4
    )
    assert not list(tmp_path.glob("*.hdf5"))
    assert any(status.startswith("armed") for status in view.statuses)
    assert any(status == "active" for status in view.statuses)
    assert [arm for arm, _ in robot.commands][:2] == ["left", "right"]


def test_completed_recording_advances_dashboard_episode_id(tmp_path, monkeypatch):
    profile = yaml.safe_load(PROFILE.read_text())
    profile["robot"]["cameras"] = {}
    profile["recording"].update(directory=str(tmp_path), episode_length=1)
    profile["preview"]["enabled"] = False
    monkeypatch.setattr(time, "sleep", lambda delay: None)
    view = SequenceView(["b", None, None, "q"])
    collect_gello.run_collection(
        FakeFollower(), TriggerReader(), profile, view=view, max_steps=4
    )
    assert view.episodes[-1] == (1, "next")
    with h5py.File(tmp_path / "demo_0.hdf5") as episode:
        assert episode.attrs["complete"]


def test_episode_override_rejects_an_existing_demo(tmp_path, monkeypatch):
    with h5py.File(tmp_path / "demo_7.hdf5", "w") as episode:
        episode.attrs["complete"] = True
        episode.create_dataset("action", shape=(1, 14), dtype="f4")
    profile = yaml.safe_load(PROFILE.read_text())
    profile["robot"]["cameras"] = {}
    profile["recording"].update(directory=str(tmp_path), episode_start=0)
    profile["preview"]["enabled"] = False
    monkeypatch.setattr(time, "sleep", lambda delay: None)
    view = SequenceView([{"episode": 7}, {"episode": 4}, "q"])
    collect_gello.run_collection(
        FakeFollower(), SafeReader(), profile, view=view, max_steps=3
    )
    assert view.episodes[-1] == (4, "next")


def test_episode_override_is_editable_while_recording_is_queued(tmp_path, monkeypatch):
    profile = yaml.safe_load(PROFILE.read_text())
    profile["robot"]["cameras"] = {}
    profile["recording"].update(directory=str(tmp_path), episode_start=0)
    profile["preview"]["enabled"] = False
    monkeypatch.setattr(time, "sleep", lambda delay: None)
    view = SequenceView(["b", {"episode": 3}, "q"])
    collect_gello.run_collection(
        FakeFollower(), SafeReader(), profile, view=view, max_steps=3
    )
    assert (3, "queued") in view.episodes
    assert not list(tmp_path.glob("*.hdf5"))


def test_demo_sidebar_helpers_sort_and_delete_only_completed_files(tmp_path):
    for demo_id, complete in ((10, True), (2, True), (1, False)):
        with h5py.File(tmp_path / f"demo_{demo_id}.hdf5", "w") as episode:
            episode.attrs["complete"] = complete
            episode.create_dataset("action", shape=(demo_id, 14), dtype="f4")

    assert list_demos(tmp_path) == [
        {"id": 2, "frames": 2},
        {"id": 10, "frames": 10},
    ]
    assert delete_demo(tmp_path, 2) == {"deleted": 2}
    assert not (tmp_path / "demo_2.hdf5").exists()
    with pytest.raises(ValueError, match="completed"):
        delete_demo(tmp_path, 1)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        delete_demo(tmp_path, 99)


def test_dashboard_places_episode_and_demo_controls_below_cameras():
    static = Path("egomimic/robot/teleop_dashboard_static")
    html = (static / "index.html").read_text()
    css = (static / "style.css").read_text()
    javascript = (static / "app.js").read_text()
    assert html.index('id="episode-number"') < html.index('id="demos"')
    assert (
        html.index('id="status"')
        < html.index('class="command-row"')
        < html.index('id="cameras"')
        < html.index('class="bottom-bar"')
        < html.index('id="open-demos"')
        < html.index('id="episode-number"')
    )
    assert 'class="demo-sidebar"' in html
    assert 'class="demo-button"' in html
    assert 'id="previous-frame"' in html
    assert 'id="next-frame"' in html
    assert 'id="playback-time"' in html
    assert ".bottom-bar { display: grid" in css
    assert ".demo-button { padding: 11px 18px" in css
    assert ".review-cameras { display: grid" in css
    assert "pendingEpisodeFrames" in javascript
    assert "message.episode_state === 'recording'" in javascript
    assert "editable until recording starts" in javascript
    assert "openDemo(demos[0])" in javascript
    assert ".command-row { display: flex" in css
    assert "overflow-y: auto" in css
    assert "method: 'DELETE'" in javascript


def test_pending_recording_creates_no_file_until_triggers_activate(
    tmp_path, monkeypatch
):
    profile = yaml.safe_load(PROFILE.read_text())
    profile["robot"]["cameras"] = {}
    profile["recording"].update(directory=str(tmp_path), episode_length=10)
    profile["preview"]["enabled"] = False
    monkeypatch.setattr(time, "sleep", lambda delay: None)
    robot = FakeFollower()
    view = SequenceView(["b", None, "q"])
    assert (
        collect_gello.run_collection(
            robot, SafeReader(), profile, view=view, max_steps=3
        )
        == 2
    )
    assert not list(tmp_path.glob("*.hdf5"))
    assert any(
        status == "armed: release both GELLO triggers" for status in view.statuses
    )
    assert view.recording == [False, False, False]


def test_stop_key_disarms_followers_without_an_extra_command(monkeypatch):
    profile = yaml.safe_load(PROFILE.read_text())
    profile["robot"]["cameras"] = {}
    profile["preview"]["enabled"] = False
    profile["recording"]["enabled"] = False
    monkeypatch.setattr(time, "sleep", lambda delay: None)
    robot = FakeFollower()
    view = SequenceView(["g", None, "x"])
    assert (
        collect_gello.run_collection(
            robot, TriggerReader(squeeze_after=1), profile, view=view, max_steps=3
        )
        == 3
    )
    assert [arm for arm, _ in robot.commands] == ["left", "right"]
