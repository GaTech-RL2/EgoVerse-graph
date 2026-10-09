"""Hardware-free checks for demo-format rollout episode recording."""

from __future__ import annotations

import json
import threading
import time
from collections import namedtuple
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from egomimic.robot import rollout_episode
from egomimic.robot.collect_demo import EpisodeWriter
from egomimic.robot.interface import ARM_OFFSET
from egomimic.robot.rollout import run_rollout, validate_rollout_config
from egomimic.robot.rollout_episode import (
    RolloutEpisodeRecorder,
    list_rollout_episodes,
    validate_episode_recording,
)

CAMERAS = {"front_img_1": (4, 6), "left_wrist_img": (4, 6)}
Usage = namedtuple("Usage", "total used free")


@pytest.fixture
def disk(monkeypatch):
    """Free space the recorder sees, in GB (shutil.disk_usage is patched)."""
    state = {"free_gb": 500.0}
    monkeypatch.setattr(
        rollout_episode.shutil,
        "disk_usage",
        lambda _path: Usage(10**13, 0, int(state["free_gb"] * 1e9)),
    )
    return state


def observation(k):
    obs = {"joint_positions": np.full(14, 0.01 * k), "ee_poses": np.full(14, 0.02 * k)}
    for name, (height, width) in CAMERAS.items():
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[..., 0], frame[..., 2] = k, 200  # BGR, as the camera drivers publish
        obs[name] = frame
    return obs


def command(k):
    joints = np.full(14, 0.1 * k)
    joints[[6, 13]] = 0.5
    return joints


def finished(recorder):
    assert recorder.wait(timeout=10)
    results = recorder.take_finished()
    assert len(results) == 1
    return results[0]


def dataset_tree(path):
    tree = {}
    with h5py.File(path, "r") as file:
        file.visititems(
            lambda name, item: tree.__setitem__(name, (item.shape, str(item.dtype)))
            if isinstance(item, h5py.Dataset)
            else None
        )
    return tree


def test_episode_recording_config_is_validated():
    assert validate_episode_recording(None)["enabled"] is False
    config = validate_episode_recording(
        {"enabled": True, "directory": "/tmp/episodes", "min_free_gb": 5}
    )
    assert config["min_free_gb"] == 5.0 and config["max_queued_rows"] == 60
    for bad in (
        {"enabled": "yes"},
        {"directory": "relative/episodes"},
        {"min_free_gb": -1},
        {"min_free_gb": True},
        {"max_queued_rows": 0},
        {"fps": 12},
    ):
        with pytest.raises(ValueError):
            validate_episode_recording(bad)
    with pytest.raises(ValueError, match="episode_recording"):
        validate_rollout_config({"episode_recording": {"enabled": 1}, "policy": {}})


def test_recorder_writes_the_gello_demo_layout_plus_a_rollout_group(tmp_path, disk):
    recorder = RolloutEpisodeRecorder(
        CAMERAS, {"enabled": True, "directory": str(tmp_path)}
    )
    episode_id = recorder.start(
        {"checkpoint": "/ckpt/model.ckpt", "policy_config": {"kind": "graph"}}
    )
    plan = np.arange(5 * 14, dtype=float).reshape(5, 14) / 100
    assert recorder.mark_plan(
        plan, executed_rows=3, inference_seconds=0.05,
        info={"controls": {"replan_every": 3}},
    )
    reference = EpisodeWriter(tmp_path / "reference" / "demo_0.hdf5", CAMERAS)
    for k in range(3):
        eepose = np.full(14, 0.2 * k)
        assert recorder.append(
            observation(k), command(k), eepose,
            target=np.full(14, 0.3 * k), now=recorder._t0 + 0.04 * k,
        )
        reference.append(observation(k), command(k), eepose)
    reference.close()
    assert recorder.close(outcome="success") == episode_id
    assert not recorder.recording
    saved = finished(recorder)["saved"]

    path = tmp_path / f"{episode_id}.hdf5"
    assert saved["filename"] == path.name and saved["frames"] == 3
    assert saved["outcome"] == "success" and saved["complete"] is True
    assert saved["plans"] == 1 and saved["checkpoint"] == "/ckpt/model.ckpt"
    episode, demo = dataset_tree(path), dataset_tree(reference.path)
    # Top level is exactly a demo; everything else lives under rollout/.
    assert {k: v for k, v in episode.items() if not k.startswith("rollout/")} == demo
    with h5py.File(path, "r") as file, h5py.File(reference.path, "r") as ref:
        for key in demo:
            np.testing.assert_array_equal(file[key][()], ref[key][()])
        assert bool(file.attrs["complete"]) and not bool(file.attrs["sim"])
        assert file.attrs["kind"] == "rollout"
        assert file.attrs["outcome"] == "success"
        assert file.attrs["end_reason"] == "saved"
        assert file.attrs["checkpoint"] == "/ckpt/model.ckpt"
        assert json.loads(file.attrs["policy_config"]) == {"kind": "graph"}
        np.testing.assert_allclose(file["rollout/timestamps"][()], [0, 0.04, 0.08])
        np.testing.assert_array_equal(file["rollout/plan_index"][()], [0, 0, 0])
        np.testing.assert_allclose(file["rollout/target_eepose"][2], 0.6, rtol=1e-6)
        np.testing.assert_allclose(file["rollout/plans/actions"][0], plan, rtol=1e-6)
        assert file["rollout/plans/executed_rows"][0] == 3
        assert file["rollout/plans/start_row"][0] == 0
        info = json.loads(file["rollout/plans/info"][0])
        assert info["controls"] == {"replan_every": 3}
    assert [episode["id"] for episode in list_rollout_episodes(tmp_path)] == [episode_id]


def test_discarded_and_empty_episodes_leave_no_file(tmp_path, disk):
    recorder = RolloutEpisodeRecorder(
        CAMERAS, {"enabled": True, "directory": str(tmp_path)}
    )
    recorder.start({})
    assert recorder.append(observation(0), command(0), np.zeros(14))
    recorder.discard()
    assert finished(recorder)["discarded"]
    recorder.start({})
    recorder.close()
    assert finished(recorder)["saved"] is None
    assert list(tmp_path.iterdir()) == []


def test_low_disk_refuses_to_start_and_stops_a_running_episode(tmp_path, disk):
    recorder = RolloutEpisodeRecorder(
        CAMERAS, {"enabled": True, "directory": str(tmp_path), "min_free_gb": 20}
    )
    disk["free_gb"] = 5.0
    with pytest.raises(RuntimeError, match="GB free"):
        recorder.start({})
    disk["free_gb"] = 500.0
    recorder.start({})
    assert recorder.append(observation(0), command(0), np.zeros(14), now=recorder._t0)
    disk["free_gb"] = 5.0
    assert not recorder.append(
        observation(1), command(1), np.zeros(14), now=recorder._t0 + 2.0
    )
    assert recorder.failure == "low_disk"
    recorder.close(outcome="success")
    saved = finished(recorder)["saved"]
    assert saved["frames"] == 1
    assert saved["complete"] is False and saved["end_reason"] == "low_disk"


def test_a_slow_writer_ends_the_episode_instead_of_blocking_control(
    tmp_path, disk, monkeypatch
):
    gate = threading.Event()
    original = EpisodeWriter.append

    def slow_append(self, *args):
        gate.wait(5)
        return original(self, *args)

    monkeypatch.setattr(rollout_episode.EpisodeWriter, "append", slow_append)
    recorder = RolloutEpisodeRecorder(
        CAMERAS, {"enabled": True, "directory": str(tmp_path), "max_queued_rows": 1}
    )
    recorder.start({})
    began = time.monotonic()
    accepted = [
        recorder.append(observation(k), command(k), np.zeros(14)) for k in range(4)
    ]
    assert time.monotonic() - began < 0.5  # never waits on the disk
    assert accepted[0] and not all(accepted)
    assert recorder.failure == "writer_error"
    gate.set()
    recorder.close()
    saved = finished(recorder)["saved"]
    assert saved["complete"] is False and saved["end_reason"] == "writer_error"


@pytest.mark.parametrize("end", ["close", "discard"])
def test_ending_a_stalled_episode_never_blocks_control(tmp_path, disk, monkeypatch, end):
    gate = threading.Event()
    original = EpisodeWriter.append

    def slow_append(self, *args):
        gate.wait(5)
        return original(self, *args)

    monkeypatch.setattr(rollout_episode.EpisodeWriter, "append", slow_append)
    recorder = RolloutEpisodeRecorder(
        CAMERAS, {"enabled": True, "directory": str(tmp_path), "max_queued_rows": 1}
    )
    recorder.start({})
    accepted = [
        recorder.append(observation(k), command(k), np.zeros(14)) for k in range(3)
    ]
    assert accepted[0] and not accepted[-1] and recorder.failure == "writer_error"
    began = time.monotonic()
    episode_id = getattr(recorder, end)()  # the writer is stalled inside one row
    assert time.monotonic() - began < 0.1
    assert episode_id is not None and not recorder.recording and recorder.busy
    gate.set()
    result = finished(recorder)
    if end == "discard":
        assert result["discarded"] is True
        assert not list(tmp_path.glob("rollout_*"))
    else:
        saved = result["saved"]
        assert saved["complete"] is False and saved["end_reason"] == "writer_error"
        # Every accepted row was written before finalizing (how many were accepted
        # depends on whether the writer had dequeued the first row yet).
        assert saved["frames"] == sum(accepted) >= 1


class EpisodeRobot:
    def __init__(self, with_fk=True):
        self.arms = ["left", "right"]
        self.camera_res = {"front_img_1": (2, 3)}
        self.q = np.zeros(14)
        self.q[[6, 13]] = 0.5
        self.commands = []
        if with_fk:
            self.forward_kinematics = lambda joints, _arm: np.asarray(joints)[:6] + 1.0

    def get_obs(self):
        return {
            "joint_positions": self.q.copy(),
            "ee_poses": self.q.copy(),
            "front_img_1": np.full((2, 3, 3), 7, dtype=np.uint8),
        }

    def solve_ik(self, pose, _arm):
        return np.asarray(pose, dtype=float)

    def set_joints(self, command, arm):
        command = np.asarray(command, dtype=float).copy()
        self.commands.append((arm, command))
        self.q[ARM_OFFSET[arm] : ARM_OFFSET[arm] + 7] = command

    def set_home(self):
        pass


class EpisodeView:
    def __init__(self, controls, requests):
        self.controls, self.requests = iter(controls), iter(requests)
        self.statuses, self.published = [], []

    def update(self, _obs):
        return next(self.controls)

    def take_episode_request(self):
        return next(self.requests, None)

    def set_episode_recording(self, recording, frames=0, saved=None):
        self.published.append((recording, frames, saved))

    def set_status(self, status):
        self.statuses.append(status)

    def clear_action_plan(self):
        pass

    def close(self):
        pass


def run(robot, view, tmp_path, monkeypatch):
    monkeypatch.setattr("egomimic.robot.rollout.time.sleep", lambda _: None)
    target = np.zeros((1, 14))
    target[:, [0, 7]] = 0.01
    target[:, [6, 13]] = 0.5
    policy = SimpleNamespace(action_type="cartesian", predict=lambda _obs: target)
    run_rollout(
        robot,
        policy,
        {
            "frequency": 30,
            "max_steps": 20,
            "max_joint_velocity": 1.0,
            "preview": {"enabled": False, "wait_for_start": True},
            "episode_recording": {"enabled": True, "directory": str(tmp_path)},
            "policy": {"kind": "graph", "checkpoint": "/ckpt/m.ckpt"},
        },
        view=view,
    )
    return target[0], sorted(tmp_path.glob("rollout_*.hdf5"))


START = {"action": "start"}


def test_rollout_records_executed_ticks_and_saves_the_labelled_outcome(
    tmp_path, disk, monkeypatch
):
    robot = EpisodeRobot()
    view = EpisodeView(
        [None, "c", None, None, None, "q"],
        [None, None, START, None, {"action": "save", "outcome": "success"}],
    )
    target, files = run(robot, view, tmp_path, monkeypatch)

    assert len(files) == 1
    with h5py.File(files[0], "r") as file:
        assert bool(file.attrs["complete"]) and file.attrs["outcome"] == "success"
        assert file.attrs["checkpoint"] == "/ckpt/m.ckpt"
        joints = file["actions/joints"][()]
        assert joints.shape == (2, 14)  # the two ticks executed while recording
        sent = np.r_[target[0:6], 0.5, target[7:13], 0.5]  # identity IK
        np.testing.assert_allclose(joints, [sent, sent], rtol=1e-6)
        expected_fk = np.r_[target[0:6] + 1.0, 0.5, target[7:13] + 1.0, 0.5]
        np.testing.assert_allclose(file["actions/eepose"][0], expected_fk, rtol=1e-6)
        np.testing.assert_allclose(file["rollout/target_eepose"][0], target, rtol=1e-6)
        np.testing.assert_array_equal(file["rollout/plan_index"][()], [0, 1])
        assert file["rollout/plans/actions"].shape == (2, 1, 14)
        assert file["observations/images/front_img_1"].shape == (2, 2, 3, 3)
    assert any(saved and saved["outcome"] == "success" for _, _, saved in view.published)


@pytest.mark.parametrize(
    "controls, complete, reason",
    [([None, "c", None, "r", "q"], True, "restart"), ([None, "c", None, "q"], False, "stop")],
)
def test_rollout_keeps_an_unsaved_episode_on_restart_or_stop(
    tmp_path, disk, monkeypatch, controls, complete, reason
):
    view = EpisodeView(controls, [None, None, START])
    _, files = run(EpisodeRobot(), view, tmp_path, monkeypatch)

    assert len(files) == 1
    with h5py.File(files[0], "r") as file:
        assert file["actions/joints"].shape[0] == 1
        assert bool(file.attrs["complete"]) is complete
        assert file.attrs["outcome"] == "unlabeled"
        assert file.attrs["end_reason"] == reason


def test_rollout_without_forward_kinematics_records_nothing(
    tmp_path, disk, monkeypatch
):
    view = EpisodeView([None, "c", None, "q"], [None, None, START])
    _, files = run(EpisodeRobot(with_fk=False), view, tmp_path, monkeypatch)

    assert files == []
    assert any("forward kinematics" in status for status in view.statuses)
