from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from egomimic.rldb.embodiment.embodiment import EMBODIMENT
from egomimic.scripts.eva_process.eva_to_zarr import _split_per_arm
from egomimic.scripts.yam_process.yam_utils import (
    YamHD5Extractor,
    load_rl2_calibration,
)


def _write_episode(path: Path, *, complete: bool = True) -> dict[str, np.ndarray]:
    timesteps = 3
    eepose = np.zeros((timesteps, 14), dtype=np.float64)
    eepose[:, :6] = np.array([0.4, -0.2, 0.3, 0.2, -0.1, 0.05])
    eepose[:, 6] = [0.0, 0.5, 1.0]
    eepose[:, 7:13] = np.array([0.3, 0.15, 0.25, -0.3, 0.2, -0.1])
    eepose[:, 13] = [1.0, 0.5, 0.0]
    joints = np.full((timesteps, 14), 0.2, dtype=np.float64)
    joints[:, 6] = eepose[:, 6]
    joints[:, 13] = eepose[:, 13]
    image = np.arange(timesteps * 4 * 5 * 3, dtype=np.uint8).reshape(timesteps, 4, 5, 3)

    with h5py.File(path, "w") as episode:
        episode.attrs["complete"] = complete
        actions = episode.create_group("actions")
        actions.create_dataset("eepose", data=eepose)
        actions.create_dataset("joints", data=joints)
        observations = episode.create_group("observations")
        observations.create_dataset("eepose", data=eepose)
        observations.create_dataset("joints", data=joints)
        observations.create_dataset("joint_positions", data=joints)
        images = observations.create_group("images")
        images.create_dataset("front_img_1", data=image)
        images.create_dataset("left_wrist_img", data=image)
        images.create_dataset("right_wrist_img", data=image)
    return {"eepose": eepose, "joints": joints, "image": image}


def test_yam_extractor_transforms_right_pose_into_left_base(tmp_path):
    path = tmp_path / "episode.hdf5"
    raw = _write_episode(path)
    calibration = load_rl2_calibration()

    features = YamHD5Extractor.process_episode(path, "both")

    left_t_right = calibration["left_base_T_right_base"]
    expected_xyz = (left_t_right[:3, :3] @ raw["eepose"][:, 7:10].T).T + left_t_right[
        :3, 3
    ]
    expected_rot = (
        left_t_right[:3, :3][None]
        @ R.from_euler("ZYX", raw["eepose"][:, 10:13]).as_matrix()
    )
    expected_ypr = R.from_matrix(expected_rot).as_euler("ZYX")

    np.testing.assert_allclose(features["obs_eepose"][:, :7], raw["eepose"][:, :7])
    np.testing.assert_allclose(features["obs_eepose"][:, 7:10], expected_xyz)
    np.testing.assert_allclose(features["obs_eepose"][:, 10:13], expected_ypr)
    np.testing.assert_allclose(features["obs_eepose"][:, 13], raw["eepose"][:, 13])
    np.testing.assert_allclose(features["obs_joints"], raw["joints"])
    assert features["images.front_img_1"].shape == (3, 3, 4, 5)
    assert features["metadata.embodiment"].dtype == np.int32
    assert np.all(features["metadata.embodiment"] == EMBODIMENT.YAM_BIMANUAL.value)


def test_yam_quaternions_do_not_receive_eva_axis_rotation(tmp_path):
    path = tmp_path / "episode.hdf5"
    _write_episode(path)
    features = YamHD5Extractor.process_episode(path, "both")
    split = _split_per_arm(
        {
            "obs_ee_pose": features["obs_eepose"],
            "cmd_ee_pose": features["cmd_eepose"],
            "obs_joints": features["obs_joints"],
            "cmd_joints": features["cmd_joints"],
        },
        "both",
        rotate_to_eva_frame=False,
    )
    expected_xyzw = R.from_euler("ZYX", features["obs_eepose"][:, 3:6]).as_quat()
    expected_wxyz = expected_xyzw[:, [3, 0, 1, 2]]
    np.testing.assert_allclose(split["left.obs_ee_pose"][:, 3:], expected_wxyz)
    assert split["left.obs_ee_pose"].shape == (3, 7)
    assert split["right.obs_ee_pose"].shape == (3, 7)


def test_yam_extractor_rejects_incomplete_or_single_arm(tmp_path):
    path = tmp_path / "episode.hdf5"
    _write_episode(path, complete=False)
    with pytest.raises(ValueError, match="incomplete"):
        YamHD5Extractor.process_episode(path, "both")
    with pytest.raises(ValueError, match="bimanual"):
        YamHD5Extractor.process_episode(path, "left")
