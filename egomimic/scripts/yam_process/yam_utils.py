"""Extract RL2 YAM HDF5 episodes using the EVA processing feature contract.

The recorder stores each arm's end-effector pose in that arm's own base frame.
EgoVerse's ``yam_bimanual`` embodiment expects one station frame, so this module
uses the RL2 top-camera calibration to express the right arm in ``left_base``.
"""

from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from egomimic.rldb.embodiment.embodiment import EMBODIMENT

DATASET_KEY_MAPPINGS = {
    "joint_positions": "joint_positions",
    "front_img_1": "front_img_1",
    "right_wrist_img": "right_wrist_img",
    "left_wrist_img": "left_wrist_img",
}

YAM_EMBODIMENT = "yam_bimanual"

# Keys whose six arm values may need a missing all-zero frame filled.
ACTION_KEYS = {"cmd_eepose", "obs_eepose", "cmd_joints", "obs_joints"}
POSE_KEYS = {"cmd_eepose", "obs_eepose"}

# Exact per-device D405 intrinsics recorded at the RL2 station. The front-camera
# K is loaded from rl2yam.yaml; the wrist cameras are not part of that extrinsic
# calibration, but their K matrices are still useful to downstream consumers.
LEFT_WRIST_INTRINSICS = np.array(
    [
        [391.8486328125, 0.0, 323.69842529296875, 0.0],
        [0.0, 391.4812927246094, 242.69020080566406, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
RIGHT_WRIST_INTRINSICS = np.array(
    [
        [392.78973388671875, 0.0, 320.9379577636719, 0.0],
        [0.0, 392.3650207519531, 236.44271850585938, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def _default_calibration_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "hydra_configs"
        / "calibration"
        / "rl2yam.yaml"
    )


@lru_cache(maxsize=None)
def load_rl2_calibration(calibration_path: str | Path | None = None) -> dict:
    """Load and validate the station calibration used for YAM conversion."""
    path = Path(calibration_path or _default_calibration_path()).resolve()
    calibration = yaml.safe_load(path.read_text())
    left = np.asarray(calibration["extrinsics"]["left"], dtype=np.float64)
    right = np.asarray(calibration["extrinsics"]["right"], dtype=np.float64)
    front_k = np.asarray(calibration["intrinsics"], dtype=np.float64)
    if left.shape != (4, 4) or right.shape != (4, 4):
        raise ValueError(f"RL2 YAM extrinsics must be 4x4 matrices: {path}")
    if front_k.shape != (3, 4):
        raise ValueError(f"RL2 YAM intrinsics must be a 3x4 matrix: {path}")
    if not all(np.isfinite(value).all() for value in (left, right, front_k)):
        raise ValueError(f"RL2 YAM calibration contains non-finite values: {path}")
    if not np.allclose(left[3], [0, 0, 0, 1]) or not np.allclose(
        right[3], [0, 0, 0, 1]
    ):
        raise ValueError(f"RL2 YAM extrinsics are not homogeneous transforms: {path}")

    left_t_right = left @ np.linalg.inv(right)
    return {
        "path": str(path),
        "source_repository": calibration.get("source_repository", ""),
        "source_revision": calibration.get("source_revision", ""),
        "camera_serial": calibration.get("camera_serial", ""),
        "left_base_T_camera": left,
        "right_base_T_camera": right,
        "left_base_T_right_base": left_t_right,
        "intrinsics": {
            "front_1": front_k,
            "left_wrist": LEFT_WRIST_INTRINSICS,
            "right_wrist": RIGHT_WRIST_INTRINSICS,
        },
        "extrinsics": {"front_1": left},
    }


def _transform_right_pose_to_left_base(data: np.ndarray, left_t_right: np.ndarray):
    """Transform the right xyz+ypr columns of a bimanual array in-place."""
    data = np.asarray(data).copy()
    right_xyz = data[:, 7:10]
    right_rotation = R.from_euler("ZYX", data[:, 10:13], degrees=False).as_matrix()
    data[:, 7:10] = (left_t_right[:3, :3] @ right_xyz.T).T + left_t_right[:3, 3]
    data[:, 10:13] = R.from_matrix(
        left_t_right[:3, :3][None, :, :] @ right_rotation
    ).as_euler("ZYX", degrees=False)
    return data


class YamHD5Extractor:
    """Read bimanual YAM recordings into the dictionary consumed by EVA Zarr code."""

    @staticmethod
    def process_episode(episode_path, arm, calibration_path=None):
        """Extract one complete YAM episode.

        YAM collection always records both followers. Images are returned as TCHW
        uint8 arrays; observed and commanded state keys use the same names as
        :class:`EvaHD5Extractor`, so the existing numeric/image split and Zarr
        writer can be reused.
        """
        if arm != "both":
            raise ValueError("RL2 YAM recordings are bimanual; arm must be 'both'")

        episode_feats = {}
        with h5py.File(episode_path, "r") as episode:
            if not bool(episode.attrs.get("complete", False)):
                raise ValueError(f"YAM episode is incomplete: {episode_path}")

            for camera in YamHD5Extractor.get_cameras(episode):
                images = np.asarray(episode["observations/images"][camera][:])
                if images.ndim != 4 or images.shape[-1] not in (1, 3, 4):
                    raise ValueError(
                        f"YAM camera {camera} must be THWC; got {images.shape}"
                    )
                mapped_key = DATASET_KEY_MAPPINGS.get(camera, camera)
                episode_feats[f"images.{mapped_key}"] = images.transpose(0, 3, 1, 2)

            for state in YamHD5Extractor.get_obs_state(episode):
                mapped_key = DATASET_KEY_MAPPINGS.get(state, state)
                episode_feats[f"obs_{mapped_key}"] = episode["observations"][state][:]

            for state in YamHD5Extractor.get_cmd_state(episode):
                mapped_key = DATASET_KEY_MAPPINGS.get(state, state)
                episode_feats[f"cmd_{mapped_key}"] = episode["actions"][state][:]

        YamHD5Extractor.validate_features(episode_feats)
        for key in ACTION_KEYS:
            if key in episode_feats:
                episode_feats[key] = YamHD5Extractor.clean_zero_data(episode_feats[key])

        calibration = load_rl2_calibration(calibration_path)
        for key in POSE_KEYS:
            episode_feats[key] = _transform_right_pose_to_left_base(
                episode_feats[key], calibration["left_base_T_right_base"]
            )

        num_timesteps = episode_feats["obs_eepose"].shape[0]
        episode_feats["metadata.embodiment"] = np.full(
            (num_timesteps, 1), EMBODIMENT.YAM_BIMANUAL.value, dtype=np.int32
        )
        return episode_feats

    @staticmethod
    def get_cameras(hdf5_data: h5py.File):
        if "observations/images" not in hdf5_data:
            raise ValueError("YAM episode is missing observations/images")
        return [key for key in hdf5_data["observations/images"] if "depth" not in key]

    @staticmethod
    def get_obs_state(hdf5_data: h5py.File):
        if "observations" not in hdf5_data:
            raise ValueError("YAM episode is missing observations")
        return [key for key in hdf5_data["observations"] if key != "images"]

    @staticmethod
    def get_cmd_state(hdf5_data: h5py.File):
        if "actions" not in hdf5_data:
            raise ValueError("YAM episode is missing actions")
        return list(hdf5_data["actions"])

    @staticmethod
    def validate_features(episode_feats: dict) -> None:
        required = {"obs_eepose", "obs_joints", "cmd_eepose", "cmd_joints"}
        missing = sorted(required - episode_feats.keys())
        if missing:
            raise ValueError(f"YAM episode is missing required features: {missing}")

        lengths = {key: len(value) for key, value in episode_feats.items()}
        if not lengths or len(set(lengths.values())) != 1:
            raise ValueError(f"YAM feature lengths do not match: {lengths}")

        for key in required:
            value = np.asarray(episode_feats[key])
            if value.ndim != 2 or value.shape[1] != 14:
                raise ValueError(
                    f"YAM {key} must have shape (T, 14); got {value.shape}"
                )
            if not np.isfinite(value).all():
                raise ValueError(f"YAM {key} contains non-finite values")
            for index in (6, 13):
                gripper = value[:, index]
                if np.any((gripper < -1e-3) | (gripper > 1.001)):
                    raise ValueError(f"YAM {key} gripper is not normalized to [0, 1]")

    @staticmethod
    def clean_zero_data(data: np.ndarray) -> np.ndarray:
        """Fill missing all-zero six-axis frames independently for each arm."""
        data = data.copy()
        for pose_slice in (slice(0, 6), slice(7, 13)):
            zero_mask = np.all(data[:, pose_slice] == 0, axis=1)
            if not np.any(zero_mask):
                continue
            nonzero_indices = np.flatnonzero(~zero_mask)
            if not len(nonzero_indices):
                continue
            for timestep in np.flatnonzero(zero_mask):
                before = nonzero_indices[nonzero_indices < timestep]
                if len(before):
                    source = before[-1]
                else:
                    source = nonzero_indices[nonzero_indices > timestep][0]
                data[timestep, pose_slice] = data[source, pose_slice]
        return data
