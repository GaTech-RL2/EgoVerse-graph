"""Measured camera-to-reference poses for world-model windows.

Aria ``obs_head_pose`` stores world-from-RGB-camera, xyz then quaternion wxyz.
EVA native consumers treat stored arm extrinsics as base-from-camera (despite
the historical T_cam_base name); preserve that actual consumer convention. Wrist-camera poses require their own measured calibration.
No unknown pose is replaced by an identity measurement.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def camera_pose_matrices(xyzwxyz):
    """Convert native xyz+wxyz camera-to-world observations to SE(3)."""
    pose = np.asarray(xyzwxyz, dtype=np.float64)
    if pose.ndim < 2 or pose.shape[-1] != 7 or not np.isfinite(pose).all():
        raise ValueError("Expected finite native xyz+wxyz camera poses")
    if not np.allclose(np.linalg.norm(pose[..., 3:], axis=-1), 1, atol=1e-3):
        raise ValueError("Native camera quaternion must be unit length")
    result = np.broadcast_to(np.eye(4), pose.shape[:-1] + (4, 4)).copy()
    result[..., :3, :3] = Rotation.from_quat(pose[..., [4, 5, 6, 3]].reshape(-1, 4)).as_matrix().reshape(pose.shape[:-1] + (3, 3))
    result[..., :3, 3] = pose[..., :3]
    return result


def relative_camera_pose(world_from_camera, valid, reference_index, *, observed=None):
    """Return per-camera xyz+first-two-rotation-columns and validity.

    Inputs are T,C,4,4 in meters, T,C validity, and one reference time for
    each camera (scalar accepted). Training selects the first sampled time of
    an arbitrary window, independently of diffusion noise. Inference supplies
    observed=T,C and rejects a reference not in real observed context.
    Invalid rows remain NaN and false; they cannot become training targets.
    """
    poses = np.asarray(world_from_camera, dtype=np.float64)
    valid = np.asarray(valid)
    if poses.ndim != 4 or poses.shape[-2:] != (4, 4) or valid.shape != poses.shape[:2] or valid.dtype != bool:
        raise ValueError("Expected T,C,4,4 poses and boolean T,C validity")
    n, c = valid.shape
    references = np.broadcast_to(np.asarray(reference_index), (c,))
    if not np.issubdtype(references.dtype, np.integer) or np.any((references < 0) | (references >= n)):
        raise ValueError("Reference indices outside selected window")
    if observed is not None:
        observed = np.asarray(observed)
        if observed.shape != valid.shape or observed.dtype != bool:
            raise ValueError("Observed camera roles must be boolean T,C")
    out = np.full((n, c, 9), np.nan, dtype=np.float32)
    usable = valid.copy()
    for camera, ref in enumerate(references):
        if not valid[ref, camera]:
            usable[:, camera] = False
            continue
        if observed is not None and not observed[ref, camera]:
            raise ValueError("Inference camera reference must be a real observed pose")
        selected = poses[valid[:, camera], camera]
        r = selected[:, :3, :3]
        if (not np.isfinite(selected).all() or not np.allclose(selected[:, 3], [0, 0, 0, 1])
                or not np.allclose(r.swapaxes(-1, -2) @ r, np.eye(3), atol=1e-5)
                or not np.allclose(np.linalg.det(r), 1, atol=1e-5)):
            raise ValueError("Invalid measured camera rigid transform")
        relative = np.linalg.inv(poses[ref, camera]) @ selected
        out[valid[:, camera], camera] = np.concatenate(
            [relative[:, :3, 3], relative[:, :3, 0], relative[:, :3, 1]], -1
        )
    return out, usable


def native_camera_world_poses(embodiment, camera_names, total_frames, *,
                              head_pose=None, extrinsics=None, robot_reference_arm="left"):
    """Build measured camera targets from the native Zarr schema without writes.

    Returns absolute T,C,4,4 and boolean validity for preparation, not already
    sample-relative targets. The window loader must select its reference after
    sampling. EVA uses the chosen arm base as a fixed world coordinate system;
    its stored front-camera calibration is base-from-camera in native consumers.
    Human RGB world pose is ``obs_head_pose`` including its optical correction.
    The schema contains no wrist-camera calibration: those cameras stay missing.
    """
    if not camera_names or len(set(camera_names)) != len(camera_names) or total_frames < 1:
        raise ValueError("Unique camera names and positive frame count required")
    out = np.full((total_frames, len(camera_names), 4, 4), np.nan, dtype=np.float64)
    valid = np.zeros((total_frames, len(camera_names)), dtype=bool)
    if "front_1" not in camera_names:
        return out, valid
    camera = camera_names.index("front_1")
    if embodiment == "eva_bimanual":
        if not isinstance(extrinsics, dict) or robot_reference_arm not in extrinsics:
            return out, valid
        measured = np.asarray(extrinsics[robot_reference_arm], dtype=np.float64)
        if measured.shape != (4, 4) or not np.isfinite(measured).all():
            raise ValueError("Malformed measured EVA camera calibration")
        # Shared validator checks orthogonality/determinant before admitting it.
        relative_camera_pose(measured[None, None], np.ones((1, 1), bool), 0)
        out[:, camera] = measured
        valid[:, camera] = True
    elif embodiment in ("human", "human_bimanual", "aria_bimanual"):
        if head_pose is None:
            return out, valid
        pose = np.asarray(head_pose, dtype=np.float64)
        if pose.shape != (total_frames, 7):
            raise ValueError("Native obs_head_pose must have T,7 layout")
        measured = (np.isfinite(pose).all(axis=1)
                    & (np.abs(pose[:,:3]).max(axis=1) < 1000)
                    & np.isclose(np.linalg.norm(pose[:, 3:], axis=1), 1, atol=1e-3))
        if measured.any():
            out[measured, camera] = camera_pose_matrices(pose[measured])
            valid[measured, camera] = True
    else:
        raise ValueError("Unsupported native camera embodiment; require explicit converter")
    return out, valid
