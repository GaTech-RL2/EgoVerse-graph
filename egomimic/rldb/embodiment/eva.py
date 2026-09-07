from __future__ import annotations

from typing import Literal

import numpy as np

from egomimic.rldb.embodiment.embodiment import Embodiment
from egomimic.rldb.embodiment.human import ARIA_INTRINSICS
from egomimic.rldb.zarr.action_chunk_transforms import (
    ActionChunkCoordinateFrameTransform,
    ConcatKeys,
    DeleteKeys,
    InterpolateLinear,
    InterpolatePose,
    NumpyToTensor,
    PoseCoordinateFrameTransform,
    SplitKeys,
    Transform,
    transforms_for_rotation_mode,
)
from egomimic.utils.pose_utils import (
    _matrix_to_xyzwxyz,
)


class Eva(Embodiment):
    INTRINSICS = ARIA_INTRINSICS
    EXTRINSICS = {
        "left": np.array(
            [
                [0.01329544, -0.71757193, 0.69635749, -0.04409191],
                [-0.99959782, -0.02698416, -0.00872107, -0.23221381],
                [0.02504862, -0.69596148, -0.7176421, 0.57323278],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        "right": np.array(
            [
                [-0.04733948, -0.76631195, 0.64072222, -0.01998031],
                [-0.9983006, 0.05811952, -0.00424732, 0.32539554],
                [-0.0339837, -0.63983444, -0.76776103, 0.64809634],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    }

    @staticmethod
    def get_transform_list(
        action_mode: Literal["cartesian"] = "cartesian",
        coord_frame: Literal["camframe", "eef_frame"] = "camframe",
        rotation_mode: Literal["euler", "quat", "6D"] = "euler",
    ) -> list[Transform]:
        """``action_mode`` is the action layout; ``coord_frame`` is where poses
        live; ``rotation_mode`` is how rotation is stored.

        Cam-frame actions are expressed in the wrist cameras via
        :attr:`EXTRINSICS`. EEF-frame actions are a delta from the current EEF
        pose. In both cases the geometric hops run in xyz+quat, then
        ``rotation_mode`` converts rotation to euler (xyz+ypr, 14D), quat (16D),
        or Zhou 6D (20D).
        """
        if action_mode != "cartesian":
            raise ValueError(f"unknown action_mode {action_mode!r}")
        if coord_frame == "camframe":
            return _build_eva_bimanual_transform_list(rotation_mode=rotation_mode)
        if coord_frame == "eef_frame":
            return _build_eva_bimanual_eef_frame_transform_list(
                rotation_mode=rotation_mode
            )
        raise ValueError(f"unknown coord_frame {coord_frame!r}")

    @classmethod
    def _get_keymap(cls, keymap_mode: str):
        if keymap_mode != "cartesian":
            raise ValueError(
                f"Unsupported keymap_mode {keymap_mode!r} for {cls.__name__}; "
                "expected 'cartesian'"
            )
        front_key = cls.VIZ_IMAGE_KEY
        right_wrist_key = "observations.images.right_wrist_img"
        left_wrist_key = "observations.images.left_wrist_img"

        key_map = {
            front_key: {
                "key_type": "camera_keys",
                "zarr_key": "images.front_1",
            },
            right_wrist_key: {
                "key_type": "camera_keys",
                "zarr_key": "images.right_wrist",
            },
            left_wrist_key: {
                "key_type": "camera_keys",
                "zarr_key": "images.left_wrist",
            },
            "right.obs_ee_pose": {
                "key_type": "proprio_keys",
                "zarr_key": "right.obs_ee_pose",
            },
            "right.obs_gripper": {
                "key_type": "proprio_keys",
                "zarr_key": "right.obs_gripper",
            },
            "left.obs_ee_pose": {
                "key_type": "proprio_keys",
                "zarr_key": "left.obs_ee_pose",
            },
            "left.obs_gripper": {
                "key_type": "proprio_keys",
                "zarr_key": "left.obs_gripper",
            },
            "right.cmd_gripper": {
                "key_type": "action_keys",
                "zarr_key": "right.cmd_gripper",
                "horizon": 45,
            },
            "left.cmd_gripper": {
                "key_type": "action_keys",
                "zarr_key": "left.cmd_gripper",
                "horizon": 45,
            },
            "right.cmd_ee_pose": {
                "key_type": "action_keys",
                "zarr_key": "right.cmd_ee_pose",
                "horizon": 45,
            },
            "left.cmd_ee_pose": {
                "key_type": "action_keys",
                "zarr_key": "left.cmd_ee_pose",
                "horizon": 45,
            },
        }

        return key_map

    @classmethod
    def dinov3_keymap(cls):
        """
        Compact keymap for alignment training: cartesian action chunk, the
        DINOv3 image embedding produced by the embedding_process pipeline, and
        the language annotation track.
        """
        return {
            "actions_cartesian": {
                "key_type": "action_keys",
                "zarr_key": "actions_cartesian",
            },
            "dino_front_1": {
                "key_type": "proprio_keys",
                "zarr_key": "dino.front_img_1",
            },
            "annotations": {
                "key_type": "annotation_keys",
                "zarr_key": "annotations",
            },
        }


def _build_eva_bimanual_revert_eef_frame_transform_list(
    *,
    action_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
    left_cmd_wristframe: str = "left.cmd_ee_pose_wristframe",
    right_cmd_wristframe: str = "right.cmd_ee_pose_wristframe",
    left_cmd_gripper: str = "left.cmd_gripper",
    right_cmd_gripper: str = "right.cmd_gripper",
    left_obs_camframe: str = "left.obs_ee_pose_camframe",
    right_obs_camframe: str = "right.obs_ee_pose_camframe",
    left_obs_gripper: str = "left.obs_gripper",
    right_obs_gripper: str = "right.obs_gripper",
    left_cmd_camframe: str = "left.cmd_ee_pose_camframe",
    right_cmd_camframe: str = "right.cmd_ee_pose_camframe",
    is_quat: bool = True,
) -> list[Transform]:
    """Revert wrist-frame EVA actions back to camera frame for visualization."""
    if is_quat:
        pose_shape = 7
    else:
        pose_shape = 6
    transform_list = [
        # Extract obs camframe poses from the concatenated obs key
        SplitKeys(
            input_key=obs_key,
            output_key_list=[
                (left_obs_camframe, pose_shape),
                (left_obs_gripper, 1),
                (right_obs_camframe, pose_shape),
                (right_obs_gripper, 1),
            ],
        ),
        # Split wrist-frame actions into per-arm chunks
        SplitKeys(
            input_key=action_key,
            output_key_list=[
                (left_cmd_wristframe, pose_shape),
                (left_cmd_gripper, 1),
                (right_cmd_wristframe, pose_shape),
                (right_cmd_gripper, 1),
            ],
        ),
        # Revert wrist frame → camera frame (inverse=False: target_se3 @ chunk_se3)
        ActionChunkCoordinateFrameTransform(
            target_world=left_obs_camframe,
            chunk_world=left_cmd_wristframe,
            transformed_key_name=left_cmd_camframe,
            mode="xyzypr",
            inverse=False,
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=right_obs_camframe,
            chunk_world=right_cmd_wristframe,
            transformed_key_name=right_cmd_camframe,
            mode="xyzypr",
            inverse=False,
        ),
        ConcatKeys(
            key_list=[
                left_cmd_camframe,
                left_cmd_gripper,
                right_cmd_camframe,
                right_cmd_gripper,
            ],
            new_key_name=action_key,
            delete_old_keys=True,
        ),
    ]
    return transform_list


def _build_eva_bimanual_eef_frame_transform_list(
    *,
    left_target_world: str = "left_extrinsics_pose",
    right_target_world: str = "right_extrinsics_pose",
    left_cmd_world: str = "left.cmd_ee_pose",
    right_cmd_world: str = "right.cmd_ee_pose",
    left_obs_pose: str = "left.obs_ee_pose",
    right_obs_pose: str = "right.obs_ee_pose",
    left_obs_gripper: str = "left.obs_gripper",
    right_obs_gripper: str = "right.obs_gripper",
    left_cmd_gripper: str = "left.cmd_gripper",
    right_cmd_gripper: str = "right.cmd_gripper",
    left_cmd_camframe: str = "left.cmd_ee_pose_camframe",
    right_cmd_camframe: str = "right.cmd_ee_pose_camframe",
    left_obs_camframe: str = "left.obs_ee_pose_camframe",
    right_obs_camframe: str = "right.obs_ee_pose_camframe",
    left_cmd_wristframe: str = "left.cmd_ee_pose_wristframe",
    right_cmd_wristframe: str = "right.cmd_ee_pose_wristframe",
    actions_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
    chunk_length: int = 100,
    stride: int = 1,
    rotation_mode: Literal["euler", "quat", "6D"] = "euler",
) -> list[Transform]:
    """EVA bimanual transform pipeline with actions expressed relative to the
    current EEF pose (wrist frame), analogous to keypoints relative to wrist pose."""
    extrinsics = Eva.EXTRINSICS
    left_extrinsics_pose = _matrix_to_xyzwxyz(extrinsics["left"][None, :])[0]
    right_extrinsics_pose = _matrix_to_xyzwxyz(extrinsics["right"][None, :])[0]
    left_extra_batch_key = {"left_extrinsics_pose": left_extrinsics_pose}
    right_extra_batch_key = {"right_extrinsics_pose": right_extrinsics_pose}

    # Step 1: transform cmd and obs into camera frame using extrinsics
    transform_list = [
        ActionChunkCoordinateFrameTransform(
            target_world=left_target_world,
            chunk_world=left_cmd_world,
            transformed_key_name=left_cmd_camframe,
            extra_batch_key=left_extra_batch_key,
            mode="xyzwxyz",
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=right_target_world,
            chunk_world=right_cmd_world,
            transformed_key_name=right_cmd_camframe,
            extra_batch_key=right_extra_batch_key,
            mode="xyzwxyz",
        ),
        PoseCoordinateFrameTransform(
            target_world=left_target_world,
            pose_world=left_obs_pose,
            transformed_key_name=left_obs_camframe,
            mode="xyzwxyz",
        ),
        PoseCoordinateFrameTransform(
            target_world=right_target_world,
            pose_world=right_obs_pose,
            transformed_key_name=right_obs_camframe,
            mode="xyzwxyz",
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=left_cmd_camframe,
            output_action_key=left_cmd_camframe,
            stride=stride,
            mode="xyzwxyz",
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=right_cmd_camframe,
            output_action_key=right_cmd_camframe,
            stride=stride,
            mode="xyzwxyz",
        ),
        InterpolateLinear(
            new_chunk_length=chunk_length,
            action_key=left_cmd_gripper,
            output_action_key=left_cmd_gripper,
            stride=stride,
        ),
        InterpolateLinear(
            new_chunk_length=chunk_length,
            action_key=right_cmd_gripper,
            output_action_key=right_cmd_gripper,
            stride=stride,
        ),
        # Step 2: transform camera-frame actions into EEF-relative (wrist) frame
        ActionChunkCoordinateFrameTransform(
            target_world=left_obs_camframe,
            chunk_world=left_cmd_camframe,
            transformed_key_name=left_cmd_wristframe,
            mode="xyzwxyz",
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=right_obs_camframe,
            chunk_world=right_cmd_camframe,
            transformed_key_name=right_cmd_wristframe,
            mode="xyzwxyz",
        ),
    ]

    transform_list.extend(
        transforms_for_rotation_mode(
            keys=[
                left_cmd_wristframe,
                right_cmd_wristframe,
                left_obs_camframe,
                right_obs_camframe,
            ],
            rotation_mode=rotation_mode,
        )
    )

    transform_list.extend(
        [
            ConcatKeys(
                key_list=[
                    left_cmd_wristframe,
                    left_cmd_gripper,
                    right_cmd_wristframe,
                    right_cmd_gripper,
                ],
                new_key_name=actions_key,
                delete_old_keys=True,
            ),
            ConcatKeys(
                key_list=[
                    left_obs_camframe,
                    left_obs_gripper,
                    right_obs_camframe,
                    right_obs_gripper,
                ],
                new_key_name=obs_key,
                delete_old_keys=True,
            ),
            DeleteKeys(
                keys_to_delete=[
                    left_cmd_world,
                    right_cmd_world,
                    left_obs_pose,
                    right_obs_pose,
                    left_cmd_camframe,
                    right_cmd_camframe,
                    left_target_world,
                    right_target_world,
                ]
            ),
            NumpyToTensor(
                keys=[
                    actions_key,
                    obs_key,
                ]
            ),
        ]
    )
    return transform_list


def _build_eva_bimanual_transform_list(
    *,
    left_target_world: str = "left_extrinsics_pose",
    right_target_world: str = "right_extrinsics_pose",
    left_cmd_world: str = "left.cmd_ee_pose",
    right_cmd_world: str = "right.cmd_ee_pose",
    left_obs_pose: str = "left.obs_ee_pose",
    right_obs_pose: str = "right.obs_ee_pose",
    left_obs_gripper: str = "left.obs_gripper",
    right_obs_gripper: str = "right.obs_gripper",
    left_cmd_gripper: str = "left.cmd_gripper",
    right_cmd_gripper: str = "right.cmd_gripper",
    left_cmd_camframe: str = "left.cmd_ee_pose_camframe",
    right_cmd_camframe: str = "right.cmd_ee_pose_camframe",
    actions_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
    chunk_length: int = 100,
    stride: int = 1,
    rotation_mode: Literal["euler", "quat", "6D"] = "euler",
) -> list[Transform]:
    """Canonical EVA bimanual transform pipeline used by tests and notebooks."""
    extrinsics = Eva.EXTRINSICS
    left_extrinsics_pose = _matrix_to_xyzwxyz(extrinsics["left"][None, :])[0]
    right_extrinsics_pose = _matrix_to_xyzwxyz(extrinsics["right"][None, :])[0]
    left_extra_batch_key = {"left_extrinsics_pose": left_extrinsics_pose}
    right_extra_batch_key = {"right_extrinsics_pose": right_extrinsics_pose}

    transform_list = [
        ActionChunkCoordinateFrameTransform(
            target_world=left_target_world,
            chunk_world=left_cmd_world,
            transformed_key_name=left_cmd_camframe,
            extra_batch_key=left_extra_batch_key,
            mode="xyzwxyz",
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=right_target_world,
            chunk_world=right_cmd_world,
            transformed_key_name=right_cmd_camframe,
            extra_batch_key=right_extra_batch_key,
            mode="xyzwxyz",
        ),
        PoseCoordinateFrameTransform(
            target_world=left_target_world,
            pose_world=left_obs_pose,
            transformed_key_name=left_obs_pose,
            mode="xyzwxyz",
        ),
        PoseCoordinateFrameTransform(
            target_world=right_target_world,
            pose_world=right_obs_pose,
            transformed_key_name=right_obs_pose,
            mode="xyzwxyz",
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=left_cmd_camframe,
            output_action_key=left_cmd_camframe,
            stride=stride,
            mode="xyzwxyz",
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=right_cmd_camframe,
            output_action_key=right_cmd_camframe,
            stride=stride,
            mode="xyzwxyz",
        ),
        InterpolateLinear(
            new_chunk_length=chunk_length,
            action_key=left_cmd_gripper,
            output_action_key=left_cmd_gripper,
            stride=stride,
        ),
        InterpolateLinear(
            new_chunk_length=chunk_length,
            action_key=right_cmd_gripper,
            output_action_key=right_cmd_gripper,
            stride=stride,
        ),
    ]

    transform_list.extend(
        transforms_for_rotation_mode(
            keys=[
                left_cmd_camframe,
                right_cmd_camframe,
                left_obs_pose,
                right_obs_pose,
            ],
            rotation_mode=rotation_mode,
        )
    )

    transform_list.extend(
        [
            ConcatKeys(
                key_list=[
                    left_cmd_camframe,
                    left_cmd_gripper,
                    right_cmd_camframe,
                    right_cmd_gripper,
                ],
                new_key_name=actions_key,
                delete_old_keys=True,
            ),
            ConcatKeys(
                key_list=[
                    left_obs_pose,
                    left_obs_gripper,
                    right_obs_pose,
                    right_obs_gripper,
                ],
                new_key_name=obs_key,
                delete_old_keys=True,
            ),
            DeleteKeys(
                keys_to_delete=[
                    left_cmd_world,
                    right_cmd_world,
                    left_target_world,
                    right_target_world,
                ]
            ),
            NumpyToTensor(
                keys=[
                    actions_key,
                    obs_key,
                ]
            ),
        ]
    )
    return transform_list


# Where the arc tokenizer stashes the untokenized chunk it consumed, so
# validation can score against the real thing instead of a reconstruction.
# Nothing trains on it. It is not declared in any keymap, but MultiDataset's
# `_infer_key_type` classifies post-transform keys by name and anything starting
# with "actions" is inferred to be an action key -- so it does get normalized on
# the way in, and the evaluator's `norm_stats.unnormalize` puts it back in
# metres. Renaming it to something not starting with "actions" would skip both
# halves; keep the two in step either way.
UNTOKENIZED_ACTION_KEY = "actions_cartesian_untokenized"


def _append_arc_tokenizer(
    transform_list: list[Transform],
    *,
    min_distance_unit: float,
    resampled_vector_length: int,
    rotation_mode: Literal["euler", "quat", "6D"] = "euler",
    dt: float | None = None,
    action_key: str = "actions_cartesian",
    preserve_action_key: str | None = UNTOKENIZED_ACTION_KEY,
) -> list[Transform]:
    """Splice the arc-length tokenizer in before the final NumpyToTensor.

    The tokenizer works on numpy arrays, so it has to run before the cast;
    NumpyToTensor then converts the (M+1, 14) result to a torch tensor.

    ``rotation_mode`` must be ``euler``: the tokenizer's chunk layout is a
    hard-coded 14D ``[xyz(3), ypr(3), grip(1)] x 2``, and it SLERPs through
    the ypr slots. quat (16D) and 6D (20D) chunks are rejected here rather
    than at the first batch, where the shape check fires deep inside a run.
    """
    if rotation_mode != "euler":
        raise ValueError(
            "the arc-length tokenizer only supports rotation_mode='euler' "
            f"(its chunk layout is 14D [xyz, ypr, grip] x 2); got {rotation_mode!r}"
        )
    from egomimic.rldb.zarr.arc_length_tokenizer import (
        TokenizeBimanualArcLengthCartesian,
    )

    kwargs = {} if dt is None else {"dt": float(dt)}
    tokenize = TokenizeBimanualArcLengthCartesian(
        action_key=action_key,
        output_action_key=action_key,
        min_distance_unit=float(min_distance_unit),
        resampled_vector_length=int(resampled_vector_length),
        preserve_action_key=preserve_action_key,
        **kwargs,
    )
    for i in range(len(transform_list) - 1, -1, -1):
        if isinstance(transform_list[i], NumpyToTensor):
            # The preserved chunk has to ride through the same cast as
            # everything else or it reaches the collate fn as a numpy array.
            if preserve_action_key is not None:
                transform_list[i].keys = list(transform_list[i].keys) + [
                    preserve_action_key
                ]
            return transform_list[:i] + [tokenize] + transform_list[i:]
    return transform_list + [tokenize]
