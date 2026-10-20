"""Bimanual raw-window transforms for baseline and canonical ARC targets."""

import numpy as np

from egomimic.rldb.embodiment.human import (
    Human,
    _build_human_cartesian_eef_frame_transform_list,
    _pad_human_cartesian_gripper,
)
from egomimic.rldb.zarr.arc_length_tokenizer import TokenizeBimanualArcLengthCartesian

VARIANTS = ("time", "arc")


class CopyKeyRows:
    def __init__(self, src, dst, n_rows):
        self.src, self.dst, self.n_rows = src, dst, int(n_rows)

    def transform(self, batch):
        value = np.asarray(batch[self.src])
        if len(value) == 0 or self.n_rows < 1:
            raise ValueError("Cannot preserve an empty source or nonpositive horizon")
        out = value[: self.n_rows].copy()
        if len(out) < self.n_rows:
            out = np.concatenate(
                (out, np.repeat(out[-1:], self.n_rows - len(out), axis=0))
            )
        batch[self.dst] = out
        return batch


def get_keymap(
    horizon: int,
    keymap_mode: str = "cartesian",
    embodiment: str = "human",
    drop_wrist_images: bool = True,
    yam_source_frames: int | None = None,
    **kwargs,
):
    """Build a Cartesian keymap, preserving declarative distance horizons.

    ``yam_source_frames`` explicitly pins a YAM Cartesian source window for
    stationery recipes. Otherwise the caller's horizon remains unchanged.
    """
    if yam_source_frames is not None:
        if type(yam_source_frames) is not int or yam_source_frames < 1:
            raise ValueError("yam_source_frames must be a positive integer")
        if embodiment != "yam" or keymap_mode != "cartesian":
            raise ValueError("yam_source_frames applies only to YAM cartesian windows")
    if embodiment == "human":
        key_map = Human.get_keymap(keymap_mode, **kwargs)
    elif embodiment == "yam":
        from egomimic.rldb.embodiment.yam import Yam

        key_map = Yam.get_keymap(keymap_mode, **kwargs)
        if drop_wrist_images:
            key_map = {
                k: v
                for k, v in key_map.items()
                if v.get("zarr_key") not in ("images.left_wrist", "images.right_wrist")
            }
    else:
        raise ValueError(f"embodiment must be 'human' or 'yam', got {embodiment!r}")
    for spec in key_map.values():
        if "horizon" not in spec:
            continue
        # A distance horizon is a resolver contract, not an integer frame count.
        if isinstance(spec["horizon"], dict):
            continue
        spec["horizon"] = int(
            horizon if yam_source_frames is None else yam_source_frames
        )
    return key_map


def get_transform_list(
    variant: str,
    chunk_length: int,
    time_rows: int = 100,
    min_distance_unit: float = 0.40,
    resampled_vector_length: int = 100,
    stride: int = 1,
    rotation_mode: str = "euler",
    embodiment: str = "human",
    source_fps: float = 30.0,
    yam_source_frames: int | None = None,
    rotation_distance_unit: float = 1.7453292519943295,
    clock_mode: str = "velocity",
    streams: str = "full_separation",
    gripper_distance_unit: float | None = None,
    gripper_stream_mode: str = "events",
):
    if not 0 < source_fps < float("inf"):
        raise ValueError("source_fps must be finite and positive")
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if embodiment == "human":
        tl = _build_human_cartesian_eef_frame_transform_list(
            stride=int(stride),
            rotation_mode=rotation_mode,
            chunk_length=int(chunk_length),
        )
        tl = _pad_human_cartesian_gripper(tl, rotation_mode=rotation_mode)
    elif embodiment == "yam":
        from egomimic.rldb.embodiment.yam import (
            _build_yam_bimanual_eef_frame_transform_list,
        )

        # Stationery pins native source frames; older main recipes retain their
        # configured interpolation horizon and stride.
        if yam_source_frames is not None and (
            type(yam_source_frames) is not int or yam_source_frames < 1
        ):
            raise ValueError("yam_source_frames must be a positive integer")
        # Native windows are already padded by the loader. TIME must also
        # preserve these rows exactly, as in the original stationery source.
        yam_chunk_length = None if yam_source_frames is not None else int(chunk_length)
        yam_stride = 1 if yam_source_frames is not None else int(stride)
        tl = _build_yam_bimanual_eef_frame_transform_list(
            stride=yam_stride,
            rotation_mode=rotation_mode,
            chunk_length=yam_chunk_length,
        )
    else:
        raise ValueError(f"embodiment must be 'human' or 'yam', got {embodiment!r}")
    tl.append(CopyKeyRows("actions_cartesian", "actions_time", int(time_rows)))
    if variant == "arc":
        tl.append(
            TokenizeBimanualArcLengthCartesian(
                min_distance_unit=min_distance_unit,
                rotation_distance_unit=rotation_distance_unit,
                resampled_vector_length=resampled_vector_length,
                clock_mode=clock_mode,
                streams=streams,
                gripper_distance_unit=gripper_distance_unit,
                gripper_stream_mode=gripper_stream_mode,
                dt=1 / source_fps
                if embodiment == "yam" and yam_source_frames is not None
                else stride / source_fps,
            )
        )
    return tl
