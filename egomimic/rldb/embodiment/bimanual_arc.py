"""Configurable bimanual Cartesian chunks and ARC timing targets.

Dataset selection and experimental parameters belong in Hydra YAML.
"""

from __future__ import annotations

from egomimic.rldb.embodiment.human import (
    Human,
    _build_human_cartesian_eef_frame_transform_list,
    _pad_human_cartesian_gripper,
)
from egomimic.rldb.zarr.e1_arc_tokenizer import CopyKeyRows, TokenizeBimanualArcLengthE1

VARIANTS = (
    "time",
    "arcmean",
    "arcvel",
    "arclogdur",
    "arcdur",
    "arcdurhyb",
    "arcvelhyb",
    "arcdurtri",
    "arcveltri",
    "arcdurevt",
    "arcvelevt",
    "arcdurref",
    "arcvelref",
)
VELOCITY_MODES = {
    "arcmean": "mean",
    "arcvel": "profile",
    "arclogdur": "logdur",
    "arcdur": "dur",
    "arcdurhyb": "durhyb",
    "arcvelhyb": "profhyb",
    "arcdurtri": "durtri",
    "arcveltri": "proftri",
    "arcdurevt": "durevt",
    "arcvelevt": "stkevt",
    "arcdurref": "durref",
    "arcvelref": "stkref",
}


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
    speed_smooth_frames: int = 7,
    velocity_norm: str = "path",
    progress_smooth_hz: float | None = None,
    embodiment: str = "human",
    fixed_spacing: bool = False,
    source_fps: float = 30.0,
    yam_source_frames: int | None = None,
    rotation_distance_unit: float = 6.283185307179586,  # hybrid / tri / evt / ref variants only; see e1_arc_tokenizer
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
        yam_chunk_length = (
            (yam_source_frames if variant == "time" else None)
            if yam_source_frames is not None
            else int(chunk_length)
        )
        yam_stride = 1 if yam_source_frames is not None else int(stride)
        tl = _build_yam_bimanual_eef_frame_transform_list(
            stride=yam_stride,
            rotation_mode=rotation_mode,
            chunk_length=yam_chunk_length,
        )
    else:
        raise ValueError(f"embodiment must be 'human' or 'yam', got {embodiment!r}")
    tl.append(CopyKeyRows("actions_cartesian", "actions_time", int(time_rows)))
    if variant != "time":
        tl.append(
            TokenizeBimanualArcLengthE1(
                action_key="actions_cartesian",
                output_action_key="actions_cartesian",
                min_distance_unit=float(min_distance_unit),
                resampled_vector_length=int(resampled_vector_length),
                dt=(
                    1.0 / source_fps
                    if embodiment == "yam" and yam_source_frames is not None
                    else float(stride) / source_fps
                ),
                velocity_norm=velocity_norm,
                velocity_mode=VELOCITY_MODES[variant],
                speed_smooth_frames=int(speed_smooth_frames),
                progress_smooth_hz=progress_smooth_hz,
                fixed_spacing=bool(fixed_spacing),
                rotation_distance_unit=float(rotation_distance_unit),
            )
        )
    return tl
