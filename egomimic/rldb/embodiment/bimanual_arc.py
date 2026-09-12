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

VARIANTS = ("time", "arcmean", "arcvel", "arclogdur", "arcdur")
VELOCITY_MODES = {
    "arcmean": "mean",
    "arcvel": "profile",
    "arclogdur": "logdur",
    "arcdur": "dur",
}


def get_keymap(
    horizon: int,
    keymap_mode: str = "cartesian",
    embodiment: str = "human",
    drop_wrist_images: bool = True,
    **kwargs,
):
    """Plain cartesian keymap with every action key's raw window set to ``horizon``."""
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
        if "horizon" in spec:
            spec["horizon"] = int(horizon)
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

        tl = _build_yam_bimanual_eef_frame_transform_list(
            stride=int(stride),
            rotation_mode=rotation_mode,
            chunk_length=int(chunk_length),
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
                dt=float(stride) / source_fps,
                velocity_norm=velocity_norm,
                velocity_mode=VELOCITY_MODES[variant],
                speed_smooth_frames=int(speed_smooth_frames),
                progress_smooth_hz=progress_smooth_hz,
                fixed_spacing=bool(fixed_spacing),
            )
        )
    return tl
