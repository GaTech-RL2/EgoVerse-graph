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
    """Build a cartesian keymap for the time or distance ARC variants.

    Human retains the caller's fixed horizon. YAM baseline is always a raw
    100-frame window; YAM ARC carries its distance/source-buffer specification
    through unchanged so the resolver chooses the per-sample source length.
    """
    yam_cls = None
    if embodiment == "human":
        key_map = Human.get_keymap(keymap_mode, **kwargs)
    elif embodiment == "yam":
        from egomimic.rldb.embodiment.yam import Yam

        yam_cls = Yam
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
        # YAM's ARC keymap carries a declarative distance horizon.  Do not
        # coerce it to an integer: the leaf resolver must inspect the source
        # poses and choose a per-sample frame count.  Baseline YAM is now
        # explicitly 100 source frames, independent of the legacy E1
        # ``chunk_length=45`` setting.
        if isinstance(spec["horizon"], dict):
            continue
        spec["horizon"] = int(
            yam_cls.ACTION_HORIZON
            if yam_cls is not None and keymap_mode == "cartesian"
            else horizon
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
            Yam,
            _build_yam_bimanual_eef_frame_transform_list,
        )

        # Baseline YAM is a raw 100-frame time chunk. ARC YAM has a
        # distance-resolved variable-length source chunk; leave it untouched
        # so its tokenizer can measure cumulative travel on the original
        # samples. ``chunk_length`` remains the human/legacy setting for other
        # embodiments and is intentionally not mutated globally.
        yam_chunk_length = Yam.ACTION_HORIZON if variant == "time" else None
        # ARC distance is defined on the stored source samples.  YAM is
        # recorded at the native 30 Hz cadence; never subsample it before
        # accumulating distance.  Keep accepting the legacy argument for
        # config compatibility, but it has no effect on YAM ARC.
        yam_stride = 1
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
                dt=(1.0 / source_fps if embodiment == "yam" else float(stride) / source_fps),
                velocity_norm=velocity_norm,
                velocity_mode=VELOCITY_MODES[variant],
                speed_smooth_frames=int(speed_smooth_frames),
                progress_smooth_hz=progress_smooth_hz,
                fixed_spacing=bool(fixed_spacing),
            )
        )
    return tl
