"""Loader-side U-Socket local-velocity ARC codec bindings.

Tokenization must run before dataset normalization so distance and rotation
budgets remain in native pixels and radians.  The native decoder runs after
the evaluator unnormalizes predicted ARC tokens.
"""

from __future__ import annotations

import numpy as np
import torch

from egomimic.pipeline.stages_arc import (
    ArcDetokenizeStage,
    ArcDetokenizeStackedStage,
    ArcDetokenizeDurationStage,
)
from egomimic.rldb.zarr.planar_arc import (
    TokenizeUSocketArcVelocity,
    TokenizeUSocketArcVelocityStacked,
    TokenizeUSocketArcVelocityCarry,
    TokenizeUSocketArcDuration,
)


def _arc_prefix(keys, action_target_offset, raw_action_horizon):
    """Drop a_t when the keymap was told to fetch it.

    get_planar_keymap requests action_horizon + action_target_offset steps so
    that the target can be aligned to the LAST observation in the window. The
    ARC factories used to take neither argument, so an offset of 1 vanished
    into **_kwargs: the loader fetched 81 steps and the tokenizer consumed all
    81 beginning at a_t, one frame earlier and one frame longer than the Paper
    DP baseline it was being compared against.
    """
    from egomimic.rldb.embodiment.pushshapes import SliceActionTarget

    if not action_target_offset:
        return []
    return [
        SliceActionTarget(
            keys, start=int(action_target_offset), horizon=int(raw_action_horizon)
        )
    ]


def get_usocket_arc_velocity_transform_list(
    keys: list[str] | None = None,
    min_distance_unit: float = 80.0,
    resampled_vector_length: int = 56,
    dt: float = 1.0 / 30.0,
    rotation_distance_unit: float | None = None,
    action_target_offset: int = 0,
    raw_action_horizon: int = 80,
    **_kwargs,
):
    keys = keys or ["actions"]
    if len(keys) != 1:
        raise ValueError("U-Socket ARC tokenization requires exactly one action key")
    return _arc_prefix(keys, action_target_offset, raw_action_horizon) + [
        TokenizeUSocketArcVelocity(
            action_key=keys[0],
            output_action_key=keys[0],
            min_distance_unit=min_distance_unit,
            resampled_vector_length=resampled_vector_length,
            dt=dt,
            rotation_distance_unit=rotation_distance_unit,
        )
    ]


class USocketArcLocalVelocityNativeDecoder:
    """Decode unnormalized two-stream ARC tokens into fixed-rate actions."""

    preserves_decoded_timing = True

    def __init__(
        self,
        resampled_vector_length: int,
        action_horizon: int,
        native_action_dim: int = 3,
        dt: float = 1.0 / 30.0,
    ):
        self.action_horizon = int(action_horizon)
        self._decoder = ArcDetokenizeStage(
            resampled_vector_length=resampled_vector_length,
            action_horizon=action_horizon,
            native_action_dim=native_action_dim,
            dt=dt,
        )

    def decode(self, actions, context: dict | None = None):
        del context
        is_tensor = torch.is_tensor(actions)
        value = actions if is_tensor else torch.as_tensor(np.asarray(actions))
        squeeze = value.ndim == 2
        if squeeze:
            value = value.unsqueeze(0)
        native = self._decoder.forward({"pred_action": value})["pred_action_native"]
        if squeeze:
            native = native.squeeze(0)
        return native if is_tensor else native.cpu().numpy()

    __call__ = decode


def get_usocket_arc_velocity_stacked_transform_list(
    keys: list[str] | None = None,
    min_distance_unit: float = 80.0,
    resampled_vector_length: int = 56,
    dt: float = 1.0 / 30.0,
    rotation_distance_unit: float | None = None,
    action_target_offset: int = 0,
    raw_action_horizon: int = 80,
    **_kwargs,
):
    """Loader transform emitting the stacked ``[M, 6]`` ARC token."""
    keys = keys or ["actions"]
    if len(keys) != 1:
        raise ValueError("U-Socket ARC tokenization requires exactly one action key")
    return _arc_prefix(keys, action_target_offset, raw_action_horizon) + [
        TokenizeUSocketArcVelocityStacked(
            action_key=keys[0],
            output_action_key=keys[0],
            min_distance_unit=min_distance_unit,
            resampled_vector_length=resampled_vector_length,
            dt=dt,
            rotation_distance_unit=rotation_distance_unit,
        )
    ]


class USocketArcLocalVelocityStackedNativeDecoder(USocketArcLocalVelocityNativeDecoder):
    """Native decoder for the stacked ``[M, 6]`` token."""

    def __init__(
        self,
        resampled_vector_length: int,
        action_horizon: int,
        native_action_dim: int = 3,
        dt: float = 1.0 / 30.0,
    ):
        self.action_horizon = int(action_horizon)
        self._decoder = ArcDetokenizeStackedStage(
            resampled_vector_length=resampled_vector_length,
            action_horizon=action_horizon,
            native_action_dim=native_action_dim,
            dt=dt,
        )


def get_usocket_arc_velocity_carry_transform_list(
    keys: list[str] | None = None,
    min_distance_unit: float = 80.0,
    resampled_vector_length: int = 56,
    dt: float = 1.0 / 30.0,
    rotation_distance_unit: float | None = None,
    action_target_offset: int = 0,
    raw_action_horizon: int = 80,
    **_kwargs,
):
    """Loader transform for the shared-clock (no angular budget) ablation."""
    keys = keys or ["actions"]
    if len(keys) != 1:
        raise ValueError("U-Socket ARC tokenization requires exactly one action key")
    if rotation_distance_unit is not None:
        raise ValueError(
            "the carry variant has no angular budget; set "
            "planar.arc_rotation_distance to null for this experiment"
        )
    return _arc_prefix(keys, action_target_offset, raw_action_horizon) + [
        TokenizeUSocketArcVelocityCarry(
            action_key=keys[0],
            output_action_key=keys[0],
            min_distance_unit=min_distance_unit,
            resampled_vector_length=resampled_vector_length,
            dt=dt,
        )
    ]


def get_usocket_arc_duration_transform_list(
    keys: list[str] | None = None,
    min_distance_unit: float = 80.0,
    resampled_vector_length: int = 56,
    dt: float = 1.0 / 30.0,
    rotation_distance_unit: float | None = None,
    action_target_offset: int = 0,
    raw_action_horizon: int = 80,
    **_kwargs,
):
    """Loader transform emitting the duration-timed stacked token."""
    keys = keys or ["actions"]
    if len(keys) != 1:
        raise ValueError("U-Socket ARC tokenization requires exactly one action key")
    return _arc_prefix(keys, action_target_offset, raw_action_horizon) + [
        TokenizeUSocketArcDuration(
            action_key=keys[0],
            output_action_key=keys[0],
            min_distance_unit=min_distance_unit,
            resampled_vector_length=resampled_vector_length,
            dt=dt,
            rotation_distance_unit=rotation_distance_unit,
        )
    ]


class USocketArcDurationNativeDecoder(USocketArcLocalVelocityNativeDecoder):
    """Native decoder for the duration-timed stacked token."""

    def __init__(
        self,
        resampled_vector_length: int,
        action_horizon: int,
        native_action_dim: int = 3,
        dt: float = 1.0 / 30.0,
    ):
        self.action_horizon = int(action_horizon)
        self._decoder = ArcDetokenizeDurationStage(
            resampled_vector_length=resampled_vector_length,
            action_horizon=action_horizon,
            native_action_dim=native_action_dim,
            dt=dt,
        )
