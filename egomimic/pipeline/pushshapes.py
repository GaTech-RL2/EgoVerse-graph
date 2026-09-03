"""Planar action decoders shared by metrics and downstream consumers."""

from __future__ import annotations

import numpy as np
import torch

from egomimic.rldb.zarr.planar_arc import PLANAR_ACTION_DIM


def _common5_to_native(actions, native_action_dim: int):
    if native_action_dim not in (2, 3, 4):
        raise ValueError("native_action_dim must be 2, 3, or 4")
    if actions.shape[-1] != PLANAR_ACTION_DIM:
        raise ValueError(
            f"expected common Planar width {PLANAR_ACTION_DIM}, got {actions.shape}"
        )
    if torch.is_tensor(actions):
        theta = torch.atan2(actions[..., 3], actions[..., 2]).unsqueeze(-1)
        native = torch.cat((actions[..., :2], theta, actions[..., 4:5]), dim=-1)
    else:
        value = np.asarray(actions)
        theta = np.arctan2(value[..., 3], value[..., 2])[..., None]
        native = np.concatenate((value[..., :2], theta, value[..., 4:5]), axis=-1)
    return native[..., :native_action_dim]


class PlanarCommon5NativeDecoder:
    """Decode a fixed-rate common-five chunk without changing its timing."""

    preserves_decoded_timing = True

    def __init__(self, action_horizon: int, native_action_dim: int):
        self.action_horizon = int(action_horizon)
        self.native_action_dim = int(native_action_dim)
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")

    def decode(self, actions, context: dict | None = None):
        del context
        value = (
            actions.unsqueeze(0)
            if torch.is_tensor(actions) and actions.ndim == 2
            else actions
        )
        value = (
            np.asarray(actions)[None]
            if not torch.is_tensor(actions) and np.asarray(actions).ndim == 2
            else value
        )
        expected = (self.action_horizon, PLANAR_ACTION_DIM)
        if value.ndim != 3 or tuple(value.shape[1:]) != expected:
            raise ValueError(
                f"expected (B, {expected[0]}, {expected[1]}), got {value.shape}"
            )
        return _common5_to_native(value, self.native_action_dim)

    __call__ = decode


class PlanarArcWaypointZeroNativeDecoder:
    """Decode the anchored first waypoint from a Planar arc token."""

    preserves_decoded_timing = True
    action_horizon = 1

    def __init__(self, resampled_vector_length: int, native_action_dim: int):
        self.num_waypoints = int(resampled_vector_length)
        self.native_action_dim = int(native_action_dim)
        if self.num_waypoints < 2:
            raise ValueError("resampled_vector_length must be at least two")

    def decode(self, actions, context: dict | None = None):
        del context
        value = (
            actions.unsqueeze(0)
            if torch.is_tensor(actions) and actions.ndim == 2
            else actions
        )
        value = (
            np.asarray(actions)[None]
            if not torch.is_tensor(actions) and np.asarray(actions).ndim == 2
            else value
        )
        expected = (self.num_waypoints + 1, PLANAR_ACTION_DIM)
        if value.ndim != 3 or tuple(value.shape[1:]) != expected:
            raise ValueError(
                f"expected (B, {expected[0]}, {expected[1]}), got {value.shape}"
            )
        return _common5_to_native(value[:, :1], self.native_action_dim)

    __call__ = decode
