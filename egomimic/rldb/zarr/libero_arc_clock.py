"""LIBERO OSC geometry with the canonical PR #273 ARC supports and clocks."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from egomimic.rldb.zarr.arc_clock_bridge import SingleArmArcClockBridge
from egomimic.rldb.zarr.libero_arc import (
    LiberoArcCodec,
    _from_rotation6d,
    _integrate_commands,
    _rotation6d,
)


class LiberoArcClockCodec(SingleArmArcClockBridge, LiberoArcCodec):
    """[XYZ, rotation6d, gripper | one, two or three canonical timing columns].

    Native actions remain seven-dimensional OSC commands. The represented command
    path is decoded in full; the existing LIBERO evaluator still executes 16 actions.
    """

    def __init__(
        self,
        *,
        clock_mode="duration",
        streams="full_separation",
        num_waypoints=36,
        horizon=32,
        dt=0.05,
        translation_scale=0.05,
        rotation_scale=0.5,
        max_translation=1.6,
        max_rotation_degrees=384.0,
        **clock_kwargs,
    ):
        LiberoArcCodec.__init__(
            self,
            num_waypoints=num_waypoints,
            horizon=horizon,
            dt=dt,
            translation_scale=translation_scale,
            rotation_scale=rotation_scale,
            max_translation=max_translation,
            max_rotation_degrees=max_rotation_degrees,
        )
        SingleArmArcClockBridge.__init__(
            self,
            shape_dim=10,
            num_waypoints=num_waypoints,
            horizon=horizon,
            dt=dt,
            clock_mode=clock_mode,
            streams=streams,
            min_distance_unit=max_translation,
            rotation_distance_unit=np.deg2rad(max_rotation_degrees),
            **clock_kwargs,
        )
        self.mode = "cartesian_clock"
        self.decode_pose_horizon = self.horizon + 1

    def encode(self, actions):
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (self.horizon, 7) or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite ({self.horizon},7) LIBERO actions")
        xyz, rotation = _integrate_commands(
            actions, self.translation_scale, self.rotation_scale
        )
        raw = np.zeros((self.horizon + 1, 14))
        raw[:, :3] = xyz
        raw[:, 3:6] = rotation.as_euler("ZYX")
        raw[:, 6] = np.r_[actions[0, 6], actions[:, 6]]
        canonical = self._encode_cartesian(raw)
        return np.column_stack(
            (
                canonical[:, :3],
                _rotation6d(Rotation.from_euler("ZYX", canonical[:, 3:6])),
                canonical[:, 6],
                canonical[:, self.timing_columns],
            )
        ).astype(np.float32)

    def _unpack(self, token):
        value = self._validate(token)
        canonical = self._empty_cartesian_token(value)
        canonical[:, :3] = value[:, :3]
        canonical[:, 3:6] = _from_rotation6d(value[:, 3:9]).as_euler("ZYX")
        canonical[:, 6] = value[:, 9]
        return canonical

    def decode(self, token):
        poses = self.cartesian.detokenize(self._unpack(token), self.decode_pose_horizon)
        rotation = Rotation.from_euler("ZYX", poses[:, 3:6])
        return np.column_stack(
            (
                np.diff(poses[:, :3], axis=0) / self.translation_scale,
                (rotation[1:] * rotation[:-1].inv()).as_rotvec() / self.rotation_scale,
                poses[1:, 6],
            )
        ).astype(np.float32)

    def token_scale(self):
        """Physical scaling, including seconds (not rates) in velocity row zero."""
        scale = np.ones(self.shape)
        scale[:, :3] = self.translation_scale * self.horizon
        hold = self.cartesian.hold_window_frames * self.dt
        scale[:, self.shape_dim :] = max(hold, self.horizon * self.dt)
        if self.clock_mode == "velocity":
            bounds = [
                np.sqrt(3) * self.translation_scale / self.dt,
                np.sqrt(3) * self.rotation_scale / self.dt,
                2.0 / self.dt,
            ]
            if self.streams == "projection":
                bounds[0] += (
                    np.linalg.norm(self.cartesian.projection_offset_m) * bounds[1]
                )
            scale[1:, self.shape_dim :] = bounds[: self.action_dim - self.shape_dim]
        return scale

    def representation_context(self):
        return {
            **super().representation_context(),
            "geometry": "libero_osc_xyz_rotation6d_gripper",
            "token_scaling": "physical_per_row_first_delay_v1",
        }
