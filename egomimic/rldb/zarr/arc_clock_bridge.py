"""Single-arm geometry adapters for the canonical Cartesian ARC clock contract.

Timing, supports, grippers, holds and clock correction belong to the Cartesian
codec. Adapters change geometry only. They expose no execution_steps override:
benchmark rollout loops retain their fixed native-action replanning cadence.
"""

from __future__ import annotations

import numpy as np

from egomimic.rldb.zarr.arc_length_tokenizer import (
    MAX_SPEED,
    PROJECTION_OFFSET_M,
    STREAM_NAMES,
    TokenizeBimanualArcLengthCartesian,
)


class SingleArmArcClockBridge:
    """Use the left arm of the canonical codec; the unused arm is constant."""

    def __init__(
        self,
        *,
        shape_dim,
        num_waypoints,
        horizon,
        dt,
        clock_mode,
        streams,
        min_distance_unit,
        rotation_distance_unit,
        hold_window_frames=None,
        gripper_stream_mode="events",
        gripper_distance_unit=None,
        projection_offset_m=PROJECTION_OFFSET_M,
        zero_dist_epsilon=1e-6,
    ):
        self.horizon = int(horizon)
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        self.decode_pose_horizon = self.horizon
        self.cartesian = TokenizeBimanualArcLengthCartesian(
            action_key="actions",
            output_action_key="actions",
            resampled_vector_length=num_waypoints,
            dt=dt,
            clock_mode=clock_mode,
            streams=streams,
            min_distance_unit=min_distance_unit,
            rotation_distance_unit=rotation_distance_unit,
            hold_window_frames=(
                self.horizon if hold_window_frames is None else hold_window_frames
            ),
            gripper_stream_mode=gripper_stream_mode,
            gripper_distance_unit=gripper_distance_unit,
            projection_offset_m=projection_offset_m,
            zero_dist_epsilon=zero_dist_epsilon,
        )
        self.num_waypoints = self.cartesian.M
        self.dt = self.cartesian.tokenizer.config.dt
        self.clock_mode, self.streams = clock_mode, streams
        self.shape_dim = int(shape_dim)
        self.action_dim = self.shape_dim + len(STREAM_NAMES[streams])
        self.shape = (self.num_waypoints, self.action_dim)
        self.timing_columns = np.arange(14, self.cartesian.shape[1], 2)

    def _validate(self, token):
        value = np.asarray(token, dtype=np.float64)
        if value.shape != self.shape or not np.isfinite(value).all():
            raise ValueError(f"Expected finite ARC token {self.shape}")
        return value

    def _encode_cartesian(self, raw):
        return self.cartesian.transform({"actions": raw})["actions"]

    def _unpack(self, token):
        """Subclasses convert their stored geometry to canonical (M,14+clocks)."""
        raise NotImplementedError

    def _empty_cartesian_token(self, token):
        value = self._validate(token)
        canonical = np.zeros(self.cartesian.shape, dtype=np.float64)
        canonical[:, self.timing_columns] = value[:, self.shape_dim :]
        return canonical

    def waypoint_clocks(self, token):
        """Exact decoded clocks for diagnostics; this does not choose replanning."""
        value = self.cartesian._native(self._unpack(token))
        clocks = {}
        for index, name in enumerate(STREAM_NAMES[self.streams]):
            distances = self.cartesian._distances(value, 0, name)
            stored = value[:, 14 + 2 * index]
            intervals = self.cartesian._intervals(
                distances, stored[1:], self.decode_pose_horizon
            )
            clocks[name] = {
                "seconds": max(stored[0], 0) + np.r_[0.0, np.cumsum(intervals)],
                "active": bool(np.any(distances > 1e-12)),
            }
        return clocks

    def represented_seconds(self, token):
        ends = [
            clock["seconds"][-1]
            for clock in self.waypoint_clocks(token).values()
            if clock["active"]
        ]
        return min(ends or [self.cartesian.hold_window_frames * self.dt])

    def representation_context(self):
        return {
            "contract": "cartesian_arc_pr273_v1",
            "clock_mode": self.clock_mode,
            "timing_layout": (
                "first_delay_absolute_v1"
                if self.clock_mode == "duration"
                else "first_delay_incoming_v1"
            ),
            "streams": self.streams,
            "layout": "shape_then_timing",
            "shape_dim": self.shape_dim,
            "action_dim": self.action_dim,
            "translation_distance_m": self.cartesian.tokenizer.config.min_distance_unit,
            "rotation_distance_radians": self.cartesian.rotation_distance_unit,
            "hold_window_frames": self.cartesian.hold_window_frames,
            "gripper_stream_mode": self.cartesian.gripper_stream_mode,
            "gripper_distance_unit": self.cartesian.gripper_distance_unit,
            "projection_offset_m": self.cartesian.projection_offset_m.tolist(),
            "zero_dist_epsilon": self.cartesian.zero_dist_epsilon,
            "max_translation_speed_m_s": MAX_SPEED,
            "replanning": "caller_fixed_native_steps",
        }
