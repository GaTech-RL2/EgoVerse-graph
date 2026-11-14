"""Main arrival clocks on LIBERO's relative OSC command geometry.

The anchor is the command-space origin (zero displacement, identity rotation,
neutral gripper command), not a measured Cartesian pose. OSC geometry remains
relative; no metric speed bound or projection is applied. Scalar/component
ablations retain their declared grouping and rotation representation.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from egomimic.rldb.zarr.cartesian_arc import (
    TokenizeBimanualArcLengthCartesian,
    absolute_times_to_seconds,
)
from egomimic.rldb.zarr.libero_arc import _from_rotation6d, _rotation6d
from egomimic.rldb.zarr.libero_arc_grouped import LiberoArcGroupedCodec

CONTRACT = "libero_osc_main_arrival_v2_unbounded"
MAIN_COMMIT = "c7e110461650d09a88b2c2dff8f02db54b1deace"


class LiberoArcArrivalCodec(LiberoArcGroupedCodec):
    def __init__(self, mode="dur", stream_spec=None, **kwargs):
        super().__init__(mode=mode, stream_spec=stream_spec or {}, **kwargs)
        self._main = TokenizeBimanualArcLengthCartesian(
            action_key="actions",
            resampled_vector_length=self.num_waypoints,
            min_distance_unit=self.max_translation or 1e30,
            rotation_distance_unit=np.deg2rad(self.max_rotation_degrees or 1e30),
            clock_mode="duration" if mode == "dur" else "velocity",
            dt=self.dt,
        )

    def representation_context(self):
        return {
            **super().representation_context(),
            "version": 2,
            "clock_contract": CONTRACT,
            "upstream_arc_commit": MAIN_COMMIT,
            "source_time_origin": "next_frame",
            "anchor": "relative_osc_origin",
            "duration_timing": "absolute_arrival_seconds",
            "velocity_timing": "incoming_segment_rates",
            "translation_speed_limit": None,
            "projection": False,
        }

    def _anchor(self):
        row = np.zeros(self.action_dim)
        if self.rotation_representation == "so3":
            row[3:9] = _rotation6d(Rotation.identity())
        return row

    def _supports(self, future, group):
        _, kind, _ = group
        if kind == "grip":
            return self._main._gripper_event_frames(future[:, self.grip_column]).astype(
                float
            )
        vectors = self._vectors(future, group)
        cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(vectors, axis=-1))]
        budget = (
            self.max_translation
            if kind == "xyz"
            else None
            if self.max_rotation_degrees is None
            else np.deg2rad(self.max_rotation_degrees)
        )
        lower, alpha = self._main._resample_coordinates(cumulative, budget or 1e30)
        return lower + alpha

    def encode(self, actions):
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (self.horizon, 7) or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite ({self.horizon},7) LIBERO actions")
        future = np.zeros((self.horizon, self.action_dim))
        future[:, :3] = np.cumsum(actions[:, :3] * self.translation_scale, axis=0)
        if self.rotation_representation == "so3":
            rotations, previous = [], Rotation.identity()
            for action in actions:
                previous = (
                    Rotation.from_rotvec(action[3:6] * self.rotation_scale) * previous
                )
                rotations.append(previous.as_quat())
            rotation = Rotation.from_quat(rotations)
            future[:, 3:9] = _rotation6d(rotation)
        else:
            future[:, 3:6] = np.cumsum(actions[:, 3:6] * self.rotation_scale, axis=0)
        future[:, self.grip_column] = actions[:, 6]
        result = np.zeros((self.num_waypoints, self.action_dim))
        frames = np.arange(self.horizon)
        supports = {}
        for group in self.groups:
            name, kind, axes = group
            selected = self._supports(future, group)
            supports[name] = selected
            if kind == "so3":
                result[:, 3:9] = _rotation6d(Slerp(frames, rotation)(selected))
            elif kind == "grip":
                result[:, self.grip_column] = future[
                    selected.astype(int), self.grip_column
                ]
            else:
                for axis in axes:
                    column = axis + (3 if kind == "angular_driver" else 0)
                    result[:, column] = np.interp(selected, frames, future[:, column])
            if name == "xyz0" and self.gripper != "separate":
                # Main's coupled gripper is interpolated on the primary clock.
                result[:, self.grip_column] = np.interp(
                    selected, frames, future[:, self.grip_column]
                )
        anchored = np.vstack((self._anchor(), result))
        for group in self.groups:
            name, _, _ = group
            times = (supports[name] + 1) * self.dt
            width = self.timing_slices[name].stop - self.timing_slices[name].start
            if self.mode == "dur":
                timing = np.repeat(times[:, None], width, axis=1)
            else:
                vectors = self._vectors(anchored, group)
                distance = (
                    np.linalg.norm(vectors, axis=-1, keepdims=True)
                    if self.timing == "group"
                    else vectors
                )
                elapsed = np.diff(np.r_[0.0, times])[:, None]
                timing = np.divide(
                    distance,
                    elapsed,
                    out=np.zeros_like(distance),
                    where=elapsed > 1e-12,
                )
            result[:, self.timing_slices[name]] = timing
        return result.astype(np.float32)

    def _intervals(self, tokens, group):
        name, _, _ = group
        vectors = self._vectors(np.vstack((self._anchor(), tokens)), group)
        stored = tokens[:, self.timing_slices[name]]
        if self.clock_mode == "group":
            distance = np.linalg.norm(vectors, axis=-1, keepdims=True)
            if self.timing == "component":
                stored = (
                    stored.mean(axis=-1, keepdims=True)
                    if self.mode == "dur"
                    else np.linalg.norm(stored, axis=-1, keepdims=True)
                )
        else:
            distance = np.abs(vectors)
        if self.mode == "dur":
            return np.column_stack(
                [
                    absolute_times_to_seconds(
                        stored[:, i], distance[:, i], max_speed=None
                    )
                    for i in range(stored.shape[1])
                ]
            )
        return np.column_stack(
            [
                self._main._intervals(distance[:, i], stored[:, i])
                for i in range(stored.shape[1])
            ]
        )

    def clocks(self, tokens):
        value = self._validate(tokens)
        return {
            g[0]: np.vstack(
                (
                    np.zeros((1, self._intervals(value, g).shape[1])),
                    np.cumsum(self._intervals(value, g), axis=0),
                )
            )
            for g in self.groups
        }

    def represented_seconds(self, tokens):
        values = self._validate(tokens)
        clocks = self.clocks(values)
        anchored = np.vstack((self._anchor(), values))
        ends = []
        for group in self.groups:
            active = np.any(np.abs(self._vectors(anchored, group)) > 1e-12, axis=0)
            if self.clock_mode == "group":
                active = np.array([active.any()])
            ends.extend(clocks[group[0]][-1, active].tolist())
        if ends:
            return min(ends)
        return min(c[-1].min() for c in clocks.values()) if self.mode == "dur" else 0.0

    def _indices(self, intervals, queries):
        return self._main._clock_indices(intervals, queries)[:3]

    def decode(self, tokens):
        tokens = self._validate(tokens)
        values = np.vstack((self._anchor(), tokens))
        queries = np.arange(self.horizon + 1) * self.dt
        intervals = {g[0]: self._intervals(tokens, g) for g in self.groups}
        clocks = {
            name: np.vstack((np.zeros((1, d.shape[1])), np.cumsum(d, axis=0)))
            for name, d in intervals.items()
        }
        xyz = np.zeros((self.horizon + 1, 3))
        for name, kind, axes in self.groups:
            if kind != "xyz":
                continue
            for i, axis in enumerate(axes):
                lo, hi, a = self._indices(
                    intervals[name][:, i if self.clock_mode == "component" else 0],
                    queries[1:],
                )
                xyz[1:, axis] = values[lo, axis] + a * (
                    values[hi, axis] - values[lo, axis]
                )
        if self.rotation_representation == "so3" and self.clock_mode == "group":
            lo, hi, a = self._indices(intervals["rot0"][:, 0], queries[1:])
            r = _from_rotation6d(values[:, 3:9])
            delta = r[hi] * r[lo].inv()
            future = Rotation.from_rotvec(delta.as_rotvec() * a[:, None]) * r[lo]
            rotation = Rotation.from_quat(
                np.vstack((Rotation.identity().as_quat(), future.as_quat()))
            )
        else:
            rotation = self._angular_path(values, clocks, queries)
        grip_name = "grip" if self.gripper == "separate" else "xyz0"
        if self.gripper == "separate":
            clock = clocks[grip_name][:, 0]
            finite = clock[np.isfinite(clock)]
            tolerance = 8 * np.finfo(np.float32).eps * max(1.0, finite[-1])
            idx = (
                np.searchsorted(clock, queries[1:] + tolerance, side="right") - 1
            ).clip(0, len(values) - 1)
            grip = values[idx, self.grip_column]
        else:
            lo, hi, a = self._indices(intervals[grip_name][:, 0], queries[1:])
            grip = values[lo, self.grip_column] + a * (
                values[hi, self.grip_column] - values[lo, self.grip_column]
            )
        return np.column_stack(
            (
                np.diff(xyz, axis=0) / self.translation_scale,
                (rotation[1:] * rotation[:-1].inv()).as_rotvec() / self.rotation_scale,
                grip,
            )
        ).astype(np.float32)
