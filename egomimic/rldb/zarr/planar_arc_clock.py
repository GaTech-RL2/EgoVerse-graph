"""Native planar future targets and explicit measured-current-pose anchors."""

from __future__ import annotations

import numpy as np

from egomimic.rldb.zarr.arc_clock_bridge import SingleArmArcClockBridge


def current_action_from_state(state, native_action_dim, state_indices=None):
    """Extract measured actuator state, never object pose or a future command."""
    state = np.asarray(state, dtype=np.float64)
    if state_indices is None:
        if native_action_dim not in (2, 3):
            raise ValueError(
                "Gripper embodiments require explicit measured-state indices"
            )
        state_indices = tuple(range(native_action_dim))
    indices = np.asarray(state_indices)
    if (
        state.ndim < 1
        or indices.shape != (native_action_dim,)
        or not np.issubdtype(indices.dtype, np.integer)
        or np.any(indices < 0)
        or np.any(indices >= state.shape[-1])
    ):
        raise ValueError("Invalid measured-current-pose state mapping")
    value = state[..., indices]
    if not np.isfinite(value).all():
        raise ValueError("Measured current pose must be finite")
    return value.copy()


class PlanarArcClockCodec(SingleArmArcClockBridge):
    """[x,y,cos(theta),sin(theta),gripper | canonical clocks].

    Positions and distances stay in native units, rates in native units/second.
    Duration correction enforces monotonicity without a physical speed bound.
    """

    def __init__(
        self,
        *,
        native_action_dim=4,
        num_waypoints=56,
        horizon=80,
        dt=1 / 30,
        clock_mode="duration",
        streams="full_separation",
        min_distance_unit=80.0,
        rotation_distance_unit=np.deg2rad(26),
        **clock_kwargs,
    ):
        if native_action_dim not in (2, 3, 4):
            raise ValueError("native_action_dim must be 2, 3, or 4")
        self.native_action_dim = int(native_action_dim)
        super().__init__(
            shape_dim=5,
            num_waypoints=num_waypoints,
            horizon=horizon,
            dt=dt,
            clock_mode=clock_mode,
            streams=streams,
            min_distance_unit=min_distance_unit,
            rotation_distance_unit=rotation_distance_unit,
            **clock_kwargs,
        )

    def _canonical_pose(self, actions):
        value = np.asarray(actions, dtype=np.float64)
        if (
            value.ndim != 2
            or len(value) < 1
            or value.shape[1] not in (self.native_action_dim, 5)
            or not np.isfinite(value).all()
        ):
            raise ValueError("Expected finite native or common-five planar actions")
        raw = np.zeros((len(value), 14))
        raw[:, :2] = value[:, :2]
        if self.native_action_dim >= 3:
            raw[:, 3] = (
                np.arctan2(value[:, 3], value[:, 2])
                if value.shape[1] == 5
                else value[:, 2]
            )
        if self.native_action_dim == 4:
            raw[:, 6] = value[:, -1]
        return raw

    def _anchor(self, initial_action):
        value = np.asarray(initial_action, dtype=np.float64)
        if value.shape != (self.native_action_dim,):
            raise ValueError("Supply one measured native current pose explicitly")
        return self._canonical_pose(value[None])[0]

    def encode(self, actions, *, initial_action):
        raw = self._canonical_pose(actions)
        canonical = self._encode_cartesian(raw, self._anchor(initial_action))
        return np.column_stack(
            (
                canonical[:, :2],
                np.cos(canonical[:, 3]),
                np.sin(canonical[:, 3]),
                canonical[:, 6],
                canonical[:, self.timing_columns],
            )
        )

    def _unpack(self, token):
        value = self._validate(token)
        canonical = self._empty_cartesian_token(value)
        canonical[:, :2] = value[:, :2]
        if self.native_action_dim >= 3:
            canonical[:, 3] = np.arctan2(value[:, 3], value[:, 2])
        if self.native_action_dim == 4:
            canonical[:, 6] = value[:, 4]
        return canonical

    def decode(self, token, *, initial_action):
        anchor = self._anchor(initial_action)
        poses = self._decode_cartesian(token, anchor)
        theta = np.unwrap(np.r_[anchor[3], poses[:, 3]])[1:]
        return np.column_stack((poses[:, :2], theta, poses[:, 6]))[
            :, : self.native_action_dim
        ]

    def waypoint_clocks(self, token, *, initial_action):
        clocks = super().waypoint_clocks(
            token, initial_action=self._anchor(initial_action)
        )
        if self.native_action_dim < 4:
            clocks.pop("gripper", None)
        if self.native_action_dim < 3:
            clocks.pop("rotation", None)
        return clocks

    def representation_context(self):
        return {
            **super().representation_context(),
            "geometry": "planar_xy_cos_sin_gripper",
            "translation_units": "native",
            "native_action_dim": self.native_action_dim,
        }


class TokenizePlanarArcClock:
    def __init__(
        self,
        action_key="actions",
        output_action_key=None,
        state_key="state_agent_obj",
        state_indices=None,
        **codec_kwargs,
    ):
        self.action_key = action_key
        self.output_action_key = output_action_key or action_key
        self.codec = PlanarArcClockCodec(**codec_kwargs)
        self.state_key, self.state_indices = state_key, state_indices

    def transform(self, batch):
        value = np.asarray(batch[self.action_key])
        dtype = value.dtype if np.issubdtype(value.dtype, np.floating) else np.float32
        state = np.asarray(batch[self.state_key])
        if state.ndim != 2 or len(state) == 0:
            raise ValueError("Arrival ARC requires an explicit observation history")
        anchor = current_action_from_state(
            state[-1], self.codec.native_action_dim, self.state_indices
        )
        batch[self.output_action_key] = self.codec.encode(
            value, initial_action=anchor
        ).astype(dtype)
        return batch


def get_planar_clock_transform_list(
    keys=None, raw_action_horizon=80, action_target_offset=1, **codec_kwargs
):
    from egomimic.rldb.embodiment.pushshapes import SliceActionTarget

    keys = keys or ["actions"]
    if len(keys) != 1:
        raise ValueError("Planar ARC clock transforms require one action key")
    return [
        SliceActionTarget(keys, int(action_target_offset), int(raw_action_horizon)),
        TokenizePlanarArcClock(
            action_key=keys[0], horizon=raw_action_horizon, **codec_kwargs
        ),
    ]
