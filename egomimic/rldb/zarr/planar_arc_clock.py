"""Native planar geometry with the canonical Cartesian ARC clock contract."""

from __future__ import annotations

import numpy as np

from egomimic.rldb.zarr.arc_clock_bridge import SingleArmArcClockBridge


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

    def encode(self, actions):
        value = np.asarray(actions, dtype=np.float64)
        if (
            value.ndim != 2
            or len(value) < 2
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
        canonical = self._encode_cartesian(raw)
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

    def decode(self, token):
        poses = self._decode_cartesian(token)
        return np.column_stack((poses[:, :2], np.unwrap(poses[:, 3]), poses[:, 6]))[
            :, : self.native_action_dim
        ]

    def waypoint_clocks(self, token):
        clocks = super().waypoint_clocks(token)
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
    def __init__(self, action_key="actions", output_action_key=None, **codec_kwargs):
        self.action_key = action_key
        self.output_action_key = output_action_key or action_key
        self.codec = PlanarArcClockCodec(**codec_kwargs)

    def transform(self, batch):
        value = np.asarray(batch[self.action_key])
        dtype = value.dtype if np.issubdtype(value.dtype, np.floating) else np.float32
        batch[self.output_action_key] = self.codec.encode(value).astype(dtype)
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
