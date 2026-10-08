"""Planar ARC with independent continuous-aperture gripper timing.

All shape columns precede timing. Grouped XY has width 8; scalar X/Y has
width 9. Unlike LIBERO's binary gripper command, Planar aperture is continuous
and is interpolated on its own total-variation stream, without a D/R cap.
"""
from __future__ import annotations

import numpy as np

from egomimic.rldb.zarr.planar_arc import TokenizePlanarArcTimed


class TokenizePlanarArcGrouped(TokenizePlanarArcTimed):
    def __init__(self, translation_groups="xy", **kwargs):
        super().__init__(**kwargs)
        if translation_groups not in {"xy", "x_y"}:
            raise ValueError("translation_groups must be xy or x_y")
        self.translation_groups = translation_groups
        self.action_dim = 8 if translation_groups == "xy" else 9

    def tokenize(self, actions):
        xy, theta, grip = self._components(np.asarray(actions, dtype=np.float64))
        output = np.zeros((self.num_waypoints, self.action_dim), dtype=np.float64)
        groups = [(0, 1)] if self.translation_groups == "xy" else [(0,), (1,)]
        streams = [(xy[:, axes], axes, self.distance, len(axes) == 1) for axes in groups]
        streams += [(theta[:, None], (2, 3), self.rotation_distance, True),
                    (grip[:, None], (4,), None, True)]
        for index, (values, columns, cap, signed) in enumerate(streams):
            arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(values, axis=0), axis=-1))]
            end = float(arc[-1]) if cap is None else min(float(arc[-1]), cap)
            points, timing, _ = self._sample_stream(values, arc, end, signed_rate=signed)
            if columns == (2, 3):
                output[:, 2:4] = np.column_stack((np.cos(points[:, 0]), np.sin(points[:, 0])))
            else:
                output[:, columns] = points
            output[:, 5 + index] = timing
        return output


def get_planar_grouped_transform_list(keys=None, raw_action_horizon=80,
        action_target_offset=1, translation_groups="xy", **kwargs):
    from egomimic.rldb.embodiment.pushshapes import SliceActionTarget

    keys = keys or ["actions"]
    if len(keys) != 1:
        raise ValueError("Grouped Planar ARC requires one action key")
    accepted = {k: kwargs[k] for k in ("min_distance_unit", "resampled_vector_length",
        "dt", "rotation_distance_unit", "timing_mode", "zero_dist_epsilon", "waypoint_sampling") if k in kwargs}
    return [SliceActionTarget(keys, int(action_target_offset), int(raw_action_horizon)),
        TokenizePlanarArcGrouped(action_key=keys[0], output_action_key=keys[0],
                                translation_groups=translation_groups, **accepted)]
