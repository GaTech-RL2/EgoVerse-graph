"""Native decoder for independently timed Planar translation, rotation, grip."""
from __future__ import annotations

import numpy as np
import torch

from egomimic.pipeline.pushshapes import PlanarArcTimedNativeDecoder


class PlanarArcGroupedNativeDecoder(PlanarArcTimedNativeDecoder):
    def __init__(self, translation_groups="xy", **kwargs):
        super().__init__(**kwargs)
        if translation_groups not in {"xy", "x_y"}:
            raise ValueError("translation_groups must be xy or x_y")
        self.translation_groups = translation_groups
        self.action_dim = 8 if translation_groups == "xy" else 9

    def _streams(self, value):
        xy_groups = [("translation", (0, 1))] if self.translation_groups == "xy" else [("x", (0,)), ("y", (1,))]
        streams = [(name, value[..., columns]) for name, columns in xy_groups]
        angle = torch.atan2(value[..., 3], value[..., 2])
        diff = torch.diff(angle, dim=-1)
        delta = torch.atan2(torch.sin(diff), torch.cos(diff))
        unwrapped = torch.cat((angle[..., :1], angle[..., :1] + delta.cumsum(-1)), dim=-1)
        streams.extend([("rotation", unwrapped[..., None]), ("gripper", value[..., 4:5])])
        return [(name, geometry, self._durations(geometry, value[:, :-1, 5 + i]))
                for i, (name, geometry) in enumerate(streams)]

    def _tensor(self, actions):
        value = actions if torch.is_tensor(actions) else torch.as_tensor(np.asarray(actions))
        if not value.is_floating_point():
            value = value.float()
        expected = (self.num_waypoints, self.action_dim)
        if value.ndim < 2 or tuple(value.shape[-2:]) != expected:
            raise ValueError(f"Expected (...,{expected[0]},{expected[1]}), got {tuple(value.shape)}")
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite grouped ARC prediction")
        return value

    def waypoint_clocks(self, actions):
        value = self._tensor(actions)
        if value.ndim != 2:
            raise ValueError("Execution selection expects a single token")
        clocks = {}
        for name, geometry, durations in self._streams(value.unsqueeze(0)):
            d = durations[0]
            clocks[name] = {"seconds": torch.cat((d.new_zeros(1), d.cumsum(0))),
                           "active": bool((torch.diff(geometry[0], dim=0).abs() > self.epsilon).any())}
        return clocks

    def decode(self, actions, context=None):
        del context
        value = self._tensor(actions)
        leading = tuple(value.shape[:-2])
        value = value.reshape(-1, self.num_waypoints, self.action_dim)
        parts = [self._sample(geometry, duration) for _, geometry, duration in self._streams(value)]
        native = torch.cat(parts, dim=-1)[..., :self.native_action_dim]
        # Explicit anchor protects constant STK streams with zero-time supports.
        angle = torch.atan2(value[:, 0, 3], value[:, 0, 2])[:, None]
        anchor = torch.cat((value[:, 0, :2], angle, value[:, 0, 4:5]), dim=-1)
        native[:, 0] = anchor[:, :self.native_action_dim]
        native = native.reshape(*leading, self.action_horizon, self.native_action_dim)
        return native if torch.is_tensor(actions) else native.cpu().numpy()

    __call__ = decode
