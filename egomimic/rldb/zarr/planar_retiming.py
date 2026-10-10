"""Recorded-command retiming; this does not simulate contact dynamics."""

from __future__ import annotations

import math

import numpy as np


class PlanarCommandRetiming:
    """Five virtual views give uniform rates without worker/RNG dependence.

    ZarrDataset supplies the selected view, excludes incomplete raw windows,
    and checks the episode clock. Apply BEFORE native-to-model conversion.
    Observations stay at the anchor; only future command targets are retimed.
    """

    def __init__(
        self, rates=(1.0,), horizon=16, fps=30.0, action_key="actions", angle_column=2
    ):
        self.rates = tuple(float(r) for r in rates)
        self.horizon = int(horizon)
        self.fps = float(fps)
        if (
            not self.rates
            or len(set(self.rates)) != len(self.rates)
            or any(not math.isfinite(r) or r <= 0 for r in self.rates)
            or self.horizon < 2
            or not math.isfinite(self.fps)
            or self.fps <= 0
        ):
            raise ValueError("Invalid retiming rates, horizon or FPS")
        self.action_key = action_key
        self.angle_column = int(angle_column)
        self.required_frames = math.ceil(max(self.rates) * (self.horizon - 1)) + 1
        self.sample_views = len(self.rates)

    def transform(self, batch):
        view = int(batch.pop("_retiming_view"))
        if not 0 <= view < self.sample_views:
            raise ValueError("Invalid retiming view")
        a = np.asarray(batch[self.action_key], dtype=np.float64)
        if (
            a.ndim != 2
            or a.shape[0] < self.required_frames
            or a.shape[1] not in (3, 4)
            or not np.isfinite(a).all()
        ):
            raise ValueError("Retiming requires a finite, unpadded native planar chunk")
        if self.angle_column != 2:
            raise ValueError("Native planar schema requires theta in column 2")
        rate = self.rates[view]
        queries = np.arange(self.horizon, dtype=np.float64) * rate
        native = a.copy()
        native[:, 2] = np.unwrap(native[:, 2])
        out = np.stack(
            [
                np.interp(queries, np.arange(len(a)), native[:, j])
                for j in range(a.shape[1])
            ],
            axis=-1,
        )
        out[:, 2] = (out[:, 2] + np.pi) % (2 * np.pi) - np.pi
        batch[self.action_key] = out.astype(np.float32)
        # PushT emits only the declared rate; discard stale physical-speed telemetry.
        batch.pop("requested_speed", None)
        batch.pop("requested_speed_value", None)
        batch["retiming_rate"] = np.asarray([rate], dtype=np.float32)
        return batch
