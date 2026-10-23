"""Baseline utilities temporarily separated from the withdrawn Cartesian ARC codec.

Remove this transitional module when the explicit ARC integration restores
the final shared utility locations; it contains no tokenization or decoding.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from egomimic.rldb.zarr.action_chunk_transforms import Transform

ARM_LAYOUT = ((0, 3, 6, slice(0, 3)), (7, 10, 13, slice(7, 10)))


def lowpass_positions(pos: np.ndarray, fc_hz: float, fs_hz: float) -> np.ndarray:
    """Zero-phase 4th-order Butterworth low-pass of a (T, 3) track (Step 0's filter).
    Chunks too short for the filter's padding are returned unchanged."""
    pos = np.asarray(pos, dtype=np.float64)
    if fc_hz is None or fc_hz <= 0 or len(pos) < 20:
        return pos
    sos = butter(4, float(fc_hz), btype="low", fs=float(fs_hz), output="sos")
    return sosfiltfilt(sos, pos, axis=0)


class CopyKeyRows(Transform):
    """``batch[dst] = batch[src][:n_rows]`` — carries the un-tokenized time chunk
    (``actions_time``) alongside the model target so the E1 evaluator can score
    every variant against the same 30 Hz ground truth."""

    def __init__(self, src: str, dst: str, n_rows: int):
        self.src, self.dst, self.n_rows = src, dst, int(n_rows)

    def transform(self, batch: dict) -> dict:
        x = np.asarray(batch[self.src], dtype=np.float64)
        batch[self.dst] = x[: self.n_rows].copy()
        return batch
