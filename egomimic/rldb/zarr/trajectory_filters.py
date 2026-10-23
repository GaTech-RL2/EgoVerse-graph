"""Position filtering for dataset anchor selection; not ARC target smoothing."""

import numpy as np
from scipy.signal import butter, sosfiltfilt


def lowpass_positions(pos: np.ndarray, fc_hz: float, fs_hz: float) -> np.ndarray:
    """Zero-phase 4th-order Butterworth low-pass of a (T, 3) track (Step 0's filter).
    Chunks too short for the filter's padding are returned unchanged."""
    pos = np.asarray(pos, dtype=np.float64)
    if fc_hz is None or fc_hz <= 0 or len(pos) < 20:
        return pos
    sos = butter(4, float(fc_hz), btype="low", fs=float(fs_hz), output="sos")
    return sosfiltfilt(sos, pos, axis=0)
