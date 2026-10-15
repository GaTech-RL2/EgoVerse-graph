"""ARC Projection: one rigid-point progress coordinate and clock per arm.

Waypoints still contain end-effector xyz, intrinsic ZYX Euler rotation and
gripper aperture. Only their placement and timing metric use p + R @ offset.
The inherited duration/profile representations and interpolation stay intact.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from egomimic.rldb.zarr.e1_arc_tokenizer import TokenizeBimanualArcLengthE1

PROJECTION_OFFSET_M = (0.02, 0.02, 0.02)
PROJECTION_VARIANTS = {"arc_projection_dur": "dur", "arc_projection_vel": "profile"}
PROJECTION_LAYOUTS = {"e1_projection_dur": "dur", "e1_projection_vel": "profile"}


class TokenizeBimanualArcProjection(TokenizeBimanualArcLengthE1):
    """Single-stream duration or velocity ARC measured at an offset local point.

    The default offset is 2 cm along EACH local axis (norm sqrt(3) * 2 cm).
    Rotation about this vector, or translation cancelling the point's rotational
    motion, is invisible to this metric. Stationary projected paths inherit E1's
    hold behavior; this is not a full SE(3) distance.
    """

    def __init__(self, *, projection_offset_m=PROJECTION_OFFSET_M, **kwargs):
        offset = np.asarray(projection_offset_m, dtype=np.float64)
        if offset.shape != (3,) or not np.isfinite(offset).all() or not np.any(offset):
            raise ValueError(
                "projection_offset_m must be a finite nonzero local 3-vector"
            )
        kwargs.setdefault("velocity_mode", "dur")
        if kwargs["velocity_mode"] not in ("dur", "profile"):
            raise ValueError(
                "ARC Projection supports only duration or velocity profile timing"
            )
        # Rotation of a 2 cm lever can be much slower than the legacy 1 cm/s
        # translation floor. Preserve that motion rather than artificially speeding it up.
        kwargs.setdefault("min_speed", 1e-6)
        if not np.isfinite(kwargs["min_speed"]) or kwargs["min_speed"] <= 0:
            raise ValueError("min_speed must be finite and positive")
        super().__init__(**kwargs)
        self.projection_offset_m = offset.copy()

    def progress_coordinates(self, pos: np.ndarray, ypr: np.ndarray) -> np.ndarray:
        return np.asarray(pos, dtype=np.float64) + Rotation.from_euler(
            "ZYX", ypr
        ).apply(self.projection_offset_m)
