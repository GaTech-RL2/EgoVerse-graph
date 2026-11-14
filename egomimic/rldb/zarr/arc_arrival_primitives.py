"""Geometry helpers copied from main c7e11046 for the isolated LIBERO runtime."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def _ypr_to_rotation(ypr: np.ndarray) -> R:
    """Convert (N, 3) or (3,) yaw-pitch-roll (ZYX euler) to a scipy Rotation."""
    return R.from_euler("ZYX", np.asarray(ypr, dtype=np.float64).reshape(-1, 3))


def _rotation_to_ypr(rotation: R) -> np.ndarray:
    """Convert a scipy Rotation of length N to (N, 3) yaw-pitch-roll."""
    ypr = rotation.as_euler("ZYX", degrees=False)
    return np.atleast_2d(ypr).astype(np.float64)


def cumulative_arc_length(pos: np.ndarray) -> np.ndarray:
    """pos: (N, 3) -> cumulative translational arc length (N,), starting at 0."""
    pos = np.asarray(pos, dtype=np.float64)
    step = np.linalg.norm(pos[1:] - pos[:-1], axis=-1)
    return np.concatenate([np.array([0.0]), np.cumsum(step)])


def cumulative_rotation_length(ypr: np.ndarray) -> np.ndarray:
    """Return cumulative SO(3) geodesic angle in radians.

    The returned coordinate is an independent rotation clock.  It deliberately
    does not depend on translational motion, so an in-place wrist rotation can
    advance even when the EEF position is stationary.
    """
    ypr = np.asarray(ypr, dtype=np.float64).reshape(-1, 3)
    if len(ypr) < 2:
        return np.zeros(len(ypr), dtype=np.float64)
    rotations = _ypr_to_rotation(ypr)
    relative = rotations[:-1].inv() * rotations[1:]
    step = np.linalg.norm(relative.as_rotvec(), axis=-1)
    return np.concatenate([np.array([0.0]), np.cumsum(step)])


def _bracket_segments(
    cumdist: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized :func:`_bracket_segment` over many arc-length targets.

    Same clamping rules as the scalar version: a target at or below
    ``cumdist[0]`` maps to ``(0, 0.0)``, at or above ``cumdist[-1]`` to
    ``(len - 2, 1.0)``, and a zero-length segment yields ``alpha = 0``.
    """
    cumdist = np.asarray(cumdist, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    n = len(cumdist)
    last = max(n - 2, 0)
    i = np.clip(np.searchsorted(cumdist, targets, side="left") - 1, 0, last)
    s0 = cumdist[i]
    s1 = cumdist[np.minimum(i + 1, n - 1)]
    span = s1 - s0
    ok = span > 1e-12
    alpha = np.where(ok, (targets - s0) / np.where(ok, span, 1.0), 0.0)
    lo = targets <= cumdist[0]
    hi = targets >= cumdist[-1]
    i = np.where(lo, 0, np.where(hi, last, i))
    alpha = np.where(lo, 0.0, np.where(hi, 1.0, alpha))
    return i, alpha


def _first_crossing_brackets(
    cumdist: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized first-crossing bracket, mirroring the per-arm scalar rule.

    Unlike :func:`_bracket_segments` this never clamps a target at or past the
    end to the last segment's ``alpha = 1``: a trailing hold consumes no
    translation, so the first frame reaching the distance is the answer.
    ``TokenizeBimanualArcLengthCartesian._translation_bracket`` is the scalar
    statement of the same rule, and the two must agree exactly.
    """
    cumdist = np.asarray(cumdist, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    n = len(cumdist)
    capped = np.minimum(targets, cumdist[-1])
    upper = np.clip(np.searchsorted(cumdist, capped, side="left"), 1, n - 1)
    interval = cumdist[upper] - cumdist[upper - 1]
    ok = interval > 1e-12
    # The numerator keeps the uncapped target, exactly as the scalar path does;
    # the clip below is what bounds an overshooting target.
    alpha = np.where(
        ok, (targets - cumdist[upper - 1]) / np.where(ok, interval, 1.0), 0.0
    )
    alpha = np.clip(alpha, 0.0, 1.0)
    lo = targets <= cumdist[0]
    return np.where(lo, 0, upper - 1), np.where(lo, 0.0, alpha)


def _slerp_segments_ypr(
    ypr: np.ndarray, i: np.ndarray, alpha: np.ndarray
) -> np.ndarray:
    """SLERP ``ypr[i] -> ypr[i + 1]`` at ``alpha``, for many pairs at once.

    Walks the same geodesic scipy's two-keyframe ``Slerp`` walks, as one
    batched rotation-vector step instead of one ``Slerp`` object per target.
    """
    ypr = np.asarray(ypr, dtype=np.float64)
    n = len(ypr)
    j = np.minimum(i + 1, n - 1)
    rot = _ypr_to_rotation(ypr)
    r0 = rot[i]
    rel = r0.inv() * rot[j]
    out = _rotation_to_ypr(r0 * R.from_rotvec(rel.as_rotvec() * alpha[:, None]))
    # Endpoints exactly: the scalar helpers short-circuit at alpha 0 / 1 and
    # return the keyframe itself, so do not let the geodesic round-trip it.
    at0 = alpha <= 0.0
    at1 = alpha >= 1.0
    out[at0] = ypr[i[at0]]
    out[at1] = ypr[j[at1]]
    return out
