"""Bimanual arc-length tokenizer for EEF pose action chunks.

Re-parameterizes a time-indexed bimanual cartesian action chunk by translational
arc length instead of time: each arm's next ``min_distance_unit`` meters of EEF
travel is resampled to ``resampled_vector_length`` waypoints spaced uniformly in
arc length, and a translational-velocity channel is appended so the original
timing can be reconstructed at deploy time.

Ported from the GR00T arc-length tokenizer
(gr00t branch ``rpunamiya/arc-length-tokenizer``,
``groot/core/utils/state_action/arc_length_tokenizer.py`` +
``groot/core/data/state_action/arc_length_action_transform.py``), adapted to
this repo's conventions:

- Rotations are xyz + ypr (ZYX euler, matching ``egomimic.utils.pose_utils``)
  instead of 6D rotation. Interpolation still happens in rotation space
  (scipy SLERP), never per-euler-angle.
- Operates on the canonical (T, 14) bimanual cartesian layout
  ``[L xyz ypr grip | R xyz ypr grip]`` produced by the embodiment transform
  pipelines, rather than per-group dicts. Optional per-arm joint chunks
  (T, J) resample along that arm's arc length (``tokenize_with_joints``).
- Each arm has its own independent cumulative arc length; a stationary arm
  yields a zero token (pose held, zero velocity) while the other arm tokenizes
  normally.
- Inputs containing the >=1e8 invalid-pose sentinel produce a 1e9-filled
  output, matching the interpolation helpers in ``pose_utils``.

Layout of the tokenized chunk (per waypoint): per arm
``[xyz(3) ypr(3) grip(1) trans_vel(velocity_dim)]``, arms concatenated —
16 dims for the scalar velocity modes (velocity_dim=1), 20 for the per-dim
modes (velocity_dim=3). Velocity stats for normalization live in
``arc_length_stats.py``.

Only numpy + scipy are required, so this module stays importable in
environments without torch / projectaria. ``TokenizeBimanualArcLength``
implements the same ``transform(batch: dict) -> dict`` interface as
``action_chunk_transforms.Transform`` and composes into those pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.optimize import isotonic_regression
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# Canonical bimanual cartesian layout: [L xyz ypr grip | R xyz ypr grip].
ARM_DIM = 7  # xyz(3) + ypr(3) + gripper(1)
BIMANUAL_CARTESIAN_DIM = 2 * ARM_DIM  # 14
# Tokenized layout appends velocity_dim(mode) trans_vel columns per arm.
# These constants give the scalar-mode (velocity_dim=1) layout; use
# BimanualArcLengthTokenizer.arc_arm_dim / .arc_dim for the mode-aware dims.
ARC_ARM_DIM = ARM_DIM + 1  # 8
ARC_BIMANUAL_DIM = 2 * ARC_ARM_DIM  # 16

# Invalid-pose convention shared with pose_utils._interpolate_euler.
INVALID_POSE_THRESHOLD = 1e8
INVALID_POSE_FILL = 1e9


class VelocityMode(str, Enum):
    """How the translational-velocity payload of a chunk is summarized.

    MEAN_SCALAR: one scalar per chunk — chord distance / chunk duration.
    MEAN_PER_DIM: one 3-vector per chunk — displacement / chunk duration.
    PER_STEP_SCALAR: one scalar per waypoint segment (last value repeated).
    PER_STEP_PER_DIM: one 3-vector per waypoint segment (last value repeated).
    """

    MEAN_SCALAR = "mean_scalar"
    MEAN_PER_DIM = "mean_per_dim"
    PER_STEP_SCALAR = "per_step_scalar"
    PER_STEP_PER_DIM = "per_step_per_dim"


def velocity_dim(mode: VelocityMode | str) -> int:
    """Columns the velocity channel occupies per arm in the tokenized layout."""
    mode = VelocityMode(mode)
    if mode in (VelocityMode.MEAN_SCALAR, VelocityMode.PER_STEP_SCALAR):
        return 1
    return 3


# ---------------------------------------------------------------------------
# Rotation helpers (ypr <-> scipy Rotation; interpolation in rotation space)
# ---------------------------------------------------------------------------


def _ypr_to_rotation(ypr: np.ndarray) -> R:
    """Convert (N, 3) or (3,) yaw-pitch-roll (ZYX euler) to a scipy Rotation."""
    return R.from_euler("ZYX", np.asarray(ypr, dtype=np.float64).reshape(-1, 3))


def _rotation_to_ypr(rotation: R) -> np.ndarray:
    """Convert a scipy Rotation of length N to (N, 3) yaw-pitch-roll."""
    ypr = rotation.as_euler("ZYX", degrees=False)
    return np.atleast_2d(ypr).astype(np.float64)


def slerp_pair_ypr(
    ypr_a: np.ndarray, ypr_b: np.ndarray, alphas: np.ndarray
) -> np.ndarray:
    """SLERP between two ypr rotations at the given alphas in [0, 1].

    returns: (len(alphas), 3) ypr.
    """
    alphas = np.clip(np.asarray(alphas, dtype=np.float64), 0.0, 1.0)
    if len(alphas) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    keyframes = _ypr_to_rotation(
        np.stack([np.asarray(ypr_a), np.asarray(ypr_b)], axis=0)
    )
    slerp = Slerp([0.0, 1.0], keyframes)
    return _rotation_to_ypr(slerp(alphas))


def slerp_through_ypr(ypr_seq: np.ndarray, num: int) -> np.ndarray:
    """SLERP through N ypr waypoints, sampling `num` points uniformly.

    returns: (num, 3) ypr.
    """
    ypr_seq = np.asarray(ypr_seq, dtype=np.float64).reshape(-1, 3)
    n = len(ypr_seq)
    if n == 0:
        return np.zeros((num, 3), dtype=np.float64)
    if n == 1:
        return np.repeat(ypr_seq, num, axis=0)
    slerp = Slerp(np.linspace(0.0, 1.0, n), _ypr_to_rotation(ypr_seq))
    return _rotation_to_ypr(slerp(np.linspace(0.0, 1.0, num)))


# ---------------------------------------------------------------------------
# Arc-length helpers (3D position)
# ---------------------------------------------------------------------------


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


def cumulative_bimanual_translation_length(actions: np.ndarray) -> np.ndarray:
    """Joint translation clock: left travel plus right travel."""
    actions = np.asarray(actions, dtype=np.float64)
    left = np.linalg.norm(np.diff(actions[:, 0:3], axis=0), axis=-1)
    right = np.linalg.norm(np.diff(actions[:, 7:10], axis=0), axis=-1)
    return np.concatenate(([0.0], np.cumsum(left + right)))


def _bracket_segment(cumdist: np.ndarray, target_s: float) -> tuple[int, float]:
    """Locate the segment containing arc length target_s.

    returns: (segment start index i, alpha in [0, 1]), clamped to the first /
    last segment when target_s falls outside [cumdist[0], cumdist[-1]].
    """
    if target_s <= cumdist[0]:
        return 0, 0.0
    if target_s >= cumdist[-1]:
        return len(cumdist) - 2, 1.0
    i = int(np.searchsorted(cumdist, target_s, side="left") - 1)
    i = max(0, min(i, len(cumdist) - 2))
    s0, s1 = cumdist[i], cumdist[i + 1]
    if s1 <= s0 + 1e-12:
        return i, 0.0
    return i, float((target_s - s0) / (s1 - s0))


def _interp_pos_at_s(
    pos: np.ndarray, cumdist: np.ndarray, target_s: float
) -> np.ndarray:
    """Linear-interp position at arc length target_s. returns: (3,)."""
    if target_s <= cumdist[0]:
        return pos[0].copy()
    if target_s >= cumdist[-1]:
        return pos[-1].copy()
    i, alpha = _bracket_segment(cumdist, target_s)
    return (1.0 - alpha) * pos[i] + alpha * pos[i + 1]


def _interp_ypr_at_s(
    ypr: np.ndarray, cumdist: np.ndarray, target_s: float
) -> np.ndarray:
    """SLERP rotation at arc length target_s. returns: (3,) ypr."""
    if target_s <= cumdist[0]:
        return ypr[0].copy()
    if target_s >= cumdist[-1]:
        return ypr[-1].copy()
    i, alpha = _bracket_segment(cumdist, target_s)
    if alpha == 0.0:
        return ypr[i].copy()
    return slerp_pair_ypr(ypr[i], ypr[i + 1], np.array([alpha]))[0]


def _interp_linear_at_s(
    values: np.ndarray, cumdist: np.ndarray, target_s: float
) -> np.ndarray:
    """Linear-interp (N, D) values at arc length target_s. returns: (D,)."""
    if target_s <= cumdist[0]:
        return values[0].copy()
    if target_s >= cumdist[-1]:
        return values[-1].copy()
    i, alpha = _bracket_segment(cumdist, target_s)
    return (1.0 - alpha) * values[i] + alpha * values[i + 1]


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


def _source_times_at_targets(
    cumdist: np.ndarray, targets: np.ndarray, dt: float
) -> np.ndarray:
    """Map arc targets to fractional source times with scalar-identical math."""
    indices, alpha = _bracket_segments(cumdist, targets)
    return (indices + alpha) * dt


def _linear_at_targets(
    values: np.ndarray, cumdist: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """Interpolate all targets while preserving the scalar operation order."""
    indices, alpha = _bracket_segments(cumdist, targets)
    blend = alpha[:, None]
    return (1.0 - blend) * values[indices] + blend * values[indices + 1]


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


def resample_at_arc_lengths(
    pos: np.ndarray,
    ypr: np.ndarray,
    gripper: np.ndarray,
    cumdist: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate pos / ypr / gripper at many arc lengths in a single pass.

    Vectorized equivalent of calling :func:`_interp_pos_at_s`,
    :func:`_interp_ypr_at_s` and :func:`_interp_linear_at_s` once per target.
    The scalar rotation helper builds a fresh two-keyframe ``Slerp`` per
    target, so a bimanual token at M = 100 costs ~200 Rotation/Slerp
    constructions per sample and dominates data loading (58 ms/sample
    measured on mecka fold chunks, ~1 s/sample in loader workers without
    ``OMP_NUM_THREADS=1``); this form measures 1.65 ms.

    returns: (pos (K, 3), ypr (K, 3), gripper (K, G)) for K targets.
    """
    targets = np.asarray(targets, dtype=np.float64)
    i, alpha = _bracket_segments(cumdist, targets)
    j = np.minimum(i + 1, len(cumdist) - 1)
    a = alpha[:, None]
    pos = np.asarray(pos, dtype=np.float64)
    gripper = np.asarray(gripper, dtype=np.float64)
    pos_out = (1.0 - a) * pos[i] + a * pos[j]
    grip_out = (1.0 - a) * gripper[i] + a * gripper[j]
    ypr_out = _slerp_segments_ypr(ypr, i, alpha)
    return pos_out, ypr_out, grip_out


def resample_by_distance(
    pos: np.ndarray,
    ypr: np.ndarray,
    gripper: np.ndarray,
    cumdist: np.ndarray,
    start_s: float,
    end_s: float,
    num_samples: int,
    *,
    start_idx: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample pos/ypr/gripper uniformly in arc length over [start_s, end_s].

    Position and gripper interpolate linearly along the original segments;
    rotation SLERPs per-segment. If start_idx is given, the first sample is
    taken explicitly from that index — this breaks arc-length-lookup ambiguity
    in stationary clusters where multiple timesteps share one cumulative arc
    length but differ in rotation/gripper.

    returns: (pos (M, 3), ypr (M, 3), gripper (M, G)).
    """
    targets = np.linspace(start_s, end_s, num_samples)
    pos_out, ypr_out, grip_out = resample_at_arc_lengths(
        pos, ypr, gripper, cumdist, targets
    )
    if start_idx is not None:
        pos_out[0] = pos[start_idx]
        ypr_out[0] = ypr[start_idx]
        grip_out[0] = gripper[start_idx]
    return pos_out, ypr_out, grip_out


def _dist_interval_indices(
    cumdist: np.ndarray, start_s: float, end_s: float
) -> tuple[int, int]:
    """Original-timestep index range [start_idx, end_idx] covering [start_s, end_s]."""
    start_idx = int(np.searchsorted(cumdist, start_s, side="left"))
    end_idx = int(np.searchsorted(cumdist, end_s, side="right") - 1)
    start_idx = max(0, min(start_idx, len(cumdist) - 1))
    end_idx = max(start_idx, min(end_idx, len(cumdist) - 1))
    return start_idx, end_idx


# ---------------------------------------------------------------------------
# Velocity payloads
# ---------------------------------------------------------------------------


def make_velocity_payload(
    pos_rs: np.ndarray, chunk_duration_steps: int, dt: float, mode: VelocityMode
) -> np.ndarray:
    """Summarize a resampled chunk's translational speed.

    args:
        pos_rs: (M, 3) arc-length-resampled positions.
        chunk_duration_steps: number of original time steps the chunk covers.
        dt: seconds per original time step.
    returns:
        MEAN_SCALAR: (1,) chord distance / total time.
        MEAN_PER_DIM: (3,) displacement / total time.
        PER_STEP_SCALAR: (M,) per-segment speed, last value repeated.
        PER_STEP_PER_DIM: (M, 3) per-segment velocity, last value repeated.
    """
    m = pos_rs.shape[0]
    total_time = max(chunk_duration_steps * dt, 1e-8)
    if mode == VelocityMode.MEAN_SCALAR:
        dist = float(np.linalg.norm(pos_rs[-1] - pos_rs[0]))
        return np.array([dist / total_time], dtype=np.float64)
    if mode == VelocityMode.MEAN_PER_DIM:
        return ((pos_rs[-1] - pos_rs[0]) / total_time).astype(np.float64)
    seg = pos_rs[1:] - pos_rs[:-1]
    per_seg_time = max(total_time / max(m - 1, 1), 1e-8)
    if mode == VelocityMode.PER_STEP_SCALAR:
        v = np.linalg.norm(seg, axis=-1) / per_seg_time
        return np.concatenate([v, v[-1:]]).astype(np.float64)
    if mode == VelocityMode.PER_STEP_PER_DIM:
        v = seg / per_seg_time
        return np.concatenate([v, v[-1:]], axis=0).astype(np.float64)
    raise ValueError(f"Unsupported velocity mode: {mode}")


def _zero_velocity(mode: VelocityMode, m: int) -> np.ndarray:
    if mode == VelocityMode.MEAN_SCALAR:
        return np.zeros((1,), dtype=np.float64)
    if mode == VelocityMode.MEAN_PER_DIM:
        return np.zeros((3,), dtype=np.float64)
    if mode == VelocityMode.PER_STEP_SCALAR:
        return np.zeros((m,), dtype=np.float64)
    if mode == VelocityMode.PER_STEP_PER_DIM:
        return np.zeros((m, 3), dtype=np.float64)
    raise ValueError(f"Unsupported velocity mode: {mode}")


def _broadcast_velocity(
    trans_vel: np.ndarray, mode: VelocityMode, m: int
) -> np.ndarray:
    """Reshape a velocity payload into a uniform (M, velocity_dim(mode)) block."""
    v = np.asarray(trans_vel, dtype=np.float64)
    if mode == VelocityMode.MEAN_SCALAR:
        return np.broadcast_to(v.reshape(1, 1), (m, 1)).copy()
    if mode == VelocityMode.MEAN_PER_DIM:
        return np.broadcast_to(v.reshape(1, 3), (m, 3)).copy()
    if mode == VelocityMode.PER_STEP_SCALAR:
        return v.reshape(m, 1)
    if mode == VelocityMode.PER_STEP_PER_DIM:
        return v.reshape(m, 3)
    raise ValueError(f"Unsupported velocity mode: {mode}")


# ---------------------------------------------------------------------------
# Single-arm tokenizer core
# ---------------------------------------------------------------------------


@dataclass
class ArcChunkToken:
    """One arc-length chunk of a single arm's trajectory.

    kind is "motion" for a normal chunk or "zero" for a degenerate one
    (stationary, or covering too many time steps for the distance unit).
    """

    kind: str
    mode: str
    pos: np.ndarray  # (M, 3) arc-length-resampled positions
    ypr: np.ndarray  # (M, 3) resampled rotations (ZYX euler)
    gripper: np.ndarray  # (M, G) resampled gripper
    trans_vel: np.ndarray  # (1,), (3,), (M,), or (M, 3) depending on mode
    start_idx: int
    end_idx: int
    num_original_steps: int
    chunk_distance: float
    dt: float
    zero_reason: str | None = None


class ArcLengthTokenizer:
    """Single-arm arc-length tokenizer for xyz + ypr (+ gripper) trajectories.

    Chunks a trajectory into fixed translational arc-length units, resamples
    each chunk to ``resampled_vector_length`` waypoints uniform in arc length,
    and stores a velocity payload for timing reconstruction.
    """

    def __init__(
        self,
        min_distance_unit: float = 0.05,
        resampled_vector_length: int = 20,
        mode: str = "mean_scalar",
        dt: float = 1.0 / 30.0,
        zero_dist_epsilon: float = 1e-6,
        max_steps_per_chunk: int = 200,
    ):
        if resampled_vector_length < 2:
            raise ValueError(
                f"resampled_vector_length must be >= 2, got {resampled_vector_length}"
            )
        self.min_distance_unit = float(min_distance_unit)
        self.M = int(resampled_vector_length)
        self.mode = VelocityMode(mode)
        self.dt = float(dt)
        self.zero_dist_epsilon = float(zero_dist_epsilon)
        self.max_steps_per_chunk = int(max_steps_per_chunk)

    @staticmethod
    def _validate_inputs(
        pos: np.ndarray, ypr: np.ndarray, gripper: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos = np.asarray(pos, dtype=np.float64)
        ypr = np.asarray(ypr, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"pos must have shape (N, 3), got {pos.shape}")
        if ypr.ndim != 2 or ypr.shape[1] != 3:
            raise ValueError(f"ypr must have shape (N, 3), got {ypr.shape}")
        if len(pos) != len(ypr):
            raise ValueError(
                f"pos and ypr must have the same length, got {len(pos)} vs {len(ypr)}"
            )
        if len(pos) < 2:
            raise ValueError(f"Need at least 2 timesteps, got {len(pos)}")
        if gripper is None:
            gripper = np.zeros((len(pos), 1), dtype=np.float64)
        else:
            gripper = np.asarray(gripper, dtype=np.float64)
            if gripper.ndim == 1:
                gripper = gripper[:, None]
            if len(gripper) != len(pos):
                raise ValueError(
                    f"gripper length {len(gripper)} does not match pos length {len(pos)}"
                )
        return pos, ypr, gripper

    # -- tokenize ----------------------------------------------------------

    def tokenize(
        self,
        pos: np.ndarray,
        ypr: np.ndarray,
        gripper: np.ndarray | None = None,
    ) -> list[ArcChunkToken]:
        """Chunk the full trajectory into consecutive non-overlapping tokens."""
        pos, ypr, gripper = self._validate_inputs(pos, ypr, gripper)
        cumdist = cumulative_arc_length(pos)
        total_s = float(cumdist[-1])

        tokens: list[ArcChunkToken] = []
        cur_s = 0.0
        while cur_s < total_s - 1e-12:
            next_s = min(cur_s + self.min_distance_unit, total_s)
            start_idx, end_idx = _dist_interval_indices(cumdist, cur_s, next_s)
            num_steps = max(1, end_idx - start_idx)
            chunk_dist = float(next_s - cur_s)

            if chunk_dist < self.zero_dist_epsilon:
                reason = "distance_below_epsilon"
            elif num_steps > self.max_steps_per_chunk:
                reason = "too_many_steps_for_distance_unit"
            else:
                reason = None

            if reason is not None:
                tokens.append(
                    self._make_zero_token(
                        pos,
                        ypr,
                        gripper,
                        cumdist,
                        cur_s,
                        next_s,
                        start_idx,
                        end_idx,
                        num_steps,
                        chunk_dist,
                        reason,
                    )
                )
            else:
                pos_rs, ypr_rs, grip_rs = resample_by_distance(
                    pos, ypr, gripper, cumdist, cur_s, next_s, self.M
                )
                tokens.append(
                    ArcChunkToken(
                        kind="motion",
                        mode=self.mode.value,
                        pos=pos_rs,
                        ypr=ypr_rs,
                        gripper=grip_rs,
                        trans_vel=make_velocity_payload(
                            pos_rs, num_steps, self.dt, self.mode
                        ),
                        start_idx=start_idx,
                        end_idx=end_idx,
                        num_original_steps=num_steps,
                        chunk_distance=chunk_dist,
                        dt=self.dt,
                    )
                )
            cur_s = next_s
        return tokens

    def tokenize_at(
        self,
        pos: np.ndarray,
        ypr: np.ndarray,
        gripper: np.ndarray | None = None,
        t: int = 0,
    ) -> ArcChunkToken:
        """Tokenize the next min_distance_unit of travel starting at timestep t.

        This is the time-synced form used for prediction targets: the token
        covers arc lengths [cumdist[t], cumdist[t] + min_distance_unit]
        (clipped to the trajectory end), with the first waypoint anchored
        exactly at timestep t.
        """
        pos, ypr, gripper = self._validate_inputs(pos, ypr, gripper)
        n = len(pos)
        if not 0 <= t < n:
            raise ValueError(f"t must be in [0, {n - 1}], got {t}")
        cumdist = cumulative_arc_length(pos)
        total_s = float(cumdist[-1])

        start_s = float(cumdist[t])
        end_s = min(start_s + self.min_distance_unit, total_s)
        chunk_dist = end_s - start_s
        _, end_idx = _dist_interval_indices(cumdist, start_s, end_s)
        num_steps = max(1, end_idx - t)

        if chunk_dist < self.zero_dist_epsilon:
            reason = "distance_below_epsilon"
        elif num_steps > self.max_steps_per_chunk:
            reason = "too_many_steps_for_distance_unit"
        else:
            reason = None

        if reason is not None:
            end = min(end_idx, n - 1)
            alphas = np.linspace(0.0, 1.0, self.M)
            return ArcChunkToken(
                kind="zero",
                mode=self.mode.value,
                pos=np.repeat(pos[t : t + 1], self.M, axis=0),
                ypr=slerp_pair_ypr(ypr[t], ypr[end], alphas),
                gripper=(1.0 - alphas[:, None]) * gripper[t]
                + alphas[:, None] * gripper[end],
                trans_vel=_zero_velocity(self.mode, self.M),
                start_idx=t,
                end_idx=end_idx,
                num_original_steps=num_steps,
                chunk_distance=chunk_dist,
                dt=self.dt,
                zero_reason=reason,
            )

        pos_rs, ypr_rs, grip_rs = resample_by_distance(
            pos, ypr, gripper, cumdist, start_s, end_s, self.M, start_idx=t
        )
        return ArcChunkToken(
            kind="motion",
            mode=self.mode.value,
            pos=pos_rs,
            ypr=ypr_rs,
            gripper=grip_rs,
            trans_vel=make_velocity_payload(pos_rs, num_steps, self.dt, self.mode),
            start_idx=t,
            end_idx=end_idx,
            num_original_steps=num_steps,
            chunk_distance=chunk_dist,
            dt=self.dt,
        )

    def tokenize_per_timestep(
        self,
        pos: np.ndarray,
        ypr: np.ndarray,
        gripper: np.ndarray | None = None,
    ) -> list[ArcChunkToken]:
        """One (overlapping) token per timestep, time-synced with observations."""
        pos, ypr, gripper = self._validate_inputs(pos, ypr, gripper)
        return [self.tokenize_at(pos, ypr, gripper, t=t) for t in range(len(pos))]

    # -- detokenize ---------------------------------------------------------

    def detokenize(
        self, tokens: list[ArcChunkToken]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reconstruct an approximate time-indexed trajectory from consecutive tokens.

        returns: (pos (N, 3), ypr (N, 3), gripper (N, G)). Consecutive chunks
        share their boundary waypoint, so each chunk after the first drops its
        first reconstructed step.
        """
        pos_parts: list[np.ndarray] = []
        ypr_parts: list[np.ndarray] = []
        grip_parts: list[np.ndarray] = []
        for token in tokens:
            p, r, g = self._detokenize_one(token)
            if pos_parts:
                p, r, g = p[1:], r[1:], g[1:]
            pos_parts.append(p)
            ypr_parts.append(r)
            grip_parts.append(g)
        if not pos_parts:
            return (
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0, 1), dtype=np.float64),
            )
        return (
            np.concatenate(pos_parts, axis=0),
            np.concatenate(ypr_parts, axis=0),
            np.concatenate(grip_parts, axis=0),
        )

    def _detokenize_one(
        self, token: ArcChunkToken
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos_rs = np.asarray(token.pos, dtype=np.float64)
        ypr_rs = np.asarray(token.ypr, dtype=np.float64)
        grip_rs = np.asarray(token.gripper, dtype=np.float64)

        if token.kind == "zero":
            n = max(2, int(token.num_original_steps))
            alphas = np.linspace(0.0, 1.0, n)
            return (
                np.repeat(pos_rs[0:1], n, axis=0),
                slerp_pair_ypr(ypr_rs[0], ypr_rs[-1], alphas),
                (1.0 - alphas[:, None]) * grip_rs[0] + alphas[:, None] * grip_rs[-1],
            )

        n_steps = max(2, self._infer_steps(token))
        src = np.linspace(0.0, 1.0, len(pos_rs))
        tgt = np.linspace(0.0, 1.0, n_steps)

        pos_out = np.stack(
            [np.interp(tgt, src, pos_rs[:, d]) for d in range(3)], axis=-1
        )
        ypr_out = slerp_through_ypr(ypr_rs, n_steps)
        grip_out = np.stack(
            [np.interp(tgt, src, grip_rs[:, d]) for d in range(grip_rs.shape[1])],
            axis=-1,
        )
        return pos_out, ypr_out, grip_out

    def _infer_steps(self, token: ArcChunkToken) -> int:
        """Number of time-indexed waypoints to emit for a motion chunk.

        The chunk covers num_original_steps step-intervals, i.e.
        num_original_steps + 1 positions; the velocity payload tells us how
        many control steps the chunk's distance corresponds to.
        """
        pos_rs = np.asarray(token.pos, dtype=np.float64)
        vel = np.asarray(token.trans_vel, dtype=np.float64)
        dt = float(token.dt)
        seg_d = np.linalg.norm(pos_rs[1:] - pos_rs[:-1], axis=-1)
        total_dist = float(seg_d.sum())

        if total_dist < self.zero_dist_epsilon:
            return max(2, int(token.num_original_steps) + 1)

        mode = VelocityMode(token.mode)
        if mode == VelocityMode.MEAN_SCALAR:
            v = max(float(vel[0]), 1e-8)
            return max(2, int(np.ceil(total_dist / (v * dt))) + 1)
        if mode == VelocityMode.MEAN_PER_DIM:
            v = max(float(np.linalg.norm(vel)), 1e-8)
            return max(2, int(np.ceil(total_dist / (v * dt))) + 1)
        if mode == VelocityMode.PER_STEP_SCALAR:
            v = np.maximum(vel[:-1], 1e-8)
            seg_steps = np.ceil(seg_d / (v * dt)).astype(int)
            return max(2, int(np.sum(np.maximum(seg_steps, 1))) + 1)
        if mode == VelocityMode.PER_STEP_PER_DIM:
            v = np.maximum(np.linalg.norm(vel[:-1], axis=-1), 1e-8)
            seg_steps = np.ceil(seg_d / (v * dt)).astype(int)
            return max(2, int(np.sum(np.maximum(seg_steps, 1))) + 1)
        raise ValueError(f"Unsupported velocity mode: {mode}")


# ---------------------------------------------------------------------------
# Bimanual layer: canonical (T, 14) chunk <-> (M, 14 + 2 * vel_dim) arc chunk
# ---------------------------------------------------------------------------


@dataclass
class BimanualArcLengthConfig:
    """Config for the bimanual arc-length tokenizer.

    min_distance_unit: meters of per-arm EEF travel covered by one token.
    resampled_vector_length: M, waypoints per token (the post-tokenization
        chunk length).
    mode: velocity payload mode; detokenize supports the mean modes only.
    dt: seconds per source time step (control period at deploy time).
    """

    min_distance_unit: float = 0.05
    resampled_vector_length: int = 20
    mode: str = "mean_scalar"
    dt: float = 1.0 / 30.0
    zero_dist_epsilon: float = 1e-6
    max_steps_per_chunk: int = 200


class BimanualArcLengthTokenizer:
    """Arc-length tokenizer over the canonical bimanual cartesian layout.

    tokenize: (T, 14) time-indexed chunk -> (M, arc_dim) arc-indexed chunk,
    where each arm's block is [xyz(3) ypr(3) grip(1) vel(velocity_dim)] —
    arc_dim is 16 for the scalar velocity modes and 20 for the per-dim modes.
    Each arm is tokenized against its own cumulative arc length, so a
    stationary arm holds pose (zero velocity) while the other arm covers its
    distance unit. The token starts at t=0 of the input chunk (time-synced
    with the observation) and spans the next min_distance_unit of that arm's
    travel.

    tokenize_with_joints additionally resamples per-arm joint chunks (T, J)
    at the same M waypoints along that arm's arc-length parameterization
    (zero tokens hold the initial joints), mirroring the GR00T action
    transform's joint handling.

    detokenize: (M, arc_dim) -> (H, 14) time-indexed chunk sampled at the
    control period ``dt`` under the mean-velocity constant-speed assumption
    (modes "mean_scalar" and "mean_per_dim"; per-step timing is not wired):
    control step k sits at arc length v * k * dt along the token (clamped to
    the token span), and every channel is interpolated there against the
    waypoints' own cumulative arc length. The trans_vel columns are consumed
    to set the timing, not returned.
    """

    def __init__(self, config: BimanualArcLengthConfig | None = None):
        self.config = config or BimanualArcLengthConfig()
        self._core = ArcLengthTokenizer(
            min_distance_unit=self.config.min_distance_unit,
            resampled_vector_length=self.config.resampled_vector_length,
            mode=self.config.mode,
            dt=self.config.dt,
            zero_dist_epsilon=self.config.zero_dist_epsilon,
            max_steps_per_chunk=self.config.max_steps_per_chunk,
        )

    @property
    def M(self) -> int:
        return self._core.M

    @property
    def velocity_dim(self) -> int:
        return velocity_dim(self._core.mode)

    @property
    def arc_arm_dim(self) -> int:
        return ARM_DIM + self.velocity_dim

    @property
    def arc_dim(self) -> int:
        return 2 * self.arc_arm_dim

    def tokenize(self, chunk: np.ndarray) -> np.ndarray:
        """args: chunk (T, 14) [L xyz ypr grip | R xyz ypr grip], T >= 2.

        returns: (M, arc_dim) [L xyz ypr grip vel | R xyz ypr grip vel].
        """
        actions, _, _ = self.tokenize_with_joints(chunk)
        return actions

    def tokenize_with_joints(
        self,
        chunk: np.ndarray,
        joints_left: np.ndarray | None = None,
        joints_right: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Tokenize the bimanual chunk plus optional per-arm joint chunks.

        args:
            chunk: (T, 14) bimanual cartesian chunk.
            joints_left / joints_right: optional (T, J) absolute joint targets,
                resampled along that arm's arc length at the same M waypoints.
        returns:
            (actions (M, arc_dim), joints_left (M, J) or None,
             joints_right (M, J) or None)
        """
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] != BIMANUAL_CARTESIAN_DIM:
            raise ValueError(
                f"BimanualArcLengthTokenizer.tokenize expects (T, "
                f"{BIMANUAL_CARTESIAN_DIM}), got {chunk.shape}"
            )
        joints = []
        for name, j in (("joints_left", joints_left), ("joints_right", joints_right)):
            if j is None:
                joints.append(None)
                continue
            j = np.asarray(j, dtype=np.float64)
            if j.ndim == 1:
                j = j[:, None]
            if j.ndim != 2 or len(j) != len(chunk):
                raise ValueError(
                    f"{name} must have shape (T, J) with T={len(chunk)}, got {j.shape}"
                )
            joints.append(j)

        if np.any(np.abs(chunk) >= INVALID_POSE_THRESHOLD):
            actions = np.full((self.M, self.arc_dim), INVALID_POSE_FILL)
            filled = [
                None if j is None else np.full((self.M, j.shape[1]), INVALID_POSE_FILL)
                for j in joints
            ]
            return actions, filled[0], filled[1]

        arms = []
        joints_out: list[np.ndarray | None] = []
        for arm_start, j in ((0, joints[0]), (ARM_DIM, joints[1])):
            arm = chunk[:, arm_start : arm_start + ARM_DIM]
            pos = arm[:, 0:3]
            token = self._core.tokenize_at(
                pos=pos, ypr=arm[:, 3:6], gripper=arm[:, 6:7], t=0
            )
            vel = _broadcast_velocity(token.trans_vel, self._core.mode, self.M)
            arms.append(
                np.concatenate([token.pos, token.ypr, token.gripper, vel], axis=-1)
            )
            joints_out.append(
                None if j is None else self._resample_arm_joints(j, pos, token)
            )
        return np.concatenate(arms, axis=-1), joints_out[0], joints_out[1]

    def _resample_arm_joints(
        self, joints: np.ndarray, pos: np.ndarray, token: ArcChunkToken
    ) -> np.ndarray:
        """Resample (T, J) joints at the token's M waypoints along the arm's
        arc length. Zero tokens hold the initial joints (matching the token's
        held pose); motion tokens interpolate against the arm's cumdist with
        the first waypoint anchored at the token's start timestep."""
        if token.kind == "zero":
            return np.repeat(
                joints[token.start_idx : token.start_idx + 1], self.M, axis=0
            )
        cumdist = cumulative_arc_length(pos)
        start_s = float(cumdist[token.start_idx])
        targets = np.linspace(start_s, start_s + token.chunk_distance, self.M)
        out = np.stack(
            [_interp_linear_at_s(joints, cumdist, float(s)) for s in targets], axis=0
        )
        out[0] = joints[token.start_idx]
        return out

    def detokenize(self, arc_chunk: np.ndarray, action_horizon: int) -> np.ndarray:
        """args: arc_chunk (M, arc_dim), action_horizon H.

        returns: (H, 14) time-indexed chunk at the control period.
        """
        actions, _, _ = self.detokenize_with_joints(arc_chunk, action_horizon)
        return actions

    def detokenize_with_joints(
        self,
        arc_chunk: np.ndarray,
        action_horizon: int,
        joints_left: np.ndarray | None = None,
        joints_right: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Detokenize the arc chunk plus optional arc-indexed joint chunks.

        args:
            arc_chunk: (M, arc_dim) tokenized bimanual chunk.
            joints_left / joints_right: optional (M, J) arc-indexed joints (as
                produced by tokenize_with_joints), interpolated at the same
                control timestamps against that arm's waypoint arc length.
        returns:
            (actions (H, 14), joints_left (H, J) or None,
             joints_right (H, J) or None)
        """
        mode = self._core.mode
        if mode not in (VelocityMode.MEAN_SCALAR, VelocityMode.MEAN_PER_DIM):
            raise NotImplementedError(
                "Arc-length detokenize supports the mean modes ('mean_scalar', "
                f"'mean_per_dim') only, got {mode.value!r}"
            )
        arc_chunk = np.asarray(arc_chunk, dtype=np.float64)
        if arc_chunk.ndim != 2 or arc_chunk.shape[1] != self.arc_dim:
            raise ValueError(
                f"BimanualArcLengthTokenizer.detokenize expects (M, "
                f"{self.arc_dim}), got {arc_chunk.shape}"
            )
        m = arc_chunk.shape[0]
        joints = []
        for name, j in (("joints_left", joints_left), ("joints_right", joints_right)):
            if j is None:
                joints.append(None)
                continue
            j = np.asarray(j, dtype=np.float64)
            if j.ndim == 1:
                j = j[:, None]
            if j.ndim != 2 or len(j) != m:
                raise ValueError(
                    f"{name} must have shape (M, J) with M={m}, got {j.shape}"
                )
            joints.append(j)

        h = int(action_horizon)
        dt = self._core.dt
        vd = self.velocity_dim

        arms = []
        joints_out: list[np.ndarray | None] = []
        for arm_start, j in ((0, joints[0]), (self.arc_arm_dim, joints[1])):
            arm = arc_chunk[:, arm_start : arm_start + self.arc_arm_dim]
            pos, ypr, grip = arm[:, 0:3], arm[:, 3:6], arm[:, 6:7]
            vel = arm[0, ARM_DIM : ARM_DIM + vd]
            v = max(float(vel[0]) if vd == 1 else float(np.linalg.norm(vel)), 0.0)

            cumdist = cumulative_arc_length(pos)
            total = float(cumdist[-1])
            if total < 1e-9 or v <= 1e-8:
                pos_t = np.repeat(pos[:1], h, axis=0)
                ypr_t = np.repeat(ypr[:1], h, axis=0)
                grip_t = np.repeat(grip[:1], h, axis=0)
                j_t = None if j is None else np.repeat(j[:1], h, axis=0)
            else:
                s = np.minimum(v * np.arange(h, dtype=np.float64) * dt, total)
                pos_t = np.stack(
                    [_interp_pos_at_s(pos, cumdist, float(sk)) for sk in s]
                )
                ypr_t = np.stack(
                    [_interp_ypr_at_s(ypr, cumdist, float(sk)) for sk in s]
                )
                grip_t = np.stack(
                    [_interp_linear_at_s(grip, cumdist, float(sk)) for sk in s]
                )
                j_t = (
                    None
                    if j is None
                    else np.stack(
                        [_interp_linear_at_s(j, cumdist, float(sk)) for sk in s]
                    )
                )
            arms.append(np.concatenate([pos_t, ypr_t, grip_t], axis=-1))
            joints_out.append(j_t)
        return np.concatenate(arms, axis=-1), joints_out[0], joints_out[1]


class TokenizeBimanualArcLength:
    """Transform-style adapter: arc-length-tokenize a bimanual cartesian chunk.

    Implements the same ``transform(batch: dict) -> dict`` interface as
    ``action_chunk_transforms.Transform`` (duck-typed so this module stays
    importable without torch/projectaria). Reads ``action_key`` (T, 14) from
    the batch and writes the (M, arc_dim) tokenized chunk to
    ``output_action_key``. If ``left_joint_key`` / ``right_joint_key`` are
    given, those (T, J) chunks are resampled along the matching arm's arc
    length and written to ``output_*_joint_key`` (default: ``<key>_arc``).
    """

    def __init__(
        self,
        action_key: str = "actions_cartesian",
        output_action_key: str = "actions_arc",
        config: BimanualArcLengthConfig | None = None,
        left_joint_key: str | None = None,
        right_joint_key: str | None = None,
        output_left_joint_key: str | None = None,
        output_right_joint_key: str | None = None,
    ):
        self.action_key = action_key
        self.output_action_key = output_action_key
        self.left_joint_key = left_joint_key
        self.right_joint_key = right_joint_key
        self.output_left_joint_key = output_left_joint_key or (
            f"{left_joint_key}_arc" if left_joint_key else None
        )
        self.output_right_joint_key = output_right_joint_key or (
            f"{right_joint_key}_arc" if right_joint_key else None
        )
        self.tokenizer = BimanualArcLengthTokenizer(config)

    def transform(self, batch: dict) -> dict:
        joints_left = (
            np.asarray(batch[self.left_joint_key]) if self.left_joint_key else None
        )
        joints_right = (
            np.asarray(batch[self.right_joint_key]) if self.right_joint_key else None
        )
        actions, joints_left_rs, joints_right_rs = self.tokenizer.tokenize_with_joints(
            np.asarray(batch[self.action_key]), joints_left, joints_right
        )
        batch[self.output_action_key] = actions
        if self.left_joint_key:
            batch[self.output_left_joint_key] = joints_left_rs
        if self.right_joint_key:
            batch[self.output_right_joint_key] = joints_right_rs
        return batch


# Canonical layout for the (M+1, 14) arc-token variant used by the FM policy
# head (see hpt_cotrain_mecka_flow_shared_head_arc.yaml). Per-slot dims:
#   [L xyz(3), L ypr(3), L grip(1), R xyz(3), R ypr(3), R grip(1)] = 14
# Geometric overlay/DTW utilities retain these progress measures; the ARC codec
# itself uses independently capped streams only.
ARC_CHUNKING_MODES = ("race", "multistream", "joint_distance")


def resolve_arc_chunking_mode(mode=None, rotation_distance_unit=None):
    # Source-window and geometric utilities retain their existing defaults.
    # The Cartesian codec does not use this selector.
    if mode is None:
        return "joint_distance" if rotation_distance_unit is not None else "multistream"
    value = str(mode).strip().lower()
    if value not in ARC_CHUNKING_MODES:
        raise ValueError(
            f"Unknown arc_chunking_mode geometric progress measure: {value!r}"
        )
    return value


CLOCK_MODES = ("velocity", "duration")
STREAM_MODES = ("full_separation", "translation_rotation", "translation", "projection")
# translation / projection: one clock per arm; rotation and gripper ride its supports.
STREAM_NAMES = {
    "full_separation": ("translation", "rotation", "gripper"),
    "translation_rotation": ("translation", "rotation"),
    "translation": ("translation",),
    "projection": ("translation",),
}
STREAM_DIMS = {
    "full_separation": 20,
    "translation_rotation": 18,
    "translation": 16,
    "projection": 16,
}
# Timing columns always follow geometry; four- and two-stream order is the same prefix.
SIX_CLOCK_COLUMNS = (0, 7, 3, 10, 6, 13)
CLOCK_COLUMNS = (0, 7, 3, 10)
SINGLE_CLOCK_COLUMNS = (0, 7)
# ARC Projection: progress is measured at p + R @ offset, 2 cm along each local axis.
PROJECTION_OFFSET_M = (0.02, 0.02, 0.02)
# Max speed of an absolute translation clock, m/s. There is no min speed: a slow arm
# also times the rotation and gripper riding its clock, and a floor replays them fast.
MAX_SPEED = 5.0


def validate_clock_mode(mode):
    if mode not in CLOCK_MODES:
        raise ValueError(f"clock_mode must be one of {CLOCK_MODES}, got {mode!r}")
    return mode


def validate_streams(streams):
    if streams not in STREAM_MODES:
        raise ValueError(f"streams must be one of {STREAM_MODES}, got {streams!r}")
    return streams


def bimanual_arc_token_shape(
    resampled_vector_length, clock_mode="velocity", streams="full_separation"
):
    validate_clock_mode(clock_mode)
    validate_streams(streams)
    m = int(resampled_vector_length)
    if m < 2:
        raise ValueError("resampled_vector_length must be at least two")
    return m, STREAM_DIMS[streams]


def stream_distances(waypoints, off, stream, streams, offset=PROJECTION_OFFSET_M):
    """Per-segment distance one arm's stream covers between emitted waypoints."""
    if stream == "translation":
        xyz = waypoints[:, off : off + 3]
        if streams == "projection":
            ypr = waypoints[:, off + 3 : off + 6]
            xyz = xyz + R.from_euler("ZYX", ypr).apply(np.asarray(offset, dtype=float))
        return np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    if stream == "rotation":
        return np.diff(cumulative_rotation_length(waypoints[:, off + 3 : off + 6]))
    return np.abs(np.diff(waypoints[:, off + 6]))


def absolute_times_to_seconds(times, distances, max_speed=None):
    """Absolute waypoint seconds -> start delay and incoming interval seconds.

    Elmo's speed-bounded isotonic decode (PR #266): project the predicted times
    onto the nearest (least-squares) non-decreasing clock whose moving intervals
    last at least distance / max_speed, by isotonic regression on the slack above
    that fastest clock. A backwards pair is averaged instead of drifting later
    rows late, and no local error sums into later rows. Row zero is the start
    delay and stays >= 0. Without motion, or without a bound, this is plain
    isotonic regression.
    """
    t = np.asarray(times, dtype=np.float64)
    cum = np.r_[0.0, np.cumsum(np.asarray(distances, dtype=np.float64))]
    if t.shape != cum.shape or not np.isfinite(t).all():
        raise ValueError("Absolute times require one finite time per waypoint")
    moving = cum[-1] > 0.0
    fastest = cum / max_speed if moving and max_speed else np.zeros_like(cum)
    slack = np.maximum(isotonic_regression(t - fastest).x, 0.0)
    clock = fastest + slack
    return np.r_[clock[0], np.diff(clock)]


def clock_token_seconds(token, clock_mode, streams=None, offset=PROJECTION_OFFSET_M):
    """Timing rows as seconds: start delay, then incoming intervals.

    Velocity row zero is already seconds. Duration rows are absolute seconds from
    the chunk anchor, decoded once by :func:`absolute_times_to_seconds` with the
    translation max speed; rotation and gripper clocks are only made monotone.
    """
    validate_clock_mode(clock_mode)
    value = np.array(token, dtype=np.float64)
    if value.ndim < 2 or value.shape[-1] not in (16, 18, 20):
        raise ValueError("Expected an ARC token with 16, 18 or 20 columns")
    if not np.isfinite(value).all():
        raise ValueError("ARC tokens must be finite")
    if clock_mode == "duration":
        streams = streams or {20: "full_separation", 18: "translation_rotation"}.get(
            value.shape[-1], "translation"
        )
        if STREAM_DIMS[validate_streams(streams)] != value.shape[-1]:
            raise ValueError(f"{streams} tokens have {STREAM_DIMS[streams]} columns")
        rows = value.reshape(-1, *value.shape[-2:])
        for row in rows:
            for index, name in enumerate(STREAM_NAMES[streams]):
                for arm, off in enumerate((0, 7)):
                    col = 14 + 2 * index + arm
                    row[:, col] = absolute_times_to_seconds(
                        row[:, col],
                        stream_distances(row, off, name, streams, offset),
                        MAX_SPEED if name == "translation" else None,
                    )
        value = rows.reshape(value.shape)
    return value


def stack_arc_token(token):
    """Internal geometry/timing view, not a selectable serialized layout."""
    value = np.asarray(token, dtype=np.float64)
    if value.shape[-1] not in (16, 18, 20):
        raise ValueError("Only the explicit 16D/18D/20D ARC contracts are supported")
    cols = {20: SIX_CLOCK_COLUMNS, 18: CLOCK_COLUMNS, 16: SINGLE_CLOCK_COLUMNS}[
        value.shape[-1]
    ]
    timing = np.zeros(value.shape[:-1] + (14,))
    timing[..., cols] = value[..., 14:]
    return np.concatenate((value[..., :14], timing), axis=-2)


class TokenizeBimanualArcLengthCartesian:
    """First-delay ARC: M x (14 geometry + two, four or six clocks).

    Row zero holds start delay. Rows i>0 describe incoming interval i-1 -> i.
    Velocity: seconds in row zero, emitted segment distance/second thereafter.
    Duration: absolute seconds from the chunk anchor in every timing row (row
    zero is the start delay), decoded by speed-bounded isotonic regression.
    ``translation`` and ``projection`` give each arm one clock; rotation and the
    gripper ride its supports. ``projection`` measures that clock at the point
    ``p + R @ projection_offset_m``.
    There is no automatic interpretation of historical checkpoint layouts.
    """

    def __init__(
        self,
        action_key="actions_cartesian",
        output_action_key="actions_cartesian",
        min_distance_unit=0.60,
        rotation_distance_unit=1.7453292519943295,
        gripper_distance_unit=None,
        gripper_stream_mode="events",
        resampled_vector_length=100,
        dt=1 / 30,
        zero_dist_epsilon=1e-6,
        hold_window_frames=400,
        preserve_action_key=None,
        preserve_action_rows=None,
        clock_mode="velocity",
        streams="full_separation",
        projection_offset_m=PROJECTION_OFFSET_M,
    ):
        from types import SimpleNamespace

        self.clock_mode = validate_clock_mode(clock_mode)
        self.streams = validate_streams(streams)
        self.projection_offset_m = np.asarray(projection_offset_m, dtype=np.float64)
        if (
            self.projection_offset_m.shape != (3,)
            or not np.isfinite(self.projection_offset_m).all()
            or not self.projection_offset_m.any()
        ):
            raise ValueError("projection_offset_m must be a finite nonzero 3-vector")
        self.action_key, self.output_action_key = action_key, output_action_key
        self.preserve_action_key, self.preserve_action_rows = (
            preserve_action_key,
            preserve_action_rows,
        )
        self.shape = bimanual_arc_token_shape(
            resampled_vector_length, clock_mode, streams
        )
        for name, value in (
            ("D", min_distance_unit),
            ("R", rotation_distance_unit),
            ("dt", dt),
            ("zero_dist_epsilon", zero_dist_epsilon),
        ):
            if value is None or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if int(hold_window_frames) < 1:
            raise ValueError("hold_window_frames must be positive")
        if preserve_action_rows is not None and int(preserve_action_rows) < 1:
            raise ValueError("preserve_action_rows must be positive")
        if gripper_stream_mode not in ("events", "distance"):
            raise ValueError("gripper_stream_mode must be events or distance")
        if streams == "full_separation" and gripper_stream_mode == "distance":
            if (
                gripper_distance_unit is None
                or not np.isfinite(gripper_distance_unit)
                or gripper_distance_unit <= 0
            ):
                raise ValueError(
                    "Separate distance gripper streams require positive gripper_distance_unit"
                )
        self.rotation_distance_unit = float(rotation_distance_unit)
        self.gripper_distance_unit = gripper_distance_unit
        self.gripper_stream_mode = gripper_stream_mode
        self.hold_window_frames = int(hold_window_frames)
        self.zero_dist_epsilon = float(zero_dist_epsilon)
        self.tokenizer = SimpleNamespace(
            M=self.shape[0],
            config=SimpleNamespace(
                dt=float(dt),
                min_distance_unit=float(min_distance_unit),
                zero_dist_epsilon=self.zero_dist_epsilon,
            ),
        )

    @property
    def M(self):
        return self.shape[0]

    @property
    def _stream_names(self):
        return STREAM_NAMES[self.streams]

    def transform(self, batch):
        raw = np.asarray(batch[self.action_key], dtype=np.float64)
        if (
            raw.ndim != 2
            or raw.shape[1] != 14
            or len(raw) < 2
            or not np.isfinite(raw).all()
        ):
            raise ValueError("ARC source must be finite (T>=2, 14)")
        if self.preserve_action_key is not None:
            preserved = raw.copy()
            if self.preserve_action_rows is not None:
                n = int(self.preserve_action_rows)
                preserved = preserved[:n]
                if len(preserved) < n:
                    preserved = np.concatenate(
                        (
                            preserved,
                            np.repeat(preserved[-1:], n - len(preserved), axis=0),
                        )
                    )
            batch[self.preserve_action_key] = preserved
        waypoints = self._hybrid_waypoints(raw)
        clocks = self._stream_targets(raw, waypoints)
        if self.clock_mode == "duration":
            clocks = np.cumsum(clocks, axis=0)  # start delay + intervals -> absolute
        batch[self.output_action_key] = np.concatenate((waypoints, clocks), axis=1)
        return batch

    def _hybrid_source_coordinates(self, raw):
        translation, rotation = [], []
        for off in (0, 7):
            cum = cumulative_arc_length(self._progress(raw, off))
            end = min(float(cum[-1]), self.tokenizer.config.min_distance_unit)
            hold = len(raw) - 1 if cum[-1] < self.zero_dist_epsilon else None
            translation.append((cum, np.linspace(0, end, self.M), hold))
            cum = cumulative_rotation_length(raw[:, off + 3 : off + 6])
            rotation.append(
                (
                    cum,
                    np.linspace(
                        0, min(float(cum[-1]), self.rotation_distance_unit), self.M
                    ),
                )
            )
        return translation, rotation

    def _progress(self, values, off):
        """Points whose arc length is the translation stream's progress."""
        xyz = values[:, off : off + 3]
        if self.streams != "projection":
            return xyz
        ypr = values[:, off + 3 : off + 6]
        return xyz + R.from_euler("ZYX", ypr).apply(self.projection_offset_m)

    def _translation_values_at_targets(self, values, cumulative, targets):
        i, a = _first_crossing_brackets(cumulative, targets)
        return values[i] + a[:, None] * (values[i + 1] - values[i])

    def _translation_source_times(self, cumulative, targets, dt):
        i, a = _first_crossing_brackets(cumulative, targets)
        return (i + a) * dt

    def _stream_targets(self, raw, waypoints):
        translation, rotation = self._hybrid_source_coordinates(raw)
        gripper = (
            self._hybrid_gripper_coordinates(raw)
            if self.streams == "full_separation"
            else None
        )
        clocks = {}
        dt = self.tokenizer.config.dt
        for arm, off in enumerate((0, 7)):
            t_cum, t_targets, hold = translation[arm]
            r_cum, r_targets = rotation[arm]
            specs = [
                ("translation", t_cum, t_targets, hold is not None),
                ("rotation", r_cum, r_targets, r_cum[-1] < self.zero_dist_epsilon),
            ][: len(self._stream_names)]
            if gripper is not None:
                cum, targets, held = gripper[arm]
                specs.append(("gripper", cum, targets, held))
            for name, cum, targets, held in specs:
                if name == "gripper" and self.gripper_stream_mode == "events":
                    times = self._gripper_event_frames(raw[:, off + 6]) * dt
                    if not held:
                        times[0] = self._motion_start_time(cum, dt)
                elif held:
                    times = np.linspace(0, self.hold_window_frames * dt, self.M)
                else:
                    times = self._translation_source_times(cum, targets, dt)
                    times[0] = self._motion_start_time(cum, dt)
                intervals = np.maximum(np.diff(times), 0)
                if self.clock_mode == "velocity":
                    distance = self._distances(waypoints, off, name)
                    if held:
                        distance[:] = 0
                    intervals = np.divide(
                        distance,
                        intervals,
                        out=np.zeros_like(distance),
                        where=intervals > 1e-12,
                    )
                clocks[name, arm] = np.r_[times[0], intervals]
        return np.stack(
            [clocks[name, arm] for name in self._stream_names for arm in (0, 1)],
            axis=1,
        )

    def _distances(self, waypoints, off, stream):
        return stream_distances(
            waypoints, off, stream, self.streams, self.projection_offset_m
        )

    def _native(self, token, clocks_in_seconds=False):
        value = np.asarray(token, dtype=np.float64)
        if value.ndim != 2 or value.shape[1] != self.shape[1] or len(value) < 2:
            raise ValueError(f"Expected (M>=2, {self.shape[1]}) ARC token")
        if not np.isfinite(value).all():
            raise ValueError("ARC tokens must be finite")
        if clocks_in_seconds and self.clock_mode != "duration":
            raise ValueError("clocks_in_seconds is only valid for duration clocks")
        return (
            value.copy()
            if clocks_in_seconds
            else clock_token_seconds(
                value, self.clock_mode, self.streams, self.projection_offset_m
            )
        )

    def _intervals(self, delta, stored, horizon):
        if self.clock_mode == "duration":
            return np.maximum(stored, 0)
        moving, usable = delta > 1e-12, np.abs(stored) > 1e-8
        if not np.any(moving):
            return np.full_like(
                delta, self.hold_window_frames * self.tokenizer.config.dt / len(delta)
            )
        out = np.divide(
            delta, np.abs(stored), out=np.zeros_like(delta), where=moving & usable
        )
        out[moving & ~usable] = self.tokenizer.config.dt * (horizon + 1)
        return out

    def control_steps(self, token, *, max_steps=None, clocks_in_seconds=False):
        value = self._native(token, clocks_in_seconds)
        ends = []
        for index, name in enumerate(self._stream_names):
            for arm, off in enumerate((0, 7)):
                delta = self._distances(value, off, name)
                moving = delta > 1e-12
                if not moving.any():
                    continue
                stored = value[1:, 14 + index * 2 + arm]
                # Decoded absolute clocks are monotone: a zero interval is a move
                # within one tick, not missing timing.
                usable = (
                    np.abs(stored) > 1e-8
                    if self.clock_mode == "velocity"
                    else np.ones_like(moving)
                )
                if np.any(moving & ~usable):
                    ends.append(float("inf"))
                else:
                    ends.append(
                        max(value[0, 14 + index * 2 + arm], 0)
                        + self._intervals(delta, stored, max_steps or 1).sum()
                    )
        duration = (
            min(ends) if ends else self.hold_window_frames * self.tokenizer.config.dt
        )
        if not np.isfinite(duration):
            if max_steps is None:
                raise ValueError(
                    "ARC prefix timing does not define finite replan boundary"
                )
            return int(max_steps)
        steps = max(1, int(np.ceil(duration / self.tokenizer.config.dt - 1e-9)))
        return min(steps, int(max_steps)) if max_steps is not None else steps

    def detokenize(self, arc_actions, action_horizon, *, clocks_in_seconds=False):
        value = self._native(arc_actions, clocks_in_seconds)
        h = int(action_horizon)
        if h < 1:
            raise ValueError("action_horizon must be positive")
        query = np.arange(h) * self.tokenizer.config.dt
        arms = []
        for arm, off in enumerate((0, 7)):
            indices, clocks = {}, {}
            for index, name in enumerate(self._stream_names):
                col = 14 + index * 2 + arm
                durations = self._intervals(
                    self._distances(value, off, name), value[1:, col], h
                )
                delay = max(value[0, col], 0)
                indices[name] = self._hybrid_clock_indices(durations, h, delay)
                clocks[name] = delay + np.r_[0, np.cumsum(durations)]
            lo, hi, a = indices["translation"]
            xyz = value[:, off : off + 3]
            xyz = xyz[lo] + a[:, None] * (xyz[hi] - xyz[lo])
            rlo, _, ra = indices.get("rotation", indices["translation"])
            ypr = _slerp_segments_ypr(value[:, off + 3 : off + 6], rlo, ra)
            grip = value[:, off + 6 : off + 7]
            if self.streams == "full_separation":
                lo, hi, a = indices["gripper"]
            if (
                self.streams == "full_separation"
                and self.gripper_stream_mode == "events"
            ):
                clock = clocks["gripper"]
                tolerance = 8 * np.finfo(np.float32).eps * max(1.0, clock[-1])
                gi = np.searchsorted(clock, query + tolerance, side="right") - 1
                grip = grip[np.clip(gi, 0, len(grip) - 1)]
            else:
                grip = grip[lo] + a[:, None] * (grip[hi] - grip[lo])
            arms.append(np.concatenate((xyz, ypr, grip), axis=1))
        return np.concatenate(arms, axis=1)

    def _hybrid_gripper_coordinates(self, raw: np.ndarray):
        """Independent normalized travel coordinates for left/right grippers."""
        result = []
        for offset in (6, 13):
            cumulative = np.concatenate(
                ([0.0], np.cumsum(np.abs(np.diff(raw[:, offset]))))
            )
            end = (
                float(cumulative[-1])
                if self.gripper_stream_mode == "events"
                else min(self.gripper_distance_unit, float(cumulative[-1]))
            )
            result.append(
                (
                    cumulative,
                    np.linspace(0.0, end, self.M),
                    cumulative[-1] < self.tokenizer.config.zero_dist_epsilon,
                )
            )
        return result

    def _gripper_event_frames(self, values: np.ndarray) -> np.ndarray:
        """PR #202 supports: initial value, command changes, endpoint, padding.

        Gripper values are held commands, not interpolated apertures. A full
        hold carries the configured source-window duration in duration mode.
        Like PR #202, rate tokens cannot encode a terminal stationary dwell.
        """
        changes = np.flatnonzero(np.abs(np.diff(values)) > 1e-9) + 1
        endpoint = len(values) - 1 if len(changes) else self.hold_window_frames
        required = np.unique(np.r_[0, changes, endpoint]).astype(np.float64)
        if len(required) > self.M:
            required = required[
                np.rint(np.linspace(0, len(required) - 1, self.M)).astype(int)
            ]
        return np.r_[required, np.repeat(required[-1], self.M - len(required))]

    @staticmethod
    def _motion_start_time(cumulative: np.ndarray, dt: float) -> float:
        """Last frame of the initial stationary plateau; no smoothing/floor."""
        moving = np.flatnonzero(np.diff(cumulative) > 1e-12)
        return float(moving[0] * dt) if len(moving) else 0.0

    def _hybrid_clock_indices(
        self, duration: np.ndarray, action_horizon: int, start_delay: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        elapsed = start_delay + np.concatenate(([0.0], np.cumsum(duration)))
        targets = self.tokenizer.config.dt * np.arange(action_horizon, dtype=np.float64)
        upper = np.searchsorted(elapsed, targets, side="right")
        upper = np.clip(upper, 1, len(elapsed) - 1)
        lower = upper - 1
        start = elapsed[lower]
        end = elapsed[upper]
        alpha = np.divide(
            targets - start,
            end - start,
            out=np.zeros_like(targets),
            where=(end - start) > self.zero_dist_epsilon,
        )
        return lower, upper, np.clip(alpha, 0.0, 1.0)

    def _hybrid_waypoints(self, raw: np.ndarray) -> np.ndarray:
        """Resample both streams against their own source coordinates."""
        translation, rotation = self._hybrid_source_coordinates(raw)
        gripper = (
            self._hybrid_gripper_coordinates(raw)
            if self.streams == "full_separation"
            else None
        )

        arms = []
        for offset, (translation_cumulative, translation_targets, hold), (
            rotation_cumulative,
            rotation_targets,
        ) in zip((0, 7), translation, rotation):
            rotation_indices, rotation_alpha = _bracket_segments(
                rotation_cumulative, rotation_targets
            )
            if hold is None:
                xyz = self._translation_values_at_targets(
                    raw[:, offset : offset + 3],
                    translation_cumulative,
                    translation_targets,
                )
            else:
                # A held translation stream remains constant for the window.
                xyz = np.repeat(raw[0:1, offset : offset + 3], self.M, axis=0)
            if gripper is not None:
                grip_cumulative, grip_targets, grip_hold = gripper[offset // 7]
                if self.gripper_stream_mode == "events":
                    frames = self._gripper_event_frames(raw[:, offset + 6])
                    indices = np.minimum(frames.astype(int), len(raw) - 1)
                    grip = raw[indices, offset + 6 : offset + 7]
                elif grip_hold:
                    grip = np.repeat(raw[0:1, offset + 6 : offset + 7], self.M, axis=0)
                else:
                    grip_index, grip_alpha = _first_crossing_brackets(
                        grip_cumulative, grip_targets
                    )
                    grip_values = raw[:, offset + 6 : offset + 7]
                    grip = (1.0 - grip_alpha[:, None]) * grip_values[
                        grip_index
                    ] + grip_alpha[:, None] * grip_values[grip_index + 1]
            elif hold is None:
                grip = self._translation_values_at_targets(
                    raw[:, offset + 6 : offset + 7],
                    translation_cumulative,
                    translation_targets,
                )
            else:
                grip = np.repeat(raw[0:1, offset + 6 : offset + 7], self.M, axis=0)
            if "rotation" in self._stream_names:
                # Rotation keeps its own clock: a translation hold must not
                # cut the independent rotation stream short.
                ypr = _slerp_segments_ypr(
                    raw[:, offset + 3 : offset + 6],
                    rotation_indices,
                    rotation_alpha,
                )
                still = (
                    rotation_cumulative[-1] < self.tokenizer.config.zero_dist_epsilon
                )
            else:  # one clock per arm: rotation at the translation supports
                ypr = _slerp_segments_ypr(
                    raw[:, offset + 3 : offset + 6],
                    *_first_crossing_brackets(
                        translation_cumulative, translation_targets
                    ),
                )
                still = hold is not None
            if still:
                ypr = np.repeat(raw[0:1, offset + 3 : offset + 6], self.M, axis=0)
            arms.append(np.concatenate([xyz, ypr, grip], axis=-1))
        return np.concatenate(arms, axis=-1)
