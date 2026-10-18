"""E1 (mixed-speed protocol) variants of the bimanual arc-length tokenizer.

Additive subclass of ``TokenizeBimanualArcLengthCartesian`` — nothing upstream
changes. Three things over the parent:

* **Vectorized tokenize / detokenize with the parent's exact semantics.** The
  parent resamples rotation with one scipy ``Slerp`` object per waypoint
  (≈ 200 Rotation/Slerp constructions per sample), which measured 1 s per
  sample in loader workers — 25 min per 100-batch epoch. Here the bracketing
  is one ``searchsorted``, positions / grippers are one linear blend, and
  rotation is one batched ``rot_i * exp(alpha · log(rot_i⁻¹ rot_{i+1}))`` —
  the same geodesic scipy's ``Slerp`` walks. ``e1_data_smoke.py --check-parent``
  asserts agreement with the parent to float precision.

* ``velocity_norm="path"`` — the trailing velocity row keeps the chord
  direction but its magnitude becomes the token's PATH speed (arc length the
  token covers / time it took) instead of chord / time. ``detokenize`` walks the
  waypoint polyline at ``||vel||``, so the upstream chord-norm token runs slow on
  any curved motion (Step 0 on mecka fold chunks: token speed / measured path
  speed = 0.23, d_clock 2.8 s median). "path" is the protocol's Arc-mean.

* ``velocity_mode="profile"`` — Arc+Vel. No velocity row; every waypoint row
  carries the per-arm speed at that arc-length position as two extra columns:
  ``(M, 16) = [14 canonical | v_L(u_m), v_R(u_m)]``. ``detokenize`` integrates
  the clock, t(u) = ∫ du / v(u) with v piecewise linear between waypoints.
  The waypoint speeds are the raw 30 Hz chunk's path speed (7-frame moving
  average — see ``chunk_speed`` for why not a Butterworth here).

* ``velocity_mode="logdur"`` — the tempo-invariance ablation's channel
  (Ideas note *Arc Tokenizer Tempo Invariance Changes*, #1 + #2). Same
  ``(M, 16)`` layout, but per arm the extra column holds, in row 0, the log of
  the token's mean slowness ``log(T_span / span)`` (s/m — the one scalar that
  carries tempo) and, in rows 1..M-1, the log of each waypoint segment's
  duration relative to the mean segment duration (a tempo-normalized profile
  that sums to the token's time by construction). ``detokenize`` rebuilds the
  clock as a cumulative sum of durations — no division by a near-zero speed,
  and a timing error is a *relative* error at any tempo.

* ``velocity_mode="dur"`` — the port of Ryan's duration-timed codec
  (EgoVerse-graph ``6d1b5f93``, "duration + progress instead of velocity").
  Same ``(M, 16)`` layout, but the per-arm column holds the **elapsed seconds**
  of each waypoint interval, absolute and unlogged: rows ``0..M-2`` are the
  intervals and row ``M-1`` repeats the last one (his ``_sample_stream`` pads
  the channel to length M the same way, and his decoder reads only the first
  M-1). ``detokenize`` sums them into the clock — no division, no log, no
  scalar/profile split. Ryan's argument is that ``_sample_stream`` computes
  each interval's ``delta_t`` and then divides it away to make a rate, which
  the decoder's first act is to invert; duration is the primitive quantity and
  it represents a hold exactly. Differences from our ``logdur``, which already
  banked the no-division half of that argument, are in the class docstring of
  ``durations_to_clock_abs``.

* ``velocity_mode="profhyb"`` (variant ``arcvelhyb``) — ``durhyb``'s token with SPEEDS in place of
  durations: the same (M, 18) layout and waypoints (xyz + gripper on the translation arc, ypr on
  the arm's own rotation arc), but rows ``0..M-2`` of each timing column hold the interval's mean
  speed, its waypoint-polyline segment length over its duration (m/s for translation, rad/s for
  rotation), and row ``M-1`` keeps the start delay. Decode turns speeds back into durations with the
  same segment lengths and runs ``durhyb``'s clock, so the two carry identical timing content and
  differ only in parameterization, as ``arcvel`` and ``arcdur`` do. A hold (a stream with no length)
  is all zeros and decodes spread over the decode horizon, which is exactly where ``durhyb``'s uniform
  hold durations put it. (2026-10-03: the first version stored the hold's 3.3 s in row ``M-1``, a row
  that is ~always 0 on Aria data; per-element quantile normalization (q99 - q1 = 0) turned it into
  6.6e8 and the cotrain run diverged.)

* ``velocity_mode="durtri"`` / ``"proftri"`` (variants ``arcdurtri`` / ``arcveltri``) — three streams per arm:
  ``durhyb`` / ``profhyb`` with the gripper moved off the translation arc onto its own stream, sampled along
  its own travel (sum |d grip|, over the whole window) at M evenly spaced targets and timed by its own column:
  ``(M, 20) = [14 canonical | translation L, R | rotation L, R | gripper L, R]``. Row ``M-1`` of the gripper
  column is its start delay; a gripper hold (travel under ``GRIPPER_HOLD_EPSILON``) keeps grip[0] and an
  all-zero column. So a grasp that closes after the arm stops, or while it turns in place, keeps its timing
  (``durhyb`` samples it on the translation arc and loses it once translation ends).

* ``velocity_mode="durevt"`` / ``"stkevt"`` (variants ``arcdurevt`` / ``arcvelevt``) — Ryan's LIBERO stream codec
  (EgoVerse-graph ryan-streams ``libero_arc_timed`` / ``libero_arc_grouped``) on the bimanual YAM, in ``durtri``'s
  ``(M, 20)`` column order. Translation and rotation supports come from his ``_support_frames`` on the real arcs
  (uniform in arc; the duration form also reserves the longest stationary stretches, steps of ``zero_dist_epsilon``
  = 1e-6 m / rad per frame or less where his default is 1e-9, as zero-distance intervals), and
  every timing column holds its stream's intervals in rows ``0..M-2`` with row ``M-1`` REPEATING row ``M-2``: there is
  no start-delay row. ``durevt`` stores elapsed seconds, ``stkevt`` the distance between the encoded supports over
  the elapsed time (m/s, rad/s). The gripper is an EVENT list instead of a travel-sampled stream: supports at the
  window start, the start and end of each gripper motion (the ``GRIPPER_MAX_EVENTS`` largest) and the window end,
  padded by repeating the last one, decoded as linear ramps with holds between them. Its column is SECONDS in both
  forms: a dwell between two ramps has no value change, so a rate would give it zero time. Decode is a plain
  cumulative clock per stream (never ``_stream_index``, which would read row ``M-1`` as a start delay).

* ``velocity_mode="durref"`` / ``"stkref"`` (variants ``arcdurref`` / ``arcvelref``) — Ryan's two-clock 'reference'
  token (``LiberoArcTimedCodec``; ``translation_groups [[0, 1, 2]]``, gripper on the translation stream) with the
  rotation stored as rotation6d: ``(M, 24) = [L xyz, L rotation6d, L grip | R xyz, R rotation6d, R grip |
  translation timing L, R | rotation timing L, R]``. Supports and timing columns are ``durevt`` / ``stkevt``'s
  (his ``_support_frames`` on the real arcs, row ``M-1`` repeats row ``M-2``, plain cumulative clocks), except that
  the translation stream has NO budget: like his runs it spans the window's whole path, so ``min_distance_unit`` is
  not read. The gripper is the aperture interpolated linearly at the translation supports and decoded linearly on
  the translation clock (his held binary command is a step; ours is continuous). Decode maps any six numbers back
  to a rotation with his Gram-Schmidt ``_from_rotation6d`` and returns ZYX Euler. The fixed-scale normalization and
  the clipped sampling that go with his token live in the recipe (``norm_stats.fixed_ranges``, the model leaf's
  ``clip_sample_range``), not here.

* ``fixed_spacing`` — theory design rule 1: waypoints are ALWAYS ``h = D / (M - 1)`` apart.
  A partial token (path shorter than D inside the window — 29 % of arm-tokens on ABC
  skirts, 50 % on stationery, 8 % on mecka) keeps its first ``n_valid`` waypoints at
  that spacing and repeats the last valid pose in the remaining rows (a plateau the
  polyline's cumulative arc length makes self-describing at decode), instead of
  stretching M waypoints over whatever path exists and silently changing the
  token's geometric scale. logdur rows past the valid length carry 0 and zero-length
  segments are excluded from the clock at decode.

* ``progress_smooth_hz`` — (#4) low-pass the chunk's positions (zero-phase
  4th-order Butterworth, the Step 0 filter) before cumulative arc length is
  accumulated, so ``D`` metres of *measured* progress is the same true path at
  every tempo (Thm 3.7: the raw polyline length of a jittery track grows with
  samples per metre, i.e. with slowness). Waypoint 0 stays the raw anchor.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation as R

from egomimic.rldb.zarr.action_chunk_transforms import Transform
from egomimic.rldb.zarr.arc_length_tokenizer import (
    TokenizeBimanualArcLengthCartesian,
    cumulative_arc_length,
)
from egomimic.utils.pose_utils import _ypr_to_rot6d

try:  # the parent fills chunks with out-of-range poses; mirror that rule
    from egomimic.rldb.zarr.arc_length_tokenizer import (
        INVALID_POSE_FILL,
        INVALID_POSE_THRESHOLD,
    )
except ImportError:  # pragma: no cover
    INVALID_POSE_THRESHOLD, INVALID_POSE_FILL = np.inf, 0.0

# (xyz offset, ypr offset, grip offset, velocity-row xyz slice) per arm in the
# canonical 14-dim layout [L xyz ypr grip | R xyz ypr grip].
ARM_LAYOUT = ((0, 3, 6, slice(0, 3)), (7, 10, 13, slice(7, 10)))
E1_ARCVEL_DIM = 16
# durhyb: [14 canonical | translation dt L, R | rotation dt L, R]
E1_ARCDURHYB_DIM = (
    18  # also profhyb: [14 canonical | translation v L, R | rotation omega L, R]
)
# durtri / proftri: [14 canonical | translation L, R | rotation L, R | gripper L, R] -- the gripper gets its own stream
E1_ARCDURTRI_DIM = 20
TRI_MODES = ("durtri", "proftri")
HYBRID_MODES = ("durhyb", "profhyb", *TRI_MODES)
SPEED_MODES = (
    "profhyb",
    "proftri",
)  # timing columns hold interval speeds, not durations
# durevt / stkevt: Ryan's stream codec with an event-list gripper. Deliberately NOT hybrid / tri / speed modes: their
# row M-1 repeats row M-2 (no start delay) and their decode is a plain cumulative clock per stream.
EVT_MODES = ("durevt", "stkevt")
EVT_RATE_FLOORS = (1e-3, 1e-3)  # stkevt / stkref decode: m/s, rad/s
# durref / stkref: Ryan's two-clock 'reference' token with rotation6d,
# [L xyz 0:3 | L rotation6d 3:9 | L grip 9 | R xyz 10:13 | R rotation6d 13:19 | R grip 19 | translation timing L, R |
# rotation timing L, R]. NOT hybrid / tri / speed / evt modes: none of the canonical-14 code may read this layout.
E1_ARCREF_DIM = 24
REF_MODES = ("durref", "stkref")
REF_ARM_LAYOUT = ((0, 3, 9), (10, 13, 19))  # (xyz, rotation6d, grip) offsets per arm
# Gripper events of durevt / stkevt (aperture in [0, 1]). On the slowpace366 validation windows (every 5th, 2026-10-11)
# 70 % of the per-frame aperture steps are exactly 0 and 85 % are under 1e-3 (0.03 / s), and an arm-window holds
# 0 / 1 / 2 / 3 / more motions in 53.3 / 24.6 / 15.7 / 5.4 / 0.99 %.
# Speeds are aperture / s: a motion contains a step at least HI fast (0.01 / frame at 30 Hz) and all its steps are
# at least LO fast (0.005 / frame); same-direction runs MERGE_GAP frames apart are one motion; a motion changes the
# aperture by MIN_TRAVEL or more; over MAX_EVENTS only the largest motions (by aperture change) are kept and a dropped
# one is absorbed by the dwell around it (keeping the FIRST ones let a contact bounce take a slot and turned a later
# full open / close into a seconds-long linear drift: GT gripper MSE over the window 0.31e-3 against 0.083e-3, worst
# aperture error 0.85 against 0.17); FIT are the rise-time levels that place a ramp's start / end in time.
GRIPPER_EVENT_SPEED_HI = 0.3
GRIPPER_EVENT_SPEED_LO = 0.15
GRIPPER_EVENT_MERGE_GAP = 1
GRIPPER_EVENT_MIN_TRAVEL = 0.03
GRIPPER_MAX_EVENTS = 3
GRIPPER_EVENT_FIT = (0.2, 0.8)
# Rotation budget of the hybrid token: a full turn, i.e. the rotation stream spans the whole source window.
# The lab hybrid's 24 deg (R24deg) fits a chunk that ENDS when a budget is spent; the E1 YAM window is a fixed
# 100 frames, and on rl2 stationery the wrist turns more than 24 deg in 63 % of arm-windows (median 52 deg,
# p90 123 deg), so a 24 deg stream froze the wrist for the rest of the window: GT round-trip geodesic error
# 11.7 deg vs 1.1 deg for plain arcdur, and 0.07 deg with this budget (1,500 val windows, 2026-09-30).
DEFAULT_ROTATION_DISTANCE_UNIT = 2 * np.pi
# durtri / proftri: a gripper stream whose total travel (sum |d grip|, grip in [0, 1]) is under this is a hold. On 1,500
# rl2 YAM stationery 100-frame windows (2026-10-06) 28.5 % of arm-windows have travel < 1e-6 and only 0.6 % more fall in
# [1e-6, 1e-3); above that travel is spread continuously. Aria human grippers are always 0 (a hold).
GRIPPER_HOLD_EPSILON = 1e-3
# Deploy-side stream tests (``stream_clocks`` / ``stream_ends``, read by egomimic.robot.arc_decoder): a wrist whose
# total geodesic turn is under HYBRID_ROT_EPS is still, and a gripper stream whose aperture range is under
# TRI_GRIPPER_MOVE_RANGE is a hold. Neither touches tokenize.
HYBRID_ROT_EPS = 1e-6
TRI_GRIPPER_MOVE_RANGE = 0.05


def _dur_col_to_speed(col: np.ndarray, seg: np.ndarray, eps: float) -> np.ndarray:
    """durhyb timing column -> profhyb: rows 0..M-2 become segment length / duration; row M-1 (start
    delay) is kept. A hold (stream length under ``eps``) is all zeros: nothing to time."""
    out = np.zeros_like(col)
    if float(seg.sum()) < eps:
        return out
    out[:-1] = seg / np.maximum(col[:-1], LOGDUR_MIN_DT)
    out[-1] = col[-1]
    return out


def _speed_col_to_dur(
    col: np.ndarray,
    seg: np.ndarray,
    eps: float,
    hold_time: float,
    floor: float | None = None,
) -> np.ndarray:
    """profhyb timing column -> the durhyb column it encodes (inverse of ``_dur_col_to_speed``). A hold
    gets durhyb's uniform hold durations over ``hold_time`` (the decode horizon).

    ``floor`` (opt-in, ``speed_floors``): an interval's duration is seg / max(|speed|, floor). Without it one predicted
    speed <= 0 becomes seg / 1e-6, a 1e2-1e5 s interval that freezes the stream for the rest of the chunk."""
    col = np.asarray(col, dtype=np.float64)
    M = len(col)
    out = np.zeros(M)
    if float(seg.sum()) < eps:
        out[:-1] = float(hold_time) / max(M - 1, 1)
        return out
    speed = (
        np.maximum(col[:-1], 1e-6)
        if floor is None
        else np.maximum(np.abs(col[:-1]), floor)
    )
    out[:-1] = seg / speed
    out[-1] = col[-1]
    return out


def rotation_arc_length(ypr: np.ndarray) -> np.ndarray:
    """Cumulative geodesic rotation angle (rad) along a (T, 3) ZYX-euler track."""
    ypr = np.asarray(ypr, dtype=np.float64)
    if len(ypr) < 2:
        return np.zeros(len(ypr))
    rot = R.from_euler("ZYX", ypr)
    return np.concatenate(([0.0], np.cumsum((rot[:-1].inv() * rot[1:]).magnitude())))


def gripper_arc_length(grip: np.ndarray) -> np.ndarray:
    """Cumulative gripper travel (sum of |d grip|) along a (T,) or (T, 1) track."""
    g = np.asarray(grip, dtype=np.float64).reshape(-1)
    return np.concatenate(([0.0], np.cumsum(np.abs(np.diff(g)))))


def _lerp_rows(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Rows of x at fractional indices f (linear between neighbours, clamped)."""
    n = len(x)
    if n < 2:
        return np.repeat(x[:1], len(f), 0)
    f = np.clip(np.asarray(f, dtype=np.float64), 0.0, n - 1)
    i = np.minimum(np.floor(f).astype(int), n - 2)
    a = (f - i)[:, None]
    return (1.0 - a) * x[i] + a * x[i + 1]


def _slerp_rows(ypr: np.ndarray, f: np.ndarray) -> np.ndarray:
    """ZYX-euler rows at fractional indices f, geodesic between neighbours."""
    n = len(ypr)
    if n < 2:
        return np.repeat(ypr[:1], len(f), 0)
    f = np.clip(np.asarray(f, dtype=np.float64), 0.0, n - 1)
    i = np.minimum(np.floor(f).astype(int), n - 2)
    return _slerp_vec(R.from_euler("ZYX", ypr), i, f - i).as_euler("ZYX", degrees=False)


def _from_rotation6d(values):
    """(..., 6) = [first matrix column, second matrix column] -> rotations, by Gram-Schmidt. Any six numbers give a
    valid rotation: a first column of norm under 1e-8 becomes (1, 0, 0), and a second column parallel to the first
    is replaced by the unit axis least aligned with it (an all-zero row is the identity). Verbatim from Ryan's
    ``libero_arc.py`` (EgoVerse-graph ryan-streams ``f335bbe8``); ``pose_utils._rot6d_to_ypr`` raises on such rows."""
    a, b = values[..., :3].copy(), values[..., 3:].copy()
    bad = np.linalg.norm(a, axis=-1) < 1e-8
    a[bad] = (1, 0, 0)
    a /= np.linalg.norm(a, axis=-1, keepdims=True)
    b -= (a * b).sum(axis=-1, keepdims=True) * a
    bad = np.linalg.norm(b, axis=-1) < 1e-8
    if bad.any():
        fallback = np.eye(3)[np.argmin(np.abs(a[bad]), axis=-1)]
        b[bad] = fallback - (fallback * a[bad]).sum(axis=-1, keepdims=True) * a[bad]
    b /= np.linalg.norm(b, axis=-1, keepdims=True)
    return R.from_matrix(np.stack((a, b, np.cross(a, b)), axis=-1))


# ---------------------------------------------------------------------------
# durevt / stkevt: stream supports (Ryan) and gripper events
# ---------------------------------------------------------------------------
def _bracket(cumulative, target):
    """Fractional frame at which ``cumulative`` first reaches ``target``: the FIRST frame of an initial or interior
    plateau, the last frame only when ``target >= cumulative[-1]``. ``np.interp`` returns the last frame of every
    plateau, i.e. it would skip a start dwell. Verbatim from Ryan's ``libero_arc_timed.py`` (EgoVerse-graph
    ryan-streams ``f335bbe8``)."""
    if target <= cumulative[0]:
        return 0.0
    if target >= cumulative[-1]:
        return float(len(cumulative) - 1)
    lower = int(np.searchsorted(cumulative, target, side="left")) - 1
    span = cumulative[lower + 1] - cumulative[lower]
    return lower + (target - cumulative[lower]) / span


def _support_frames(cumulative, end, count, mode, epsilon=1e-9):
    """Uniform arc allocation, reserving longest dwells for duration tokens.

    Verbatim from Ryan's ``libero_arc_timed.py`` (EgoVerse-graph ryan-streams ``f335bbe8``), whose dwell reservation
    and quota allocation follow duration_support_frames in the pinned native planar codec, including its fixed
    support budget. Returns ``count`` fractional source frames, first 0: ``mode='stk'`` spaces them uniformly in arc
    up to ``end``; ``mode='dur'`` also makes both edges of the longest stopped stretches (steps <= ``epsilon``)
    supports, so a dwell is its own zero-distance interval.
    """
    frames = np.arange(len(cumulative), dtype=np.float64)
    if end <= epsilon:
        return np.linspace(0, frames[-1], count)
    targets = np.linspace(0, end, count)
    uniform = np.array([_bracket(cumulative, t) for t in targets])
    if mode == "stk":
        return uniform
    stopped = np.diff(cumulative) <= epsilon
    if not stopped.any():
        return uniform
    starts = np.flatnonzero(stopped & np.r_[True, ~stopped[:-1]])
    ends = np.flatnonzero(stopped & np.r_[~stopped[1:], True]) + 1
    end_frame = uniform[-1]
    holds = [
        (float(a), min(float(b), end_frame))
        for a, b in zip(starts, ends)
        if a < end_frame
    ]
    selected = {0.0, end_frame}
    for start, stop in sorted(holds, key=lambda hold: (hold[0] - hold[1], hold[0])):
        candidate = selected | {start, stop}
        if len(candidate) <= count:
            selected = candidate
    boundaries = np.array(sorted(selected))
    if len(boundaries) == 2:
        return uniform
    boundary_arc = np.interp(boundaries, frames, cumulative)
    ranks = np.linspace(0, 1, count)
    boundary_rank = np.interp(boundary_arc, targets, ranks)
    weights = np.diff(boundary_rank)
    quotas = (count - len(boundaries)) * weights / weights.sum()
    counts = 1 + np.floor(quotas).astype(int)
    remaining = count - 1 - counts.sum()
    counts[np.argsort(-(quotas - np.floor(quotas)), kind="stable")[:remaining]] += 1
    supports = [0.0]
    for index, pieces in enumerate(counts):
        if boundary_arc[index + 1] - boundary_arc[index] <= epsilon:
            supports.extend(
                np.linspace(boundaries[index], boundaries[index + 1], pieces + 1)[1:-1]
            )
        else:
            rank = np.linspace(
                boundary_rank[index], boundary_rank[index + 1], pieces + 1
            )[1:-1]
            supports.extend(
                _bracket(cumulative, t) for t in np.interp(rank, ranks, targets)
            )
        supports.append(boundaries[index + 1])
    return np.asarray(supports)


def _level_crossing(x: np.ndarray, level: float, sign: float, a: int, b: int):
    """First fractional frame in [a, b] where x crosses ``level`` going in direction ``sign``; None if never."""
    y = sign * (x[a : b + 1] - level)
    idx = np.flatnonzero((y[:-1] < 0) & (y[1:] >= 0))
    if len(idx) == 0:
        return None
    i = idx[0]
    return a + i + (-y[i]) / (y[i + 1] - y[i])


def gripper_event_supports(grip: np.ndarray, dt: float):
    """Event supports of one gripper track (T,): (times in fractional frames, aperture values), always
    ``[0, start_1, end_1, ..., start_n, end_n, T-1]`` with n <= ``GRIPPER_MAX_EVENTS`` and duplicates kept, so even
    intervals are dwells and odd intervals are ramps (no motion gives ``[0, T-1]``). Over the cap the
    ``GRIPPER_MAX_EVENTS`` largest motions are kept (ties: the earliest); a dropped one is absorbed by the dwell
    around it, whose ends then differ by that motion's travel.

    A motion is a run of same-sign steps of at least ``GRIPPER_EVENT_SPEED_LO`` (runs ``GRIPPER_EVENT_MERGE_GAP``
    apart with no reversal between them are one: a duplicated sample inside a ramp) that contains a step of at least
    ``GRIPPER_EVENT_SPEED_HI`` and moves the aperture by ``GRIPPER_EVENT_MIN_TRAVEL`` or more. Its two supports keep
    the aperture at its first and last frame but sit, in time, on the line through its ``GRIPPER_EVENT_FIT`` level
    crossings: the ramps are S-shaped, and a straight line between the raw first / last frames starts too early
    (GT gripper MSE over the first 25 ticks 0.57e-3 against 0.076e-3 with the fit). Exact for a one-frame step."""
    g = np.asarray(grip, dtype=np.float64).reshape(-1)
    s = np.diff(g)
    v_hi, v_lo = GRIPPER_EVENT_SPEED_HI * dt, GRIPPER_EVENT_SPEED_LO * dt
    motions = []
    for sign in (1.0, -1.0):
        m = np.r_[False, sign * s >= v_lo, False]
        edges = np.flatnonzero(m[1:] != m[:-1])
        runs = []
        for a, b in zip(edges[0::2], edges[1::2]):
            if (
                runs
                and a - runs[-1][1] <= GRIPPER_EVENT_MERGE_GAP
                and not np.any(sign * s[runs[-1][1] : a] < -v_lo)
            ):
                runs[-1][1] = b
            else:
                runs.append([a, b])
        motions += [
            (int(a), int(b))
            for a, b in runs
            if (sign * s[a:b]).max() >= v_hi
            and abs(g[b] - g[a]) >= GRIPPER_EVENT_MIN_TRAVEL
        ]
    kept = []
    for a, b in sorted(motions):
        if kept:  # a bridged gap can make opposite-direction motions overlap
            a = max(a, kept[-1][1])
        if b > a:
            kept.append((a, b))
    # Stable sort by travel, so ties keep the earliest; then back in time order.
    kept = sorted(
        sorted(kept, key=lambda m: -abs(g[m[1]] - g[m[0]]))[:GRIPPER_MAX_EVENTS]
    )
    lo, hi = GRIPPER_EVENT_FIT
    times, values = [0.0], [g[0]]
    for a, b in kept:
        net = g[b] - g[a]
        t_lo = _level_crossing(g, g[a] + lo * net, np.sign(net), a, b)
        t_hi = _level_crossing(g, g[a] + hi * net, np.sign(net), a, b)
        if t_lo is None or t_hi is None or t_hi <= t_lo:
            ts, te = float(a), float(b)
        else:
            w = (t_hi - t_lo) / (hi - lo)
            ts, te = t_lo - lo * w, t_hi + (1 - hi) * w
        ts = min(max(ts, a, times[-1]), b)
        te = max(min(te, b), ts)
        times += [ts, te]
        values += [g[a], g[b]]
    times.append(float(len(g) - 1))
    values.append(g[-1])
    return np.asarray(times), np.asarray(values)


# ---------------------------------------------------------------------------
# vectorized arc-length resampling (parent semantics)
# ---------------------------------------------------------------------------
def _bracket_vec(cum: np.ndarray, targets: np.ndarray):
    """Vectorized ``_bracket_segment``: segment index i and alpha per target."""
    n = len(cum)
    i = np.searchsorted(cum, targets, side="left") - 1
    i = np.clip(i, 0, max(n - 2, 0))
    s0, s1 = cum[i], cum[np.minimum(i + 1, n - 1)]
    span = s1 - s0
    alpha = np.where(
        span > 1e-12, (targets - s0) / np.where(span > 1e-12, span, 1.0), 0.0
    )
    lo, hi = targets <= cum[0], targets >= cum[-1]
    i = np.where(lo, 0, np.where(hi, max(n - 2, 0), i))
    alpha = np.where(lo, 0.0, np.where(hi, 1.0, alpha))
    return i, np.clip(alpha, 0.0, 1.0)


def _slerp_vec(rot: R, i: np.ndarray, alpha: np.ndarray) -> R:
    """Geodesic interpolation between rot[i] and rot[i+1] at alpha (what scipy's Slerp does)."""
    n = len(rot)
    j = np.minimum(i + 1, n - 1)
    r0, r1 = rot[i], rot[j]
    rel = r0.inv() * r1
    return r0 * R.from_rotvec(rel.as_rotvec() * alpha[:, None])


def resample_at_s(pos, ypr, grip, cum, targets, rot: R | None = None):
    """Interpolate (pos, ypr, grip) at arc lengths ``targets`` against ``cum``.

    Same edge rules as the parent's ``_interp_*_at_s``: below cum[0] → first
    row, at/above cum[-1] → last row, alpha == 0 → the bracketing row itself.
    """
    i, alpha = _bracket_vec(cum, targets)
    j = np.minimum(i + 1, len(cum) - 1)
    a = alpha[:, None]
    pos_out = (1.0 - a) * pos[i] + a * pos[j]
    grip_out = (1.0 - a) * grip[i] + a * grip[j]
    rot = R.from_euler("ZYX", ypr) if rot is None else rot
    ypr_out = _slerp_vec(rot, i, alpha).as_euler("ZYX", degrees=False)
    exact0, exact1 = alpha <= 0.0, alpha >= 1.0
    ypr_out[exact0] = ypr[i[exact0]]
    ypr_out[exact1] = ypr[j[exact1]]
    return pos_out, ypr_out, grip_out


def chunk_speed(pos: np.ndarray, dt: float, smooth_frames: int = 7) -> np.ndarray:
    """Per-frame PATH speed of a (T, 3) chunk: d(arc length)/dt, smoothed with a
    centred ``smooth_frames`` moving average (edge-padded).

    Deliberately not an IIR low-pass: a zero-phase Butterworth on a 200-frame
    chunk has a ~10-frame transient at the anchor, exactly where the token's
    first waypoints (and E_time) live. Path speed rather than chord speed so the
    integral clock traverses the token's own arc length — jitter included — at
    the rate it was actually traversed.
    """
    pos = np.asarray(pos, dtype=np.float64)
    if len(pos) < 2:
        return np.zeros(len(pos))
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt
    v = np.concatenate([step, step[-1:]])
    w = max(1, int(smooth_frames))
    if w > 1 and len(v) >= w:
        vp = np.pad(v, (w // 2, w - 1 - w // 2), mode="edge")
        v = np.convolve(vp, np.ones(w) / w, mode="valid")
    return v


def integral_clock(cum: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """Time-of-progress at the waypoints: t(u_m) = ∫_0^{u_m} du / v(u)."""
    v = np.asarray(speed, dtype=np.float64)
    seg = np.diff(cum)
    dtime = seg * 0.5 * (1.0 / v[:-1] + 1.0 / v[1:])
    return np.concatenate(([0.0], np.cumsum(dtime)))


def lowpass_positions(pos: np.ndarray, fc_hz: float, fs_hz: float) -> np.ndarray:
    """Zero-phase 4th-order Butterworth low-pass of a (T, 3) track (Step 0's filter).
    Chunks too short for the filter's padding are returned unchanged."""
    pos = np.asarray(pos, dtype=np.float64)
    if fc_hz is None or fc_hz <= 0 or len(pos) < 20:
        return pos
    sos = butter(4, float(fc_hz), btype="low", fs=float(fs_hz), output="sos")
    return sosfiltfilt(sos, pos, axis=0)


LOGDUR_CLIP = np.log(20.0)  # |log(segment duration / mean segment duration)| cap
LOGDUR_MIN_DT = 1e-4  # s, floor on a segment duration at tokenize time


def durations_to_clock(
    col: np.ndarray, span, min_speed: float = 0.01, max_speed: float = 5.0
) -> np.ndarray:
    """logdur column (M,) + polyline span (m) -> time-of-progress at the M waypoints (s).

    The decoded mean speed is bounded to [min_speed, max_speed] — the same
    floor the profile mode's integral clock uses — so a wild row-0 prediction
    cannot make one chunk's clock dominate d_clock (an untrained head gave
    10^5 s in the launcher smoke). Tokenized values sit well inside the range
    (fold data: 0.10–0.42 m/s), so training targets are never clipped.
    """
    col = np.asarray(col, dtype=np.float64)
    cum = None
    if (
        np.ndim(span) > 0
    ):  # a cumulative-arc-length array: zero-length (padded) segments carry no time
        cum = np.asarray(span, dtype=np.float64)
        span = float(cum[-1])
    t_span = max(float(span), 1e-9) * np.exp(
        np.clip(col[0], -np.log(max_speed), -np.log(min_speed))
    )
    # rows 1.. are log(segment duration / mean segment duration): at tokenization their
    # exponentials sum to M-1 exactly, so normalising here is exact for true tokens and
    # keeps a predicted profile from rescaling the total time that row 0 owns.
    w = np.exp(np.clip(col[1:], -LOGDUR_CLIP, LOGDUR_CLIP))
    if cum is not None:
        w = w * (np.diff(cum) > 1e-9)
    seg = w / max(float(w.sum()), 1e-12) * t_span
    return np.concatenate(([0.0], np.cumsum(seg)))


def durations_to_clock_abs(
    col: np.ndarray, span, min_speed: float = 0.01, max_speed: float = 5.0
) -> np.ndarray:
    """``dur`` column (M,) + polyline span (m) -> time-of-progress at the M waypoints (s).

    Ryan's duration codec, ported (EgoVerse-graph ``6d1b5f93``). The channel is
    the elapsed SECONDS of each waypoint interval, so the clock is a running
    sum of it: ``t = [0, cumsum(dt_i)]``. Rows ``0..M-2`` are the intervals;
    row ``M-1`` repeats the last one and is never read, mirroring his
    ``_sample_stream`` padding.

    How this differs from ``durations_to_clock`` (our ``logdur``), which is the
    whole point of running it as a row:

    * **Absolute, not scalar x normalized profile.** logdur splits tempo into
      row 0 (log mean slowness, which owns the total time) and rows 1.. (log
      segment durations relative to the mean, renormalised at decode so a
      predicted profile cannot rescale the total). Here every interval carries
      its own absolute seconds and the total is just their sum, so a profile
      error does move the total.
    * **Linear, not log.** An L2 loss on this channel penalises absolute
      timing error; on logdur it penalises relative error at any tempo. Which
      one a mixed-tempo policy wants is exactly the open question.
    * **Unclipped.** logdur clamps ``|log(seg / mean seg)|`` at ``LOGDUR_CLIP``
      (= log 20), so a hold longer than 20x the mean segment is truncated;
      absolute duration represents it exactly. This is Ryan's "duration
      represents a hold exactly" point, and it is the one that bites on
      hold-heavy sources (ABC stationery: 54 % of frames).
    * His third argument -- the decoder reads only ``rate.abs()`` so the
      predicted sign of omega is dead -- does not transfer: our rotation rides
      the same per-arm arc clock as position and there is no angular rate
      channel to carry a dead sign.

    Durations are clamped non-negative: time cannot run backwards, and a
    negative prediction would make the clock non-monotone and break the
    progress lookup (his clamp too). The one guard that is ours and not his:
    if the implied mean speed leaves ``[min_speed, max_speed]`` the clock is
    rescaled uniformly -- the same bound ``durations_to_clock`` puts on
    logdur's row 0, kept here so neither row gets a safety net the other
    lacks. It never fires on true tokens (fold data 0.10-0.42 m/s).
    """
    col = np.asarray(col, dtype=np.float64)
    M = len(col)
    cum = None
    if (
        np.ndim(span) > 0
    ):  # a cumulative-arc-length array: zero-length (padded) segments carry no time
        cum = np.asarray(span, dtype=np.float64)
        span = float(cum[-1])
    seg = np.maximum(col[: M - 1], 0.0)
    if cum is not None:
        seg = seg * (np.diff(cum) > 1e-9)
    t_span = float(seg.sum())
    span = max(float(span), 1e-9)
    lo, hi = span / max_speed, span / min_speed
    if t_span > 1e-12 and not (lo <= t_span <= hi):
        seg = seg * (min(max(t_span, lo), hi) / t_span)
    return np.concatenate(([0.0], np.cumsum(seg)))


def absolute_times_to_clock(
    col: np.ndarray, cum: np.ndarray, min_speed: float = 0.01, max_speed: float = 5.0
) -> np.ndarray:
    """Waypoint seconds from the chunk anchor, without accumulating predictions.

    Anchor row zero at zero, project backwards predictions to a monotone clock,
    and hold padded supports at the preceding time. Retain the duration codec's
    whole-clock speed bounds. A local timing error never sums into later rows.
    """
    clock = np.asarray(col, dtype=np.float64).copy()
    cum = np.asarray(cum, dtype=np.float64)
    if clock.ndim != 1 or clock.shape != cum.shape or len(clock) < 2:
        raise ValueError("Absolute times require matching waypoint and arc vectors")
    if not np.isfinite(clock).all() or not np.isfinite(cum).all():
        raise ValueError("Absolute times and arc supports must be finite")
    clock[0] = 0.0
    for i in range(1, len(clock)):
        clock[i] = (
            max(clock[i - 1], clock[i], 0.0)
            if cum[i] - cum[i - 1] > 1e-9
            else clock[i - 1]
        )
    span = max(float(cum[-1]), 1e-9)
    lo, hi = span / max_speed, span / min_speed
    if clock[-1] > 1e-12 and not lo <= clock[-1] <= hi:
        clock *= min(max(clock[-1], lo), hi) / clock[-1]
    return clock


class CopyKeyRows(Transform):
    """``batch[dst] = batch[src][:n_rows]`` — carries the un-tokenized time chunk
    (``actions_time``) alongside the model target so the E1 evaluator can score
    every variant against the same 30 Hz ground truth."""

    def __init__(self, src: str, dst: str, n_rows: int):
        self.src, self.dst, self.n_rows = src, dst, int(n_rows)

    def transform(self, batch: dict) -> dict:
        x = np.asarray(batch[self.src], dtype=np.float64)
        if x.ndim == 0 or len(x) == 0:
            raise ValueError(f"{self.src!r} must contain at least one time row")
        out = x[: self.n_rows].copy()
        if len(out) < self.n_rows:
            # Keep the evaluator's control horizon rectangular at episode
            # tails. This is the same repeat-last convention used by the Zarr
            # loader and avoids variable-length ``actions_time`` tensors when
            # ARC's source window is shorter than the requested 100 steps.
            out = np.concatenate(
                [out, np.repeat(out[-1:], self.n_rows - len(out), axis=0)], axis=0
            )
        batch[self.dst] = out
        return batch


class TokenizeBimanualArcLengthE1(TokenizeBimanualArcLengthCartesian):
    def __init__(
        self,
        *,
        velocity_norm: str = "chord",
        velocity_mode: str = "mean",
        speed_smooth_frames: int = 7,
        duration_timing: str = "delta",
        min_speed: float = 0.01,
        progress_smooth_hz: float | None = None,
        fixed_spacing: bool = False,
        rotation_distance_unit: float = DEFAULT_ROTATION_DISTANCE_UNIT,
        gripper_hold_epsilon: float = GRIPPER_HOLD_EPSILON,
        speed_floors: Sequence[float] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fixed_spacing = bool(fixed_spacing)
        self.rotation_distance_unit = float(rotation_distance_unit)
        self.gripper_hold_epsilon = float(gripper_hold_epsilon)
        if velocity_norm not in ("chord", "path"):
            raise ValueError(
                f"velocity_norm must be 'chord' or 'path', got {velocity_norm!r}"
            )
        if velocity_mode not in (
            "mean",
            "profile",
            "logdur",
            "dur",
            *HYBRID_MODES,
            *EVT_MODES,
            *REF_MODES,
        ):
            raise ValueError(
                f"velocity_mode must be 'mean', 'profile', 'logdur', 'dur', 'durhyb', 'profhyb', 'durtri', 'proftri', 'durevt', 'stkevt', 'durref' or 'stkref', got {velocity_mode!r}"
            )
        if (
            velocity_mode in (*HYBRID_MODES, *EVT_MODES, *REF_MODES)
            and self.fixed_spacing
        ):
            raise ValueError(f"{velocity_mode} does not implement fixed_spacing")
        if velocity_mode in EVT_MODES and self.M < 2 * GRIPPER_MAX_EVENTS + 2:
            raise ValueError(
                f"{velocity_mode} needs at least {2 * GRIPPER_MAX_EVENTS + 2} waypoints for its gripper events, got {self.M}"
            )
        # Opt-in decode floors (translation m/s, rotation rad/s, gripper 1/s) for the speed columns: an interval lasts
        # distance / max(|speed|, floor). None keeps the legacy decode (stkevt / stkref: EVT_RATE_FLOORS).
        if speed_floors is not None:
            speed_floors = tuple(float(x) for x in speed_floors)
            if len(speed_floors) != 3 or not all(
                np.isfinite(x) and x > 0 for x in speed_floors
            ):
                raise ValueError(
                    f"speed_floors must be three finite positive numbers, got {speed_floors}"
                )
            if velocity_mode not in (*SPEED_MODES, "stkevt", "stkref"):
                raise ValueError(
                    f"speed_floors applies only to speed-column modes, not {velocity_mode!r}"
                )
        self.speed_floors = speed_floors
        if duration_timing not in ("delta", "absolute"):
            raise ValueError("duration_timing must be 'delta' or 'absolute'")
        if duration_timing != "delta" and velocity_mode != "dur":
            raise ValueError("Absolute waypoint timing requires velocity_mode='dur'")
        self.duration_timing = duration_timing
        self.velocity_norm = velocity_norm
        self.velocity_mode = velocity_mode
        self.speed_smooth_frames = int(speed_smooth_frames)
        self.min_speed = float(min_speed)
        self.progress_smooth_hz = (
            None if progress_smooth_hz in (None, 0, 0.0) else float(progress_smooth_hz)
        )

    @property
    def wide(self) -> bool:
        """(M, 16) layouts: a per-waypoint timing column per arm."""
        return self.velocity_mode in (
            "profile",
            "logdur",
            "dur",
            *HYBRID_MODES,
            *EVT_MODES,
            *REF_MODES,
        )

    @property
    def token_dim(self) -> int:
        if self.velocity_mode in REF_MODES:
            return E1_ARCREF_DIM
        if self.velocity_mode in (*TRI_MODES, *EVT_MODES):
            return E1_ARCDURTRI_DIM
        return E1_ARCDURHYB_DIM if self.velocity_mode in HYBRID_MODES else E1_ARCVEL_DIM

    def _tokenize_arm_hybrid(self, arm: np.ndarray):
        """durhyb, one arm (T, 7) -> (waypoints (M, 7), translation dt (M,), rotation dt (M,), gripper dt (M,) or None).

        xyz and gripper are sampled along the translation arc (budget D), ypr along the arm's own
        rotation arc (budget ``rotation_distance_unit``), each at M evenly spaced targets. Each
        stream stores the elapsed seconds of its intervals in rows 0..M-2 and, in row M-1 (padding
        in ``dur``), its START DELAY: the time before the stream's first motion, so a wrist turn
        that begins after the arm stops translating starts on time at decode. A translation hold
        keeps the arm put and times the gripper uniformly over the window (the plain token drops
        that gripper motion); a rotation hold keeps ypr[0].

        durtri: the gripper is a third stream instead, sampled along its own travel (sum |d grip|,
        the whole window) with its own duration column; a gripper hold (travel under
        ``gripper_hold_epsilon``) keeps grip[0] and an all-zero column, as a rotation hold does.
        """
        M, dt = self.M, self.tokenizer.config.dt
        eps = self.tokenizer.config.zero_dist_epsilon
        pos_raw, ypr, grip = arm[:, 0:3], arm[:, 3:6], arm[:, 6:7]
        n = len(pos_raw)
        frames = np.arange(n, dtype=np.float64)

        def targets(
            cum, budget, hold_eps=eps
        ):  # fractional source frame at each waypoint, or None for a hold
            span = min(float(budget), float(cum[-1])) if n > 1 else 0.0
            return (
                None
                if span < hold_eps
                else np.interp(np.linspace(0.0, span, M), cum, frames)
            )

        def durations(f):  # rows 0..M-2 interval seconds, row M-1 start delay
            seg = np.maximum(np.diff(f) * dt, LOGDUR_MIN_DT)
            return np.concatenate([seg, [max(float(f[0]), 0.0) * dt]])

        pos = self._progress_positions(pos_raw)
        f_t = targets(
            cumulative_arc_length(pos), self.tokenizer.config.min_distance_unit
        )
        if f_t is None:
            f_t = np.linspace(0.0, max(n - 1, 0), M)
            xyz = np.repeat(pos_raw[:1], M, 0)
        else:
            xyz = _lerp_rows(pos, f_t)
            xyz[0] = pos_raw[0]
        g_dt = None
        if self.velocity_mode in TRI_MODES:
            f_g = targets(gripper_arc_length(grip), np.inf, self.gripper_hold_epsilon)
            if f_g is None:
                grip_wp, g_dt = np.repeat(grip[:1], M, 0), np.zeros(M)
            else:
                grip_wp, g_dt = _lerp_rows(grip, f_g), durations(f_g)
                grip_wp[0] = grip[0]
        else:
            grip_wp = _lerp_rows(grip, f_t)
        f_r = targets(rotation_arc_length(ypr), self.rotation_distance_unit)
        if f_r is None:
            ypr_wp, r_dt = np.repeat(ypr[:1], M, 0), np.zeros(M)
        else:
            ypr_wp, r_dt = _slerp_rows(ypr, f_r), durations(f_r)
            ypr_wp[0] = ypr[0]
        t_dt = durations(f_t) if n > 1 else np.zeros(M)
        return np.concatenate([xyz, ypr_wp, grip_wp], axis=1), t_dt, r_dt, g_dt

    @staticmethod
    def _stream_clock(col: np.ndarray, span: float, lo: float, hi: float):
        """Time-of-progress at one hybrid stream's M waypoints (s), start delay included
        (rows 0..M-2 interval seconds, row M-1 start delay)."""
        M = len(col)
        if span > 1e-9:
            clock = durations_to_clock_abs(col, span, min_speed=lo, max_speed=hi)
        else:  # a hold: nothing to bound the speed by; time the rows as stored
            clock = np.concatenate(([0.0], np.cumsum(np.maximum(col[: M - 1], 0.0))))
        return clock + max(float(col[M - 1]), 0.0)

    @staticmethod
    def _stream_index(
        col: np.ndarray, span: float, t: np.ndarray, lo: float, hi: float
    ):
        """Fractional waypoint index at times t from one stream's duration column
        (rows 0..M-2 interval seconds, row M-1 start delay)."""
        M = len(col)
        clock = TokenizeBimanualArcLengthE1._stream_clock(col, span, lo, hi)
        if clock[-1] - clock[0] <= 1e-9:
            return np.zeros(len(t))
        return np.interp(t, clock, np.arange(M, dtype=np.float64))

    def as_duration_token(self, arc: np.ndarray, action_horizon: int) -> np.ndarray:
        """A hybrid / tri token with DURATION timing columns: profhyb / proftri speed columns are converted
        (segment length / speed; a held stream gets uniform hold durations over the decode horizon, the
        ``speed_floors`` apply) and duration tokens are returned as they are. Decode-side only."""
        arc = np.asarray(arc, dtype=np.float64)
        if self.velocity_mode not in SPEED_MODES:
            return arc
        out = arc.copy()
        h = int(action_horizon)
        eps, hold = (
            self.tokenizer.config.zero_dist_epsilon,
            self.tokenizer.config.dt * (h - 1),
        )
        floors = self.speed_floors or (None, None, None)
        for k, (xyz_off, ypr_off, grip_off, _) in enumerate(ARM_LAYOUT):
            cum_t = cumulative_arc_length(arc[:, xyz_off : xyz_off + 3])
            cum_r = rotation_arc_length(arc[:, ypr_off : ypr_off + 3])
            out[:, 14 + k] = _speed_col_to_dur(
                arc[:, 14 + k], np.diff(cum_t), eps, hold, floors[0]
            )
            out[:, 16 + k] = _speed_col_to_dur(
                arc[:, 16 + k], np.diff(cum_r), eps, hold, floors[1]
            )
            if self.velocity_mode in TRI_MODES:
                cum_g = gripper_arc_length(arc[:, grip_off : grip_off + 1])
                out[:, 18 + k] = _speed_col_to_dur(
                    arc[:, 18 + k],
                    np.diff(cum_g),
                    self.gripper_hold_epsilon,
                    hold,
                    floors[2],
                )
        return out

    def _detokenize_hybrid(
        self, arc: np.ndarray, h: int, t: np.ndarray | None = None
    ) -> np.ndarray:
        if t is None:
            t = self.tokenizer.config.dt * np.arange(h, dtype=np.float64)
        arc = self.as_duration_token(arc, h)
        arms = []
        for k, (xyz_off, ypr_off, grip_off, _) in enumerate(ARM_LAYOUT):
            xyz_wp = arc[:, xyz_off : xyz_off + 3]
            ypr_wp = arc[:, ypr_off : ypr_off + 3]
            grip_wp = arc[:, grip_off : grip_off + 1]
            cum_t, cum_r = cumulative_arc_length(xyz_wp), rotation_arc_length(ypr_wp)
            t_col, r_col = arc[:, 14 + k], arc[:, 16 + k]
            tri = self.velocity_mode in TRI_MODES
            if tri:
                cum_g, g_col = gripper_arc_length(grip_wp), arc[:, 18 + k]
            f_t = self._stream_index(t_col, float(cum_t[-1]), t, self.min_speed, 5.0)
            f_r = self._stream_index(r_col, float(cum_r[-1]), t, 0.01, 20.0)
            # gripper speed bounds wide enough never to bind on real tokens (it is a rescue for wild predictions only)
            f_g = (
                self._stream_index(g_col, float(cum_g[-1]), t, 1e-3, 100.0)
                if tri
                else f_t
            )
            arms.append(
                np.concatenate(
                    [
                        _lerp_rows(xyz_wp, f_t),
                        _slerp_rows(ypr_wp, f_r),
                        _lerp_rows(grip_wp, f_g),
                    ],
                    axis=-1,
                )
            )
        return np.concatenate(arms, axis=-1)

    def _tokenize_arm_evt(self, arm: np.ndarray):
        """durevt / stkevt, one arm (T, 7) -> (waypoints (M, 7), translation, rotation, gripper timing (M,)).

        xyz and ypr are read at Ryan's support frames of their own REAL arcs (budgets D and
        ``rotation_distance_unit``; no speed deadband). For his dwell reservation a step is stopped when it is
        ``zero_dist_epsilon`` (1e-6 m / rad per frame) or less: his default is 1e-9, and 2.5 % of the real translation
        steps (0.8 % of the rotation ones) are float32 dust between the two, which would fragment the dwells. The
        gripper is the event list of ``gripper_event_supports`` padded by repeating its last support. Timing rows
        0..M-2 are the intervals between supports and row M-1 repeats row M-2: elapsed seconds, or for stkevt's
        translation / rotation the distance between the ENCODED supports over the elapsed time (0 for a zero-length
        interval). The gripper column is seconds in both forms.
        """
        cfg = self.tokenizer.config
        M, dt, eps = self.M, cfg.dt, cfg.zero_dist_epsilon
        mode = "dur" if self.velocity_mode == "durevt" else "stk"
        pos_raw, ypr, grip = arm[:, 0:3], arm[:, 3:6], arm[:, 6]
        pos = self._progress_positions(pos_raw)
        cum_t, cum_r = cumulative_arc_length(pos), rotation_arc_length(ypr)
        f_t = _support_frames(
            cum_t, min(cum_t[-1], cfg.min_distance_unit), M, mode, epsilon=eps
        )
        f_r = _support_frames(
            cum_r, min(cum_r[-1], self.rotation_distance_unit), M, mode, epsilon=eps
        )
        xyz, ypr_wp = _lerp_rows(pos, f_t), _slerp_rows(ypr, f_r)
        xyz[0], ypr_wp[0] = pos_raw[0], ypr[0]
        t_col, r_col = np.diff(f_t) * dt, np.diff(f_r) * dt
        if mode == "stk":
            dist_t = np.diff(cumulative_arc_length(xyz))
            dist_r = np.diff(rotation_arc_length(ypr_wp))
            t_col = np.divide(dist_t, t_col, out=np.zeros(M - 1), where=t_col > 1e-9)
            r_col = np.divide(dist_r, r_col, out=np.zeros(M - 1), where=r_col > 1e-9)
        times, values = gripper_event_supports(grip, dt)
        pad = M - len(times)
        times = np.concatenate([times, np.repeat(times[-1], pad)])
        values = np.concatenate([values, np.repeat(values[-1], pad)])
        wp = np.concatenate([xyz, ypr_wp, values[:, None]], axis=1)
        return (wp, *(np.r_[c, c[-1]] for c in (t_col, r_col, np.diff(times) * dt)))

    def _evt_seconds(self, col: np.ndarray, dist: np.ndarray, floor: float):
        """Interval seconds from rows 0..M-2 of a durevt / durref column (seconds, clamped at 0) or a stkevt / stkref
        rate column (distance between the supports / max(|rate|, floor): the predicted sign is dead and an interval
        lasts at most distance / floor. At the default 1e-3 that is 2 to 4 s for a translation interval and 9 to 25 s
        for a rotation one: still a stall for the executed prefix; raise it with ``speed_floors``)."""
        if self.velocity_mode in ("durevt", "durref"):
            return np.maximum(col, 0.0)
        return dist / np.maximum(np.abs(col), floor)

    def _detokenize_evt(
        self, arc: np.ndarray, h: int, t: np.ndarray | None = None
    ) -> np.ndarray:
        """Whole-token decode: every stream runs its own clock, the cumulative sum of its interval seconds from
        t = 0, is interpolated between supports (right-continuous at zero-length intervals) and holds its last support
        once the clock ends. No start delay, no mean-speed rescale, no dependence on the decode horizon."""
        if t is None:
            t = self.tokenizer.config.dt * np.arange(h, dtype=np.float64)
        floors = self.speed_floors or EVT_RATE_FLOORS
        rows = np.arange(len(arc), dtype=np.float64)

        def index(seconds):  # fractional support index at times t
            f = np.interp(t, np.concatenate(([0.0], np.cumsum(seconds))), rows)
            f[t == 0] = 0.0
            return f

        arms = []
        for k, (xyz_off, ypr_off, grip_off, _) in enumerate(ARM_LAYOUT):
            xyz_wp = arc[:, xyz_off : xyz_off + 3]
            ypr_wp = arc[:, ypr_off : ypr_off + 3]
            grip_wp = arc[:, grip_off : grip_off + 1]
            t_iv = self._evt_seconds(
                arc[:-1, 14 + k], np.diff(cumulative_arc_length(xyz_wp)), floors[0]
            )
            r_iv = self._evt_seconds(
                arc[:-1, 16 + k], np.diff(rotation_arc_length(ypr_wp)), floors[1]
            )
            g_iv = np.maximum(arc[:-1, 18 + k], 0.0)  # seconds in both forms
            arms.append(
                np.concatenate(
                    [
                        _lerp_rows(xyz_wp, index(t_iv)),
                        _slerp_rows(ypr_wp, index(r_iv)),
                        _lerp_rows(grip_wp, index(g_iv)),
                    ],
                    axis=-1,
                )
            )
        return np.concatenate(arms, axis=-1)

    def _tokenize_arm_ref(self, arm: np.ndarray):
        """durref / stkref, one arm (T, 7) -> (waypoints (M, 10) [xyz, rotation6d, grip], translation timing (M,),
        rotation timing (M,)).

        ``_tokenize_arm_evt``'s translation and rotation streams with two differences: the translation stream has no
        budget (it spans the window's whole path, as Ryan's runs do; D = 0.40 m binds in 29.5 % of arm-windows and
        the gripper rides this stream), and the gripper is the aperture interpolated linearly at the translation
        supports. The rotation supports are slerped as ypr and stored as rotation6d (first two matrix columns);
        stkref's rotation rate is the geodesic angle between those supports over the elapsed time.
        """
        cfg = self.tokenizer.config
        M, dt, eps = self.M, cfg.dt, cfg.zero_dist_epsilon
        mode = "dur" if self.velocity_mode == "durref" else "stk"
        pos_raw, ypr, grip = arm[:, 0:3], arm[:, 3:6], arm[:, 6:7]
        pos = self._progress_positions(pos_raw)
        cum_t, cum_r = cumulative_arc_length(pos), rotation_arc_length(ypr)
        f_t = _support_frames(cum_t, cum_t[-1], M, mode, epsilon=eps)
        f_r = _support_frames(
            cum_r, min(cum_r[-1], self.rotation_distance_unit), M, mode, epsilon=eps
        )
        xyz, ypr_wp = _lerp_rows(pos, f_t), _slerp_rows(ypr, f_r)
        xyz[0], ypr_wp[0] = pos_raw[0], ypr[0]
        t_col, r_col = np.diff(f_t) * dt, np.diff(f_r) * dt
        if mode == "stk":
            dist_t = np.diff(cumulative_arc_length(xyz))
            dist_r = np.diff(rotation_arc_length(ypr_wp))
            t_col = np.divide(dist_t, t_col, out=np.zeros(M - 1), where=t_col > 1e-9)
            r_col = np.divide(dist_r, r_col, out=np.zeros(M - 1), where=r_col > 1e-9)
        wp = np.concatenate([xyz, _ypr_to_rot6d(ypr_wp), _lerp_rows(grip, f_t)], axis=1)
        return (wp, *(np.r_[c, c[-1]] for c in (t_col, r_col)))

    def _detokenize_ref(
        self, arc: np.ndarray, h: int, t: np.ndarray | None = None
    ) -> np.ndarray:
        """Whole-token decode to the canonical (h, 14) chunk with ZYX Euler: ``_detokenize_evt``'s two clocks per
        arm, xyz and gripper linear on the translation clock, rotation6d rows mapped to rotations (Gram-Schmidt) and
        slerped on the rotation clock. A stream holds its last support once its clock ends."""
        if t is None:
            t = self.tokenizer.config.dt * np.arange(h, dtype=np.float64)
        floors = self.speed_floors or EVT_RATE_FLOORS
        rows = np.arange(len(arc), dtype=np.float64)

        def index(seconds):  # fractional support index at times t
            f = np.interp(t, np.concatenate(([0.0], np.cumsum(seconds))), rows)
            f[t == 0] = 0.0
            return f

        arms = []
        for k, (xyz_off, rot_off, grip_off) in enumerate(REF_ARM_LAYOUT):
            xyz_wp = arc[:, xyz_off : xyz_off + 3]
            ypr_wp = _from_rotation6d(arc[:, rot_off : rot_off + 6]).as_euler("ZYX")
            grip_wp = arc[:, grip_off : grip_off + 1]
            f_t = index(
                self._evt_seconds(
                    arc[:-1, 20 + k], np.diff(cumulative_arc_length(xyz_wp)), floors[0]
                )
            )
            f_r = index(
                self._evt_seconds(
                    arc[:-1, 22 + k], np.diff(rotation_arc_length(ypr_wp)), floors[1]
                )
            )
            arms.append(
                np.concatenate(
                    [
                        _lerp_rows(xyz_wp, f_t),
                        _slerp_rows(ypr_wp, f_r),
                        _lerp_rows(grip_wp, f_t),
                    ],
                    axis=-1,
                )
            )
        return np.concatenate(arms, axis=-1)

    def _progress_positions(self, pos: np.ndarray) -> np.ndarray:
        """Positions used for arc length / resampling / timing (#4 when smoothing is on)."""
        if self.progress_smooth_hz is None:
            return np.asarray(pos, dtype=np.float64)
        return lowpass_positions(
            pos, self.progress_smooth_hz, 1.0 / self.tokenizer.config.dt
        )

    # -- one arm, parent semantics of ArcLengthTokenizer.tokenize_at(t=0) -----
    def progress_coordinates(self, pos: np.ndarray, ypr: np.ndarray) -> np.ndarray:
        """Geometric path used for a single stream's support placement and clock.

        Pose values remain at the end effector. Subclasses may measure progress
        at another rigidly attached point, including during decode.
        """
        return pos

    def _tokenize_arm(self, arm: np.ndarray):
        """arm: (T, 7) [xyz ypr grip] → (waypoints (M, 7), mean velocity (3,), num_steps, span)."""
        cfg = self.tokenizer.config
        M, D, dt = self.M, cfg.min_distance_unit, cfg.dt
        pos_raw, ypr, grip = arm[:, 0:3], arm[:, 3:6], arm[:, 6:7]
        pos = self._progress_positions(pos_raw)
        n = len(pos)
        cum = cumulative_arc_length(self.progress_coordinates(pos, ypr))
        end_s = min(D, float(cum[-1]))
        end_idx = int(np.searchsorted(cum, end_s, side="right") - 1)
        end_idx = max(0, min(end_idx, n - 1))
        num_steps = max(1, end_idx)
        alphas = np.linspace(0.0, 1.0, M)
        if end_s < cfg.zero_dist_epsilon or num_steps > cfg.max_steps_per_chunk:
            # zero token: hold the pose, slerp rotation to the end frame, ramp the gripper
            end = min(end_idx, n - 1)
            rot = R.from_euler("ZYX", np.stack([ypr[0], ypr[end]]))
            ypr_rs = _slerp_vec(rot, np.zeros(M, dtype=int), alphas).as_euler(
                "ZYX", degrees=False
            )
            ypr_rs[0] = ypr[0]
            ypr_rs[-1] = ypr[end]
            wp = np.concatenate(
                [
                    np.repeat(pos_raw[:1], M, 0),
                    ypr_rs,
                    (1 - alphas[:, None]) * grip[0] + alphas[:, None] * grip[end],
                ],
                axis=1,
            )
            return wp, np.zeros(3), num_steps, end_s
        if self.fixed_spacing:
            h = D / max(M - 1, 1)
            n_valid = int(min(M, np.floor(end_s / h + 1e-9) + 1))
            end_s = (n_valid - 1) * h
            end_idx = max(
                0, min(int(np.searchsorted(cum, end_s, side="right") - 1), n - 1)
            )
            num_steps = max(1, end_idx)
            targets = np.concatenate(
                [np.arange(n_valid) * h, np.full(M - n_valid, end_s)]
            )  # plateau past the valid length
        else:
            targets = np.linspace(0.0, end_s, M)
        pos_rs, ypr_rs, grip_rs = resample_at_s(pos, ypr, grip, cum, targets)
        pos_rs[0], ypr_rs[0], grip_rs[0] = (
            pos_raw[0],
            ypr[0],
            grip[0],
        )  # start_idx=0 anchoring (raw)
        vel = (pos_rs[-1] - pos_rs[0]) / max(num_steps * dt, 1e-8)  # MEAN_PER_DIM
        return np.concatenate([pos_rs, ypr_rs, grip_rs], axis=1), vel, num_steps, end_s

    # -- tokenize ----------------------------------------------------------
    def transform(self, batch: dict) -> dict:
        chunk = np.asarray(batch[self.action_key], dtype=np.float64)
        M = self.M
        dt = self.tokenizer.config.dt
        if chunk.ndim != 2 or chunk.shape[1] != 14:
            raise ValueError(f"expected (T, 14) chunk, got {chunk.shape}")
        if np.any(np.abs(chunk) >= INVALID_POSE_THRESHOLD):
            rows, cols = (M, self.token_dim) if self.wide else (M + 1, 14)
            batch[self.output_action_key] = np.full(
                (rows, cols), INVALID_POSE_FILL, dtype=np.float64
            )
            return batch

        if self.velocity_mode in REF_MODES:
            (wp_l, t_l, r_l), (wp_r, t_r, r_r) = (
                self._tokenize_arm_ref(chunk[:, xyz_off : xyz_off + 7])
                for xyz_off, _, _, _ in ARM_LAYOUT
            )
            batch[self.output_action_key] = np.concatenate(
                [wp_l, wp_r, np.stack([t_l, t_r, r_l, r_r], axis=1)], axis=1
            )
            return batch

        if self.velocity_mode in EVT_MODES:
            (wp_l, t_l, r_l, g_l), (wp_r, t_r, r_r, g_r) = (
                self._tokenize_arm_evt(chunk[:, xyz_off : xyz_off + 7])
                for xyz_off, _, _, _ in ARM_LAYOUT
            )
            batch[self.output_action_key] = np.concatenate(
                [wp_l, wp_r, np.stack([t_l, t_r, r_l, r_r, g_l, g_r], axis=1)], axis=1
            )
            return batch

        if self.velocity_mode in HYBRID_MODES:
            wps, t_dts, r_dts, g_dts = [], [], [], []
            for xyz_off, _, _, _ in ARM_LAYOUT:
                wp, t_dt, r_dt, g_dt = self._tokenize_arm_hybrid(
                    chunk[:, xyz_off : xyz_off + 7]
                )
                if self.velocity_mode in SPEED_MODES:
                    eps = self.tokenizer.config.zero_dist_epsilon
                    t_dt = _dur_col_to_speed(
                        t_dt, np.diff(cumulative_arc_length(wp[:, 0:3])), eps
                    )
                    r_dt = _dur_col_to_speed(
                        r_dt, np.diff(rotation_arc_length(wp[:, 3:6])), eps
                    )
                    if g_dt is not None:
                        g_dt = _dur_col_to_speed(
                            g_dt,
                            np.diff(gripper_arc_length(wp[:, 6])),
                            self.gripper_hold_epsilon,
                        )
                wps.append(wp)
                t_dts.append(t_dt)
                r_dts.append(r_dt)
                if g_dt is not None:
                    g_dts.append(g_dt)
            batch[self.output_action_key] = np.concatenate(
                [np.concatenate(wps, axis=1), np.stack(t_dts + r_dts + g_dts, axis=1)],
                axis=1,
            )
            return batch

        wps, vels, spans = [], [], []
        for xyz_off, _, _, _ in ARM_LAYOUT:
            wp, vel, _, span = self._tokenize_arm(chunk[:, xyz_off : xyz_off + 7])
            wps.append(wp)
            vels.append(vel)
            spans.append(span)
        waypoints = np.concatenate(wps, axis=1)  # (M, 14)

        if self.wide:
            prof = np.zeros((M, 2), dtype=np.float64)
            for k, (xyz_off, ypr_off, _, _) in enumerate(ARM_LAYOUT):
                pos = self._progress_positions(chunk[:, xyz_off : xyz_off + 3])
                pos = self.progress_coordinates(pos, chunk[:, ypr_off : ypr_off + 3])
                cum = cumulative_arc_length(pos)
                span = spans[k]
                if span <= 1e-8 or len(pos) < 2:
                    if self.velocity_mode == "logdur":
                        prof[0, k] = -np.log(
                            self.min_speed
                        )  # stationary arm: slowness at the floor speed
                    continue
                if self.fixed_spacing:
                    h = self.tokenizer.config.min_distance_unit / max(M - 1, 1)
                    n_valid = int(min(M, np.floor(span / h + 1e-9) + 1))
                    u = np.concatenate(
                        [
                            np.arange(n_valid) * h,
                            np.full(M - n_valid, (n_valid - 1) * h),
                        ]
                    )
                else:
                    u = np.linspace(0.0, span, M)
                fidx = np.interp(u, cum, np.arange(len(cum)))
                if self.velocity_mode == "profile":
                    v = chunk_speed(pos, dt, self.speed_smooth_frames)
                    prof[:, k] = np.maximum(np.interp(fidx, np.arange(len(v)), v), 0.0)
                elif self.velocity_mode == "dur":
                    # Ryan's codec: the elapsed seconds of each waypoint interval,
                    # absolute. `fidx` is the fractional frame index at each waypoint,
                    # so an interval's duration is its difference x dt -- the same
                    # `seg` logdur then takes the log of, which makes the two rows
                    # carry identical timing content and differ only in how it is
                    # parameterized. Padded (plateau) segments stay 0; row M-1 repeats
                    # the last interval, as his `_sample_stream` pads.
                    valid = np.diff(u) > 1e-9
                    seg = np.maximum(np.diff(fidx) * dt, LOGDUR_MIN_DT) * valid
                    if self.duration_timing == "absolute":
                        # Same supports and ground-truth clock as delta dur;
                        # each row directly supervises its time from the anchor.
                        prof[:, k] = np.concatenate(([0.0], np.cumsum(seg)))
                    else:
                        prof[: M - 1, k] = seg
                        prof[M - 1, k] = seg[-1]
                else:  # logdur: row 0 = log mean slowness, rows 1.. = log relative segment durations
                    valid = (
                        np.diff(u) > 1e-9
                    )  # padded (plateau) segments carry no time and get 0
                    seg = np.maximum(np.diff(fidx) * dt, LOGDUR_MIN_DT) * valid
                    n_seg = max(int(valid.sum()), 1)
                    t_span = float(seg.sum())
                    prof[0, k] = np.log(t_span / max(float(u[-1]), 1e-9))
                    ratio = np.zeros(M - 1)
                    ratio[valid] = np.clip(
                        np.log(seg[valid] / (t_span / n_seg)), -LOGDUR_CLIP, LOGDUR_CLIP
                    )
                    prof[1:, k] = ratio
            batch[self.output_action_key] = np.concatenate([waypoints, prof], axis=1)
            return batch

        # mean mode: the parent's (M+1, 14) layout with its velocity row
        vel_token = np.zeros(14, dtype=np.float64)
        default_dur = max(M - 1, 1) * dt
        for k, (xyz_off, ypr_off, grip_off, vsl) in enumerate(ARM_LAYOUT):
            wp = wps[k]
            vel = vels[k]
            speed = float(np.linalg.norm(vel))
            chord = float(np.linalg.norm(wp[-1, 0:3] - wp[0, 0:3]))
            dur = (chord / speed) if speed > 1e-8 else default_dur
            if self.velocity_norm == "path" and speed > 1e-8 and chord > 1e-8:
                span_wp = float(cumulative_arc_length(wp[:, 0:3])[-1])
                if span_wp > 1e-8:
                    vel = vel * ((span_wp / dur) / speed)
            vel_token[vsl] = vel
            vel_token[ypr_off : ypr_off + 3] = (wp[-1, 3:6] - wp[0, 3:6]) / max(
                dur, 1e-8
            )
            vel_token[grip_off] = float(wp[-1, 6] - wp[0, 6]) / max(dur, 1e-8)
        batch[self.output_action_key] = np.concatenate(
            [waypoints, vel_token[None]], axis=0
        )
        return batch

    # -- detokenize --------------------------------------------------------
    def detokenize(
        self,
        arc_actions: np.ndarray,
        action_horizon: int,
        times: np.ndarray | None = None,
    ) -> np.ndarray:
        """``times`` (H,), optional: the token-clock seconds to read each output row at.
        Default is the uniform control grid ``dt * arange(H)``, replay at the demonstrated
        tempo. A deploy-side caller (``egomimic.robot.arc_decoder``) passes a warped,
        non-decreasing grid to replay the same path at another tempo; both arms are read
        at the same ``times``, so bimanual timing is preserved. Training never passes it."""
        arc = np.asarray(arc_actions, dtype=np.float64)
        h = int(action_horizon)
        dt = self.tokenizer.config.dt
        if times is None:
            t = dt * np.arange(h, dtype=np.float64)
        else:
            t = np.asarray(times, dtype=np.float64)
            if (
                t.shape != (h,)
                or not np.isfinite(t).all()
                or t[0] < 0
                or (np.diff(t) < 0).any()
            ):
                raise ValueError(
                    f"times must be a finite non-decreasing ({h},) grid starting at >= 0"
                )
        profile = self.wide
        if profile:
            if arc.ndim != 2 or arc.shape[1] != self.token_dim:
                raise ValueError(
                    f"{self.velocity_mode} detokenize expects (M, {self.token_dim}), got {arc.shape}"
                )
            if self.velocity_mode in REF_MODES:
                return self._detokenize_ref(arc, h, t)
            if self.velocity_mode in EVT_MODES:
                return self._detokenize_evt(arc, h, t)
            if self.velocity_mode in HYBRID_MODES:
                return self._detokenize_hybrid(arc, h, t)
            M = arc.shape[0]
        else:
            if arc.ndim != 2 or arc.shape[1] != 14:
                raise ValueError(f"detokenize expects (M+1, 14), got {arc.shape}")
            M = arc.shape[0] - 1
        arms = []
        for k, (xyz_off, ypr_off, grip_off, vsl) in enumerate(ARM_LAYOUT):
            xyz_wp = arc[:M, xyz_off : xyz_off + 3]
            ypr_wp = arc[:M, ypr_off : ypr_off + 3]
            grip_wp = arc[:M, grip_off : grip_off + 1]
            cum = cumulative_arc_length(self.progress_coordinates(xyz_wp, ypr_wp))
            total = float(cum[-1])
            if profile:
                degenerate = total < 1e-9
                s = (
                    None
                    if degenerate
                    else np.interp(t, self._wide_clock(arc[:, 14 + k], cum), cum)
                )
            else:
                speed = float(np.linalg.norm(arc[M, vsl]))
                if self.velocity_norm == "chord":
                    # the lab's token stores a CHORD rate; walking arc length at it runs slow
                    # (parent fix df5b8498): convert to the arc rate the traversal needs.
                    chord = float(np.linalg.norm(xyz_wp[-1] - xyz_wp[0]))
                    if chord > 1e-6:
                        speed = speed * (total / chord)
                degenerate = total < 1e-9 or speed < 1e-8
                s = None if degenerate else np.minimum(speed * t, total)
            if degenerate:
                arms.append(
                    np.concatenate(
                        [
                            np.repeat(xyz_wp[:1], h, 0),
                            np.repeat(ypr_wp[:1], h, 0),
                            np.repeat(grip_wp[:1], h, 0),
                        ],
                        axis=-1,
                    )
                )
                continue
            pos_t, ypr_t, grip_t = resample_at_s(xyz_wp, ypr_wp, grip_wp, cum, s)
            arms.append(np.concatenate([pos_t, ypr_t, grip_t], axis=-1))
        return np.concatenate(arms, axis=-1)  # (H, 14)

    def _wide_clock(self, col: np.ndarray, cum: np.ndarray) -> np.ndarray:
        """Time-of-progress at the M waypoints from one arm's timing column."""
        if self.velocity_mode == "dur" and self.duration_timing == "absolute":
            return absolute_times_to_clock(col, cum, min_speed=self.min_speed)
        if self.velocity_mode in EVT_MODES:  # the translation clock: a running sum
            floors = self.speed_floors or EVT_RATE_FLOORS
            seconds = self._evt_seconds(col[:-1], np.diff(cum), floors[0])
            return np.concatenate(([0.0], np.cumsum(seconds)))
        if self.velocity_mode == "logdur":
            return durations_to_clock(col, cum, min_speed=self.min_speed)
        if self.velocity_mode in (
            "dur",
            "durhyb",
            "durtri",
        ):  # durhyb / durtri: the translation clock
            return durations_to_clock_abs(col, cum, min_speed=self.min_speed)
        if self.velocity_mode in SPEED_MODES:  # the translation clock, from speeds
            cfg = self.tokenizer.config
            dur = _speed_col_to_dur(
                col,
                np.diff(cum),
                cfg.zero_dist_epsilon,
                cfg.dt * (len(col) - 1),
                floor=None if self.speed_floors is None else self.speed_floors[0],
            )
            return durations_to_clock_abs(dur, cum, min_speed=self.min_speed)
        return integral_clock(cum, np.maximum(col, self.min_speed))

    def clock_at_waypoints(self, arc_actions: np.ndarray) -> list[np.ndarray]:
        """Per arm, the token's implied time-of-progress at its M waypoints (s)."""
        if self.velocity_mode in REF_MODES:
            raise ValueError(
                f"clock_at_waypoints reads the canonical 14-column layout, not {self.velocity_mode}'s (M, {E1_ARCREF_DIM})"
            )
        arc = np.asarray(arc_actions, dtype=np.float64)
        M = arc.shape[0] if self.wide else arc.shape[0] - 1
        out = []
        for k, (xyz_off, ypr_off, _, vsl) in enumerate(ARM_LAYOUT):
            cum = cumulative_arc_length(
                self.progress_coordinates(
                    arc[:M, xyz_off : xyz_off + 3], arc[:M, ypr_off : ypr_off + 3]
                )
            )
            if self.wide:
                out.append(self._wide_clock(arc[:, 14 + k], cum))
            else:
                speed = float(np.linalg.norm(arc[M, vsl]))
                if self.velocity_norm == "chord":
                    chord = float(
                        np.linalg.norm(
                            arc[0, xyz_off : xyz_off + 3]
                            - arc[M - 1, xyz_off : xyz_off + 3]
                        )
                    )
                    if chord > 1e-6:
                        speed = speed * (float(cum[-1]) / chord)
                out.append(cum / max(speed, self.min_speed))
        return out

    def stream_clocks(self, arc_actions: np.ndarray, action_horizon: int) -> list[dict]:
        """Per arm, the token's clocks at its M waypoints (s): ``translation``, ``rotation`` and
        ``gripper`` (None when that channel rides the translation clock) plus ``moving``,
        which of them move in this token. Decode-side only: the deploy decoder warps tempo
        on the translation clocks and ends a chunk at the first MOVING stream's end."""
        if not self.wide:
            raise ValueError("stream_clocks reads the wide (M, token_dim) layouts")
        arc = np.asarray(arc_actions, dtype=np.float64)
        if arc.ndim != 2 or arc.shape[1] != self.token_dim or arc.shape[0] < 2:
            raise ValueError(
                f"{self.velocity_mode} stream_clocks expects (M, {self.token_dim}), got {arc.shape}"
            )
        h = int(action_horizon)
        floors = self.speed_floors or EVT_RATE_FLOORS

        def running(seconds):
            return np.concatenate(([0.0], np.cumsum(seconds)))

        out = []
        if self.velocity_mode in REF_MODES:
            for k, (xyz_off, rot_off, grip_off) in enumerate(REF_ARM_LAYOUT):
                xyz = arc[:, xyz_off : xyz_off + 3]
                ypr = _from_rotation6d(arc[:, rot_off : rot_off + 6]).as_euler("ZYX")
                cum_t, cum_r = cumulative_arc_length(xyz), rotation_arc_length(ypr)
                out.append(
                    {
                        "translation": running(
                            self._evt_seconds(
                                arc[:-1, 20 + k], np.diff(cum_t), floors[0]
                            )
                        ),
                        "rotation": running(
                            self._evt_seconds(
                                arc[:-1, 22 + k], np.diff(cum_r), floors[1]
                            )
                        ),
                        "gripper": None,
                        "moving": {
                            "translation": bool(
                                cum_t[-1] >= 1e-9 or np.ptp(arc[:, grip_off]) > 1e-9
                            ),
                            "rotation": bool(cum_r[-1] > HYBRID_ROT_EPS),
                            "gripper": False,
                        },
                    }
                )
            return out
        durations = (
            self.as_duration_token(arc, h)
            if self.velocity_mode in HYBRID_MODES
            else arc
        )
        for k, (xyz_off, ypr_off, grip_off, _) in enumerate(ARM_LAYOUT):
            xyz, ypr = arc[:, xyz_off : xyz_off + 3], arc[:, ypr_off : ypr_off + 3]
            grip = arc[:, grip_off]
            if self.velocity_mode in EVT_MODES:
                cum_t, cum_r = cumulative_arc_length(xyz), rotation_arc_length(ypr)
                clocks = {
                    "translation": running(
                        self._evt_seconds(arc[:-1, 14 + k], np.diff(cum_t), floors[0])
                    ),
                    "rotation": running(
                        self._evt_seconds(arc[:-1, 16 + k], np.diff(cum_r), floors[1])
                    ),
                    "gripper": running(np.maximum(arc[:-1, 18 + k], 0.0)),
                }
                moving = {
                    "translation": bool(cum_t[-1] >= 1e-9),
                    "rotation": bool(cum_r[-1] > HYBRID_ROT_EPS),
                    "gripper": bool(np.ptp(grip) >= TRI_GRIPPER_MOVE_RANGE),
                }
            elif self.velocity_mode in HYBRID_MODES:
                tri = self.velocity_mode in TRI_MODES
                cum_t, cum_r = cumulative_arc_length(xyz), rotation_arc_length(ypr)
                clocks = {
                    "translation": self._stream_clock(
                        durations[:, 14 + k], float(cum_t[-1]), self.min_speed, 5.0
                    ),
                    "rotation": self._stream_clock(
                        durations[:, 16 + k], float(cum_r[-1]), 0.01, 20.0
                    ),
                    "gripper": (
                        self._stream_clock(
                            durations[:, 18 + k],
                            float(gripper_arc_length(grip[:, None])[-1]),
                            1e-3,
                            100.0,
                        )
                        if tri
                        else None
                    ),
                }
                moving = {
                    "translation": bool(
                        cum_t[-1] >= 1e-9 or (not tri and np.ptp(grip) > 1e-9)
                    ),
                    "rotation": bool(cum_r[-1] > HYBRID_ROT_EPS),
                    "gripper": bool(tri and np.ptp(grip) >= TRI_GRIPPER_MOVE_RANGE),
                }
            else:  # dur / logdur / profile (and the projection codec): one clock per arm
                cum = cumulative_arc_length(self.progress_coordinates(xyz, ypr))
                clocks = {
                    "translation": self._wide_clock(arc[:, 14 + k], cum),
                    "rotation": None,
                    "gripper": None,
                }
                moving = {
                    "translation": bool(cum[-1] >= 1e-9),
                    "rotation": False,
                    "gripper": False,
                }
            out.append({**clocks, "moving": moving})
        return out

    def stream_ends(self, arc_actions: np.ndarray, action_horizon: int) -> list[float]:
        """Clock time (s) at which each MOVING stream reaches its last waypoint, left arm
        first; a still stream contributes nothing. Past the latest every row repeats."""
        ends = []
        for arm in self.stream_clocks(arc_actions, action_horizon):
            for name in ("translation", "rotation", "gripper"):
                if arm["moving"][name] and arm[name] is not None:
                    ends.append(float(arm[name][-1]))
        return ends

    hybrid_stream_ends = stream_ends
