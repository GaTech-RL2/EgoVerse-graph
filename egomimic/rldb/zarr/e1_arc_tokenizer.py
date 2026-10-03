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
  differ only in parameterization, as ``arcvel`` and ``arcdur`` do. A translation hold has no length
  to time, so its row ``M-1`` holds the hold's duration (the gripper ramps over it).

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

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation as R

from egomimic.rldb.zarr.action_chunk_transforms import Transform
from egomimic.rldb.zarr.arc_length_tokenizer import (
    TokenizeBimanualArcLengthCartesian,
    cumulative_arc_length,
)

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
E1_ARCDURHYB_DIM = 18  # also profhyb: [14 canonical | translation v L, R | rotation omega L, R]
HYBRID_MODES = ("durhyb", "profhyb")
# Rotation budget of the hybrid token: a full turn, i.e. the rotation stream spans the whole source window.
# The lab hybrid's 24 deg (R24deg) fits a chunk that ENDS when a budget is spent; the E1 YAM window is a fixed
# 100 frames, and on rl2 stationery the wrist turns more than 24 deg in 63 % of arm-windows (median 52 deg,
# p90 123 deg), so a 24 deg stream froze the wrist for the rest of the window: GT round-trip geodesic error
# 11.7 deg vs 1.1 deg for plain arcdur, and 0.07 deg with this budget (1,500 val windows, 2026-09-30).
DEFAULT_ROTATION_DISTANCE_UNIT = 2 * np.pi


def _dur_col_to_speed(col: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """durhyb timing column -> profhyb: rows 0..M-2 become segment length / duration; row M-1 (start
    delay) is kept. A hold (no length) keeps its rows' total time in row M-1 instead."""
    out = np.zeros_like(col)
    if float(seg.sum()) < 1e-9:
        out[-1] = float(np.maximum(col[:-1], 0.0).sum())
        return out
    out[:-1] = seg / np.maximum(col[:-1], LOGDUR_MIN_DT)
    out[-1] = col[-1]
    return out


def _speed_col_to_dur(col: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """profhyb timing column -> the durhyb column it encodes (inverse of ``_dur_col_to_speed``)."""
    col = np.asarray(col, dtype=np.float64)
    M = len(col)
    out = np.zeros(M)
    if float(seg.sum()) < 1e-9:  # hold: spread the stored time evenly over the rows
        out[:-1] = max(float(col[-1]), 0.0) / max(M - 1, 1)
        return out
    out[:-1] = seg / np.maximum(col[:-1], 1e-6)
    out[-1] = col[-1]
    return out


def rotation_arc_length(ypr: np.ndarray) -> np.ndarray:
    """Cumulative geodesic rotation angle (rad) along a (T, 3) ZYX-euler track."""
    ypr = np.asarray(ypr, dtype=np.float64)
    if len(ypr) < 2:
        return np.zeros(len(ypr))
    rot = R.from_euler("ZYX", ypr)
    return np.concatenate(([0.0], np.cumsum((rot[:-1].inv() * rot[1:]).magnitude())))


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
    M = len(col)
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
        min_speed: float = 0.01,
        progress_smooth_hz: float | None = None,
        fixed_spacing: bool = False,
        rotation_distance_unit: float = DEFAULT_ROTATION_DISTANCE_UNIT,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fixed_spacing = bool(fixed_spacing)
        self.rotation_distance_unit = float(rotation_distance_unit)
        if velocity_norm not in ("chord", "path"):
            raise ValueError(
                f"velocity_norm must be 'chord' or 'path', got {velocity_norm!r}"
            )
        if velocity_mode not in ("mean", "profile", "logdur", "dur", *HYBRID_MODES):
            raise ValueError(
                f"velocity_mode must be 'mean', 'profile', 'logdur', 'dur', 'durhyb' or 'profhyb', got {velocity_mode!r}"
            )
        if velocity_mode in HYBRID_MODES and self.fixed_spacing:
            raise ValueError(f"{velocity_mode} does not implement fixed_spacing")
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
        return self.velocity_mode in ("profile", "logdur", "dur", *HYBRID_MODES)

    @property
    def token_dim(self) -> int:
        return E1_ARCDURHYB_DIM if self.velocity_mode in HYBRID_MODES else E1_ARCVEL_DIM

    def _tokenize_arm_hybrid(self, arm: np.ndarray):
        """durhyb, one arm (T, 7) -> (waypoints (M, 7), translation dt (M,), rotation dt (M,)).

        xyz and gripper are sampled along the translation arc (budget D), ypr along the arm's own
        rotation arc (budget ``rotation_distance_unit``), each at M evenly spaced targets. Each
        stream stores the elapsed seconds of its intervals in rows 0..M-2 and, in row M-1 (padding
        in ``dur``), its START DELAY: the time before the stream's first motion, so a wrist turn
        that begins after the arm stops translating starts on time at decode. A translation hold
        keeps the arm put and times the gripper uniformly over the window (the plain token drops
        that gripper motion); a rotation hold keeps ypr[0].
        """
        M, dt = self.M, self.tokenizer.config.dt
        eps = self.tokenizer.config.zero_dist_epsilon
        pos_raw, ypr, grip = arm[:, 0:3], arm[:, 3:6], arm[:, 6:7]
        n = len(pos_raw)
        frames = np.arange(n, dtype=np.float64)

        def targets(cum, budget):  # fractional source frame at each waypoint, or None for a hold
            span = min(float(budget), float(cum[-1])) if n > 1 else 0.0
            return None if span < eps else np.interp(np.linspace(0.0, span, M), cum, frames)

        def durations(f):  # rows 0..M-2 interval seconds, row M-1 start delay
            seg = np.maximum(np.diff(f) * dt, LOGDUR_MIN_DT)
            return np.concatenate([seg, [max(float(f[0]), 0.0) * dt]])

        pos = self._progress_positions(pos_raw)
        f_t = targets(cumulative_arc_length(pos), self.tokenizer.config.min_distance_unit)
        if f_t is None:
            f_t = np.linspace(0.0, max(n - 1, 0), M)
            xyz = np.repeat(pos_raw[:1], M, 0)
        else:
            xyz = _lerp_rows(pos, f_t)
            xyz[0] = pos_raw[0]
        grip_wp = _lerp_rows(grip, f_t)
        f_r = targets(rotation_arc_length(ypr), self.rotation_distance_unit)
        if f_r is None:
            ypr_wp, r_dt = np.repeat(ypr[:1], M, 0), np.zeros(M)
        else:
            ypr_wp, r_dt = _slerp_rows(ypr, f_r), durations(f_r)
            ypr_wp[0] = ypr[0]
        t_dt = durations(f_t) if n > 1 else np.zeros(M)
        return np.concatenate([xyz, ypr_wp, grip_wp], axis=1), t_dt, r_dt

    @staticmethod
    def _stream_index(col: np.ndarray, span: float, t: np.ndarray, lo: float, hi: float):
        """Fractional waypoint index at times t from one stream's duration column
        (rows 0..M-2 interval seconds, row M-1 start delay)."""
        M = len(col)
        if span > 1e-9:
            clock = durations_to_clock_abs(col, span, min_speed=lo, max_speed=hi)
        else:  # a hold: nothing to bound the speed by; time the rows as stored
            clock = np.concatenate(([0.0], np.cumsum(np.maximum(col[: M - 1], 0.0))))
        if clock[-1] <= 1e-9:
            return np.zeros(len(t))
        return np.interp(t, clock + max(float(col[M - 1]), 0.0), np.arange(M, dtype=np.float64))

    def _detokenize_hybrid(self, arc: np.ndarray, h: int) -> np.ndarray:
        t = self.tokenizer.config.dt * np.arange(h, dtype=np.float64)
        arms = []
        for k, (xyz_off, ypr_off, grip_off, _) in enumerate(ARM_LAYOUT):
            xyz_wp = arc[:, xyz_off : xyz_off + 3]
            ypr_wp = arc[:, ypr_off : ypr_off + 3]
            grip_wp = arc[:, grip_off : grip_off + 1]
            cum_t, cum_r = cumulative_arc_length(xyz_wp), rotation_arc_length(ypr_wp)
            t_col, r_col = arc[:, 14 + k], arc[:, 16 + k]
            if self.velocity_mode == "profhyb":
                t_col, r_col = _speed_col_to_dur(t_col, np.diff(cum_t)), _speed_col_to_dur(r_col, np.diff(cum_r))
            f_t = self._stream_index(t_col, float(cum_t[-1]), t, self.min_speed, 5.0)
            f_r = self._stream_index(r_col, float(cum_r[-1]), t, 0.01, 20.0)
            arms.append(
                np.concatenate(
                    [_lerp_rows(xyz_wp, f_t), _slerp_rows(ypr_wp, f_r), _lerp_rows(grip_wp, f_t)],
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
    def _tokenize_arm(self, arm: np.ndarray):
        """arm: (T, 7) [xyz ypr grip] → (waypoints (M, 7), mean velocity (3,), num_steps, span)."""
        cfg = self.tokenizer.config
        M, D, dt = self.M, cfg.min_distance_unit, cfg.dt
        pos_raw, ypr, grip = arm[:, 0:3], arm[:, 3:6], arm[:, 6:7]
        pos = self._progress_positions(pos_raw)
        n = len(pos)
        cum = cumulative_arc_length(pos)
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

        if self.velocity_mode in HYBRID_MODES:
            wps, t_dts, r_dts = [], [], []
            for xyz_off, _, _, _ in ARM_LAYOUT:
                wp, t_dt, r_dt = self._tokenize_arm_hybrid(chunk[:, xyz_off : xyz_off + 7])
                if self.velocity_mode == "profhyb":
                    t_dt = _dur_col_to_speed(t_dt, np.diff(cumulative_arc_length(wp[:, 0:3])))
                    r_dt = _dur_col_to_speed(r_dt, np.diff(rotation_arc_length(wp[:, 3:6])))
                wps.append(wp)
                t_dts.append(t_dt)
                r_dts.append(r_dt)
            batch[self.output_action_key] = np.concatenate(
                [np.concatenate(wps, axis=1), np.stack(t_dts + r_dts, axis=1)], axis=1
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
            for k, (xyz_off, _, _, _) in enumerate(ARM_LAYOUT):
                pos = self._progress_positions(chunk[:, xyz_off : xyz_off + 3])
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
    def detokenize(self, arc_actions: np.ndarray, action_horizon: int) -> np.ndarray:
        arc = np.asarray(arc_actions, dtype=np.float64)
        h = int(action_horizon)
        dt = self.tokenizer.config.dt
        t = dt * np.arange(h, dtype=np.float64)
        profile = self.wide
        if profile:
            if arc.ndim != 2 or arc.shape[1] != self.token_dim:
                raise ValueError(
                    f"{self.velocity_mode} detokenize expects (M, {self.token_dim}), got {arc.shape}"
                )
            if self.velocity_mode in HYBRID_MODES:
                return self._detokenize_hybrid(arc, h)
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
            cum = cumulative_arc_length(xyz_wp)
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
        if self.velocity_mode == "logdur":
            return durations_to_clock(col, cum, min_speed=self.min_speed)
        if self.velocity_mode in ("dur", "durhyb"):  # durhyb: the translation clock
            return durations_to_clock_abs(col, cum, min_speed=self.min_speed)
        if self.velocity_mode == "profhyb":  # the translation clock, from speeds
            return durations_to_clock_abs(_speed_col_to_dur(col, np.diff(cum)), cum, min_speed=self.min_speed)
        return integral_clock(cum, np.maximum(col, self.min_speed))

    def clock_at_waypoints(self, arc_actions: np.ndarray) -> list[np.ndarray]:
        """Per arm, the token's implied time-of-progress at its M waypoints (s)."""
        arc = np.asarray(arc_actions, dtype=np.float64)
        M = arc.shape[0] if self.wide else arc.shape[0] - 1
        out = []
        for k, (xyz_off, _, _, vsl) in enumerate(ARM_LAYOUT):
            cum = cumulative_arc_length(arc[:M, xyz_off : xyz_off + 3])
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
