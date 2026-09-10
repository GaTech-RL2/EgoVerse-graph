"""Arc-matched, DTW and time-domain metrics for bimanual cartesian chunks.

Pure functions over ``(T, 14)`` numpy chunks -- no evaluator, no trainer, no
model. Ported from the main repo's arc branch, where they lived inside an
HPT-coupled eval class; keeping them free of that lets this fork's
:class:`~egomimic.eval.arc_bimanual_cartesian_eval.ArcBimanualCartesianEval`
call them, and lets them be tested without a graph.

Four families, all scored against the SAME ground truth -- the time-indexed
chunk the loader preserved -- so an arc run and a time-indexed baseline land on
comparable charts:

``arcmatch``
    Re-tokenize prediction and ground truth onto a matched per-arm span (the
    shorter of the two travelled distances) and score the waypoints. Travel is
    divided out, so this measures path SHAPE alone. Reported with and without
    the velocity row; the gap between the two is timing error.

``dtw``
    Warp the prediction against the ground-truth chunk. Elastic in time, so
    unlike arcmatch it does see a travel mismatch -- a prediction that stops
    short or overruns pays for it here.

``chunk``
    Plain MSE against the raw ground-truth chunk, split into xyz, ypr, gripper
    and final-step terms. The "detok"/"baseline" family upstream.

``pose_err_m``
    Position and rotation as ONE number in metres, via a lever arm, so radians
    are converted to the length they cost instead of being summed with metres
    under an arbitrary unit choice.

Read ``arcmatch_span_m`` and ``arcmatch_travel_ratio`` next to the arcmatch
scores: because travel is divided out, a policy that stalls scores well on
shape while those two collapse.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from egomimic.rldb.zarr.arc_length_tokenizer import (
    _dist_interval_indices,
    cumulative_arc_length,
    resample_by_distance,
)

# Canonical bimanual cartesian layout, per arm: [xyz(3), ypr(3), grip(1)].
ARM_BLOCKS = ((0, 3, 6), (7, 10, 13))  # (xyz offset, ypr offset, gripper index)
ARM_XYZ = ((0, 1, 2), (7, 8, 9))
ARM_YPR = ((3, 4, 5), (10, 11, 12))
XYZ_COLS = [c for arm in ARM_XYZ for c in arm]
# Position + gripper. Rotation is excluded from every paired MSE so radians are
# never summed with metres; it gets the geodesic chart instead.
PAIRED_COLS = [0, 1, 2, 6, 7, 8, 9, 13]
YPR_COLS = [c for arm in ARM_YPR for c in arm]
GRIP_COLS = [6, 13]
BIMANUAL_DIM = 14


def _validate(traj: np.ndarray, label: str) -> np.ndarray:
    traj = np.asarray(traj, dtype=np.float64)
    if traj.ndim != 2 or traj.shape[1] != BIMANUAL_DIM:
        raise ValueError(f"{label} must be (T, {BIMANUAL_DIM}), got {traj.shape}")
    if len(traj) < 2:
        raise ValueError(f"{label} needs at least two timesteps, got {len(traj)}")
    return traj


def _arm_views(traj: np.ndarray):
    """(T, 14) -> per-arm (pos (T,3), ypr (T,3), grip (T,1))."""
    for xyz_off, ypr_off, grip_i in ARM_BLOCKS:
        yield (
            traj[:, xyz_off : xyz_off + 3],
            traj[:, ypr_off : ypr_off + 3],
            traj[:, grip_i : grip_i + 1],
        )


def mse(a: np.ndarray, b: np.ndarray, cols=None) -> float:
    if cols is not None:
        a, b = a[..., cols], b[..., cols]
    return float(np.mean((a - b) ** 2))


# -- arc-span tokenization --------------------------------------------------


def arm_travel(traj: np.ndarray) -> np.ndarray:
    """(T, 14) -> (2,) total translational arc length per arm, in metres."""
    traj = _validate(traj, "arm_travel input")
    return np.array(
        [float(cumulative_arc_length(pos)[-1]) for pos, _, _ in _arm_views(traj)]
    )


def match_spans(pred_ti: np.ndarray, gt_ti: np.ndarray) -> np.ndarray:
    """Per-arm matched span: the shorter of the two travelled distances.

    Whichever side travels less sets the window, so both are re-tokenized over
    a stretch of motion they both actually cover. Without the cut, a prediction
    that runs further is scored against ground truth it never reached.
    """
    return np.minimum(arm_travel(pred_ti), arm_travel(gt_ti))


def tokenize_span(
    traj: np.ndarray, spans: np.ndarray, num_points: int, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize a time-indexed chunk over a caller-supplied PER-ARM span.

    The real tokenizer with its distance freed rather than pinned to D. Each
    arm is resampled to ``num_points`` samples uniform in ITS OWN arc length
    over ``[0, span_arm]``, using the same interpolators the data pipeline uses
    (linear on xyz and gripper, slerp on rotation), so these are the waypoints
    it would have produced had D been ``span_arm``.

    The velocity row is recomputed here rather than read off a model output, so
    a time-indexed run -- which predicts no velocity token at all -- still has
    one, and both run types get it by the same formula.

    returns: (waypoints (num_points, 14), velocity (14,)).
    """
    traj = _validate(traj, "tokenize_span input")
    if num_points < 2:
        raise ValueError("num_points must be at least two")
    if dt <= 0:
        raise ValueError("dt must be positive")
    waypoints = np.zeros((num_points, BIMANUAL_DIM), dtype=np.float64)
    velocity = np.zeros(BIMANUAL_DIM, dtype=np.float64)
    for arm, (pos, ypr, grip) in enumerate(_arm_views(traj)):
        cum = cumulative_arc_length(pos)
        end_s = float(min(spans[arm], cum[-1]))
        p, y, g = resample_by_distance(
            pos, ypr, grip, cum, 0.0, end_s, num_points, start_idx=0
        )
        offset = arm * 7
        waypoints[:, offset : offset + 3] = p
        waypoints[:, offset + 3 : offset + 6] = y
        waypoints[:, offset + 6] = g[:, 0]
        # duration = source steps the span covers, times dt. Falling back to one
        # step keeps a stationary arm's velocity finite and zero rather than
        # NaN, which is how the tokenizer handles the same case.
        start_i, end_i = _dist_interval_indices(cum, 0.0, end_s)
        duration = max(end_i - start_i, 1) * dt
        velocity[offset : offset + 3] = (p[-1] - p[0]) / duration
        velocity[offset + 3 : offset + 6] = (y[-1] - y[0]) / duration
        velocity[offset + 6] = (g[-1, 0] - g[0, 0]) / duration
    return waypoints, velocity


def clip_to_distance(traj: np.ndarray, distance: float) -> np.ndarray:
    """Truncate a chunk at the first row where EITHER arm passes ``distance``.

    Both arms keep the same row count because warping runs on whole rows.
    """
    traj = _validate(traj, "clip_to_distance input")
    ends = []
    for pos, _, _ in _arm_views(traj):
        cum = cumulative_arc_length(pos)
        over = np.nonzero(cum > distance)[0]
        ends.append(int(over[0]) + 1 if len(over) else len(traj))
    return traj[: max(2, min(ends))]


# -- dynamic time warping ---------------------------------------------------


def dtw_path(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Warping path between two (N, 3) position sequences.

    Standard DTW recurrence, but the forward pass sweeps ANTI-DIAGONALS rather
    than rows: every cell on ``i + j = k`` depends only on diagonals ``k-1`` and
    ``k-2``, so each diagonal is one vectorized numpy step. That turns an
    O(N*M) python loop into O(N+M) numpy calls, which is the difference between
    DTW being affordable in a per-epoch val loop and not.

    returns: (ia, ib) index arrays of equal length, the aligned pairs.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(
            f"dtw_path needs two (N, D) sequences of equal width, got "
            f"{a.shape} and {b.shape}"
        )
    na, nb = len(a), len(b)
    distance = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    accumulated = np.full((na + 1, nb + 1), np.inf)
    accumulated[0, 0] = 0.0
    for k in range(2, na + nb + 1):
        i = np.arange(max(1, k - nb), min(na, k - 1) + 1)
        if not len(i):
            continue
        j = k - i
        accumulated[i, j] = distance[i - 1, j - 1] + np.minimum(
            np.minimum(accumulated[i - 1, j], accumulated[i, j - 1]),
            accumulated[i - 1, j - 1],
        )
    ia, ib = [], []
    i, j = na, nb
    while i > 0 and j > 0:
        ia.append(i - 1)
        ib.append(j - 1)
        step = int(
            np.argmin(
                (
                    accumulated[i - 1, j],
                    accumulated[i, j - 1],
                    accumulated[i - 1, j - 1],
                )
            )
        )
        if step == 0:
            i -= 1
        elif step == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    return np.array(ia[::-1]), np.array(ib[::-1])


# -- rotation ---------------------------------------------------------------


def geodesic_deg(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean geodesic angle in degrees between two ``(..., 14)`` chunks.

    The angle of the single rotation carrying one orientation to the other --
    the shortest path on SO(3). Bounded in [0, 180], never wraps, unaffected by
    gimbal lock, and unchanged by expressing both rotations in another frame,
    none of which holds for a per-axis ypr difference.
    """
    per_arm = []
    for cols in ARM_YPR:
        rp = R.from_euler("ZYX", pred[..., list(cols)].reshape(-1, 3))
        rg = R.from_euler("ZYX", gt[..., list(cols)].reshape(-1, 3))
        per_arm.append(float(np.degrees((rp * rg.inv()).magnitude()).mean()))
    return float(np.mean(per_arm))


def pose_err_m(pred: np.ndarray, gt: np.ndarray, lever_m: float) -> float:
    """Position AND rotation error as one number, in metres.

    Summing squared metres with squared radians would make the score depend on
    an arbitrary unit choice -- rescaling the rotation representation would
    "improve" it. So rotation is converted into the length it costs: a geodesic
    error of theta radians displaces a point rigidly attached ``lever_m`` from
    the EEF origin by ``lever_m * theta``. The result is the RMS displacement of
    that point,

        sqrt( mean( |dp|^2 + (lever_m * theta)^2 ) )

    which reads as "a point this far out on the gripper is off by X metres".
    Both terms are frame-invariant -- translation because the camera revert is
    rigid, rotation because conjugation preserves the geodesic angle -- so the
    combined number is too.

    ``lever_m`` is the one judgement call, and it is explicit rather than
    smuggled in through units: it says what a radian is worth in metres.
    """
    if lever_m < 0:
        raise ValueError("lever_m must be non-negative")
    per_arm = []
    for xyz_cols, ypr_cols in zip(ARM_XYZ, ARM_YPR):
        dp2 = ((pred[..., list(xyz_cols)] - gt[..., list(xyz_cols)]) ** 2).sum(-1)
        rp = R.from_euler("ZYX", pred[..., list(ypr_cols)].reshape(-1, 3))
        rg = R.from_euler("ZYX", gt[..., list(ypr_cols)].reshape(-1, 3))
        theta = (rp * rg.inv()).magnitude().reshape(dp2.shape)
        per_arm.append(dp2 + (float(lever_m) * theta) ** 2)
    return float(np.sqrt(np.mean(per_arm)))


# -- families ---------------------------------------------------------------


def arcmatch_metrics(
    predictions: list[np.ndarray],
    ground_truth: list[np.ndarray],
    *,
    num_points: int,
    dt: float,
    lever_m: float,
) -> dict[str, float]:
    """Re-tokenize both sides onto the matched per-arm span, then score."""
    pred_w, gt_w, pred_v, gt_v, spans, ratios = [], [], [], [], [], []
    for pred, gt in zip(predictions, ground_truth):
        span = match_spans(pred, gt)
        if not np.all(np.isfinite(span)):
            continue
        pw, pv = tokenize_span(pred, span, num_points, dt)
        gw, gv = tokenize_span(gt, span, num_points, dt)
        pred_w.append(pw)
        gt_w.append(gw)
        pred_v.append(pv)
        gt_v.append(gv)
        spans.append(span)
        ratios.append(arm_travel(pred) / np.maximum(arm_travel(gt), 1e-6))
    if not pred_w:
        return {}
    pred_w, gt_w = np.stack(pred_w), np.stack(gt_w)
    # With-velocity variant: the velocity row appended as one more sample,
    # which is what an arc head actually predicts alongside the waypoints.
    pred_full = np.concatenate([pred_w, np.stack(pred_v)[:, None, :]], axis=1)
    gt_full = np.concatenate([gt_w, np.stack(gt_v)[:, None, :]], axis=1)
    return {
        "arcmatch_paired_mse": mse(pred_w, gt_w, PAIRED_COLS),
        "arcmatch_withvel_paired_mse": mse(pred_full, gt_full, PAIRED_COLS),
        "arcmatch_xyz_mse": mse(pred_w, gt_w, XYZ_COLS),
        "arcmatch_rot_geodesic_deg": geodesic_deg(pred_w, gt_w),
        "arcmatch_pose_err_m": pose_err_m(pred_w, gt_w, lever_m),
        "arcmatch_final_xyz_l2_m": float(
            np.mean(
                [
                    np.linalg.norm(
                        pred_w[:, -1, list(cols)] - gt_w[:, -1, list(cols)], axis=-1
                    )
                    for cols in ARM_XYZ
                ]
            )
        ),
        # Arc-matching divides travel out, so read these two beside the scores:
        # a policy that stalls looks good on shape while these collapse.
        "arcmatch_span_m": float(np.mean(spans)),
        "arcmatch_travel_ratio": float(np.mean(ratios)),
    }


def dtw_metrics(
    predictions: list[np.ndarray],
    ground_truth: list[np.ndarray],
    *,
    max_samples: int = 8,
    clip_distance: float | None = None,
) -> dict[str, float]:
    """Warp the prediction against the ground-truth chunk.

    Warping absorbs a speed difference but not a length one, so a prediction
    that stops short or runs long pays for it here -- this is the family that
    sees the extent arcmatch divides out. Capped at ``max_samples`` because DTW
    is quadratic in chunk length.
    """
    paired, l2 = [], []
    for pred, gt in zip(predictions[:max_samples], ground_truth[:max_samples]):
        target = gt if clip_distance is None else clip_to_distance(gt, clip_distance)
        for arm, cols in enumerate(ARM_XYZ):
            cols = list(cols)
            ia, ib = dtw_path(pred[:, cols], target[:, cols])
            l2.append(
                float(
                    np.mean(
                        np.linalg.norm(pred[ia][:, cols] - target[ib][:, cols], axis=-1)
                    )
                )
            )
            block = list(range(arm * 7, arm * 7 + 7))
            paired.append(mse(pred[ia][:, block], target[ib][:, block]))
    if not l2:
        return {}
    return {
        "dtw_xyz_l2_m": float(np.mean(l2)),
        "dtw_paired_mse": float(np.mean(paired)),
    }


def chunk_metrics(
    predictions: np.ndarray, ground_truth: np.ndarray, *, lever_m: float
) -> dict[str, float]:
    """Plain MSE against the raw ground-truth chunk, split by component.

    Requires both sides to already share a row count; the caller equalizes.
    """
    pred = np.asarray(predictions, dtype=np.float64)
    gt = np.asarray(ground_truth, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(
            f"chunk_metrics needs matching shapes, got {pred.shape} and {gt.shape}"
        )
    return {
        "chunk_paired_mse": mse(pred, gt, PAIRED_COLS),
        "chunk_xyz_mse": mse(pred, gt, XYZ_COLS),
        "chunk_ypr_mse": mse(pred, gt, YPR_COLS),
        "chunk_grip_mse": mse(pred, gt, GRIP_COLS),
        "chunk_rot_geodesic_deg": geodesic_deg(pred, gt),
        "chunk_pose_err_m": pose_err_m(pred, gt, lever_m),
        "chunk_final_xyz_l2_m": float(
            np.mean(
                [
                    np.linalg.norm(
                        pred[:, -1, list(cols)] - gt[:, -1, list(cols)], axis=-1
                    )
                    for cols in ARM_XYZ
                ]
            )
        ),
    }
