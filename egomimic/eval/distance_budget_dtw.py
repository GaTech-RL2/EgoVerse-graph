"""Distance-budgeted episode rollouts and exact, GT-frame-balanced global DTW.

The alignment is symmetric in its allowed moves, but its *reporting weight* is
one vote per recorded GT timestamp. Timing remains a separate diagnostic.
"""

from __future__ import annotations

import math

import numpy as np

METRIC_VERSION = "joint_distance_global_dtw_v1"
METRIC_FRAME_KEY = "evaluation.eef_to_world"
XYZ_COLS = (0, 1, 2, 7, 8, 9)


def world_xyz(actions: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Restore wrist-relative XYZ to a common world frame, without normalization."""
    values = np.asarray(actions, dtype=np.float64)
    anchors = np.asarray(anchors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 14 or anchors.shape != (2, 4, 4):
        raise ValueError("DTW requires (T,14) actions and (2,4,4) EEF-to-world anchors")
    xyz = np.concatenate(
        [
            values[:, offset : offset + 3] @ anchors[arm, :3, :3].T
            + anchors[arm, :3, 3]
            for arm, offset in enumerate((0, 7))
        ],
        axis=1,
    )
    if not np.isfinite(xyz).all():
        raise ValueError("Nonfinite world-frame DTW trajectory")
    return xyz


def joint_cumulative_distance(xyz: np.ndarray) -> np.ndarray:
    values = np.asarray(xyz, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6 or not len(values):
        raise ValueError("Expected nonempty (T,6) bimanual XYZ")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite GT positions")
    steps = np.linalg.norm(np.diff(values, axis=0).reshape(-1, 2, 3), axis=2).sum(
        axis=1
    )
    return np.concatenate(([0.0], np.cumsum(steps)))


def distance_windows(
    cumulative: np.ndarray, budget: float
) -> tuple[np.ndarray, np.ndarray]:
    """K=ceil(L/budget), sampled at first GT frame reaching each milestone.

    Repeated anchors are intentional if a recorded frame crosses multiple
    milestones. A stationary episode receives one full-prefix prediction.
    """
    cumulative = np.asarray(cumulative, dtype=np.float64)
    if (
        not np.isfinite(budget)
        or budget <= 0
        or cumulative.ndim != 1
        or not len(cumulative)
        or not np.isfinite(cumulative).all()
        or cumulative[0] != 0
        or np.any(np.diff(cumulative) < 0)
    ):
        raise ValueError("Invalid cumulative GT distance or execution budget")
    length = float(cumulative[-1])
    if length <= 1e-12:
        return np.array([0]), np.array([budget])
    ratio = length / budget
    nearest = round(ratio)
    count = int(
        nearest
        if math.isclose(ratio, nearest, abs_tol=1e-10, rel_tol=1e-10)
        else math.ceil(ratio)
    )
    starts = np.arange(max(1, count), dtype=np.float64) * budget
    anchors = np.searchsorted(cumulative, starts, side="left")
    return anchors, np.minimum(budget, np.maximum(0.0, length - starts))


def global_dtw(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    *,
    max_cells: int = 50_000_000,
    return_path: bool = False,
) -> dict:
    """Exact endpoint-anchored DTW with one shared six-XYZ alignment.

    Minimize summed squared XYZ error along the monotone path. Average matches
    *within each GT timestamp*, then average GT timestamps. Horizontal AND
    vertical moves cover shorter AND longer predictions without dropping any
    samples. No resampling, spatial scaling, band, or approximate alignment.
    DP uses O(N*M) bytes for backpointers and O(M) floating-point storage.
    """
    pred = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(ground_truth, dtype=np.float64)
    if (
        pred.ndim != 2
        or gt.ndim != 2
        or pred.shape[1:] != (6,)
        or gt.shape[1:] != (6,)
        or not len(pred)
        or not len(gt)
    ):
        raise ValueError("DTW requires nonempty (T,6) XYZ sequences")
    if not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise ValueError("DTW does not accept NaN/Inf samples")
    n, m = len(pred), len(gt)
    if n * m > int(max_cells):
        raise ValueError(
            f"Exact DTW needs {n * m} cells; limit={max_cells}. "
            "Increase dtw_max_cells explicitly; samples were not discarded."
        )
    back = np.empty((n, m), dtype=np.uint8)
    previous = np.full(m, np.inf)
    for i in range(n):
        costs = np.square(gt - pred[i]).mean(axis=1)
        diagonal = np.concatenate(([0.0 if i == 0 else np.inf], previous[:-1]))
        # Vectorized form of cur[j] = cost[j] + min(diag, up, cur[j-1]).
        cumulative = np.cumsum(costs)
        before = np.concatenate(([0.0], cumulative[:-1]))
        current = cumulative + np.minimum.accumulate(
            np.minimum(previous, diagonal) - before
        )
        left = np.concatenate(([np.inf], current[:-1]))
        back[i] = np.argmin(np.stack((diagonal, previous, left)), axis=0)
        previous = current
    i, j = n - 1, m - 1
    path = []
    while True:
        path.append((i, j))
        if i == 0 and j == 0:
            break
        move = back[i, j]
        i -= int(move != 2)
        j -= int(move != 1)
        if i < 0 or j < 0:
            raise RuntimeError("Invalid DTW backtrace")
    path = np.asarray(path[::-1], dtype=np.int64)
    costs = np.square(pred[path[:, 0]] - gt[path[:, 1]]).mean(axis=1)
    counts = np.bincount(path[:, 1], minlength=m)
    per_gt = np.bincount(path[:, 1], weights=costs, minlength=m) / counts
    result = {
        "xyz_mse": float(per_gt.mean()),
        "gt_frames": m,
        "predicted_samples": n,
        "gt_coverage": float(np.count_nonzero(counts) / m),
        "prediction_coverage": float(len(np.unique(path[:, 0])) / n),
        "path_pairs": len(path),
        "path_cost_sum": float(costs.sum()),
    }
    if return_path:
        result.update(path=path, per_gt_cost=per_gt, matches_per_gt=counts)
    return result


def fractional_waypoint_prefix(token: np.ndarray, fraction: float) -> np.ndarray:
    """Trim an already-capped per-waypoint token at fractional source progress.

    Only the final partial episode budget may interpolate an M-based boundary.
    Velocities remain unchanged: no GT-clock retiming or spatial rescaling.
    """
    from egomimic.rldb.zarr.arc_length_tokenizer import slerp_pair_ypr

    value = np.asarray(token, dtype=np.float64)
    m = len(value) // 2
    if m < 2 or len(value) != 2 * m or not 0 < fraction <= 1:
        raise ValueError("Expected per-waypoint token and fraction in (0,1]")
    if fraction == 1:
        return value.copy()
    progress = fraction * (m - 1)
    lower = min(int(math.floor(progress)), m - 2)
    alpha = progress - lower
    terminal = (1 - alpha) * value[lower] + alpha * value[lower + 1]
    for offset in (3, 10):
        terminal[offset : offset + 3] = slerp_pair_ypr(
            value[lower, offset : offset + 3],
            value[lower + 1, offset : offset + 3],
            np.array([alpha]),
        )[0]
    # An exact waypoint endpoint must not be repeated (zero extra interval).
    if alpha <= 1e-12 and lower >= 1:
        return np.concatenate((value[: lower + 1], value[m : m + lower + 1]))
    return np.concatenate(
        (
            value[: lower + 1],
            terminal[None],
            value[m : m + lower + 1],
            value[m + lower + 1 : m + lower + 2],
        )
    )


def score_distance_dtw_episode(evaluator, records: list[dict]) -> dict:
    """Build one rollout and perform exactly one global DTW for the episode."""
    from egomimic.eval.open_loop_sim import (
        arc_execution_prefix,
        arc_prefix_control_steps,
        executed_arc_waypoints,
    )
    from egomimic.rldb.zarr.arc_length_tokenizer import (
        TokenizeBimanualArcLengthCartesian,
    )

    records = sorted(records, key=lambda item: int(item["frame"]))
    if any(METRIC_FRAME_KEY not in item for item in records):
        raise ValueError(
            "Distance DTW requires evaluation.eef_to_world metadata from "
            "the updated wrist-frame data transforms; refusing mixed frames"
        )
    gt = np.concatenate(
        [
            world_xyz(item["ground_truth"][:1], item[METRIC_FRAME_KEY])
            for item in records
        ]
    )
    cumulative = joint_cumulative_distance(gt)
    is_arc = evaluator._is_arc_prediction(records[0]["prediction"])
    if any(
        evaluator._is_arc_prediction(item["prediction"]) != is_arc for item in records
    ):
        raise ValueError("Mixed ARC/baseline representations within an episode")
    budget = None
    if is_arc:
        if evaluator.velocity_mode != "per_waypoint":
            raise ValueError("Distance-budget DTW requires per-waypoint ARC velocity")
        m = evaluator.resampled_vector_length
        fraction = evaluator.execute_fraction
        if evaluator.arc_execution_cap_mode == "waypoints":
            fraction = (executed_arc_waypoints(m, fraction) - 1) / (m - 1)
        budget = evaluator.min_distance_unit * fraction
        anchors, budgets = distance_windows(cumulative, budget)
        tokenizer = TokenizeBimanualArcLengthCartesian(
            min_distance_unit=evaluator.min_distance_unit,
            rotation_distance_unit=getattr(evaluator, "rotation_distance_unit", None),
            resampled_vector_length=m,
            dt=evaluator.control_dt,
            velocity_mode=evaluator.velocity_mode,
            translation_horizon_mode=getattr(
                evaluator, "translation_horizon_mode", "joint"
            ),
        )
    else:
        anchors = np.arange(0, len(records), evaluator.execute_steps)
        budgets = np.zeros(len(anchors))
    predictions, steps = [], []
    total = 0
    for anchor, remaining_budget in zip(anchors, budgets):
        record = records[int(anchor)]
        if is_arc:
            fraction = min(1.0, float(remaining_budget / budget))
            cap_fraction = evaluator.execute_fraction
            if evaluator.arc_execution_cap_mode == "distance":
                cap_fraction *= fraction
            partial = arc_execution_prefix(
                record["prediction"],
                cap_fraction,
                evaluator.velocity_mode,
                evaluator.min_distance_unit,
                evaluator.arc_execution_cap_mode,
            )
            if evaluator.arc_execution_cap_mode == "waypoints" and fraction < 1:
                partial = fractional_waypoint_prefix(partial, fraction)
            # Decode the entire prefix on its OWN clock, even if slower than GT.
            n = arc_prefix_control_steps(
                partial,
                1.0,
                evaluator.velocity_mode,
                evaluator.control_dt,
                evaluator.min_distance_unit,
                rotation_distance_unit=getattr(
                    evaluator, "rotation_distance_unit", None
                ),
                arc_execution_cap_mode="waypoints",
                translation_horizon_mode=getattr(
                    evaluator, "translation_horizon_mode", "joint"
                ),
            )
            if n > evaluator.dtw_max_prediction_steps:
                raise ValueError(
                    f"DTW prefix has {n} samples; increase dtw_max_prediction_steps explicitly"
                )
            decoded = tokenizer.detokenize(partial, action_horizon=n)
        else:
            decoded, n = evaluator._decode_prediction_with_steps(
                record["prediction"], max_steps=len(records) - int(anchor)
            )
        total += n
        if total * len(gt) > evaluator.dtw_max_cells:
            raise ValueError(
                "Episode exceeds dtw_max_cells; no GT/prediction samples were discarded"
            )
        predictions.append(world_xyz(decoded, record[METRIC_FRAME_KEY]))
        steps.append(n)
    prediction = np.concatenate(predictions)
    result = global_dtw(prediction, gt, max_cells=evaluator.dtw_max_cells)
    result.update(
        metric_version=METRIC_VERSION,
        segments=len(anchors),
        anchor_frames=[int(records[int(i)]["frame"]) for i in anchors],
        segment_control_steps=steps,
        execution_distance_budget_m=budget,
        segment_distance_budgets_m=budgets.tolist() if is_arc else None,
        gt_joint_distance_m=float(cumulative[-1]),
        stationary_fallback=bool(is_arc and cumulative[-1] <= 1e-12),
        predicted_duration_s=float(total * evaluator.control_dt),
        gt_duration_s=float(len(gt) * evaluator.control_dt),
        duration_ratio=float(total / len(gt)),
        rollout_mode="gt_distance_budget" if is_arc else "fixed_control_frames",
    )
    return result


def summarize_distance_dtw(episodes: list[dict]) -> dict:
    scores = [item["distance_dtw"] for item in episodes]
    total_gt = sum(item["gt_frames"] for item in scores)
    return {
        "metric_version": METRIC_VERSION,
        "xyz_mse": sum(item["xyz_mse"] * item["gt_frames"] for item in scores)
        / total_gt,
        "episode_xyz_mse": float(np.mean([item["xyz_mse"] for item in scores])),
        "gt_frames": total_gt,
        "predicted_samples": sum(item["predicted_samples"] for item in scores),
        "segments": sum(item["segments"] for item in scores),
        "gt_coverage": sum(item["gt_coverage"] * item["gt_frames"] for item in scores)
        / total_gt,
        "prediction_coverage": min(item["prediction_coverage"] for item in scores),
        "duration_ratio": sum(item["predicted_samples"] for item in scores) / total_gt,
    }
