"""Distance-budgeted episode rollouts and exact, GT-frame-balanced global DTW.

The alignment is symmetric in its allowed moves, but its *reporting weight* is
one vote per recorded GT timestamp. Timing remains a separate diagnostic.
"""

from __future__ import annotations

import numpy as np

METRIC_VERSION = "arc_arrival_global_dtw_v7"
METRIC_FRAME_KEY = "evaluation.eef_to_world"
XYZ_COLS = (0, 1, 2, 7, 8, 9)
ARC_DISTANCE_SEMANTICS = "min_cumulative_per_arm_translation"


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


def per_arm_cumulative_distance(xyz: np.ndarray) -> np.ndarray:
    """Cumulative translation for each arm, preserving which arm moved first."""
    values = np.asarray(xyz, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6 or not len(values):
        raise ValueError("Expected nonempty (T,6) bimanual XYZ")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite GT positions")
    steps = np.linalg.norm(np.diff(values, axis=0).reshape(-1, 2, 3), axis=2)
    return np.concatenate((np.zeros((1, 2)), np.cumsum(steps, axis=0)))


def translation_progress(xyz: np.ndarray) -> np.ndarray:
    """Shared progress after independently accumulating each arm's distance."""
    return np.min(per_arm_cumulative_distance(xyz), axis=1)


def joint_cumulative_distance(xyz: np.ndarray) -> np.ndarray:
    # Preserve the original accumulation order, including exact threshold
    # rounding used to select GT frame anchors in historical joint-D scores.
    values = np.asarray(xyz, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6 or not len(values):
        raise ValueError("Expected nonempty (T,6) bimanual XYZ")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite GT positions")
    steps = np.linalg.norm(np.diff(values, axis=0).reshape(-1, 2, 3), axis=2).sum(
        axis=1
    )
    return np.concatenate(([0.0], np.cumsum(steps)))


def global_dtw(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    *,
    max_cells: int = 50_000_000,
    return_path: bool = False,
) -> dict:
    """Exact endpoint-anchored DTW with one shared XYZ alignment (6 or 3 columns).

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
        or pred.shape[1:] != gt.shape[1:]
        or pred.shape[1] not in (3, 6)
        or not len(pred)
        or not len(gt)
    ):
        raise ValueError("DTW requires nonempty (T,6) or per-arm (T,3) XYZ sequences")
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


def score_distance_dtw_episode(evaluator, records: list[dict]) -> dict:
    """Build one rollout and perform exactly one global DTW for the episode."""
    from egomimic.eval.open_loop_sim import (
        executed_arc_waypoints,
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
    joint_distance = joint_cumulative_distance(gt)
    is_arc = evaluator._is_arc_prediction(records[0]["prediction"])
    if any(
        evaluator._is_arc_prediction(item["prediction"]) != is_arc for item in records
    ):
        raise ValueError("Mixed ARC/baseline representations within an episode")
    budget = None
    cumulative = translation_progress(gt) if is_arc else joint_distance
    if is_arc:
        m = evaluator.resampled_vector_length
        fraction = (executed_arc_waypoints(m, evaluator.execute_fraction) - 1) / (m - 1)
        budget = evaluator.min_distance_unit * fraction
    else:
        anchors = np.arange(0, len(records), evaluator.execute_steps)
    if is_arc:
        # Recorded-observation open-loop segment rollout: from the current frame, execute the
        # first 30% of the predicted waypoints with every stream on its own
        # clock, stop when the first moving stream runs out, then replan from
        # the recorded observation at that frame. Both arms share the replan
        # points and are scored by one DTW, exactly as the open-loop metric
        # and the videos execute a chunk.
        predictions, segment_steps, anchor_frames, clipped = [], [], [], []
        cursor, total = 0, 0
        while cursor < len(records):
            record = records[cursor]
            decoded, n = evaluator._decode_prediction_with_steps(
                record["prediction"],
                initial_action=record.get("arc.current_action"),
                max_steps=len(records) - cursor,
            )
            predictions.append(world_xyz(decoded, record[METRIC_FRAME_KEY]))
            segment_steps.append(n)
            anchor_frames.append(int(record["frame"]))
            cursor += n
            total += n
        prediction = np.concatenate(predictions)
        result = global_dtw(prediction, gt, max_cells=evaluator.dtw_max_cells)
        segments = len(segment_steps)
        segment_budgets = None
    else:
        prediction, segment_steps, total, clipped = _baseline_rollout(
            evaluator, records, anchors, len(gt)
        )
        result = global_dtw(prediction, gt, max_cells=evaluator.dtw_max_cells)
        segments = len(anchors)
        anchor_frames = [int(records[int(i)]["frame"]) for i in anchors]
        segment_budgets = None
    result.update(
        metric_version=METRIC_VERSION,
        segments=segments,
        anchor_frames=anchor_frames,
        segment_control_steps=segment_steps,
        clipped_segments=len(clipped),
        clipped_anchor_indices=clipped,
        execution_distance_budget_m=budget,
        segment_distance_budgets_m=segment_budgets,
        distance_semantics=ARC_DISTANCE_SEMANTICS if is_arc else None,
        distance_window_semantics="first_stream_replan" if is_arc else None,
        gt_joint_distance_m=float(joint_distance[-1]),
        gt_mode_progress_m=float(cumulative[-1]),
        gt_per_arm_distance_m=per_arm_cumulative_distance(gt)[-1].tolist(),
        rotation_distance_unit=(
            getattr(evaluator, "rotation_distance_unit", None) if is_arc else None
        ),
        rotation_clock=(
            "shared_independent"
            if is_arc and getattr(evaluator, "rotation_distance_unit", None) is not None
            else None
        ),
        stationary_fallback=bool(is_arc and cumulative[-1] <= 1e-12),
        predicted_duration_s=float(total * evaluator.control_dt),
        gt_duration_s=float(len(gt) * evaluator.control_dt),
        duration_ratio=float(total / len(gt)),
        rollout_mode=(
            "first_stream_replan"
            if is_arc
            else ("gt_distance_budget" if is_arc else "fixed_control_frames")
        ),
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
        "clipped_segments": sum(item.get("clipped_segments", 0) for item in scores),
    }


def _baseline_rollout(evaluator, records, anchors, gt_frames):
    predictions, steps = [], []
    total = 0
    for anchor in anchors:
        record = records[int(anchor)]
        decoded, n = evaluator._decode_prediction_with_steps(
            record["prediction"], max_steps=len(records) - int(anchor)
        )
        total += n
        if total * gt_frames > evaluator.dtw_max_cells:
            raise ValueError("Episode exceeds dtw_max_cells")
        predictions.append(world_xyz(decoded, record[METRIC_FRAME_KEY]))
        steps.append(n)
    return np.concatenate(predictions), steps, total, []
