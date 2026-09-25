#!/usr/bin/env python3
"""Measure D for the RL2 `organize_stationary` slice.

D is the distance budget one ARC chunk spends. To make the ARC variants
comparable to the time-indexed baseline, D is the MEDIAN arc length a baseline
100-frame chunk travels. The median rather than the mean: chunk distance is
right-skewed, a handful of fast reaches drag the mean above what a typical chunk
covers, and a D above the typical chunk makes most chunks run long.

The three chunking modes do not read the same clock, so this reports both:

  joint_distance  -> left travel + right travel  (one shared clock)
  race            -> min(left, right)            (first arm to reach D ends it)
  multistream     -> each arm separately         (per-arm clocks)

A single D means a different chunk duration under each mode. Both candidate
values are reported so that trade-off is visible before D is pinned.

The distance definitions are imported from the tokenizer rather than
reimplemented, so the measured value is in exactly the units the tokenizer will
later consume as `min_distance_unit`.

Read-only: syncs episodes from S3 into the shared dataset dir and writes one
JSON report. It submits nothing and trains nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr

from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.arc_length_tokenizer import (
    cumulative_arc_length,
    cumulative_bimanual_rotation_length,
    cumulative_bimanual_translation_length,
)
from egomimic.rldb.zarr.zarr_dataset_multi import S3EpisodeResolver

# Byte-identical to the predicate in both stationery_rl2_organize_* data configs.
# The measurement must cover exactly the episodes the runs will train on.
FILTER_LAMBDA = (
    "lambda row: row['embodiment'] == 'yam_bimanual' and "
    "row['lab'] == 'rl2' and row['task'] == 'organize_stationary' and "
    "row['zarr_processed_path'] != '' and row['is_deleted'] == False"
)

# The ARC clock runs on raw world-frame command poses, before the eef_frame
# transform, so D is measured on the same arrays the tokenizer reads.
POSE_KEYS = {"left": "left.cmd_ee_pose", "right": "right.cmd_ee_pose"}


def load_actions(store: Path) -> np.ndarray | None:
    """Assemble the (T, 14) layout the tokenizer indexes.

    Columns are [left xyz, left ypr, left gripper, right xyz, right ypr, right
    gripper]; cumulative_bimanual_* reads 0:3 / 3:6 and 7:10 / 10:13. The
    gripper columns are zero-filled because no distance clock consults them.
    """
    try:
        group = zarr.open(str(store), mode="r")
    except Exception:
        return None
    poses = {}
    for side, key in POSE_KEYS.items():
        if key not in group:
            return None
        pose = np.asarray(group[key], dtype=np.float64)
        if pose.ndim != 2 or pose.shape[1] < 6:
            return None
        poses[side] = pose[:, :6]
    length = min(len(poses["left"]), len(poses["right"]))
    if length < 2:
        return None
    actions = np.zeros((length, 14), dtype=np.float64)
    actions[:, 0:6] = poses["left"][:length]
    actions[:, 7:13] = poses["right"][:length]
    return actions


def chunk_measurements(actions: np.ndarray, horizon: int, stride: int) -> dict:
    """Terminal clock readings for each horizon-length window."""
    out = {k: [] for k in ("joint_t", "left_t", "right_t", "min_t", "joint_r")}
    # A chunk is the horizon frames a policy emits at once, so the last valid
    # start is T - horizon; shorter tails would understate the budget.
    for start in range(0, max(0, len(actions) - horizon + 1), stride):
        window = actions[start : start + horizon]
        left = float(cumulative_arc_length(window[:, 0:3])[-1])
        right = float(cumulative_arc_length(window[:, 7:10])[-1])
        out["joint_t"].append(float(cumulative_bimanual_translation_length(window)[-1]))
        out["left_t"].append(left)
        out["right_t"].append(right)
        out["min_t"].append(min(left, right))
        out["joint_r"].append(float(cumulative_bimanual_rotation_length(window)[-1]))
    return out


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        **{f"p{p}": float(np.percentile(array, p)) for p in (5, 25, 50, 75, 95)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip the S3 sync; still resolve the episode list through the filter.",
    )
    args = parser.parse_args()

    filters = DatasetFilter(filter_lambdas=[FILTER_LAMBDA])
    if args.no_sync:
        # Still resolve through the filter. Listing the dataset dir instead would
        # sweep in every unrelated episode that shares that directory.
        rows = S3EpisodeResolver._get_filtered_paths(filters)
    else:
        rows = S3EpisodeResolver.sync_from_filters(
            bucket_name="rldb", filters=filters, local_dir=args.dataset_dir
        )

    pooled = {k: [] for k in ("joint_t", "left_t", "right_t", "min_t", "joint_r")}
    used, skipped = [], []
    for _, episode_hash in rows:
        actions = load_actions(args.dataset_dir / episode_hash)
        if actions is None:
            skipped.append(episode_hash)
            continue
        for key, values in chunk_measurements(
            actions, args.horizon, args.stride
        ).items():
            pooled[key].extend(values)
        used.append(episode_hash)

    report = {
        "filter": FILTER_LAMBDA,
        "action_horizon": args.horizon,
        "stride": args.stride,
        "episodes_selected": len(rows),
        "episodes_measured": len(used),
        "episodes_skipped": skipped,
        "joint_translation_m": summarize(pooled["joint_t"]),
        "left_translation_m": summarize(pooled["left_t"]),
        "right_translation_m": summarize(pooled["right_t"]),
        "first_arm_translation_m": summarize(pooled["min_t"]),
        "joint_rotation_rad": summarize(pooled["joint_r"]),
    }
    # D is the median. The mean and the full percentile spread stay in the
    # report so the skew is visible before the value is pinned into a config.
    if report["joint_translation_m"]:
        report["candidate_D"] = {
            "statistic": "median",
            "joint_clock": report["joint_translation_m"]["p50"],
            "first_arm_clock": report["first_arm_translation_m"]["p50"],
            "note": (
                "joint_clock matches joint_distance chunking. Under race the "
                "same number is reached later, since one arm alone must travel "
                "it; first_arm_clock is the race-equivalent budget."
            ),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if used else 1


if __name__ == "__main__":
    raise SystemExit(main())
