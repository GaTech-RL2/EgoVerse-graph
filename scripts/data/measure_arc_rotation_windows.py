"""Measure SO(3) travel over fixed-frame windows of explicitly named episodes.

No images, recursive discovery, or ARC distance capping. A 100-frame window
contains 99 relative rotations. Report left/right separately and mark recording
gaps instead of silently treating discontinuities as ordinary 30 Hz intervals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial.transform import Rotation


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"windows": 0}
    return {
        "windows": len(values),
        "mean_degrees": float(values.mean()),
        "median_degrees": float(np.median(values)),
        "p10_degrees": float(np.quantile(values, 0.1)),
        "p90_degrees": float(np.quantile(values, 0.9)),
    }


def measure(root, episode, frames=100):
    group = zarr.open_group(str(Path(root) / episode), mode="r")
    embodiment = str(group.attrs["embodiment"])
    count = int(group.attrs["total_frames"])
    steps = frames - 1
    if frames < 2 or count < frames:
        raise ValueError(f"{episode}: need at least {frames} frames, got {count}")
    gap_mask = None
    if "obs_rgb_timestamps_ns" in group:
        timestamps = np.asarray(group["obs_rgb_timestamps_ns"][:count])
        intervals = np.diff(timestamps) * 1e-9
        gaps = (intervals <= 0) | (intervals > 0.05)
        gap_mask = (
            np.convolve(gaps.astype(int), np.ones(steps, dtype=int), mode="valid") > 0
        )
    result = {
        "episode": episode,
        "embodiment": embodiment,
        "frames": count,
        "window_frames": frames,
        "overlapping_windows": True,
        "timestamp_gap_threshold_s": 0.05 if gap_mask is not None else None,
    }
    values = {}
    for arm in ("left", "right"):
        key = (
            f"{arm}.obs_ee_pose"
            if embodiment == "human_bimanual"
            else f"{arm}.cmd_ee_pose"
        )
        pose = np.asarray(group[key][:count], dtype=np.float64)
        quat = pose[:, [4, 5, 6, 3]]  # Native xyz + wxyz.
        if not np.isfinite(quat).all() or np.any(np.linalg.norm(quat, axis=1) < 1e-9):
            raise ValueError(
                f"{episode}/{key}: invalid quaternion in valid frame range"
            )
        rotations = Rotation.from_quat(quat)
        increments = np.rad2deg((rotations[:-1].inv() * rotations[1:]).magnitude())
        cumulative = np.r_[0.0, np.cumsum(increments)]
        travel = cumulative[steps:] - cumulative[:-steps]
        continuous = travel[~gap_mask] if gap_mask is not None else travel
        result[arm] = {
            "all_windows": stats(travel),
            "without_timestamp_gaps": stats(continuous),
        }
        values[arm] = (travel, continuous)
    return result, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--episode", action="append", required=True)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    episodes, pooled = [], {}
    for episode in args.episode:
        result, values = measure(args.dataset_root, episode, args.frames)
        episodes.append(result)
        for arm, arrays in values.items():
            bucket = pooled.setdefault((result["embodiment"], arm), [[], []])
            for part, array in zip(bucket, arrays):
                part.append(array)
    summaries = {}
    for (embodiment, arm), arrays in pooled.items():
        summaries.setdefault(embodiment, {})[arm] = {
            "all_windows": stats(np.concatenate(arrays[0])),
            "without_timestamp_gaps": stats(np.concatenate(arrays[1])),
        }
    report = {
        "definition": "Sum of 99 adjacent SO(3) geodesic angles for 100-frame windows, not endpoint angle.",
        "scope": "Only the explicitly named episodes; frame-weighted within each embodiment, no human/robot pooling.",
        "summary": summaries,
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
