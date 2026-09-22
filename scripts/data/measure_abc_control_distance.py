"""Estimate joint EEF travel on training-only 100-frame windows, never images.

Uses the same episode split as MultiDataset and a deterministic, stratified
20% sample of training episodes. All complete sliding windows in each selected
episode contribute equally, so weighting matches uniform control-frame draws.
"""

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import boto3
import numpy as np
import zarr

from egomimic.rldb.zarr.control_rate import control_frame_indices
from egomimic.rldb.zarr.zarr_dataset_multi import split_dataset_names
from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df

TASKS = (
    "fold and stack the towels",
    "take the fake fruits out of the plastic bag",
    "sort the stationery into containers",
    "insert the wireless bluetooth earbuds into the charging case",
)


def window_distances(left, right, horizon=100):
    left, right = (
        np.asarray(left, dtype=np.float64),
        np.asarray(right, dtype=np.float64),
    )
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 3:
        raise ValueError("Expected matching (T,3) left/right XYZ arrays")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Non-finite EEF coordinates")
    step = np.linalg.norm(np.diff(left, axis=0), axis=1)
    step += np.linalg.norm(np.diff(right, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(step)]
    # 100 successive commanded poses contain 99 inter-pose displacements.
    return cumulative[horizon - 1 :] - cumulative[: len(left) - horizon + 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    df = episode_table_to_df(create_default_engine())
    numeric_cache = args.output.parent / "distance-numeric-cache"
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    df = df[(df.lab == "abc") & (df.embodiment == "yam_bimanual") & df.task.isin(TASKS)]
    df = df[~df.is_deleted.fillna(False) & df.zarr_processed_path.fillna("").ne("")]
    rows = {r["episode_hash"]: r for r in df.to_dict("records")}
    train, valid = split_dataset_names(rows, valid_ratio=0.2, seed=42)
    selected = []
    for task in TASKS:
        ids = sorted(
            (key for key in train if rows[key]["task"] == task),
            key=lambda key: hashlib.sha256(("42:" + key).encode()).hexdigest(),
        )
        if not ids:
            raise ValueError(f"No training episodes for {task}")
        selected.extend(ids[: max(1, round(len(ids) * 0.20))])
    print(
        json.dumps(
            {
                "matched": len(rows),
                "train": len(train),
                "valid": len(valid),
                "sampled": len(selected),
            }
        ),
        flush=True,
    )

    def measure(key):
        row = rows[key]
        path = args.root / key
        if not path.is_dir():
            path = numeric_cache / key
            parsed = urlparse(row["zarr_processed_path"])
            prefix = parsed.path.lstrip("/").rstrip("/") + "/"
            for page in s3.get_paginator("list_objects_v2").paginate(
                Bucket=parsed.netloc, Prefix=prefix
            ):
                for obj in page.get("Contents", []):
                    relative = obj["Key"][len(prefix) :]
                    if relative not in (
                        "zarr.json",
                        ".zattrs",
                        ".zgroup",
                        ".zmetadata",
                    ) and not relative.startswith(
                        ("left.cmd_ee_pose/", "right.cmd_ee_pose/")
                    ):
                        continue
                    target = path / relative
                    if not target.is_file() or target.stat().st_size != obj["Size"]:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        s3.download_file(parsed.netloc, obj["Key"], str(target))
        store = zarr.open_group(str(path), mode="r")
        n = int(store.attrs["total_frames"])
        fps = float(store.attrs["fps"])
        indices = control_frame_indices(n, fps, 30)
        if len(indices) < 100:
            raise ValueError(f"{key}: fewer than 100 control frames")
        left = np.asarray(store["left.cmd_ee_pose"][:n, :3])[indices]
        right = np.asarray(store["right.cmd_ee_pose"][:n, :3])[indices]
        dist = window_distances(left, right)
        return {
            "episode": key,
            "task": row["task"],
            "source_fps": fps,
            "windows": len(dist),
            "sum_m": float(dist.sum()),
            "sum_squared_m": float(np.square(dist).sum()),
            "mean_m": float(dist.mean()),
        }

    results, errors = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(measure, key): key for key in selected}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:
                errors.append(
                    {"episode": futures[future], "error_type": type(error).__name__}
                )
            if (len(results) + len(errors)) % 25 == 0:
                print(
                    f"measured={len(results)} errors={len(errors)} total={len(selected)}",
                    flush=True,
                )
    if errors:
        raise RuntimeError(
            f"Calibration incomplete; refusing biased D: {errors[:5]} ({len(errors)} errors)"
        )
    count = sum(row["windows"] for row in results)
    mean = sum(row["sum_m"] for row in results) / count
    report = {
        "definition": "sum of left and right EEF XYZ path lengths; nearest-frame 30Hz resampling; 100 poses / 99 intervals; complete sliding windows",
        "lab": "abc",
        "tasks": TASKS,
        "seed": 42,
        "split_seed": 42,
        "sample_fraction_of_training_episodes": 0.20,
        "horizon": 100,
        "control_dt": 1 / 30,
        "train_episode_count": len(train),
        "valid_episode_count": len(valid),
        "train_episode_ids_sha256": hashlib.sha256(
            "".join(k + "\n" for k in sorted(train)).encode()
        ).hexdigest(),
        "sampled_episodes": len(results),
        "windows": count,
        "mean_joint_distance_m": mean,
        "recommended_D_m": max(0.01, round(mean, 2)),
        "per_task": {
            task: {
                "episodes": sum(row["task"] == task for row in results),
                "mean_m": sum(row["sum_m"] for row in results if row["task"] == task)
                / sum(row["windows"] for row in results if row["task"] == task),
            }
            for task in TASKS
        },
        "episodes": sorted(results, key=lambda row: row["episode"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.json")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}), flush=True)


if __name__ == "__main__":
    main()
