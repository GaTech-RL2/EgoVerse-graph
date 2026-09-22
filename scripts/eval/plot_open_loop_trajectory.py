#!/usr/bin/env python3
"""Plot executed GT/predicted paths and their time parameterization."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ARM_OFFSETS = (("Left arm", 0), ("Right arm", 7))


def _combined_distance(actions: np.ndarray) -> np.ndarray:
    intervals = np.linalg.norm(np.diff(actions[:, 0:3], axis=0), axis=-1)
    intervals += np.linalg.norm(np.diff(actions[:, 7:10], axis=0), axis=-1)
    return np.concatenate(([0.0], np.cumsum(intervals)))


def _load(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    with np.load(path) as data:
        pred = np.asarray(data["prediction"], dtype=np.float64)
        gt = np.asarray(data["ground_truth"], dtype=np.float64)
        dt = float(data["control_dt"])
    if pred.ndim != 2 or gt.ndim != 2 or pred.shape[1] != 14 or gt.shape[1] != 14:
        raise ValueError(f"{path} does not contain (T, 14) trajectories")
    return pred, gt, dt


def plot_comparison(
    baseline_path: Path, arc_path: Path, output_path: Path, epoch: int
) -> None:
    rows = [
        ("Baseline", *_load(baseline_path)),
        ("ARC D40 / M100", *_load(arc_path)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

    for row, (label, pred, gt, dt) in enumerate(rows):
        marker_every = max(1, min(len(pred), len(gt)) // 6)
        for column, (arm_label, offset) in enumerate(ARM_OFFSETS):
            ax = axes[row, column]
            ax.plot(
                gt[:, offset],
                gt[:, offset + 1],
                "o-",
                color="#16803c",
                label="GT",
                markersize=3,
                markevery=marker_every,
            )
            ax.plot(
                pred[:, offset],
                pred[:, offset + 1],
                "s-",
                color="#c7352d",
                label="Pred",
                markersize=3,
                markevery=marker_every,
            )
            ax.scatter(
                gt[0, offset], gt[0, offset + 1], color="#16803c", marker="o", s=55
            )
            ax.scatter(
                pred[0, offset],
                pred[0, offset + 1],
                color="#c7352d",
                marker="s",
                s=45,
            )
            ax.set_title(f"{label} — {arm_label}")
            ax.set_xlabel("camera-frame x (m)")
            ax.set_ylabel("camera-frame y (m)")
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=0.25)
            ax.legend(loc="best")

        ax = axes[row, 2]
        gt_time = np.arange(len(gt)) * dt
        pred_time = np.arange(len(pred)) * dt
        ax.plot(gt_time, _combined_distance(gt), color="#16803c", label="GT")
        ax.plot(pred_time, _combined_distance(pred), color="#c7352d", label="Pred")
        ax.set_title(f"{label} — combined arm travel")
        ax.set_xlabel("execution time (s)")
        ax.set_ylabel("left + right distance (m)")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

    fig.suptitle(
        f"Open-loop executed prefixes at shared checkpoint epoch {epoch}\n"
        "Markers are equally spaced in control time: shifted markers on the same "
        "path indicate a speed mismatch.",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--arc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    args = parser.parse_args()
    plot_comparison(args.baseline, args.arc, args.output, args.epoch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
