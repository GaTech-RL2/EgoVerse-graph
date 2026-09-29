"""Chunk continuity metrics and trajectory renderers."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from egomimic.eval.diagnostics._io import atomic_json, atomic_npz, sha256, utc_now

CHUNK_SEAM_SCHEMA_VERSION = 1


def _trajectory(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] < 2:
        raise ValueError(
            f"{name} must have shape (steps, action_dim>=2), got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def circular_delta(first, second) -> np.ndarray:
    """Shortest signed difference ``second - first`` in radians."""

    return np.arctan2(
        np.sin(np.asarray(second) - first), np.cos(np.asarray(second) - first)
    )


def _derivative(values: np.ndarray, order: int, dt: float) -> np.ndarray:
    if values.shape[0] <= order:
        return np.empty((0, values.shape[1]), dtype=np.float32)
    return np.diff(values, n=order, axis=0) / float(dt) ** order


def chunk_seam_metrics(
    previous_tail: Any,
    new_head: Any,
    *,
    dt: float = 1.0,
    theta_index: int | None = 2,
    grip_index: int | None = 3,
) -> dict[str, float | int | None]:
    """Compare two predictions for the same future action offsets.

    The old unused tail and the freshly predicted head are aligned by future
    offset. Position, velocity, and jerk use the first two native coordinates;
    theta uses circular distance, and grip uses absolute distance.
    """

    old = _trajectory(previous_tail, "previous_tail")
    new = _trajectory(new_head, "new_head")
    if old.shape[1] != new.shape[1]:
        raise ValueError(f"action dimensions differ: {old.shape[1]} vs {new.shape[1]}")
    if not np.isfinite(dt) or float(dt) <= 0.0:
        raise ValueError("dt must be finite and positive")
    count = min(old.shape[0], new.shape[0])
    old = old[:count]
    new = new[:count]
    position_error = np.linalg.norm(new[:, :2] - old[:, :2], axis=1)
    old_velocity = _derivative(old[:, :2], 1, dt)
    new_velocity = _derivative(new[:, :2], 1, dt)
    old_jerk = _derivative(old[:, :2], 3, dt)
    new_jerk = _derivative(new[:, :2], 3, dt)

    def rms_difference(first: np.ndarray, second: np.ndarray) -> float:
        if first.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(second - first, dtype=np.float64))))

    result: dict[str, float | int | None] = {
        "aligned_steps": int(count),
        "action_dim": int(old.shape[1]),
        "position_mean_l2": float(position_error.mean()),
        "position_max_l2": float(position_error.max()),
        "velocity_rms_difference": rms_difference(old_velocity, new_velocity),
        "jerk_rms_difference": rms_difference(old_jerk, new_jerk),
        "rotation_mean_abs_radians": None,
        "grip_mean_abs_difference": None,
    }
    if theta_index is not None and 0 <= int(theta_index) < old.shape[1]:
        result["rotation_mean_abs_radians"] = float(
            np.mean(np.abs(circular_delta(old[:, theta_index], new[:, theta_index])))
        )
    if grip_index is not None and 0 <= int(grip_index) < old.shape[1]:
        result["grip_mean_abs_difference"] = float(
            np.mean(np.abs(new[:, grip_index] - old[:, grip_index]))
        )
    return result


def write_chunk_seam_artifact(
    events: Sequence[dict[str, Any]],
    output_dir: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write variable-length seam events without object arrays."""

    if not events:
        raise ValueError("cannot write an empty chunk-seam artifact")
    final = Path(output_dir).resolve()
    if final.exists():
        raise FileExistsError(f"chunk-seam artifact already exists: {final}")
    staging = final.with_name(f".{final.name}.tmp-{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=False)

    previous_rows = []
    new_rows = []
    executed_rows = []
    offsets = [0]
    timesteps = []
    embodiment_ids = []
    episode_indices = []
    for event in events:
        previous = _trajectory(event["previous_tail"], "previous_tail")
        new = _trajectory(event["new_head"], "new_head")
        executed = _trajectory(event.get("executed_head", new), "executed_head")
        count = min(previous.shape[0], new.shape[0], executed.shape[0])
        if len({previous.shape[1], new.shape[1], executed.shape[1]}) != 1:
            raise ValueError("chunk-seam event action dimensions differ")
        previous_rows.append(previous[:count])
        new_rows.append(new[:count])
        executed_rows.append(executed[:count])
        offsets.append(offsets[-1] + count)
        timesteps.append(int(event["t"]))
        embodiment_ids.append(int(event["embodiment_id"]))
        episode_indices.append(int(event["episode_index"]))

    arrays_path = staging / "chunk_seams.npz"
    atomic_npz(
        arrays_path,
        previous_tail=np.concatenate(previous_rows),
        new_head=np.concatenate(new_rows),
        executed_head=np.concatenate(executed_rows),
        event_offsets=np.asarray(offsets, dtype=np.int64),
        timestep=np.asarray(timesteps, dtype=np.int64),
        embodiment_id=np.asarray(embodiment_ids, dtype=np.int64),
        episode_index=np.asarray(episode_indices, dtype=np.int64),
    )
    manifest = {
        "schema_version": CHUNK_SEAM_SCHEMA_VERSION,
        "created_at": utc_now(),
        "event_count": len(events),
        "arrays": arrays_path.name,
        "arrays_sha256": sha256(arrays_path),
        "metadata": dict(metadata or {}),
    }
    atomic_json(staging / "manifest.json", manifest)
    atomic_json(
        staging / "COMPLETE.json",
        {"manifest_sha256": sha256(staging / "manifest.json")},
    )
    final.parent.mkdir(parents=True, exist_ok=True)
    os.rename(staging, final)
    return final


def validate_chunk_seam_artifact(artifact_dir: str | Path) -> dict[str, Any]:
    root = Path(artifact_dir).resolve(strict=True)
    manifest_path = root / "manifest.json"
    complete = json.loads((root / "COMPLETE.json").read_text())
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != CHUNK_SEAM_SCHEMA_VERSION:
        raise ValueError("chunk-seam schema version mismatch")
    if complete.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("chunk-seam manifest hash mismatch")
    arrays_path = root / manifest["arrays"]
    if sha256(arrays_path) != manifest["arrays_sha256"]:
        raise ValueError("chunk-seam array hash mismatch")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        offsets = arrays["event_offsets"]
        if offsets.shape != (int(manifest["event_count"]) + 1,):
            raise ValueError("chunk-seam offset count mismatch")
        if offsets[0] != 0 or np.any(np.diff(offsets) <= 0):
            raise ValueError("chunk-seam event offsets are invalid")
        if int(offsets[-1]) != int(arrays["previous_tail"].shape[0]):
            raise ValueError("chunk-seam row count mismatch")
    return manifest


def _render_seam_event(
    previous: np.ndarray,
    new: np.ndarray,
    executed: np.ndarray,
    output: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = min(len(previous), len(new), len(executed))
    previous, new, executed = previous[:count], new[:count], executed[:count]
    offset = np.arange(count)
    old_velocity = _derivative(previous[:, :2], 1, 1.0)
    new_velocity = _derivative(new[:, :2], 1, 1.0)
    old_jerk = _derivative(previous[:, :2], 3, 1.0)
    new_jerk = _derivative(new[:, :2], 3, 1.0)

    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    axes[0, 0].plot(previous[:, 0], previous[:, 1], "o-", label="previous unused tail")
    axes[0, 0].plot(new[:, 0], new[:, 1], "o-", label="new raw head")
    axes[0, 0].plot(executed[:, 0], executed[:, 1], "x--", label="executed head")
    axes[0, 0].set(title="position path", xlabel="x", ylabel="y")
    for dim, label in ((0, "x"), (1, "y")):
        axes[0, 1].plot(offset, previous[:, dim], label=f"old {label}")
        axes[0, 1].plot(offset, new[:, dim], linestyle="--", label=f"new {label}")
    axes[0, 1].set(title="position by future offset", xlabel="future offset")

    if previous.shape[1] >= 3:
        axes[1, 0].plot(offset, previous[:, 2], label="previous theta")
        axes[1, 0].plot(offset, new[:, 2], linestyle="--", label="new theta")
        axes[1, 0].set(title="rotation", ylabel="radians")
    else:
        axes[1, 0].text(0.5, 0.5, "no rotation channel", ha="center", va="center")
    if previous.shape[1] >= 4:
        axes[1, 1].plot(offset, previous[:, 3], label="previous grip")
        axes[1, 1].plot(offset, new[:, 3], linestyle="--", label="new grip")
        axes[1, 1].set(title="grip")
    else:
        axes[1, 1].text(0.5, 0.5, "no grip channel", ha="center", va="center")

    axes[2, 0].plot(
        np.arange(len(old_velocity)),
        np.linalg.norm(old_velocity, axis=1),
        label="previous",
    )
    axes[2, 0].plot(
        np.arange(len(new_velocity)),
        np.linalg.norm(new_velocity, axis=1),
        linestyle="--",
        label="new",
    )
    axes[2, 0].set(title="position velocity magnitude", xlabel="future offset")
    axes[2, 1].plot(
        np.arange(len(old_jerk)), np.linalg.norm(old_jerk, axis=1), label="previous"
    )
    axes[2, 1].plot(
        np.arange(len(new_jerk)),
        np.linalg.norm(new_jerk, axis=1),
        linestyle="--",
        label="new",
    )
    axes[2, 1].set(title="position jerk magnitude", xlabel="future offset")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def render_chunk_seam_report(
    artifact_dir: str | Path, output_dir: str | Path
) -> dict[str, Path]:
    """Render each replan seam plus a machine-readable metric summary."""

    root = Path(artifact_dir).resolve(strict=True)
    manifest = validate_chunk_seam_artifact(root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    rendered = []
    with np.load(root / manifest["arrays"], allow_pickle=False) as arrays:
        offsets = arrays["event_offsets"]
        for index in range(int(manifest["event_count"])):
            start, end = int(offsets[index]), int(offsets[index + 1])
            previous = arrays["previous_tail"][start:end]
            new = arrays["new_head"][start:end]
            executed = arrays["executed_head"][start:end]
            metrics = chunk_seam_metrics(previous, new)
            row = {
                "event_index": index,
                "embodiment_id": int(arrays["embodiment_id"][index]),
                "episode_index": int(arrays["episode_index"][index]),
                "timestep": int(arrays["timestep"][index]),
                **metrics,
            }
            rows.append(row)
            figure = output / f"chunk_seam_event_{index:04d}.png"
            _render_seam_event(
                previous,
                new,
                executed,
                figure,
                title=(
                    f"Chunk seam {index}: emb{row['embodiment_id']} "
                    f"episode {row['episode_index']} at t={row['timestep']}"
                ),
            )
            rendered.append(figure)

    json_path = output / "chunk_seam_metrics.json"
    atomic_json(json_path, {"schema_version": 1, "events": rows})
    csv_path = output / "chunk_seam_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"metrics_json": json_path, "metrics_csv": csv_path, "figures": rendered}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--artifact", required=True)
    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--artifact", required=True)
    render_parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.command == "validate":
        print(json.dumps(validate_chunk_seam_artifact(args.artifact), indent=2))
    else:
        print(
            json.dumps(
                render_chunk_seam_report(args.artifact, args.output_dir),
                indent=2,
                default=str,
            )
        )


if __name__ == "__main__":
    main()
