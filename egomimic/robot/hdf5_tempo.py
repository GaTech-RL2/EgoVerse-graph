"""Classify locally collected YAM HDF5 episodes by Cartesian tempo.

Typical station usage::

    python -m egomimic.robot.hdf5_tempo \
      --data demos/yam_gello \
      --watch-seconds 10 \
      --output demos/yam_gello/tempo_report.json

Use ``--reference`` with a previously saved report to keep bucket boundaries
fixed across a collection session. Without a reference, boundaries are fitted
from the completed HDF5 episodes currently present under ``--data``.

The frame-speed signal matches the repository's E1 Step-0 convention: observed
Cartesian positions are low-pass filtered with a zero-phase fourth-order
Butterworth filter, finite-differenced at the recording rate, and reduced by
taking the maximum speed over the two arms. The episode tempo is the mean of
that signal after samples at or below ``--hold-threshold`` are removed.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import butter, sosfiltfilt

BUCKET_NAMES = ("slow", "medium", "fast")
CUT_METHODS = ("equal_width", "equal_width_robust", "tertile")
POSE_KEYS = {
    "observations": "observations/eepose",
    "actions": "actions/eepose",
}


@dataclass(frozen=True)
class EpisodeTempo:
    path: str
    frames: int
    duration_s: float
    fps: float
    tempo_mps: float
    moving_fraction: float
    median_mps: float
    p95_mps: float


def buckets(tempo: float, cuts: Sequence[float]) -> str:
    """Assign ``tempo`` to slow/medium/fast using two ascending cuts."""
    cut_array = np.asarray(cuts, dtype=np.float64)
    if cut_array.shape != (2,) or not np.all(np.isfinite(cut_array)):
        raise ValueError(f"cuts must contain exactly two finite values, got {cuts!r}")
    if cut_array[0] > cut_array[1]:
        raise ValueError(f"cuts must be ascending, got {cuts!r}")
    return BUCKET_NAMES[int(np.searchsorted(cut_array, tempo, side="right"))]


def compute_cut_sets(tempos: Sequence[float]) -> dict[str, list[float]]:
    """Return equal-width, robust equal-width, and tertile boundaries."""
    values = np.asarray(tempos, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("cannot compute cuts without at least one finite tempo")

    minimum, maximum = float(np.min(values)), float(np.max(values))
    q05, q95 = np.quantile(values, [0.05, 0.95])
    return {
        "equal_width": np.linspace(minimum, maximum, 4)[1:3].tolist(),
        "equal_width_robust": np.linspace(float(q05), float(q95), 4)[1:3].tolist(),
        "tertile": np.quantile(values, [1.0 / 3.0, 2.0 / 3.0]).tolist(),
    }


def _validate_pose_array(pose: np.ndarray, *, key: str, path: Path) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] < 10:
        raise ValueError(
            f"{path}: {key} must have shape (frames, >=10), got {pose.shape}"
        )
    if pose.shape[0] < 2:
        raise ValueError(f"{path}: {key} needs at least two frames")
    if not np.all(np.isfinite(pose[:, [0, 1, 2, 7, 8, 9]])):
        raise ValueError(f"{path}: {key} contains non-finite Cartesian positions")
    return pose


def _lowpass_positions(
    pos: np.ndarray, cutoff_hz: float, sample_hz: float
) -> np.ndarray:
    """Exact lightweight copy of the repository's E1 Step-0 position filter."""
    pos = np.asarray(pos, dtype=np.float64)
    if cutoff_hz <= 0 or len(pos) < 20:
        return pos
    sos = butter(4, cutoff_hz, btype="low", fs=sample_hz, output="sos")
    return sosfiltfilt(sos, pos, axis=0)


def frame_speed(
    pose: np.ndarray,
    *,
    fps: float,
    lowpass_hz: float = 3.0,
) -> np.ndarray:
    """Step-0 per-frame speed: filtered XYZ, maximum over both arms."""
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be positive and finite, got {fps}")
    arm_speeds = []
    for xyz_slice in (slice(0, 3), slice(7, 10)):
        xyz = _lowpass_positions(pose[:, xyz_slice], lowpass_hz, fps)
        step_speed = np.linalg.norm(np.diff(xyz, axis=0), axis=1) * fps
        arm_speeds.append(np.concatenate([step_speed, step_speed[-1:]]))
    return np.maximum(arm_speeds[0], arm_speeds[1])


def read_episode_tempo(
    path: Path,
    *,
    fps: float,
    pose_source: str = "observations",
    lowpass_hz: float = 3.0,
    hold_threshold: float = 0.05,
    require_complete: bool = True,
) -> EpisodeTempo | None:
    """Read one episode without loading camera arrays; skip incomplete files."""
    key = POSE_KEYS[pose_source]
    with h5py.File(path, "r") as episode:
        if require_complete and not bool(episode.attrs.get("complete", False)):
            return None
        if key not in episode:
            raise KeyError(f"{path}: missing required dataset {key!r}")
        pose = _validate_pose_array(episode[key][:], key=key, path=path)

    speed = frame_speed(pose, fps=fps, lowpass_hz=lowpass_hz)
    moving = speed[speed > hold_threshold]
    if moving.size == 0:
        raise ValueError(
            f"{path}: no samples exceed hold threshold {hold_threshold:g} m/s"
        )
    return EpisodeTempo(
        path=str(path),
        frames=int(pose.shape[0]),
        duration_s=float(pose.shape[0] / fps),
        fps=float(fps),
        tempo_mps=float(np.mean(moving)),
        moving_fraction=float(moving.size / speed.size),
        median_mps=float(np.median(speed)),
        p95_mps=float(np.quantile(speed, 0.95)),
    )


def discover_hdf5(inputs: Iterable[Path], *, recursive: bool = False) -> list[Path]:
    paths: set[Path] = set()
    for candidate in inputs:
        candidate = candidate.expanduser().resolve()
        if candidate.is_file():
            if candidate.suffix.lower() in {".hdf5", ".h5"}:
                paths.add(candidate)
            continue
        if not candidate.is_dir():
            continue
        iterator = candidate.rglob("*.hdf5") if recursive else candidate.glob("*.hdf5")
        paths.update(path for path in iterator if path.is_file())
    return sorted(paths)


def load_reference_cuts(path: Path) -> dict[str, list[float]]:
    payload = json.loads(path.read_text())
    cut_sets = payload.get("cut_sets", payload)
    if not isinstance(cut_sets, Mapping):
        raise TypeError(f"{path}: expected a cut_sets mapping")
    result: dict[str, list[float]] = {}
    for method in CUT_METHODS:
        if method not in cut_sets:
            raise ValueError(f"{path}: missing reference cuts for {method}")
        cuts = [float(value) for value in cut_sets[method]]
        buckets(0.0, cuts)  # validates shape/order/finiteness
        result[method] = cuts
    return result


def build_report(
    episodes: Sequence[EpisodeTempo],
    *,
    cut_sets: Mapping[str, Sequence[float]] | None = None,
    pose_source: str,
    lowpass_hz: float,
    hold_threshold: float,
) -> dict[str, object]:
    if not episodes:
        raise ValueError("no completed HDF5 episodes were available")
    if cut_sets is None:
        cut_sets = compute_cut_sets([episode.tempo_mps for episode in episodes])
        cut_source = "fitted_from_input_episodes"
    else:
        cut_sets = {
            name: [float(value) for value in cuts] for name, cuts in cut_sets.items()
        }
        cut_source = "reference_file"

    records = []
    for episode in sorted(episodes, key=lambda item: item.path):
        record = asdict(episode)
        record["buckets"] = {
            method: buckets(episode.tempo_mps, cuts)
            for method, cuts in cut_sets.items()
        }
        records.append(record)
    return {
        "metric": "mean_active_max_bimanual_cartesian_speed_mps",
        "pose_source": pose_source,
        "lowpass_hz": lowpass_hz,
        "hold_threshold_mps": hold_threshold,
        "cut_source": cut_source,
        "cut_sets": cut_sets,
        "episode_count": len(records),
        "episodes": records,
    }


def _print_report(report: Mapping[str, object]) -> None:
    cut_sets = report["cut_sets"]
    print(
        f"Completed episodes: {report['episode_count']} | "
        f"metric={report['metric']} | cuts={report['cut_source']}"
    )
    for method in CUT_METHODS:
        cuts = cut_sets[method]
        print(f"  {method:18s}: {cuts[0]:.4f}, {cuts[1]:.4f} m/s")
    if (
        report["cut_source"] == "fitted_from_input_episodes"
        and report["episode_count"] < 3
    ):
        print(
            "  WARNING: fewer than three episodes; fitted bucket labels are not "
            "meaningful. Use --reference for live checks."
        )
    print()
    print(
        f"{'episode':28s} {'tempo':>8s} {'moving':>8s} "
        f"{'equal':>8s} {'robust':>8s} {'tertile':>8s}"
    )
    for record in report["episodes"]:
        labels = record["buckets"]
        print(
            f"{Path(record['path']).name:28.28s} "
            f"{record['tempo_mps']:8.3f} "
            f"{record['moving_fraction'] * 100:7.1f}% "
            f"{labels['equal_width']:>8s} "
            f"{labels['equal_width_robust']:>8s} "
            f"{labels['tertile']:>8s}"
        )


def _write_json_atomic(path: Path, report: Mapping[str, object]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def _file_signature(paths: Sequence[Path]) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths
    )


def _analyze_once(args: argparse.Namespace) -> tuple[tuple[str, int, int], ...]:
    paths = discover_hdf5(args.data, recursive=args.recursive)
    signature = _file_signature(paths)
    episodes = []
    skipped_incomplete = 0
    skipped_unreadable = 0
    for path in paths:
        try:
            episode = read_episode_tempo(
                path,
                fps=args.fps,
                pose_source=args.pose_source,
                lowpass_hz=args.lowpass_hz,
                hold_threshold=args.hold_threshold,
                require_complete=not args.include_incomplete,
            )
        except OSError as exc:
            if args.include_incomplete:
                raise
            skipped_unreadable += 1
            print(f"Skipping unreadable/in-progress episode {path.name}: {exc}")
            continue
        if episode is None:
            skipped_incomplete += 1
        else:
            episodes.append(episode)
    if not episodes:
        print(
            f"No completed HDF5 episodes found ({len(paths)} files, "
            f"{skipped_incomplete} incomplete)."
        )
        return signature

    reference = load_reference_cuts(args.reference) if args.reference else None
    report = build_report(
        episodes,
        cut_sets=reference,
        pose_source=args.pose_source,
        lowpass_hz=args.lowpass_hz,
        hold_threshold=args.hold_threshold,
    )
    report["skipped_incomplete"] = skipped_incomplete
    report["skipped_unreadable"] = skipped_unreadable
    _print_report(report)
    if args.output:
        _write_json_atomic(args.output, report)
        print(f"\nWrote {args.output.expanduser().resolve()}")
    return signature


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        default=[Path("demos/yam_gello")],
        help="HDF5 file(s) or directories (default: demos/yam_gello).",
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--pose-source",
        choices=tuple(POSE_KEYS),
        default="observations",
        help="Use recorded follower poses (default) or commanded poses.",
    )
    parser.add_argument("--lowpass-hz", type=float, default=3.0)
    parser.add_argument(
        "--hold-threshold",
        type=float,
        default=0.05,
        help="Exclude max-arm speed samples at or below this value in m/s.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="Prior report JSON supplying fixed cut_sets for live classification.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    parser.add_argument(
        "--watch-seconds",
        type=float,
        default=0.0,
        help="Rescan when files change; 0 runs once (recommended live value: 10).",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Analyze episodes without complete=true (unsafe during active writes).",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.hold_threshold < 0:
        raise ValueError("--hold-threshold must be non-negative")
    if args.lowpass_hz < 0 or args.lowpass_hz >= args.fps / 2:
        raise ValueError("--lowpass-hz must be in [0, fps/2)")
    if args.watch_seconds < 0:
        raise ValueError("--watch-seconds must be non-negative")

    previous = _analyze_once(args)
    if args.watch_seconds <= 0:
        return
    print(f"\nWatching every {args.watch_seconds:g}s; Ctrl-C to exit.")
    try:
        while True:
            time.sleep(args.watch_seconds)
            paths = discover_hdf5(args.data, recursive=args.recursive)
            signature = _file_signature(paths)
            if signature != previous:
                print("\nHDF5 files changed; refreshing tempo report.")
                previous = _analyze_once(args)
    except KeyboardInterrupt:
        print("\nTempo watch stopped.")


if __name__ == "__main__":
    main()
