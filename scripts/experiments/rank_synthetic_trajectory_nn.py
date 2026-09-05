#!/usr/bin/env python3
"""Rank a complete synthetic-flow matrix from exported trajectory NPZ files.

This is deliberately a post-hoc evaluator: model checkpoints are hashed but
never loaded. A scheduled exporter must first produce the compact trajectory
files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    """Hash array identity, including dtype and shape."""
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode())
    digest.update(json.dumps(contiguous.shape).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _resolve(path_text: str, relative_to: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else relative_to / path


def _load_matrix(manifest_path: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads(manifest_path.read_text())
    variants = manifest.get("variants")
    seeds = manifest.get("seeds")
    config_paths = manifest.get("configs")
    if not variants or not seeds or not config_paths:
        raise ValueError("matrix manifest must define variants, seeds, and configs")
    expected = {(str(variant), int(seed)) for variant in variants for seed in seeds}
    configs = []
    observed: dict[tuple[str, int], Path] = {}
    for path_text in config_paths:
        config_path = _resolve(path_text, manifest_path.parent)
        if not config_path.is_file():
            raise FileNotFoundError(f"matrix config does not exist: {config_path}")
        config = json.loads(config_path.read_text())
        key = (str(config["variant"]), int(config["seed"]))
        if key in observed:
            raise ValueError(
                f"duplicate config for variant={key[0]} seed={key[1]}: "
                f"{observed[key]} and {config_path}"
            )
        observed[key] = config_path
        configs.append({"path": config_path, "config": config, "key": key})
    missing = sorted(expected - set(observed))
    extra = sorted(set(observed) - expected)
    if missing or extra:
        raise ValueError(f"matrix is incomplete: missing={missing}, extra={extra}")
    return manifest, configs


def _parse_explicit_npzs(values: list[str]) -> dict[str, Path]:
    parsed = {}
    for value in values:
        run_id, separator, path_text = value.partition("=")
        if not separator or not run_id or not path_text:
            raise ValueError(
                "--trajectory-npz must use the exact form RUN_ID=/path/to/file.npz"
            )
        if run_id in parsed:
            raise ValueError(f"duplicate --trajectory-npz for {run_id}")
        parsed[run_id] = Path(path_text)
    return parsed


def _run_id(config: dict) -> str:
    wandb = config.get("wandb") or {}
    return str(wandb.get("id") or Path(config["output_dir"]).name)


def _trajectory_from_root(root: Path, run_id: str) -> Path:
    candidates = [
        root / f"{run_id}.npz",
        root / run_id / "trajectory.npz",
        root / run_id / "validation_trajectory.npz",
    ]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one trajectory for {run_id}; checked {candidates}, "
            f"found {matches}"
        )
    return matches[0]


def _terminal_checkpoint(config: dict) -> Path:
    output = Path(config["output_dir"])
    summary = output / "summary.json"
    if not summary.is_file():
        raise FileNotFoundError(
            f"run is incomplete because summary.json is missing: {summary}"
        )
    max_steps = int(config["max_steps"])
    pattern = f"*global-step-{max_steps:06d}.pt"
    matches = sorted((output / "checkpoints").glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"run is incomplete: expected one terminal checkpoint matching "
            f"{output / 'checkpoints' / pattern}, found {matches}"
        )
    return matches[0]


def _load_evaluation_targets(path: Path, particles: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"evaluation dataset does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "split" not in archive or "target_3d" not in archive:
            raise ValueError("evaluation dataset must contain split and target_3d")
        indices = np.flatnonzero(archive["split"] == 1)
        if len(indices) < particles:
            raise ValueError(
                f"evaluation dataset has {len(indices)} validation points, fewer "
                f"than the required {particles}"
            )
        target = np.asarray(archive["target_3d"][indices[:particles]])
    if target.shape != (particles, 3) or not np.isfinite(target).all():
        raise ValueError(
            f"evaluation target must be finite with shape {(particles, 3)}, "
            f"got {target.shape}"
        )
    return target


def _load_trajectory(path: Path, particles: int) -> tuple[np.ndarray, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"trajectory does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"times", "points", "target"}:
            raise ValueError(
                f"trajectory {path} must contain only times, points, and target; "
                f"found {archive.files}"
            )
        times = np.asarray(archive["times"])
        points = np.asarray(archive["points"])
        target = np.asarray(archive["target"])
    if times.ndim != 1 or len(times) < 2 or not np.all(np.diff(times) > 0):
        raise ValueError(f"trajectory {path} has an invalid time grid")
    if not np.isclose(times[0], 0.0) or not np.isclose(times[-1], 1.0):
        raise ValueError(f"trajectory {path} time grid must span [0, 1]")
    expected = (len(times), particles, 3)
    if points.shape != expected or target.shape != (particles, 3):
        raise ValueError(
            f"trajectory {path} expected points {expected} and target "
            f"{(particles, 3)}, got {points.shape} and {target.shape}"
        )
    if not np.isfinite(times).all() or not np.isfinite(points).all():
        raise ValueError(f"trajectory {path} contains non-finite values")
    if not np.isfinite(target).all():
        raise ValueError(f"trajectory {path} target contains non-finite values")
    return times, points, target


def _directional_nn_mse(
    samples: np.ndarray, targets: np.ndarray, chunk_size: int
) -> float:
    total = 0.0
    for start in range(0, len(samples), chunk_size):
        chunk = samples[start : start + chunk_size]
        squared = np.square(chunk[:, None, :] - targets[None, :, :]).sum(axis=-1)
        total += float(squared.min(axis=1).sum(dtype=np.float64))
    return total / len(samples)


def symmetric_nn_mse(
    samples: np.ndarray, targets: np.ndarray, chunk_size: int
) -> tuple[float, float, float]:
    generated_to_target = _directional_nn_mse(samples, targets, chunk_size)
    target_to_generated = _directional_nn_mse(targets, samples, chunk_size)
    return (
        0.5 * (generated_to_target + target_to_generated),
        generated_to_target,
        target_to_generated,
    )


def _declared_evaluation_dataset(configs: list[dict]) -> Path:
    declared = set()
    for entry in configs:
        path_text = entry["config"].get("evaluation_dataset")
        if not path_text:
            raise ValueError(
                "configs do not declare one common evaluation_dataset; pass "
                "--evaluation-dataset explicitly"
            )
        declared.add(_resolve(path_text, entry["path"].parent).resolve())
    if len(declared) != 1:
        raise ValueError(
            "configs do not declare one common evaluation_dataset; pass "
            "--evaluation-dataset explicitly"
        )
    return Path(declared.pop())


def _write_markdown(result: dict, output: Path) -> None:
    lines = [
        "# Post-hoc symmetric nearest-neighbor ranking",
        "",
        f"- Evaluation particles: {result['comparison']['particles']}",
        f"- Evaluation dataset SHA-256: `{result['comparison']['dataset']['sha256']}`",
        f"- Target cloud SHA-256: `{result['comparison']['target_sha256']}`",
        f"- Time grid SHA-256: `{result['comparison']['times_sha256']}`",
        "",
        "| Rank | Variant | Symmetric NN MSE (mean ± sample std) | Per seed |",
        "|---:|---|---:|---|",
    ]
    for rank, row in enumerate(result["ranking"], start=1):
        values = ", ".join(
            f"{entry['seed']}: {entry['symmetric_nn_mse']:.9g}"
            for entry in row["per_seed"]
        )
        lines.append(
            f"| {rank} | {row['variant']} | {row['mean']:.9g} ± "
            f"{row['sample_std']:.9g} | {values} |"
        )
    lines.extend(
        [
            "",
            "The symmetric metric is the average of generated→target and "
            "target→generated nearest-neighbor squared distances. All rows use "
            "the exact same fixed target cloud, particle count, and time grid.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trajectory-root", type=Path)
    source.add_argument(
        "--trajectory-npz",
        action="append",
        default=[],
        metavar="RUN_ID=PATH",
        help="repeat once for every run in the complete matrix",
    )
    parser.add_argument(
        "--evaluation-dataset",
        type=Path,
        help="shared post-hoc dataset; defaults to the configs' common dataset",
    )
    parser.add_argument("--expected-particles", type=int, default=2048)
    parser.add_argument("--distance-chunk-size", type=int, default=512)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_particles <= 1:
        raise ValueError("expected-particles must exceed one")
    if args.distance_chunk_size <= 0:
        raise ValueError("distance-chunk-size must be positive")

    manifest, configs = _load_matrix(args.manifest)
    explicit = _parse_explicit_npzs(args.trajectory_npz)
    expected_run_ids = {_run_id(entry["config"]) for entry in configs}
    if len(expected_run_ids) != len(configs):
        raise ValueError("matrix configs must have unique run IDs")
    if explicit and set(explicit) != expected_run_ids:
        raise ValueError(
            "explicit trajectory set does not match complete matrix: "
            f"missing={sorted(expected_run_ids - set(explicit))}, "
            f"extra={sorted(set(explicit) - expected_run_ids)}"
        )

    dataset = args.evaluation_dataset or _declared_evaluation_dataset(configs)
    reference_target = _load_evaluation_targets(dataset, args.expected_particles)
    dataset_sha256 = sha256_file(dataset)
    declared_dataset = (manifest.get("datasets") or {}).get("evaluation") or {}
    if args.evaluation_dataset is None and declared_dataset.get("sha256"):
        if declared_dataset["sha256"] != dataset_sha256:
            raise ValueError(
                "evaluation dataset hash differs from the matrix manifest: "
                f"declared={declared_dataset['sha256']} actual={dataset_sha256}"
            )

    reference_times = None
    run_rows = []
    for entry in configs:
        config = entry["config"]
        variant, seed = entry["key"]
        run_id = _run_id(config)
        trajectory_path = (
            explicit[run_id]
            if explicit
            else _trajectory_from_root(args.trajectory_root, run_id)
        )
        checkpoint = _terminal_checkpoint(config)
        times, points, target = _load_trajectory(
            trajectory_path, args.expected_particles
        )
        if not np.array_equal(target, reference_target):
            raise ValueError(
                f"trajectory target does not exactly match evaluation dataset: "
                f"{trajectory_path}"
            )
        if reference_times is None:
            reference_times = times
        elif not np.array_equal(times, reference_times):
            raise ValueError(
                f"trajectory time grid differs from the first run: {trajectory_path}"
            )
        metric, precision, support = symmetric_nn_mse(
            points[-1], target, args.distance_chunk_size
        )
        run_rows.append(
            {
                "variant": variant,
                "seed": seed,
                "run_id": run_id,
                "symmetric_nn_mse": metric,
                "generated_to_target_nn_mse": precision,
                "target_to_generated_nn_mse": support,
                "config": {
                    "path": str(entry["path"]),
                    "sha256": sha256_file(entry["path"]),
                },
                "checkpoint": {
                    "path": str(checkpoint),
                    "global_step": int(config["max_steps"]),
                    "sha256": sha256_file(checkpoint),
                },
                "dataset": {"path": str(dataset), "sha256": dataset_sha256},
                "trajectory": {
                    "path": str(trajectory_path),
                    "sha256": sha256_file(trajectory_path),
                },
            }
        )

    ranking = []
    for variant in manifest["variants"]:
        per_seed = sorted(
            (
                {
                    "seed": row["seed"],
                    "symmetric_nn_mse": row["symmetric_nn_mse"],
                }
                for row in run_rows
                if row["variant"] == variant
            ),
            key=lambda item: item["seed"],
        )
        values = np.asarray(
            [item["symmetric_nn_mse"] for item in per_seed], dtype=np.float64
        )
        ranking.append(
            {
                "variant": variant,
                "per_seed": per_seed,
                "mean": float(values.mean()),
                "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
        )
    ranking.sort(key=lambda row: (row["mean"], row["variant"]))
    result = {
        "schema_version": 1,
        "metric": "symmetric_nearest_neighbor_mse",
        "comparison": {
            "particles": args.expected_particles,
            "dataset": {"path": str(dataset), "sha256": dataset_sha256},
            "target_sha256": sha256_array(reference_target),
            "times_sha256": sha256_array(reference_times),
            "time_steps": len(reference_times),
        },
        "ranking": ranking,
        "runs": sorted(run_rows, key=lambda row: (row["variant"], row["seed"])),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    _write_markdown(result, args.output_markdown)


if __name__ == "__main__":
    main()
