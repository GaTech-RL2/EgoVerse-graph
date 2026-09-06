#!/usr/bin/env python3
"""Materialize matched uniform sphere- and cube-surface datasets."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from egomimic.synthetic import generate_gaussian_sphere_cube


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-dim", type=int, default=8)
    parser.add_argument("--sphere-radius", type=float, default=2.0)
    parser.add_argument(
        "--cube-half-extent", type=float, default=math.sqrt(12.0 / 5.0)
    )
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    if not 0 < args.train_fraction < 1 or not 0 <= args.val_fraction < 1:
        raise ValueError("invalid split fractions")
    if args.train_fraction + args.val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be below one")

    batch = generate_gaussian_sphere_cube(
        args.count,
        seed=args.seed,
        sphere_radius=args.sphere_radius,
        cube_half_extent=args.cube_half_extent,
        source_dim=args.source_dim,
    )
    rng = np.random.default_rng(args.seed + 1)
    permutation = rng.permutation(args.count)
    split = np.full(args.count, 2, dtype=np.uint8)
    train_end = round(args.count * args.train_fraction)
    val_end = train_end + round(args.count * args.val_fraction)
    split[permutation[:train_end]] = 0
    split[permutation[train_end:val_end]] = 1

    args.output_dir.mkdir(parents=True)
    common = {
        "source_latent": batch.source_latent.numpy(),
        "source_gaussian_latent": batch.source_gaussian_latent.numpy(),
        "source_3d": batch.source_3d.numpy(),
        "source_gaussian_3d": batch.source_gaussian_3d.numpy(),
        "surface_uniform": batch.surface_uniform.numpy(),
        "split": split,
    }
    paths = {
        "sphere": args.output_dir / f"{args.prefix}_sphere.npz",
        "cube": args.output_dir / f"{args.prefix}_cube.npz",
    }
    np.savez_compressed(
        paths["sphere"], **common, target_3d=batch.sphere_target_3d.numpy()
    )
    np.savez_compressed(paths["cube"], **common, target_3d=batch.cube_target_3d.numpy())
    manifest = {
        "schema_version": 1,
        "kind": "matched_uniform_sphere_and_cube_surfaces",
        "count": args.count,
        "seed": args.seed,
        "source_dim": args.source_dim,
        "sphere_radius": args.sphere_radius,
        "cube_half_extent": args.cube_half_extent,
        "scale_contract": "equal expected squared radius",
        "pairing": "shared base Gaussian/CDF variables and identical split; pairing is not consumed by training",
        "split_counts": {str(index): int((split == index).sum()) for index in range(3)},
        "files": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
