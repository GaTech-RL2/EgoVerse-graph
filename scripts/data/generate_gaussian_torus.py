#!/usr/bin/env python3
"""Materialize the paired Gaussian-to-torus benchmark as NPZ + JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from egomimic.synthetic import generate_gaussian_torus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4_096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--major-radius", type=float, default=2.0)
    parser.add_argument("--minor-radius", type=float, default=0.65)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("refusing to overwrite an existing dataset")
    if not 0 < args.train_fraction < 1 or not 0 <= args.val_fraction < 1:
        raise ValueError("invalid split fractions")
    if args.train_fraction + args.val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be below one")

    batch = generate_gaussian_torus(
        args.count,
        seed=args.seed,
        major_radius=args.major_radius,
        minor_radius=args.minor_radius,
    )
    rng = np.random.default_rng(args.seed + 1)
    permutation = rng.permutation(args.count)
    split = np.full(args.count, 2, dtype=np.uint8)
    train_end = round(args.count * args.train_fraction)
    val_end = train_end + round(args.count * args.val_fraction)
    split[permutation[:train_end]] = 0
    split[permutation[train_end:val_end]] = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        source_2d=batch.source_2d.numpy(),
        source_3d=batch.source_3d.numpy(),
        target_3d=batch.target_3d.numpy(),
        angles=batch.angles.numpy(),
        split=split,
    )
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "kind": "paired_2d_gaussian_to_3d_torus",
        "count": args.count,
        "seed": args.seed,
        "major_radius": args.major_radius,
        "minor_radius": args.minor_radius,
        "source_distribution": "N(0,I_2), embedded as [z0,z1,0] for 3D flow",
        "target_distribution": "analytic torus with CDF-derived uniform angles",
        "pairing": "deterministic standard-normal-CDF angular parameterization",
        "linear_cfm_velocity": "target_3d - source_3d",
        "split_labels": {"0": "train", "1": "validation", "2": "test"},
        "split_counts": {str(i): int((split == i).sum()) for i in range(3)},
        "npz_sha256": digest,
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
