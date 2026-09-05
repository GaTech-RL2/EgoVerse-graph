#!/usr/bin/env python3
"""Materialize the paired Gaussian-to-torus benchmark as NPZ + JSON."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from egomimic.synthetic import generate_gaussian_torus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4_096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--major-radius", type=float, default=2.0)
    parser.add_argument("--minor-radius", type=float, default=0.65)
    parser.add_argument("--source-dim", type=int, default=2)
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
        source_dim=args.source_dim,
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
        source_latent=batch.source_latent.numpy(),
        source_gaussian_latent=batch.source_gaussian_latent.numpy(),
        source_2d=batch.source_2d.numpy(),
        source_3d=batch.source_3d.numpy(),
        source_gaussian_3d=batch.source_gaussian_3d.numpy(),
        target_3d=batch.target_3d.numpy(),
        angles=batch.angles.numpy(),
        split=split,
    )
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "kind": f"paired_{args.source_dim}d_gaussian_to_3d_torus",
        "count": args.count,
        "seed": args.seed,
        "major_radius": args.major_radius,
        "minor_radius": args.minor_radius,
        "source_distribution": f"paired N(0,I_{args.source_dim}) plus independent N(0,I_{args.source_dim}) inference noise; first two paired coordinates determine the torus target",
        "source_dim": args.source_dim,
        "target_distribution": "analytic torus with CDF-derived uniform angles",
        "pairing": "deterministic standard-normal-CDF angular parameterization",
        "paired_physical_path": "linear interpolation from source_3d to target_3d",
        "model_latent_flow": "source_latent to encoder(target_3d); learned endpoint depends on model parameters",
        "split_labels": {"0": "train", "1": "validation", "2": "test"},
        "split_counts": {str(i): int((split == i).sum()) for i in range(3)},
        "npz_sha256": digest,
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
