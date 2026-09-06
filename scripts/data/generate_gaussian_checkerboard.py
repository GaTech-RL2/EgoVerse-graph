#!/usr/bin/env python3
"""Materialize an independent Gaussian-to-2D-checkerboard benchmark."""

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

from egomimic.synthetic import generate_gaussian_checkerboard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4_096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-dim", type=int, default=8)
    parser.add_argument("--cluster-std", type=float, default=0.12)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    args = parser.parse_args()
    manifest_path = args.output.with_suffix(".json")
    if args.output.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite an existing dataset")
    if not 0 < args.train_fraction < 1 or not 0 <= args.val_fraction < 1:
        raise ValueError("invalid split fractions")
    if args.train_fraction + args.val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be below one")

    batch = generate_gaussian_checkerboard(
        args.count,
        seed=args.seed,
        source_dim=args.source_dim,
        cluster_std=args.cluster_std,
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
        source_gaussian_latent=batch.source_gaussian_latent.numpy(),
        target_2d=batch.target_2d.numpy(),
        mode_indices=batch.mode_indices.numpy(),
        mode_centers=batch.mode_centers.numpy(),
        mode_weights=batch.mode_weights.numpy(),
        split=split,
    )
    manifest = {
        "schema_version": 1,
        "kind": "independent_gaussian_to_asymmetric_2d_checkerboard_mixture",
        "count": args.count,
        "seed": args.seed,
        "source_dim": args.source_dim,
        "cluster_std": args.cluster_std,
        "mode_centers": batch.mode_centers.tolist(),
        "mode_weights": batch.mode_weights.tolist(),
        "source_distribution": f"independent N(0,I_{args.source_dim})",
        "target_distribution": "eight disconnected 2D Gaussian islands",
        "split_counts": {
            str(index): int((split == index).sum()) for index in range(3)
        },
        "npz_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
