#!/usr/bin/env python3
"""Audit the native-action round trip of a fixed-horizon Planar ARC codec."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import zarr

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from egomimic.pipeline.pushshapes import PlanarArcTrajectoryNativeDecoder
from egomimic.rldb.embodiment.pushshapes import get_planar_arc_length_transform_list


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def angle_error(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(actual - expected), np.cos(actual - expected))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--domain", help="Manifest domain; inferred for a single-domain split"
    )
    parser.add_argument(
        "--waypoint-sampling", choices=("uniform", "curvature"), default="uniform"
    )
    parser.add_argument("--raw-action-horizon", type=int, default=40)
    parser.add_argument("--action-target-offset", type=int, default=1)
    parser.add_argument("--token-horizon", type=int, default=32)
    parser.add_argument("--waypoints", type=int, default=16)
    parser.add_argument("--arc-distance", type=float, default=40.0)
    parser.add_argument("--rotation-radius", type=float, default=0.0)
    parser.add_argument("--hybrid-rotation-unit", type=float, default=0.14776)
    parser.add_argument("--max-anchor-error", type=float, default=1e-4)
    args = parser.parse_args(argv)

    if args.output.exists():
        raise FileExistsError(args.output)
    if args.raw_action_horizon <= 1 or args.action_target_offset < 0:
        raise ValueError(
            "raw horizon must exceed one and target offset must be non-negative"
        )
    if args.waypoints <= 0 or args.token_horizon != 2 * args.waypoints:
        raise ValueError("duration ARC requires exactly two token rows per waypoint")

    split = json.loads(args.split_manifest.read_text())
    domains = split["domains"]
    domain_name = args.domain
    if domain_name is None:
        if len(domains) != 1:
            raise ValueError("--domain is required for a multi-domain manifest")
        domain_name = next(iter(domains))
    domain = domains[domain_name]
    valid_ids = list(domain["valid_ids"])
    if (
        not valid_ids
        or len(valid_ids) != len(set(valid_ids))
        or len(valid_ids) != int(domain["valid_count"])
    ):
        raise ValueError("split manifest valid IDs/count mismatch")
    if set(valid_ids) & set(domain["train_ids"]):
        raise ValueError("split manifest has train/validation ID overlap")

    transforms = get_planar_arc_length_transform_list(
        action_horizon=args.token_horizon,
        raw_action_horizon=args.raw_action_horizon,
        action_target_offset=args.action_target_offset,
        min_distance_unit=args.arc_distance,
        resampled_vector_length=args.waypoints,
        dt=1.0 / 30.0,
        rotation_radius=args.rotation_radius,
        hybrid_rotation_unit=args.hybrid_rotation_unit,
        velocity_mode="duration",
        waypoint_sampling=args.waypoint_sampling,
    )
    decoder = None
    rows = []
    for episode_id in valid_ids:
        episode_path = args.dataset_root / f"{episode_id}.zarr"
        group = zarr.open_group(str(episode_path), mode="r")
        actions = np.asarray(group["actions"])
        if actions.ndim != 2 or actions.shape[1] not in (2, 3, 4):
            raise ValueError(
                f"invalid native action shape for {episode_id}: {actions.shape}"
            )
        if decoder is None:
            decoder = PlanarArcTrajectoryNativeDecoder(
                resampled_vector_length=args.waypoints,
                native_action_dim=actions.shape[1],
                raw_action_horizon=args.raw_action_horizon,
                dt=1.0 / 30.0,
                rotation_radius=args.rotation_radius,
                velocity_mode="duration",
            )
        total_frames = int(group.attrs.get("total_frames", len(actions)))
        usable_frames = min(total_frames, len(actions))
        loader_horizon = args.raw_action_horizon + args.action_target_offset
        if usable_frames < loader_horizon:
            raise ValueError(
                f"{episode_id} has only {usable_frames} usable action rows"
            )
        start = (usable_frames - loader_horizon) // 2
        loader_window = actions[start : start + loader_horizon].astype(np.float32)
        if loader_window.shape != (loader_horizon, decoder.native_action_dim):
            raise ValueError(
                f"unexpected native window shape for {episode_id}: {loader_window.shape}"
            )

        batch = {"actions": loader_window.copy()}
        for transform in transforms:
            batch = transform.transform(batch)
        token = batch["actions"]
        if token.shape != (args.token_horizon, 5):
            raise ValueError(f"unexpected token shape for {episode_id}: {token.shape}")
        decoded = np.asarray(decoder.decode(token))[0]
        target = loader_window[
            args.action_target_offset : args.action_target_offset
            + args.raw_action_horizon
        ]
        if decoded.shape != target.shape or not np.all(np.isfinite(decoded)):
            raise ValueError(
                f"invalid decoded trajectory for {episode_id}: {decoded.shape}"
            )
        native_error = decoded - target
        if decoder.native_action_dim >= 3:
            native_error[:, 2] = angle_error(decoded[:, 2], target[:, 2])
        anchor_error = float(np.max(np.abs(native_error[0])))
        if anchor_error > args.max_anchor_error:
            raise ValueError(
                f"target anchor mismatch for {episode_id}: {anchor_error} > {args.max_anchor_error}"
            )
        xy_error = decoded[:, :2] - target[:, :2]
        row = {
            "episode_id": episode_id,
            "window_start": start,
            "loader_window_rows": loader_horizon,
            "target_rows": len(target),
            "token_shape": list(token.shape),
            "decoded_shape": list(decoded.shape),
            "anchor_error_max_abs": anchor_error,
            "xy_rmse": float(np.sqrt(np.mean(np.square(xy_error)))),
            "native_max_abs_error_per_channel": np.max(
                np.abs(native_error), axis=0
            ).tolist(),
            "represented_duration_seconds": float(
                np.sum(token[args.waypoints : -1, 0])
            ),
            "stored_observation_alignment": group.attrs.get("observation_alignment"),
        }
        if decoder.native_action_dim >= 3:
            row["theta_rmse_rad"] = float(
                np.sqrt(np.mean(np.square(native_error[:, 2])))
            )
        if decoder.native_action_dim == 4:
            row["grip_rmse"] = float(np.sqrt(np.mean(np.square(native_error[:, 3]))))
        rows.append(row)

    result = {
        "schema_version": 1,
        "evidence_scope": "codec round trip; not a causal-storage audit or a closed-loop policy score",
        "dataset": {
            "domain": domain_name,
            "root": str(args.dataset_root),
            "split_manifest": str(args.split_manifest),
            "split_manifest_sha256": sha256(args.split_manifest),
            "valid_episode_count": len(valid_ids),
            "valid_names_sha256": domain["valid_names_sha256"],
        },
        "contract": {
            "loader_window_horizon": args.raw_action_horizon
            + args.action_target_offset,
            "action_target_offset": args.action_target_offset,
            "raw_action_horizon": args.raw_action_horizon,
            "token_horizon": args.token_horizon,
            "waypoints": args.waypoints,
            "native_action_dim": decoder.native_action_dim,
            "waypoint_sampling": args.waypoint_sampling,
            "timing": "per_waypoint_duration_cubic_v1",
            "anchor_invariant": "decoded[0] == raw_window[action_target_offset]",
        },
        "summary": {
            "anchor_error_max_abs": max(row["anchor_error_max_abs"] for row in rows),
            "xy_rmse_mean": float(np.mean([row["xy_rmse"] for row in rows])),
        },
        "episodes": rows,
    }
    for metric in ("theta_rmse_rad", "grip_rmse"):
        if metric in rows[0]:
            result["summary"][f"{metric}_mean"] = float(
                np.mean([row[metric] for row in rows])
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
