"""Paired, episode-disjoint LIBERO global-basis reconstruction diagnostic.

This reads the released replay through its native action/split conventions.
It does not train, evaluate policies, or claim simulator success from MSE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial.transform import Slerp

from egomimic.benchmarks.libero.catalog import validate_task_coverage
from egomimic.rldb.zarr.libero_arc import _integrate_commands
from egomimic.rldb.zarr.libero_arc_global import LiberoArcGlobalCodec
from egomimic.rldb.zarr.libero_dataset import validation_mask


def digest_array(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(array.tobytes()).hexdigest()


def sample_windows(path, *, episodes_per_task=2, windows_per_episode=2, seed=42):
    replay = zarr.open_group(str(path), mode="r")
    actions = np.asarray(replay["data/action"][:], dtype=np.float32)
    ends = np.asarray(replay["meta/episode_ends"][:], dtype=np.int64)
    tasks = np.asarray(replay["data/task_uid"][:]).reshape(-1)
    validate_task_coverage(np.unique(tasks), "libero_10")
    if actions.shape != (int(ends[-1]), 7) or not np.isfinite(actions).all():
        raise ValueError("Invalid released action array")
    starts = np.r_[0, ends[:-1]]
    valid = validation_mask(len(ends), val_ratio=0.1, seed=seed)
    rng = np.random.default_rng(seed)
    windows, identities = [], []
    for split, mask in (("train", ~valid), ("valid", valid)):
        for task in np.unique(tasks):
            episodes = np.flatnonzero(mask & (tasks[starts] == task))
            if not len(episodes):
                raise ValueError(f"No {split} episode for task {task}")
            selected = rng.choice(
                episodes, min(episodes_per_task, len(episodes)), replace=False
            )
            for episode in sorted(selected):
                begin, end = int(starts[episode]), int(ends[episode])
                if not np.all(tasks[begin:end] == task):
                    raise ValueError("Task identity changes within an episode")
                offsets = rng.choice(
                    end - begin, min(windows_per_episode, end - begin), replace=False
                )
                for offset in sorted(offsets):
                    raw = actions[begin + offset : min(begin + offset + 32, end)]
                    raw = np.pad(raw, ((0, 32 - len(raw)), (0, 0)), mode="edge")
                    windows.append(raw)
                    identities.append(
                        {
                            "split": split,
                            "task_uid": int(task),
                            "episode": int(episode),
                            "offset": int(offset),
                        }
                    )
    return (
        np.stack(windows),
        identities,
        {
            "dataset": str(Path(path).resolve()),
            "seed": seed,
            "validation_ratio": 0.1,
            "train_episode_ids": np.flatnonzero(~valid).tolist(),
            "valid_episode_ids": np.flatnonzero(valid).tolist(),
            "actions_sha256": digest_array(actions),
            "episode_ends_sha256": digest_array(ends),
            "task_uid_sha256": digest_array(tasks),
            "windows_sha256": digest_array(np.stack(windows)),
        },
    )


def _geometry(actions):
    xyz, rotation = _integrate_commands(actions, 0.05, 0.5)
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=-1))]
    angle = np.r_[0.0, np.cumsum((rotation[1:] * rotation[:-1].inv()).magnitude())]
    grid = np.linspace(0, 1, 129)
    keep_t, keep_r = (
        np.r_[True, np.diff(distance) > 1e-12],
        np.r_[True, np.diff(angle) > 1e-12],
    )
    pose = np.column_stack(
        [
            np.interp(grid * distance[-1], distance[keep_t], xyz[keep_t, i])
            for i in range(3)
        ]
    )
    orientation = (
        Slerp(angle[keep_r] / angle[-1], rotation[keep_r])(
            np.clip(grid, 0, angle[keep_r][-1] / angle[-1])
        )
        if keep_r.sum() > 1
        else rotation[np.zeros(len(grid), dtype=int)]
    )
    return pose, orientation, xyz, rotation


def errors(raw, decoded):
    difference = decoded - raw
    expected, predicted = _geometry(raw), _geometry(decoded)
    return {
        "action_mse": float(np.mean(difference**2)),
        "translation_command_mse": float(np.mean(difference[:, :3] ** 2)),
        "rotation_command_mse": float(np.mean(difference[:, 3:6] ** 2)),
        "gripper_sign_error": float(
            np.mean(np.sign(decoded[:, 6]) != np.sign(raw[:, 6]))
        ),
        "geometric_xyz_rmse_m": float(
            np.sqrt(np.mean((expected[0] - predicted[0]) ** 2))
        ),
        "geometric_rotation_rmse_rad": float(
            np.sqrt(np.mean((predicted[1] * expected[1].inv()).magnitude() ** 2))
        ),
        "time_indexed_xyz_rmse_m": float(
            np.sqrt(np.mean((expected[2] - predicted[2]) ** 2))
        ),
        "time_indexed_rotation_rmse_rad": float(
            np.sqrt(np.mean((predicted[3] * expected[3].inv()).magnitude() ** 2))
        ),
    }


def fit_bounded_scaling(tokens, train_mask, geometry, floor_fraction=0.05):
    """Diagnostic affine scale, fitted only to designated training windows.

    The floor is 5% of each physical block's RMS feature standard deviation
    (at least 1e-4). It avoids enormous gains on nearly constant high modes.
    This does not change the codec, and is not automatically installed in a
    training config. Native OAT action normalization remains a separate layer.
    """
    tokens = np.asarray(tokens, dtype=np.float64)
    train_mask = np.asarray(train_mask)
    if (
        tokens.ndim != 3
        or train_mask.dtype != bool
        or train_mask.shape != (len(tokens),)
        or train_mask.sum() < 2
        or not np.isfinite(tokens).all()
        or not 0 < floor_fraction <= 1
    ):
        raise ValueError("Finite tokens and at least two training windows required")
    center = tokens[train_mask].mean(axis=0)
    std = tokens[train_mask].std(axis=0)
    scale = std.copy().reshape(-1)
    blocks = (
        (0, 3 * geometry),
        (3 * geometry, 9 * geometry),
        (9 * geometry, 9 * geometry + 31),
        (9 * geometry + 31, 9 * geometry + 62),
        (9 * geometry + 62, tokens[0].size - 2),
        (tokens[0].size - 2, tokens[0].size),
    )
    for start, end in blocks:
        values = scale[start:end]
        floor = max(1e-4, floor_fraction * float(np.sqrt(np.mean(values**2))))
        scale[start:end] = np.maximum(values, floor)
    return center, scale.reshape(center.shape), std


def noise_audit(codec, tokens, raw, identities, sigmas=(0.001, 0.01), seed=42):
    """Perturb final denoiser targets, not intermediate diffusion noise states.

    Fixed per-component residual RMS is not equal physical error or compute.
    Pair random draws between scaling variants and report target dispersion
    too, so smaller inverse scales cannot masquerade as a policy improvement.
    """
    train = np.array([row["split"] == "train" for row in identities])
    physical = tokens.astype(np.float64) / codec.token_scale()
    center, scale, std = fit_bounded_scaling(physical, train, codec.geometry)
    clean = np.stack([codec.decode(row) for row in tokens])
    modes = {
        "physical": (np.zeros_like(center), np.ones_like(scale)),
        "bounded_train_std": (center, scale),
    }
    result = {
        "fit_windows": int(train.sum()),
        "evaluation_windows": int((~train).sum()),
        "fit_split": "train_only",
        "not_policy_scores": True,
        "native_normalization_unchanged": True,
        "std_quantiles": np.quantile(std, [0, 0.1, 0.5, 0.9, 1]).tolist(),
        "affine": {
            "kind": "bounded_train_std_v1",
            "basis": codec.basis,
            "geometry": codec.geometry,
            "clock": codec.clock,
            "geometry_fit_grid": codec.geometry_fit_grid,
            "fit_split": "train_only",
            "training_windows_sha256": digest_array(raw[train]),
            "center": center.tolist(),
            "scale": scale.tolist(),
        },
        "variants": {},
    }
    for name, (offset, divisor) in modes.items():
        normalized = (physical - offset) / divisor
        roundtrip = ((normalized * divisor + offset) * codec.token_scale()).astype(
            np.float32
        )
        np.testing.assert_allclose(roundtrip, tokens, rtol=1e-6, atol=1e-7)
        variant = {
            "train_feature_std_quantiles": np.quantile(
                normalized[train].std(axis=0), [0, 0.1, 0.5, 0.9, 1]
            ).tolist(),
            "valid_target_abs_p99": float(
                np.quantile(np.abs(normalized[~train]), 0.99)
            ),
            "perturbations": {},
        }
        for sigma in sigmas:
            rng = np.random.default_rng(seed)
            predicted = normalized[~train] + sigma * rng.standard_normal(
                normalized[~train].shape
            )
            perturbed = ((predicted * divisor + offset) * codec.token_scale()).astype(
                np.float32
            )
            decoded = np.stack([codec.decode(row) for row in perturbed])
            delta = np.mean((decoded - clean[~train]) ** 2, axis=(1, 2))
            total = [errors(a, b) for a, b in zip(raw[~train], decoded)]
            variant["perturbations"][str(sigma)] = {
                "decoded_delta_mse_mean": float(delta.mean()),
                "decoded_delta_mse_p95": float(np.quantile(delta, 0.95)),
                "native_error": {
                    key: float(np.mean([row[key] for row in total])) for key in total[0]
                },
            }
        result["variants"][name] = variant
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path)
    source.add_argument(
        "--frozen-input",
        type=Path,
        help="Existing results.json + predictions.npz directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry", type=int, nargs="+", default=[16, 32, 48, 64])
    parser.add_argument(
        "--bases",
        nargs="+",
        choices=("uniform", "fourier", "chebyshev"),
        default=["uniform", "fourier", "chebyshev"],
    )
    parser.add_argument("--noise-audit", action="store_true")
    parser.add_argument(
        "--chebyshev-grid", choices=("uniform", "chebyshev_lobatto"), default="uniform"
    )
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--windows-per-episode", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.frozen_input:
        manifest = args.frozen_input / "INPUT.json"
        arrays = args.frozen_input / "input-windows.npz"
        if not manifest.exists():
            manifest, arrays = (
                args.frozen_input / "results.json",
                args.frozen_input / "predictions.npz",
            )
        previous = json.loads(manifest.read_text())
        with np.load(arrays, allow_pickle=False) as data:
            windows = data["raw"]
        identities, provenance = previous["windows"], previous["provenance"]
        if digest_array(windows) != provenance["windows_sha256"] or len(windows) != len(
            identities
        ):
            raise ValueError("Frozen reconstruction window identity mismatch")
    else:
        windows, identities, provenance = sample_windows(
            args.dataset,
            episodes_per_task=args.episodes_per_task,
            windows_per_episode=args.windows_per_episode,
        )
    # Preserve the exact action-only sample before any candidate can fail, so
    # repairs need no repeated data traversal or changed random selection.
    (args.output / "INPUT.json").write_text(
        json.dumps({"windows": identities, "provenance": provenance}, indent=2) + "\n"
    )
    np.savez_compressed(args.output / "input-windows.npz", raw=windows)
    candidates, predictions, measurements = [], [], []
    audits = {}
    for geometry in args.geometry:
        for basis in args.bases:
            codec = LiberoArcGlobalCodec(
                basis=basis,
                geometry=geometry,
                max_translation=1.6,
                max_rotation_degrees=384,
                geometry_fit_grid=(
                    args.chebyshev_grid if basis == "chebyshev" else "uniform"
                ),
            )
            name = f"{basis}_g{geometry}_t31"
            tokens = np.stack([codec.encode(raw) for raw in windows])
            prediction = np.stack([codec.decode(token) for token in tokens])
            candidates.append(
                {
                    "id": name,
                    "basis": basis,
                    "geometry": geometry,
                    "clock": 31,
                    "scalars": codec.scalars,
                    "rows": codec.num_waypoints,
                    "geometry_fit_grid": codec.geometry_fit_grid,
                }
            )
            predictions.append(prediction)
            measurements.append(
                [errors(raw, decoded) for raw, decoded in zip(windows, prediction)]
            )
            if args.noise_audit:
                audits[name] = noise_audit(codec, tokens, windows, identities)
            print(
                name,
                "mean_action_mse",
                np.mean([m["action_mse"] for m in measurements[-1]]),
                flush=True,
            )
    summary = {}
    for split in ("train", "valid"):
        subset = [i for i, row in enumerate(identities) if row["split"] == split]
        summary[split] = {
            candidate["id"]: {
                key: {
                    "mean": float(np.mean(values)),
                    "p95": float(np.quantile(values, 0.95)),
                    "max": float(np.max(values)),
                }
                for key in measurements[index][0]
                for values in [[measurements[index][i][key] for i in subset]]
            }
            for index, candidate in enumerate(candidates)
        }
    result = {
        "kind": "offline_reconstruction_not_simulator_success",
        "provenance": provenance,
        "windows": identities,
        "candidates": candidates,
        "summary": summary,
        "noise_audit": audits,
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(
        args.output / "predictions.npz", raw=windows, predictions=np.stack(predictions)
    )
    plot(result, args.output)


def plot(result, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"uniform": "#475569", "fourier": "#2563eb", "chebyshev": "#d97706"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    metrics = [
        ("action_mse", "Time-indexed native action MSE"),
        ("geometric_xyz_rmse_m", "Translation shape RMSE (m)"),
        ("geometric_rotation_rmse_rad", "Rotation shape RMSE (rad)"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        for basis, color in colors.items():
            candidates = [c for c in result["candidates"] if c["basis"] == basis]
            ax.plot(
                [c["scalars"] for c in candidates],
                [
                    result["summary"]["valid"][c["id"]][metric]["mean"]
                    for c in candidates
                ],
                "o-",
                label=basis,
                color=color,
            )
        ax.set(xlabel="Total representation scalars", title=title, yscale="log")
        ax.grid(alpha=0.2)
    axes[0].legend()
    fig.suptitle(
        "LIBERO-10 held-out demonstrations · identical clocks and grip · not policy rollouts"
    )
    fig.savefig(output / "reconstruction.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
