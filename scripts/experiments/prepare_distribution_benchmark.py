#!/usr/bin/env python3
"""Prepare paired direct-CFM versus Action-Flow distribution benchmarks."""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from egomimic.synthetic.action_adapter_flow import SyntheticActionAdapterFlow
from egomimic.synthetic.shared_latent_flow import SyntheticDirectFlow


DISTRIBUTIONS = (
    "duffing",
    "henon",
    "benford",
    "logistic",
    "cauchy",
    "lorenz",
)
MODEL_SEEDS = (42, 43, 44)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split(count: int, seed: int) -> np.ndarray:
    order = np.random.default_rng(seed).permutation(count)
    split = np.full(count, 2, dtype=np.uint8)
    train_end = int(0.90 * count)
    val_end = train_end + int(0.05 * count)
    split[order[:train_end]] = 0
    split[order[train_end:val_end]] = 1
    return split


def _rk4_step(state, time, dt, derivative):
    k1 = derivative(state, time)
    k2 = derivative(state + 0.5 * dt * k1, time + 0.5 * dt)
    k3 = derivative(state + 0.5 * dt * k2, time + 0.5 * dt)
    k4 = derivative(state + dt * k3, time + dt)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def _trajectory(count: int, state, derivative, *, dt: float, burn: int, stride: int):
    values = []
    time = 0.0
    state = np.asarray(state, dtype=np.float64)
    for index in range(burn + count * stride):
        state = _rk4_step(state, time, dt, derivative)
        time += dt
        if index >= burn and (index - burn) % stride == 0:
            values.append(state.copy())
    return np.asarray(values, dtype=np.float64)


def _duffing(count: int, seed: int) -> np.ndarray:
    del seed
    omega = 1.2

    def derivative(state, time):
        x, velocity = state
        acceleration = -0.2 * velocity + x - x**3 + 0.30 * math.cos(omega * time)
        return np.array((velocity, acceleration), dtype=np.float64)

    xy = _trajectory(count, (0.1, 0.0), derivative, dt=0.02, burn=12_000, stride=8)
    sample_times = (np.arange(count) * 8 + 12_000) * 0.02
    return np.column_stack((xy, np.sin(omega * sample_times)))


def _henon(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x, y = 0.1 + 1e-3 * rng.normal(), 0.0
    values = []
    for index in range(2_000 + count):
        x, y = 1.0 - 1.4 * x * x + y, 0.3 * x
        if index >= 2_000:
            values.append((x, y))
    return np.asarray(values, dtype=np.float64)


def _benford(count: int, seed: int) -> np.ndarray:
    # A continuous significand distribution: S=10^U has Benford first digits.
    uniform = np.random.default_rng(seed).uniform(0.0, 1.0, size=count)
    return np.power(10.0, uniform)[:, None]


def _logistic(count: int, seed: int) -> np.ndarray:
    x = 0.173 + 1e-5 * np.random.default_rng(seed).normal()
    values = []
    history = []
    for index in range(2_000 + count + 2):
        x = 4.0 * x * (1.0 - x)
        if index >= 2_000:
            history.append(x)
    for index in range(count):
        values.append(history[index : index + 3])
    return np.asarray(values, dtype=np.float64)


def _cauchy(count: int, seed: int) -> np.ndarray:
    raw = np.random.default_rng(seed).standard_cauchy((count, 2))
    raw = np.clip(raw, -25.0, 25.0)
    raw[:, 1] = 0.65 * raw[:, 0] + math.sqrt(1.0 - 0.65**2) * raw[:, 1]
    return raw


def _lorenz(count: int, seed: int) -> np.ndarray:
    initial = np.array((1.0, 1.0, 1.0), dtype=np.float64)
    initial += 1e-4 * np.random.default_rng(seed).normal(size=3)

    def derivative(state, _time):
        x, y, z = state
        return np.array((10.0 * (y - x), x * (28.0 - z) - y, x * y - 8.0 * z / 3.0))

    return _trajectory(count, initial, derivative, dt=0.005, burn=10_000, stride=10)


def _native(distribution: str, count: int, seed: int) -> np.ndarray:
    generators = {
        "duffing": _duffing,
        "henon": _henon,
        "benford": _benford,
        "logistic": _logistic,
        "cauchy": _cauchy,
        "lorenz": _lorenz,
    }
    return generators[distribution](count, seed)


def _high_features(native: np.ndarray, *, seed: int, output_dim: int = 32):
    if native.shape[1] >= output_dim:
        raise ValueError("native dimension must be below high-dimensional width")
    rng = np.random.default_rng(seed)
    remaining = output_dim - native.shape[1]
    weights = rng.normal(size=(native.shape[1], remaining)) / math.sqrt(native.shape[1])
    bias = rng.uniform(-math.pi, math.pi, size=remaining)
    nonlinear = np.sin(native @ weights + bias)
    return np.concatenate((native, nonlinear), axis=1)


def _standardize_pair(train_raw: np.ndarray, eval_raw: np.ndarray, split: np.ndarray):
    selected = train_raw[split == 0]
    mean = selected.mean(axis=0)
    scale = selected.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (train_raw - mean) / scale, (eval_raw - mean) / scale, mean, scale


def _write_dataset_pair(root: Path, distribution: str, regime: str):
    data_seed = 20_260_911
    train_split = _split(4_096, data_seed + 1)
    eval_split = _split(40_960, data_seed + 2)
    train_native = _native(distribution, 4_096, data_seed + 3)
    eval_native = _native(distribution, 40_960, data_seed + 4)
    if regime.startswith("high"):
        output_dim = int(regime.removeprefix("high"))
        if output_dim < 4:
            raise ValueError(f"invalid high-dimensional regime: {regime}")
        train_raw = _high_features(
            train_native, seed=data_seed + 5, output_dim=output_dim
        )
        eval_raw = _high_features(
            eval_native, seed=data_seed + 5, output_dim=output_dim
        )
    elif regime == "low":
        train_raw, eval_raw = train_native, eval_native
    else:
        raise ValueError(f"unknown dimension regime: {regime}")
    train_target, eval_target, mean, scale = _standardize_pair(
        train_raw, eval_raw, train_split
    )
    action_dim = int(train_target.shape[1])
    latent_dim = max(8, 2 * action_dim)
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for label, target, split, seed in (
        ("train", train_target, train_split, data_seed + 10),
        ("eval", eval_target, eval_split, data_seed + 20),
    ):
        rng = np.random.default_rng(seed)
        path = data_dir / f"{distribution}-{regime}-{label}.npz"
        np.savez_compressed(
            path,
            target=target.astype(np.float32),
            source_direct=rng.normal(size=target.shape).astype(np.float32),
            source_latent=rng.normal(size=(len(target), latent_dim)).astype(np.float32),
            split=split,
            normalization_mean=mean.astype(np.float64),
            normalization_scale=scale.astype(np.float64),
            native_dimension=np.asarray(train_native.shape[1], dtype=np.int64),
        )
        paths[label] = path
    return paths, action_dim, latent_dim


def _parameter_count(module: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def _matched_models(action_dim: int, latent_dim: int):
    action_model = {
        "action_dim": action_dim,
        "latent_dim": latent_dim,
        "adapter_family": "nonlinear",
        "residual_width": 32,
        "residual_depth": 2,
        "field_width": 128,
        "field_depth": 4,
    }
    action_parameters = _parameter_count(SyntheticActionAdapterFlow(**action_model))
    candidates = []
    for width in range(8, 513):
        config = {"data_dim": action_dim, "field_width": width, "field_depth": 4}
        count = _parameter_count(SyntheticDirectFlow(**config))
        candidates.append((abs(count - action_parameters), width, count, config))
    _, _, direct_parameters, direct_model = min(candidates)
    return action_model, action_parameters, direct_model, direct_parameters


def _config(
    *,
    root: Path,
    distribution: str,
    regime: str,
    seed: int,
    method: str,
    train_path: Path,
    eval_path: Path,
    action_dim: int,
    latent_dim: int,
    source_commit: str,
):
    action_model, action_parameters, direct_model, direct_parameters = _matched_models(
        action_dim, latent_dim
    )
    run_id = f"{distribution}-{regime}-{method}-s{seed}-200k-v1"
    common = {
        "distribution_name": distribution,
        "dimension_regime": regime,
        "source_commit": source_commit,
        "seed": seed,
        "dataset": str(train_path),
        "evaluation_dataset": str(eval_path),
        "evaluation_particles": 2_048,
        "target_key": "target",
        "output_dir": str(root / "runs" / run_id),
        "flow_samples": 14,
        "learning_rate": 3e-4,
        "batch_size": 512,
        "max_steps": 200_000,
        "inference_steps": 32,
        "log_every": 500,
        "checkpoint_every": 50_000,
        "diagnostic_noise_samples": 4_096,
        "wandb": {
            "entity": "rl2-group",
            "project": "synthetic-distribution-fairness",
            "id": run_id,
            "name": run_id,
            "group": f"{distribution}-{regime}",
            "resume": "never",
            "dir": str(root / "wandb"),
        },
        "fairness": {
            "paired_dataset": True,
            "paired_model_seed": seed,
            "action_flow_parameters": action_parameters,
            "direct_flow_parameters": direct_parameters,
            "parameter_ratio_direct_over_action": direct_parameters / action_parameters,
        },
    }
    if method == "direct":
        common.update(
            architecture="direct_flow",
            source_key="source_direct",
            model=direct_model,
        )
    elif method == "action_flow":
        common.update(
            architecture="action_adapter_flow",
            source_key="source_latent",
            model=action_model,
            adapter_objective="action_velocity",
            lambda_reconstruction=1.0,
            lambda_scale=0.0,
            lambda_action_velocity=1.0,
            clean_gradient_mode="all_stopgrad",
        )
    else:
        raise ValueError(f"unknown method: {method}")
    return run_id, common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--dimension-regimes",
        nargs="+",
        choices=("low", "high32", "high100"),
        default=("low", "high32"),
    )
    args = parser.parse_args()
    if len(args.source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.source_commit
    ):
        raise ValueError("source-commit must be a full lowercase Git SHA")
    root = args.root.resolve()
    config_dir = root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for distribution in DISTRIBUTIONS:
        for regime in args.dimension_regimes:
            paths, action_dim, latent_dim = _write_dataset_pair(
                root, distribution, regime
            )
            for seed in MODEL_SEEDS:
                for method in ("direct", "action_flow"):
                    run_id, config = _config(
                        root=root,
                        distribution=distribution,
                        regime=regime,
                        seed=seed,
                        method=method,
                        train_path=paths["train"],
                        eval_path=paths["eval"],
                        action_dim=action_dim,
                        latent_dim=latent_dim,
                        source_commit=args.source_commit,
                    )
                    path = config_dir / f"{run_id}.json"
                    path.write_text(json.dumps(config, indent=2) + "\n")
                    records.append(
                        {
                            "run_id": run_id,
                            "config": str(path),
                            "distribution": distribution,
                            "dimension_regime": regime,
                            "seed": seed,
                            "method": method,
                            "action_dim": action_dim,
                            "latent_dim": latent_dim
                            if method == "action_flow"
                            else action_dim,
                            "parameters": (
                                config["fairness"]["action_flow_parameters"]
                                if method == "action_flow"
                                else config["fairness"]["direct_flow_parameters"]
                            ),
                        }
                    )
    datasets = sorted((root / "data").glob("*.npz"))
    manifest = {
        "schema_version": 1,
        "source_commit": args.source_commit,
        "base_commit": "3a813037eb16eaed1fc95c97364bab5ea167e7e8",
        "definitions": {
            "low": "native 1D, 2D, or 3D system state",
            "high32": "same intrinsic samples in deterministic injective nonlinear 32D embedding",
            "high100": "same intrinsic samples in deterministic injective nonlinear 100D embedding",
            "benford": "continuous significand S=10^U, U uniform on [0,1]",
            "cauchy": "correlated Cauchy truncated to [-25,25] before train-only standardization",
        },
        "datasets": [{"path": str(path), "sha256": _sha256(path)} for path in datasets],
        "runs": records,
    }
    (root / "BENCHMARK_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"runs": len(records), "datasets": len(datasets)}, sort_keys=True))


if __name__ == "__main__":
    main()
