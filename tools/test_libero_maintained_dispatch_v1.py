"""Pure metadata guards on the actual YAML's externally parsed JSON, no models."""

import copy
import json
import sys

from libero_maintained_dispatch_v1 import (
    METRICS,
    PROFILE,
    evaluator_overrides,
    require_finite_native_metrics,
    scheduled_only,
    validate_data_receipt,
    validate_resolved,
)
from typed_libero_profile_v3 import phase_bindings


def assign(config, path, value):
    cursor = config
    keys = path.split(".")
    for key in keys[:-1]:
        cursor = cursor[int(key)] if isinstance(cursor, list) else cursor[key]
    if isinstance(cursor, list):
        cursor[int(keys[-1])] = value
    else:
        cursor[keys[-1]] = value


def check(config):
    cfg = copy.deepcopy(config)
    cfg["callbacks"]["dit_half"] = {
        "_target_": "egomimic.utils.libero_dit_half.LiberoDiTHalf"
    }
    for phase in ("preflight", "smoke", "full"):
        variant = copy.deepcopy(cfg)
        for key, value in phase_bindings(phase).items():
            assign(variant, key, value)
        validate_resolved(variant, phase)
    full = copy.deepcopy(cfg)
    for key, value in phase_bindings("full").items():
        assign(full, key, value)
    mutations = [
        (
            "data.train_datasets.libero_panda._target_",
            "egomimic.rldb.zarr.libero_dataset.LiberoDataset._from_resolver",
        ),
        ("normalizer._target_", "egomimic.rldb.zarr.libero_dataset.LiberoNormalizer"),
        ("norm_stats.precomputed_norm_path", "/planar/norm_stats.json"),
        ("data.valid_datasets.libero_panda.mode", "train"),
        ("data.valid_datasets.libero_panda.valid_ratio", 0.05),
        ("model.pipeline.stages.6.field.backbone.num_heads", 12),
        ("callbacks.dit_half._target_", "invented"),
        ("callbacks.ema.validate_with_ema", False),
        ("evaluator.energy_sample_count", 8),
        ("data.train_dataloader_params.libero_panda.batch_size", 64),
    ]
    for key, value in mutations:
        variant = copy.deepcopy(full)
        assign(variant, key, value)
        try:
            validate_resolved(variant, "full")
        except ValueError:
            pass
        else:
            raise AssertionError(("bad dispatch accepted", key))
    evaluator_overrides(PROFILE)
    assert not any("validation_view" in item for item in evaluator_overrides(PROFILE))
    metrics = {
        "Train": {"normalized_reconst_mse": 1.0},
        "Valid": {key: 1.0 for key in METRICS},
    }
    require_finite_native_metrics(metrics)
    for key in METRICS:
        bad = copy.deepcopy(metrics)
        bad["Valid"][key] = float("nan")
        try:
            require_finite_native_metrics(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(("nonfinite accepted", key))
    receipt = {
        "suite": "libero10",
        "replay_path": "/fixture.zarr",
        "dataset_logical_sha256": "fixture",
        "episodes": 500,
        "valid_ratio": 0.01,
        "split_seed": 42,
        "frames": 1000,
        "action_shape": [1000, 7],
        "train_episode_indices": list(range(495)),
        "valid_episode_indices": list(range(495, 500)),
    }
    validate_data_receipt(
        receipt, replay_path="/fixture.zarr", expected_logical_sha="fixture"
    )
    for key, value in (
        ("valid_ratio", 0.05),
        ("valid_episode_indices", [0, 496, 497, 498, 499]),
        ("replay_path", "/wrong"),
        ("action_shape", [1000, 4]),
    ):
        bad = copy.deepcopy(receipt)
        bad[key] = value
        try:
            validate_data_receipt(
                bad, replay_path="/fixture.zarr", expected_logical_sha="fixture"
            )
        except ValueError:
            pass
        else:
            raise AssertionError(("bad receipt accepted", key))
    try:
        scheduled_only()
    except RuntimeError:
        pass
    else:
        raise AssertionError("login compute allowed")
    return {
        "status": "LIBERO_MAINTAINED_DISPATCH_METADATA_PASS_ONLY",
        "phases": 3,
        "semantic_negatives": len(mutations),
        "metric_negatives": len(METRICS),
        "receipt_negatives": 4,
        "actual_constructor": False,
        "gpu_ready": False,
    }


if __name__ == "__main__":
    print(json.dumps(check(json.load(sys.stdin)), indent=2))
