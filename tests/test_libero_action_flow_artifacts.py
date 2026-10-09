"""Tiny pure-metadata validation; no model or data construction."""

import pytest

from egomimic.benchmarks.libero.action_flow_artifacts import (
    REQUIRED,
    SEED_SHA,
    canonical_sha,
    validate_payload,
)


def fixture():
    identity = {
        "suite": "libero10",
        "source": "libero_panda",
        "action_dim": 7,
        "action_horizon": 16,
        "seed": 42,
        "sample_count": 32,
        "energy_seed_bank_sha256": SEED_SHA,
        "inference_method": "euler",
        "inference_steps": 50,
        "normalization_scope": "training_episodes_only",
        "effective_batch_size": 32,
        "homogeneous": "not_applicable_single_source",
        "source_commit": "a" * 40,
        "resolved_config_sha256": "b" * 64,
        "split_sha256": "c" * 64,
        "normalizer_state_sha256": "d" * 64,
        "dataset_logical_sha256": "e" * 64,
    }
    tensor_metadata = {
        name: {
            "shape": [32, 16, 7],
            "axes": ["batch", "horizon", "action"],
            "finite": True,
            "sha256": "f" * 64,
        }
        for name in (
            "normalized_prediction",
            "native_prediction",
            "normalized_target",
            "native_target",
        )
    }
    for name in ("normalized_samples", "native_samples"):
        tensor_metadata[name] = {
            "shape": [32, 32, 16, 7],
            "axes": ["sample", "batch", "horizon", "action"],
            "finite": True,
            "sha256": "f" * 64,
        }
    for name in ("diagnostic/latent/clean", "diagnostic/decoded/reconstruction"):
        tensor_metadata[name] = {"finite": True, "sha256": "f" * 64}
    return {
        "schema": "libero-native-action-flow-metrics/v2",
        "identity": identity,
        "identity_sha256": canonical_sha(identity),
        "source": "libero_panda",
        "global_step": 2,
        "batches": [
            {
                "batch_index": 0,
                "batch_size": 32,
                "normalized_target_sha256": "f" * 64,
                "metrics": {key: 1.0 for key in REQUIRED},
                # Pure metadata fixture: validates reference contracts only,
                # never proves payload bytes, actual episodes or a run PASS.
                "tensor_payload": {
                    "schema": "libero-native-tensors/v1",
                    "filename": "fixture.pt",
                    "sha256": "f" * 64,
                    "tensors": tensor_metadata,
                    "alignment": {"fixture": "explicit-synthetic-metadata"},
                    "shared_analysis": {
                        "sha256": "f" * 64,
                        "identity_sha256": "f" * 64,
                        "path": "fixture-analysis.pt",
                        "global_step": 2,
                    },
                },
            }
        ],
    }


def test_native_complete():
    assert validate_payload(fixture())


def test_legacy_metric_only_schema_does_not_certify_native_tensors():
    payload = fixture()
    payload["schema"] = "libero-native-action-flow-metrics/v1"
    with pytest.raises(ValueError, match="wrong native artifact schema"):
        validate_payload(payload)


def test_reference_rejects_wrong_sample_axes():
    payload = fixture()
    payload["batches"][0]["tensor_payload"]["tensors"]["native_samples"]["axes"] = [
        "batch",
        "sample",
        "horizon",
        "action",
    ]
    with pytest.raises(ValueError, match="K32 tensor contract"):
        validate_payload(payload)


@pytest.mark.parametrize("key", REQUIRED)
def test_reject_nonfinite(key):
    payload = fixture()
    payload["batches"][0]["metrics"][key] = float("nan")
    with pytest.raises(ValueError):
        validate_payload(payload)


@pytest.mark.parametrize(
    "key",
    (
        "source_commit",
        "resolved_config_sha256",
        "split_sha256",
        "normalizer_state_sha256",
        "dataset_logical_sha256",
        "energy_seed_bank_sha256",
        "normalization_scope",
        "inference_method",
        "inference_steps",
        "effective_batch_size",
    ),
)
def test_reject_identity(key):
    payload = fixture()
    payload["identity"][key] = "wrong"
    payload["identity_sha256"] = canonical_sha(payload["identity"])
    with pytest.raises(ValueError):
        validate_payload(payload)


def test_missing_first_batch():
    payload = fixture()
    payload["batches"][0]["batch_index"] = 1
    with pytest.raises(ValueError):
        validate_payload(payload)


def test_no_implicit_checkpoint_completion():
    payload = fixture()
    payload["global_step"] = 0
    with pytest.raises(ValueError):
        validate_payload(payload)
