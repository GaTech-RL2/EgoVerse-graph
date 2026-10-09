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
    return {
        "schema": "libero-native-action-flow-metrics/v1",
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
            }
        ],
    }


def test_native_complete():
    assert validate_payload(fixture())


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
