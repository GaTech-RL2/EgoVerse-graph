"""Rollout ABI transport and embodiment routing; no simulator needed."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

path = Path(__file__).parents[1] / "egomimic/benchmarks/ravens/rollout.py"
spec = importlib.util.spec_from_file_location("ravens_rollout", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def training_scene_request(tmp_path):
    manifest = {
        "complete": True,
        "episodes": [
            {"seed": i * 2, "sha256": "a" * 64, "frames": 710, "reward": 1.0}
            for i in range(500)
        ],
    }
    raw = json.dumps(manifest).encode()
    dataset_sha = hashlib.sha256(raw).hexdigest()
    (tmp_path / "dataset.json").write_bytes(raw)
    split = {
        "train": [f"episode_{i:06d}" for i in range(450)],
        "valid": [f"episode_{i:06d}" for i in range(450, 500)],
        "seed": 42,
        "dataset_manifest_sha256": dataset_sha,
    }
    raw_split = json.dumps(split).encode()
    (tmp_path / "split.json").write_bytes(raw_split)
    return {
        "evaluation_split": "train",
        "seeds": [0],
        "collection_seeds": list(range(0, 1000, 2)),
        "dataset_manifest": str(tmp_path / "dataset.json"),
        "dataset_manifest_sha256": dataset_sha,
        "split_manifest": str(tmp_path / "split.json"),
        "split_manifest_sha256": hashlib.sha256(raw_split).hexdigest(),
    }


def test_training_scene_requires_actual_train_membership(tmp_path):
    request = training_scene_request(tmp_path)
    selected = module.validate_scene_selection(request)
    assert selected[0]["episode_id"] == "episode_000000"
    assert selected[0]["evaluation_split"] == "train"
    request["seeds"] = [900]
    with pytest.raises(ValueError, match="validation, not training"):
        module.validate_scene_selection(request)
    request["seeds"] = [20000]
    with pytest.raises(ValueError, match="not recorded"):
        module.validate_scene_selection(request)


def test_unseen_default_keeps_collection_exclusion(tmp_path):
    request = training_scene_request(tmp_path)
    del request["evaluation_split"]
    with pytest.raises(ValueError, match="overlap"):
        module.validate_scene_selection(request)
    request["seeds"] = [20000]
    assert module.validate_scene_selection(request)[20000] == {
        "evaluation_split": "unseen"
    }


def test_training_scene_rejects_changed_split_and_mapping(tmp_path):
    request = training_scene_request(tmp_path)
    request["collection_seeds"] = list(reversed(request["collection_seeds"]))
    with pytest.raises(ValueError, match="mapping mismatch"):
        module.validate_scene_selection(request)
    request = training_scene_request(tmp_path)
    Path(request["split_manifest"]).write_text("{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        module.validate_scene_selection(request)


def test_simulator_import_does_not_import_torch():
    # This file itself is importable in the simulation-only Python3.10 ABI.
    assert "torch" not in module.__dict__


def test_transport_roundtrip():
    values = {
        "camera": np.arange(36, dtype=np.uint8).reshape(3, 4, 3),
        "state": np.array([1, 2, 3], np.float32),
    }
    decoded = module.unpack(module.pack(values))
    for key in values:
        assert values[key].dtype == decoded[key].dtype
        np.testing.assert_array_equal(values[key], decoded[key])


def test_transport_rejects_wrong_shape_and_dtype():
    payload = module.pack({"state": np.zeros(3, np.float32)})
    payload["state"]["shape"] = [4]
    with pytest.raises(ValueError, match="shape/byte"):
        module.unpack(payload)
    with pytest.raises(ValueError, match="only uint8"):
        module.pack({"state": np.zeros(3, np.float64)})


def test_policy_routing_is_configurable():
    import torch

    from egomimic.benchmarks.libero.rollout import GraphPolicy
    from egomimic.rldb.zarr.libero_dataset import OBS_KEYS

    class Norm:
        def normalize(self, batch, embodiment):
            self.embodiment = embodiment
            return batch

        def unnormalize(self, batch, embodiment):
            assert embodiment == self.embodiment
            return batch

    class Algo:
        nets = torch.nn.Identity()

        def process_batch_for_training(self, batch):
            assert list(batch) == ["ravens_ur5"]
            return batch

        def forward_eval(self, batch):
            return {"ravens_ur5": {"pred_action": torch.zeros(1, 32, 7)}}

    policy = GraphPolicy(Algo(), Norm(), embodiment="ravens_ur5")
    obs = {
        key: np.zeros((128, 128, 3), np.uint8)
        if key.endswith("rgb")
        else np.zeros(1, np.float32)
        for key in OBS_KEYS
    }
    policy.reset(obs)
    assert policy.predict().shape == (16, 7)


def test_protocol_from_actual_af_model_only_checkpoint():
    from omegaconf import OmegaConf

    saved = OmegaConf.create({"model": {"action_horizon": 32, "action_dim": 7}})
    trained = OmegaConf.create(
        {
            "model": {"action_horizon": 32, "action_dim": 7},
            "benchmark": {
                "suite": "ravens-continuous-v1",
                "horizon": 32,
                "n_obs_steps": 2,
                "n_action_steps": 16,
            },
        }
    )
    assert module.resolve_protocol(saved, trained)["horizon"] == 32
    trained.model.action_horizon = 100
    with pytest.raises(ValueError, match="differs from checkpoint"):
        module.resolve_protocol(saved, trained)


def test_protocol_rejects_wrong_trained_history():
    from omegaconf import OmegaConf

    saved = OmegaConf.create({"model": {"action_dim": 7}})
    trained = OmegaConf.create(
        {
            "model": {"action_dim": 7},
            "benchmark": {
                "suite": "ravens-continuous-v1",
                "horizon": 32,
                "n_obs_steps": 1,
                "n_action_steps": 16,
            },
        }
    )
    with pytest.raises(ValueError, match="trained action/history"):
        module.resolve_protocol(saved, trained)
