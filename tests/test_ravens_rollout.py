"""Rollout ABI transport and embodiment routing; no simulator needed."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

path = Path(__file__).parents[1] / "egomimic/benchmarks/ravens/rollout.py"
spec = importlib.util.spec_from_file_location("ravens_rollout", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


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
