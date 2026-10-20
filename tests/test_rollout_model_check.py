"""Checkpoint preflight must restore on CPU and return before hardware opens."""

import json
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from egomimic.robot import rollout


def check_config(tmp_path):
    config = {
        "robot": {"kind": "yam"},
        "policy": {"kind": "graph", "device": "cuda:0", "checkpoint": "bound.ckpt"},
    }
    path = tmp_path / "rollout.yaml"
    OmegaConf.save(OmegaConf.create(config), path)
    return path


def test_check_model_forces_cpu_loads_and_exits_before_hardware(
    tmp_path, monkeypatch, capsys
):
    path = check_config(tmp_path)
    monkeypatch.setattr("sys.argv", ["rollout", "--config", str(path), "--check-model"])
    monkeypatch.setattr(rollout, "validate_rollout_config", lambda config: None)
    closed = []

    def load(config):
        assert config["device"] == "cpu"
        return SimpleNamespace(
            graph=SimpleNamespace(nets=torch.nn.Linear(2, 3)),
            action_type="cartesian",
            close=lambda: closed.append(True),
        )

    monkeypatch.setattr(rollout, "load_policy", load)
    monkeypatch.setattr(
        rollout, "create_robot", lambda _: pytest.fail("Opened hardware")
    )
    rollout.main()
    report = json.loads(capsys.readouterr().out)
    assert report["parameters"] == 9
    assert report["device"] == "cpu"
    assert not report["hardware_opened"]
    assert closed == [True]
    assert OmegaConf.load(path).policy.device == "cuda:0"


def test_check_model_propagates_load_failure_without_devices(tmp_path, monkeypatch):
    path = check_config(tmp_path)
    monkeypatch.setattr("sys.argv", ["rollout", "--config", str(path), "--check-model"])
    monkeypatch.setattr(rollout, "validate_rollout_config", lambda config: None)

    def fail(config):
        raise ValueError("Checkpoint binding mismatch")

    monkeypatch.setattr(rollout, "load_policy", fail)
    monkeypatch.setattr(
        rollout, "create_robot", lambda _: pytest.fail("Opened hardware")
    )
    with pytest.raises(ValueError, match="binding mismatch"):
        rollout.main()
