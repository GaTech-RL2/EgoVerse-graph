"""Regression coverage for checkpoint requests during startup and saving."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.train.train_synthetic_manifold import (
    _atomic_torch_save,
    _CheckpointRequests,
)


def test_request_received_during_save_stays_pending():
    requests = _CheckpointRequests()
    requests.request(0, None)
    snapshot = requests.received
    requests.request(0, None)
    requests.saved = snapshot
    assert requests.received != requests.saved


def test_failed_atomic_save_preserves_checkpoint_and_removes_temporary(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"step": 1}, checkpoint)
    original = checkpoint.read_bytes()

    def fail_save(state, temporary):
        Path(temporary).write_bytes(b"incomplete")
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="interrupted write"):
        _atomic_torch_save({"step": 2}, checkpoint)
    assert checkpoint.read_bytes() == original
    assert list(tmp_path.iterdir()) == [checkpoint]


def _assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for lhs, rhs in zip(left, right, strict=True):
            _assert_equal(lhs, rhs)
    else:
        assert left == right


@pytest.mark.skipif(not hasattr(signal, "SIGUSR2"), reason="requires POSIX signals")
def test_startup_signal_checkpoints_and_resumes_exactly(tmp_path):
    source = Path(__file__).resolve().parents[1]
    trainer = source / "scripts/train/train_synthetic_manifold.py"
    dataset = tmp_path / "dataset.npz"
    rng = np.random.default_rng(17)
    np.savez_compressed(
        dataset,
        source_gaussian_latent=rng.normal(size=(40, 8)).astype(np.float32),
        target_3d=rng.normal(size=(40, 3)).astype(np.float32),
        split=np.array([0] * 30 + [1] * 5 + [2] * 5, dtype=np.uint8),
    )
    config = {
        "architecture": "action_adapter_flow",
        "adapter_objective": "action_velocity",
        "seed": 42,
        "dataset": str(dataset),
        "source_key": "source_gaussian_latent",
        "output_dir": str(tmp_path / "interrupted"),
        "model": {
            "latent_dim": 8,
            "adapter_family": "nonlinear",
            "residual_width": 8,
            "residual_depth": 1,
            "field_width": 8,
            "field_depth": 1,
        },
        "flow_samples": 2,
        "lambda_reconstruction": 100.0,
        "learning_rate": 0.0003,
        "batch_size": 4,
        "max_steps": 4,
        "inference_steps": 2,
        "diagnostic_noise_samples": 16,
        "angular_bins": 4,
        "log_every": 1,
        "checkpoint_every": 50_000,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    # Inject a real OS signal precisely at the first heavyweight import. There
    # is no sleep-based startup race and no production-only test hook. Exit only
    # after the requested atomic checkpoint is published, simulating a restart.
    bootstrap = """
import builtins
import os
import runpy
import signal
import sys

original_import = builtins.__import__
sent = False
def signal_before_numpy(name, *args, **kwargs):
    global sent
    if name == 'numpy' and not sent:
        sent = True
        assert callable(signal.getsignal(signal.SIGUSR2)), 'handler installed too late'
        os.kill(os.getpid(), signal.SIGUSR2)
    return original_import(name, *args, **kwargs)
builtins.__import__ = signal_before_numpy
original_replace = os.replace
def interrupt_after_checkpoint(src, dst):
    original_replace(src, dst)
    if str(dst).endswith('.pt'):
        os._exit(75)
os.replace = interrupt_after_checkpoint
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}

    def run(command, expected=0):
        result = subprocess.run(
            command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
        )
        assert result.returncode == expected, result.stdout + result.stderr

    run(
        [sys.executable, "-c", bootstrap, str(trainer), "--config", str(config_path)],
        expected=75,
    )
    output = Path(config["output_dir"])
    checkpoints = list((output / "checkpoints").glob("*.pt"))
    assert len(checkpoints) == 1
    checkpoint = checkpoints[0]
    assert checkpoint.name.endswith("global-step-000001.pt")
    checkpoint_bytes = checkpoint.read_bytes()
    run([
        sys.executable, str(trainer), "--config", str(config_path),
        "--resume", str(checkpoint),
    ])
    assert checkpoint.read_bytes() == checkpoint_bytes
    final_path = next((output / "checkpoints").glob("*global-step-000004.pt"))
    resumed = torch.load(final_path, map_location="cpu", weights_only=False)
    assert resumed["resume"]["rng_restored"] is True
    assert [json.loads(row)["step"] for row in
            (output / "metrics.jsonl").read_text().splitlines()] == [1, 2, 3, 4]
    summary = json.loads((output / "summary.json").read_text())
    assert np.isfinite(summary["validation_generation_symmetric_nn_mse"])
    assert np.isfinite(summary["validation_action_velocity_mse"])

    config["output_dir"] = str(tmp_path / "uninterrupted")
    config_path.write_text(json.dumps(config))
    run([sys.executable, str(trainer), "--config", str(config_path)])
    control_path = next((tmp_path / "uninterrupted/checkpoints").glob("*.pt"))
    control = torch.load(control_path, map_location="cpu", weights_only=False)
    for key in ("model", "optimizer", "rng", "step"):
        _assert_equal(resumed[key], control[key])
    control_summary = json.loads(
        (tmp_path / "uninterrupted/summary.json").read_text()
    )
    assert summary == control_summary
