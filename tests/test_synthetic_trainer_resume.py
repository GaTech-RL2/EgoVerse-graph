import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from scripts.train.train_synthetic_manifold import (
    _capture_rng_state,
    _restore_rng_state,
)


def _run(source: Path, config: Path, resume: Path | None = None) -> None:
    command = [
        sys.executable,
        str(source / "scripts/train/train_synthetic_manifold.py"),
        "--config",
        str(config),
    ]
    if resume is not None:
        command.extend(("--resume", str(resume)))
    subprocess.run(command, cwd=source, check=True)


def test_training_resume_preserves_checkpoints_and_appends_metrics(tmp_path):
    source = Path(__file__).parents[1]
    dataset = tmp_path / "dataset.npz"
    rng = np.random.default_rng(7)
    split = np.array([0] * 30 + [1] * 5 + [2] * 5, dtype=np.uint8)
    np.savez_compressed(
        dataset,
        source_latent=rng.normal(size=(40, 2)).astype(np.float32),
        target_3d=rng.normal(size=(40, 3)).astype(np.float32),
        split=split,
    )
    output = tmp_path / "run"
    config_path = tmp_path / "config.json"
    config = {
        "method": "unite",
        "seed": 42,
        "dataset": str(dataset),
        "source_key": "source_latent",
        "output_dir": str(output),
        "model": {
            "latent_dim": 2,
            "codec_width": 8,
            "codec_depth": 1,
            "field_width": 8,
            "field_depth": 1,
        },
        "flow_samples": 2,
        "reconstruction_noise_range": [0.5, 1.0],
        "learning_rate": 0.0003,
        "batch_size": 4,
        "max_steps": 2,
        "inference_steps": 2,
        "log_every": 1,
        "checkpoint_every": 1,
    }
    config_path.write_text(json.dumps(config))
    _run(source, config_path)
    checkpoint = next((output / "checkpoints").glob("*global-step-000002.pt"))
    legacy_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    legacy_state.pop("rng")
    torch.save(legacy_state, checkpoint)

    config["max_steps"] = 4
    config_path.write_text(json.dumps(config))
    _run(source, config_path, checkpoint)

    steps = sorted(
        int(path.stem.rsplit("-", 1)[-1])
        for path in (output / "checkpoints").glob("*.pt")
    )
    assert steps == [1, 2, 3, 4]
    metric_steps = [
        json.loads(row)["step"]
        for row in (output / "metrics.jsonl").read_text().splitlines()
    ]
    assert metric_steps == [1, 2, 3, 4]
    final = torch.load(
        next((output / "checkpoints").glob("*global-step-000004.pt")),
        map_location="cpu",
        weights_only=False,
    )
    assert final["step"] == 4
    assert final["resume"]["checkpoint"] == str(checkpoint)
    assert final["resume"]["rng_restored"] is False
    assert "rng" in final


def test_restore_rng_state_moves_saved_cuda_states_to_cpu(monkeypatch):
    generator = torch.Generator().manual_seed(9)
    state = _capture_rng_state(generator)

    class DeviceState:
        def __init__(self):
            self.cpu_called = False

        def cpu(self):
            self.cpu_called = True
            return torch.tensor([1, 2, 3], dtype=torch.uint8)

    device_state = DeviceState()
    state["cuda"] = [device_state]
    captured = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", captured.append)

    _restore_rng_state(state, generator)

    assert device_state.cpu_called
    assert captured[0][0].device.type == "cpu"
    assert captured[0][0].dtype == torch.uint8


def test_action_adapter_training_uses_50k_default_and_real_validation(tmp_path):
    source = Path(__file__).parents[1]
    dataset = tmp_path / "dataset.npz"
    rng = np.random.default_rng(17)
    split = np.array([0] * 30 + [1] * 5 + [2] * 5, dtype=np.uint8)
    np.savez_compressed(
        dataset,
        source_gaussian_latent=rng.normal(size=(40, 8)).astype(np.float32),
        target_3d=rng.normal(size=(40, 3)).astype(np.float32),
        split=split,
    )
    output = tmp_path / "action-adapter-run"
    config_path = tmp_path / "action-adapter.json"
    config = {
        "architecture": "action_adapter_flow",
        "adapter_objective": "none",
        "seed": 42,
        "dataset": str(dataset),
        "evaluation_dataset": str(dataset),
        "evaluation_particles": 5,
        "source_key": "source_gaussian_latent",
        "output_dir": str(output),
        "model": {
            "latent_dim": 8,
            "adapter_family": "fixed_affine",
            "field_width": 8,
            "field_depth": 1,
        },
        "flow_samples": 2,
        "lambda_reconstruction": 0.0,
        "lambda_scale": 0.0,
        "lambda_path": 0.0,
        "learning_rate": 0.0003,
        "batch_size": 4,
        "max_steps": 2,
        "inference_steps": 2,
        "diagnostic_noise_samples": 16,
        "angular_bins": 4,
        "log_every": 1,
    }
    config_path.write_text(json.dumps(config))
    _run(source, config_path)
    checkpoints = list((output / "checkpoints").glob("*.pt"))
    assert len(checkpoints) == 1
    assert checkpoints[0].name.endswith("global-step-000002.pt")
    summary = json.loads((output / "summary.json").read_text())
    for key in (
        "validation_generation_symmetric_nn_mse",
        "validation_path_consistency_mse",
        "validation_scale_loss",
        "validation_torus_surface_rmse",
        "validation_angular_histogram_l1",
        "validation_decoder_jacobian_singular_max",
    ):
        assert np.isfinite(summary[key])
