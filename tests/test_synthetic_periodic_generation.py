"""Periodic generation must observe training without changing its updates."""

import json
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.train import train_synthetic_manifold as trainer


def _assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_equal(a, b)
    else:
        assert left == right


def test_periodic_generation_preserves_optimizer_rng_and_logs_exact_cloud_once(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    data = tmp_path / "train.npz"
    evaluation = tmp_path / "evaluation.npz"
    rng = np.random.default_rng(41)
    for path in (data, evaluation):
        np.savez_compressed(
            path, source_3d=rng.normal(size=(12, 3)).astype(np.float32),
            target_3d=rng.normal(size=(12, 3)).astype(np.float32),
            split=np.array([0] * 8 + [1] * 3 + [2], dtype=np.uint8),
        )
    expected_source, expected_target = trainer.SyntheticTrajectoryEval.load_validation_data(
        evaluation, "source_3d", 3
    )
    original_evaluate = trainer.SyntheticTrajectoryEval.evaluate
    evaluation_calls = []

    def noisy_evaluate(model, source, target, *, steps):
        assert not model.training
        assert not torch.is_grad_enabled()
        assert not torch.is_inference_mode_enabled()
        torch.testing.assert_close(source, expected_source, rtol=0, atol=0)
        torch.testing.assert_close(target, expected_target, rtol=0, atol=0)
        assert steps == 2
        evaluation_calls.append(steps)
        # A future stochastic sampler must not silently change training draws.
        random.random()
        np.random.normal()
        torch.randn(11)
        return original_evaluate(model, source, target, steps=steps)

    monkeypatch.setattr(trainer.SyntheticTrajectoryEval, "evaluate", noisy_evaluate)
    logged_runs = []

    class FakeRun:
        def __init__(self):
            self.rows = []

        def log(self, row, *, step):
            self.rows.append((step, dict(row)))

        def finish(self):
            pass

    def init(**_kwargs):
        run = FakeRun()
        logged_runs.append(run)
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    checkpoints = []
    for every in (0, 1):
        config = {
            "architecture": "direct_flow", "seed": 42, "dataset": str(data),
            "evaluation_dataset": str(evaluation), "evaluation_particles": 3,
            "source_key": "source_3d",
            "output_dir": str(tmp_path / f"run-{every}"),
            "model": {"data_dim": 3, "field_width": 8, "field_depth": 1},
            "flow_samples": 2, "learning_rate": 0.0003, "batch_size": 4,
            "max_steps": 4, "inference_steps": 2, "log_every": 2,
            "checkpoint_every": 4, "generation_log_every": every,
            "wandb": {"mode": "disabled"},
        }
        config_path = tmp_path / f"config-{every}.json"
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr(sys, "argv", ["train", "--config", str(config_path)])
        before_calls = len(evaluation_calls)
        trainer.main()
        assert len(evaluation_calls) - before_calls == (1 if every == 0 else 4)
        output = tmp_path / f"run-{every}"
        paths = list((output / "checkpoints").glob("*.pt"))
        assert len(paths) == 1
        checkpoints.append(torch.load(paths[0], map_location="cpu", weights_only=False))
        rows = [json.loads(row) for row in (output / "metrics.jsonl").read_text().splitlines()]
        generation_rows = [row for row in rows if "generation_symmetric_nn_mse" in row]
        expected_steps = [] if every == 0 else [1, 2, 3]
        assert [row["step"] for row in generation_rows] == expected_steps
        for row in generation_rows:
            assert np.isfinite(row["generation_symmetric_nn_mse"])
            assert row["generation_eval_seconds"] > 0
        wb_rows = [row for _, row in logged_runs[-1].rows if "generation_symmetric_nn_mse" in row]
        assert wb_rows == generation_rows
        terminal_rows = [(step, row) for step, row in logged_runs[-1].rows
                         if "validation_generation_symmetric_nn_mse" in row]
        assert len(terminal_rows) == 1
        assert terminal_rows[0][0] == 4
    for key in ("model", "optimizer", "rng"):
        _assert_equal(checkpoints[0][key], checkpoints[1][key])


def test_periodic_generation_restores_mixed_modes_and_rng_on_error(monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout())
    model.train()
    model[1].eval()
    original_modes = [module.training for module in model.modules()]
    original_rng = trainer._capture_rng_state(torch.Generator())

    def fail(*_args, **_kwargs):
        random.random()
        np.random.normal()
        torch.randn(5)
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr(trainer.SyntheticTrajectoryEval, "evaluate", fail)
    with pytest.raises(RuntimeError, match="evaluation failed"):
        trainer._periodic_generation_metrics(model, torch.zeros(3, 3), torch.zeros(3, 3), steps=2)
    assert [module.training for module in model.modules()] == original_modes
    restored = trainer._capture_rng_state(torch.Generator())
    for key in ("python", "numpy", "torch", "cuda"):
        _assert_equal(original_rng[key], restored[key])
