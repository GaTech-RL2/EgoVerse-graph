"""Only successful final-step validation can suppress duplicate publication."""

import ast
from pathlib import Path

import pytest
import torch

from egomimic.eval.image_proprio_eval import ImageProprioEval
from tests.test_pixel_prediction_diagnostics import batch_fixture, fake_model


# Execute the maintained runner guard directly without unrelated optional
# runtime imports (the local test environment does not include tabulate).
def _runner_guard():
    path = Path(__file__).parents[1] / "egomimic/trainHydra.py"
    node = next(
        item
        for item in ast.parse(path.read_text()).body
        if isinstance(item, ast.FunctionDef) and item.name == "_needs_final_validation"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace["_needs_final_validation"]


needs_final_validation = _runner_guard()


def evaluator_fixture(tmp_path=None):
    batch, _ = batch_fixture()
    model, prediction, _ = fake_model(batch)
    diagnostics = (
        None
        if tmp_path is None
        else {
            "artifact_dir": str(tmp_path),
            "write_video": False,
            "log_wandb": False,
        }
    )
    evaluator = ImageProprioEval(objective="visual_only", pixel_diagnostics=diagnostics)
    evaluator.model = model
    evaluator.on_validation_start()
    return evaluator, batch, model, prediction


def finish(evaluator, batch):
    evaluator.on_validation_step(batch, 0)
    evaluator.log_epoch_metrics()
    evaluator.on_validation_end()


def test_completed_pass_suppresses_only_same_step_and_start_resets_it(tmp_path):
    evaluator, batch, model, _ = evaluator_fixture(tmp_path)
    assert evaluator.has_completed_validation(model.global_step) is False
    assert needs_final_validation(evaluator, model.global_step) is True
    finish(evaluator, batch)
    assert evaluator.has_completed_validation(model.global_step) is True
    assert needs_final_validation(evaluator, model.global_step) is False
    assert needs_final_validation(evaluator, model.global_step + 1) is True
    evaluator.on_validation_start()
    assert needs_final_validation(evaluator, model.global_step) is True
    with pytest.raises(FileExistsError):
        evaluator.on_validation_step(batch, 0)
    evaluator.log_epoch_metrics()
    evaluator.on_validation_end()
    assert needs_final_validation(evaluator, model.global_step) is True


@pytest.mark.parametrize(
    "case", ["empty", "zero_targets", "sanity", "unlogged", "changed_step"]
)
def test_incomplete_pass_keeps_final_validation(case):
    evaluator, batch, model, _ = evaluator_fixture()
    if case == "sanity":
        model.trainer.sanity_checking = True
        evaluator.on_validation_start()
    if case == "zero_targets":
        batch["robot"]["world"]["image"]["observed"].fill_(True)
    if case != "empty":
        evaluator.on_validation_step(batch, 0)
    if case != "unlogged":
        evaluator.log_epoch_metrics()
    if case == "changed_step":
        model.global_step += 1
    evaluator.on_validation_end()
    assert evaluator.has_completed_validation(model.global_step) is False
    assert needs_final_validation(evaluator, model.global_step) is True


def test_failed_artifact_publication_cannot_mark_complete(tmp_path, monkeypatch):
    evaluator, batch, model, _ = evaluator_fixture(tmp_path)
    evaluator.on_validation_step(batch, 0)
    evaluator.log_epoch_metrics()

    def fail(_):
        assert evaluator.has_completed_validation(model.global_step) is False
        raise RuntimeError("publication failed")

    monkeypatch.setattr(evaluator.pixel_diagnostics, "log_to_wandb", fail)
    with pytest.raises(RuntimeError, match="publication failed"):
        evaluator.on_validation_end()
    assert needs_final_validation(evaluator, model.global_step) is True


@pytest.mark.parametrize("case", ["prediction", "accumulator", "metric_publication"])
def test_nonfinite_or_failed_metric_pass_cannot_mark_complete(case):
    evaluator, batch, model, prediction = evaluator_fixture()
    if case == "prediction":
        prediction["image"].fill_(float("nan"))
        with pytest.raises(ValueError, match="Non-finite"):
            evaluator.on_validation_step(batch, 0)
    else:
        evaluator.on_validation_step(batch, 0)
        if case == "accumulator":
            key = next(iter(evaluator.totals))
            evaluator.totals[key] = (torch.tensor(float("inf")), torch.tensor(1.0))
        else:

            def fail(*args, **kwargs):
                raise RuntimeError("metric publication failed")

            model.log = fail
        with pytest.raises((ValueError, RuntimeError)):
            evaluator.log_epoch_metrics()
    evaluator.on_validation_end()
    assert needs_final_validation(evaluator, model.global_step) is True
