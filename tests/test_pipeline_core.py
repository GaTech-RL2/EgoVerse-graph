import pytest
import torch

from egomimic.pipeline.core import Pipeline, Stage, sum_losses


class _FeatureStage(Stage):
    reads = ["obs/*"]
    writes = ["feature/*"]

    def forward(self, batch):
        batch["feature/x"] = batch["obs/x"] + 1
        return batch


class _LossStage(Stage):
    train_only = True
    reads = ["feature/x"]
    writes = ["loss/value"]

    def forward(self, batch):
        batch["loss/value"] = batch["feature/x"].square().mean()
        return batch


class _ModeStage(Stage):
    reads = ["target"]
    writes = ["loss/mode"]
    reads_by_mode = {"rollout": ["condition"]}
    writes_by_mode = {"rollout": ["pred_action"]}


def test_pipeline_registers_and_executes_stages():
    pipeline = Pipeline([_FeatureStage(), _LossStage()])
    source = torch.tensor([2.0], requires_grad=True)

    output = pipeline({"obs/x": source})
    total = sum_losses(output)
    total.backward()

    assert total.item() == 9.0
    assert source.grad.item() == 6.0


def test_pipeline_plan_resolves_wildcards_and_train_only_stages():
    feature = _FeatureStage()
    loss = _LossStage()
    pipeline = Pipeline([feature, loss])

    runnable, excluded = pipeline.plan(["obs/x"], mode="train")
    assert runnable == [feature, loss]
    assert excluded == []

    runnable, excluded = pipeline.plan(["obs/x"], mode="rollout")
    assert runnable == [feature]
    assert excluded == [(loss, ["<train-only>"])]


def test_stage_mode_contract_and_explain_are_exact():
    stage = _ModeStage()
    pipeline = Pipeline([stage])

    assert stage.contract("train") == (("target",), ("loss/mode",))
    assert stage.contract("rollout") == (("condition",), ("pred_action",))
    assert "EXCLUDED: missing target" in pipeline.explain([], mode="train")
    assert "pred_action" in pipeline.explain(["condition"], mode="rollout")

    with pytest.raises(ValueError, match="train\\|rollout"):
        stage.contract("validation")


def test_sum_losses_rejects_graph_without_an_objective():
    with pytest.raises(RuntimeError, match=r"no loss/\* keys"):
        sum_losses({"log/value": torch.tensor(1.0)})
