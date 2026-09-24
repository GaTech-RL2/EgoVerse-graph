"""Explicit module initialization never partial-loads or guesses a namespace."""

import hashlib

import pytest
import torch
from torch import nn

from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pipeline.core import Stage
from egomimic.pipeline.initialization import configure_trainability, initialize_weights


class Projection(Stage):
    reads = ("x",)
    writes = ("pred_action",)

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 2)

    def forward(self, batch):
        batch["pred_action"] = self.projection(batch["x"])
        return batch


def spec(tmp_path, value):
    path = tmp_path / "weights.pt"
    torch.save(value, path)
    return {
        "source": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "stage_id": "head",
        "module_path": "projection",
    }


def test_hash_verified_initialization_and_explicit_freeze(tmp_path):
    source = nn.Linear(3, 2)
    graph = PipelineAlgo(
        [Projection()],
        device="cpu",
        stage_ids={"head": 0},
        initialization=[spec(tmp_path, source.state_dict())],
        trainability=[{"stage_id": "head", "trainable": False}],
    )
    torch.testing.assert_close(
        graph.pipeline.stages[0].projection.weight, source.weight
    )
    assert not any(p.requires_grad for p in graph.nets.parameters())
    configure_trainability(graph.pipeline, [{"stage_id": "head", "trainable": True}])
    assert all(p.requires_grad for p in graph.nets.parameters())


@pytest.mark.parametrize("failure", ["hash", "missing", "dtype", "shape", "nan"])
def test_rejected_initialization_leaves_all_parameters_unchanged(tmp_path, failure):
    graph = PipelineAlgo([Projection()], device="cpu", stage_ids={"head": 0})
    before = {k: v.clone() for k, v in graph.nets.state_dict().items()}
    state = nn.Linear(3, 2).state_dict()
    if failure == "missing":
        state.pop("bias")
    if failure == "dtype":
        state["bias"] = state["bias"].double()
    if failure == "shape":
        state["bias"] = torch.zeros(3)
    if failure == "nan":
        state["bias"][0] = float("nan")
    setting = spec(tmp_path, state)
    if failure == "hash":
        setting["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        initialize_weights(graph.pipeline, [setting])
    for k, v in graph.nets.state_dict().items():
        torch.testing.assert_close(v, before[k])


def test_trainability_schedule_replays_on_resume_and_preserves_optimizer_membership():
    from egomimic.pl_utils.trainability_behavior import TrainabilitySchedule

    graph = PipelineAlgo([Projection()], device="cpu", stage_ids={"head": 0})

    class Context:
        model = graph
        global_step = 0

        def _default_training_step(self, batch, batch_idx):
            return self.global_step

    context = Context()
    behavior = TrainabilitySchedule(
        [
            {"at_step": 0, "targets": [{"stage_id": "head", "trainable": False}]},
            {"at_step": 2, "targets": [{"stage_id": "head", "trainable": True}]},
        ]
    )
    behavior.bind(context)
    optimizer = torch.optim.Adam(graph.nets.parameters())
    assert not any(p.requires_grad for p in graph.nets.parameters())
    context.global_step = 2
    assert behavior.training_step({}, 0) == 2
    assert all(p.requires_grad for p in graph.nets.parameters())
    assert {id(p) for p in optimizer.param_groups[0]["params"]} == {
        id(p) for p in graph.nets.parameters()
    }
    behavior.on_load_checkpoint({"global_step": 1})
    assert not any(p.requires_grad for p in graph.nets.parameters())
    behavior.on_load_checkpoint({"global_step": 2})
    assert all(p.requires_grad for p in graph.nets.parameters())
