"""Compare likelihood mechanics against the retained source-bound wrapper."""

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.pl_utils.pl_model_action_flow_likelihood import (
    ActionFlowLikelihoodModelWrapper,
)
from egomimic.pl_utils.training_behavior_action_flow_likelihood import (
    ActionFlowLikelihoodTrainingBehavior,
)
from tests.test_action_flow_likelihood import batch, stages


def test_likelihood_recipe_changes_only_framework_selection():
    root = Path(__file__).resolve().parents[1]
    path = "egomimic/hydra_configs/model/bf/us_action_flow_bc_bridge_likelihood.yaml"
    candidate = yaml.safe_load((root / path).read_text())
    assert candidate.pop("_target_") == "egomimic.pl_utils.pl_model.ModelWrapper"
    assert candidate.pop("training_behavior") == {
        "_target_": "egomimic.pl_utils.training_behavior_action_flow_likelihood.ActionFlowLikelihoodTrainingBehavior"
    }
    # Canonical YAML payload (without the framework target), independently
    # extracted from exact training source7c253e5. No full-history checkout is
    # required in shallow CI, and every scientific field is included.
    digest = hashlib.sha256(
        json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert digest == "e55a71bec35d62e694119ea3c42fed1a8e7855f2d97fc3e7310c21a37831e033"


def test_port_preserves_historical_tensor_and_checkpoint_algorithms():
    root = Path(__file__).resolve().parents[1] / "egomimic/pl_utils"

    def methods(path, name):
        tree = ast.parse((root / path).read_text())
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == name
        )
        return {
            node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)
        }

    historical = methods(
        "pl_model_action_flow_likelihood.py", "ActionFlowLikelihoodModelWrapper"
    )
    candidate = methods(
        "training_behavior_action_flow_likelihood.py",
        "ActionFlowLikelihoodTrainingBehavior",
    )

    class ContextToHistorical(ast.NodeTransformer):
        def visit_Attribute(self, node):
            node = self.generic_visit(node)
            if (
                isinstance(node.value, ast.Attribute)
                and node.value.attr == "context"
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "self"
            ):
                node.value = ast.Name(id="self", ctx=ast.Load())
            return node

    for name in (
        "_log_telemetry",
        "_capture_gradient_routes",
        "_log_prediction_metrics",
        "validation_step",
        "on_save_checkpoint",
    ):
        port_name = (
            "log_prediction_metrics" if name == "_log_prediction_metrics" else name
        )
        port = ContextToHistorical().visit(copy.deepcopy(candidate[port_name]))
        port.name = name
        assert ast.dump(port, include_attributes=False) == ast.dump(
            historical[name], include_attributes=False
        ), name


@pytest.mark.parametrize("cadence", [0, 1])
def test_training_validation_checkpoint_and_rng_match_historical(monkeypatch, cadence):
    def observe(shared):
        torch.manual_seed(73)
        algo = PipelineAlgo(stages(dropout=0.3), device="cpu")
        wrapper = (
            ModelWrapper(
                pipeline=algo,
                enable_grad_norm=False,
                training_behavior=ActionFlowLikelihoodTrainingBehavior(
                    gradient_telemetry_cadence=cadence
                ),
            )
            if shared
            else ActionFlowLikelihoodModelWrapper(
                pipeline=algo,
                enable_grad_norm=False,
                gradient_telemetry_cadence=cadence,
            )
        )
        logged = {}
        monkeypatch.setattr(
            wrapper,
            "log",
            lambda name, value, **kw: logged.update(
                {name: (torch.as_tensor(value).detach().clone(), kw)}
            ),
        )
        optimizer = torch.optim.SGD(wrapper.parameters(), lr=0.001)
        losses, gradients = [], []
        for index in range(2):
            optimizer.zero_grad()
            loss = wrapper.training_step({"example": batch()}, index)
            losses.append(loss.detach().clone())
            loss.backward()
            gradients.append(
                [
                    None if p.grad is None else p.grad.detach().clone()
                    for p in wrapper.parameters()
                ]
            )
            wrapper.on_after_backward()
            wrapper.on_before_optimizer_step(optimizer)
            optimizer.step()
        wrapper.eval()
        before = torch.get_rng_state().clone()
        with torch.inference_mode():
            wrapper.validation_step({"example": batch(), "inactive": None}, 2, 1)
        # batch() constructs random data, so compare resulting RNG across sources.
        after = torch.get_rng_state().clone()
        checkpoint = {}
        wrapper.on_save_checkpoint(checkpoint)
        return (
            copy.deepcopy(wrapper.nets.state_dict()),
            losses,
            gradients,
            {k: v for k, v in logged.items() if not k.startswith("Timing/")},
            before,
            after,
            {k: v for k, v in checkpoint.items() if k.startswith("action_flow_")},
        )

    historical, candidate = observe(False), observe(True)
    assert historical[0].keys() == candidate[0].keys()
    for key in historical[0]:
        assert torch.equal(historical[0][key], candidate[0][key]), key
    for a, b in zip(historical[1], candidate[1]):
        assert torch.equal(a, b)
    for left, right in zip(historical[2], candidate[2]):
        for a, b in zip(left, right):
            assert (a is None) == (b is None)
            if a is not None:
                assert torch.equal(a, b)
    assert historical[3].keys() == candidate[3].keys()
    for key, (value, options) in historical[3].items():
        assert torch.equal(value, candidate[3][key][0]), key
        assert options == candidate[3][key][1], key
    assert torch.equal(historical[4], candidate[4])
    assert torch.equal(historical[5], candidate[5])
    assert historical[6] == candidate[6]
