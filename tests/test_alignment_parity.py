"""Numerical parity with immutable source methods and configured graph training."""

from copy import deepcopy
from functools import partial

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from egomimic.models.cores.hpt_transformer import MultiheadAttention, SimpleTransformer
from egomimic.pipeline.core import Pipeline
from egomimic.pipeline.stages_alignment import (
    RepresentationAlignment,
    ScheduledLossReduction,
)
from egomimic.pipeline.stages_hpt import HPTTrunkStage
from tests.test_retained_recipe_steps import CONFIGS, small_cpu_model

pytest.importorskip("geomloss")
SoftDTWLossPyTorch = pytest.importorskip("tslearn.metrics").SoftDTWLossPyTorch
LegacyAlignment = pytest.importorskip(
    "tests.fixtures.legacy_alignment_reference"
).LegacyAlignment


@pytest.mark.parametrize("supervision", [None, "mse", "soft_dtw"])
def test_sinkhorn_loss_and_gradients_match_source(supervision):
    torch.manual_seed(121)
    left = torch.randn(3, 4, 8, requires_grad=True)
    right = torch.randn(3, 4, 8, requires_grad=True)
    actions = torch.randn(3, 5, 6)
    other_actions = actions.flip(0) + 0.01
    reference = LegacyAlignment()
    reference.device = torch.device("cpu")
    reference.ot_6dof = False
    reference.use_dtw = supervision == "soft_dtw"
    reference.dtw = SoftDTWLossPyTorch(gamma=0.1)
    expected, expected_distance = reference.compute_ot(
        left, right, actions, other_actions, supervision is not None, 0.5
    )
    expected_grads = torch.autograd.grad(expected, (left, right))
    stage = RepresentationAlignment(
        "left",
        "right",
        supervision=supervision,
        cost_layout="legacy_broadcast",
        left_actions="actions",
        right_actions="other_actions",
        left_action_indices=[0, 1, 2],
        right_action_indices=[0, 1, 2],
    )
    result = stage(
        {
            "left": left,
            "right": right,
            "actions": actions,
            "other_actions": other_actions,
        }
    )
    torch.testing.assert_close(result["alignment/loss"], expected)
    torch.testing.assert_close(result["log/feature_distance"], expected_distance)
    for actual, expected_gradient in zip(
        torch.autograd.grad(result["alignment/loss"], (left, right)), expected_grads
    ):
        torch.testing.assert_close(actual, expected_gradient)


def test_freeze_boundary_preserves_alignment_gradients():
    torch.manual_seed(42)
    trunk = SimpleTransformer(
        partial(MultiheadAttention, embed_dim=8, num_heads=2, batch_first=True),
        embed_dim=8,
        num_blocks=3,
    )
    stage = HPTTrunkStage(
        trunk,
        8,
        action_token_count=2,
        squeeze_action_token=False,
        representation_block=1,
        detach_condition_before_block=2,
    )
    result = stage({"hpt/tokens": torch.randn(2, 3, 8)})
    result["condition"].square().sum().backward(retain_graph=True)
    assert trunk.blocks[0].attn.in_proj_weight.grad is None
    assert trunk.blocks[1].attn.in_proj_weight.grad is None
    assert trunk.blocks[2].attn.in_proj_weight.grad.abs().sum() > 0
    stage.zero_grad()
    result["hpt/representation"].square().sum().backward()
    assert trunk.blocks[0].attn.in_proj_weight.grad.abs().sum() > 0
    assert trunk.blocks[1].attn.in_proj_weight.grad.abs().sum() > 0
    assert trunk.blocks[2].attn.in_proj_weight.grad is None
    # The resumed path is the source implementation, with unchanged weights.
    _, blocks = trunk(torch.randn(2, 5, 8))
    reference = LegacyAlignment()
    reference.trunk = {"trunk": trunk}
    reference.postprocess_tokens = lambda value: value
    torch.testing.assert_close(
        trunk.resume_from_block(blocks, 2), reference.resume_from_depth(blocks, 2)
    )
    assert "hpt/representation" not in stage.execute(
        {"hpt/tokens": torch.randn(2, 3, 8)}, mode="inference"
    )


def test_loss_schedule_and_resume_use_training_batches():
    def graph():
        return Pipeline(
            [
                ScheduledLossReduction(
                    {
                        "source_losses": {"start_batch": 2},
                        "alignment": {"weight": 0.7, "start_batch": 3},
                    },
                    denominator_key="source_count",
                )
            ]
        )

    reducer = graph()
    batch = {
        "source_losses": torch.tensor([2.0, 4.0]),
        "alignment": torch.tensor(10.0),
        "source_count": 2,
    }
    assert reducer(batch)["loss/total"] == 0
    assert reducer(batch)["loss/total"] == 3
    restored = graph()
    restored.load_state_dict(deepcopy(reducer.state_dict()), strict=True)
    assert restored(batch)["loss/total"] == pytest.approx(6.5)
    assert restored(batch)["log/training_batches"] == 4


def test_egobridge_two_source_optimizer_and_independent_feature_passes():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        config = compose(
            config_name="train_zarr_cartesian", overrides=["model=egobridge"]
        )
    cpu = small_cpu_model(config, "hpt")
    cpu.model.pipeline.stages[1].representation_block = 0
    cpu.model.pipeline.loss_pipeline.stages[0].supervision = "mse"
    model = instantiate(cpu.model.pipeline, device="cpu")
    calls = []
    hook = model.pipeline.stage_by_id("stems").register_forward_hook(
        lambda *args: calls.append(1)
    )
    batch = {}
    for name, identity, width in [("eva_bimanual", 6, 14), ("human_bimanual", 3, 12)]:
        batch[name] = {
            "embodiment": torch.tensor([identity, identity]),
            "actions_cartesian": torch.randn(2, 100, width),
            "observations.state.ee_pose": torch.randn(2, width),
            **{
                f"observations.images.{key}": torch.rand(2, 3, 32, 32)
                for key in ("front_img_1", "left_wrist_img", "right_wrist_img")
            },
        }
    optimizer = torch.optim.Adam(model.nets.parameters(), lr=1e-4)
    for _ in range(2):
        optimizer.zero_grad()
        predictions = model.forward_training(batch)
        losses = model.compute_losses(predictions, batch)
        expected = (
            losses["source_0_loss"]
            + losses["source_1_loss"]
            + 0.7 * losses["log_alignment_loss"]
        ) / 2
        torch.testing.assert_close(losses["loss"], expected)
        losses["loss"].backward()
        assert model.pipeline.stage_by_id("trunk").action_token.grad.abs().sum() > 0
        optimizer.step()
    hook.remove()
    assert (
        len(calls) == 8
    ), "Two sources, two steps, independent BC and OT feature passes"
    clone = instantiate(cpu.model.pipeline, device="cpu")
    clone.nets.load_state_dict(deepcopy(model.nets.state_dict()), strict=True)
    assert clone.nets["loss_pipeline"].stages[1].training_batches == 2
    clone.nets.eval()
    assert clone.forward_eval({"human_bimanual": batch["human_bimanual"]})[
        "human_bimanual"
    ]["pred_action"].shape == (2, 100, 12)


def test_alignment_rejects_unmatched_batches():
    stage = RepresentationAlignment("left", "right")
    with pytest.raises(ValueError, match="matching"):
        stage({"left": torch.zeros(2, 3), "right": torch.zeros(3, 3)})
