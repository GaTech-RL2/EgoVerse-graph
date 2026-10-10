import pytest
import torch

from egomimic.pipeline.stages_matched_dp import (
    SharedNativeDPStage,
    shared_native_transformers,
)


def test_co_training_shares_attention_but_keeps_native_io_private():
    models = shared_native_transformers(16, 144, 1, 8, 0.0)
    robot, human = models.values()
    assert robot.layers is human.layers
    assert robot.pos_emb is human.pos_emb
    assert robot.proj_u is not human.proj_u
    assert robot.proj_d is not human.proj_d
    for width, model in ((14, robot), (138, human)):
        model.zero_grad(set_to_none=True)
        output = model(
            torch.randn(2, 100, width), torch.tensor([0, 99]), torch.randn(2, 16)
        )
        assert output.shape == (2, 100, width)
        output.square().mean().backward()
        gradient = robot.layers[0].mha.in_proj_weight.grad
        assert (
            gradient is not None
            and torch.isfinite(gradient).all()
            and gradient.abs().sum() > 0
        )


def test_native_diffusion_training_and_ddim_sampling_work_for_both_embodiments():
    stage = SharedNativeDPStage(
        condition_dim=16,
        inference_steps=2,
        denoiser="transformer",
        transformer_width=144,
        transformer_depth=1,
        transformer_heads=8,
        transformer_dropout=0.0,
    )
    optimizer = torch.optim.AdamW(stage.parameters(), lr=1e-4)
    for embodiment, width in ((7, 14), (3, 138)):
        stage.train()
        optimizer.zero_grad(set_to_none=True)
        output = stage.execute(
            {
                "embodiment": embodiment,
                "condition": torch.randn(2, 16),
                "target": torch.randn(2, 100, width),
            },
            mode="train",
        )
        loss = output["loss/diffusion_noise"]
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        stage.eval()
        with torch.no_grad():
            prediction = stage.execute(
                {"embodiment": embodiment, "condition": torch.randn(2, 16)},
                mode="inference",
            )["pred_action"]
        assert prediction.shape == (2, 100, width)
        assert torch.isfinite(prediction).all()


def test_rejects_wrong_native_input_shape():
    model = shared_native_transformers(16, 144, 1, 8, 0.0)["human_bimanual"]
    with pytest.raises(ValueError, match="native_action_dim"):
        model(torch.zeros(1, 100, 14), torch.tensor([1]), torch.zeros(1, 16))
