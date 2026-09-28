import pytest
import torch
from omegaconf import OmegaConf

from egomimic.models.stems.byte_language import ByteLanguageStem, encode_bytes
from egomimic.pipeline.stages_masked_flow import (
    MaskedFlowNoisingStage,
    MaskedFlowVelocityLossStage,
)


def stem():
    torch.manual_seed(17)
    model = ByteLanguageStem(width=16, heads=2, ffn_dim=32, layers=2, dropout=0.0)
    model.init_cross_attn(
        OmegaConf.create(
            dict(
                crossattn_latent=3,
                modality_embed_dim=16,
                crossattn_heads=2,
                crossattn_dim_head=8,
                crossattn_modality_dropout=0.0,
            )
        )
    )
    return model.eval()


def test_language_padding_does_not_change_instruction_features():
    model = stem()
    single = model.compute_latent(encode_bytes(["Push blue left."]))
    mixed = model.compute_latent(
        encode_bytes(
            ["Push blue left.", "Push the block to the right of the reference marker."]
        )
    )
    torch.testing.assert_close(single[0], mixed[0], atol=2e-6, rtol=2e-5)
    changed = model.compute_latent(["Push blue right."])
    assert not torch.allclose(single, changed)
    mixed.sum().backward()
    assert model.embedding.weight.grad.abs().sum() > 0
    for layer in model.layers:
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in layer.parameters()
        )


def test_byte_language_refuses_truncation_and_invalid_masks():
    with pytest.raises(ValueError, match="exceeds"):
        encode_bytes(["a" * 255])
    model = stem()
    batch = encode_bytes(["left", "right"])
    batch["attention_mask"][:] = True
    with pytest.raises(ValueError, match="padding"):
        model.compute_latent(batch)


def test_masked_loss_uses_valid_scalar_denominator_and_zero_padded_gradient():
    prediction = torch.tensor([[[1.0, 2.0], [100.0, 100.0]]], requires_grad=True)
    batch = {
        "flow/predicted_velocity": prediction,
        "flow/velocity_target": torch.zeros_like(prediction),
        "action_valid": torch.tensor([[True, False]]),
    }
    loss = MaskedFlowVelocityLossStage()(batch)["loss/flow_velocity"]
    assert loss.item() == 2.5
    loss.backward()
    torch.testing.assert_close(
        prediction.grad, torch.tensor([[[1.0, 2.0], [0.0, 0.0]]])
    )


def test_draw_replay_ignores_padded_target_before_cross_timestep_attention():
    noising = MaskedFlowNoisingStage(
        action_horizon=3, action_dim=2, noise_key="noise", time_key="time"
    )
    batch = {
        "target": torch.zeros(1, 3, 2),
        "action_valid": torch.tensor([[True, True, False]]),
        "noise": torch.ones(1, 3, 2),
        "time": torch.tensor([0.25]),
    }
    out = noising({k: v.clone() for k, v in batch.items()})
    batch["target"][:, -1] = float("nan")
    other = noising(batch)
    for key in ("target", "flow/noisy_action", "flow/velocity_target"):
        torch.testing.assert_close(out[key], other[key])
    torch.testing.assert_close(out["flow/noisy_action"], torch.full((1, 3, 2), 0.25))


def test_empty_mask_and_mismatched_draws_fail():
    stage = MaskedFlowNoisingStage(
        action_horizon=2, action_dim=1, noise_key="noise", time_key="time"
    )
    batch = {
        "target": torch.zeros(1, 2, 1),
        "action_valid": torch.tensor([[False, False]]),
        "noise": torch.zeros(1, 2, 1),
        "time": torch.tensor([0.5]),
    }
    with pytest.raises(ValueError, match="valid timestep"):
        stage(batch)
    batch["action_valid"][:] = True
    batch["time"] = torch.zeros(1, 1)
    with pytest.raises(ValueError, match="batch size"):
        stage(batch)
