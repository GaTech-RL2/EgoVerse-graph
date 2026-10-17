import pytest
import torch
from torch.func import jvp

from egomimic.models.temporal_action_codec import (
    TemporalActionDecoder,
    TemporalActionEncoder,
)


@pytest.mark.parametrize("action_dim", [14, 138])
def test_native_horizons_and_no_reconstruction_bypass(action_dim):
    torch.manual_seed(42)
    encoder = TemporalActionEncoder(action_dim, 100, 16, 64, [8, 16, 32])
    decoder = TemporalActionDecoder(action_dim, 100, 16, 64, [8, 16, 32])
    encoder.eval()
    decoder.eval()
    actions = torch.randn(2, 100, action_dim)
    latent = encoder(actions)
    assert latent.shape == (2, 64, 16)
    assert decoder(latent).shape == actions.shape
    assert torch.equal(latent, encoder(actions))
    assert not torch.allclose(decoder(latent), decoder(torch.zeros_like(latent)))
    loss = (decoder(latent) - actions).square().mean()
    loss.backward()
    assert encoder.backbone.down_blocks[0][0].layers[0].block[0].weight.grad.norm() > 0
    assert decoder.backbone.output.weight.grad.norm() > 0
    with pytest.raises(ValueError, match="Expected"):
        decoder(latent[:, :-1])


def test_decoder_jvp_matches_finite_difference_and_trains_parameters():
    torch.manual_seed(42)
    decoder = TemporalActionDecoder(14, 100, 16, 64, [8, 16, 32]).double().eval()
    z = torch.randn(1, 64, 16, dtype=torch.float64)
    direction = torch.randn_like(z)
    _, tangent = jvp(decoder, (z,), (direction,))
    eps = 1e-5
    numerical = (decoder(z + eps * direction) - decoder(z - eps * direction)) / (
        2 * eps
    )
    torch.testing.assert_close(tangent, numerical, rtol=1e-4, atol=1e-5)
    tangent.square().mean().backward()
    assert decoder.backbone.output.weight.grad.norm() > 0


def test_checkpointed_joint_action_velocity_path_reaches_encoder_and_field():
    torch.manual_seed(42)
    encoder = TemporalActionEncoder(14, 100, 16, 64, [8, 16, 32]).train()
    decoder = TemporalActionDecoder(14, 100, 16, 64, [8, 16, 32]).train()
    actions = torch.randn(2, 100, 14)
    z = encoder(actions)
    field = torch.nn.Linear(16, 16)
    residual = field(z) - (torch.randn_like(z) - z)
    # Match ContentDecoderStage's AD boundary: checkpointing off inside JVP.
    decoder.gradient_checkpointing = False
    _, tangent = jvp(decoder, (z,), (residual,))
    decoder.gradient_checkpointing = True
    loss = tangent.square().mean() + (decoder(z) - actions).square().mean()
    loss.backward()
    assert field.weight.grad.norm() > 0
    assert encoder.backbone.output.weight.grad.norm() > 0
    assert decoder.backbone.output.weight.grad.norm() > 0


def test_diagnostic_hooks_are_real_equal_shape_activations():
    encoder = TemporalActionEncoder(14, 100, 16, 64, [8, 16, 32]).eval()
    values = []
    handles = [
        block.register_forward_hook(
            lambda module, args, output, i=i: values.append(
                encoder.canonicalize_diagnostic_block_output(output, i)
            )
        )
        for i, block in enumerate(encoder.blocks)
    ]
    encoder(torch.randn(2, 100, 14))
    for handle in handles:
        handle.remove()
    assert len(values) == 2
    assert values[0].shape == values[1].shape == (2, 64, 8)


def test_config_replacement_preserves_embodiments_and_latent_boundaries():
    from omegaconf import OmegaConf

    from egomimic.pipeline.stages_speed import replace_action_codec_configs

    spec = {
        "_target_": "old",
        "action_dim": 14,
        "action_horizon": 100,
        "latent_dim": 16,
        "num_latent_tokens": 64,
    }
    stages = [
        {"_target_": "test.ContentEncoderStage", "encoders": {"yam": dict(spec)}},
        {"_target_": "test.ContentDecoderStage", "decoders": {"yam": dict(spec)}},
    ]
    replacement = OmegaConf.create(
        {key: {"yam": {**spec, "_target_": "new"}} for key in ("encoders", "decoders")}
    )
    replace_action_codec_configs(stages, replacement)
    assert stages[0]["encoders"]["yam"]["_target_"] == "new"
    replacement.encoders.yam.latent_dim = 8
    with pytest.raises(ValueError, match="changes yam.latent_dim"):
        replace_action_codec_configs(stages, replacement)
    replacement.encoders.yam.latent_dim = 16
    del replacement.decoders.yam
    with pytest.raises(ValueError, match="embodiment mismatch"):
        replace_action_codec_configs(stages, replacement)
