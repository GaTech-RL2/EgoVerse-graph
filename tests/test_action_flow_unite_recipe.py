from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

from egomimic.models.action_flow_unite import (
    UniteActionFlowContentEncoder,
    UniteActionFlowVelocityField,
)
from egomimic.models.unite_dit import UniteDiTBackbone
from egomimic.pipeline.stages_action_flow import (
    ConditionalVelocityStage,
    ContentDecoderStage,
    LatentBridgeStage,
)


def _backbone(*, horizon: int = 2) -> UniteDiTBackbone:
    return UniteDiTBackbone(
        input_dim=4,
        output_dim=4,
        horizon=horizon,
        condition_dim=8,
        max_condition_tokens=1,
        max_content_tokens=4,
        hidden_dim=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        qk_norm=True,
        use_rope=True,
        use_rmsnorm=True,
        use_swiglu=True,
        block_norm=True,
        time_fourier_dim=8,
        context_adaln_summary=False,
        in_context_start=1,
        in_context_len=4,
        trainable_position_embeddings=False,
        trainable_register_position_embeddings=True,
        trainable_content_position_embeddings=False,
        trainable_condition_position_embeddings=False,
        trainable_in_context_position_embeddings=True,
        gradient_checkpointing=False,
    )


def test_unite_action_flow_adapters_preserve_compact_register_contract():
    encoder = UniteActionFlowContentEncoder(
        backbone=_backbone(),
        action_dim=3,
        action_horizon=4,
        latent_dim=4,
        num_latent_tokens=2,
        condition_dim=8,
    )
    field = UniteActionFlowVelocityField(
        backbone=_backbone(),
        input_dim=4,
        output_dim=4,
        horizon=2,
        condition_dim=8,
        condition_dropout_probability=0.1,
    )
    latent = encoder(torch.randn(3, 4, 3))
    velocity = field(
        latent,
        torch.rand(3),
        torch.randn(3, 8),
        condition_drop_mask=torch.tensor([False, True, False]),
    )
    assert latent.shape == velocity.shape == (3, 2, 4)
    assert encoder.backbone is not field.backbone
    assert encoder.blocks is encoder.backbone.blocks
    assert field.blocks is field.backbone.blocks


def test_unite_diagnostic_activations_align_registers_after_context_insertion():
    encoder = UniteActionFlowContentEncoder(
        backbone=_backbone(),
        action_dim=3,
        action_horizon=4,
        latent_dim=4,
        num_latent_tokens=2,
        condition_dim=8,
    )
    before = torch.randn(3, 6, 32)
    after = torch.randn(3, 10, 32)
    assert encoder.canonicalize_diagnostic_block_output(before, 0).shape == (3, 2, 32)
    torch.testing.assert_close(
        encoder.canonicalize_diagnostic_block_output(after, 1),
        after[:, 4:6],
    )


def test_unite_bridge_maps_shifted_clean_fraction_to_action_flow_time():
    torch.manual_seed(7)
    stage = LatentBridgeStage(
        samples_per_content=128,
        condition_dropout_probability=0.1,
        time_sampling="unite_lognormal_shifted",
        lognorm_mu=0.0,
        lognorm_sigma=1.0,
        timestep_shift_alpha=0.5,
        independent_noise_per_sample=True,
        independent_condition_dropout_per_sample=True,
    )
    clean = torch.zeros(2, 2, 4)
    batch = stage(
        {
            "action_flow/clean_latent": clean,
            "sampler/noise": torch.zeros_like(clean),
            "condition": torch.randn(2, 8),
        }
    )
    time = batch["action_flow/time"]
    noise = batch["action_flow/noise"]
    assert time.shape == (256,)
    assert bool(((time > 0.0) & (time < 1.0)).all())
    assert not torch.equal(noise[0], noise[1])
    assert batch["action_flow/condition_drop_mask"].shape == (256,)


class _ConstantField(torch.nn.Module):
    def forward(self, value, time, condition, condition_drop_mask=None):
        return torch.ones_like(value)


def test_dopri5_integrates_from_gaussian_t1_to_clean_t0(monkeypatch):
    captured = {}

    def fake_odeint(function, initial, times, *, method, atol, rtol):
        captured["times"] = times
        derivative = function(times[0], initial)
        endpoint = initial + (times[-1] - times[0]) * derivative
        return torch.stack((initial, endpoint))

    monkeypatch.setitem(sys.modules, "torchdiffeq", SimpleNamespace(odeint=fake_odeint))
    stage = ConditionalVelocityStage(
        field=_ConstantField(),
        num_inference_steps=50,
        inference_method="dopri5",
        timestep_shift_alpha=0.5,
    )
    result = stage.execute(
        {
            "sampler/noise": torch.ones(2, 2, 4),
            "condition": torch.randn(2, 8),
        },
        mode="inference",
    )
    torch.testing.assert_close(captured["times"][[0, -1]], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(
        result["action_flow/generated_latent"], torch.zeros(2, 2, 4)
    )


def test_reconstruction_noising_start_one_is_exactly_clean():
    decoder = torch.nn.Identity()
    stage = ContentDecoderStage(
        decoder=decoder,
        reconstruction_noising_start=1.0,
        reconstruction_noising_probability=1.0,
    )
    clean = torch.randn(2, 2, 4)
    result = stage(
        {
            "action_flow/clean_latent": clean,
            "action_flow/state": clean,
            "action_flow/velocity_residual": torch.zeros_like(clean),
        }
    )
    torch.testing.assert_close(result["action_flow/reconstruction"], clean)
