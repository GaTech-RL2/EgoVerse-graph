from pathlib import Path

import pytest
import torch
import torch.nn as nn

from egomimic.pipeline.core import Pipeline, sum_losses
from egomimic.pipeline.stages_action_flow import (
    ActionFlowObjectiveStage,
    ConditionalVelocityStage,
    ContentDecoderStage,
    ContentEncoderStage,
    LatentBridgeStage,
)
from egomimic.pipeline.stages_sampler import GaussianLatentNoise


class _LastDimLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, value):
        return self.linear(value)


class _TinyField(nn.Module):
    def __init__(self, latent_dim: int, condition_dim: int):
        super().__init__()
        self.state_weight = nn.Parameter(torch.tensor(0.25))
        self.time_weight = nn.Parameter(torch.tensor(0.5))
        self.condition_projection = nn.Linear(condition_dim, latent_dim)
        self.null_condition = nn.Parameter(torch.zeros(condition_dim))
        self.seen_masks = []

    def forward(self, value, time, condition, *, condition_drop_mask=None):
        if condition_drop_mask is None:
            raise AssertionError("the caller must supply the shared mask")
        self.seen_masks.append(condition_drop_mask.clone())
        effective_condition = torch.where(
            condition_drop_mask[:, None],
            self.null_condition[None].expand_as(condition),
            condition,
        )
        projected = self.condition_projection(effective_condition)
        return (
            self.state_weight * value
            + self.time_weight * time[:, None, None]
            + projected[:, None, :]
        )


class _ConstantField(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)
        self.seen_times = []
        self.seen_masks = []

    def forward(self, value, time, condition, *, condition_drop_mask=None):
        self.seen_times.append(time.clone())
        self.seen_masks.append(condition_drop_mask.clone())
        return torch.full_like(value, self.value)


def test_bridge_reuses_one_base_noise_and_drop_mask_across_independent_times():
    batch_size, count, horizon, latent_dim = 3, 5, 4, 2
    clean = torch.linspace(
        -1.0,
        1.0,
        batch_size * horizon * latent_dim,
    ).reshape(batch_size, horizon, latent_dim)
    clean.requires_grad_()
    noise = clean.new_tensor(
        [
            [[2.0, -2.0]] * horizon,
            [[3.0, -3.0]] * horizon,
            [[4.0, -4.0]] * horizon,
        ]
    )
    condition = torch.randn(batch_size, 7)
    stage = LatentBridgeStage(
        samples_per_content=count,
        condition_dropout_probability=0.5,
    )

    torch.manual_seed(114)
    output = stage(
        {
            "action_flow/clean_latent": clean,
            "sampler/noise": noise,
            "condition": condition,
        }
    )

    base_index = torch.arange(batch_size).repeat_interleave(count)
    assert torch.equal(output["action_flow/base_index"], base_index)
    assert torch.equal(output["action_flow/noise"], noise[base_index])
    assert torch.equal(output["action_flow/condition"], condition[base_index])
    assert torch.equal(
        output["action_flow/condition_drop_mask"],
        output["action_flow/base_condition_drop_mask"][base_index],
    )
    assert output["action_flow/condition_drop_mask"].dtype == torch.bool

    time = output["action_flow/time"]
    assert time.shape == (batch_size * count,)
    assert bool(((0.0 <= time) & (time < 1.0)).all())
    reshaped_time = time.reshape(batch_size, count)
    assert all(int(torch.unique(row).numel()) > 1 for row in reshaped_time)

    clean_many = clean[base_index]
    noise_many = noise[base_index]
    expected_state = (1.0 - time[:, None, None]) * clean_many + time[
        :, None, None
    ] * noise_many
    torch.testing.assert_close(output["action_flow/state"], expected_state)
    torch.testing.assert_close(
        output["action_flow/target_velocity"], noise_many - clean_many
    )

    output["action_flow/state"].sum().backward()
    assert clean.grad is not None
    assert bool(torch.isfinite(clean.grad).all())
    assert float(clean.grad.abs().sum()) > 0.0


def test_base_noise_and_bridge_times_are_resampled_online():
    sampler = GaussianLatentNoise(num_tokens=4, latent_dim=3)
    bridge = LatentBridgeStage(
        samples_per_content=3,
        condition_dropout_probability=0.0,
    )
    condition = torch.randn(2, 5)
    clean = torch.randn(2, 4, 3)

    torch.manual_seed(71)
    first_noise = sampler({"condition": condition})["sampler/noise"].clone()
    first = bridge(
        {
            "condition": condition,
            "sampler/noise": first_noise,
            "action_flow/clean_latent": clean,
        }
    )
    second_noise = sampler({"condition": condition})["sampler/noise"].clone()
    second = bridge(
        {
            "condition": condition,
            "sampler/noise": second_noise,
            "action_flow/clean_latent": clean,
        }
    )

    assert not torch.equal(first_noise, second_noise)
    assert not torch.equal(first["action_flow/time"], second["action_flow/time"])
    for output, base_noise in ((first, first_noise), (second, second_noise)):
        base_index = output["action_flow/base_index"]
        assert torch.equal(output["action_flow/noise"], base_noise[base_index])


def test_decoded_noise_moment_objective_updates_only_decoder_path():
    decoder_module = _LastDimLinear(3, 4)
    decoder = ContentDecoderStage(decoder_module, decode_noise=True)
    objective = ActionFlowObjectiveStage(
        flow_weight=0.0,
        reconstruction_weight=0.0,
        action_velocity_weight=0.0,
        moment_weight=1.0,
    )
    clean = torch.randn(2, 5, 3, requires_grad=True)
    state = torch.randn(4, 5, 3, requires_grad=True)
    residual = torch.randn(4, 5, 3, requires_grad=True)
    output = {
        "target": torch.randn(2, 5, 4),
        "sampler/noise": torch.randn(2, 5, 3),
        "action_flow/clean_latent": clean,
        "action_flow/state": state,
        "action_flow/velocity_residual": residual,
    }
    output = decoder(output)
    output = objective(output)
    output["loss/action_flow"].backward()

    assert clean.grad is not None and float(clean.grad.abs().sum()) == 0.0
    assert state.grad is not None and float(state.grad.abs().sum()) == 0.0
    assert residual.grad is not None and float(residual.grad.abs().sum()) == 0.0
    gradients = [parameter.grad for parameter in decoder_module.parameters()]
    assert any(
        gradient is not None
        and bool(torch.isfinite(gradient).all())
        and float(gradient.abs().sum()) > 0.0
        for gradient in gradients
    )


def test_training_objective_preserves_all_joint_gradient_routes():
    encoder = _LastDimLinear(4, 3)
    field = _TinyField(latent_dim=3, condition_dim=5)
    decoder = _LastDimLinear(3, 4)
    stages = (
        ContentEncoderStage(encoder),
        LatentBridgeStage(
            samples_per_content=4,
            condition_dropout_probability=0.0,
        ),
        ConditionalVelocityStage(field, num_inference_steps=3),
        ContentDecoderStage(decoder),
        ActionFlowObjectiveStage(
            flow_weight=1.0,
            reconstruction_weight=10.0,
            action_velocity_weight=1.0,
        ),
    )
    target = torch.randn(2, 6, 4)
    condition = torch.randn(2, 5, requires_grad=True)
    output = {
        "target": target,
        "condition": condition,
        "sampler/noise": torch.randn(2, 6, 3),
    }

    torch.manual_seed(9)
    for stage in stages:
        output = stage(output)

    component_keys = (
        "log/action_flow_fm",
        "log/action_flow_reconstruction",
        "log/action_flow_reconstruction_l1",
        "log/action_flow_action_velocity",
    )
    assert all(output[key].requires_grad for key in component_keys)
    assert torch.equal(sum_losses(output), output["loss/action_flow"])
    output["loss/action_flow"].backward()

    for module in (encoder, field, decoder):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert any(
            gradient is not None
            and bool(torch.isfinite(gradient).all())
            and float(gradient.abs().sum()) > 0.0
            for gradient in gradients
        )
    assert condition.grad is not None
    assert float(condition.grad.abs().sum()) > 0.0


def test_decoder_jvp_matches_explicit_linear_jacobian_and_is_differentiable():
    decoder = _LastDimLinear(2, 3)
    with torch.no_grad():
        decoder.linear.weight.copy_(
            torch.tensor([[1.0, 2.0], [-3.0, 4.0], [0.5, -0.25]])
        )
        decoder.linear.bias.copy_(torch.tensor([7.0, 8.0, 9.0]))
    stage = ContentDecoderStage(decoder)
    clean = torch.randn(2, 4, 2)
    state = torch.randn(6, 4, 2, requires_grad=True)
    residual = torch.randn(6, 4, 2, requires_grad=True)

    output = stage(
        {
            "action_flow/clean_latent": clean,
            "action_flow/state": state,
            "action_flow/velocity_residual": residual,
        }
    )
    expected = torch.einsum("...d,od->...o", residual, decoder.linear.weight)
    torch.testing.assert_close(
        output["action_flow/decoded_velocity_residual"], expected
    )

    output["action_flow/decoded_velocity_residual"].square().mean().backward()
    assert residual.grad is not None
    assert decoder.linear.weight.grad is not None
    assert float(decoder.linear.weight.grad.abs().sum()) > 0.0


def test_objective_averages_k_samples_and_applies_only_declared_weights():
    target = torch.zeros(2, 3, 4)
    reconstruction = torch.ones_like(target)
    residual = torch.full((28, 3, 8), 2.0)
    decoded_residual = torch.full((28, 3, 4), 3.0)
    values = {
        "target": target,
        "action_flow/reconstruction": reconstruction,
        "action_flow/velocity_residual": residual,
        "action_flow/decoded_velocity_residual": decoded_residual,
    }

    rec1 = ActionFlowObjectiveStage(reconstruction_weight=1.0)(dict(values))
    rec10 = ActionFlowObjectiveStage(reconstruction_weight=10.0)(dict(values))

    assert float(rec1["log/action_flow_fm"]) == pytest.approx(4.0)
    assert float(rec1["log/action_flow_reconstruction"]) == pytest.approx(1.0)
    assert float(rec1["log/action_flow_action_velocity"]) == pytest.approx(9.0)
    assert float(rec1["loss/action_flow"]) == pytest.approx(14.0)
    assert float(rec10["loss/action_flow"]) == pytest.approx(23.0)


def test_reverse_euler_is_unguided_and_uses_the_same_field_and_decoder():
    field = _ConstantField(2.0)
    velocity = ConditionalVelocityStage(field, num_inference_steps=4)
    decoder = nn.Identity()
    content = ContentDecoderStage(decoder)
    noise = torch.full((2, 3, 2), 3.0)
    condition = torch.randn(2, 5)

    output = velocity.execute(
        {"sampler/noise": noise, "condition": condition},
        mode="inference",
    )
    output = content.execute(output, mode="inference")

    torch.testing.assert_close(
        output["action_flow/generated_latent"], torch.ones_like(noise)
    )
    torch.testing.assert_close(output["pred_action"], torch.ones_like(noise))
    assert output["action_flow/trajectory"].shape == (5, 2, 3, 2)
    expected_times = (1.0, 0.75, 0.5, 0.25)
    assert len(field.seen_times) == len(expected_times)
    for seen, expected in zip(field.seen_times, expected_times):
        torch.testing.assert_close(seen, torch.full((2,), expected))
    assert all(not bool(mask.any()) for mask in field.seen_masks)
    assert velocity.field is field
    assert content.decoder is decoder


def test_mode_contracts_form_one_static_train_and_inference_graph():
    encoder = ContentEncoderStage(_LastDimLinear(4, 3))
    bridge = LatentBridgeStage(samples_per_content=2)
    velocity = ConditionalVelocityStage(_TinyField(3, 5), num_inference_steps=2)
    decoder = ContentDecoderStage(_LastDimLinear(3, 4))
    objective = ActionFlowObjectiveStage()
    pipeline = Pipeline([encoder, bridge, velocity, decoder, objective])

    training, training_excluded = pipeline.plan(
        ["target", "condition", "sampler/noise"], mode="train"
    )
    inference, inference_excluded = pipeline.plan(
        ["condition", "sampler/noise"], mode="inference"
    )

    assert training == [encoder, bridge, velocity, decoder, objective]
    assert training_excluded == []
    assert inference == [velocity, decoder]
    assert inference_excluded == [
        (encoder, ["<train-only>"]),
        (bridge, ["<train-only>"]),
        (objective, ["<train-only>"]),
    ]
    assert velocity.contract("inference") == (
        ("sampler/noise", "condition"),
        (
            "action_flow/generated_latent",
            "action_flow/trajectory",
            "log/action_flow_inference_steps",
        ),
    )
    assert decoder.contract("inference") == (
        ("action_flow/generated_latent",),
        ("pred_action",),
    )


def test_stage_source_keeps_the_pipeline_boundary_generic():
    source = (
        (Path(__file__).parents[1] / "egomimic/pipeline/stages_action_flow.py")
        .read_text()
        .lower()
    )
    for forbidden in (
        "embodiment",
        "domain",
        "ac_key",
        "action_key",
        "egomimic.models",
        "unite",
        "monotonic",
        "scale_loss",
        "noisy_reconstruction",
    ):
        assert forbidden not in source


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"samples_per_content": 0}, "positive"),
        ({"condition_dropout_probability": -0.1}, r"\[0, 1\]"),
        ({"condition_dropout_probability": 1.1}, r"\[0, 1\]"),
    ],
)
def test_bridge_rejects_invalid_sampling_contract(kwargs, match):
    with pytest.raises(ValueError, match=match):
        LatentBridgeStage(**kwargs)


def test_bridge_rejects_noise_that_does_not_match_the_clean_latent():
    stage = LatentBridgeStage(samples_per_content=2)
    with pytest.raises(ValueError, match="noise must match"):
        stage(
            {
                "action_flow/clean_latent": torch.randn(2, 4, 3),
                "sampler/noise": torch.randn(2, 4, 4),
                "condition": torch.randn(2, 5),
            }
        )
