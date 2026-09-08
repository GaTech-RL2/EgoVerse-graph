from __future__ import annotations

import hashlib
import json
from pathlib import Path

import hydra
import pytest
import torch
from hydra import compose, initialize_config_dir
from torch import nn

from egomimic.models.action_flow_codec import ContextFreeSequenceDecoder
from egomimic.models.action_flow_likelihood import (
    GaussianBridgeSchedule,
    TimeDependentSequenceMean,
)
from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pipeline.core import Pipeline
from egomimic.pipeline.stages_action_flow_likelihood import (
    ConditionalReverseMeanStage,
    GaussianBridgeNoisingStage,
    GaussianBridgeObjectiveStage,
    LikelihoodDecoderStage,
    LikelihoodReferenceStage,
)
from egomimic.pl_utils.pl_model_action_flow_likelihood import (
    ActionFlowLikelihoodModelWrapper,
)


class TinyField(nn.Module):
    def __init__(self, latent_dim=3, condition_dim=2):
        super().__init__()
        self.linear = nn.Linear(latent_dim + condition_dim + 1, latent_dim)
        self.calls = []

    def forward(self, state, time, condition, *, condition_drop_mask):
        self.calls.append(time.detach().clone())
        condition = torch.where(condition_drop_mask[:, None], 0, condition)
        features = torch.cat(
            (
                state,
                time[:, None, None].expand(-1, state.shape[1], 1),
                condition[:, None, :].expand(-1, state.shape[1], -1),
            ),
            -1,
        )
        return self.linear(features)


def stages(num_levels=4, samples=3, dropout=0.0):
    return [
        LikelihoodReferenceStage(
            TimeDependentSequenceMean(
                input_dim=2,
                latent_dim=3,
                horizon=2,
                hidden_dim=8,
                depth=1,
                num_heads=2,
                feedforward_dim=16,
            ),
            num_levels=num_levels,
            interior_samples_per_content=samples,
        ),
        GaussianBridgeNoisingStage(
            num_levels=num_levels, condition_dropout_probability=dropout
        ),
        ConditionalReverseMeanStage(TinyField(), num_levels=num_levels),
        LikelihoodDecoderStage(
            ContextFreeSequenceDecoder(
                latent_dim=3,
                output_dim=2,
                horizon=2,
                hidden_dim=8,
                depth=1,
                num_heads=2,
                feedforward_dim=16,
            )
        ),
        GaussianBridgeObjectiveStage(num_levels=num_levels),
    ]


def batch():
    return {
        "target": torch.randn(2, 2, 2),
        "condition": torch.randn(2, 2, requires_grad=True),
        "sampler/noise": torch.randn(2, 2, 3),
    }


def test_terminal_mean_is_exact_zero_and_codec_is_context_free():
    mean = TimeDependentSequenceMean()
    action = torch.randn(2, 16, 4)
    assert torch.count_nonzero(mean(action, torch.ones(2))) == 0
    assert mean(action, torch.full((2,), 1 / 32)).shape == (2, 16, 8)
    with pytest.raises(TypeError):
        mean(action, torch.ones(2), condition=torch.ones(2, 67))
    assert sum(p.numel() for p in mean.parameters()) < 15000


def test_reference_noising_exact_targets_independent_samples_and_shared_drop_mask():
    torch.manual_seed(7)
    modules = stages(samples=14, dropout=0.3)
    result = modules[1](modules[0](batch()))
    p = "likelihood/"
    levels, noise = result[p + "levels"], result[p + "interior_noise"]
    schedule = modules[1].schedule
    assert levels.shape == (28,) and bool(((levels >= 2) & (levels <= 4)).all())
    assert not torch.equal(noise[0], noise[1])
    expected_state = (
        result[p + "current_mean"] + schedule.sigmas[levels - 1, None, None] * noise
    )
    expected_target = (
        result[p + "previous_mean"]
        + 0.95 * schedule.sigmas[levels - 2, None, None] * noise
    )
    torch.testing.assert_close(result[p + "state"][:28], expected_state)
    torch.testing.assert_close(result[p + "posterior_target"], expected_target)
    torch.testing.assert_close(
        result[p + "posterior_variance"],
        (1 - 0.95**2) * schedule.sigmas[levels - 2].square(),
    )
    masks = result[p + "condition_drop_mask"]
    assert torch.equal(masks[:28].view(2, 14), masks[28:, None].expand(2, 14))
    assert result[p + "posterior_target"].requires_grad


def test_likelihood_reduction_sums_full_chunk_not_toy_coefficient():
    stage = GaussianBridgeObjectiveStage(num_levels=32, tau=0.02)
    values = {
        "likelihood/interior_prediction": torch.ones(28, 16, 8),
        "likelihood/posterior_target": torch.zeros(28, 16, 8),
        "likelihood/posterior_variance": torch.ones(28),
        "likelihood/boundary_prediction": torch.ones(2, 16, 4),
        "target": torch.zeros(2, 16, 4),
    }
    result = stage(values)
    torch.testing.assert_close(
        result["log/ActionFlow/InteriorBridgeNLL"], torch.tensor(31 * 128 / 2)
    )
    torch.testing.assert_close(
        result["log/ActionFlow/BoundaryNLL"], torch.tensor(80000.0)
    )
    torch.testing.assert_close(result["log/MSE"], torch.tensor(1.0))
    assert result["loss/likelihood"] == result["log/ActionFlow/TotalLoss"]
    assert not any("recon" in k.lower() or "fm" in k.lower() for k in result)


def test_both_likelihood_terms_update_private_means_and_shared_conditioned_field():
    torch.manual_seed(5)
    modules = stages()
    pipeline = Pipeline(modules)
    value = batch()
    result = pipeline(value)
    mean = modules[0].mean_encoder.network.output_projection.weight
    field = modules[2].field.linear.weight
    decoder = modules[3].decoder.output_projection.weight
    targets = (mean, field, decoder, value["condition"])
    interior = torch.autograd.grad(
        result["log/ActionFlow/InteriorBridgeNLL"],
        targets,
        retain_graph=True,
        allow_unused=True,
    )
    boundary = torch.autograd.grad(
        result["log/ActionFlow/BoundaryNLL"],
        targets,
        retain_graph=True,
        allow_unused=True,
    )
    assert interior[2] is None
    assert all(
        g is not None and torch.count_nonzero(g) > 0
        for i, g in enumerate(interior)
        if i != 2
    )
    assert all(g is not None and torch.count_nonzero(g) > 0 for g in boundary)
    assert len(modules[2].field.calls) == 1
    assert modules[2].field.calls[0].shape == (2 * (3 + 1),)
    optimizer = torch.optim.AdamW(pipeline.parameters(), lr=1e-4)
    optimizer.zero_grad()
    result["loss/likelihood"].backward()
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in pipeline.parameters())


def test_inference_uses_all_discrete_calls_innovations_and_output_noise(monkeypatch):
    modules = stages(num_levels=32)
    pipeline = Pipeline(modules).eval()
    shapes = []

    def noise_like(value, **kwargs):
        shapes.append(tuple(value.shape))
        return torch.ones_like(value, **kwargs)

    monkeypatch.setattr(torch, "randn_like", noise_like)
    value = batch()
    result = pipeline.execute(
        {"condition": value["condition"], "sampler/noise": value["sampler/noise"]},
        mode="inference",
    )
    calls = modules[2].field.calls
    assert len(calls) == 32
    torch.testing.assert_close(
        torch.stack(calls)[:, 0], torch.arange(32, 0, -1).float() / 32
    )
    assert shapes == [(2, 2, 3)] * 31 + [(2, 2, 2)]
    assert result["likelihood/trajectory"].shape == (33, 2, 2, 3)
    torch.testing.assert_close(
        result["pred_action"], result["likelihood/action_mean"] + 0.02
    )
    assert result["log/ActionFlow/SamplerCalls"] == 32
    assert result["log/ActionFlow/LatentInnovations"] == 31
    assert "loss/likelihood" not in result and "likelihood/boundary_mean" not in result


def test_seeded_actual_sampler_replays_including_output_noise():
    pipeline = Pipeline(stages()).eval()
    value = batch()
    torch.manual_seed(42)
    first = pipeline.execute(value, mode="inference")["pred_action"]
    torch.manual_seed(42)
    second = pipeline.execute(value, mode="inference")["pred_action"]
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.manual_seed(43)
    assert not torch.equal(
        first, pipeline.execute(value, mode="inference")["pred_action"]
    )


def test_wrapper_uses_likelihood_labels_and_records_observed_gradient_routes(
    monkeypatch,
):
    torch.manual_seed(3)
    algo = PipelineAlgo(stages(), device="cpu")
    model = ActionFlowLikelihoodModelWrapper(
        pipeline=algo, enable_grad_norm=False, gradient_telemetry_cadence=1
    )
    recorded = {}
    monkeypatch.setattr(
        model, "log", lambda name, value, **kwargs: recorded.__setitem__(name, value)
    )
    loss = model.training_step({"example": batch()}, 0)
    assert torch.isfinite(loss) and loss.requires_grad
    assert {
        "Train/MSE",
        "Train/ActionFlow/InteriorBridgeNLL",
        "Train/ActionFlow/BoundaryNLL",
        "Train/ActionFlow/TotalLoss",
    } <= set(recorded)
    assert not any(
        "FlowMatchingLoss" in name or "ReconstructionLoss" in name for name in recorded
    )
    checkpoint = {}
    model.on_save_checkpoint(checkpoint)
    manifest = checkpoint["action_flow_gradient_route_manifest"]
    assert set(manifest["routes"]) == {"InteriorBridgeNLL", "BoundaryNLL"}
    interior = [x["name"] for x in manifest["routes"]["InteriorBridgeNLL"]]
    boundary = [x["name"] for x in manifest["routes"]["BoundaryNLL"]]
    assert any("mean_encoder" in name for name in interior)
    assert any("field" in name for name in interior)
    assert not any("decoder" in name for name in interior)
    assert any("decoder" in name for name in boundary)
    core = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    assert (
        hashlib.sha256(
            json.dumps(
                core, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode()
        ).hexdigest()
        == manifest["manifest_sha256"]
    )
    assert checkpoint["action_flow_likelihood_contract"]["latent_fm_objective"] is False


def test_two_update_smoke_captures_gradient_routes_on_second_update(monkeypatch):
    model = ActionFlowLikelihoodModelWrapper(
        pipeline=PipelineAlgo(stages(), device="cpu"),
        enable_grad_norm=False,
        gradient_telemetry_cadence=2,
    )
    recorded = {}
    monkeypatch.setattr(
        model, "log", lambda name, value, **kwargs: recorded.__setitem__(name, value)
    )
    current_step = [0]
    monkeypatch.setattr(
        ActionFlowLikelihoodModelWrapper,
        "global_step",
        property(lambda self: current_step[0]),
    )
    model.training_step({"example": batch()}, 0)
    assert model._gradient_route_manifest is None
    current_step[0] = 1
    loss = model.training_step({"example": batch()}, 1)
    loss.backward()
    model.on_after_backward()
    model.on_before_optimizer_step(None)
    assert model._gradient_route_manifest is not None
    assert "Train/ActionFlow/GradientNorm/TotalPreclip" in recorded
    assert recorded["Train/ActionFlow/Compute/PeakAllocatedBytes"] == 0


def test_full_hydra_likelihood_graph_has_unblocked_inference_plan():
    config_dir = Path(__file__).parents[1] / "egomimic/hydra_configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve())):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=[
                "+experiment=pusht/action_flow_bc_usocket_bridge_likelihood_s42",
                "++paths.root_dir=.",
            ],
        )
    assert cfg.train.action_flow_method == cfg.model.action_flow_method
    assert not cfg.evaluator.action_flow_diagnostics.enabled
    algo = hydra.utils.instantiate(cfg.model.pipeline, device="cpu")
    pipeline = algo.pipeline
    runnable, excluded = pipeline.plan(
        ["front_img_1", "state_agent_obj"], mode="inference"
    )
    assert [type(s).__name__ for s in runnable] == [
        "FusedObsEncoder",
        "GaussianLatentNoise",
        "ConditionalReverseMeanStage",
        "LikelihoodDecoderStage",
    ]
    assert all(missing == ["<train-only>"] for _, missing in excluded)
    assert sum(p.numel() for p in pipeline.stages[0].parameters()) == 11197088
    assert sum(p.numel() for p in pipeline.stages[3].parameters()) == 10768
    assert sum(p.numel() for p in pipeline.stages[5].parameters()) == 39506641
    assert sum(p.numel() for p in pipeline.stages[6].parameters()) == 10744
    # PipelineAlgo is the orchestration adapter; registered parameters live in nets.
    assert sum(p.numel() for p in algo.nets.parameters()) == 50725241


@pytest.mark.parametrize(
    "kwargs", [{"num_levels": 1}, {"rho": 1}, {"sigma_min": 0}, {"sigma_max": 2}]
)
def test_invalid_reference_schedule_fails_closed(kwargs):
    with pytest.raises(ValueError):
        GaussianBridgeSchedule(**kwargs)
