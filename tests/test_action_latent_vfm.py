from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from egomimic.models.adaln_backbone import AdaLNBackbone
from egomimic.pipeline.stages_action_latent_vfm import (
    FlowBridgeNoisingStage,
    LatentVelocityFieldStage,
    ReconstructionBridgeNoisingStage,
    TinyActionLatentEncoder,
    VelocityEndpointStage,
    VelocityFlowObjectiveStage,
)
from egomimic.pl_utils.pl_model_action_latent_vfm import (
    ActionLatentVFMModelWrapper,
)
from tools.config_graph import build_graph

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = (
    ROOT
    / "egomimic/hydra_configs/experiment/pusht/action_latent_vfm_usocket_val01_h16.yaml"
)


class _RecordingVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.conditions = []

    def forward(self, latent, time, condition):
        self.conditions.append(condition.detach().clone())
        return self.scale * latent


class _MetricPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.velocity = LatentVelocityFieldStage(
            denoising_module=_RecordingVelocity(),
            condition_dim=6,
        )
        self.stages = nn.ModuleList((self.velocity,))


class _MetricAlgo:
    def __init__(self):
        self.nets = nn.ModuleDict({"pipeline": _MetricPipeline()})
        self.device = torch.device("cpu")

    @property
    def pipeline(self):
        return self.nets["pipeline"]

    def process_batch_for_training(self, batch):
        return batch

    def forward_training(self, batch):
        weight = self.pipeline.velocity.denoising_module.scale
        results = OrderedDict()
        for source, values in batch.items():
            results[source] = {
                "target": weight.new_zeros((values["count"], 1, 1)),
                "loss/action_latent_reconstruction": (weight - 0.0).square(),
                "loss/action_latent_fm": (weight - 2.0).square(),
                "log/action_latent_reconstruction_l1": weight.detach().abs(),
            }
        return results


def test_action_encoder_maps_h16_actions_to_n16_d8_latent():
    encoder = TinyActionLatentEncoder(
        action_dim=4,
        action_horizon=16,
        num_latent_tokens=16,
        latent_dim=8,
        hidden_dim=32,
        depth=2,
        num_heads=4,
    )
    actions = torch.randn(3, 16, 4)
    output = encoder(actions)
    assert output.shape == (3, 16, 8)
    torch.testing.assert_close(output, encoder(actions))


def test_general_adaln_backbone_has_no_content_or_sampler_contract():
    backbone = AdaLNBackbone(
        input_dim=8,
        output_dim=8,
        horizon=16,
        condition_dim=128,
        hidden_dim=32,
        depth=2,
        num_heads=4,
        gradient_checkpointing=False,
    )
    output = backbone(torch.randn(3, 16, 8), torch.rand(3), torch.randn(3, 128))
    assert output.shape == (3, 16, 8)
    names = dict(backbone.named_parameters())
    assert not any("content" in name for name in names)
    assert not any("sampler" in name or "velocity" in name for name in names)


def test_fixed_euler_diagnostics_capture_real_denoiser_layers():
    backbone = AdaLNBackbone(
        input_dim=8,
        output_dim=8,
        horizon=16,
        condition_dim=6,
        hidden_dim=32,
        depth=4,
        num_heads=4,
        gradient_checkpointing=False,
    )
    stage = LatentVelocityFieldStage(
        denoising_module=backbone,
        condition_dim=6,
        num_inference_steps=2,
        cfg_scale=1.0,
    ).eval()
    states, predictions, activations = stage.diagnostic_rollout(
        clean_latent=torch.randn(3, 16, 8),
        noise=torch.randn(3, 16, 8),
        condition=torch.randn(3, 6),
        raw_noise_levels=[0.0, 0.5, 1.0],
    )
    assert states.shape == (3, 3, 16, 8)
    assert predictions.shape == (3, 3, 16, 8)
    assert tuple(activations) == tuple(f"block_{index:02d}" for index in range(4))
    assert all(value.shape == (3, 3, 16, 32) for value in activations.values())


def test_fm_detaches_clean_target_but_reconstruction_preserves_encoder_gradient():
    clean = torch.randn(2, 16, 8, requires_grad=True)
    fm = FlowBridgeNoisingStage(samples_per_reconstruction=14)({"latent/clean": clean})
    assert fm["fm/noisy_latent"].shape == (28, 16, 8)
    assert not fm["fm/noisy_latent"].requires_grad
    assert not fm["fm/target_velocity"].requires_grad

    reconstruction = ReconstructionBridgeNoisingStage(0.5, 1.0)({"latent/clean": clean})
    assert reconstruction["reconstruction/noisy_latent"].requires_grad


def test_velocity_endpoint_and_reconstruction_loss_update_local_prediction():
    noisy = torch.randn(2, 16, 8)
    velocity = torch.randn(2, 16, 8, requires_grad=True)
    time = torch.tensor([0.5, 0.75])
    batch = VelocityEndpointStage()(
        {
            "reconstruction/noisy_latent": noisy,
            "reconstruction/time": time,
            "reconstruction/pred_velocity": velocity,
        }
    )
    expected = noisy + (1.0 - time.reshape(-1, 1, 1)) * velocity
    assert torch.allclose(batch["reconstruction/pred_clean_latent"], expected)


def test_flow_objective_sums_fourteen_full_mean_terms():
    batch = VelocityFlowObjectiveStage(samples_per_reconstruction=14)(
        {
            "fm/pred_velocity": torch.ones(28, 16, 8),
            "fm/target_velocity": torch.zeros(28, 16, 8),
        }
    )
    assert float(batch["loss/action_latent_fm"]) == 14.0


def test_condition_dropout_replaces_condition_but_not_noisy_latent():
    denoiser = _RecordingVelocity()
    stage = LatentVelocityFieldStage(
        denoising_module=denoiser,
        condition_dim=6,
        condition_dropout_probability=1.0,
        cfg_scale=4.0,
    )
    stage.train()
    condition = torch.randn(2, 6)
    fm_latent = torch.randn(4, 16, 8)
    reconstruction_latent = torch.randn(2, 16, 8)
    output = stage.execute(
        {
            "condition": condition,
            "fm/noisy_latent": fm_latent,
            "fm/time": torch.rand(4),
            "fm/condition_repeat": 2,
            "reconstruction/noisy_latent": reconstruction_latent,
            "reconstruction/time": torch.rand(2),
        },
        mode="train",
    )
    assert torch.equal(output["fm/pred_velocity"], fm_latent)
    assert torch.equal(output["reconstruction/pred_velocity"], reconstruction_latent)
    for observed in denoiser.conditions:
        assert torch.allclose(observed, stage._null_condition_like(observed))
    assert float(output["log/action_latent_fm_condition_drop_fraction"]) == 1.0
    assert (
        float(output["log/action_latent_reconstruction_condition_drop_fraction"]) == 1.0
    )


def test_wrapper_weights_components_and_uses_shared_denoiser_telemetry(monkeypatch):
    wrapper = ActionLatentVFMModelWrapper(pipeline=_MetricAlgo())
    logged = {}
    monkeypatch.setattr(
        wrapper,
        "log",
        lambda name, value, **kwargs: logged.setdefault(name, (value, kwargs)),
    )
    batch = OrderedDict(
        (
            ("small", {"count": 1}),
            ("large", {"count": 3}),
        )
    )
    loss = wrapper.training_step(batch, 0)
    assert float(loss) == 2.0
    assert float(logged["Train/ActionLatentVFM/TotalLoss"][0]) == 2.0

    parameter = wrapper._velocity_stage().denoising_module.scale
    parameter.grad = torch.tensor(7.0)
    before = parameter.grad.clone()
    wrapper._measure_gradient_conflict(
        (parameter - 0.0).square(), (parameter - 2.0).square()
    )
    assert float(logged["log/unite_gradient_cosine"][0]) == -1.0
    assert "log/unite_recon_grad_norm" in logged
    assert "log/unite_denoise_grad_norm" in logged
    torch.testing.assert_close(parameter.grad, before)


def test_graph_is_lint_clean_and_declares_condition_dropout():
    graphs = {
        mode: build_graph(
            EXPERIMENT,
            mode=mode,
            overrides=("model=bf/us_action_latent_vfm_nt16_d8_h512_s42",),
        )
        for mode in ("train", "inference")
    }
    assert not graphs["train"]["lint"]
    assert not graphs["inference"]["lint"]
    velocity = next(
        node
        for node in graphs["train"]["nodes"]
        if node["t"] == "LatentVelocityFieldStage"
    )
    assert velocity["p"]["condition_dropout_probability"] == 0.3
    assert (
        velocity["p"]["denoising_module"]["_target_"]
        == "egomimic.models.adaln_backbone.AdaLNBackbone"
    )
    assert "max_condition_tokens" not in velocity["p"]["denoising_module"]
    assert "max_content_tokens" not in velocity["p"]["denoising_module"]
    assert velocity["p"]["denoising_module"]["hidden_dim"] == 512
    assert velocity["p"]["denoising_module"]["depth"] == 12
    experiment = OmegaConf.load(EXPERIMENT)
    assert experiment.callbacks.model_checkpoint.every_n_train_steps == 40000
    assert experiment.evaluator.unite_diagnostics.enabled is True
