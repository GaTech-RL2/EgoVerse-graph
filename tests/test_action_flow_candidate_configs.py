"""Exercise the maintained Hydra rows and their candidate-specific telemetry."""

from pathlib import Path

import hydra
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from egomimic.eval.action_flow_diagnostics import ActionFlowDiagnostics
from egomimic.pl_utils.pl_model_action_flow import ActionFlowModelWrapper


CONFIG_DIR = Path(__file__).parents[1] / "egomimic/hydra_configs"
ROWS = [
    ("latent_fm_sg_recon1", "latent_fm_stopgrad", 1.0),
    ("graph_section", "graph_section_diagnostic", 0.0),
]


def _compose(row):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        return compose(
            config_name="train_zarr_cartesian",
            overrides=[
                f"+experiment=pusht/action_flow_bc_usocket_{row}_s42",
                "++paths.root_dir=.",
            ],
        )


@pytest.mark.parametrize("row,method,reconstruction_weight", ROWS)
def test_candidate_row_instantiates_exact_dimensions_and_shared_codec(
    row, method, reconstruction_weight
):
    cfg = _compose(row)
    assert cfg.model.action_flow_method == cfg.train.action_flow_method == method
    assert cfg.run_provenance.objective.method == method
    assert cfg.model.action_horizon == 16
    assert cfg.model.latent_dim == 8
    assert cfg.model.action_dim == 4
    assert cfg.model.reconstruction_weight == reconstruction_weight
    assert cfg.run_provenance.objective.decoded_noise_scale_weight == 0
    assert cfg.model.flow_samples_per_content == 14
    assert cfg.model.condition_dropout_probability == 0.3
    assert cfg.model.optimizer.lr == 3e-5

    algo = hydra.utils.instantiate(cfg.model.pipeline, device="cpu")
    stages = algo.pipeline.stages
    encoder, field, decoder, objective = stages[3], stages[5], stages[6], stages[7]
    assert field.field.horizon == 16
    assert field.field.input_dim == field.field.output_dim == 8
    assert field.field.condition_dim == 67
    assert field.field.depth == 12
    assert 39_000_000 <= sum(p.numel() for p in field.parameters()) <= 41_000_000
    assert objective.reconstruction_weight == reconstruction_weight
    assert stages[1].num_tokens == 16 and stages[1].latent_dim == 8
    parameters = list(algo.nets.parameters())
    assert len(parameters) == len({id(parameter) for parameter in parameters})
    if method == "latent_fm_stopgrad":
        assert field.flow_clean_gradient_mode == "all_stopgrad"
        assert objective.residual_key == field.flow_residual_key
        assert decoder.residual_key == field.residual_key
        assert field.flow_residual_key != field.residual_key
        assert encoder.encoder.depth == decoder.decoder.depth == 2
    else:
        assert field.flow_clean_gradient_mode == "full"
        assert encoder.encoder.graph is decoder.decoder.graph
        encoder_parameters = {id(parameter) for parameter in encoder.parameters()}
        residual_parameters = {id(parameter) for parameter in decoder.decoder.residual.parameters()}
        assert encoder_parameters.isdisjoint(residual_parameters)
        assert decoder.decoder.latent_dim - decoder.decoder.action_dim == 4
        assert not cfg.run_provenance.objective.reconstruction_is_optimizer_objective
        assert cfg.run_provenance.requirements.r6 == "FAIL_restricted_diagnostic"
        assert not cfg.evaluator.action_flow_diagnostics.capture_activations
        target = torch.randn(2, 16, 4)
        torch.testing.assert_close(
            decoder.decoder(encoder.encoder(target)), target, rtol=0, atol=1e-6
        )


@pytest.mark.parametrize("row,method,reconstruction_weight", ROWS)
def test_candidate_wrapper_gradient_routes_and_compute_contract(
    monkeypatch, row, method, reconstruction_weight
):
    cfg = _compose(row)
    # Preserve the real Hydra codec/factory/stage wiring and all sequence
    # dimensions; replace only the expensive shared field width/depth. Inputs
    # are already encoded observations, so image/data nodes are not exercised.
    pipeline_cfg = OmegaConf.create(
        OmegaConf.to_container(cfg.model.pipeline, resolve=True)
    )
    pipeline_cfg.stages = list(pipeline_cfg.stages)[3:]
    field_config = pipeline_cfg.stages[2].field
    field_config.hidden_dim = 32
    field_config.depth = 2
    field_config.num_heads = 4
    field_config.feedforward_dim = 64
    field_config.time_embedding_dim = 32
    algo = hydra.utils.instantiate(pipeline_cfg, device="cpu")
    wrapper = ActionFlowModelWrapper(pipeline=algo, gradient_telemetry_cadence=1)
    logged = {}
    monkeypatch.setattr(
        wrapper, "log", lambda key, value, **kwargs: logged.update({key: value})
    )
    torch.manual_seed(51)
    loss = wrapper.training_step({"fixture": {
        "target": torch.randn(2, 16, 4),
        "condition": torch.randn(2, 67),
        "sampler/noise": torch.randn(2, 16, 8),
    }}, 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert wrapper._gradient_route_manifest is not None
    routes = wrapper._gradient_route_manifest["routes"]
    intersections = wrapper._gradient_route_manifest["intersections"]
    assert routes["FM"] and routes["ActionVelocity"]
    assert intersections["FM__ActionVelocity"]
    assert all(torch.isfinite(p.grad).all() for p in wrapper.parameters() if p.grad is not None)
    calls = 2 if method == "latent_fm_stopgrad" else 1
    assert logged["Train/ActionFlow/Compute/FieldForwardCallsPerStep"] == calls
    assert logged["Train/ActionFlow/Compute/FieldSampleEquivalentsPerStep"] == calls * 14
    if method == "latent_fm_stopgrad":
        assert intersections["FM__Reconstruction"] == []
        assert logged["Train/ActionFlow/GradientCosineDefined/FM__Reconstruction"] == 0
        assert logged["Train/ActionFlow/GradientIntersectionParameterCount/FM__Reconstruction"] == 0
        assert routes["Reconstruction"]
    else:
        assert "Reconstruction" not in routes
        assert "Train/ActionFlow/GradientNorm/Reconstruction" not in logged
        torch.testing.assert_close(
            loss,
            logged["Train/ActionFlow/FlowMatchingLoss"]
            + logged["Train/ActionFlow/ActionVelocityLoss"],
        )


def test_graph_factory_rejects_independent_encoder_configuration():
    cfg = _compose("graph_section")
    pipeline_cfg = OmegaConf.create(
        OmegaConf.to_container(cfg.model.pipeline, resolve=True)
    )
    pipeline_cfg.stages = [pipeline_cfg.stages[3], pipeline_cfg.stages[6]]
    pipeline_cfg.stages[0].encoder = {"_target_": "torch.nn.Identity"}
    with pytest.raises(hydra.errors.InstantiationException, match="second codec"):
        hydra.utils.instantiate(pipeline_cfg, device="cpu")


@pytest.mark.parametrize("row,method,reconstruction_weight", ROWS)
def test_candidate_diagnostic_config_constructs_without_stale_activation_settings(
    tmp_path, row, method, reconstruction_weight
):
    cfg = _compose(row)
    diagnostics = cfg.evaluator.action_flow_diagnostics
    diagnostics.noise_seed_bank_path = str(
        CONFIG_DIR / "evaluator/energy_score_seed_bank_k32_v1.json"
    )
    diagnostics.artifact_root = str(tmp_path / "diagnostics")
    config = OmegaConf.to_container(diagnostics, resolve=True)
    instance = ActionFlowDiagnostics(config)
    assert instance.jacobian_samples == 2
    assert instance.capture_activations == (method == "latent_fm_stopgrad")
    if method == "graph_section_diagnostic":
        assert instance.activation_layer_map == ()
        assert instance.cknna_k == 0
