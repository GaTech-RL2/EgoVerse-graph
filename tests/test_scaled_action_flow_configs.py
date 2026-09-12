from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir


CONFIG_DIR = Path(__file__).parents[1] / "egomimic" / "hydra_configs"


def _compose(row: str, monkeypatch):
    monkeypatch.setenv("PUSHSHAPES_USOCKET_ROOT", "/verified/usocket")
    monkeypatch.setenv("PUSHSHAPES_CHAIN_GRIPPER_ROOT", "/verified/chain")
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        return compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment=pusht/{row}", "++paths.root_dir=."],
        )


ROWS = (
    "action_flow_usocket_latent_fm_sg_unite_h512d14h16_sum14_cfg4_val10k_s42",
    "action_flow_chain_points6_latent_fm_sg_unite_h512d14h16_sum14_cfg4_val10k_s42",
    "action_flow_cotrain_uc_latent_fm_sg_unite_h512d14h16_sum14_cfg4_val10k_s42",
)


@pytest.mark.parametrize("row", ROWS)
def test_scaled_rows_preserve_linked_action_flow_recipe(row, monkeypatch):
    cfg = _compose(row, monkeypatch)
    stages = cfg.model.pipeline.stages
    objective = stages[-1]
    bridge = next(stage for stage in stages if stage._target_.endswith("LatentBridgeStage"))
    field = next(
        stage for stage in stages if stage._target_.endswith("ConditionalVelocityStage")
    )

    assert cfg.model._target_.endswith("ActionFlowModelWrapper")
    assert cfg.model.flow_mini_batch == 14
    assert cfg.model.flow_samples_per_content == 14
    assert cfg.model.condition_dim == 128
    assert cfg.model.condition_dropout_probability == pytest.approx(0.1)
    assert cfg.model.cfg_scale == pytest.approx(4.0)
    assert cfg.model.num_inference_steps == 50
    assert bridge.time_sampling == "lognormal_shifted"
    assert bridge.independent_noise_per_sample
    assert bridge.independent_condition_dropout_per_sample
    assert field.inference_method == "dopri5"
    assert field.dopri5_atol == pytest.approx(1e-6)
    assert field.dopri5_rtol == pytest.approx(1e-3)
    assert field.field.backbone.hidden_dim == 512
    assert field.field.backbone.depth == 14
    assert field.field.backbone.num_heads == 16
    assert objective.flow_aggregation == "sum_samples"
    assert objective.action_velocity_weight == pytest.approx(1.0)
    assert cfg.model.optimizer._target_.endswith("ReleasedUniteCompositeOptimizer")
    assert cfg.model.optimizer.lr == pytest.approx(1e-4)
    assert cfg.model.scheduler.base_lr_1 == pytest.approx(1e-4)
    assert cfg.model.scheduler.base_lr_2 == pytest.approx(5e-5)
    assert cfg.model.scheduler.final_lr == pytest.approx(5e-5)
    assert cfg.trainer.max_steps == 150_000
    assert cfg.trainer.val_check_interval == 10_000
    assert cfg.callbacks.model_checkpoint.every_n_train_steps == 30_000


def test_cotrain_row_has_two_private_codecs_and_one_shared_field(monkeypatch):
    cfg = _compose(ROWS[2], monkeypatch)
    stages = cfg.model.pipeline.stages
    encoder = next(
        stage for stage in stages if stage._target_.endswith("RoutedContentEncoderStage")
    )
    decoder = next(
        stage for stage in stages if stage._target_.endswith("RoutedContentDecoderStage")
    )
    fields = [
        stage for stage in stages if stage._target_.endswith("ConditionalVelocityStage")
    ]
    routes = {"pushshapes_sim_u_socket", "pushshapes_sim_chain_gripper"}

    assert set(encoder.encoders) == routes
    assert set(decoder.decoders) == routes
    assert encoder.encoders.pushshapes_sim_u_socket.action_dim == 4
    assert encoder.encoders.pushshapes_sim_chain_gripper.action_dim == 6
    assert decoder.decoders.pushshapes_sim_u_socket.action_dim == 4
    assert decoder.decoders.pushshapes_sim_chain_gripper.action_dim == 6
    assert len(fields) == 1
    assert cfg.data.train_dataloader_params.pushshapes_sim_u_socket.batch_size == 32
    assert cfg.data.train_dataloader_params.pushshapes_sim_chain_gripper.batch_size == 32
    assert cfg.run_provenance.objective.domain_aggregation == (
        "equal_mean_from_one_batch32_per_domain"
    )
