from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from egomimic.trainHydra import _instantiate_model_wrapper


CONFIG_DIR = Path(__file__).parents[1] / "egomimic" / "hydra_configs"
ROW = "action_flow_cotrain_uc_latent_fm_sg_unite_h384d12h12_sum14_cfg4_val10k_s42"


def test_original_size_cotrain_contract(monkeypatch):
    monkeypatch.setenv("PUSHSHAPES_USOCKET_ROOT", "/verified/usocket")
    monkeypatch.setenv("PUSHSHAPES_CHAIN_GRIPPER_ROOT", "/verified/chain")
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment=pusht/{ROW}", "++paths.root_dir=."],
        )

    stages = cfg.model.pipeline.stages
    encoder = next(s for s in stages if s._target_.endswith("RoutedContentEncoderStage"))
    decoder = next(s for s in stages if s._target_.endswith("RoutedContentDecoderStage"))
    fields = [s for s in stages if s._target_.endswith("ConditionalVelocityStage")]
    field = fields[0]
    objective = stages[-1]
    routes = {"pushshapes_sim_u_socket", "pushshapes_sim_chain_gripper"}

    assert cfg.model._target_.endswith("ModelWrapper")
    assert cfg.model.hidden_dim == 384
    assert cfg.model.flow_samples_per_content == 14
    assert cfg.model.flow_mini_batch == 14
    assert cfg.model.cfg_scale == pytest.approx(4.0)
    assert cfg.model.num_inference_steps == 50
    assert field.inference_method == "dopri5"
    assert field.field.backbone.hidden_dim == 384
    assert field.field.backbone.depth == 12
    assert field.field.backbone.num_heads == 12
    assert len(fields) == 1
    assert set(encoder.encoders) == routes
    assert set(decoder.decoders) == routes
    assert encoder.encoders.pushshapes_sim_u_socket.action_dim == 4
    assert encoder.encoders.pushshapes_sim_chain_gripper.action_dim == 6
    assert decoder.decoders.pushshapes_sim_u_socket.action_dim == 4
    assert decoder.decoders.pushshapes_sim_chain_gripper.action_dim == 6
    assert objective.flow_aggregation == "sum_samples"
    assert objective.action_velocity_weight == pytest.approx(1.0)
    assert cfg.model.optimizer.lr == pytest.approx(1e-4)
    assert cfg.model.scheduler.base_lr_1 == pytest.approx(1e-4)
    assert cfg.model.scheduler.base_lr_2 == pytest.approx(5e-5)
    assert cfg.model.scheduler.final_lr == pytest.approx(5e-5)
    assert cfg.trainer.max_steps == 150_000
    assert cfg.trainer.val_check_interval == 10_000
    assert cfg.callbacks.model_checkpoint.every_n_train_steps == 30_000
    assert cfg.data.train_dataloader_params.pushshapes_sim_u_socket.batch_size == 32
    assert cfg.data.train_dataloader_params.pushshapes_sim_chain_gripper.batch_size == 32
    assert cfg.run_provenance.objective.domain_aggregation == (
        "equal_mean_from_one_batch32_per_domain"
    )

    model = _instantiate_model_wrapper(cfg)
    assert model is not None
