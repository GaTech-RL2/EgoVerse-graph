from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.trainHydra import _resolve_model_wrapper_class
from tools.validate_action_flow_config import (
    STOPGRAD_UNITE_METHOD,
    _validate_dimensions_and_modules,
    action_flow_method,
    validate_method_contract,
)


CONFIG_DIR = Path(__file__).parents[1] / "egomimic" / "hydra_configs"
LAUNCHER = Path(__file__).parents[1] / "scripts" / "train" / "launch_action_flow_usocket.sbatch"
ROW = "action_flow_chain_points6_latent_fm_sg_unite_h384d12h12_sum14_cfg4_val10k_s42"


def _compose(monkeypatch):
    monkeypatch.setenv("PUSHSHAPES_CHAIN_GRIPPER_ROOT", "/verified/chain")
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        return compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment=pusht/{ROW}", "++paths.root_dir=."],
        )


def test_h384_points6_row_preserves_requested_contract(monkeypatch):
    cfg = _compose(monkeypatch)
    stages = cfg.model.pipeline.stages
    encoder = next(stage for stage in stages if stage._target_.endswith("ContentEncoderStage"))
    field = next(stage for stage in stages if stage._target_.endswith("ConditionalVelocityStage"))
    decoder = next(stage for stage in stages if stage._target_.endswith("ContentDecoderStage"))
    objective = stages[-1]

    assert cfg.model._target_ == "egomimic.pl_utils.pl_model.ModelWrapper"
    assert _resolve_model_wrapper_class(cfg) is ModelWrapper
    assert cfg.model.action_dim == 6
    assert cfg.planar.action_dims.pushshapes_sim_chain_gripper == 6
    assert cfg.run_provenance.action_contract.representation == (
        "left_xy_center_xy_right_xy_points6"
    )
    assert list(cfg.evaluator.semantic_blocks) == [[0, 2], [2, 4], [4, 6]]
    assert encoder.encoder.action_dim == 6
    assert decoder.decoder.action_dim == 6
    for backbone in (encoder.encoder.backbone, field.field.backbone):
        assert backbone.hidden_dim == 384
        assert backbone.depth == 12
        assert backbone.num_heads == 12
    assert decoder.decoder.hidden_dim == 384
    assert decoder.decoder.depth == 12
    assert decoder.decoder.num_heads == 12
    assert cfg.model.flow_mini_batch == 14
    assert cfg.model.flow_samples_per_content == 14
    assert objective.flow_aggregation == "sum_samples"
    assert objective.action_velocity_weight == pytest.approx(1.0)
    assert field.inference_method == "dopri5"
    assert field.dopri5_atol == pytest.approx(1e-6)
    assert field.dopri5_rtol == pytest.approx(1e-3)
    assert cfg.model.cfg_scale == pytest.approx(4.0)
    assert cfg.model.optimizer.lr == pytest.approx(1e-4)
    assert cfg.model.scheduler.base_lr_2 == pytest.approx(5e-5)
    assert cfg.model.scheduler.final_lr == pytest.approx(5e-5)
    dataset = cfg.data.train_datasets.pushshapes_sim_chain_gripper
    assert dataset.resolver.expected_episode_count == 4918
    assert dataset.expected_train_episode_count == 4869
    assert dataset.expected_valid_episode_count == 49
    assert cfg.trainer.max_steps == 150_000
    assert cfg.trainer.val_check_interval == 10_000
    assert cfg.callbacks.model_checkpoint.every_n_train_steps == 30_000


def test_h384_points6_uses_strict_routed_preflight_instead_of_legacy_name_gate():
    launcher = LAUNCHER.read_text()
    assert 'test "$AF_CLEAN_CHAIN_4918" = true' in launcher
    assert "cotrain or native-points6 experiment names/topologies" in launcher


def test_h384_points6_is_registered_as_stopgrad_unite(monkeypatch):
    cfg = _compose(monkeypatch)
    assert action_flow_method(cfg, f"pusht/{ROW}") == STOPGRAD_UNITE_METHOD
    assert validate_method_contract(cfg, f"pusht/{ROW}") == STOPGRAD_UNITE_METHOD


def test_h384_points6_full_dimension_gate_accepts_native_six_dim(monkeypatch):
    cfg = _compose(monkeypatch)
    pipeline = instantiate(cfg.model.pipeline)
    dimensions, parameters = _validate_dimensions_and_modules(
        cfg, list(pipeline.pipeline.stages)
    )
    assert dimensions["action"] == [16, 6]
    assert parameters["encoder_e"]["total"] == 32_726_064
    assert parameters["decoder_g"]["total"] == 21_304_326
