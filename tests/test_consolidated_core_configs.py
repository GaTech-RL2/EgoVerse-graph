from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

CONFIG_DIR = Path(__file__).parents[1] / "egomimic" / "hydra_configs"


def _compose(row, *extra):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        return compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment=pusht/{row}", "++paths.root_dir=.", *extra],
        )


@pytest.mark.parametrize(
    "row",
    (
        "action_flow_bc_usocket_latent_fm_sg_recon1_200m_adamw_lr1e5_s42",
        "action_flow_bc_usocket_latent_fm_sg_recon1_200m_muon_lr1e5_s42",
        "action_flow_usocket_latent_fm_sg_unite_h384_s42",
        "action_flow_usocket_latent_fm_sg_unite_h384_sum14_cfg4_val8_s42",
        "action_flow_usocket_latent_fm_sg_unite_denoiser90m_sum14_cfg4_finalval1_s42",
        "action_flow_bc_usocket_latent_fm_sg_unite_arch_denoiser90m_sum14_adamw_lr1e5_s42",
        "action_flow_usocket_latent_fm_sg_unite_h384_sum14_cfg4_val8_scale1_s42",
        "action_flow_chain_latent_fm_sg_unite_h384_sum14_cfg4_val8_s42",
    ),
)
def test_action_flow_core_rows_compose_with_portable_dataset_root(
    row, monkeypatch
):
    root = "/verified/cluster/datasets/Tsim_v2"
    monkeypatch.setenv("PUSHSHAPES_DATA_ROOT", root)

    cfg = _compose(row)

    for split in (cfg.data.train_datasets, cfg.data.valid_datasets):
        for dataset in split.values():
            assert str(dataset.resolver.folder_path).startswith(f"{root}/")
    assert cfg.model._target_ == "egomimic.pl_utils.pl_model.ModelWrapper"
    assert cfg.model.training_behavior._target_.endswith(
        "ActionFlowTrainingBehavior"
    )
    assert cfg.model.action_horizon == 16
    assert cfg.model.flow_samples_per_content == 14


def test_action_flow_denoiser90m_changes_only_field_capacity(monkeypatch):
    monkeypatch.setenv("PUSHSHAPES_DATA_ROOT", "/verified/cluster/datasets/Tsim_v2")
    cfg = _compose(
        "action_flow_usocket_latent_fm_sg_unite_denoiser90m_sum14_cfg4_finalval1_s42"
    )

    encoder = cfg.model.pipeline.stages[4].encoder.backbone
    field = cfg.model.pipeline.stages[6].field.backbone
    decoder = cfg.model.pipeline.stages[7].decoder

    assert (encoder.hidden_dim, encoder.depth, encoder.num_heads) == (384, 12, 12)
    assert (field.hidden_dim, field.depth, field.num_heads) == (640, 12, 16)
    assert (decoder.hidden_dim, decoder.depth, decoder.num_heads) == (384, 12, 12)
    assert cfg.model.gradient_checkpointing is False
    assert encoder.gradient_checkpointing is False
    assert field.gradient_checkpointing is False
    assert decoder.gradient_checkpointing is False
    assert cfg.model.flow_samples_per_content == 14
    assert cfg.model.flow_loss_aggregation == "sum_samples"
    assert cfg.model.action_flow_method == "latent_fm_stopgrad_unite"
    assert cfg.trainer.val_check_interval == 150_000
    assert cfg.trainer.limit_val_batches == 0
    assert cfg.run_provenance.validation.full_run_enabled is False
    assert cfg.run_provenance.validation.smoke_limit_batches == 1
    assert cfg.run_provenance.architecture.denoiser.parameters == 90_396_000
    assert cfg.run_provenance.architecture.total_parameters == 155_626_932


def test_action_flow_denoiser90m_adamw_reuses_only_unite_modules(monkeypatch):
    monkeypatch.setenv("PUSHSHAPES_DATA_ROOT", "/verified/cluster/datasets/Tsim_v2")
    cfg = _compose(
        "action_flow_bc_usocket_latent_fm_sg_unite_arch_denoiser90m_sum14_adamw_lr1e5_s42"
    )

    stages = cfg.model.pipeline.stages
    encoder = stages[3].encoder.backbone
    field = stages[5].field.backbone
    decoder = stages[6].decoder

    assert (encoder.hidden_dim, encoder.depth, encoder.num_heads) == (384, 12, 12)
    assert (field.hidden_dim, field.depth, field.num_heads) == (640, 12, 16)
    assert (decoder.hidden_dim, decoder.depth, decoder.num_heads) == (384, 12, 12)
    assert len(stages) == 8
    assert stages[0]._target_.endswith("FusedObsEncoder")
    assert tuple(stages[0].inputs) == ("front_img_1", "state_agent_obj")
    assert stages[1]._target_.endswith("GaussianLatentNoise")
    assert stages[4]._target_.endswith("LatentBridgeStage")
    assert stages[4].time_sampling == "uniform"
    assert stages[5]._target_.endswith("ConditionalVelocityStage")
    assert stages[5].inference_method == "euler"
    assert stages[5].cfg_scale == pytest.approx(1.0)
    assert stages[6]._target_.endswith("ContentDecoderStage")
    assert stages[6].reconstruction_noising_probability == pytest.approx(0.0)
    assert stages[7]._target_.endswith("ActionFlowObjectiveStage")
    assert stages[7].moment_weight == pytest.approx(0.0)
    assert stages[7].flow_aggregation == "sum_samples"
    assert cfg.norm_stats.norm_mode == "quantile"
    assert cfg.model.condition_dropout_probability == pytest.approx(0.3)
    assert cfg.model.num_inference_steps == 16
    assert cfg.model.action_flow_method == "latent_fm_stopgrad"
    assert cfg.model.num_latent_tokens == 8
    assert cfg.model.latent_dim == 16
    assert cfg.model.optimizer._target_ == "torch.optim.AdamW"
    assert cfg.model.optimizer.lr == pytest.approx(1.0e-5)
    assert cfg.model.optimizer.weight_decay == pytest.approx(1.0e-4)
    assert cfg.model.optimizer_named_parameters is False
    assert cfg.model.scheduler._target_.endswith("warmup_cosine_scheduler")
    assert cfg.model.scheduler.max_steps == 240_000
    assert cfg.model.scheduler.warmup_steps == 8_000
    assert cfg.model.scheduler.eta_min == pytest.approx(1.0e-6)
    assert cfg.model.flow_samples_per_content == 14
    assert cfg.model.action_horizon == 16
    assert cfg.trainer.max_steps == 240_000
    assert cfg.trainer.val_check_interval == 240_000
    assert cfg.trainer.limit_val_batches == 0
    assert cfg.trainer.precision == "bf16"
    assert cfg.callbacks.model_checkpoint.every_n_train_steps == 40_000
    assert cfg.run_provenance.validation.interval_optimizer_steps == 240_000
    assert cfg.run_provenance.architecture.unite_usage == (
        "encoder_decoder_and_denoiser_modules_only"
    )
    assert cfg.run_provenance.recipe.classifier_free_guidance is False

    # Composition alone does not exercise constructor enums such as the stage's
    # ``euler`` implementation name. Instantiate the real graph so this typed
    # launch row cannot pass while remaining non-executable.
    pipeline = instantiate(cfg.model.pipeline, device="cpu")
    assert len(pipeline.pipeline.stages) == 8


def test_released_unite_cotrain_core_row_composes(monkeypatch):
    root = "/verified/cluster/datasets/Tsim_v2"
    monkeypatch.setenv("PUSHSHAPES_DATA_ROOT", root)

    cfg = _compose(
        "unite_cotrain_usocket_chain_val01_h16",
        "model=bf/ct_unite_register_separate_nt8_h384_s42",
    )

    assert tuple(cfg.data.train_datasets) == (
        "pushshapes_sim_u_socket",
        "pushshapes_sim_chain_gripper",
    )
    for split in (cfg.data.train_datasets, cfg.data.valid_datasets):
        for dataset in split.values():
            assert str(dataset.resolver.folder_path).startswith(f"{root}/")
    assert cfg.model.num_latent_tokens == 8
    assert cfg.model.hidden_dim == 384
    assert cfg.model.action_horizon == 16
    assert cfg.model._target_ == "egomimic.pl_utils.pl_model.ModelWrapper"
    assert cfg.model.training_behavior._target_.endswith(
        "ReleasedUniteTrainingBehavior"
    )
