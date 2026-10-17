from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

CONFIG_DIR = Path(__file__).parents[1] / "egomimic/hydra_configs"


@pytest.mark.parametrize("codec", ["transformer", "small", "unet"])
@pytest.mark.parametrize("phase", ["full", "smoke"])
def test_corrected_recipe_reaches_consumers_and_preserves_smoke(
    monkeypatch, codec, phase
):
    for name in [
        "MATCHED_AF_SPLIT_SHA256",
        "MATCHED_SPLIT_SHA256",
        "MATCHED_NORMALIZATION_SHA256",
        "MATCHED_SOURCE_COMMIT",
    ]:
        monkeypatch.setenv(name, "test-identity")
    monkeypatch.setenv("AF_COTRAIN_NORM", "/test/train-only-norm.json")
    experiment = (
        "yam_human_af64_unet1d_large_codecs"
        if codec == "unet"
        else "yam_human_af64_observation640_total180m_open_loop"
    )
    overrides = [
        f"+experiment=e1/{experiment}",
        "+stationary_recipe=history2_sum_cfg4",
        "++paths.root_dir=.",
    ]
    if codec == "small":
        overrides += [
            "model.pipeline.stages.6.field.backbone.hidden_dim=576",
            "model.pipeline.stages.6.field.backbone.depth=12",
        ]
    if phase == "smoke":
        overrides += [
            "trainer.max_steps=8",
            "trainer.val_check_interval=8",
            "callbacks.model_checkpoint.every_n_train_steps=8",
        ]
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        cfg = compose(config_name="train_zarr_cartesian", overrides=overrides)
    assert cfg.model.cfg_scale == cfg.model.pipeline.stages[6].cfg_scale == 4.0
    assert (
        cfg.model.flow_loss_aggregation
        == cfg.model.pipeline.stages[-1].flow_aggregation
        == "sum_samples"
    )
    assert cfg.model.pipeline.stages[-1].flow_samples_per_content == 14
    for source in [
        cfg.data.train_datasets.yam_bimanual,
        cfg.data.train_datasets.human_bimanual,
        cfg.data.valid_datasets.yam.yam_bimanual,
        cfg.data.valid_datasets.human.human_bimanual,
    ]:
        assert source.resolver.observation_history_steps == 2
    cadence = 8 if phase == "smoke" else 15000
    assert (
        cfg.trainer.val_check_interval
        == cfg.callbacks.model_checkpoint.every_n_train_steps
        == cadence
    )
    assert cfg.trainer.accumulate_grad_batches == 1
    assert cfg.trainer.check_val_every_n_epoch is None
    assert cfg.trainer.num_sanity_val_steps == 0
    assert cfg.callbacks.model_checkpoint.every_n_epochs is None
    assert cfg.callbacks.model_checkpoint.save_on_train_epoch_end is False
    assert cfg.callbacks.model_checkpoint.save_top_k == -1
    assert list(cfg.stationary_speed.human_rates) == [0.2, 0.4, 0.6, 0.8, 1.0]
    assert list(cfg.stationary_speed.yam_rates) == [1.0]
    assert cfg.model.pipeline.conditioning_input == "retiming_multiplier"
    assert cfg.model.num_latent_tokens == 64 and cfg.model.latent_dim == 16
