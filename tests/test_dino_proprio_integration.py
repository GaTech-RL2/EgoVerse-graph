"""Exercise composed configs through optimization and scheduled validation.

Tiny synthetic fixtures test framework wiring, not LIBERO experiment quality.
"""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import open_dict

from egomimic.pl_utils.pl_model import ModelWrapper
from tests.test_dino_proprio_data import actions, prepared


def configuration(arm):
    directory = Path(__file__).parents[1] / "egomimic/hydra_configs"
    with initialize_config_dir(config_dir=str(directory), version_base="1.2"):
        return compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment=dino_proprio/{arm}"],
        )


@pytest.mark.parametrize(
    "arm,parameters",
    [
        ("visual_state", 115206921),
        ("direct_action", 114021127),
    ],
)
def test_exact_pilot_configs_dimensions_and_optimizer(arm, parameters):
    cfg = configuration(arm)
    stage = cfg.model.pipeline.stages[0]
    with torch.device("meta"):
        model = instantiate(stage.model)
    assert sum(p.numel() for p in model.parameters()) == parameters
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == parameters
    assert model.image_dim == model.width == 768
    assert len(model.blocks) == 8
    assert len(model.blocks[0].qkv) == 2
    assert not any("track" in key for key in model.state_dict())
    assert cfg.model.optimizer.lr == 1e-4
    assert cfg.model.optimizer.weight_decay == 0.01
    assert cfg.model.scheduler is None
    assert cfg.trainer.max_steps == 1000
    assert cfg.callbacks.model_checkpoint.every_n_train_steps == 250
    assert cfg.callbacks.model_checkpoint.save_top_k == -1
    assert cfg.data.min_frames == cfg.data.max_frames == 4
    assert cfg.data.max_past_seconds is cfg.data.max_future_seconds is None
    if arm == "direct_action":
        assert cfg.data.action_manifest_path is None
        assert not hasattr(model, "image_out")
    else:
        assert model.pose_conditioning == "low_noise"


@pytest.mark.parametrize("arm", ["visual_state", "direct_action"])
def test_optimizer_scheduled_validation_and_checkpoint(tmp_path, arm):
    cfg = configuration(arm)
    manifest = prepared(tmp_path)
    with open_dict(cfg):
        cfg.data.manifest_path = str(manifest)
        if arm == "direct_action":
            cfg.data.action_manifest_path = str(actions(manifest))
        stage = cfg.model.pipeline.stages[0]
        stage.warmup_steps = 0
        stage.pose_ramp_steps = 0
        stage.sampler_steps = 2
        stage.model.pose_dims = {"gripper": 4}
        stage.model.action_dims = {"gripper": 3} if arm == "direct_action" else None
        stage.model.image_dim = 8
        stage.model.width = 12
        stage.model.depth = 1
        stage.model.heads = 3
        stage.model.attention_query_chunk_size = 4
        stage.model.max_tokens = 100
    data = instantiate(cfg.data)
    context = data.prepare_context(
        mode="train", normalization="prepared", normalizer=None
    )
    evaluator = instantiate(cfg.evaluator)
    model = ModelWrapper(config_tree=cfg, evaluator=evaluator, enable_grad_norm=False)
    model.data_context = context
    context.bind(model.model, evaluator)
    data.configure_evaluation(evaluator.data_requirements())
    before = next(model.parameters()).detach().clone()
    callback = ModelCheckpoint(
        dirpath=tmp_path / "checkpoints",
        every_n_train_steps=2,
        save_top_k=-1,
        save_last=True,
        save_on_train_epoch_end=False,
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        logger=False,
        max_steps=3,
        max_epochs=-1,
        min_epochs=0,
        limit_val_batches=1,
        val_check_interval=2,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[callback],
        default_root_dir=tmp_path,
    )
    trainer.fit(model, datamodule=data)
    assert trainer.global_step == 3
    assert data.consumed_batches == 3
    assert not torch.equal(before, next(model.parameters()).detach())
    expected = (
        ["image_normalized_mse", "proprio_normalized_mse"]
        if arm == "visual_state"
        else ["action_command_mse"]
    )
    for metric in expected:
        found = [
            v
            for k, v in trainer.callback_metrics.items()
            if k.startswith("val/gripper/") and k.endswith("/" + metric)
        ]
        assert found and all(torch.isfinite(value) for value in found)
    assert callback.last_model_path
    checkpoint = torch.load(callback.last_model_path, weights_only=False)
    assert checkpoint["global_step"] == 2
    assert checkpoint["data_context"] == context.snapshot()
    assert "mot_rng_state" in checkpoint
    optimizer = trainer.optimizers[0]
    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.999)
