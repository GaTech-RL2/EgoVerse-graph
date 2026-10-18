"""Generic pixel pipeline reaches optimization, RGB validation and checkpoints."""

import json

import pytest
import torch
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import open_dict

from egomimic.pl_utils.pl_model import ModelWrapper
from tests.test_dino_proprio_integration import configuration
from tests.test_image_proprio_validation_lifecycle import needs_final_validation
from tests.test_pixel_attention_profiles import pixel_stage, recipe
from tests.test_pixel_proprio_data import pixel_manifest


@pytest.mark.parametrize("mode", ["continuous", "discrete"])
@pytest.mark.parametrize(
    "objective", ["visual_only", "visual_state_action", "images_only"]
)
def test_pixel_pipeline_optimization_validation_and_checkpoint(
    tmp_path, mode, objective
):
    images_only = objective == "images_only"
    cfg = recipe(mode, "native") if images_only else configuration("visual_state")
    objective = "visual_only" if images_only else objective
    manifest = pixel_manifest(
        tmp_path, include_sharpa=False, frames=12, actions=not images_only
    )
    if images_only:
        # The actual image-only loader must not depend on measured states or
        # their normalizer, even when the reusable source manifest records them.
        metadata = json.loads(manifest.read_text())
        for episode in metadata["episodes"]:
            (manifest.parent / episode["path"]).unlink()
        (manifest.parent / metadata["normalization"]["path"]).unlink()
    chunk = 2 if mode == "discrete" else 1
    with open_dict(cfg):
        cfg.data = {
            "_target_": "egomimic.rldb.pixel_proprio_datamodule.PixelProprioDataModule",
            "manifest_path": str(manifest),
            "task": "visual_state_action"
            if objective == "visual_state_action"
            else "visual_diffusion",
            "mode": mode,
            "sampling_strategy": "discrete_window"
            if mode == "discrete"
            else "full_episode_uniform",
            "min_frames": 4,
            "max_frames": 4,
            "discrete_frames": 4,
            "stride": chunk,
            "batch_size": 1,
            "num_workers": 0,
            "checkpointable_sampling": True,
            "validation_anchors_per_episode": 2,
            "image_only": images_only,
        }
        if images_only:
            stage = pixel_stage(cfg)
            pixel_stage(cfg, "pixel_sampler").sampler.steps = 2
        else:
            stage = cfg.model.pipeline.stages[0]
            cfg.model.pipeline.stage_ids = {"image_proprio": 0}
            stage._target_ = "egomimic.pipeline.stages_image_proprio.ImageProprioStage"
            stage.warmup_steps = stage.pose_ramp_steps = 0
            stage.sampler_steps = 2
        stage.model._target_ = "egomimic.models.image_proprio.ImageProprioModel"
        stage.model.objective = objective
        stage.model.pose_dims = None if images_only else {"libero_gripper": 9 * chunk}
        stage.model.action_dims = (
            {"libero_gripper": 7 * chunk}
            if objective == "visual_state_action"
            else None
        )
        stage.model.image_dim = 12
        stage.model.width = 16
        stage.model.depth = 1
        stage.model.heads = 2
        if images_only:
            # Keep the production three-axis position encoding, with dimensions
            # matching this tiny model and the fixture's 4px / 2px patch grid.
            stage.model.rope_axis_dims = [4, 2, 2]
            stage.model.spatial_grid_size = [2, 2]
        stage.model.max_tokens = 100
        stage.model.attention_query_chunk_size = 0 if images_only else 16
        cfg.evaluator = {
            "_target_": "egomimic.eval.image_proprio_eval.ImageProprioEval",
            "objective": objective,
            "pixel_diagnostics": {
                "artifact_dir": str(tmp_path / "pixel_diagnostics"),
                "write_video": True,
                "log_wandb": False,
            },
        }
    data = instantiate(cfg.data)
    context = data.prepare_context(
        mode="train", normalization="prepared", normalizer=None
    )
    evaluator = instantiate(cfg.evaluator)
    model = ModelWrapper(config_tree=cfg, evaluator=evaluator, enable_grad_norm=False)
    model.data_context = context
    context.bind(model.model, evaluator)
    data.configure_evaluation(evaluator.data_requirements())
    if images_only:
        for loader in (data.train_dataloader(), data.val_dataloader()):
            batch = next(iter(loader))
            assert batch
            assert all(set(item["world"]) == {"image"} for item in batch.values())
        network = model.model.pipeline.stage_by_id("pixel_denoiser").model
        assert network.modalities == ("image",)
        assert network.pose_dims == {} and network.action_dims == {}
        assert all(len(block.qkv) == 1 for block in network.blocks)
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
        max_steps=2,
        max_epochs=-1,
        min_epochs=0,
        limit_val_batches=2,
        val_check_interval=2,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[callback],
        default_root_dir=tmp_path,
    )
    trainer.fit(model, datamodule=data)
    assert trainer.global_step == data.consumed_batches == 2
    # The scheduled terminal pass already published immutable RGB artifacts.
    # This is the same guard used by trainHydra's val_at_end path.
    assert evaluator.has_completed_validation(trainer.global_step) is True
    assert needs_final_validation(evaluator, trainer.global_step) is False
    if needs_final_validation(evaluator, trainer.global_step):
        trainer.validate(model, datamodule=data)
    assert needs_final_validation(evaluator, trainer.global_step + 1) is True
    assert not torch.equal(before, next(model.parameters()).detach())
    metrics = {
        k: v for k, v in trainer.callback_metrics.items() if k.startswith("val/")
    }
    assert metrics and all(torch.isfinite(value) for value in metrics.values())
    assert any(k.endswith("image_normalized_mse") for k in metrics)
    if objective == "visual_only":
        assert all(k.endswith("image_normalized_mse") for k in metrics)
    else:
        assert any(k.endswith("proprio_normalized_mse") for k in metrics)
        assert any(k.endswith("action_command_mse") for k in metrics)
    checkpoint = torch.load(callback.last_model_path, weights_only=False)
    assert checkpoint["global_step"] == 2
    assert checkpoint["data_context"] == context.snapshot()
    assert "mot_rng_state" in checkpoint
    if images_only:
        assert trainer.optimizers[0].param_groups[0]["lr"] == 5e-5
        restored = ModelWrapper(
            config_tree=cfg, evaluator=instantiate(cfg.evaluator), enable_grad_norm=False
        )
        restored.load_state_dict(checkpoint["state_dict"], strict=True)
        for key, value in restored.state_dict().items():
            assert torch.equal(value, checkpoint["state_dict"][key])
    reports = list((tmp_path / "pixel_diagnostics").rglob("diagnostic.json"))
    assert reports
    for path in reports:
        report = json.loads(path.read_text())
        assert report["objective"] == objective
        assert report["pixel_metadata"]["chunk_size"] == chunk
        assert set(report["cameras"]) == {"0", "1"}
        for camera in report["cameras"].values():
            assert (path.parent / camera["video"]).is_file()
            assert (path.parent / camera["contact_sheet"]).is_file()
        if objective == "visual_only":
            assert "proprio" not in report and "action" not in report
        else:
            assert report["action"]["value_space"] == "native_action_commands"
            assert report["proprio"]["value_space"] == "normalized_measured_state"
