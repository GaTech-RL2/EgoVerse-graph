"""Compose actual pixel recipes without loading data or the optional CUDA runtime."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "egomimic/hydra_configs"
EXECUTION_KEYS = {
    "attention_backend",
    "attention_dtype",
    "strict_fp32",
    "attention_query_chunk_size",
    "activation_checkpointing",
}


def recipe(arm, profile=None, execution=None):
    overrides = [
        f"+experiment=pixel_proprio/video_{arm}",
        "hydra/launcher=basic",
        "data.manifest_path=/fixture/prepared.json",
        "paths.output_dir=/fixture/output",
    ]
    if profile is not None:
        overrides.append(f"world_model_attention={profile}")
    if execution is not None:
        overrides.append(f"+pixel_execution={execution}")
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        return compose(config_name="train_zarr_cartesian", overrides=overrides)


def scientific_stage(cfg):
    stage = OmegaConf.to_container(cfg.model.pipeline, resolve=True)
    model = stage["stages"][stage["stage_ids"]["pixel_denoiser"]]["model"]
    for key in EXECUTION_KEYS:
        model.pop(key)
    return stage


def pixel_stage(cfg, name="pixel_denoiser"):
    return cfg.model.pipeline.stages[cfg.model.pipeline.stage_ids[name]]


@pytest.mark.parametrize("arm", ["discrete", "continuous"])
def test_pixel_default_is_explicit_flash4_profile(arm):
    default, explicit = recipe(arm), recipe(arm, "flash4_bf16_mixed")
    assert OmegaConf.to_container(default, resolve=False) == OmegaConf.to_container(
        explicit, resolve=False
    )
    model = pixel_stage(default).model
    assert model.attention_backend == "flash4"
    assert model.attention_dtype == "bfloat16"
    assert not model.strict_fp32 and not default.runtime.strict_fp32
    assert default.runtime.disable_tf32
    assert default.trainer.precision == "bf16-mixed"


@pytest.mark.parametrize("arm", ["discrete", "continuous"])
@pytest.mark.parametrize(
    "profile,backend,dtype,strict,disable_tf32,trainer_precision",
    [
        ("native", "sdpa", "native", False, False, "32-true"),
        ("unchunked_fp32", "efficient", "native", True, True, "32-true"),
        ("flash4_bf16", "flash4", "bfloat16", False, True, "32-true"),
        ("flash4_bf16_mixed", "flash4", "bfloat16", False, True, "bf16-mixed"),
    ],
)
def test_pixel_profile_override_preserves_science(
    arm, profile, backend, dtype, strict, disable_tf32, trainer_precision
):
    cfg, baseline = recipe(arm, profile), recipe(arm, "native")
    model = pixel_stage(cfg).model
    assert model.attention_backend == backend
    assert model.attention_dtype == dtype
    assert model.strict_fp32 is strict and cfg.runtime.strict_fp32 is strict
    assert cfg.runtime.disable_tf32 is disable_tf32
    assert model.attention_query_chunk_size == 0
    assert not model.activation_checkpointing
    assert scientific_stage(cfg) == scientific_stage(baseline)
    for section in ("data", "callbacks"):
        assert OmegaConf.to_container(
            cfg[section], resolve=True
        ) == OmegaConf.to_container(baseline[section], resolve=True)
    trainer = OmegaConf.to_container(cfg.trainer, resolve=True)
    baseline_trainer = OmegaConf.to_container(baseline.trainer, resolve=True)
    assert trainer.pop("precision") == trainer_precision
    baseline_trainer.pop("precision")
    assert trainer == baseline_trainer
    assert cfg.trainer.max_steps == 300000 and cfg.trainer.val_check_interval == 250
    assert cfg.model.optimizer == baseline.model.optimizer
    assert cfg.model.optimizer.lr == 5e-5 and cfg.model.optimizer.weight_decay == 0.01
    assert cfg.model.scheduler is None and not cfg.model.enable_grad_norm
    assert model.objective == "visual_only" and model.action_dims is None
    assert cfg.data.image_only and not model.phase_a_condition_on_proprio
    assert model.pose_dims is None
    assert model.image_dim == model.width == 768
    assert model.depth == 8 and model.heads == 6 and model.mlp_ratio == 4
    assert model.max_cameras == 2 and model.use_camera_embedding
    assert not model.use_modality_embedding and not model.use_embodiment_embedding
    assert not model.use_phase_embedding
    assert model.noise_conditioning == "adaln_zero"
    assert not model.allow_patch_compression
    assert pixel_stage(cfg, "pixel_loss").weight == 1
    assert pixel_stage(cfg, "pixel_sampler").sampler.steps == 20
    assert all(
        "warmup_steps" not in stage and "pose_ramp_steps" not in stage
        for stage in cfg.model.pipeline.stages
    )
    assert cfg.data.mode == arm and model.time_mode == arm
    if arm == "discrete":
        assert cfg.data.sampling_strategy == "discrete_window"
        assert cfg.data.discrete_frames == 16 and cfg.data.stride == 4
        assert model.max_tokens == 3200
    else:
        assert cfg.data.sampling_strategy == "full_episode_uniform"
        assert cfg.data.min_frames == 2 and cfg.data.max_frames == 64
        # Retain the existing model ceiling; stage1 sampling supplies <=12,800.
        assert model.max_tokens == 25600


@pytest.mark.parametrize("arm", ["discrete", "continuous"])
def test_composed_image_only_model_has_one_expert_and_no_readouts(arm):
    cfg = recipe(arm)
    with torch.device("meta"):
        model = instantiate(pixel_stage(cfg).model)
    assert model.modalities == ("image",)
    assert model.pose_dims == {} and model.action_dims == {}
    assert sum(p.numel() for p in model.parameters()) == 88_178_688
    assert not hasattr(model, "phase")
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert all(len(block.qkv) == 1 for block in model.blocks)
    assert not any(
        key.startswith(("pose_in.", "pose_out.", "action_out.", "proprio_role."))
        for key in model.state_dict()
    )


@pytest.mark.parametrize("arm", ["discrete", "continuous"])
def test_pixel_native_precision_fallback_remains_cpu_compatible(arm):
    cfg = recipe(arm, "native")
    with open_dict(cfg):
        del cfg.world_model_attention.trainer_precision
    assert cfg.trainer.precision == "32-true"


@pytest.mark.parametrize(
    "name", ["video_only", "video_discrete", "readout_continuous", "readout_discrete"]
)
def test_continuous_stage_one_cap_does_not_change_other_recipe_defaults(name):
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=[
                f"+experiment=pixel_proprio/{name}",
                "hydra/launcher=basic",
            ],
        )
    assert cfg.data.min_frames == 2 and cfg.data.max_frames == 128
    assert cfg.data.validation_sampling_strategy == "discrete_window"
    assert cfg.data.validation_discrete_frames == 16
    assert cfg.data.validation_stride == 4


@pytest.mark.parametrize("arm", ["discrete", "continuous"])
def test_four_gpu_profile_changes_only_execution_and_global_batch(arm):
    baseline, cfg = recipe(arm), recipe(arm, execution="ddp4")
    assert scientific_stage(cfg) == scientific_stage(baseline)
    assert cfg.model.optimizer == baseline.model.optimizer
    assert cfg.trainer.precision == "bf16-mixed"
    assert cfg.trainer.strategy == "ddp"
    assert cfg.trainer.devices == cfg.data.distributed_world_size == 4
    assert cfg.trainer.num_nodes == cfg.data.batch_size == 1
    assert cfg.trainer.accumulate_grad_batches == 1
    assert cfg.trainer.use_distributed_sampler is False
    assert cfg.data.checkpointable_sampling is True
    assert cfg.runtime.cluster_environment == "torch_elastic"
    assert cfg.runtime.slurm_requeue_owner == "none"
    actual_data = OmegaConf.to_container(cfg.data, resolve=True)
    original_data = OmegaConf.to_container(baseline.data, resolve=True)
    actual_data.pop("distributed_world_size")
    original_data.pop("distributed_world_size", None)
    assert actual_data == original_data
    actual_trainer = OmegaConf.to_container(cfg.trainer, resolve=True)
    original_trainer = OmegaConf.to_container(baseline.trainer, resolve=True)
    for key in (
        "strategy",
        "devices",
        "num_nodes",
        "accumulate_grad_batches",
        "use_distributed_sampler",
    ):
        actual_trainer.pop(key, None)
        original_trainer.pop(key, None)
    assert actual_trainer == original_trainer
    assert cfg.callbacks == baseline.callbacks
    assert recipe(arm, "native", "ddp4").trainer.precision == "32-true"
