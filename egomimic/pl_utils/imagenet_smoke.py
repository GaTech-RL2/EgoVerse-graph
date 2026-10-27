"""Dedicated real ImageNet smoke/capacity evidence; never a production initializer.

Historical temporal Elmo smoke contracts remain unchanged. Importing this
module performs no CUDA operation; runtime entry points require the actual
allocated four-H100 execution before touching model/checkpoint tensors.
"""

import copy
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

import torch
from hydra.utils import instantiate
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf

from egomimic.eval.image_proprio_eval import ImageProprioEval
from egomimic.pipeline.checkpoint_binding import validate_checkpoint_binding
from egomimic.pipeline.construction import checkpoint_construction
from egomimic.pl_utils.joint_smoke import (
    ConditionalOptimizerUsage, check_conditional_adam_state, load_smoke_configs,
    require, weights_digest, _optimizer_capacity_reserve_bytes,
)
from egomimic.pl_utils.pl_model import ModelWrapper
from scripts.train.joint.lr_policy import from_config, update_lr, verify_scheduler
from scripts.train.joint.native_recovery import conditional_record, preserved_rng, state_digest

WORLD_SIZE = 4
STEPS = 50
TOTAL_PARAMETERS = 314227000
EXECUTED_PARAMETERS = 159006592
VALIDATION_ANCHORS = 1000
LOCAL_VALIDATION_ANCHORS = 250
METRIC = "val/imagenet/image_generation/image_normalized_mse"
_HEX = re.compile(r"[0-9a-f]{64}")


def _plain(value):
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def _sha(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_once(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def _torch_once(path, value):
    with Path(path).open("xb") as stream:
        torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())


def _read_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 8 * 1024 * 1024,
            "Missing, linked or oversized smoke proof")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate smoke proof key")
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError("Nonfinite smoke proof JSON: " + value)
    value = json.loads(path.read_bytes(), object_pairs_hook=unique, parse_constant=nonfinite)
    require(isinstance(value, dict), "Smoke proof must be a mapping")
    return value


def validate_imagenet_config(cfg, *, smoke=True):
    """Pure exact scientific/runtime-shape contract, callable in CPU fixtures."""
    # A retained .hydra/config.yaml may legitimately contain an unrelated
    # paths.work_dir=${hydra:runtime.cwd}. Postflight is not a Hydra launcher;
    # resolve only the complete contract sections we actually authenticate.
    c = {key: _plain(cfg[key]) for key in ("data", "model", "trainer", "evaluator", "callbacks")}
    c.update({key: _plain(cfg.get(key)) for key in ("seed", "norm_stats", "normalizer", "val_at_start", "val_at_end")})
    data, model, trainer = c["data"], c["model"], c["trainer"]
    require(c.get("seed") == 42 and c.get("norm_stats") is None
            and c.get("normalizer") is None and not c.get("val_at_start", False)
            and c.get("val_at_end") is False, "ImageNet normalization/validation policy changed")
    require(data.get("_target_") == "egomimic.rldb.imagenet_imagefolder.ImageNetImageFolderDataModule"
            and data.get("expected_classes") == 1000 and data.get("expected_train_images") == 1281167
            and data.get("expected_val_images") == 50000 and data.get("batch_size") == 8
            and data.get("validation_batch_size") == 1 and data.get("distributed_world_size") == WORLD_SIZE
            and data.get("seed") == 42, "ImageNet source/count/batch/topology contract changed")
    for key in ("manifest_sha256", "class_language_sha256", "language_features_sha256", "validation_anchor_sha256"):
        require(isinstance(data.get(key), str) and _HEX.fullmatch(data[key]) is not None,
                "Actual ImageNet metadata SHA256 is unbound: " + key)
    for key in ("manifest_path", "data_root", "class_language_path", "language_features_path", "validation_anchor_path"):
        require(isinstance(data.get(key), str) and Path(data[key]).is_absolute()
                and "UNBOUND" not in data[key], "Actual ImageNet source path is unbound: " + key)
    require(trainer.get("devices") == 4 and trainer.get("num_nodes") == 1
            and trainer.get("precision") == "bf16-mixed" and trainer.get("accelerator") == "gpu"
            and trainer.get("accumulate_grad_batches") == 1 and trainer.get("gradient_clip_val") is None
            and trainer.get("use_distributed_sampler") is False
            and trainer.get("num_sanity_val_steps") == 0 and trainer.get("check_val_every_n_epoch") is None
            and trainer.get("strategy", {}).get("_target_") == "lightning.pytorch.strategies.DDPStrategy"
            and trainer["strategy"].get("find_unused_parameters") is True,
            "ImageNet actual DDP/BF16/optimizer-step policy changed")
    require(trainer.get("max_steps") == (50 if smoke else 300000)
            and trainer.get("val_check_interval") == (50 if smoke else 20000),
            "ImageNet scheduled train/validation cadence changed")
    limit = trainer.get("limit_val_batches")
    require((type(limit) is float and limit == 1.0) or (smoke and type(limit) is int and limit == 250),
            "ImageNet validation must consume every exact1000 frozen anchor")
    evaluator = c.get("evaluator", {})
    require(evaluator.get("_target_") == "egomimic.eval.image_proprio_eval.ImageProprioEval"
            and evaluator.get("objective") == "image_generation" and evaluator.get("sample_seed") == 42
            and evaluator.get("active_streams_from_data") is False, "Wrong ImageNet generation evaluator")
    require(model.get("_target_") == "egomimic.pl_utils.pl_model.ModelWrapper"
            and model.get("optimizer") == dict(_target_="torch.optim.AdamW", _partial_=True,
                                               lr=5e-6, weight_decay=.01)
            and from_config(model) == dict(peak_lr=5e-6, warmup_optimizer_steps=3000,
                                           schedule="linear_warmup_then_constant"),
            "ImageNet AdamW/warmup policy changed")
    behavior = model.get("training_behavior", {})
    require(behavior.get("_target_") == "egomimic.pl_utils.world_model_behavior.WorldModelTrainingBehavior"
            and behavior.get("distributed_noise_seed") == 42, "Missing shared RNG/sampler behavior")
    pipeline = model["pipeline"]
    stages = pipeline["stages"]
    require(pipeline.get("_target_") == "egomimic.pipeline.algo.PipelineAlgo" and len(stages) == 4
            and pipeline.get("stage_ids") == dict(joint_noising=0, joint_denoiser=1, image_loss=2, joint_sampler=3)
            and pipeline.get("initialization") is None and pipeline.get("trainability") is None
            and pipeline.get("loss_pipeline") is None and not pipeline.get("training_passes")
            and pipeline.get("compatibility_mode", "current") == "current",
            "ImageNet must use the unchanged four-stage joint pipeline")
    require(stages[0].get("_target_") == "egomimic.pipeline.stages_joint_state_action.JointNoisingStage"
            and stages[0]["noising"].get("_target_") == "egomimic.models.joint_image_proprio.CoupledImageProprioNoising"
            and stages[0]["noising"].get("maximum") == 1.0
            and stages[0]["noising"].get("active_streams") == ["image", "proprio", "action", "camera"],
            "ImageNet noising changed")
    denoiser = stages[1]
    net = denoiser["model"]
    require(denoiser.get("_target_") == "egomimic.pipeline.stages_joint_state_action.JointDenoiserStage"
            and denoiser.get("language_key") == "language_features"
            and net.get("_target_") == "egomimic.models.joint_image_proprio.JointImageProprioModel"
            and net.get("active_streams") == ["image", "proprio", "action", "camera"]
            and net.get("pose_dims") == dict(eva_bimanual=20, human_bimanual=126)
            and net.get("action_dims") == dict(eva_bimanual=20)
            and net.get("camera_dims") == dict(eva_bimanual=9, human_bimanual=9)
            and net.get("proprio_width") == 512 and net.get("action_width") == net.get("camera_width") == 384
            and net.get("separate_proprio_experts") is True and net.get("language_feature_dim") == 1024
            and net.get("language_width") == 128 and net.get("camera_max_count") == 3,
            "Full ImageNet shared/private bank architecture changed")
    image = net["image_model"]
    expected = dict(_target_="egomimic.models.image_proprio.ImageProprioModel", objective="visual_only",
                    image_dim=768, width=1024, depth=8, heads=8, mlp_ratio=4,
                    require_observed_context=False, phase_a_condition_on_proprio=False,
                    pose_dims=None, action_dims=None, qk_norm=True, use_rmsnorm=True, use_swiglu=True,
                    swiglu_multiple_of=64, allow_patch_compression=False, activation_checkpointing=False,
                    noise_conditioning="adaln_zero", position_encoding="spatiotemporal_rope",
                    rope_axis_dims=[64, 32, 32], spatial_grid_size=[10, 10], time_mode="continuous",
                    attention_backend="flash4", attention_dtype="bfloat16", strict_fp32=False,
                    token_execution="packed_valid", compile_mode="default", compile_dynamic=True,
                    compile_backend="inductor", loss_reduction="mean_per_example")
    require(all(image.get(key) == value for key, value in expected.items()),
            "Full ImageNet FA4/packed/compiled QK-RMS-SwiGLU image architecture changed")
    require(stages[2] == dict(_target_="egomimic.pipeline.stages_joint_state_action.JointLossStage",
                             stream="image", weight=1.0,
                             loss=dict(_target_="egomimic.models.pixel_diffusion.PlainPixelMSE", reduction="mean_per_example"))
            and stages[3] == dict(_target_="egomimic.pipeline.stages_joint_state_action.JointSamplerStage",
                                  language_key="language_features",
                                  sampler=dict(_target_="egomimic.models.joint_image_proprio.JointImageProprioSampler",
                                               state_steps=20, action_steps=20)),
            "ImageNet plain image objective/native20-step sampler changed")
    callbacks = [item for item in c.get("callbacks", {}).values() if isinstance(item, dict)]
    require(sum(item.get("_target_") == "egomimic.pl_utils.joint_smoke.ConditionalOptimizerUsage"
                for item in callbacks) == 1, "Exactly one maintained conditional usage callback is required")
    if smoke:
        require(not any(item.get("_target_") == "lightning.pytorch.callbacks.ModelCheckpoint" for item in callbacks),
                "Smoke permits only its dedicated diagnostic checkpoint save")
        require(sum(item.get("_target_") == "egomimic.pl_utils.imagenet_smoke.ImageNetSmokeProof"
                    for item in callbacks) == 1, "Exactly one dedicated ImageNet smoke callback is required")
    else:
        checkpoints = [item for item in callbacks if item.get("_target_") == "lightning.pytorch.callbacks.ModelCheckpoint"]
        require(len(checkpoints) == 1 and checkpoints[0].get("every_n_train_steps") == 20000
                and checkpoints[0].get("every_n_epochs") == 0 and checkpoints[0].get("save_on_train_epoch_end") is False
                and checkpoints[0].get("save_top_k") == -1 and checkpoints[0].get("save_last") == "link"
                and not any(item.get("_target_") == "egomimic.pl_utils.imagenet_smoke.ImageNetSmokeProof" for item in callbacks),
                "Full ImageNet training retains only its scheduled20k production saves")
    return from_config(model)


def partition_parameters(module, network, *, expected_total=TOTAL_PARAMETERS,
                         expected_active=EXECUTED_PARAMETERS):
    """Map actual parameters, including inactive language fusions and identity."""
    native = dict(network.named_parameters())
    active, inactive = set(), set()
    for name, param in native.items():
        if name.startswith(("image_model.", "language_projection.", "language_fusions.image.")):
            active.add(id(param))
        else:
            require(name.startswith(("proprio_banks.", "action.", "camera.", "camera_identity.",
                                     "language_fusions.proprio__", "language_fusions.action.",
                                     "language_fusions.camera.")), "Unknown inactive parameter namespace: " + name)
            inactive.add(id(param))
    named = dict(module.named_parameters())
    require(set(map(id, named.values())) == active | inactive and active and inactive,
            "Wrapper/native parameter identities differ")
    require(sum(p.numel() for p in named.values()) == expected_total
            and sum(p.numel() for p in named.values() if id(p) in active) == expected_active,
            "Full or image-executed parameter capacity changed")
    return {name: ("active" if id(p) in active else "inactive") for name, p in named.items()}


def check_parameter_updates(names, partition, counts, expected_step=STEPS):
    require(isinstance(partition, dict) and set(partition.values()) == {"active", "inactive"}
            and isinstance(counts, dict) and isinstance(names, list)
            and len(names) == len(set(names)) and set(names) == set(partition)
            and set(counts) == set(partition), "ImageNet parameter usage identity/order is incomplete")
    require(all(type(counts[name]) is int and counts[name] == (expected_step if partition[name] == "active" else 0)
                for name in names), "ImageNet active/inactive optimizer usage differs")


def validate_fit_rows(rows, policy, job_id):
    """Pure verifier of all four actual producers; never manufactures fit rows."""
    require(isinstance(rows, list) and len(rows) == 4 and isinstance(job_id, str) and job_id.isdecimal(),
            "Need four actual ImageNet rank proofs")
    first = rows[0]
    for rank, record in enumerate(rows):
        require(record.get("schema") == "imagenet-joint-fit-v1" and record.get("status") == "FIT_PASS"
                and record.get("job_id") == job_id and record.get("rank") == rank
                and record.get("world_size") == 4 and record.get("global_step") == 50
                and record.get("rank_batch_size") == 8 and record.get("global_batch_size") == 32
                and record.get("distributed_backend") == "nccl" and record.get("trainer_precision") == "bf16-mixed"
                and record.get("validation_batches") == 1000 and record.get("local_validation_batches") == 250
                and record.get("validation_batch_indices") == list(range(rank * 250, (rank + 1) * 250))
                and record.get("parameters") == TOTAL_PARAMETERS
                and record.get("executed_parameters") == EXECUTED_PARAMETERS,
                "ImageNet fit proof allocation/count/anchor identity differs")
        require(record.get("bf16_training_outputs") == record.get("fp32_losses") == 50
                and all(record.get(key) is True for key in ("fp32_gradients_verified", "all_training_bf16_outputs",
                    "all50_lr_verified", "inactive_banks_unchanged", "language_conditioning_verified")),
                "Incomplete real ImageNet optimization/precision proof")
        rates = record.get("optimizer_update_lrs")
        require(record.get("lr_policy") == policy and isinstance(rates, list) and len(rates) == 50
                and all(type(rate) in (int, float) and math.isfinite(rate)
                        and math.isclose(rate, update_lr(policy, step), rel_tol=1e-12)
                        for step, rate in enumerate(rates)), "Missing actual all50 optimizer LR proof")
        check_parameter_updates(record.get("parameter_names"), record.get("parameter_partition", {}),
                                record.get("parameter_update_counts", {}))
        require(record.get("parameter_names") == first.get("parameter_names")
                and record.get("parameter_partition") == first.get("parameter_partition")
                and record.get("parameter_update_counts") == first.get("parameter_update_counts")
                and record.get("expert_parameters") == first.get("expert_parameters"),
                "Conditional parameter identities/counts differ across ranks")
        expert_counts = record.get("expert_parameters", {})
        require(isinstance(expert_counts, dict) and expert_counts
                and all(type(count) is int and count > 0 for count in expert_counts.values())
                and sum(expert_counts.values()) == TOTAL_PARAMETERS, "Per-expert counts do not cover full model")
        require(record.get("inactive_initial_sha256") == record.get("inactive_final_sha256")
                == first.get("inactive_initial_sha256")
                and record.get("weights_sha256") == first.get("weights_sha256")
                and record.get("initial_weights_sha256") == first.get("initial_weights_sha256")
                and record.get("weights_sha256") != record.get("initial_weights_sha256")
                and record.get("all_rank_weights") == [record.get("weights_sha256")] * 4
                and record.get("all_rank_initial_weights") == [record.get("initial_weights_sha256")] * 4,
                "ImageNet final/initial/inactive all-rank weight proof differs")
        for key in ("weights_sha256", "initial_weights_sha256", "inactive_final_sha256", "data_context_sha256"):
            require(isinstance(record.get(key), str) and _HEX.fullmatch(record[key]), "Missing exact fit digest")
        require(record.get("data_context_sha256") == first.get("data_context_sha256")
                and record.get("metrics") == first.get("metrics") and set(record.get("metrics", {})) == {METRIC}
                and math.isfinite(record["metrics"][METRIC]), "ImageNet validation metric/context reduction differs")
        require(record.get("active_streams") == ["image"] and record.get("attention_backend") == "flash4"
                and record.get("token_execution") == "packed_valid" and record.get("compile_mode") == "default"
                and "H100" in record.get("gpu_name", "") and record.get("gpu_capability", [0])[0] >= 9
                and isinstance(record.get("gpu_uuid"), str) and record["gpu_uuid"], "Missing H100/compiled FA4 proof")
        require(record.get("measured_updates") == 40
                and all(type(record.get(key)) in (int, float) and math.isfinite(record[key]) and record[key] > 0
                        for key in ("warm_wall_seconds", "validation_checkpoint_endcheck_seconds"))
                and all(type(record.get(key)) is int and record[key] > 0
                        for key in ("peak_allocated_bytes", "peak_reserved_bytes")), "Missing measured timing/capacity")
        require(record.get("checkpoint_name") == first.get("checkpoint_name")
                and re.fullmatch(r"smoke-epoch[0-9]{4,}-step000000050\.ckpt", record["checkpoint_name"]),
                "Diagnostic checkpoint name must retain true epoch/step")
    require(len({record["gpu_uuid"] for record in rows}) == 4, "ImageNet ranks must own distinct GPUs")
    return first


def validate_rng_payload(payload):
    """Pure shape/type and CPU-generator checks; does not access CUDA."""
    import random
    import numpy as np
    require(isinstance(payload, dict) and payload.get("schema") == "mot_rng_state.distributed.v1"
            and payload.get("world_size") == 4 and isinstance(payload.get("ranks"), list)
            and len(payload["ranks"]) == 4, "Checkpoint needs all four actual rank RNG states")
    for state in payload["ranks"]:
        require(isinstance(state, dict) and set(state) == {"torch_cpu", "torch_cuda", "numpy", "python", "cuda_scope"}
                and state["cuda_scope"] == "current_device"
                and isinstance(state["torch_cuda"], list) and len(state["torch_cuda"]) == 1,
                "Each saved rank must bind one current-device CUDA RNG state")
        for tensor in [state["torch_cpu"], state["torch_cuda"][0]]:
            require(torch.is_tensor(tensor) and tensor.dtype == torch.uint8 and tensor.ndim == 1
                    and tensor.numel() > 0 and tensor.device.type == "cpu", "Invalid saved RNG byte tensor")
        torch.Generator(device="cpu").set_state(state["torch_cpu"])
        np.random.RandomState().set_state(state["numpy"])
        random.Random().setstate(state["python"])
    return payload["ranks"]


def _network(module):
    return module.model.pipeline.stage_by_id("joint_denoiser").model


def _gpu_runtime(device, *, distributed):
    job = os.environ.get("SLURM_JOB_ID", "")
    require(job.isdecimal() and torch.cuda.is_available() and torch.cuda.device_count() == 4,
            "ImageNet evidence requires an actual allocated four-visible-GPU job")
    require(torch.device(device).type == "cuda", "ImageNet smoke cannot certify CPU execution")
    if distributed:
        require(torch.distributed.is_initialized() and torch.distributed.get_world_size() == 4
                and torch.distributed.get_backend() == "nccl", "ImageNet requires actual four-rank NCCL")
    props = torch.cuda.get_device_properties(device)
    require("H100" in props.name and props.major >= 9, "ImageNet requires actual H100 hardware")
    return props


def _network_contract(module):
    network = _network(module)
    image = network.image_model
    require(network.active_streams("imagenet") == ("image",)
            and tuple(network.streams) == ("image", "proprio", "action", "camera")
            and network.language_feature_dim == 1024 and network.language_width == 128,
            "Actual ImageNet model stream/language interfaces differ")
    require(image.compile_mode == "default" and image.compile_dynamic is True
            and image.compile_backend == "inductor" and image.token_execution == "packed_valid"
            and image.qk_norm and image.use_rmsnorm and image.use_swiglu
            and all(block.attention_backend == "flash4" for block in image.blocks),
            "Actual ImageNet network is not full compiled FA4/QK/RMS/SwiGLU")
    require(len({p.device for p in module.parameters()}) == 1
            and all(p.device.type == "cuda" and p.requires_grad and p.dtype == torch.float32 and torch.isfinite(p).all()
                for p in module.parameters()), "Stored ImageNet banks must all be trainable finite FP32")
    partition = partition_parameters(module, network)
    return network, partition


def _validate_sample(sample, *, batch_size, hidden=False):
    require(sample.get("embodiment") == "imagenet" and set(sample.get("world", {})) == {"image"},
            "ImageNet sample must contain only the genuine image stream")
    world = sample["world"]["image"]
    require(world["value"].shape == (batch_size, 100, 768) and world["value"].dtype == torch.float32
            and world["valid"].dtype == world["target_valid"].dtype == world["observed"].dtype == torch.bool
            and world["valid"].all() and world["target_valid"].all() and not world["observed"].any(),
            "ImageNet needs full T1/C1/P100 RGB targets with no observed pixels")
    if hidden:
        require(torch.isnan(world["value"]).all(), "Hidden target sample must replace every RGB value by NaN")
    else:
        require(torch.isfinite(world["value"]).all() and (world["value"].abs() <= 1.00001).all(),
                "ImageNet actual RGB targets must be finite normalized pixels")
    require(sample["language_features"].shape == (batch_size, 1024)
            and sample["language_features"].dtype == torch.float32
            and torch.isfinite(sample["language_features"]).all()
            and len(sample["pixel_metadata"]) == len(sample["task_kind"]) == batch_size
            and all(kind == "image_generation" for kind in sample["task_kind"]),
            "ImageNet requires actual finite class-language conditioning")
    require(isinstance(sample.get("task_id"), (list, tuple)) and len(sample["task_id"]) == batch_size
            and isinstance(sample.get("language_instruction"), (list, tuple))
            and len(sample["language_instruction"]) == batch_size, "Each ImageNet row needs its exact class text/ID")
    for row, details in enumerate(sample["pixel_metadata"]):
        feature = sample["language_features"][row].detach().cpu().contiguous().numpy().astype("<f4", copy=False)
        feature_sha = hashlib.sha256(feature.tobytes()).hexdigest()
        require(details.get("frame_indices") == [0] and details.get("timestamps_seconds") == [0.0]
                and details.get("image_size") == [160, 160] and details.get("patch_size") == 16
                and details.get("camera_names") == ["image"] and _HEX.fullmatch(details.get("source_sha256", ""))
                and details.get("class_feature_sha256") == feature_sha
                and details.get("wnid") == sample["task_id"][row]
                and details.get("class_name") == sample["language_instruction"][row],
                "Missing exact row-bound ImageNet source/class/vector metadata")
    reference = copy.copy(sample)
    if hidden:
        reference["world"] = {"image": dict(world, value=torch.zeros_like(world["value"]))}
    ImageProprioEval._validate_image_generation({"imagenet": reference})


def _inactive_digest(module, partition):
    return weights_digest({name: p for name, p in module.named_parameters() if partition[name] == "inactive"})


def _cpu(value):
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value, dict): return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list): return [_cpu(item) for item in value]
    if isinstance(value, tuple): return tuple(_cpu(item) for item in value)
    return copy.deepcopy(value)


class ImageNetSmokeProof(Callback):
    """Exact50 updates plus all1000 scheduled anchors, with inactive-bank proof."""
    def __init__(self, output, expected_steps=50, warmup_steps=10):
        require(type(expected_steps) is int and expected_steps == 50 and type(warmup_steps) is int
                and warmup_steps == 10, "ImageNet smoke is exactly50 updates, warming through10")
        self.output, self.expected_steps, self.warmup_steps = Path(output), expected_steps, warmup_steps

    def on_fit_start(self, trainer, pl_module):
        self.props = _gpu_runtime(pl_module.device, distributed=True)
        require(trainer.world_size == 4 and trainer.precision == "bf16-mixed"
                and trainer.global_step == 0 and not trainer.ckpt_path
                and len(trainer.optimizers) == 1 and not trainer.optimizers[0].state,
                "ImageNet smoke must start fresh with exact actual topology/precision")
        require(isinstance(trainer.optimizers[0], torch.optim.AdamW)
                and all(g["weight_decay"] == .01 for g in trainer.optimizers[0].param_groups),
                "Actual ImageNet optimizer must be unchanged AdamW weight decay")
        cfg = OmegaConf.load(self.output / ".hydra/config.yaml")
        self.policy = validate_imagenet_config(cfg)
        require(Path(cfg.paths.output_dir).resolve() == self.output.resolve(), "Smoke output/config path differs")
        self.network, self.partition = _network_contract(pl_module)
        self.names = [name for name, _ in pl_module.named_parameters()]
        self.usage = {name: 0 for name in self.names}
        usages = [cb for cb in trainer.callbacks if isinstance(cb, ConditionalOptimizerUsage)]
        require(len(usages) == 1, "Actual trainer needs exactly one conditional usage callback")
        self.usage_callback = usages[0]
        data = trainer.datamodule
        require(len(data.datasets["val"]) == 1000 and len(data.val_dataloader()) == 250
                and data.context.fingerprint() == pl_module.data_context.fingerprint(),
                "Smoke must retain the complete production1000-anchor context")
        self.optimizer_lrs, self.fp32_losses, self.bf16_outputs = [], 0, 0
        self.validation_indices, self.training_sample_ids = [], []
        self.initial_inactive = _inactive_digest(pl_module, self.partition)
        self.initial_weights = weights_digest(pl_module.state_dict())
        initial = [None] * 4
        torch.distributed.all_gather_object(initial, self.initial_weights)
        require(len(set(initial)) == 1, "Fresh ImageNet DDP weights differ")
        self.all_initial_weights = initial
        verify_scheduler(self.policy, [c.scheduler.state_dict() for c in trainer.lr_scheduler_configs], 0)
        require(all(math.isclose(g["lr"], update_lr(self.policy, 0), rel_tol=1e-12)
                    for g in trainer.optimizers[0].param_groups), "Wrong first ImageNet optimizer LR")
        def observe(module, args, output):
            require(args[1] == "imagenet" and set(output) == {"image"}
                    and all(v.dtype == torch.bfloat16 and torch.isfinite(v).all() for v in output.values()),
                    "Actual ImageNet model output must be finite BF16 image only")
            if pl_module.training and torch.is_grad_enabled(): self.bf16_outputs += 1
        self.hook = self.network.register_forward_hook(observe)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        require(set(batch) == {"imagenet"}, "Only ImageNet may enter this dedicated smoke")
        _validate_sample(batch["imagenet"], batch_size=8)
        if len(self.training_sample_ids) < 3:
            self.training_sample_ids.append([m["sample_id"] for m in batch["imagenet"]["pixel_metadata"]])

    def on_before_backward(self, trainer, pl_module, loss):
        require(loss.dtype == torch.float32 and torch.isfinite(loss), "ImageNet loss must be finite FP32")
        self.fp32_losses += 1

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        rate = update_lr(self.policy, int(trainer.global_step))
        require(all(math.isclose(g["lr"], rate, rel_tol=1e-12) for g in optimizer.param_groups),
                "Actual ImageNet optimizer LR deviates from pinned warmup")
        for name, parameter in pl_module.named_parameters():
            if self.partition[name] == "inactive":
                require(parameter.grad is None and not optimizer.state.get(parameter),
                        "Inactive vector/language/identity bank received gradient or Adam state")
            else:
                require(parameter.grad is not None and parameter.grad.dtype == torch.float32
                        and torch.isfinite(parameter.grad).all(), "Active ImageNet gradient must be finite FP32")
                self.usage[name] += 1
        self.optimizer_lrs.append(rate)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step in (10, 50):
            torch.cuda.synchronize(pl_module.device); torch.distributed.barrier()
            if trainer.global_step == 10:
                torch.cuda.reset_peak_memory_stats(pl_module.device)
                self.warm_start = time.perf_counter()
            else:
                self.warm_wall_seconds = time.perf_counter() - self.warm_start
                self.peak_allocated = torch.cuda.max_memory_allocated(pl_module.device)
                self.peak_reserved = torch.cuda.max_memory_reserved(pl_module.device)

    def on_validation_start(self, trainer, pl_module):
        require(trainer.global_step == 50 and not trainer.sanity_checking,
                "Only real scheduled validation after50 updates is admitted")
        self.validation_start = time.perf_counter()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        require(dataloader_idx == 0 and set(batch) == {"imagenet"}, "Wrong validation source/loader")
        sample = batch["imagenet"]
        _validate_sample(sample, batch_size=1)
        indices = sample.get("validation_batch_index")
        require(isinstance(indices, list) and len(indices) == 1 and type(indices[0]) is int,
                "Actual global validation anchor identity is missing")
        self.validation_indices.append(indices[0])
        if trainer.global_rank == 0 and batch_idx == 0:
            hidden = _cpu(sample)
            hidden["world"]["image"]["value"].fill_(float("nan"))
            _validate_sample(hidden, batch_size=1, hidden=True)
            _torch_once(self.output / "sample-input-imagenet.pt", hidden)

    def on_train_end(self, trainer, pl_module):
        self.hook.remove()
        require(trainer.global_step == self.fp32_losses == self.bf16_outputs == 50
                and len(self.optimizer_lrs) == 50 and trainer.datamodule.consumed_batches == 50,
                "Incomplete real50-update optimization/precision/cursor evidence")
        rank = trainer.global_rank
        require(self.validation_indices == list(range(rank * 250, (rank + 1) * 250)),
                "Each rank must consume its exact250 canonical anchors once")
        evaluator = pl_module.evaluator
        require(evaluator.has_completed_validation(50) and evaluator.global_validation_batches == 1000
                and evaluator._validation_batches == 250, "All1000 scheduled generation anchors must finish")
        metrics = {str(k): float(v) for k, v in trainer.callback_metrics.items() if str(k).startswith("val/")}
        require(set(metrics) == {METRIC} and math.isfinite(metrics[METRIC]), "Missing finite generation metric")
        check_parameter_updates(self.names, self.partition, self.usage)
        require(self.usage_callback.counts == self.usage, "Maintained conditional usage proof disagrees")
        conditional_record(trainer.optimizers[0], pl_module.named_parameters(), self.usage, 50)
        final_inactive = _inactive_digest(pl_module, self.partition)
        require(final_inactive == self.initial_inactive, "Inactive banks changed despite absent image interfaces")
        final = weights_digest(pl_module.state_dict())
        weights = [None] * 4
        torch.distributed.all_gather_object(weights, final)
        require(len(set(weights)) == 1 and final != self.initial_weights, "ImageNet ranks did not optimize identically")
        checkpoint_name = f"smoke-epoch{trainer.current_epoch:04d}-step000000050.ckpt"
        require(not (self.output / checkpoint_name).exists(), "Diagnostic checkpoint is immutable")
        # Every rank calls the maintained save: its RNG gather is never rank0-only.
        trainer.save_checkpoint(self.output / checkpoint_name)
        if rank == 0:
            with (self.output / "smoke-config.yaml").open("x") as stream:
                stream.write(OmegaConf.to_yaml(pl_module.hparams.config_tree, resolve=True))
                stream.flush(); os.fsync(stream.fileno())
        torch.distributed.barrier()
        record = dict(schema="imagenet-joint-fit-v1", status="FIT_PASS", job_id=os.environ["SLURM_JOB_ID"],
            rank=rank, world_size=4, global_step=50, rank_batch_size=8, global_batch_size=32,
            distributed_backend=torch.distributed.get_backend(), trainer_precision=trainer.precision,
            validation_batches=1000, local_validation_batches=250,
            validation_batch_indices=self.validation_indices, parameters=TOTAL_PARAMETERS,
            executed_parameters=EXECUTED_PARAMETERS, expert_parameters=self.network.expert_parameter_counts(),
            weights_sha256=final, all_rank_weights=weights, initial_weights_sha256=self.initial_weights,
            all_rank_initial_weights=self.all_initial_weights, inactive_initial_sha256=self.initial_inactive,
            inactive_final_sha256=final_inactive, inactive_banks_unchanged=True,
            parameter_names=self.names, parameter_partition=self.partition, parameter_update_counts=self.usage,
            bf16_training_outputs=self.bf16_outputs, fp32_losses=self.fp32_losses,
            fp32_gradients_verified=True, all_training_bf16_outputs=True,
            optimizer_update_lrs=self.optimizer_lrs, lr_policy=self.policy, all50_lr_verified=True,
            metrics=metrics, data_context_sha256=pl_module.data_context.fingerprint(), checkpoint_name=checkpoint_name,
            training_sample_ids=self.training_sample_ids, language_conditioning_verified=True,
            active_streams=["image"], attention_backend="flash4", token_execution="packed_valid",
            compile_mode="default", gpu_uuid=str(self.props.uuid), gpu_name=self.props.name,
            gpu_capability=list(torch.cuda.get_device_capability(pl_module.device)),
            measured_updates=40, warm_wall_seconds=self.warm_wall_seconds,
            warm_timing_interval="after_completed_update10_to_after_completed_update50; excludes_scheduled_validation",
            peak_allocated_bytes=self.peak_allocated, peak_reserved_bytes=self.peak_reserved,
            validation_checkpoint_endcheck_seconds=time.perf_counter()-self.validation_start,
            validation_checkpoint_endcheck_interval="scheduled_validation_start_through_diagnostic_checkpoint_and_train_end_checks; composite_not_validation_only")
        _json_once(self.output / f"fit-rank{rank}.json", record)


def validate_imagenet_checkpoint(checkpoint, cfg, model_config, *, expected_step,
                                 expected_context_sha256):
    """Allocated strict reload reusable at50 or a scheduled production step.

    The caller supplies real checkpoint bytes and the immutable target step.
    This function does not select a recovery checkpoint or authorize a run.
    It returns a strictly restored wrapper/data/optimizer for native sampling.
    """
    _gpu_runtime(torch.device("cuda", 0), distributed=False)
    require(type(expected_step) is int and expected_step > 0
            and (expected_step == 50 or expected_step % 20000 == 0 and expected_step <= 300000),
            "Only diagnostic50 or actual scheduled production checkpoints are supported")
    policy = validate_imagenet_config(cfg, smoke=expected_step == 50)
    require(checkpoint.get("global_step") == expected_step
            and isinstance(expected_context_sha256, str) and _HEX.fullmatch(expected_context_sha256),
            "Checkpoint step/context binding differs")
    state = checkpoint.get("state_dict")
    require(isinstance(state, dict) and state and all(torch.is_tensor(t) and t.dtype == torch.float32
            and torch.isfinite(t).all() for t in state.values()), "Checkpoint model must be complete finite FP32")
    optimizers, schedulers = checkpoint.get("optimizer_states", []), checkpoint.get("lr_schedulers", [])
    require(len(optimizers) == 1 and optimizers[0].get("state"), "Missing actual full AdamW state")
    verify_scheduler(policy, schedulers, expected_step)
    require(all(math.isclose(group["lr"], update_lr(policy, expected_step), rel_tol=1e-12)
                and group.get("initial_lr") == policy["peak_lr"] for group in optimizers[0]["param_groups"]),
            "Wrong next-update LR/optimizer peak")
    usage = checkpoint.get("native_conditional_optimizer_usage")
    require(isinstance(usage, dict), "Checkpoint lacks maintained conditional optimizer usage")
    check_conditional_adam_state(optimizers[0], usage, expected_step)
    ranks = validate_rng_payload(checkpoint.get("mot_rng_state"))
    data = instantiate(cfg.data)
    context = data.prepare_context(mode="eval", normalization=None, normalizer=None,
                                   restored_state=checkpoint.get("data_context"))
    require(context.fingerprint() == expected_context_sha256 and len(data.datasets["val"]) == 1000,
            "Strict restored ImageNet data/anchor context differs")
    cursor = checkpoint.get("ImageNetImageFolderDataModule")
    require(isinstance(cursor, dict) and set(cursor) == {"schema", "consumed_batches", "data_context_sha256"}
            and cursor.get("schema") == "imagenet-source-draw-state-v1"
            and cursor.get("consumed_batches") == expected_step
            and cursor.get("data_context_sha256") == expected_context_sha256,
            "Missing exact ImageNet consumed-draw checkpoint state")
    data.load_state_dict(cursor)
    require(data.state_dict() == cursor, "Strict ImageNet sampler restoration differs")
    validate_checkpoint_binding(checkpoint, model_config, context)
    with checkpoint_construction():
        wrapper = ModelWrapper(config_tree=model_config, enable_grad_norm=False).cuda().eval()
        context.bind(wrapper.model)
    wrapper.data_context = context
    wrapper.load_state_dict(state, strict=True)
    network, partition = _network_contract(wrapper)
    names = [name for name, _ in wrapper.named_parameters()]
    require(usage.get("parameter_names") == names, "Conditional checkpoint parameter order differs")
    check_parameter_updates(names, partition, usage["parameter_update_counts"], expected_step)
    optimizer = instantiate(cfg.model.optimizer, params=wrapper.parameters())
    if callable(optimizer): optimizer = optimizer()
    scheduler = instantiate(cfg.model.scheduler, optimizer=optimizer)
    if callable(scheduler): scheduler = scheduler()
    optimizer.load_state_dict(optimizers[0]); scheduler.load_state_dict(schedulers[0])
    require(state_digest(optimizer.state_dict()) == state_digest(optimizers[0])
            and scheduler.state_dict() == schedulers[0]
            and all(math.isclose(g["lr"], update_lr(policy, expected_step), rel_tol=1e-12)
                    for g in optimizer.param_groups), "Strict AdamW/scheduler reload changed state")
    actual_usage = conditional_record(optimizer, wrapper.named_parameters(), usage["parameter_update_counts"], expected_step)
    require(actual_usage == usage, "Strict conditional optimizer reload differs")
    # Validate that each actual payload can be restored, without advancing or
    # replacing the scientific training RNG. This is not a resumed train step.
    from egomimic.pl_utils.world_model_behavior import _capture_rng_state, _restore_rng_state
    restored_rng = []
    with preserved_rng():
        for rank, saved in enumerate(ranks):
            _restore_rng_state(saved)
            actual = _capture_rng_state(current_cuda_only=True)
            require(state_digest(actual) == state_digest(saved), "Saved rank RNG payload does not restore exactly")
            restored_rng.append(dict(rank=rank, sha256=state_digest(actual)))
    proof = dict(parameters=TOTAL_PARAMETERS, executed_parameters=EXECUTED_PARAMETERS,
        expert_parameters=network.expert_parameter_counts(), parameter_names=names,
        parameter_update_counts=usage["parameter_update_counts"], parameter_partition=partition,
        data_context_sha256=context.fingerprint(), strict_full_wrapper_reload=True,
        optimizer_sampler_all_rank_rng_verified=True, lr_scheduler_strictly_restored=True,
        fp32_stored_model_and_adam=True, all_rank_rng_payload_restorable=True,
        all_rank_rng_payload_restoration=restored_rng, lr_policy=policy,
        next_optimizer_lr=update_lr(policy, expected_step),
        inactive_weights_sha256=_inactive_digest(wrapper, partition))
    return wrapper, data, optimizer, proof


def sample_imagenet_hidden(wrapper, sample):
    """Actual20-step native inference with no clean RGB target access."""
    _validate_sample(sample, batch_size=1, hidden=True)
    network = _network(wrapper)
    sampler = wrapper.model.pipeline.stage_by_id("joint_sampler").sampler
    require(sampler.state_steps == 20 and sampler.action_steps == 20, "Wrong maintained native integration schedule")
    calls = []
    def observe(module, args, output):
        require(args[1] == "imagenet" and set(args[0]) == set(output) == {"image"}
                and all(torch.isfinite(value).all() for value in output.values()),
                "Hidden-target generation returned an invalid stream")
        calls.append(args[3]["image"].detach().clone())
    hook = network.register_forward_hook(observe)
    wrapper.model.device = next(wrapper.parameters()).device
    groups = wrapper.model.process_batch_for_training({"imagenet": sample})
    try:
        with preserved_rng(), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            torch.manual_seed(42)
            torch.cuda.synchronize(); started = time.perf_counter()
            generated = wrapper.model.forward_eval(groups)["imagenet"]["image_proprio_prediction"]
            torch.cuda.synchronize(); elapsed = time.perf_counter() - started
        require(len(calls) == 20 and all(torch.allclose(level, torch.full_like(level, (20-index)/20))
                for index, level in enumerate(calls)), "Native sample did not execute exactly20descending steps")
        require(set(generated) == {"image"} and generated["image"].shape == (1, 100, 768)
                and generated["image"].dtype == torch.float32 and torch.isfinite(generated["image"]).all(),
                "Every hidden-target generated RGB value must be finite FP32")
        return dict(sample_state_steps=20, sample_action_steps=0, sample_finite=True,
                    actual_denoiser_calls=20, sample_seconds=elapsed,
                    hidden_rgb_values_all_nan=True, genuine_finite_language_retained=True)
    finally:
        hook.remove()


def verify_imagenet_smoke(output, *, job_id=None):
    """Allocated postflight; local fixture receipts cannot pass this entry point."""
    _gpu_runtime(torch.device("cuda", 0), distributed=False)
    require(os.environ.get("RANK", "0") == "0", "Run postflight once on the allocated rank0 process")
    root = Path(output)
    job = os.environ["SLURM_JOB_ID"]
    require(job_id is None or job_id == job, "Postflight belongs to another actual allocation")
    require(not (root / "smoke-completion.json").exists(), "Smoke completion is immutable")
    cfg, model_config = load_smoke_configs(root)
    policy = validate_imagenet_config(cfg)
    rows = [_read_json(root / f"fit-rank{rank}.json") for rank in range(4)]
    first = validate_fit_rows(rows, policy, job)
    checkpoint_path = root / first["checkpoint_name"]
    require(checkpoint_path.is_file() and not checkpoint_path.is_symlink(), "Missing own immutable diagnostic checkpoint")
    checkpoint_sha = _sha(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu", mmap=True)
    require(weights_digest(checkpoint["state_dict"]) == first["weights_sha256"],
            "Diagnostic checkpoint differs from all actual optimized ranks")
    wrapper, data, optimizer, proof = validate_imagenet_checkpoint(checkpoint, cfg, model_config,
        expected_step=50, expected_context_sha256=first["data_context_sha256"])
    require(proof["parameter_names"] == first["parameter_names"]
            and proof["parameter_partition"] == first["parameter_partition"]
            and proof["parameter_update_counts"] == first["parameter_update_counts"]
            and proof["inactive_weights_sha256"] == first["inactive_final_sha256"]
            and proof["expert_parameters"] == first["expert_parameters"],
            "Strict checkpoint differs from actual conditional/inactive rank proofs")
    sample_path = root / "sample-input-imagenet.pt"
    require(sample_path.is_file() and not sample_path.is_symlink(), "Missing actual hidden validation sample")
    sample = torch.load(sample_path, weights_only=False, map_location="cpu")
    _validate_sample(sample, batch_size=1, hidden=True)
    # Reopen the real first frozen anchor and class cache: the saved hidden
    # sample may differ only by its deliberately removed RGB target values.
    from egomimic.rldb.imagenet_imagefolder import collate_imagenet_images
    from egomimic.rldb.pixel_proprio import PixelValidationIndex
    expected = collate_imagenet_images([data.datasets["val"][PixelValidationIndex(0, 0)]])["imagenet"]
    expected["world"]["image"]["value"].fill_(float("nan"))
    require(state_digest(sample) == state_digest(expected), "Hidden sample lost actual anchor/class-language provenance")
    sampled = sample_imagenet_hidden(wrapper, sample)
    require(_sha(checkpoint_path) == checkpoint_sha, "Diagnostic checkpoint changed during postflight")
    receipt = dict(schema="imagenet-native-smoke-v1", status="PASS", job_id=job, world_size=4,
        global_step=50, rank_batch_size=8, global_batch_size=32, distributed_backend="nccl", trainer_precision="bf16-mixed",
        validation_batches=1000, local_validation_batches=250,
        **proof, **sampled, all50_lr_verified=True, optimizer_update_lrs=first["optimizer_update_lrs"],
        all_rank_weights_agree=True, all_training_bf16_outputs=True, language_conditioning_verified=True,
        inactive_banks_unchanged=True, metrics=first["metrics"], active_streams=["image"],
        checkpoint=dict(name=checkpoint_path.name, path=str(checkpoint_path.resolve()), sha256=checkpoint_sha,
                        bytes=checkpoint_path.stat().st_size),
        sample_input_sha256=_sha(sample_path), measured_updates=40,
        warm_wall_seconds_by_rank=[row["warm_wall_seconds"] for row in rows],
        warm_wall_seconds=max(row["warm_wall_seconds"] for row in rows),
        peak_allocated_bytes=max(row["peak_allocated_bytes"] for row in rows),
        peak_reserved_bytes=max(row["peak_reserved_bytes"] for row in rows),
        validation_checkpoint_endcheck_seconds=max(row["validation_checkpoint_endcheck_seconds"] for row in rows),
        validation_checkpoint_endcheck_interval="scheduled_validation_start_through_diagnostic_checkpoint_and_train_end_checks; composite_not_validation_only",
        attention_backend="flash4", token_execution="packed_valid", compile_mode="default", ranks=rows,
        diagnostic_checkpoint_is_production_initializer=False)
    _json_once(root / "smoke-completion.json", receipt)
    return receipt


def validate_imagenet_capacity(config_path, output, *, source_manifest_sha256, data_manifest_sha256):
    """One actual rank8/T1/C1/P100 compiled backward; zero optimizer updates."""
    from torch.nn.parallel import DistributedDataParallel
    from egomimic.rldb.imagenet_imagefolder import collate_imagenet_images
    from egomimic.rldb.pixel_proprio import DistributedPixelTrainingBatchSampler
    require(os.environ.get("SLURM_JOB_ID", "").isdecimal(), "Capacity requires an actual allocated job")
    world, rank, local = (int(os.environ[key]) for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK"))
    require(world == 4 and 0 <= rank < 4 and 0 <= local < 4 and torch.cuda.device_count() == 4,
            "ImageNet capacity requires four distinct visible rank devices")
    torch.cuda.set_device(local)
    created = not torch.distributed.is_initialized()
    if created: torch.distributed.init_process_group("nccl")
    props = _gpu_runtime(torch.device("cuda", local), distributed=True)
    destination = Path(output)
    require(not destination.exists(), "Capacity proof is immutable")
    require(isinstance(source_manifest_sha256, str) and _HEX.fullmatch(source_manifest_sha256),
            "Capacity requires exact staged source-manifest binding")
    cfg = OmegaConf.load(config_path)
    validate_imagenet_config(cfg)
    require(data_manifest_sha256 == cfg.data.manifest_sha256 == _sha(cfg.data.manifest_path),
            "Actual capacity data manifest differs from request/config")
    data = instantiate(cfg.data)
    context = data.prepare_context(mode="train", normalization=None, normalizer=None)
    draws = next(iter(DistributedPixelTrainingBatchSampler(data.datasets["train"], 8, 42, world_size=4, rank=rank)))
    sample = collate_imagenet_images([data.datasets["train"][draw] for draw in draws])
    _validate_sample(sample["imagenet"], batch_size=8)
    torch.manual_seed(42)
    wrapper = ModelWrapper(config_tree=dict(model=_plain(cfg.model)), enable_grad_norm=False).cuda()
    context.bind(wrapper.model); wrapper.data_context = context
    network, partition = _network_contract(wrapper)
    class CapacityPass(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.wrapper = wrapper
        def forward(self, batch):
            predicted = self.wrapper.model.forward_training(batch)
            require(set(predicted) == {"imagenet"}
                    and set(predicted["imagenet"]["joint/prediction"]) == {"image"}
                    and predicted["imagenet"]["joint/prediction"]["image"].dtype == torch.bfloat16,
                    "Actual capacity prediction must execute image-only BF16")
            return self.wrapper.model.compute_losses(predicted, batch)["loss"]
    wrapped = DistributedDataParallel(CapacityPass(), device_ids=[local], find_unused_parameters=True)
    batch = wrapper.model.process_batch_for_training(sample)
    free_before, total = torch.cuda.mem_get_info(local)
    reserve_bytes = _optimizer_capacity_reserve_bytes(wrapper.parameters())
    reserve = torch.empty(reserve_bytes // 4, dtype=torch.float32, device=f"cuda:{local}")
    reserve.zero_()
    torch.cuda.reset_peak_memory_stats(local); started = time.perf_counter()
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16): loss = wrapped(batch)
        require(loss.dtype == torch.float32 and torch.isfinite(loss), "Capacity loss must be finite FP32")
        loss.backward()
        for name, parameter in wrapper.named_parameters():
            require((parameter.grad is None) if partition[name] == "inactive" else
                    (parameter.grad is not None and parameter.grad.dtype == torch.float32
                     and torch.isfinite(parameter.grad).all()), "Capacity active/inactive gradient contract differs")
        torch.cuda.synchronize(local)
        free_after, _ = torch.cuda.mem_get_info(local)
        record = dict(schema="imagenet-native-capacity-v1", status="PASS", job_id=os.environ["SLURM_JOB_ID"],
            rank=rank, world_size=4, rank_batch_size=8, global_batch_size=32, frames=1, cameras=1,
            patches=100, image_tokens_per_sample=100, image_size=[160, 160], patch_size=16,
            parameters=TOTAL_PARAMETERS, executed_parameters=EXECUTED_PARAMETERS,
            expert_parameters=network.expert_parameter_counts(), data_context_sha256=context.fingerprint(),
            source_manifest_sha256=source_manifest_sha256, data_manifest_sha256=data_manifest_sha256,
            config_sha256=_sha(config_path), optimizer_updates=0, compiled_backward_fp32finite=True,
            attention_backend="flash4", token_execution="packed_valid", compile_mode="default", compile_dynamic=True,
            bf16_forward=True, fp32_weights_gradients=True, inactive_gradients_absent=True,
            gpu_uuid=str(props.uuid), gpu_name=props.name, gpu_capability=list(torch.cuda.get_device_capability(local)),
            optimizer_memory_reserve_bytes=reserve_bytes,
            optimizer_memory_reserve_policy="2FP32AdamMoments+1FP32ForeachWorkspace; memory_only_no_optimizer_state",
            free_before_capacity_bytes=free_before, fit_free_bytes=free_after, device_memory_bytes=total,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(local),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(local), seconds=time.perf_counter()-started,
            loss=float(loss.detach()), actual_source_keys=[m["sample_id"] for m in sample["imagenet"]["pixel_metadata"]])
        destination.parent.mkdir(parents=True, exist_ok=True)
        _json_once(destination.parent / f"{destination.stem}-rank{rank}.json", record)
        rows = [None] * 4; torch.distributed.all_gather_object(rows, record)
        require(len({row["gpu_uuid"] for row in rows}) == 4
                and all(row["data_context_sha256"] == context.fingerprint() for row in rows),
                "Capacity needs distinct owned GPUs and identical full contexts")
        if rank == 0:
            result = dict(record, ranks=rows, all_rank_gpu_uuids=[row["gpu_uuid"] for row in rows],
                          peak_allocated_bytes=max(row["peak_allocated_bytes"] for row in rows),
                          peak_reserved_bytes=max(row["peak_reserved_bytes"] for row in rows),
                          fit_free_bytes=min(row["fit_free_bytes"] for row in rows))
            _json_once(destination, result)
            return result
    finally:
        del reserve
        if created: torch.distributed.destroy_process_group()
