#!/usr/bin/env python3
"""Read-only preflight for the executable Action Flow experiment family.

This tool composes Hydra and instantiates the real pipeline on CPU.  It does
not construct a data module, read episode data, start Lightning, initialize an
optimizer, load a checkpoint, or contact W&B.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch.nn as nn
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

# Direct ``python tools/...`` execution otherwise places only ``tools/`` on
# sys.path. Pin the repository root before importing the package under audit.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from egomimic.eval.energy_score import (  # noqa: E402
    USOCKET_ENERGY_DISTANCE_CONFIG,
    normalize_usocket_energy_distance_config,
)
from egomimic.eval.planar_action_eval import (  # noqa: E402
    USOCKET_NATIVE_ERROR_CONFIG,
    normalize_usocket_native_error_config,
)
from egomimic.models.action_flow_codec import (  # noqa: E402
    ContextFreeSequenceDecoder,
    ContextFreeSequenceEncoder,
)
from egomimic.models.action_flow_transformer import AdaLNSequenceField  # noqa: E402
from egomimic.pipeline.algo import PipelineAlgo  # noqa: E402
from egomimic.pipeline.stages_action_flow import (  # noqa: E402
    ActionFlowObjectiveStage,
    ConditionalVelocityStage,
    ContentDecoderStage,
    ContentEncoderStage,
    LatentBridgeStage,
)
from egomimic.pipeline.stages_io import ActionTargetBuilder  # noqa: E402
from egomimic.pipeline.stages_sampler import (  # noqa: E402
    FusedObsEncoder,
    GaussianLatentNoise,
)
from egomimic.rldb.zarr.content_manifest import (  # noqa: E402
    validate_content_manifest,
)

SCHEMA_VERSION = 1
DEFAULT_CONFIG_ROOT = REPOSITORY_ROOT / "egomimic" / "hydra_configs"
CANONICAL_CONTENT_MANIFEST_RELATIVE_PATH = Path(
    "egomimic/hydra_configs/data/pusht/manifests/usocket_3000_v2_clean_content_v1.json"
)
EXPECTED_STAGE_TYPES = (
    FusedObsEncoder,
    GaussianLatentNoise,
    ActionTargetBuilder,
    ContentEncoderStage,
    LatentBridgeStage,
    ConditionalVelocityStage,
    ContentDecoderStage,
    ActionFlowObjectiveStage,
)
EXPECTED_STAGE_TARGETS = tuple(
    f"{stage_type.__module__}.{stage_type.__name__}"
    for stage_type in EXPECTED_STAGE_TYPES
)
EXPECTED_INFERENCE_TYPES = (
    FusedObsEncoder,
    GaussianLatentNoise,
    ConditionalVelocityStage,
    ContentDecoderStage,
)
FORBIDDEN_PIPELINE_KEYS = frozenset(
    {
        "ac_keys",
        "action_keys",
        "domain",
        "domains",
        "embodiment",
        "embodiments",
        "robot",
        "robots",
    }
)
FORBIDDEN_IMPLEMENTATION_TOKENS = (
    "unite",
    "crosstransformer",
    "cross_transformer",
    "diffusionnoisingstage",
)
ALLOWED_PAIR_DIFFERENCES = frozenset(
    {
        "description",
        "model.pipeline.stages.7.reconstruction_weight",
        "model.reconstruction_weight",
        "name",
        "run_provenance.objective.reconstruction_weight",
    }
)


class PreflightError(RuntimeError):
    """Raised when an executable config violates the family contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightError(message)


def _exact(actual: Any, expected: Any, label: str) -> None:
    _require(actual == expected, f"{label}: expected {expected!r}, got {actual!r}")


def _float(actual: Any, expected: float, label: str) -> None:
    value = float(actual)
    _require(
        math.isfinite(value)
        and math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12),
        f"{label}: expected {expected!r}, got {actual!r}",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compose_experiment(
    experiment: str,
    *,
    config_root: Path = DEFAULT_CONFIG_ROOT,
    overrides: Sequence[str] = (),
) -> DictConfig:
    """Compose one experiment without entering a Hydra run directory."""

    experiment = str(experiment)
    _require(bool(experiment), "experiment must be non-empty")
    config_root = Path(config_root).resolve()
    _require(config_root.is_dir(), f"config root does not exist: {config_root}")
    selected_overrides = [
        f"+experiment={experiment}",
        "++paths.root_dir=.",
        *map(str, overrides),
    ]
    with initialize_config_dir(
        version_base=None,
        config_dir=str(config_root),
    ):
        return compose(
            config_name="train_zarr_cartesian",
            overrides=selected_overrides,
            return_hydra_config=True,
        )


def resolved_config_payload(config: DictConfig) -> tuple[dict[str, Any], str]:
    """Resolve a config with stable sentinels for Hydra runtime-only values."""

    _require("hydra" in config, "composed config must include Hydra metadata")
    with open_dict(config.hydra.runtime):
        config.hydra.runtime.cwd = "<RUNTIME_CWD>"
        config.hydra.runtime.output_dir = "<RUNTIME_OUTPUT_DIR>"
    HydraConfig.instance().set_config(config)

    selected = OmegaConf.masked_copy(
        config, [key for key in config.keys() if key != "hydra"]
    )
    if OmegaConf.select(selected, "logger.wandb.id") is not None:
        with open_dict(selected.logger.wandb):
            selected.logger.wandb.id = "<RUNTIME_RUN_ID>"
    payload = OmegaConf.to_container(selected, resolve=True)
    _require(isinstance(payload, dict), "resolved config must be a mapping")
    return payload, _json_sha256(payload)


def _walk(value: Any, path: tuple[str, ...] = ()):
    if isinstance(value, Mapping):
        for key, item in value.items():
            key = str(key)
            yield path + (key,), key, item
            yield from _walk(item, path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, path + (str(index),))


def _validate_generic_surface(config: DictConfig, pipeline: nn.Module) -> None:
    raw = OmegaConf.to_container(config.model.pipeline, resolve=True)
    for path, key, value in _walk(raw):
        _require(
            key.lower() not in FORBIDDEN_PIPELINE_KEYS,
            f"pipeline config contains forbidden key {'.'.join(path)}",
        )
        if isinstance(value, str):
            compact = value.lower().replace("-", "").replace("_", "")
            for token in FORBIDDEN_IMPLEMENTATION_TOKENS:
                normalized = token.replace("_", "")
                _require(
                    normalized not in compact,
                    f"pipeline config contains forbidden implementation {value!r}",
                )

    parameters = tuple(inspect.signature(PipelineAlgo.__init__).parameters)
    _exact(parameters, ("self", "stages", "device"), "PipelineAlgo constructor")
    for module in pipeline.modules():
        qualified = f"{type(module).__module__}.{type(module).__name__}".lower()
        for token in FORBIDDEN_IMPLEMENTATION_TOKENS:
            _require(
                token.replace("_", "") not in qualified.replace("_", ""),
                f"pipeline instantiated forbidden module {qualified}",
            )


def _parameter_manifest(module: nn.Module) -> dict[str, Any]:
    entries = [
        {
            "dtype": str(parameter.dtype),
            "name": name,
            "numel": int(parameter.numel()),
            "requires_grad": bool(parameter.requires_grad),
            "shape": list(parameter.shape),
        }
        for name, parameter in module.named_parameters(remove_duplicate=True)
    ]
    names = [entry["name"] for entry in entries]
    _require(
        len(names) == len(set(names)), "parameter manifest contains duplicate names"
    )
    manifest_sha256 = _json_sha256(entries)
    return {
        "entries": entries,
        "manifest_sha256": manifest_sha256,
        "total": sum(entry["numel"] for entry in entries),
        "trainable": sum(entry["numel"] for entry in entries if entry["requires_grad"]),
    }


def _validate_dimensions_and_modules(
    config: DictConfig,
    stages: Sequence[nn.Module],
) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    _exact(int(config.model.action_horizon), 16, "model action horizon")
    _exact(int(config.model.action_dim), 4, "model action dimension")
    _exact(int(config.model.latent_dim), 8, "model latent dimension")
    _exact(int(config.model.condition_dim), 67, "model condition dimension")

    observation = stages[0]
    noise = stages[1]
    encoder_stage = stages[3]
    bridge = stages[4]
    field_stage = stages[5]
    decoder_stage = stages[6]
    objective = stages[7]
    _require(isinstance(encoder_stage.encoder, ContextFreeSequenceEncoder), "wrong E")
    _require(isinstance(field_stage.field, AdaLNSequenceField), "wrong field v")
    _require(isinstance(decoder_stage.decoder, ContextFreeSequenceDecoder), "wrong g")

    encoder = encoder_stage.encoder
    decoder = decoder_stage.decoder
    field = field_stage.field
    codec_expected = {
        "horizon": 16,
        "hidden_dim": 20,
        "depth": 2,
        "num_heads": 4,
        "feedforward_dim": 80,
    }
    for label, codec in (("encoder E", encoder), ("decoder g", decoder)):
        for attribute, expected in codec_expected.items():
            _exact(getattr(codec, attribute), expected, f"{label} {attribute}")
        _float(codec.dropout, 0.0, f"{label} dropout")
    _exact(encoder.input_dim, 4, "encoder E input dimension")
    _exact(encoder.output_dim, 8, "encoder E output dimension")
    _exact(decoder.input_dim, 8, "decoder g input dimension")
    _exact(decoder.output_dim, 4, "decoder g output dimension")
    _exact(
        tuple(inspect.signature(encoder.forward).parameters),
        ("content",),
        "encoder E context-free forward signature",
    )
    _exact(
        tuple(inspect.signature(decoder.forward).parameters),
        ("content",),
        "decoder g context-free forward signature",
    )

    field_expected = {
        "input_dim": 8,
        "output_dim": 8,
        "horizon": 16,
        "condition_dim": 67,
        "hidden_dim": 512,
        "depth": 12,
        "num_heads": 8,
        "feedforward_dim": 2048,
        "time_embedding_dim": 512,
    }
    for attribute, expected in field_expected.items():
        _exact(getattr(field, attribute), expected, f"field v {attribute}")
    _float(field.time_scale, 1_000.0, "field v normalized-time scale")
    _float(field.dropout, 0.0, "field v dropout")
    _float(
        field.condition_dropout_probability,
        0.3,
        "field v condition dropout",
    )

    field_config = config.model.pipeline.stages[5].field
    _require(
        "time_scale" in field_config,
        "field time_scale must be explicit in the resolved model config",
    )
    _float(field_config.time_scale, 1_000.0, "configured field time_scale")
    _exact(noise.num_tokens, 16, "Gaussian source horizon")
    _exact(noise.latent_dim, 8, "Gaussian source latent dimension")
    _exact(observation.n_obs_steps, 1, "observation steps")

    low_dim_width = sum(
        int(spec["input_dim"]) for spec in observation.encoder.obs_specs.values()
    )
    image_width = sum(
        int(getattr(module, "feature_dimension"))
        for module in observation.encoder.img_encoders.values()
    )
    _exact(low_dim_width, 3, "observation low-dimensional width")
    _exact(image_width, 64, "observation image-feature width")
    _exact(low_dim_width + image_width, 67, "observation condition width")

    _exact(bridge.samples_per_content, 14, "bridge samples per content")
    _float(bridge.condition_dropout_probability, 0.3, "bridge condition dropout")
    _exact(field_stage.num_inference_steps, 16, "inference field evaluations")
    _float(objective.flow_weight, 1.0, "FM weight")
    _float(objective.action_velocity_weight, 1.0, "action-velocity weight")
    _float(
        objective.reconstruction_weight,
        float(config.model.reconstruction_weight),
        "reconstruction weight",
    )
    _require(
        float(config.model.reconstruction_weight) in {1.0, 10.0, 100.0},
        "reconstruction weight must be exactly 1, 10, or 100",
    )

    parameters = {
        "observation_encoder": _parameter_manifest(observation),
        "encoder_e": _parameter_manifest(encoder),
        "field_v": _parameter_manifest(field),
        "decoder_g": _parameter_manifest(decoder),
    }
    _require(parameters["encoder_e"]["total"] <= 12_000, "E exceeds 12K")
    _require(parameters["decoder_g"]["total"] <= 12_000, "g exceeds 12K")
    _require(
        39_000_000 <= parameters["field_v"]["total"] <= 41_000_000,
        "field v must contain 39M to 41M parameters",
    )
    for label, counts in parameters.items():
        _require(counts["total"] > 0, f"{label} has no parameters")
        _exact(counts["trainable"], counts["total"], f"{label} trainable count")

    dimensions = {
        "action": [16, 4],
        "condition": 67,
        "image_feature": 64,
        "latent": [16, 8],
        "normalized_state": 3,
    }
    return dimensions, parameters


def _validate_topology(
    config: DictConfig,
    pipeline_algo: PipelineAlgo,
) -> dict[str, Any]:
    stage_configs = tuple(config.model.pipeline.stages)
    stage_targets = tuple(str(stage._target_) for stage in stage_configs)
    _exact(stage_targets, EXPECTED_STAGE_TARGETS, "configured stage topology")
    stages = tuple(pipeline_algo.pipeline.stages)
    _exact(
        tuple(type(stage) for stage in stages),
        EXPECTED_STAGE_TYPES,
        "instantiated stage topology",
    )

    train, train_excluded = pipeline_algo.pipeline.plan(
        ("front_img_1", "state_agent_obj", "actions"), mode="train"
    )
    inference, inference_excluded = pipeline_algo.pipeline.plan(
        ("front_img_1", "state_agent_obj"), mode="inference"
    )
    _require(
        not train_excluded, f"training graph has excluded stages: {train_excluded}"
    )
    blocked_inference = [
        (type(stage).__name__, missing)
        for stage, missing in inference_excluded
        if missing != ["<train-only>"]
    ]
    _require(
        not blocked_inference,
        f"inference graph has blocked stages: {blocked_inference}",
    )
    _exact(tuple(type(stage) for stage in train), EXPECTED_STAGE_TYPES, "train plan")
    _exact(
        tuple(type(stage) for stage in inference),
        EXPECTED_INFERENCE_TYPES,
        "inference plan",
    )
    train_field = next(
        stage for stage in train if isinstance(stage, ConditionalVelocityStage)
    )
    inference_field = next(
        stage for stage in inference if isinstance(stage, ConditionalVelocityStage)
    )
    train_decoder = next(
        stage for stage in train if isinstance(stage, ContentDecoderStage)
    )
    inference_decoder = next(
        stage for stage in inference if isinstance(stage, ContentDecoderStage)
    )
    _require(train_field is inference_field, "train/inference field stage was copied")
    _require(
        train_decoder is inference_decoder, "train/inference decoder stage was copied"
    )
    _require(
        train_field.field is stages[5].field,
        "train/inference does not use the configured field instance",
    )
    _require(
        train_decoder.decoder is stages[6].decoder,
        "train/inference does not use the configured decoder instance",
    )
    return {
        "configured_targets": list(stage_targets),
        "inference_order": [type(stage).__name__ for stage in inference],
        "shared_decoder_instance": True,
        "shared_field_instance": True,
        "train_order": [type(stage).__name__ for stage in train],
    }


def _validate_optimization(config: DictConfig) -> dict[str, Any]:
    optimizer = config.model.optimizer
    _exact(str(optimizer._target_), "torch.optim.AdamW", "optimizer target")
    _exact(bool(optimizer._partial_), True, "optimizer partial construction")
    _float(optimizer.lr, 3.0e-5, "learning rate")
    _exact([float(value) for value in optimizer.betas], [0.9, 0.999], "Adam betas")
    _float(optimizer.eps, 1.0e-8, "Adam epsilon")
    _float(optimizer.weight_decay, 1.0e-4, "weight decay")

    scheduler = config.model.scheduler
    _exact(
        str(scheduler._target_),
        "egomimic.utils.schedulers.warmup_cosine_scheduler",
        "scheduler target",
    )
    _exact(bool(scheduler._partial_), True, "scheduler partial construction")
    _exact(int(scheduler.max_steps), 240_000, "scheduler maximum steps")
    _exact(int(scheduler.warmup_steps), 8_000, "scheduler warmup steps")
    _float(scheduler.warmup_start_factor, 0.1, "warmup start factor")
    _float(scheduler.eta_min, 3.0e-6, "scheduler floor")

    trainer = config.trainer
    _exact(int(trainer.max_steps), 240_000, "trainer maximum steps")
    _exact(int(trainer.val_check_interval), 10_000, "validation cadence")
    _float(trainer.gradient_clip_val, 3.0, "gradient clip")
    _exact(str(trainer.gradient_clip_algorithm), "norm", "gradient clip algorithm")
    _exact(str(trainer.precision), "bf16", "training precision")

    checkpoint = config.callbacks.model_checkpoint
    _exact(
        int(checkpoint.every_n_train_steps),
        40_000,
        "checkpoint cadence",
    )
    _exact(checkpoint.every_n_epochs, None, "epoch checkpoint cadence")
    _exact(int(checkpoint.save_top_k), -1, "checkpoint retention")
    _require("{step}" in str(checkpoint.filename), "checkpoint name must include step")

    return {
        "checkpoint_every_steps": 40_000,
        "checkpoint_filename": str(checkpoint.filename),
        "gradient_clip_norm": 3.0,
        "max_steps": 240_000,
        "optimizer": {
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "lr": 3.0e-5,
            "target": "torch.optim.AdamW",
            "weight_decay": 1.0e-4,
        },
        "precision": "bf16",
        "scheduler": {
            "eta_min": 3.0e-6,
            "max_steps": 240_000,
            "target": "egomimic.utils.schedulers.warmup_cosine_scheduler",
            "warmup_start_factor": 0.1,
            "warmup_steps": 8_000,
        },
        "validation_every_steps": 10_000,
    }


def _validate_data_and_launch(
    config: DictConfig, *, config_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    _exact(str(config.mode), "train", "run mode")
    _exact(config.ckpt_path, None, "from-scratch checkpoint")
    _exact(int(config.launch_params.gpus_per_node), 1, "GPUs per node")
    _exact(int(config.launch_params.nodes), 1, "launch nodes")
    _exact(int(config.trainer.devices), 1, "trainer devices")
    _exact(int(config.trainer.num_nodes), 1, "trainer nodes")
    _exact(int(config.trainer.accumulate_grad_batches), 1, "gradient accumulation")

    train_sources = tuple(config.data.train_datasets.keys())
    valid_sources = tuple(config.data.valid_datasets.keys())
    _exact(train_sources, valid_sources, "train/validation source identity")
    _exact(len(train_sources), 1, "initial source count")
    source = train_sources[0]
    train = config.data.train_datasets[source]
    valid = config.data.valid_datasets[source]
    _exact(str(train.mode), "train", "training dataset mode")
    _exact(str(valid.mode), "valid", "validation dataset mode")
    _float(train.valid_ratio, 0.01, "training split ratio")
    _float(valid.valid_ratio, 0.01, "validation split ratio")
    _exact(int(train.split_seed), 42, "training split seed")
    _exact(int(valid.split_seed), 42, "validation split seed")
    _exact(int(train.resolver.expected_episode_count), 2_999, "episode inventory")
    _exact(int(train.expected_train_episode_count), 2_970, "train episodes")
    _exact(int(train.expected_valid_episode_count), 29, "validation episodes")
    _exact(
        str(train.resolver.expected_episode_names_sha256),
        str(valid.resolver.expected_episode_names_sha256),
        "train/validation inventory hash",
    )
    _exact(
        str(train.expected_train_episode_names_sha256),
        str(valid.expected_train_episode_names_sha256),
        "train episode hash",
    )
    _exact(
        str(train.expected_valid_episode_names_sha256),
        str(valid.expected_valid_episode_names_sha256),
        "validation episode hash",
    )

    train_batch = int(config.data.train_dataloader_params[source].batch_size)
    world_size = int(config.launch_params.gpus_per_node) * int(
        config.launch_params.nodes
    )
    global_batch = (
        train_batch * world_size * int(config.trainer.accumulate_grad_batches)
    )
    _exact(train_batch, 32, "per-GPU train batch")
    _exact(global_batch, 32, "effective global batch")
    _exact(int(config.planar.batch_size), 32, "declared batch size")

    provenance = config.run_provenance
    _exact(int(provenance.split_seed), 42, "provenance split seed")
    _float(provenance.valid_ratio, 0.01, "provenance validation ratio")
    _exact(
        int(provenance.train_episode_count_per_domain), 2_970, "provenance train count"
    )
    _exact(int(provenance.valid_episode_count_per_domain), 29, "provenance valid count")
    _exact(
        int(provenance.union_episode_count_per_domain), 2_999, "provenance union count"
    )
    _exact(int(provenance.id_overlap_count), 0, "episode ID overlap")
    _exact(int(provenance.resolved_path_overlap_count), 0, "physical path overlap")

    repository_root = Path(config_root).resolve().parents[1]
    manifest_path = Path(str(provenance.split_manifest_path))
    if not manifest_path.is_absolute():
        manifest_path = repository_root / manifest_path
    _require(manifest_path.is_file(), f"split manifest missing: {manifest_path}")
    manifest_sha256 = _sha256(manifest_path)
    _exact(
        manifest_sha256, str(provenance.split_manifest_sha256), "split manifest hash"
    )
    manifest = json.loads(manifest_path.read_text())
    _exact(manifest.get("status"), "PASS", "split manifest status")
    _exact(int(manifest.get("split_seed")), 42, "split manifest seed")
    _float(manifest.get("valid_ratio"), 0.01, "split manifest ratio")
    _exact(
        int(manifest.get("cross_domain_train_valid_resolved_path_overlap_count")),
        0,
        "split manifest cross-source path overlap",
    )
    _exact(tuple(manifest["domains"]), train_sources, "split manifest sources")
    domain = manifest["domains"][source]
    for key, expected in (
        ("total_count", 2_999),
        ("train_count", 2_970),
        ("valid_count", 29),
        ("union_count", 2_999),
        ("id_overlap_count", 0),
        ("resolved_path_overlap_count", 0),
    ):
        _exact(int(domain[key]), expected, f"split manifest {key}")
    _exact(bool(domain["union_matches_inventory"]), True, "complete corpus coverage")
    _exact(
        str(domain["inventory_names_sha256"]),
        str(train.resolver.expected_episode_names_sha256),
        "inventory names hash",
    )
    _exact(
        str(domain["train_names_sha256"]),
        str(train.expected_train_episode_names_sha256),
        "train names hash",
    )
    _exact(
        str(domain["valid_names_sha256"]),
        str(train.expected_valid_episode_names_sha256),
        "validation names hash",
    )

    configured_content_manifest_path = str(provenance.content_manifest_path)
    _exact(
        configured_content_manifest_path,
        CANONICAL_CONTENT_MANIFEST_RELATIVE_PATH.as_posix(),
        "canonical dataset-content manifest path",
    )
    content_manifest_path = repository_root / CANONICAL_CONTENT_MANIFEST_RELATIVE_PATH
    _require(
        content_manifest_path.is_file(),
        f"dataset-content manifest missing: {content_manifest_path}",
    )
    configured_content_manifest_sha256 = str(provenance.content_manifest_sha256).lower()
    configured_aggregate_sha256 = str(
        provenance.dataset_content_aggregate_sha256
    ).lower()
    canonical_content_manifest_sha256 = _sha256(content_manifest_path)
    _exact(
        configured_content_manifest_sha256,
        canonical_content_manifest_sha256,
        "canonical dataset-content manifest hash",
    )
    content_manifest = json.loads(content_manifest_path.read_text())
    try:
        content_identity = validate_content_manifest(content_manifest)
    except RuntimeError as error:
        raise PreflightError(str(error)) from error
    canonical_aggregate_sha256 = str(content_identity["aggregate_sha256"]).lower()
    _exact(
        configured_aggregate_sha256,
        canonical_aggregate_sha256,
        "canonical dataset aggregate content hash",
    )
    _exact(
        int(content_identity["episode_count"]),
        2_999,
        "dataset-content manifest episode count",
    )
    evaluator_content = OmegaConf.to_container(
        config.evaluator.energy_score_provenance.dataset_content,
        resolve=True,
    )
    _require(
        isinstance(evaluator_content, Mapping),
        "evaluator dataset-content provenance must be a mapping",
    )
    _exact(
        set(evaluator_content),
        {"aggregate_sha256", "manifest_path", "manifest_sha256"},
        "evaluator dataset-content provenance keys",
    )
    _exact(
        str(evaluator_content["manifest_path"]),
        configured_content_manifest_path,
        "evaluator dataset-content manifest path provenance",
    )
    _exact(
        str(evaluator_content["manifest_sha256"]).lower(),
        configured_content_manifest_sha256,
        "evaluator dataset-content manifest hash provenance",
    )
    _exact(
        str(evaluator_content["aggregate_sha256"]).lower(),
        configured_aggregate_sha256,
        "evaluator dataset aggregate content hash provenance",
    )

    objective = provenance.objective
    _float(objective.flow_weight, 1.0, "provenance FM weight")
    _float(objective.action_velocity_weight, 1.0, "provenance action weight")
    _float(
        objective.reconstruction_weight,
        float(config.model.reconstruction_weight),
        "provenance reconstruction weight",
    )
    _exact(int(objective.flow_samples_per_content), 14, "provenance bridge samples")
    _float(objective.decoded_noise_scale_weight, 0.0, "decoded-noise scale weight")
    _float(objective.monotonic_weight, 0.0, "monotonicity weight")
    _exact(str(provenance.inference.sampler), "reverse_euler", "inference sampler")
    _exact(int(provenance.inference.steps), 16, "inference sampler steps")
    _exact(
        bool(provenance.inference.classifier_free_guidance),
        False,
        "canonical classifier-free guidance",
    )
    _exact(
        int(provenance.energy_score_contract.sample_count), 32, "EnergyScore samples"
    )
    _exact(bool(config.evaluator.energy_score_enabled), True, "EnergyScore enabled")
    energy_provenance = config.evaluator.energy_score_provenance
    _exact(
        energy_provenance.source_commit,
        provenance.source_commit,
        "EnergyScore source commit provenance",
    )
    _exact(
        energy_provenance.normalization_sha256,
        provenance.normalization_sha256,
        "EnergyScore normalization provenance",
    )
    _exact(
        str(energy_provenance.split_manifest_sha256),
        str(provenance.split_manifest_sha256),
        "EnergyScore split provenance",
    )
    _exact(
        str(config.evaluator.energy_score_validation_view.split_manifest_sha256),
        str(provenance.split_manifest_sha256),
        "EnergyScore validation-view split provenance",
    )
    try:
        distance = normalize_usocket_energy_distance_config(
            OmegaConf.to_container(config.evaluator.energy_score_distance, resolve=True)
        )
    except (TypeError, ValueError) as error:
        raise PreflightError(str(error)) from error
    _exact(distance, USOCKET_ENERGY_DISTANCE_CONFIG, "EnergyScore distance")
    try:
        provenance_distance = normalize_usocket_energy_distance_config(
            OmegaConf.to_container(
                provenance.energy_score_contract.distance,
                resolve=True,
            )
        )
        evaluator_distance = normalize_usocket_energy_distance_config(
            OmegaConf.to_container(
                energy_provenance.distance_contract,
                resolve=True,
            )
        )
    except (TypeError, ValueError) as error:
        raise PreflightError(str(error)) from error
    _exact(
        provenance_distance,
        distance,
        "EnergyScore provenance distance",
    )
    _exact(evaluator_distance, distance, "evaluator EnergyScore distance provenance")

    diagnostics = config.evaluator.action_flow_diagnostics
    _exact(bool(diagnostics.enabled), True, "Action Flow diagnostics enabled")
    _exact(
        [float(value) for value in diagnostics.raw_noise_levels],
        [0.0, 0.25, 0.5, 0.75, 1.0],
        "Action Flow diagnostic noise levels",
    )
    _exact(int(diagnostics.max_batches_per_rank), 1, "diagnostic batch limit")
    _exact(int(diagnostics.max_samples), 16, "diagnostic sample limit")
    _exact(int(diagnostics.jacobian_samples), 2, "diagnostic Jacobian sample limit")
    _exact(bool(diagnostics.capture_activations), True, "activation capture")
    _exact(
        {
            int(key): int(value)
            for key, value in diagnostics.activation_layer_map.items()
        },
        {0: 0, 1: 11},
        "diagnostic activation layer map",
    )
    _exact(int(diagnostics.cknna_k), 10, "diagnostic CKNNA k")
    try:
        native_error = normalize_usocket_native_error_config(
            OmegaConf.to_container(diagnostics.native_error, resolve=True)
        )
    except (TypeError, ValueError) as error:
        raise PreflightError(str(error)) from error
    _exact(native_error, USOCKET_NATIVE_ERROR_CONFIG, "diagnostic native error")
    _exact(
        diagnostics.provenance.source_commit,
        provenance.source_commit,
        "diagnostic source commit provenance",
    )
    _exact(
        diagnostics.provenance.normalization_sha256,
        provenance.normalization_sha256,
        "diagnostic normalization provenance",
    )
    _exact(
        str(diagnostics.provenance.split_manifest_sha256),
        str(provenance.split_manifest_sha256),
        "diagnostic split provenance",
    )
    diagnostic_content = OmegaConf.to_container(
        diagnostics.provenance.dataset_content,
        resolve=True,
    )
    _require(
        isinstance(diagnostic_content, Mapping),
        "diagnostic dataset-content provenance must be a mapping",
    )
    _exact(
        set(diagnostic_content),
        {"aggregate_sha256", "manifest_sha256"},
        "diagnostic dataset-content provenance keys",
    )
    _exact(
        str(diagnostic_content["manifest_sha256"]).lower(),
        configured_content_manifest_sha256,
        "diagnostic dataset-content manifest provenance",
    )
    _exact(
        str(diagnostic_content["aggregate_sha256"]).lower(),
        configured_aggregate_sha256,
        "diagnostic dataset aggregate content provenance",
    )
    _exact(
        str(diagnostics.validation_view.split_manifest_sha256),
        str(provenance.split_manifest_sha256),
        "diagnostic validation-view split provenance",
    )
    _exact(
        int(diagnostics.validation_view.per_rank_batch_size),
        16,
        "diagnostic validation batch size",
    )
    _exact(
        int(diagnostics.validation_view.world_size),
        1,
        "diagnostic validation world size",
    )
    _exact(
        str(diagnostics.noise_seed_bank_sha256),
        str(provenance.energy_score_contract.seed_bank_sha256),
        "diagnostic seed-bank hash",
    )
    seed_bank = Path(config_root) / "evaluator" / "energy_score_seed_bank_k32_v1.json"
    _require(seed_bank.is_file(), f"diagnostic seed bank missing: {seed_bank}")
    _exact(
        _sha256(seed_bank),
        str(diagnostics.noise_seed_bank_sha256),
        "diagnostic seed-bank file hash",
    )
    _exact(
        str(diagnostics.provenance.decoder_jacobian_evaluation),
        "declared_bridge_state_at_each_fixed_noise_level",
        "diagnostic Jacobian evaluation point",
    )

    launch = {
        "accumulate_grad_batches": 1,
        "effective_global_batch": global_batch,
        "gpus_per_node": 1,
        "nodes": 1,
        "per_gpu_batch": train_batch,
        "world_size": world_size,
    }
    data = {
        "content_manifest_path": str(provenance.content_manifest_path),
        "content_manifest_sha256": configured_content_manifest_sha256,
        "dataset_content_aggregate_sha256": configured_aggregate_sha256,
        "episode_counts": {"total": 2_999, "train": 2_970, "validation": 29},
        "inventory_names_sha256": str(train.resolver.expected_episode_names_sha256),
        "source": str(source),
        "split_manifest_path": str(provenance.split_manifest_path),
        "split_manifest_sha256": manifest_sha256,
        "split_seed": 42,
        "train_names_sha256": str(train.expected_train_episode_names_sha256),
        "valid_names_sha256": str(train.expected_valid_episode_names_sha256),
        "valid_ratio": 0.01,
        "zero_id_overlap": True,
        "zero_resolved_path_overlap": True,
    }
    return data, launch


def validate_config(
    config: DictConfig,
    *,
    experiment: str,
    config_root: Path = DEFAULT_CONFIG_ROOT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one composed config and return report plus resolved payload."""

    resolved, resolved_hash = resolved_config_payload(config)
    _exact(
        str(config.model._target_),
        "egomimic.pl_utils.pl_model_action_flow.ActionFlowModelWrapper",
        "model wrapper",
    )
    configured_targets = tuple(
        str(stage._target_) for stage in config.model.pipeline.stages
    )
    _exact(configured_targets, EXPECTED_STAGE_TARGETS, "stage topology")
    _require(
        "time_scale" in config.model.pipeline.stages[5].field,
        "field time_scale must be explicit",
    )
    _float(config.model.pipeline.stages[5].field.time_scale, 1_000.0, "time scale")

    data, launch = _validate_data_and_launch(config, config_root=config_root)
    optimization = _validate_optimization(config)
    pipeline_algo = instantiate(config.model.pipeline, device="cpu")
    _require(isinstance(pipeline_algo, PipelineAlgo), "pipeline did not instantiate")
    stages = tuple(pipeline_algo.pipeline.stages)
    topology = _validate_topology(config, pipeline_algo)
    _validate_generic_surface(config, pipeline_algo.pipeline)
    dimensions, parameters = _validate_dimensions_and_modules(config, stages)
    parameters["pipeline_total"] = _parameter_manifest(pipeline_algo.nets)
    accounted = sum(
        counts["total"]
        for name, counts in parameters.items()
        if name != "pipeline_total"
    )
    _exact(accounted, parameters["pipeline_total"]["total"], "parameter accounting")

    reconstruction_weight = float(config.model.reconstruction_weight)
    report = {
        "config_name": str(config.name),
        "dimensions": dimensions,
        "experiment": str(experiment),
        "launch": launch,
        "objective": {
            "action_velocity_weight": 1.0,
            "condition_dropout_probability": 0.3,
            "flow_samples_per_content": 14,
            "flow_weight": 1.0,
            "reconstruction_weight": reconstruction_weight,
        },
        "optimization": optimization,
        "parameters": parameters,
        "resolved_config_sha256": resolved_hash,
        "schema_version": SCHEMA_VERSION,
        "split": data,
        "status": "PASS",
        "topology": topology,
    }
    del stages, pipeline_algo
    gc.collect()
    return report, resolved


def validate_experiment(
    experiment: str,
    *,
    config_root: Path = DEFAULT_CONFIG_ROOT,
    overrides: Sequence[str] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = compose_experiment(
        experiment,
        config_root=config_root,
        overrides=overrides,
    )
    return validate_config(
        config,
        experiment=experiment,
        config_root=config_root,
    )


def _differences(left: Any, right: Any, path: tuple[str, ...] = ()) -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences = []
        for key in sorted(set(left) | set(right)):
            child = path + (str(key),)
            if key not in left or key not in right:
                differences.append(".".join(child))
            else:
                differences.extend(_differences(left[key], right[key], child))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        differences = []
        if len(left) != len(right):
            return [".".join(path)]
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(
                _differences(left_item, right_item, path + (str(index),))
            )
        return differences
    return [] if left == right else [".".join(path)]


def validate_pair(
    first_experiment: str,
    second_experiment: str,
    *,
    config_root: Path = DEFAULT_CONFIG_ROOT,
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    first, first_config = validate_experiment(
        first_experiment,
        config_root=config_root,
        overrides=overrides,
    )
    second, second_config = validate_experiment(
        second_experiment,
        config_root=config_root,
        overrides=overrides,
    )
    differences = tuple(_differences(first_config, second_config))
    _exact(
        frozenset(differences),
        ALLOWED_PAIR_DIFFERENCES,
        "paired experiment differences",
    )
    paired_weights = {
        first["objective"]["reconstruction_weight"],
        second["objective"]["reconstruction_weight"],
    }
    _require(len(paired_weights) == 2, "paired reconstruction weights must differ")
    _require(
        paired_weights <= {1.0, 10.0, 100.0},
        "paired reconstruction weights must be approved",
    )
    return {
        "comparison": {
            "differing_paths": list(differences),
            "only_declared_reconstruction_differences": True,
            "status": "PASS",
        },
        "experiments": [first, second],
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
    }


def _emit(payload: Mapping[str, Any], output: Path | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(rendered)
        return
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--compare-experiment")
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        if args.compare_experiment:
            payload = validate_pair(
                args.experiment,
                args.compare_experiment,
                config_root=args.config_root,
                overrides=args.override,
            )
        else:
            payload, _ = validate_experiment(
                args.experiment,
                config_root=args.config_root,
                overrides=args.override,
            )
    except Exception as exc:
        payload = {
            "error": {"message": str(exc), "type": type(exc).__name__},
            "schema_version": SCHEMA_VERSION,
            "status": "FAIL",
        }
        _emit(payload, args.output)
        return 1

    _emit(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
