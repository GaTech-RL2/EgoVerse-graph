#!/usr/bin/env python3
"""Fail-closed verification for the USocket Action Flow training smoke.

The smoke is deliberately narrower than a general checkpoint validator.  It
binds a two-step Lightning run to one of the two approved Action Flow configs,
checks the exact optimizer/validation execution contract, strictly reconstructs
the ActionFlowModelWrapper on CPU, and proves that W&B and both immutable
validation artifact streams completed successfully.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

# ``python scripts/train/...`` otherwise places only ``scripts/train`` on the
# import path.  Resolve imports from the exact checkout being verified.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from egomimic.eval.artifact_paths import artifact_execution_identity  # noqa: E402
from egomimic.eval.energy_score import (  # noqa: E402
    USOCKET_ENERGY_DISTANCE_CONFIG,
    normalize_usocket_energy_distance_config,
    usocket_energy_distance_metadata,
)
from egomimic.eval.planar_action_eval import (  # noqa: E402
    USOCKET_NATIVE_ERROR_CONFIG,
    normalize_usocket_native_error_config,
)
from egomimic.pl_utils.pl_model_action_flow import (  # noqa: E402
    ActionFlowModelWrapper,
)
from scripts.train.validate_slurm_job_contract import (  # noqa: E402
    GPU_PROFILES,
    gpu_name_allowed,
    validate_gpu_profile,
)
from tools.validate_action_flow_config import (  # noqa: E402
    GRAPH_METHOD,
    LEGACY_METHOD,
    LIKELIHOOD_METHOD,
    STOPGRAD_METHOD,
    PreflightError,
    _validate_dimensions_and_modules,
    action_flow_method,
    method_stage_targets,
    method_wrapper_target,
    validate_method_contract,
)

SCHEMA_VERSION = 1
EXPECTED_PARAMETER_COUNT = 50_725_221
APPROVED_EXPERIMENTS = {
    "pusht/action_flow_bc_usocket_latent_fm_sg_recon1_s42": (
        "action_flow_bc_usocket_latent_fm_sg_recon1_s42",
        1.0,
        1.0,
    ),
    "pusht/action_flow_bc_usocket_latent_fm_sg_recon1_lr1e5_s42": (
        "action_flow_bc_usocket_latent_fm_sg_recon1_lr1e5_s42",
        1.0,
        1.0,
    ),
    "pusht/action_flow_bc_usocket_bridge_likelihood_s42": (
        "action_flow_bc_usocket_bridge_likelihood_s42",
        0.0,
        0.0,
    ),
    "pusht/action_flow_bc_usocket_graph_section_s42": (
        "action_flow_bc_usocket_graph_section_s42",
        0.0,
        1.0,
    ),
    "pusht/action_flow_bc_usocket_recon1_s42": (
        "action_flow_bc_usocket_recon1_s42",
        1.0,
        1.0,
    ),
    "pusht/action_flow_bc_usocket_recon10_s42": (
        "action_flow_bc_usocket_recon10_s42",
        10.0,
        1.0,
    ),
    "pusht/action_flow_bc_usocket_recon10_warmup10k_s42": (
        "action_flow_bc_usocket_recon10_warmup10k_s42",
        10.0,
        1.0,
    ),
    "pusht/action_flow_bc_usocket_recon10_warmup10k_flow001_s42": (
        "action_flow_bc_usocket_recon10_warmup10k_flow001_s42",
        10.0,
        0.01,
    ),
    "pusht/action_flow_bc_usocket_recon100_s42": (
        "action_flow_bc_usocket_recon100_s42",
        100.0,
        1.0,
    ),
}
LOW_LR_EXPERIMENT = "pusht/action_flow_bc_usocket_latent_fm_sg_recon1_lr1e5_s42"
EXPECTED_STAGE_TARGETS = (
    "egomimic.pipeline.stages_sampler.FusedObsEncoder",
    "egomimic.pipeline.stages_sampler.GaussianLatentNoise",
    "egomimic.pipeline.stages_io.ActionTargetBuilder",
    "egomimic.pipeline.stages_action_flow.ContentEncoderStage",
    "egomimic.pipeline.stages_action_flow.LatentBridgeStage",
    "egomimic.pipeline.stages_action_flow.ConditionalVelocityStage",
    "egomimic.pipeline.stages_action_flow.ContentDecoderStage",
    "egomimic.pipeline.stages_action_flow.ActionFlowObjectiveStage",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
ENERGY_CHECKPOINT_STATUS = (
    "unavailable_during_validation_artifact_write; post-run_smoke_"
    "verifier_binds_checkpoint_and_artifact_by_global_step_and_records_"
    "both_file_hashes"
)
CANONICAL_CONTENT_MANIFEST_SHA256 = (
    "a1c81fb0ce8967aba795383a293180f9ba08a0ecfdd6f4a878afb20b39733761"
)
CANONICAL_DATASET_CONTENT_AGGREGATE_SHA256 = (
    "80f835ad37c3d5c5b7b2d5c3e1656c307ee567a1f63f51081165bf404b8ceb52"
)
SOURCE_LABEL = "pushshapes_sim_u_socket"
NATIVE_LEVELS = ("t0000", "t0250", "t0500", "t0750", "t1000")


class SmokeVerificationError(RuntimeError):
    """Raised when any required smoke evidence is missing or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeVerificationError(message)


def _select(config: DictConfig, path: str) -> Any:
    value = OmegaConf.select(config, path, default=None)
    _require(value is not None, f"resolved config is missing {path}")
    return value


def _exact(config: DictConfig, path: str, expected: Any) -> None:
    actual = _select(config, path)
    _require(actual == expected, f"{path}: expected {expected!r}, got {actual!r}")


def _float(config: DictConfig, path: str, expected: float) -> None:
    actual = _select(config, path)
    try:
        value = float(actual)
    except (TypeError, ValueError) as error:
        raise SmokeVerificationError(f"{path} is not numeric: {actual!r}") from error
    _require(
        math.isfinite(value)
        and math.isclose(value, expected, rel_tol=0.0, abs_tol=1.0e-12),
        f"{path}: expected {expected!r}, got {actual!r}",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: str, label: str) -> str:
    normalized = str(value).lower()
    _require(bool(SHA256_RE.fullmatch(normalized)), f"{label} is not SHA-256")
    return normalized


def _resolve_path(value: Any, *, relative_to: Path, label: str) -> Path:
    rendered = str(value)
    _require(bool(rendered), f"{label} path is empty")
    path = Path(rendered).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    try:
        path = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise SmokeVerificationError(f"{label} does not exist: {path}") from error
    _require(path.is_file(), f"{label} is not a file: {path}")
    return path


def _resolve_normalization_path(value: Any, *, relative_to: Path) -> Path:
    rendered = str(value)
    _require(bool(rendered), "train-only normalization artifact path is empty")
    path = Path(rendered).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    try:
        path = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise SmokeVerificationError(
            f"train-only normalization artifact does not exist: {path}"
        ) from error
    if path.is_dir():
        path = path / "norm_stats.json"
    _require(path.is_file(), f"train-only normalization artifact is not a file: {path}")
    return path


def _git_head(repository_root: Path = REPOSITORY_ROOT) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _require(result.returncode == 0, f"could not resolve source HEAD: {result.stderr}")
    head = result.stdout.strip().lower()
    _require(bool(COMMIT_RE.fullmatch(head)), f"invalid source HEAD: {head!r}")
    return head


def _recorded_hash(config: DictConfig, paths: Sequence[str]) -> str | None:
    values = []
    for path in paths:
        value = OmegaConf.select(config, path, default=None)
        if value not in (None, ""):
            values.append(_validate_sha256(str(value), path))
    _require(len(set(values)) <= 1, f"recorded hashes disagree across {tuple(paths)}")
    return values[0] if values else None


def _plain_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    _require(isinstance(value, Mapping), f"{label} is not a mapping")
    return {str(key): child for key, child in value.items()}


def _same_mapping(config: DictConfig, paths: Sequence[str], *, label: str) -> dict:
    values = [
        _plain_mapping(_select(config, path), label=f"{label} at {path}")
        for path in paths
    ]
    _require(all(value == values[0] for value in values[1:]), f"{label} differs")
    return values[0]


def _resolve_expected_hash(
    *,
    supplied: str | None,
    recorded: str | None,
    actual: str,
    label: str,
) -> str:
    expected = _validate_sha256(supplied, label) if supplied is not None else recorded
    # If neither was available before verification, the verifier itself records
    # the content hash in SMOKE_RESULT.json.  Split and normalization always have
    # an independent config/provenance record and therefore never take this path.
    if expected is None:
        expected = actual
    _require(actual == expected, f"{label} mismatch: {actual} != {expected}")
    if supplied is not None and recorded is not None:
        _require(
            _validate_sha256(supplied, label) == recorded,
            f"supplied and recorded {label} values disagree",
        )
    return expected


def _validate_config(
    *,
    config_path: Path,
    experiment: str,
    run_dir: Path,
    expected_head: str,
    expected_config_sha256: str | None,
    expected_split_sha256: str | None,
    expected_normalization_sha256: str | None,
    expected_content_manifest_sha256: str | None = None,
    expected_dataset_content_aggregate_sha256: str | None = None,
) -> tuple[DictConfig, dict[str, Any]]:
    _require(experiment in APPROVED_EXPERIMENTS, f"unapproved experiment: {experiment}")
    expected_name, reconstruction_weight, flow_weight = APPROVED_EXPERIMENTS[experiment]
    expected_lr, expected_eta_min = (
        (1.0e-5, 1.0e-6)
        if experiment == LOW_LR_EXPERIMENT
        else (3.0e-5, 3.0e-6)
    )
    config = OmegaConf.load(config_path)
    try:
        method = validate_method_contract(config, experiment)
    except PreflightError as error:
        raise SmokeVerificationError(str(error)) from error

    _exact(config, "name", expected_name)
    _exact(
        config,
        "model._target_",
        method_wrapper_target(method),
    )
    targets = tuple(str(stage._target_) for stage in config.model.pipeline.stages)
    _require(
        targets == method_stage_targets(method), f"unexpected stage topology: {targets}"
    )

    for path, expected in (
        ("model.action_horizon", 16),
        ("model.action_dim", 4),
        ("model.latent_dim", 8),
        ("model.condition_dim", 67),
        ("model.flow_samples_per_content", 14),
        ("model.num_inference_steps", 16),
        ("model.pipeline.stages.0.n_obs_steps", 1),
        ("model.pipeline.stages.1.num_tokens", 16),
        ("model.pipeline.stages.1.latent_dim", 8),
        ("model.pipeline.stages.4.samples_per_content", 14),
        ("model.pipeline.stages.5.num_inference_steps", 16),
        ("model.pipeline.stages.5.field.input_dim", 8),
        ("model.pipeline.stages.5.field.output_dim", 8),
        ("model.pipeline.stages.5.field.horizon", 16),
        ("model.pipeline.stages.5.field.condition_dim", 67),
        ("model.pipeline.stages.5.field.hidden_dim", 512),
        ("model.pipeline.stages.5.field.depth", 12),
        ("model.pipeline.stages.5.field.num_heads", 8),
        ("model.pipeline.stages.5.field.feedforward_dim", 2048),
        ("model.pipeline.stages.5.field.time_embedding_dim", 512),
        ("trainer.max_steps", 2),
        ("trainer.val_check_interval", 1),
        ("trainer.limit_val_batches", 1),
        ("trainer.num_sanity_val_steps", 0),
        ("trainer.accumulate_grad_batches", 1),
        ("trainer.devices", 1),
        ("trainer.num_nodes", 1),
        ("launch_params.gpus_per_node", 1),
        ("launch_params.nodes", 1),
        ("callbacks.model_checkpoint.every_n_train_steps", 1),
        ("callbacks.model_checkpoint.save_top_k", -1),
        ("model.gradient_telemetry_cadence", 2),
        ("model.scheduler.max_steps", 240_000),
        ("model.scheduler.warmup_steps", 8_000),
        ("seed", 42),
        ("planar.action_horizon", 16),
        ("planar.observation_horizon", 1),
        ("planar.batch_size", 32),
        ("run_provenance.split_seed", 42),
        ("run_provenance.train_episode_count_per_domain", 2_970),
        ("run_provenance.valid_episode_count_per_domain", 29),
        ("run_provenance.union_episode_count_per_domain", 2_999),
        ("run_provenance.id_overlap_count", 0),
        ("run_provenance.resolved_path_overlap_count", 0),
        ("run_provenance.objective.flow_samples_per_content", 14),
        ("run_provenance.inference.steps", 16),
        ("run_provenance.energy_score_contract.sample_count", 32),
        ("evaluator.energy_score_max_batches_per_rank", 1),
        ("evaluator.energy_score_validation_view.world_size", 1),
        ("evaluator.energy_score_validation_view.per_rank_batch_size", 16),
    ):
        if method == LIKELIHOOD_METHOD and path in {
            "model.flow_samples_per_content",
            "model.num_inference_steps",
            "model.pipeline.stages.4.samples_per_content",
            "model.pipeline.stages.5.num_inference_steps",
            "run_provenance.objective.flow_samples_per_content",
            "run_provenance.inference.steps",
        }:
            continue  # Validated against the discrete-chain contract above.
        _exact(config, path, expected)

    if "_warmup10k_" in experiment or experiment.endswith("_warmup10k_s42"):
        _exact(config, "model.reconstruction_only_warmup_steps", 1)
        _exact(
            config,
            "run_provenance.objective.requested_full_reconstruction_only_warmup_steps",
            10_000,
        )
        _exact(
            config,
            "run_provenance.objective.effective_reconstruction_only_warmup_steps",
            1,
        )

    for path, expected in (
        ("model.condition_dropout_probability", 0.3),
        ("model.reconstruction_weight", reconstruction_weight),
        ("model.pipeline.stages.4.condition_dropout_probability", 0.3),
        ("model.pipeline.stages.5.field.time_scale", 1_000.0),
        ("model.pipeline.stages.5.field.condition_dropout_probability", 0.3),
        ("model.flow_weight", flow_weight),
        ("model.pipeline.stages.7.flow_weight", flow_weight),
        ("model.pipeline.stages.7.reconstruction_weight", reconstruction_weight),
        ("model.pipeline.stages.7.action_velocity_weight", 1.0),
        ("model.reconstruction_weight", reconstruction_weight),
        ("model.optimizer.lr", expected_lr),
        ("model.optimizer.eps", 1.0e-8),
        ("model.optimizer.weight_decay", 1.0e-4),
        ("model.scheduler.warmup_start_factor", 0.1),
        ("model.scheduler.eta_min", expected_eta_min),
        ("trainer.gradient_clip_val", 3.0),
        ("run_provenance.valid_ratio", 0.01),
        ("run_provenance.objective.flow_weight", flow_weight),
        ("run_provenance.objective.reconstruction_weight", reconstruction_weight),
        ("run_provenance.objective.action_velocity_weight", 1.0),
        ("run_provenance.objective.decoded_noise_scale_weight", 0.0),
        ("run_provenance.objective.monotonic_weight", 0.0),
    ):
        if method == LIKELIHOOD_METHOD and (
            path.startswith("model.pipeline.stages.7.")
            or path in {"model.reconstruction_weight", "model.flow_weight"}
            or path.startswith("run_provenance.objective.")
        ):
            continue  # This method has NLL components, not FM/reconstruction.
        _float(config, path, expected)

    _exact(config, "mode", "train")
    _require(config.ckpt_path is None, "smoke must initialize from scratch")
    _exact(config, "trainer.accelerator", "gpu")
    _exact(config, "trainer.strategy", "auto")
    _exact(config, "trainer.precision", "bf16")
    _exact(config, "trainer.gradient_clip_algorithm", "norm")
    _exact(config, "trainer.sync_batchnorm", False)
    _exact(config, "model.optimizer._target_", "torch.optim.AdamW")
    _exact(config, "model.optimizer._partial_", True)
    _require(
        [float(value) for value in config.model.optimizer.betas] == [0.9, 0.999],
        "model.optimizer.betas must be [0.9, 0.999]",
    )
    _exact(
        config,
        "model.scheduler._target_",
        "egomimic.utils.schedulers.warmup_cosine_scheduler",
    )
    _exact(config, "model.scheduler._partial_", True)
    _exact(config, "callbacks.model_checkpoint.save_last", "link")
    _require(
        "{step}" in str(_select(config, "callbacks.model_checkpoint.filename")),
        "checkpoint filename must contain immutable step identity",
    )
    _exact(config, "logger.wandb.offline", False)
    _exact(config, "evaluator.energy_score_enabled", True)
    _exact(
        config,
        "run_provenance.inference.sampler",
        (
            "gaussian_bridge_reverse_chain"
            if method == LIKELIHOOD_METHOD
            else "reverse_euler"
        ),
    )
    _exact(config, "run_provenance.inference.classifier_free_guidance", False)
    _exact(config, "run_provenance.action_contract.prediction_horizon", 16)
    _exact(
        config,
        "run_provenance.action_contract.representation",
        "x_y_cos_theta_sin_theta",
    )
    _exact(config, "norm_stats.norm_mode", "quantile")
    _float(config, "norm_stats.sample_frac", 1.0)

    distance_contract = _same_mapping(
        config,
        (
            "evaluator.energy_score_distance",
            "evaluator.energy_score_provenance.distance_contract",
            "run_provenance.energy_score_contract.distance",
        ),
        label="typed USocket EnergyScore distance contract",
    )
    try:
        normalized_distance = normalize_usocket_energy_distance_config(
            distance_contract
        )
    except (TypeError, ValueError) as error:
        raise SmokeVerificationError(
            f"invalid typed USocket EnergyScore distance contract: {error}"
        ) from error
    _require(
        normalized_distance == USOCKET_ENERGY_DISTANCE_CONFIG,
        "typed USocket EnergyScore distance contract differs",
    )
    if method == LIKELIHOOD_METHOD:
        _exact(
            config,
            "evaluator.native_decoder._target_",
            "egomimic.pipeline.pushshapes.USocketRotVecNativeDecoder",
        )
        native_error_contract = dict(USOCKET_NATIVE_ERROR_CONFIG)
    else:
        try:
            native_error_contract = normalize_usocket_native_error_config(
                _plain_mapping(
                    _select(config, "evaluator.action_flow_diagnostics.native_error"),
                    label="Action Flow native-error contract",
                )
            )
        except (TypeError, ValueError) as error:
            raise SmokeVerificationError(
                f"invalid Action Flow native-error contract: {error}"
            ) from error
    _require(
        native_error_contract == USOCKET_NATIVE_ERROR_CONFIG,
        "Action Flow native-error contract differs",
    )

    sources = tuple(config.data.train_datasets)
    _require(sources == (SOURCE_LABEL,), f"unexpected training source: {sources}")
    _require(
        tuple(config.data.valid_datasets) == sources,
        "training and validation sources differ",
    )
    _exact(
        config,
        f"data.train_dataloader_params.{SOURCE_LABEL}.batch_size",
        32,
    )
    _exact(
        config,
        f"data.valid_dataloader_params.{SOURCE_LABEL}.batch_size",
        16,
    )
    global_batch = (
        int(config.data.train_dataloader_params[SOURCE_LABEL].batch_size)
        * int(config.trainer.devices)
        * int(config.trainer.num_nodes)
        * int(config.trainer.accumulate_grad_batches)
    )
    _require(
        global_batch == 32, f"effective global batch must be 32, got {global_batch}"
    )

    expected_head = expected_head.lower()
    _require(bool(COMMIT_RE.fullmatch(expected_head)), "expected HEAD is not a commit")
    recorded_head = OmegaConf.select(
        config, "run_provenance.source_commit", default=None
    )
    if recorded_head is not None:
        _require(
            str(recorded_head).lower() == expected_head,
            "resolved config source HEAD mismatch",
        )
    actual_head = _git_head()
    _require(actual_head == expected_head, f"checkout HEAD mismatch: {actual_head}")

    config_hash = _sha256(config_path)
    config_hash = _resolve_expected_hash(
        supplied=expected_config_sha256,
        recorded=None,
        actual=config_hash,
        label="resolved config SHA-256",
    )

    split_path = _resolve_path(
        _select(config, "run_provenance.split_manifest_path"),
        relative_to=REPOSITORY_ROOT,
        label="split manifest",
    )
    split_hash = _sha256(split_path)
    split_recorded = _recorded_hash(
        config,
        (
            "run_provenance.split_manifest_sha256",
            "evaluator.energy_score_validation_view.split_manifest_sha256",
        ),
    )
    split_hash = _resolve_expected_hash(
        supplied=expected_split_sha256,
        recorded=split_recorded,
        actual=split_hash,
        label="split manifest SHA-256",
    )

    normalization_path = _resolve_normalization_path(
        _select(config, "norm_stats.precomputed_norm_path"),
        relative_to=REPOSITORY_ROOT,
    )
    normalization_hash = _sha256(normalization_path)
    normalization_recorded = _recorded_hash(
        config,
        (
            "run_provenance.normalization_sha256",
            "run_provenance.normalization_stats_sha256",
            "run_provenance.train_only_normalization_sha256",
        ),
    )
    _require(
        normalization_recorded is not None or expected_normalization_sha256 is not None,
        "normalization SHA-256 must be supplied or recorded",
    )
    normalization_hash = _resolve_expected_hash(
        supplied=expected_normalization_sha256,
        recorded=normalization_recorded,
        actual=normalization_hash,
        label="normalization SHA-256",
    )

    content_manifest_path = _resolve_path(
        _select(config, "run_provenance.content_manifest_path"),
        relative_to=REPOSITORY_ROOT,
        label="dataset content manifest",
    )
    content_manifest_hash = _sha256(content_manifest_path)
    recorded_content_hash = _recorded_hash(
        config,
        (
            "run_provenance.content_manifest_sha256",
            "evaluator.energy_score_provenance.dataset_content.manifest_sha256",
            "evaluator.action_flow_diagnostics.provenance.dataset_content.manifest_sha256",
        ),
    )
    _require(recorded_content_hash is not None, "dataset content manifest hash missing")
    content_manifest_hash = _resolve_expected_hash(
        supplied=expected_content_manifest_sha256,
        recorded=recorded_content_hash,
        actual=content_manifest_hash,
        label="dataset content manifest SHA-256",
    )
    _require(
        content_manifest_hash == CANONICAL_CONTENT_MANIFEST_SHA256,
        "dataset content manifest is not the canonical USocket corpus manifest",
    )
    try:
        content_manifest_payload = json.loads(content_manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SmokeVerificationError(
            f"dataset content manifest is not valid JSON: {content_manifest_path}"
        ) from error
    _require(
        isinstance(content_manifest_payload, Mapping),
        "dataset content manifest payload is not a mapping",
    )
    aggregate_recorded = _recorded_hash(
        config,
        (
            "run_provenance.dataset_content_aggregate_sha256",
            "evaluator.energy_score_provenance.dataset_content.aggregate_sha256",
            "evaluator.action_flow_diagnostics.provenance.dataset_content.aggregate_sha256",
        ),
    )
    _require(aggregate_recorded is not None, "dataset aggregate content hash missing")
    manifest_aggregate = _validate_sha256(
        str(content_manifest_payload.get("aggregate_sha256", "")),
        "dataset content manifest aggregate SHA-256",
    )
    expected_aggregate = (
        _validate_sha256(
            expected_dataset_content_aggregate_sha256,
            "dataset aggregate content SHA-256",
        )
        if expected_dataset_content_aggregate_sha256 is not None
        else aggregate_recorded
    )
    _require(
        aggregate_recorded == expected_aggregate == manifest_aggregate,
        "dataset aggregate content SHA-256 mismatch",
    )
    _require(
        manifest_aggregate == CANONICAL_DATASET_CONTENT_AGGREGATE_SHA256,
        "dataset aggregate is not the canonical USocket corpus content",
    )

    energy_content_path = _resolve_path(
        _select(
            config, "evaluator.energy_score_provenance.dataset_content.manifest_path"
        ),
        relative_to=REPOSITORY_ROOT,
        label="EnergyScore dataset content manifest",
    )
    _require(
        energy_content_path == content_manifest_path,
        "EnergyScore/run dataset content manifest paths differ",
    )

    for evaluator_prefix in (
        "evaluator.energy_score_provenance",
        "evaluator.action_flow_diagnostics.provenance",
    ):
        if (
            method == LIKELIHOOD_METHOD
            and evaluator_prefix == "evaluator.action_flow_diagnostics.provenance"
        ):
            continue
        _exact(config, f"{evaluator_prefix}.source_commit", expected_head)
        _exact(config, f"{evaluator_prefix}.normalization_sha256", normalization_hash)
        _exact(config, f"{evaluator_prefix}.split_manifest_sha256", split_hash)

    if method != LIKELIHOOD_METHOD:
        _exact(config, "evaluator.action_flow_diagnostics.enabled", True)
        _exact(config, "evaluator.action_flow_diagnostics.max_batches_per_rank", 1)
        _exact(
            config,
            "evaluator.action_flow_diagnostics.validation_view.world_size",
            1,
        )
        diagnostic_split = _select(
            config,
            "evaluator.action_flow_diagnostics.validation_view.split_manifest_sha256",
        )
        _require(
            str(diagnostic_split) == split_hash, "diagnostic split identity mismatch"
        )

    for path, label in (
        ("evaluator.artifact_root", "EnergyScore artifact root"),
        (
            "evaluator.action_flow_diagnostics.artifact_root",
            "Action Flow diagnostic artifact root",
        ),
    ):
        if method == LIKELIHOOD_METHOD and path.startswith(
            "evaluator.action_flow_diagnostics."
        ):
            continue
        root = Path(str(_select(config, path))).expanduser()
        if not root.is_absolute():
            root = run_dir / root
        root = root.resolve()
        _require(
            root == run_dir or run_dir in root.parents,
            f"{label} escapes the smoke run directory: {root}",
        )

    return config, {
        "config_sha256": config_hash,
        "content_manifest_path": str(content_manifest_path),
        "content_manifest_sha256": content_manifest_hash,
        "dataset_content_aggregate_sha256": manifest_aggregate,
        "energy_score_distance": usocket_energy_distance_metadata(normalized_distance),
        "native_error": native_error_contract,
        "normalization_path": str(normalization_path),
        "normalization_sha256": normalization_hash,
        "repo_head": actual_head,
        "split_manifest_path": str(split_path),
        "split_manifest_sha256": split_hash,
    }


def _finite_tree(value: Any, label: str) -> tuple[int, int]:
    tensor_count = 0
    numeric_count = 0

    def visit(item: Any, path: str) -> None:
        nonlocal tensor_count, numeric_count
        if torch.is_tensor(item):
            _require(item.device.type != "meta", f"{path} is a meta tensor")
            candidate = item.dequantize() if item.is_quantized else item
            if candidate.layout != torch.strided:
                candidate = candidate.to_dense()
            _require(
                not candidate.numel() or bool(torch.isfinite(candidate).all()),
                f"{path} contains non-finite tensor values",
            )
            tensor_count += 1
            return
        if isinstance(item, Mapping):
            for key, child in item.items():
                visit(child, f"{path}.{key}")
            return
        if isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
            return
        if isinstance(item, numbers.Number) and not isinstance(item, bool):
            _require(math.isfinite(float(item)), f"{path} is non-finite")
            numeric_count += 1

    visit(value, label)
    return tensor_count, numeric_count


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_gradient_route_manifest(
    manifest: Any,
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    method: str = LEGACY_METHOD,
) -> dict[str, Any]:
    _require(isinstance(manifest, Mapping), "gradient route manifest is missing")
    _require(
        set(manifest)
        == {
            "intersections",
            "manifest_sha256",
            "route_sha256",
            "routes",
            "schema_version",
        },
        "gradient route manifest has unexpected keys",
    )
    _require(manifest.get("schema_version") == 1, "wrong gradient manifest schema")
    routes = manifest.get("routes")
    route_hashes = manifest.get("route_sha256")
    intersections = manifest.get("intersections")
    expected_labels = (
        ("InteriorBridgeNLL", "BoundaryNLL")
        if method == LIKELIHOOD_METHOD
        else (
            ("FM", "ActionVelocity")
            if method == GRAPH_METHOD
            else ("FM", "Reconstruction", "ActionVelocity")
        )
    )
    _require(
        isinstance(routes, Mapping) and tuple(routes) == expected_labels,
        "gradient route labels differ",
    )
    _require(
        isinstance(route_hashes, Mapping) and tuple(route_hashes) == expected_labels,
        "gradient route hashes differ",
    )
    expected_parameters = {name: parameter for name, parameter in named_parameters}
    _require(expected_parameters, "restored model has no named trainable parameters")

    route_names: dict[str, list[str]] = {}
    for label in expected_labels:
        route = routes[label]
        _require(isinstance(route, list) and route, f"{label} gradient route is empty")
        names = []
        for entry in route:
            _require(
                isinstance(entry, Mapping)
                and set(entry) == {"dtype", "name", "numel", "shape"},
                f"{label} gradient route entry is malformed",
            )
            name = entry["name"]
            _require(
                isinstance(name, str) and name in expected_parameters,
                f"{label} gradient route names an unknown parameter",
            )
            parameter = expected_parameters[name]
            _require(list(parameter.shape) == entry["shape"], f"{name} shape mismatch")
            _require(int(parameter.numel()) == entry["numel"], f"{name} size mismatch")
            _require(str(parameter.dtype) == entry["dtype"], f"{name} dtype mismatch")
            names.append(name)
        _require(len(names) == len(set(names)), f"{label} gradient route repeats names")
        _require(
            route_hashes[label] == _canonical_json_sha256(route),
            f"{label} gradient route hash mismatch",
        )
        route_names[label] = names

    expected_intersections = {}
    for index, left in enumerate(expected_labels):
        for right in expected_labels[index + 1 :]:
            right_names = set(route_names[right])
            expected_intersections[f"{left}__{right}"] = [
                name for name in route_names[left] if name in right_names
            ]
    _require(
        intersections == expected_intersections,
        "gradient route intersections do not match route entries",
    )
    for pair, names in expected_intersections.items():
        expected_empty = method == STOPGRAD_METHOD and pair == "FM__Reconstruction"
        _require(
            bool(names) is not expected_empty,
            f"unexpected shared gradient pathway: {pair}",
        )

    stage_prefixes = {
        "observation": "nets.pipeline.stages.0.",
        "encoder": "nets.pipeline.stages.3.",
        "field": "nets.pipeline.stages.5.",
        "decoder": "nets.pipeline.stages.6.",
    }
    expected_reachability = {
        "FM": ("observation", "encoder", "field"),
        "Reconstruction": ("encoder", "decoder"),
        "ActionVelocity": ("observation", "encoder", "field", "decoder"),
    }
    if method == STOPGRAD_METHOD:
        expected_reachability["FM"] = ("observation", "field")
    elif method == GRAPH_METHOD:
        del expected_reachability["Reconstruction"]
    elif method == LIKELIHOOD_METHOD:
        expected_reachability = {
            "InteriorBridgeNLL": ("observation", "encoder", "field"),
            "BoundaryNLL": ("observation", "encoder", "field", "decoder"),
        }
    for label, active_groups in expected_reachability.items():
        for group, prefix in stage_prefixes.items():
            observed = any(name.startswith(prefix) for name in route_names[label])
            expected = group in active_groups
            _require(
                observed is expected,
                f"{label} gradient reachability for {group} is {observed}, "
                f"expected {expected}",
            )
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    _require(
        manifest["manifest_sha256"] == _canonical_json_sha256(core),
        "gradient route manifest hash mismatch",
    )
    return {
        "manifest_sha256": manifest["manifest_sha256"],
        "route_parameter_counts": {
            label: sum(expected_parameters[name].numel() for name in names)
            for label, names in route_names.items()
        },
        "route_sha256": dict(route_hashes),
        "shared_path_parameter_counts": {
            pair: sum(expected_parameters[name].numel() for name in names)
            for pair, names in expected_intersections.items()
        },
    }


def _validate_checkpoint_loss_schedule(
    loss_schedule: Any,
    config: DictConfig | None,
    *,
    reconstruction_weight: float,
    flow_weight: float,
) -> int:
    _require(config is not None, "checkpoint loss schedule needs its exact config")
    warmup_steps = OmegaConf.select(
        config, "model.reconstruction_only_warmup_steps", default=0
    )
    _require(
        isinstance(warmup_steps, int)
        and not isinstance(warmup_steps, bool)
        and warmup_steps in (0, 1),
        "two-update smoke requires a configured reconstruction warmup of 0 or 1",
    )
    _require(
        loss_schedule
        == {
            "joint_objective_begins_at_global_step": warmup_steps,
            "reconstruction_only_optimizer_steps": warmup_steps,
            "joint_flow_weight": flow_weight,
            "joint_reconstruction_weight": reconstruction_weight,
            "joint_action_velocity_weight": 1.0,
            "schema_version": 1,
        },
        f"unexpected Action Flow loss schedule: {loss_schedule}",
    )
    return warmup_steps


def _validate_checkpoint(
    run_dir: Path,
    *,
    reconstruction_weight: float,
    flow_weight: float,
    method: str = LEGACY_METHOD,
    config: DictConfig | None = None,
) -> dict[str, Any]:
    checkpoint_dir = run_dir / "checkpoints"
    last_path = checkpoint_dir / "last.ckpt"
    _require(last_path.is_file(), f"missing smoke checkpoint: {last_path}")
    _require(last_path.is_symlink(), "last.ckpt must be a symlink")
    immutable_path = last_path.resolve(strict=True)
    _require(
        immutable_path.parent == checkpoint_dir.resolve(),
        "last.ckpt target escapes the checkpoint directory",
    )
    _require(
        immutable_path.name != last_path.name and immutable_path.suffix == ".ckpt",
        "last.ckpt does not resolve to an immutable checkpoint",
    )

    payload = torch.load(last_path, map_location="cpu", weights_only=False, mmap=True)
    _require(isinstance(payload, Mapping), "last checkpoint is not a mapping")
    _require(payload.get("global_step") == 2, "last checkpoint global_step is not 2")
    state_dict = payload.get("state_dict")
    optimizer_states = payload.get("optimizer_states")
    scheduler_states = payload.get("lr_schedulers")
    loops = payload.get("loops")
    gradient_route_manifest = payload.get("action_flow_gradient_route_manifest")
    loss_schedule = payload.get("action_flow_loss_schedule")
    likelihood_contract = payload.get("action_flow_likelihood_contract")
    _require(
        isinstance(state_dict, Mapping) and state_dict, "checkpoint has no state_dict"
    )
    _require(
        isinstance(optimizer_states, list) and len(optimizer_states) == 1,
        "checkpoint must contain exactly one optimizer state",
    )
    _require(
        isinstance(optimizer_states[0], Mapping)
        and bool(optimizer_states[0].get("state")),
        "AdamW optimizer state is empty",
    )
    _require(
        isinstance(scheduler_states, list) and len(scheduler_states) == 1,
        "checkpoint must contain exactly one scheduler state",
    )
    _require(isinstance(loops, Mapping) and loops, "checkpoint loop state is empty")
    if loss_schedule is not None:
        expected_warmup_steps = _validate_checkpoint_loss_schedule(
            loss_schedule,
            config,
            reconstruction_weight=reconstruction_weight,
            flow_weight=flow_weight,
        )
    state_tensors, state_scalars = _finite_tree(state_dict, "checkpoint.state_dict")
    optimizer_tensors, optimizer_scalars = _finite_tree(
        optimizer_states, "checkpoint.optimizer_states"
    )
    scheduler_tensors, scheduler_scalars = _finite_tree(
        scheduler_states, "checkpoint.lr_schedulers"
    )
    _require(state_tensors > 0, "checkpoint state_dict contains no tensors")

    immutable_payload = torch.load(
        immutable_path, map_location="cpu", weights_only=False, mmap=True
    )
    _require(
        isinstance(immutable_payload, Mapping)
        and immutable_payload.get("global_step") == 2,
        "immutable checkpoint is not step 2",
    )
    _require(
        immutable_payload.get("action_flow_gradient_route_manifest")
        == gradient_route_manifest,
        "immutable/last checkpoint gradient-route manifests differ",
    )
    _require(
        immutable_payload.get("action_flow_loss_schedule") == loss_schedule,
        "immutable/last checkpoint loss schedules differ",
    )
    del immutable_payload, payload

    wrapper_type = ActionFlowModelWrapper
    if method == LIKELIHOOD_METHOD:
        from egomimic.pl_utils.pl_model_action_flow_likelihood import (
            ActionFlowLikelihoodModelWrapper,
        )

        wrapper_type = ActionFlowLikelihoodModelWrapper
        _require(
            isinstance(likelihood_contract, Mapping),
            "likelihood checkpoint lacks scientific contract",
        )
        _require(
            likelihood_contract
            == {
                "schema_version": 1,
                "method": LIKELIHOOD_METHOD,
                "num_levels": 32,
                "interior_samples_per_content": 14,
                "sigma_min": 0.1,
                "sigma_max": 1.0,
                "rho": 0.95,
                "tau": 0.02,
                "reduction": "sum_chunk_coordinates_mean_examples_constant_free_gaussian_bound",
                "learned_reference_targets": "attached",
                "sampler": "stochastic_reverse_gaussian_chain_with_action_output_noise",
                "clean_reconstruction_objective": False,
                "latent_fm_objective": False,
            },
            "likelihood checkpoint scientific contract mismatch",
        )
    try:
        restored = wrapper_type.load_from_checkpoint(
            last_path,
            map_location="cpu",
            strict=True,
            weights_only=False,
        )
    except Exception as error:
        raise SmokeVerificationError(
            f"strict ActionFlowModelWrapper reload failed: {last_path}"
        ) from error
    _require(
        type(restored) is wrapper_type,
        f"checkpoint restored unexpected wrapper {type(restored)!r}",
    )
    if loss_schedule is not None:
        _require(
            restored.reconstruction_only_warmup_steps == expected_warmup_steps,
            "strict reload lost the reconstruction-only warmup",
        )
    parameter_count = sum(parameter.numel() for parameter in restored.parameters())
    if method in (LEGACY_METHOD, STOPGRAD_METHOD):
        _require(
            parameter_count == EXPECTED_PARAMETER_COUNT,
            f"parameter count mismatch: {parameter_count} != {EXPECTED_PARAMETER_COUNT}",
        )
    else:
        _require(config is not None, "typed candidate reload needs its exact config")
        _validate_dimensions_and_modules(config, tuple(restored.model.pipeline.stages))
    # These properties fail closed on duplicated or disconnected owners.
    if method == LIKELIHOOD_METHOD:
        stages = restored.model.pipeline.stages
        owners = (stages[3].mean_encoder, stages[5].field, stages[6].decoder)
    else:
        owners = (restored.encoder_e, restored.field_v, restored.decoder_g)
    _require(
        len({id(owner) for owner in owners}) == 3, "Action Flow owners are aliased"
    )
    trainable = tuple(
        (name, parameter)
        for name, parameter in restored.nets.named_parameters(
            prefix="nets", remove_duplicate=True
        )
        if parameter.requires_grad
    )
    gradient_routes = _validate_gradient_route_manifest(
        gradient_route_manifest, trainable, method
    )
    del restored

    return {
        "checkpoint_sha256": _sha256(last_path),
        "file_size_bytes": last_path.stat().st_size,
        "gradient_routes": gradient_routes,
        "global_step": 2,
        "loss_schedule": loss_schedule,
        "likelihood_contract": likelihood_contract,
        "immutable_checkpoint_path": str(immutable_path),
        "immutable_checkpoint_sha256": _sha256(immutable_path),
        "optimizer_state_count": 1,
        "parameter_count": parameter_count,
        "scheduler_state_count": 1,
        "strict_checkpoint_reload": "passed",
        "finite_scan": {
            "optimizer_numeric_scalars": optimizer_scalars,
            "optimizer_tensors": optimizer_tensors,
            "scheduler_numeric_scalars": scheduler_scalars,
            "scheduler_tensors": scheduler_tensors,
            "state_dict_numeric_scalars": state_scalars,
            "state_dict_tensors": state_tensors,
        },
    }


def _wandb_history(path: Path) -> tuple[dict[int, dict[str, float]], int]:
    try:
        from wandb.proto import wandb_internal_pb2
        from wandb.sdk.internal.datastore import DataStore
    except ImportError as error:
        raise SmokeVerificationError(
            "W&B runtime is required for smoke proof"
        ) from error

    store = DataStore()
    store.open_for_scan(str(path))
    rows: dict[int, dict[str, float]] = {}
    exits: list[int] = []
    try:
        while (payload := store.scan_data()) is not None:
            record = wandb_internal_pb2.Record()
            record.ParseFromString(payload)
            kind = record.WhichOneof("record_type")
            if kind == "exit":
                exits.append(int(record.exit.exit_code))
            if kind != "history":
                continue
            decoded: dict[str, Any] = {}
            for item in record.history.item:
                key = item.key or ".".join(item.nested_key)
                try:
                    decoded[key] = json.loads(item.value_json)
                except (json.JSONDecodeError, TypeError):
                    continue
            if "trainer/global_step" not in decoded:
                continue
            step = int(decoded["trainer/global_step"])
            for key, value in decoded.items():
                if not key.startswith(("Train/", "Valid/", "Optimizer/", "Timing/")):
                    continue
                if isinstance(value, bool):
                    continue
                try:
                    rows.setdefault(step, {})[key] = float(value)
                except (TypeError, ValueError):
                    continue
    finally:
        store.close()
    _require(bool(exits), "W&B stream contains no exit record")
    _require(exits[-1] == 0, f"W&B final exit code is not zero: {exits}")
    return rows, exits[-1]


def _metric(row: Mapping[str, float], name: str) -> float | None:
    for candidate in (name, f"{name}_step", f"{name}_epoch"):
        if candidate in row:
            return float(row[candidate])
    return None


def _complete_row(
    rows: Mapping[int, Mapping[str, float]],
    required: Sequence[str],
    *,
    minimum_step: int,
    label: str,
) -> tuple[int, dict[str, float]]:
    for step in sorted(rows, reverse=True):
        if step < minimum_step:
            continue
        values = {name: _metric(rows[step], name) for name in required}
        if all(value is not None for value in values.values()):
            concrete = {name: float(value) for name, value in values.items()}
            _require(
                all(math.isfinite(value) for value in concrete.values()),
                f"{label} contains non-finite metrics: {concrete}",
            )
            return step, concrete
    raise SmokeVerificationError(
        f"W&B has no complete {label} row at step >= {minimum_step}: {tuple(required)}"
    )


def _validate_history(
    rows: Mapping[int, Mapping[str, float]],
    *,
    reconstruction_weight: float = 1.0,
    flow_weight: float = 1.0,
    expect_reconstruction_warmup: bool = False,
    method: str = LEGACY_METHOD,
) -> dict[str, Any]:
    component_names = (
        "TotalLoss",
        "FlowMatchingLoss",
        "ReconstructionLoss",
        "ReconstructionL1",
        "ActionVelocityLoss",
    )
    if method == LIKELIHOOD_METHOD:
        component_names = ("TotalLoss", "InteriorBridgeNLL", "BoundaryNLL")
    components = tuple(f"Train/ActionFlow/{name}" for name in component_names)
    per_source_components = tuple(f"{name}/{SOURCE_LABEL}" for name in components)
    gradient_labels = (
        ("InteriorBridgeNLL", "BoundaryNLL")
        if method == LIKELIHOOD_METHOD
        else (
            ("FM", "ActionVelocity")
            if method == GRAPH_METHOD
            else ("FM", "Reconstruction", "ActionVelocity")
        )
    )
    gradient_pairs = tuple(
        f"{left}__{right}"
        for i, left in enumerate(gradient_labels)
        for right in gradient_labels[i + 1 :]
    )
    telemetry = [
        *(f"Train/ActionFlow/GradientNorm/{label}" for label in gradient_labels),
        *(
            f"Train/ActionFlow/GradientParameterCount/{label}"
            for label in gradient_labels
        ),
        *(f"Train/ActionFlow/GradientCosine/{pair}" for pair in gradient_pairs),
        *(f"Train/ActionFlow/GradientCosineDefined/{pair}" for pair in gradient_pairs),
        *(
            f"Train/ActionFlow/GradientIntersectionParameterCount/{pair}"
            for pair in gradient_pairs
        ),
        "Train/MSE",
        f"Train/MSE/{SOURCE_LABEL}",
        "Train/ActionFlow/Compute/FieldForwardCallsPerStep",
        "Train/ActionFlow/Compute/FieldSampleEquivalentsPerStep",
        "Train/ActionFlow/Compute/DecoderJVPCallsPerStep",
        "Train/ActionFlow/Compute/PeakAllocatedBytes",
        "Train/ActionFlow/Schedule/ReconstructionOnly",
        "Train/ActionFlow/Schedule/EffectiveFlowWeight",
        "Train/ActionFlow/Schedule/EffectiveActionVelocityWeight",
    ]
    if method == LIKELIHOOD_METHOD:
        telemetry = [name for name in telemetry if "/Schedule/" not in name]
    train_step, train = _complete_row(
        rows,
        (*components, *per_source_components, *telemetry),
        minimum_step=1,
        label="Action Flow training/gradient telemetry",
    )
    for name in telemetry:
        empty_pair = method == STOPGRAD_METHOD and name.endswith("/FM__Reconstruction")
        if (
            "GradientNorm" in name
            or "GradientParameterCount" in name
            or "IntersectionParameterCount" in name
        ):
            _require(
                train[name] == 0.0 if empty_pair else train[name] > 0.0,
                f"gradient reachability differs: {name}",
            )
        if "GradientCosineDefined" in name:
            _require(
                train[name] == (0.0 if empty_pair else 1.0),
                f"gradient cosine defined flag differs: {name}",
            )
        if "GradientCosine/" in name:
            _require(-1.0 <= train[name] <= 1.0, f"gradient cosine is invalid: {name}")
    for name, expected in (
        (
            "Train/ActionFlow/Compute/FieldForwardCallsPerStep",
            2.0 if method == STOPGRAD_METHOD else 1.0,
        ),
        (
            "Train/ActionFlow/Compute/FieldSampleEquivalentsPerStep",
            (
                28.0
                if method == STOPGRAD_METHOD
                else 15.0 if method == LIKELIHOOD_METHOD else 14.0
            ),
        ),
        (
            "Train/ActionFlow/Compute/DecoderJVPCallsPerStep",
            0.0 if method == LIKELIHOOD_METHOD else 1.0,
        ),
    ):
        _require(train[name] == expected, f"unexpected compute telemetry: {name}")
    _require(
        train["Train/ActionFlow/Compute/PeakAllocatedBytes"] > 0.0,
        "CUDA peak allocation telemetry is empty",
    )
    _require(
        method == LIKELIHOOD_METHOD
        or train["Train/ActionFlow/Schedule/ReconstructionOnly"] == 0.0,
        "latest smoke optimizer step is not joint",
    )
    _require(
        method == LIKELIHOOD_METHOD
        or (
            math.isclose(
                train["Train/ActionFlow/Schedule/EffectiveFlowWeight"],
                flow_weight,
                rel_tol=0.0,
                abs_tol=1.0e-6,
            )
            and train["Train/ActionFlow/Schedule/EffectiveActionVelocityWeight"] == 1.0
        ),
        "joint smoke step did not enable both delayed objectives",
    )
    for suffix in ("", f"/{SOURCE_LABEL}"):
        expected_total = (
            (
                train[f"Train/ActionFlow/InteriorBridgeNLL{suffix}"]
                + train[f"Train/ActionFlow/BoundaryNLL{suffix}"]
            )
            if method == LIKELIHOOD_METHOD
            else (
                flow_weight * train[f"Train/ActionFlow/FlowMatchingLoss{suffix}"]
                + reconstruction_weight
                * train[f"Train/ActionFlow/ReconstructionLoss{suffix}"]
                + train[f"Train/ActionFlow/ActionVelocityLoss{suffix}"]
            )
        )
        _require(
            math.isclose(
                train[f"Train/ActionFlow/TotalLoss{suffix}"],
                expected_total,
                rel_tol=1.0e-5,
                abs_tol=1.0e-7,
            ),
            f"training weighted total is inconsistent for {suffix or 'macro'}",
        )

    warmup_step = None
    if expect_reconstruction_warmup:
        required_warmup = (*components, *per_source_components, *telemetry[-3:])
        for step in sorted(rows):
            values = {name: _metric(rows[step], name) for name in required_warmup}
            if not all(value is not None for value in values.values()):
                continue
            concrete = {name: float(value) for name, value in values.items()}
            if concrete["Train/ActionFlow/Schedule/ReconstructionOnly"] != 1.0:
                continue
            _require(
                concrete["Train/ActionFlow/Schedule/EffectiveFlowWeight"] == 0.0
                and concrete["Train/ActionFlow/Schedule/EffectiveActionVelocityWeight"]
                == 0.0,
                "warmup smoke step enabled a delayed objective",
            )
            for suffix in ("", f"/{SOURCE_LABEL}"):
                expected_total = (
                    reconstruction_weight
                    * concrete[f"Train/ActionFlow/ReconstructionLoss{suffix}"]
                )
                _require(
                    math.isclose(
                        concrete[f"Train/ActionFlow/TotalLoss{suffix}"],
                        expected_total,
                        rel_tol=1.0e-5,
                        abs_tol=1.0e-7,
                    ),
                    "reconstruction-only smoke total includes a delayed loss",
                )
            warmup_step = step
            break
        _require(warmup_step is not None, "W&B contains no reconstruction-only step")

    validity = []
    for base in (
        "Valid/MSE",
        "Valid/Native_MSE",
        "Valid/EnergyScore@32",
        "Valid/EnergyScoreAccuracy@32",
        "Valid/EnergyScoreDiversity@32",
    ):
        validity.extend((base, f"{base}/{SOURCE_LABEL}"))
    validity.extend(f"Valid/ActionFlow/{name}" for name in component_names)
    diagnostics = (
        "Valid/ActionFlow/CleanReconstructionMSE",
        "Valid/ActionFlow/CleanReconstructionNativeMSE",
        "Valid/ActionFlow/Latent/clean/RMS",
        "Valid/ActionFlow/Latent/generated/EffectiveRank",
        "Valid/ActionFlow/DecoderJacobian/clean/spectral_norm_mean",
        "Valid/ActionFlow/DecodedFullNoise/token_radius_mean",
        "Valid/ActionFlow/DenoisingTrajectory/LatentMSE/t1000",
        "Valid/ActionFlow/DenoisingTrajectory/DecodedMSE/t0000",
        "Valid/ActionFlow/DenoisingTrajectory/DecodedNativeMSE/t0000",
        "Valid/ActionFlow/Alignment/FinalLatentCosine/t1000",
        "Valid/ActionFlow/Alignment/CKA/encoder_00__field_00/t0500",
        "Valid/ActionFlow/Alignment/CKNNA/encoder_00__field_00/t0500",
    )
    if method == LIKELIHOOD_METHOD:
        diagnostics = ()
    elif method == GRAPH_METHOD:
        diagnostics = tuple(name for name in diagnostics if "/Alignment/CK" not in name)
    diagnostics = (
        *diagnostics,
        *(
            f"{name}/{SOURCE_LABEL}"
            for name in diagnostics
            if "NativeMSE" in name or "decoded_native_mse" in name
        ),
    )
    valid_step, valid = _complete_row(
        rows,
        (*validity, *diagnostics),
        minimum_step=1,
        label="scheduled validation",
    )
    expected_valid_total = (
        (
            valid["Valid/ActionFlow/InteriorBridgeNLL"]
            + valid["Valid/ActionFlow/BoundaryNLL"]
        )
        if method == LIKELIHOOD_METHOD
        else (
            flow_weight * valid["Valid/ActionFlow/FlowMatchingLoss"]
            + reconstruction_weight * valid["Valid/ActionFlow/ReconstructionLoss"]
            + valid["Valid/ActionFlow/ActionVelocityLoss"]
        )
    )
    _require(
        math.isclose(
            valid["Valid/ActionFlow/TotalLoss"],
            expected_valid_total,
            rel_tol=1.0e-5,
            abs_tol=1.0e-7,
        ),
        "validation weighted total is inconsistent",
    )
    return {
        "train_step": train_step,
        "warmup_step": warmup_step,
        "valid_step": valid_step,
        "train": train,
        "valid": valid,
    }


def _artifact_root(config: DictConfig, path: str, run_dir: Path) -> Path:
    root = Path(str(_select(config, path))).expanduser()
    if not root.is_absolute():
        root = run_dir / root
    return root.resolve()


def _step_two_artifact(root: Path, *, label: str) -> tuple[Path, Mapping[str, Any]]:
    execution = artifact_execution_identity()
    if execution is not None:
        attempt_root = root / (
            f"job-{execution['slurm_job_id']}"
            f"-restart-{execution['slurm_restart_count']}"
        )
        if attempt_root.is_dir():
            root = attempt_root
        else:
            # Posthoc verification may run in a separate CPU allocation.
            # Its scheduler ID is not the job that produced these artifacts.
            execution = None
    _require(root.is_dir(), f"missing {label} artifact root: {root}")
    leftovers = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and (path.suffix in {".tmp", ".temporary"} or ".temporary" in path.name)
    ]
    _require(not leftovers, f"unfinished {label} artifacts: {leftovers}")
    candidates = sorted(
        [
            *root.glob("epoch-*-step-2/rank-0-batch-*.pt"),
            *(
                []
                if execution is not None
                else root.glob("job-*-restart-*/epoch-*-step-2/rank-0-batch-*.pt")
            ),
        ]
    )
    _require(
        len(candidates) == 1, f"expected one step-2 {label} artifact: {candidates}"
    )
    payload = torch.load(candidates[0], map_location="cpu", weights_only=False)
    _require(isinstance(payload, Mapping), f"{label} artifact is not a mapping")
    _require(payload.get("global_step") == 2, f"{label} artifact is not step 2")
    namespace = candidates[0].parent.parent.name
    if namespace.startswith("job-"):
        recorded = payload.get("execution")
        _require(
            isinstance(recorded, Mapping), f"{label} artifact lacks execution identity"
        )
        expected_namespace = (
            f"job-{recorded.get('slurm_job_id')}"
            f"-restart-{recorded.get('slurm_restart_count')}"
        )
        _require(
            namespace == expected_namespace,
            f"{label} artifact execution identity mismatch",
        )
    _finite_tree(payload, f"{label} artifact")
    return candidates[0], payload


def _target_tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().float().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _validate_artifacts(
    *,
    config: DictConfig,
    run_dir: Path,
    identities: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    split_sha256 = str(identities["split_manifest_sha256"])
    energy_root = _artifact_root(config, "evaluator.artifact_root", run_dir)
    energy_path, energy = _step_two_artifact(energy_root, label="EnergyScore")
    _require(energy.get("schema_version") == 2, "wrong typed EnergyScore schema")
    _require(energy.get("metric") == "EnergyScore@32", "wrong EnergyScore metric")
    _require(energy.get("sample_count") == 32, "EnergyScore sample count is not 32")
    _require(
        energy.get("deterministic_seed") == 420_042,
        "EnergyScore deterministic seed differs",
    )
    seeds = energy.get("seed_bank")
    _require(
        isinstance(seeds, list) and len(seeds) == 32 and len(set(seeds)) == 32,
        "EnergyScore artifact does not contain 32 unique seeds",
    )
    seed_hash = str(config.evaluator.seed_bank_sha256)
    _require(
        energy.get("seed_bank_sha256") == seed_hash,
        "EnergyScore seed-bank hash mismatch",
    )
    expected_distance = usocket_energy_distance_metadata(USOCKET_ENERGY_DISTANCE_CONFIG)
    _require(
        energy.get("distance") == expected_distance, "EnergyScore distance differs"
    )
    _require(
        set(energy.get("domains", {})) == {SOURCE_LABEL},
        "EnergyScore artifact source mismatch",
    )
    domain = energy["domains"][SOURCE_LABEL]
    predictions = domain.get("predictions")
    targets = domain.get("targets")
    native_predictions = domain.get("native_predictions")
    native_targets = domain.get("native_targets")
    _require(
        torch.is_tensor(predictions) and tuple(predictions.shape) == (32, 16, 16, 4),
        "EnergyScore predictions do not have shape (32, 16, 16, 4)",
    )
    _require(
        torch.is_tensor(targets) and tuple(targets.shape) == (16, 16, 4),
        "EnergyScore targets do not have shape (16, 16, 4)",
    )
    _require(
        torch.is_tensor(native_predictions)
        and tuple(native_predictions.shape) == (32, 16, 16, 3),
        "typed EnergyScore native predictions have the wrong shape",
    )
    _require(
        torch.is_tensor(native_targets) and tuple(native_targets.shape) == (16, 16, 3),
        "typed EnergyScore native targets have the wrong shape",
    )
    conditions = domain.get("condition_ids")
    _require(
        isinstance(conditions, list) and len(conditions) == 16,
        "EnergyScore condition identities are incomplete",
    )
    for index, condition in enumerate(conditions):
        _require(
            isinstance(condition, Mapping)
            and condition.get("batch_position") == index
            and isinstance(condition.get("episode_hash"), str)
            and bool(condition["episode_hash"])
            and isinstance(condition.get("frame_index"), int)
            and condition["frame_index"] >= 0,
            "EnergyScore condition identity order differs",
        )
        recorded_target = _validate_sha256(
            str(condition.get("normalized_target_sha256", "")),
            f"EnergyScore condition {index} target SHA-256",
        )
        _require(
            recorded_target == _target_tensor_sha256(targets[index]),
            f"EnergyScore condition {index} target identity mismatch",
        )
    expected_energy_view = _plain_mapping(
        config.evaluator.energy_score_validation_view,
        label="EnergyScore validation view",
    )
    _require(
        energy.get("validation_view") == expected_energy_view,
        "EnergyScore validation view differs",
    )
    energy_identity = energy.get("identity")
    _require(isinstance(energy_identity, Mapping), "typed EnergyScore identity missing")
    _require(
        energy.get("identity_sha256") == _canonical_json_sha256(energy_identity),
        "typed EnergyScore identity hash mismatch",
    )
    _require(
        energy_identity.get("schema") == "energy-score-validation-artifact/v2",
        "typed EnergyScore identity schema differs",
    )
    _require(
        energy_identity.get("metric")
        == {
            "name": "EnergyScore@32",
            "sample_count": 32,
            "deterministic_seed": 420_042,
        },
        "typed EnergyScore metric identity differs",
    )
    for artifact_key, identity_key in (
        ("source_commit", "repo_head"),
        ("normalization_sha256", "normalization_sha256"),
        ("split_manifest_sha256", "split_manifest_sha256"),
    ):
        _require(
            energy_identity.get(artifact_key) == identities[identity_key],
            f"typed EnergyScore {artifact_key} differs",
        )
    resolved_config = (run_dir / ".hydra/config.yaml").resolve(strict=True)
    try:
        artifact_config = Path(
            str(energy_identity.get("resolved_config_path", ""))
        ).resolve(strict=True)
    except FileNotFoundError as error:
        raise SmokeVerificationError(
            "typed EnergyScore resolved config is unavailable"
        ) from error
    _require(
        artifact_config == resolved_config
        and energy_identity.get("resolved_config_sha256")
        == identities["config_sha256"],
        "typed EnergyScore resolved-config binding differs",
    )
    expected_wandb = {
        "entity": str(_select(config, "logger.wandb.entity")),
        "project": str(_select(config, "logger.wandb.project")),
        "run_id": str(_select(config, "logger.wandb.id")),
    }
    _require(
        energy_identity.get("wandb") == expected_wandb,
        "typed EnergyScore W&B identity differs",
    )
    expected_content = {
        "manifest_path": identities["content_manifest_path"],
        "manifest_sha256": identities["content_manifest_sha256"],
        "aggregate_sha256": identities["dataset_content_aggregate_sha256"],
    }
    _require(
        energy_identity.get("dataset_content") == expected_content,
        "typed EnergyScore dataset-content identity differs",
    )
    _require(
        energy_identity.get("distance") == expected_distance,
        "typed EnergyScore identity distance differs",
    )
    _require(
        energy_identity.get("seed_bank_sha256") == seed_hash,
        "typed EnergyScore identity seed bank differs",
    )
    _require(
        energy_identity.get("validation_view") == expected_energy_view,
        "typed EnergyScore identity validation view differs",
    )
    _require(
        energy_identity.get("validation_conditions") == {SOURCE_LABEL: conditions},
        "typed EnergyScore validation conditions differ",
    )
    binding = energy_identity.get("checkpoint_binding")
    _require(
        isinstance(binding, Mapping)
        and binding.get("global_step") == checkpoint["global_step"] == 2
        and binding.get("checkpoint_sha256") is None
        and binding.get("sha256_status") == ENERGY_CHECKPOINT_STATUS,
        "typed EnergyScore checkpoint binding differs",
    )

    if action_flow_method(config) == LIKELIHOOD_METHOD:
        # No clean-latent/ODE artifact exists for the discrete likelihood model.
        # Its actual learned-reference gradient routes are checkpoint-bound.
        _require(
            checkpoint.get("likelihood_contract"),
            "likelihood scientific checkpoint receipt missing",
        )
        return {
            "energy_score": {
                "path": str(energy_path),
                "sha256": _sha256(energy_path),
                "checkpoint_global_step": checkpoint["global_step"],
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "identity_sha256": energy["identity_sha256"],
            },
            "method_specific": {
                "likelihood_contract": checkpoint["likelihood_contract"],
                "gradient_route_manifest_sha256": checkpoint["gradient_routes"][
                    "manifest_sha256"
                ],
                "ode_diagnostics": "not_applicable_discrete_gaussian_reverse_chain",
            },
        }

    diagnostic_root = _artifact_root(
        config, "evaluator.action_flow_diagnostics.artifact_root", run_dir
    )
    diagnostic_path, diagnostic = _step_two_artifact(
        diagnostic_root, label="Action Flow diagnostics"
    )
    _require(
        diagnostic.get("metric") == "ActionFlowValidationDiagnostics",
        "wrong Action Flow diagnostic metric",
    )
    _require(diagnostic.get("schema_version") == 1, "wrong diagnostic schema")
    _require(
        set(diagnostic.get("sources", {})) == {SOURCE_LABEL},
        "Action Flow diagnostic source mismatch",
    )
    _require(
        diagnostic.get("sampler", {}).get("name") == "reverse_euler",
        "Action Flow diagnostic sampler mismatch",
    )
    identity = diagnostic.get("identity")
    identity_hash = diagnostic.get("identity_sha256")
    _require(isinstance(identity, Mapping), "diagnostic identity is missing")
    _require(
        _canonical_json_sha256(identity) == identity_hash,
        "Action Flow diagnostic identity hash mismatch",
    )
    _require(
        identity.get("native_error") == USOCKET_NATIVE_ERROR_CONFIG,
        "Action Flow diagnostic native-error identity differs",
    )
    _require(
        diagnostic.get("statistics", {}).get("native_action_error")
        == USOCKET_NATIVE_ERROR_CONFIG,
        "Action Flow diagnostic native-error statistics differ",
    )
    expected_diagnostic_view = _plain_mapping(
        config.evaluator.action_flow_diagnostics.validation_view,
        label="Action Flow diagnostic validation view",
    )
    _require(
        identity.get("validation_view") == expected_diagnostic_view
        and diagnostic.get("validation_view") == expected_diagnostic_view,
        "Action Flow diagnostic validation view differs",
    )
    diagnostic_provenance = identity.get("provenance")
    _require(
        isinstance(diagnostic_provenance, Mapping)
        and diagnostic.get("provenance") == diagnostic_provenance,
        "Action Flow diagnostic provenance differs",
    )
    _require(
        diagnostic_provenance.get("source_commit") == identities["repo_head"]
        and diagnostic_provenance.get("normalization_sha256")
        == identities["normalization_sha256"]
        and diagnostic_provenance.get("split_manifest_sha256") == split_sha256
        and diagnostic_provenance.get("dataset_content")
        == {
            "manifest_sha256": identities["content_manifest_sha256"],
            "aggregate_sha256": identities["dataset_content_aggregate_sha256"],
        },
        "Action Flow diagnostic provenance identity differs",
    )
    source_payload = diagnostic["sources"][SOURCE_LABEL]
    computed = source_payload.get("computed")
    _require(isinstance(computed, Mapping), "diagnostic computed payload missing")
    clean_native = computed.get("clean_reconstruction_native_mse_by_condition")
    trajectory_native = computed.get("trajectory_decoded_native_mse_by_condition")
    _require(
        torch.is_tensor(clean_native) and tuple(clean_native.shape) == (16,),
        "diagnostic clean native errors have wrong shape",
    )
    _require(
        torch.is_tensor(trajectory_native)
        and tuple(trajectory_native.shape) == (17, 16),
        "diagnostic trajectory native errors have wrong shape",
    )
    fixed_metrics = computed.get("fixed_level_metrics")
    _require(
        isinstance(fixed_metrics, Mapping) and set(fixed_metrics) == set(NATIVE_LEVELS),
        "diagnostic fixed-level bins differ",
    )
    for label in NATIVE_LEVELS:
        values = fixed_metrics[label]
        _require(isinstance(values, Mapping), f"diagnostic {label} is malformed")
        for metric in (
            "state_decoded_native_mse",
            "predicted_clean_decoded_native_mse",
            "final_decoded_native_mse",
        ):
            value = values.get(metric)
            _require(
                torch.is_tensor(value) and tuple(value.shape) == (16,),
                f"diagnostic {metric}/{label} has wrong shape",
            )
    sidecar = Path(f"{diagnostic_path}.sha256")
    _require(sidecar.is_file(), f"missing diagnostic SHA sidecar: {sidecar}")
    sidecar_payload = json.loads(sidecar.read_text())
    diagnostic_sha256 = _sha256(diagnostic_path)
    _require(
        sidecar_payload.get("sha256") == diagnostic_sha256
        and sidecar_payload.get("artifact") == diagnostic_path.name
        and sidecar_payload.get("identity_sha256") == identity_hash,
        "Action Flow diagnostic sidecar mismatch",
    )

    return {
        "action_flow_diagnostics": {
            "path": str(diagnostic_path),
            "sha256": diagnostic_sha256,
            "sidecar_path": str(sidecar),
            "sidecar_sha256": _sha256(sidecar),
        },
        "energy_score": {
            "path": str(energy_path),
            "sha256": _sha256(energy_path),
            "checkpoint_global_step": checkpoint["global_step"],
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "identity_sha256": energy["identity_sha256"],
        },
    }


def _validate_preflight(
    *,
    run_dir: Path,
    expected_sha256: str | None,
    expected_head: str,
    experiment: str,
    split_sha256: str,
    normalization_sha256: str,
    content_manifest_sha256: str,
    dataset_content_aggregate_sha256: str,
) -> dict[str, Any] | None:
    if expected_sha256 is None:
        return None
    expected_sha256 = _validate_sha256(expected_sha256, "preflight result SHA-256")
    candidates = sorted(run_dir.glob("provenance/restart-*/PREFLIGHT_RESULT.json"))
    _require(candidates, "expected preflight result was not copied into run provenance")
    hashes = {_sha256(path) for path in candidates}
    _require(
        hashes == {expected_sha256},
        f"copied preflight result SHA-256 mismatch: {sorted(hashes)}",
    )
    payload = json.loads(candidates[0].read_text())
    _require(payload.get("status") == "PASS", "preflight result did not pass")
    source = payload.get("source")
    _require(
        isinstance(source, Mapping) and source.get("head") == expected_head,
        "preflight source HEAD mismatch",
    )
    _require(payload.get("experiment") == experiment, "preflight experiment mismatch")
    _require(
        payload.get("split_manifest_sha256") == split_sha256,
        "preflight split identity mismatch",
    )
    _require(
        payload.get("normalization_sha256") == normalization_sha256,
        "preflight normalization identity mismatch",
    )
    _require(
        payload.get("content_manifest_sha256") == content_manifest_sha256,
        "preflight dataset content manifest identity mismatch",
    )
    _require(
        payload.get("dataset_content_aggregate_sha256")
        == dataset_content_aggregate_sha256,
        "preflight dataset aggregate content identity mismatch",
    )
    return {
        "path": str(candidates[0]),
        "sha256": expected_sha256,
        "copy_count": len(candidates),
    }


def _validate_gpu_probes(
    run_dir: Path, gpu_profile: str = "h100-h200"
) -> list[dict[str, Any]]:
    _require(gpu_profile in GPU_PROFILES, "unsupported smoke GPU profile")
    candidates = sorted(run_dir.glob("provenance/restart-*/gpu_probe.json"))
    _require(candidates, "smoke has no scheduled one-GPU BF16 probe")
    records = []
    for path in candidates:
        payload = json.loads(path.read_text())
        _require(payload.get("status") == "PASSED", f"GPU probe did not pass: {path}")
        _require(
            payload.get("world_size") == 1, f"GPU probe was not world size 1: {path}"
        )
        _require(payload.get("bf16_supported") is True, f"BF16 is unsupported: {path}")
        compute = payload.get("bf16_forward_backward")
        _require(
            isinstance(compute, Mapping)
            and compute.get("finite") is True
            and compute.get("dtype") == "torch.bfloat16"
            and compute.get("gradient_dtype") == "torch.bfloat16",
            f"GPU probe lacks finite BF16 forward/backward: {path}",
        )
        nccl = payload.get("nccl")
        _require(
            isinstance(nccl, Mapping)
            and nccl.get("world_size") == 1
            and nccl.get("all_reduce") == 1.0
            and nccl.get("destroyed") is True,
            f"GPU probe lacks one-rank NCCL proof: {path}",
        )
        gpu_name = str(payload.get("gpu_name", ""))
        _require(
            gpu_name_allowed(gpu_name, gpu_profile),
            f"smoke GPU does not match {gpu_profile!r}: {gpu_name!r}",
        )
        record = {"gpu_name": gpu_name, "path": str(path), "sha256": _sha256(path)}
        if gpu_profile == "smoke-bf16":
            contract_path = path.with_name("SLURM_JOB_CONTRACT.json")
            _require(contract_path.is_file(), "alternate smoke GPU lacks Slurm proof")
            contract = json.loads(contract_path.read_text())
            expected = contract.get("expected", {})
            _require(
                contract.get("status") == "SLURM_JOB_CONTRACT_VALIDATED"
                and not contract.get("failures")
                and expected.get("gpu_profile") == gpu_profile
                and expected.get("run_kind") == "smoke",
                "alternate smoke GPU has invalid Slurm/profile proof",
            )
            validate_gpu_profile(gpu_profile, "smoke", expected.get("constraint", ""))
            record["slurm_contract_path"] = str(contract_path)
            record["slurm_contract_sha256"] = _sha256(contract_path)
        records.append(record)
    return records


def verify_smoke(
    run_dir: Path,
    *,
    experiment: str,
    expected_head: str,
    expected_config_sha256: str | None = None,
    expected_split_sha256: str | None = None,
    expected_normalization_sha256: str | None = None,
    expected_content_manifest_sha256: str | None = None,
    expected_dataset_content_aggregate_sha256: str | None = None,
    expected_reconstruction_weight: float | None = None,
    expected_flow_weight: float | None = None,
    expected_preflight_sha256: str | None = None,
    gpu_profile: str = "h100-h200",
) -> dict[str, Any]:
    run_dir = Path(run_dir).expanduser().resolve(strict=True)
    _require(run_dir.is_dir(), f"run directory is not a directory: {run_dir}")
    expected_head = str(expected_head).lower()
    config_path = run_dir / ".hydra/config.yaml"
    _require(config_path.is_file(), f"missing resolved Hydra config: {config_path}")
    config, identities = _validate_config(
        config_path=config_path,
        experiment=experiment,
        run_dir=run_dir,
        expected_head=expected_head,
        expected_config_sha256=expected_config_sha256,
        expected_split_sha256=expected_split_sha256,
        expected_normalization_sha256=expected_normalization_sha256,
        expected_content_manifest_sha256=expected_content_manifest_sha256,
        expected_dataset_content_aggregate_sha256=(
            expected_dataset_content_aggregate_sha256
        ),
    )
    approved_reconstruction_weight = APPROVED_EXPERIMENTS[experiment][1]
    approved_flow_weight = APPROVED_EXPERIMENTS[experiment][2]
    method = action_flow_method(config, experiment)
    if method == LIKELIHOOD_METHOD:
        _require(
            expected_reconstruction_weight is None and expected_flow_weight is None,
            "likelihood smoke does not accept FM/reconstruction weight arguments",
        )
    if expected_reconstruction_weight is not None:
        _require(
            math.isclose(
                float(expected_reconstruction_weight),
                approved_reconstruction_weight,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ),
            "requested reconstruction weight disagrees with approved experiment",
        )
    if expected_flow_weight is not None:
        _require(
            math.isclose(
                float(expected_flow_weight),
                approved_flow_weight,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ),
            "requested flow weight disagrees with approved experiment",
        )
    preflight = _validate_preflight(
        run_dir=run_dir,
        expected_sha256=expected_preflight_sha256,
        expected_head=expected_head,
        experiment=experiment,
        split_sha256=identities["split_manifest_sha256"],
        normalization_sha256=identities["normalization_sha256"],
        content_manifest_sha256=identities["content_manifest_sha256"],
        dataset_content_aggregate_sha256=identities["dataset_content_aggregate_sha256"],
    )
    _require(
        OmegaConf.select(config, "run_provenance.gpu_profile", default="h100-h200")
        == gpu_profile,
        "smoke GPU profile disagrees with resolved provenance",
    )
    gpu_probes = _validate_gpu_probes(run_dir, gpu_profile)
    checkpoint = _validate_checkpoint(
        run_dir,
        reconstruction_weight=approved_reconstruction_weight,
        flow_weight=approved_flow_weight,
        method=method,
        config=config,
    )
    expect_reconstruction_warmup = "warmup10k" in experiment
    if expect_reconstruction_warmup:
        _require(
            checkpoint["loss_schedule"]
            == {
                "joint_objective_begins_at_global_step": 1,
                "reconstruction_only_optimizer_steps": 1,
                "joint_flow_weight": approved_flow_weight,
                "joint_reconstruction_weight": approved_reconstruction_weight,
                "joint_action_velocity_weight": 1.0,
                "schema_version": 1,
            },
            "checkpoint does not preserve the scaled smoke loss schedule",
        )

    streams = [
        *run_dir.glob("wandb/run-*/run-*.wandb"),
        *run_dir.glob("wandb/offline-run-*/run-*.wandb"),
    ]
    _require(len(streams) == 1, f"expected exactly one W&B stream: {streams}")
    rows, exit_code = _wandb_history(streams[0])
    metrics = _validate_history(
        rows,
        reconstruction_weight=APPROVED_EXPERIMENTS[experiment][1],
        flow_weight=APPROVED_EXPERIMENTS[experiment][2],
        expect_reconstruction_warmup=expect_reconstruction_warmup,
        method=method,
    )
    artifacts = _validate_artifacts(
        config=config, run_dir=run_dir, identities=identities, checkpoint=checkpoint
    )

    result = {
        "artifacts": artifacts,
        "checkpoint": checkpoint,
        "experiment": experiment,
        "action_flow_method": method,
        "gpu_probes": gpu_probes,
        "gpu_profile": gpu_profile,
        "identities": identities,
        "metrics": metrics,
        "preflight": preflight,
        "run_dir": str(run_dir),
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "wandb": {
            "exit_code": exit_code,
            "stream_path": str(streams[0]),
            "stream_sha256": _sha256(streams[0]),
        },
    }
    destination = run_dir / "SMOKE_RESULT.json"
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if destination.exists():
        _require(
            destination.read_text() == rendered,
            f"refusing to overwrite differing smoke record: {destination}",
        )
    else:
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.temporary"
        )
        _require(not temporary.exists(), f"temporary smoke record exists: {temporary}")
        temporary.write_text(rendered)
        try:
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--experiment",
        "--expected-experiment",
        dest="experiment",
        choices=tuple(APPROVED_EXPERIMENTS),
        required=True,
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-config-sha256")
    parser.add_argument("--expected-split-sha256")
    parser.add_argument("--expected-normalization-sha256")
    parser.add_argument("--expected-content-manifest-sha256")
    parser.add_argument("--expected-dataset-content-aggregate-sha256")
    parser.add_argument("--expected-reconstruction-weight", type=float)
    parser.add_argument("--expected-flow-weight", type=float)
    parser.add_argument("--expected-preflight-sha256")
    parser.add_argument("--gpu-profile", choices=GPU_PROFILES, default="h100-h200")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = verify_smoke(
            args.run_dir,
            experiment=args.experiment,
            expected_head=args.expected_head,
            expected_config_sha256=args.expected_config_sha256,
            expected_split_sha256=args.expected_split_sha256,
            expected_normalization_sha256=args.expected_normalization_sha256,
            expected_content_manifest_sha256=args.expected_content_manifest_sha256,
            expected_dataset_content_aggregate_sha256=(
                args.expected_dataset_content_aggregate_sha256
            ),
            expected_reconstruction_weight=args.expected_reconstruction_weight,
            expected_flow_weight=args.expected_flow_weight,
            expected_preflight_sha256=args.expected_preflight_sha256,
            gpu_profile=args.gpu_profile,
        )
    except Exception as error:
        print(f"[action-flow-smoke] FAIL: {error}", file=sys.stderr)
        return 1
    print(f"[action-flow-smoke] PASS {result['run_dir']}/SMOKE_RESULT.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
