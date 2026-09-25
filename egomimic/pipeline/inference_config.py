"""Build and validate model-owned graph inference configuration artifacts.

Training writes one ``inference-config.yaml`` beside its checkpoints.  The
artifact contains only model semantics (history, native output, decoding, and
explicitly exposed inference controls); station calibration and safety remain
in the robot rollout YAML.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

ARTIFACT_KIND = "egomimic.graph-inference"
SCHEMA_VERSION = 1

ACTION_TARGET = "egomimic.pipeline.stages_io.ActionTargetBuilder"
FLOW_DENOISER = "egomimic.pipeline.stages_flow.FlowDenoiserStage"
DIFFUSION_DENOISER = "egomimic.pipeline.stages_diffusion.DiffusionDenoiserStage"
FUSED_OBS_ENCODER = "egomimic.pipeline.stages_sampler.FusedObsEncoder"


def _as_config(config: DictConfig | Mapping[str, Any]) -> DictConfig:
    if OmegaConf.is_config(config):
        return config
    if not isinstance(config, Mapping):
        raise TypeError("Training configuration must be a mapping")
    return OmegaConf.create(config)


def _plain_node(config: DictConfig, path: str) -> Any:
    node = OmegaConf.select(config, path, default=None)
    if node is None:
        return None
    if OmegaConf.is_config(node):
        return OmegaConf.to_container(node, resolve=True)
    return node


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def model_pipeline_sha256(training: DictConfig | Mapping[str, Any]) -> str:
    """Hash the fully resolved model pipeline that determines inference."""
    config = _as_config(training)
    pipeline = _plain_node(config, "model.pipeline")
    if not isinstance(pipeline, Mapping):
        raise ValueError("Training config must define model.pipeline")
    return _canonical_sha256(pipeline)


def _uses_abc_arc(config: DictConfig) -> bool:
    return any(
        "arc_tokenizer" in str(OmegaConf.select(config, path, default=None))
        for path in (
            "abc.action_mode", "run_provenance.action_contract.representation"
        )
    )


def inference_contract_sha256(
    training: DictConfig | Mapping[str, Any],
) -> str:
    """Hash every resolved training field used to derive robot inference."""
    config = _as_config(training)
    source = {
        "pipeline": _plain_node(config, "model.pipeline"),
        "e1": {
            name: OmegaConf.select(config, f"e1.{name}", default=None)
            for name in ("variant", "D", "M", "time_rows")
        },
        "evaluator_dt": OmegaConf.select(config, "evaluator.dt", default=None),
        "defaults": {
            name: OmegaConf.select(config, f"inference_config.{name}", default=None)
            for name in (
                "flow_inference_steps",
                "diffusion_inference_steps",
                "max_flow_inference_steps",
                "replan_every",
                "action_dt",
            )
        },
    }
    # ABC uses the same action tensor width as Cartesian output, so its codec
    # declaration is part of the contract even when the pipeline shape matches.
    # Preserve existing Cartesian/E1 hashes: ancillary ABC settings do not
    # change their rollout codec. Only ARC declarations extend the contract.
    if _uses_abc_arc(config):
        source["abc"] = {
            name: OmegaConf.select(config, f"abc.{name}", default=None)
            for name in (
                "action_mode", "arc_chunking_mode", "arc_distance",
                "arc_rotation_distance", "arc_waypoints", "arc_velocity_mode",
            )
        }
        action_contract = _plain_node(config, "run_provenance.action_contract")
        if action_contract is not None:
            source["action_contract"] = action_contract
    return _canonical_sha256(source)


def _pipeline_stages(config: DictConfig) -> list[dict[str, Any]]:
    stages = _plain_node(config, "model.pipeline.stages")
    if not isinstance(stages, list) or not all(
        isinstance(stage, dict) for stage in stages
    ):
        raise ValueError("model.pipeline.stages must be a list of mappings")
    return stages


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _positive_float(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be a positive number")
    result = float(value)
    if not result > 0:
        raise ValueError(f"{label} must be a positive number")
    return result


def _unsupported(pipeline_hash: str, contract_hash: str, reason: str) -> dict[str, Any]:
    return {
        "kind": ARTIFACT_KIND,
        "schema_version": SCHEMA_VERSION,
        "status": "unsupported",
        "model_pipeline_sha256": pipeline_hash,
        "inference_contract_sha256": contract_hash,
        "reason": reason,
    }


def _history_length(stages: list[dict[str, Any]]) -> int:
    encoders = [stage for stage in stages if stage.get("_target_") == FUSED_OBS_ENCODER]
    if not encoders:
        return 1
    if len(encoders) != 1:
        raise ValueError("Inference export requires at most one FusedObsEncoder stage")
    return _positive_int(encoders[0].get("n_obs_steps", 1), "n_obs_steps")


def _inference_step_spec(
    config: DictConfig,
    stage: Mapping[str, Any],
    stage_target: str,
) -> tuple[str, str, int, int, str]:
    if stage_target == FLOW_DENOISER:
        configured = OmegaConf.select(
            config, "inference_config.flow_inference_steps", default=10
        )
        default = stage.get("num_inference_steps") if configured is None else configured
        label = "Euler integration steps"
        description = (
            "Number of Flow solver steps per prediction; higher values increase "
            "inference latency."
        )
        attribute_path = "num_inference_steps"
        maximum = OmegaConf.select(
            config, "inference_config.max_flow_inference_steps", default=100
        )
    else:
        policy = stage.get("policy")
        if not isinstance(policy, Mapping):
            raise ValueError("DiffusionDenoiserStage must define policy")
        configured = OmegaConf.select(
            config, "inference_config.diffusion_inference_steps", default=None
        )
        default = (
            policy.get("num_inference_steps") if configured is None else configured
        )
        label = "Diffusion denoising steps"
        description = (
            "Number of denoising steps per prediction; higher values increase "
            "inference latency."
        )
        attribute_path = "policy.num_inference_steps"
        scheduler = policy.get("noise_scheduler", {})
        maximum = (
            scheduler.get("num_train_timesteps", 100)
            if isinstance(scheduler, Mapping)
            else 100
        )
    default = _positive_int(default, "inference step default")
    maximum = _positive_int(maximum, "inference step maximum")
    if default > maximum:
        raise ValueError(
            f"Inference step default {default} exceeds configured maximum {maximum}"
        )
    return label, description, default, maximum, attribute_path


def _decoder_contract(
    config: DictConfig,
    *,
    variant: str,
    native_horizon: int,
    native_dim: int,
) -> tuple[int, dict[str, Any] | None]:
    if variant == "time":
        if native_dim != 14:
            raise ValueError(
                f"time Cartesian inference requires native width 14, got {native_dim}"
            )
        return native_horizon, None
    if variant == "arcmean":
        raise ValueError(
            "arcmean stores only token-wide mean timing and is not a supported "
            "closed-loop rollout contract; use arcvel, arcdur, or arclogdur"
        )
    layouts = {
        "arcvel": "e1_profile",
        "arcdur": "e1_dur",
        "arclogdur": "e1_logdur",
    }
    if variant not in layouts:
        raise ValueError(f"Unknown E1 action variant {variant!r}")
    waypoints = _positive_int(OmegaConf.select(config, "e1.M", default=None), "e1.M")
    if (native_horizon, native_dim) != (waypoints, 16):
        raise ValueError(
            f"{variant} requires native shape [{waypoints}, 16], got "
            f"[{native_horizon}, {native_dim}]"
        )
    output_horizon = _positive_int(
        OmegaConf.select(config, "e1.time_rows", default=None), "e1.time_rows"
    )
    distance = _positive_float(OmegaConf.select(config, "e1.D", default=None), "e1.D")
    dt = OmegaConf.select(config, "inference_config.action_dt", default=None)
    if dt is None:
        dt = OmegaConf.select(config, "evaluator.dt", default=None)
    dt = _positive_float(dt, "ARC action dt")
    return output_horizon, {
        "_target_": "egomimic.robot.arc_decoder.BimanualArcDecoder",
        "token_layout": layouts[variant],
        "min_distance_unit": distance,
        "resampled_vector_length": waypoints,
        "dt": dt,
        "action_horizon": output_horizon,
    }


def build_inference_config(
    training: DictConfig | Mapping[str, Any],
) -> dict[str, Any]:
    """Derive a fail-closed inference artifact from one resolved training config."""
    config = _as_config(training)
    pipeline_hash = model_pipeline_sha256(config)
    contract_hash = inference_contract_sha256(config)
    try:
        if _uses_abc_arc(config):
            raise ValueError(
                "ABC ARC requires a mode-aware hybrid rollout adapter; automatic "
                "robot inference export is unsupported. Token rows are not "
                "Cartesian control frames. Training and offline evaluation "
                "continue to use the ARC tokenizer/detokenizer."
            )
        stages = _pipeline_stages(config)
        action_targets = [
            stage for stage in stages if stage.get("_target_") == ACTION_TARGET
        ]
        if len(action_targets) != 1:
            raise ValueError(
                "rollout export requires exactly one ActionTargetBuilder stage"
            )
        action_key = action_targets[0].get("action_key")
        if action_key != "actions_cartesian":
            raise ValueError(
                "robot rollout export supports only actions_cartesian; "
                f"model uses {action_key!r}"
            )
        denoisers = [
            stage
            for stage in stages
            if stage.get("_target_") in {FLOW_DENOISER, DIFFUSION_DENOISER}
        ]
        if len(denoisers) != 1:
            raise ValueError(
                "rollout export requires exactly one FlowDenoiserStage or "
                "DiffusionDenoiserStage"
            )
        stage = denoisers[0]
        stage_target = str(stage["_target_"])
        family = "flow" if stage_target == FLOW_DENOISER else "diffusion"
        native_horizon = _positive_int(
            stage.get("action_horizon"), "denoiser action_horizon"
        )
        native_dim = _positive_int(stage.get("action_dim"), "denoiser action_dim")
        declared_variant = OmegaConf.select(config, "e1.variant", default=None)
        variant = "time" if declared_variant is None else str(declared_variant)
        output_horizon, decoder = _decoder_contract(
            config,
            variant=variant,
            native_horizon=native_horizon,
            native_dim=native_dim,
        )
        history_length = _history_length(stages)
        label, description, steps, max_steps, attribute_path = _inference_step_spec(
            config, stage, stage_target
        )
        replan_default = _positive_int(
            OmegaConf.select(config, "inference_config.replan_every", default=30),
            "inference_config.replan_every",
        )
        replan_default = min(replan_default, output_horizon)
        match: dict[str, Any] = {"stage_target": stage_target}
        if declared_variant is not None:
            match["variant"] = variant
        profile = {
            "match": match,
            "native_shape": [native_horizon, native_dim],
            "overrides": {
                "inference_steps": {
                    "label": label,
                    "description": description,
                    "type": "integer",
                    "min": 1,
                    "max": max_steps,
                    "step": 1,
                    "default": steps,
                    "target": {
                        "kind": "stage_attribute",
                        "attribute_path": attribute_path,
                    },
                },
                "replan_every": {
                    "label": "Repredict every",
                    "description": (
                        "Execute this many actions before requesting a fresh prediction."
                    ),
                    "type": "integer",
                    "min": 1,
                    "max": output_horizon,
                    "step": 1,
                    "default": replan_default,
                    "target": {
                        "kind": "policy_attribute",
                        "attribute_path": "replan_every",
                    },
                },
            },
            "adapter": {"decoder": decoder},
        }
        graph = {
            "input": {"history_length": history_length},
            "output": {
                "representation": "cartesian",
                "shape": [output_horizon, 14],
            },
            "profiles": {f"{family}_{variant}": profile},
        }
    except (KeyError, TypeError, ValueError) as error:
        return _unsupported(pipeline_hash, contract_hash, str(error))
    return {
        "kind": ARTIFACT_KIND,
        "schema_version": SCHEMA_VERSION,
        "status": "ready",
        "model_pipeline_sha256": pipeline_hash,
        "inference_contract_sha256": contract_hash,
        "inference_graph_sha256": _canonical_sha256(graph),
        "inference_graph": graph,
    }


def validate_inference_config(
    artifact: Mapping[str, Any],
    training: DictConfig | Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and return a checkpoint-bound inference graph."""
    if not isinstance(artifact, Mapping):
        raise TypeError("Inference config artifact must be a mapping")
    if artifact.get("kind") != ARTIFACT_KIND:
        raise ValueError("Inference config artifact kind is unsupported")
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Inference config artifact schema version is unsupported")
    if artifact.get("status") != "ready":
        reason = artifact.get("reason", "model has no rollout contract")
        raise ValueError(f"Selected model is not rollout-ready: {reason}")
    expected_pipeline_hash = model_pipeline_sha256(training)
    if artifact.get("model_pipeline_sha256") != expected_pipeline_hash:
        raise ValueError(
            "Inference config does not match the selected resolved model pipeline"
        )
    expected_contract_hash = inference_contract_sha256(training)
    if artifact.get("inference_contract_sha256") != expected_contract_hash:
        raise ValueError(
            "Inference config does not match the selected model codec and "
            "inference defaults"
        )
    graph = artifact.get("inference_graph")
    if not isinstance(graph, Mapping):
        raise ValueError("Inference config must contain an inference_graph mapping")
    if artifact.get("inference_graph_sha256") != _canonical_sha256(graph):
        raise ValueError("Inference graph content hash does not match the artifact")
    return dict(graph)


def load_inference_config(
    path: str | Path,
    training: DictConfig | Mapping[str, Any],
) -> dict[str, Any]:
    try:
        artifact = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except OSError as error:
        raise ValueError(f"Could not read inference config artifact: {path}") from error
    return validate_inference_config(artifact, training)


def checkpoint_run_prefix(checkpoint: str | Path) -> str:
    """Return the immutable run prefix used by relayed checkpoint sidecars."""
    stem = Path(checkpoint).stem
    positions = [
        position
        for marker in ("__epoch-", "__step-")
        if (position := stem.find(marker)) >= 0
    ]
    return stem[: min(positions)] if positions else stem


def find_inference_config(checkpoint: str | Path) -> Path | None:
    """Find the model-owned artifact stored beside one immutable checkpoint."""
    checkpoint = Path(checkpoint)
    run_prefix = checkpoint_run_prefix(checkpoint)
    for name in (
        f"{run_prefix}.inference-config.yaml",
        "inference-config.yaml",
    ):
        candidate = checkpoint.parent / name
        if candidate.is_file():
            return candidate.resolve(strict=True)
    return None


def write_inference_config(
    training: DictConfig | Mapping[str, Any], path: str | Path
) -> tuple[Path, dict[str, Any]]:
    """Atomically write one immutable artifact, tolerating identical DDP writers."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    artifact = build_inference_config(training)
    payload = OmegaConf.to_yaml(OmegaConf.create(artifact), resolve=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{os.urandom(6).hex()}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            try:
                existing = destination.read_text(encoding="utf-8")
            except OSError as error:
                raise RuntimeError(
                    f"Could not verify existing inference config: {destination}"
                ) from error
            if existing != payload:
                raise RuntimeError(
                    "Refusing to overwrite a different inference config artifact at "
                    f"{destination}"
                )
    finally:
        temporary.unlink(missing_ok=True)
    return destination, artifact


def export_configured_inference_artifact(
    training: DictConfig | Mapping[str, Any],
) -> tuple[Path, dict[str, Any]] | None:
    """Apply the global trainHydra inference-config export settings."""
    config = _as_config(training)
    settings = _plain_node(config, "inference_config")
    if settings is None:
        return None
    if not isinstance(settings, Mapping):
        raise TypeError("inference_config must be a mapping")
    enabled = settings.get("enabled", True)
    if type(enabled) is not bool:
        raise TypeError("inference_config.enabled must be a boolean")
    if not enabled:
        return None
    path = settings.get("output_path")
    if not isinstance(path, str) or not path:
        raise ValueError("inference_config.output_path must be a nonempty path")
    return write_inference_config(config, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a model-owned inference-config.yaml; opens no hardware."
    )
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    training = OmegaConf.load(args.training_config)
    path, artifact = write_inference_config(training, args.output)
    print(f"Wrote {artifact['status']} inference config: {path}")


if __name__ == "__main__":
    main()
