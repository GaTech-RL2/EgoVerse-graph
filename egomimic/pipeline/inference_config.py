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
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from egomimic.pipeline.inference_controls import validate_control

ARTIFACT_KIND = "egomimic.graph-inference"
SCHEMA_VERSION = 2


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


def inference_contract_sha256(training: DictConfig | Mapping[str, Any]) -> str:
    """Bind only the resolved, explicitly declared model semantics."""
    config = _as_config(training)
    return _canonical_sha256(
        {
            "pipeline": _plain_node(config, "model.pipeline"),
            "inference": _plain_node(config, "model.inference"),
        }
    )


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _output_contract(node, name):
    if not isinstance(node, Mapping):
        raise ValueError(f"{name} must be a mapping")
    for key in ("key", "representation"):
        if not isinstance(node.get(key), str) or not node[key]:
            raise ValueError(f"{name}.{key} must be a nonempty string")
    shape = node.get("shape")
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError(f"{name}.shape must be [horizon, width]")
    for dimension in shape:
        _positive_int(dimension, f"{name}.shape")
    timing = node.get("timing")
    if not isinstance(timing, Mapping) or not isinstance(timing.get("kind"), str):
        raise ValueError(f"{name} must declare timing semantics")
    if timing.get("temporally_resolved") is not True:
        raise ValueError(
            f"{name} timing cannot reconstruct intervals; diagnostic-only representations are not deployable"
        )
    if "dt" in timing:
        dt = timing["dt"]
        if type(dt) not in (int, float) or not math.isfinite(dt) or dt <= 0:
            raise ValueError(f"{name}.timing.dt must be finite and positive")


def validate_declared_graph(graph, pipeline):
    """Validate structure and declared bindings without interpreting families."""
    if not isinstance(graph, Mapping):
        raise ValueError("Model must declare model.inference in its YAML")
    inputs = graph.get("input")
    if not isinstance(inputs, Mapping):
        raise ValueError("model.inference.input must be a mapping")
    _positive_int(inputs.get("history_length"), "input.history_length")
    keys = inputs.get("keys")
    if (
        not isinstance(keys, list)
        or not keys
        or any(not isinstance(k, str) or not k for k in keys)
    ):
        raise ValueError("input.keys must declare the required observation keys")
    history_keys = inputs.get("history_keys", keys)
    if not isinstance(history_keys, list) or not set(history_keys) <= set(keys):
        raise ValueError("input.history_keys must be a subset of input.keys")
    constants = inputs.get("constants", {})
    if not isinstance(constants, Mapping) or not set(constants) <= set(keys):
        raise ValueError("input.constants must declare values for required input keys")
    _output_contract(graph.get("native_output"), "native_output")
    _output_contract(graph.get("output"), "output")
    compatibility = graph.get("compatibility")
    if not isinstance(compatibility, Mapping) or not isinstance(
        compatibility.get("normalizer_schema"), Mapping
    ):
        raise ValueError("model.inference.compatibility must declare normalizer_schema")
    if "tokenizer" not in compatibility:
        raise ValueError(
            "model.inference.compatibility must declare tokenizer (null for native actions)"
        )
    stages = pipeline.get("stages", [])
    stage_ids = pipeline.get("stage_ids", {})
    if not isinstance(stage_ids, Mapping) or any(
        not isinstance(k, str)
        or not k
        or type(v) is not int
        or not 0 <= v < len(stages)
        for k, v in stage_ids.items()
    ):
        raise ValueError(
            "pipeline.stage_ids must map stable names to valid stage positions"
        )
    profiles = graph.get("profiles")
    if not isinstance(profiles, Mapping) or len(profiles) != 1:
        raise ValueError(
            "model.inference must declare exactly one resolved deployment profile"
        )
    for name, profile in profiles.items():
        if not isinstance(profile, Mapping) or profile.get("stage_id") not in stage_ids:
            raise ValueError(f"Inference profile {name!r} requires a declared stage_id")
        if profile.get("native_shape") != graph["native_output"]["shape"]:
            raise ValueError(
                f"Inference profile {name!r} native_shape differs from native_output"
            )
        adapter = profile.get("adapter", {})
        if not isinstance(adapter, Mapping) or "decoder" not in adapter:
            raise ValueError(
                f"Inference profile {name!r} must declare decoder (null for identity)"
            )
        if (
            adapter["decoder"] is None
            and not adapter.get("_target_")
            and graph["native_output"]["shape"] != graph["output"]["shape"]
        ):
            raise ValueError(
                "Nonidentity native/canonical shapes require a configured decoder"
            )
        controls = profile.get("overrides", {})
        if not isinstance(controls, Mapping):
            raise ValueError(f"Inference profile {name!r} overrides must be a mapping")
        for control, spec in controls.items():
            validate_control(control, spec, stage_ids=stage_ids)
    return dict(graph)


def validate_input_constants(graph, provided):
    """Check the selected input profile before weights or hardware are opened."""
    for key, expected in graph.get("input", {}).get("constants", {}).items():
        if key not in provided or provided[key] != expected:
            raise ValueError(
                f"Inference input profile requires {key}={expected!r}; "
                f"adapter declares {provided.get(key)!r}"
            )


def build_inference_config(
    training: DictConfig | Mapping[str, Any], *, data_context=None
) -> dict[str, Any]:
    """Validate, hash and serialize the declaration; never guess semantics."""
    config = _as_config(training)
    artifact = {
        "kind": ARTIFACT_KIND,
        "schema_version": SCHEMA_VERSION,
        "model_pipeline_sha256": model_pipeline_sha256(config),
        "inference_contract_sha256": inference_contract_sha256(config),
    }
    if data_context is not None:
        from egomimic.pipeline.checkpoint_binding import checkpoint_binding

        artifact["checkpoint_binding"] = checkpoint_binding(config, data_context)
    try:
        declaration = _plain_node(config, "model.inference")
        if (
            isinstance(declaration, Mapping)
            and declaration.get("status") == "unsupported"
        ):
            reason = declaration.get("reason")
            if not isinstance(reason, str) or not reason:
                raise ValueError(
                    "Nondeployable models must declare an actionable reason"
                )
            raise ValueError(reason)
        graph = validate_declared_graph(
            declaration, _plain_node(config, "model.pipeline")
        )
    except (KeyError, TypeError, ValueError) as error:
        return {**artifact, "status": "unsupported", "reason": str(error)}
    return {
        **artifact,
        "status": "ready",
        "inference_graph": graph,
        "inference_graph_sha256": _canonical_sha256(graph),
        "normalizer_schema_sha256": _canonical_sha256(
            graph["compatibility"]["normalizer_schema"]
        ),
        "tokenizer_sha256": _canonical_sha256(graph["compatibility"]["tokenizer"]),
        "decoder_sha256": _canonical_sha256(
            [profile["adapter"]["decoder"] for profile in graph["profiles"].values()]
        ),
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
    expected = build_inference_config(training)
    if expected["status"] != "ready":
        raise ValueError(
            f"Selected model contract is not deployable: {expected['reason']}"
        )
    for field in (
        "inference_graph_sha256",
        "normalizer_schema_sha256",
        "tokenizer_sha256",
        "decoder_sha256",
    ):
        if artifact.get(field) != expected[field]:
            raise ValueError(
                f"Inference artifact {field} differs from the model declaration"
            )
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
    training: DictConfig | Mapping[str, Any], path: str | Path, *, data_context=None
) -> tuple[Path, dict[str, Any]]:
    """Atomically write one immutable artifact, tolerating identical DDP writers."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    artifact = build_inference_config(training, data_context=data_context)
    payload = OmegaConf.to_yaml(OmegaConf.create(artifact), resolve=True)
    _write_immutable_text(destination, payload)
    return destination, artifact


def _write_immutable_text(destination, payload):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
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


def export_configured_inference_artifact(
    training: DictConfig | Mapping[str, Any],
    *,
    data_context=None,
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
    result = write_inference_config(config, path, data_context=data_context)
    if data_context is not None:
        from egomimic.pl_utils.data_context import serializable_state

        # Use the same immutable writer semantics as the inference sidecar.
        context_path = Path(path).with_name("data-context.json")
        payload = (
            json.dumps(
                {"data_context": serializable_state(data_context.snapshot())},
                sort_keys=True,
                allow_nan=False,
                indent=2,
            )
            + "\n"
        )
        _write_immutable_text(context_path, payload)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a declaration or verify a bound checkpoint sidecar; opens no hardware."
    )
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--checkpoint", help="Existing bound graph checkpoint to verify"
    )
    parser.add_argument("--data-context", help="Complete training data-context.json")
    args = parser.parse_args()
    training = OmegaConf.load(args.training_config)
    if bool(args.checkpoint) != bool(args.data_context):
        parser.error("--checkpoint and --data-context must be supplied together")
    context = None
    if args.checkpoint:
        import torch
        from hydra.utils import instantiate

        from egomimic.pipeline.checkpoint_binding import validate_checkpoint_binding
        from egomimic.pl_utils.data_context import DataContext

        context = instantiate(training.data_context_loader, path=args.data_context)
        if not isinstance(context, DataContext):
            raise TypeError("data_context_loader must return a complete DataContext")
        checkpoint = torch.load(
            args.checkpoint, map_location="cpu", weights_only=False, mmap=True
        )
        validate_checkpoint_binding(checkpoint, training, context)
    path, artifact = write_inference_config(training, args.output, data_context=context)
    binding = "checkpoint-bound" if context else "unbound declaration; cannot deploy"
    print(f"Wrote {artifact['status']} inference config ({binding}): {path}")


if __name__ == "__main__":
    main()
