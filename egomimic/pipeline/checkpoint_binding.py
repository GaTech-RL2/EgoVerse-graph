"""Exact model/declaration/data bindings shared by resume and inference loaders."""

from collections.abc import Mapping

from egomimic.pipeline.inference_config import (
    inference_contract_sha256,
    model_pipeline_sha256,
)


def checkpoint_binding(training, context):
    return {
        "schema_version": 1,
        "model_pipeline_sha256": model_pipeline_sha256(training),
        "inference_contract_sha256": inference_contract_sha256(training),
        "data_context_sha256": context.fingerprint(),
    }


def validate_checkpoint_binding(checkpoint, training, context):
    binding = checkpoint.get("inference_binding")
    if not isinstance(binding, Mapping):
        raise ValueError(
            "Checkpoint lacks a verified model/data binding. Use its immutable original "
            "runtime or an explicitly verified migration; do not substitute today's recipe or normalizer."
        )
    expected = checkpoint_binding(training, context)
    if dict(binding) != expected:
        mismatches = [
            key for key, value in expected.items() if binding.get(key) != value
        ]
        raise ValueError(f"Checkpoint model/data binding mismatch: {mismatches}")
    if "data_context" not in checkpoint:
        raise ValueError("Bound checkpoints must contain their immutable data_context")
    from egomimic.pl_utils.data_context import state_fingerprint

    if state_fingerprint(checkpoint["data_context"]) != expected["data_context_sha256"]:
        raise ValueError(
            "Checkpoint data context content differs from its verified binding"
        )
    return expected


def validate_artifact_binding(artifact, checkpoint, training, context):
    expected = validate_checkpoint_binding(checkpoint, training, context)
    if artifact.get("checkpoint_binding") != expected:
        raise ValueError(
            "Inference artifact is not bound to this checkpoint's model and complete data context"
        )
