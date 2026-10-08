"""Fail-closed architecture/exactness binding for a maintained callback proposal.

This module never executes model operations, changes source, or grants launch
authority. A structural PASS is not numerical/GPU/full-model proof.
"""
import hashlib
import json
from pathlib import Path


def digest(contract):
    # Proof location/hash is evidence for a scientific identity, not part of it:
    # including its own payload SHA would create an impossible circular hash.
    identity = {key: value for key, value in contract.items() if not key.startswith("raw_proof_")}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_config(config, contract):
    if contract["expected_speed_stage_count"] not in (0, 1):
        raise ValueError("Expected speed-stage count must explicitly be zero or one")
    targets = [stage["_target_"] for stage in config["model"]["pipeline"]["stages"]]
    if targets != contract["stage_targets"]:
        raise ValueError("Complete ordered architecture identity mismatch")
    names = [target.rsplit(".", 1)[-1] for target in targets]
    if names[:2] != ["KeyedFeatureProjection", "FusedObsEncoder"]:
        raise ValueError("Native replay helper requires verified two-stage prefix")
    if names.count("SharedSpeedCondition") != contract["expected_speed_stage_count"]:
        raise ValueError("Speed conditioning changed architecture")
    velocities = [index for index, name in enumerate(names) if name == "ConditionalVelocityStage"]
    if len(velocities) != 1:
        raise ValueError("Unique native velocity stage required")
    for name in ("RoutedContentEncoderStage", "RoutedContentDecoderStage"):
        rows = [stage for stage in config["model"]["pipeline"]["stages"] if stage["_target_"].endswith("."+name)]
        if len(rows) != 1 or {str(k): v for k, v in rows[0]["route_aliases"].items()} != contract["route_aliases"]:
            raise ValueError("Source routes changed")
    projection = config["model"]["pipeline"]["stages"][0]
    if {str(k): v for k, v in projection["selector_aliases"].items()} != contract["route_aliases"]:
        raise ValueError("Projection routes changed")
    if set(config["data"]["train_datasets"]) != set(contract["source_rows"]):
        raise ValueError("Training sources changed")
    for source, rows in contract["source_rows"].items():
        if config["data"]["train_dataloader_params"][source]["batch_size"] != rows:
            raise ValueError("Per-source batch changed")
    if contract["expected_speed_stage_count"] == 0 and contract["numerical_contract"] != "raw-required":
        raise ValueError("Scalar numerical waiver cannot apply to noaug")
    return {"velocity_index": velocities[0],
            "sampler_override_path": f"model.pipeline.stages.{velocities[0]}.inference_method",
            "sampler_steps_override_path": f"model.pipeline.stages.{velocities[0]}.num_inference_steps",
            "architecture_contract_sha256": digest(contract)}


def validate_runtime(stages, contract):
    targets = [type(stage).__module__+"."+type(stage).__name__ for stage in stages]
    if targets != contract["stage_targets"]:
        raise ValueError("Constructed stage identity differs from immutable contract")
    if sum(type(stage).__name__ == "SharedSpeedCondition" for stage in stages) != contract["expected_speed_stage_count"]:
        raise ValueError("Constructed speed-stage count mismatch")
    if {str(k): v for k, v in stages[0].selector_aliases.items()} != contract["route_aliases"]:
        raise ValueError("Constructed projection routes changed")
    for stage in stages:
        if type(stage).__name__ in ("RoutedContentEncoderStage", "RoutedContentDecoderStage"):
            if {str(k): v for k, v in stage.route_aliases.items()} != contract["route_aliases"]:
                raise ValueError("Constructed source route mismatch")
    return digest(contract)


def validate_source_batches(batch, contract):
    if set(batch) != set(contract["source_rows"]):
        raise ValueError("Every optimizer update must retain both source batches")
    for source, rows in contract["source_rows"].items():
        value = batch[source]["embodiment"]
        if not hasattr(value, "shape") or len(value.shape) not in (1, 2) or value.shape[0] != rows:
            raise ValueError("Actual per-source row count changed")


def require_exactness_proof(task, contract):
    """Noaug production installation stays gated until source-specific raw proof."""
    if contract["numerical_contract"] == "known-numerical-delta-authorized":
        if contract["expected_speed_stage_count"] != 1 or contract.get("numerical_delta_authorization") != "fresh-scalar-only":
            raise ValueError("Numerical delta requires explicit fresh-scalar scope")
        return False
    if contract["numerical_contract"] != "raw-required":
        raise ValueError("Unknown numerical contract")
    relative = Path(contract["raw_proof_relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Task-local raw proof path required")
    data = (Path(task)/relative).read_bytes()
    if hashlib.sha256(data).hexdigest() != contract["raw_proof_sha256"]:
        raise ValueError("Raw proof hash mismatch")
    proof = json.loads(data)
    if proof["status"] != "EXACT_1000_STEP_LOSS_PASS" or proof["exact_raw_checks"] != 13000:
        raise ValueError("Complete scheduled raw-loss proof required")
    if proof["source_commit"] != contract["source_commit"] or proof["architecture_contract_sha256"] != digest(contract):
        raise ValueError("Raw proof scientific identity mismatch")
    if proof.get("gpu_executed") is not True:
        raise ValueError("Synthetic/CPU proof cannot clear actual BF16 grouping gate")
    return True
