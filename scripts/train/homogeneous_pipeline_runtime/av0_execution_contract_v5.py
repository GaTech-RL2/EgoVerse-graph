"""Scalar adapter-aware strict contract; preserves generic runtime/rawproof checks."""
from copy import deepcopy
from typed_execution_contract_v1 import digest,validate_runtime,validate_source_batches,require_exactness_proof
from typed_execution_contract_v1 import validate_config as native_validate_config

BUILD="egomimic.pipeline.stages_speed.build_speed_conditioned_pipeline"
SPEED="egomimic.pipeline.stages_speed.SharedSpeedCondition"
def expanded_config(config):
    config=deepcopy(config)
    pipeline=config["model"]["pipeline"]
    if pipeline["_target_"]!=BUILD or pipeline["encoding"]!="scalar":
        raise ValueError("Exact scalar adapter required")
    stages=pipeline["stages"]
    if len(stages)!=9 or any(s["_target_"]==SPEED for s in stages):
        raise ValueError("Adapter base must have exactly9 native stages")
    encoders=[i for i,s in enumerate(stages) if s["_target_"].endswith(".RoutedContentEncoderStage")]
    if len(encoders)!=1:raise ValueError("Unique insertion boundary required")
    if stages[-1]["action_velocity_weight"]!=0:
        raise ValueError("AV0 objective required")
    stages.insert(encoders[0],{"_target_":SPEED})
    return config

def validate_config(config,contract):
    if contract["numerical_contract"]!="raw-required" or contract["objective"]["action_velocity_weight"]!=0:
        raise ValueError("AV0 cannot inherit scalar numerical waiver")
    return native_validate_config(expanded_config(config),contract)
