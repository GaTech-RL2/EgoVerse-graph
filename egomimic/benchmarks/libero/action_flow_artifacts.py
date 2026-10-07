"""Native LIBERO metric evidence only; never a training or evaluation entry."""
import hashlib
import json
import math
import os
from pathlib import Path

SEED_SHA = "88657b829905d4374823db145ded19b99cec4735f76694734473bcee068bb5b6"
REQUIRED = ("reconst_mse", "reconst_mae", "translation_mse", "rotation_command_mse",
    "gripper_mse", "gripper_sign_accuracy", "command_endpoint_error_m",
    "normalized_reconst_mse", "energy_score32_native_equal_components",
    "energy_accuracy32_native_equal_components", "energy_diversity32_native_equal_components",
    "diagnostic_clean_latent_rms", "diagnostic_clean_decoded_action_normalized_rms")

def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def validate_identity(identity):
    expected = {"suite":"libero10", "source":"libero_panda", "action_dim":7,
        "action_horizon":16, "seed":42, "sample_count":32,
        "energy_seed_bank_sha256":SEED_SHA, "inference_method":"euler",
        "inference_steps":50, "normalization_scope":"training_episodes_only",
        "effective_batch_size":32, "homogeneous":"not_applicable_single_source"}
    for key,value in expected.items():
        if identity.get(key) != value: raise ValueError(("native identity mismatch",key))
    for key in ("source_commit","resolved_config_sha256","split_sha256","normalizer_state_sha256","dataset_logical_sha256"):
        value=identity.get(key,"")
        size=40 if key=="source_commit" else 64
        if len(value)!=size or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(("invalid native identity hash",key))
    return True

def validate_payload(payload):
    if payload.get("schema") != "libero-native-action-flow-metrics/v1":
        raise ValueError("wrong native artifact schema")
    validate_identity(payload["identity"])
    if canonical_sha(payload["identity"]) != payload["identity_sha256"]:
        raise ValueError("native identity digest mismatch")
    if payload["global_step"] < 1 or payload["source"] != "libero_panda":
        raise ValueError("actual optimizer/source missing")
    rows=payload["batches"]
    if not rows or rows[0]["batch_index"]!=0: raise ValueError("first-batch ES32 evidence missing")
    for index,row in enumerate(rows):
        if row["batch_index"]!=index or row["batch_size"]<1: raise ValueError("batch coverage missing")
        needed=REQUIRED if index==0 else REQUIRED[:8]
        for key in needed:
            value=row["metrics"].get(key)
            if not isinstance(value,(float,int)) or not math.isfinite(value):
                raise ValueError(("missing/nonfinite native metric",index,key))
        if len(row["normalized_target_sha256"])!=64: raise ValueError("target hash missing")
    return True

def write_artifact(root, payload):
    validate_payload(payload)
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    path=root/f'{payload["group"]}-step-{payload["global_step"]:06d}.json'
    if path.exists():raise ValueError("native artifact overwrite prohibited")
    temp=path.with_suffix(".json.tmp")
    with temp.open("x") as handle:
        json.dump(payload,handle,sort_keys=True,indent=2,allow_nan=False)
        handle.flush();os.fsync(handle.fileno())
    os.rename(temp,path)
    return path
