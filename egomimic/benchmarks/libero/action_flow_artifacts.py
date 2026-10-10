"""Native LIBERO metric evidence only; never a training or evaluation entry."""

import hashlib
import json
import math
import os
from pathlib import Path

SEED_SHA = "88657b829905d4374823db145ded19b99cec4735f76694734473bcee068bb5b6"
REQUIRED = (
    "reconst_mse",
    "reconst_mae",
    "translation_mse",
    "rotation_command_mse",
    "gripper_mse",
    "gripper_sign_accuracy",
    "command_endpoint_error_m",
    "normalized_reconst_mse",
    "energy_score32_native_equal_components",
    "energy_accuracy32_native_equal_components",
    "energy_diversity32_native_equal_components",
    "diagnostic_clean_latent_rms",
    "diagnostic_clean_decoded_action_normalized_rms",
)


def canonical_sha(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def validate_identity(identity):
    from egomimic.benchmarks.libero.native_launch_profiles import (
        historical_profile_for_suite,
    )

    profile = historical_profile_for_suite(identity.get("suite"))
    suite = profile.suite
    if type(identity.get("seed")) is not int or identity["seed"] not in (
        {42, 43} if profile.suite in {"libero_goal", "libero_object"} else {42}
    ):
        raise ValueError("native metric training seed mismatch")
    expected = {
        "suite": suite,
        "source": "libero_panda",
        "action_dim": 7,
        "action_horizon": 16,
        "sample_count": 32,
        "energy_seed_bank_sha256": SEED_SHA,
        "inference_method": "euler",
        "inference_steps": 50,
        "normalization_scope": "training_episodes_only",
        "effective_batch_size": 32,
        "homogeneous": "not_applicable_single_source",
    }
    for key, value in expected.items():
        if identity.get(key) != value:
            raise ValueError(("native identity mismatch", key))
    for key in (
        "source_commit",
        "resolved_config_sha256",
        "split_sha256",
        "normalizer_state_sha256",
        "dataset_logical_sha256",
    ):
        value = identity.get(key, "")
        size = 40 if key == "source_commit" else 64
        if len(value) != size or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(("invalid native identity hash", key))
    return True


def validate_payload(payload):
    if payload.get("schema") != "libero-native-action-flow-metrics/v2":
        raise ValueError("wrong native artifact schema")
    validate_identity(payload["identity"])
    if canonical_sha(payload["identity"]) != payload["identity_sha256"]:
        raise ValueError("native identity digest mismatch")
    if payload["global_step"] < 1 or payload["source"] != "libero_panda":
        raise ValueError("actual optimizer/source missing")
    rows = payload["batches"]
    if not rows or rows[0]["batch_index"] != 0:
        raise ValueError("first-batch ES32 evidence missing")
    for index, row in enumerate(rows):
        if row["batch_index"] != index or row["batch_size"] < 1:
            raise ValueError("batch coverage missing")
        needed = REQUIRED if index == 0 else REQUIRED[:8]
        for key in needed:
            value = row["metrics"].get(key)
            if not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(("missing/nonfinite native metric", index, key))
        if len(row["normalized_target_sha256"]) != 64:
            raise ValueError("target hash missing")
        validate_tensor_reference(row["tensor_payload"], row["batch_size"], index == 0)
        if (
            index == 0
            and row["tensor_payload"]["shared_analysis"].get("global_step")
            != payload["global_step"]
        ):
            raise ValueError("shared analysis optimizer step mismatch")
    return True


def write_artifact(root, payload):
    validate_payload(payload)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for row in payload["batches"]:
        verify_tensor_payload(root, row["tensor_payload"])
    path = root / f'{payload["group"]}-step-{payload["global_step"]:06d}.json'
    if path.exists():
        raise ValueError("native artifact overwrite prohibited")
    temp = path.with_suffix(".json.tmp")
    with temp.open("x") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    # Publish without overwriting even under concurrent validation callbacks.
    os.link(temp, path)
    temp.unlink()
    return path


def validate_tensor_reference(ref, batch_size, stochastic):
    if ref.get("schema") != "libero-native-tensors/v1":
        raise ValueError("missing native tensor payload")
    if len(ref.get("sha256", "")) != 64 or Path(
        ref.get("filename", "")
    ).name != ref.get("filename"):
        raise ValueError("invalid payload hash/path")
    metadata = ref["tensors"]
    required = (
        "normalized_prediction",
        "native_prediction",
        "normalized_target",
        "native_target",
    )
    for name in required:
        item = metadata.get(name, {})
        if (
            item.get("shape") != [batch_size, 16, 7]
            or item.get("axes") != ["batch", "horizon", "action"]
            or item.get("finite") is not True
        ):
            raise ValueError(("tensor contract", name))
    if stochastic:
        shared = ref.get("shared_analysis", {})
        for key in ("sha256", "identity_sha256"):
            if len(shared.get(key, "")) != 64:
                raise ValueError(
                    "immutable shared same-pass analysis reference missing"
                )
        if not shared.get("path"):
            raise ValueError("immutable shared analysis path missing")
        for name in ("normalized_samples", "native_samples"):
            item = metadata.get(name, {})
            if (
                item.get("shape") != [32, batch_size, 16, 7]
                or item.get("axes") != ["sample", "batch", "horizon", "action"]
                or item.get("finite") is not True
            ):
                raise ValueError(("K32 tensor contract", name))
        for name in ("diagnostic/latent/clean", "diagnostic/decoded/reconstruction"):
            if metadata.get(name, {}).get("finite") is not True:
                raise ValueError(("same-pass diagnostic missing", name))
    for name, item in metadata.items():
        if item.get("finite") is not True or len(item.get("sha256", "")) != 64:
            raise ValueError(("nonfinite/unhashed tensor", name))
    if not ref.get("alignment"):
        raise ValueError("sample alignment absent")
    return True


def write_tensor_payload(root, group, global_step, batch_index, tensors, alignment):
    import io

    import torch

    cpu = {}
    metadata = {}
    for name, value in tensors.items():
        if not torch.is_tensor(value):
            raise TypeError(("tensor required", name))
        value = value.detach().cpu().contiguous().clone()
        if not bool(torch.isfinite(value).all()):
            raise ValueError(("nonfinite tensor", name))
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        axes = (
            ["sample", "batch", "horizon", "action"]
            if name.endswith("samples")
            else ["batch", "horizon", "action"]
            if name
            in {
                "normalized_prediction",
                "native_prediction",
                "normalized_target",
                "native_target",
            }
            else [f"source_dimension{i}" for i in range(value.ndim)]
        )
        cpu[name] = value
        metadata[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "axes": axes,
            "finite": True,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    # Alignment is captured directly from this batch, not inferred from its order.
    captured = {}
    for name, value in alignment.items():
        captured[name] = (
            value.detach().cpu().clone() if torch.is_tensor(value) else value
        )
    stream = io.BytesIO()
    torch.save(
        {"schema": "libero-native-tensors/v1", "tensors": cpu, "alignment": captured},
        stream,
    )
    raw = stream.getvalue()
    sha = hashlib.sha256(raw).hexdigest()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    filename = f"{group}-step-{global_step:06d}-batch-{batch_index:04d}-{sha}.pt"
    path = root / filename
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "schema": "libero-native-tensors/v1",
        "filename": filename,
        "sha256": sha,
        "bytes": len(raw),
        "tensors": metadata,
        "alignment": sorted(captured),
    }


def verify_tensor_payload(root, ref):
    import torch

    path = Path(root) / ref["filename"]
    raw = path.read_bytes()
    if len(raw) != ref["bytes"] or hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise ValueError("immutable payload bytes/hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != "libero-native-tensors/v1" or set(
        payload["tensors"]
    ) != set(ref["tensors"]):
        raise ValueError("payload schema/tensor inventory mismatch")
    if sorted(payload["alignment"]) != ref["alignment"]:
        raise ValueError("payload alignment inventory mismatch")
    for name, tensor in payload["tensors"].items():
        info = ref["tensors"][name]
        if (
            not torch.is_tensor(tensor)
            or list(tensor.shape) != info["shape"]
            or str(tensor.dtype) != info["dtype"]
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError(("payload tensor mismatch/nonfinite", name))
        raw = tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        if hashlib.sha256(raw).hexdigest() != info["sha256"]:
            raise ValueError(("payload tensor hash mismatch", name))
    shared = ref.get("shared_analysis")
    if shared is not None:
        shared_path = Path(shared["path"])
        if hashlib.sha256(shared_path.read_bytes()).hexdigest() != shared["sha256"]:
            raise ValueError("shared same-pass analysis artifact SHA mismatch")
        shared_payload = torch.load(shared_path, map_location="cpu", weights_only=True)
        if (
            shared_payload.get("identity_sha256") != shared["identity_sha256"]
            or shared_payload.get("global_step") != shared["global_step"]
            or shared_payload.get("noise_seed") != shared["noise_seed"]
        ):
            raise ValueError("shared same-pass analysis identity/step/seed mismatch")
        # Prove the shared computed artifact consumed the exact saved raw pass.
        same_pass = {}

        def capture(prefix, values):
            for key, value in values.items():
                name = prefix + str(key)
                if torch.is_tensor(value):
                    same_pass[name] = value
                elif isinstance(value, dict):
                    capture(name + "/", value)

        capture("diagnostic/", shared_payload["sources"]["libero_panda"]["diagnostic"])
        expected = {
            name for name in payload["tensors"] if name.startswith("diagnostic/")
        }
        if set(same_pass) != expected:
            raise ValueError("shared analysis raw diagnostic inventory mismatch")
        for name, tensor in same_pass.items():
            saved = payload["tensors"][name]
            # Maintained shared _cpu_tree deliberately promotes floating
            # diagnostics to float32; raw native payload preserves its dtype.
            expected = saved.float() if saved.is_floating_point() else saved
            if (
                tensor.dtype != expected.dtype
                or tensor.shape != expected.shape
                or not torch.equal(tensor, expected)
            ):
                raise ValueError(("shared analysis consumed different raw pass", name))
    batch_size = ref["tensors"]["normalized_target"]["shape"][0]
    for key in ("episode_hash", "frame_index"):
        if (
            key not in payload["alignment"]
            or len(payload["alignment"][key]) != batch_size
        ):
            raise ValueError(("payload alignment length mismatch", key))
    return True
