"""Read-only admission of the one CUDA-visible H100 assigned to this job.

NVIDIA documents GPU Recovery Action as the replacement for deprecated reset
flags. An N/A reset flag is therefore acceptable with Recovery Action=None.
N/A ECC values are not zeros: require a usable legacy total or both SRAM/DRAM
coverage in each of the volatile and aggregate sections. Row-remap pending and
failure flags must both be explicitly clear on this H100 workload.

Accept both row_remapper and the remapped_rows XML emitted by the allocated
580.105.08 driver. Every reported pending/failure alias must be explicitly
clear; one clear schema cannot hide an unsafe second schema. Row counts are
historical remapping totals, recorded separately from these status flags.

Reference: https://docs.nvidia.com/deploy/nvidia-smi/index.html
Unit fixtures are synthetic schema examples. The allocation's actual XML is
saved alongside the JSON receipt; these checks never reset or modify hardware.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import uuid
import xml.etree.ElementTree as ET


class HealthError(ValueError):
    pass


def _uuid(value):
    if isinstance(value, bytes):
        value = str(uuid.UUID(bytes=value)) if len(value) == 16 else value.decode("ascii")
    value = str(value)
    if value.startswith("GPU-"):
        value = value[4:]
    try:
        return "GPU-" + str(uuid.UUID(value))
    except (ValueError, AttributeError) as error:
        raise HealthError("CUDA did not expose a valid full GPU UUID") from error


def _text(node, path):
    value = node.findtext(path)
    return value.strip() if value is not None else None


def _clear(value, label):
    if value is None or value.lower() not in {"no", "false", "0"}:
        raise HealthError(f"{label} is not explicitly clear: {value!r}")


def _ecc(scope, label):
    if scope is None:
        raise HealthError(f"Missing {label} ECC counters")
    numeric, unavailable = {}, []

    def visit(node, prefix=""):
        path = f"{prefix}/{node.tag}" if prefix else node.tag
        if list(node):
            for child in node:
                visit(child, path)
            return
        if not any(word in path.lower() for word in ("uncorrectable", "uncorrected", "double_bit")):
            return
        value = (node.text or "").strip()
        if value.upper() in {"N/A", "NOT SUPPORTED"}:
            unavailable.append(path)
        elif not re.fullmatch(r"\d+", value):
            raise HealthError(f"Ambiguous {label} ECC counter {path}: {value!r}")
        else:
            numeric[path] = int(value)
            if numeric[path] != 0:
                raise HealthError(f"Nonzero {label} uncorrectable ECC counter {path}: {value}")

    visit(scope)
    names = {path.rsplit("/", 1)[-1] for path in numeric}
    legacy_total = any(path.endswith("/double_bit/total") for path in numeric)
    dram = bool(names & {"dram_uncorrectable", "dram_uncorrected"})
    sram = bool(names & {"sram_uncorrectable", "sram_uncorrected"}) or {
        "sram_uncorrectable_parity", "sram_uncorrectable_secded"
    }.issubset(names)
    if not (legacy_total or (dram and sram)):
        raise HealthError(f"Incomplete {label} uncorrectable ECC coverage")
    return {"zero_counters": numeric, "unavailable_redundant_counters": unavailable}


def _row_remapper(gpu):
    """Normalize both XML schemas without interpreting missing flags as clear."""
    sections = [
        node for node in gpu if node.tag in {"row_remapper", "remapped_rows"}
    ]
    if not sections:
        raise HealthError("Missing H100 row-remapper status")
    result = {"pending": {}, "failure": {}, "counts": {}, "schemas": []}
    for index, row in enumerate(sections):
        # Preserve every value, including duplicate aliases, so contradictory
        # status fields cannot be hidden by Element.find's first-match behavior.
        prefix = f"{row.tag}[{index}]"
        result["schemas"].append(row.tag)
        for label, aliases in {
            "pending": ("pending", "row_remapping_pending", "remapped_row_pending"),
            "failure": (
                "remapping_failure_occurred", "row_remapping_failure",
                "remapped_row_failure",
            ),
        }.items():
            fields = [child for child in row if child.tag in aliases]
            if not fields:
                raise HealthError(f"Missing row-remapper {label} flag in {row.tag}")
            for field_index, field in enumerate(fields):
                path = f"{prefix}/{field.tag}[{field_index}]"
                value = (field.text or "").strip()
                _clear(value, path)
                result[label][path] = value
        for field_index, field in enumerate(row):
            if field.tag not in {"remapped_row_corr", "remapped_row_unc"}:
                continue
            path = f"{prefix}/{field.tag}[{field_index}]"
            value = (field.text or "").strip()
            if not re.fullmatch(r"\d+", value):
                raise HealthError(f"Ambiguous row-remapper count {path}: {value!r}")
            # A completed remapping is not a pending or failed remapping.
            # The existing admission policy requires clear status and zero
            # uncorrectable ECC, not zero historical correctable row remaps.
            result["counts"][path] = int(value)
    return result


def inspect_xml(xml, expected_uuid):
    """Validate selected-device XML without consulting or modifying hardware."""
    expected_uuid = _uuid(expected_uuid)
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as error:
        raise HealthError("nvidia-smi did not return valid XML") from error
    devices = root.findall("gpu")
    if len(devices) != 1:
        raise HealthError("Selected-device nvidia-smi query must return exactly one GPU")
    gpu = devices[0]
    if _uuid(_text(gpu, "uuid")) != expected_uuid:
        raise HealthError("nvidia-smi UUID differs from the CUDA-visible device")
    name = _text(gpu, "product_name") or ""
    if not re.search(r"\bH100\b", name):
        raise HealthError(f"Expected an H100, got {name!r}")
    checks = {"uuid": expected_uuid, "product_name": name}
    checks["uncorrectable_ecc"] = {
        scope: _ecc(gpu.find(f"ecc_errors/{scope}"), scope)
        for scope in ("volatile", "aggregate")
    }
    recovery = _text(gpu, "gpu_recovery_action")
    reset = {
        field: _text(gpu, f"gpu_reset_status/{field}")
        for field in ("reset_required", "drain_and_reset_recommended")
    }
    for field, value in reset.items():
        if value is not None and value.upper() not in {"N/A", "NOT SUPPORTED"}:
            _clear(value, field)
    if recovery is not None and recovery.upper() not in {"N/A", "NOT SUPPORTED"}:
        if recovery.lower() != "none":
            raise HealthError(f"GPU recovery action required: {recovery}")
        recovery_basis = "gpu_recovery_action"
    else:
        for field, value in reset.items():
            _clear(value, field)
        recovery_basis = "legacy_reset_flags"
    checks["recovery"] = {"action": recovery, "reset_flags": reset, "basis": recovery_basis}
    checks["row_remapper"] = _row_remapper(gpu)
    # Newer drivers may expose these repair flags; absent/unsupported optional
    # fields are covered by the required recovery status, not treated as zeros.
    repair = {}
    for leaf in gpu.iter():
        if leaf.tag in {"sram_threshold_exceeded", "channel_repair_pending", "tpc_repair_pending", "unrepairable_memory"}:
            value = (leaf.text or "").strip()
            if value.upper() not in {"N/A", "NOT SUPPORTED"}:
                _clear(value, leaf.tag)
            repair[leaf.tag] = value
    checks["optional_repair_flags"] = repair
    return checks


def admit_gpu(receipt_path):
    """Save admission evidence and raise on an unhealthy/ambiguous allocation."""
    import torch

    receipt_path = Path(receipt_path)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path = receipt_path.with_suffix(".xml")
    receipt = {
        "schema": "dino-proprio-gpu-admission-v1",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FAIL",
        "policy": "one-H100-zero-volatile-and-aggregate-uncorrectable-ecc-clear-recovery-and-row-remap",
    }
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise HealthError("Exactly one CUDA-visible GPU is required")
        properties = torch.cuda.get_device_properties(0)
        if not re.search(r"\bH100\b", properties.name):
            raise HealthError(f"CUDA-visible GPU is not H100: {properties.name!r}")
        assigned_uuid = _uuid(getattr(properties, "uuid", None))
        receipt["cuda_device"] = {"ordinal": 0, "name": properties.name, "uuid": assigned_uuid}
        command = ["nvidia-smi", "-q", "-x", "-i", assigned_uuid]
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
        xml_path.write_text(result.stdout)
        receipt["xml"] = {"path": str(xml_path), "sha256": hashlib.sha256(result.stdout.encode()).hexdigest()}
        receipt["checks"] = inspect_xml(result.stdout, assigned_uuid)
        receipt["status"] = "PASS"
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        raise HealthError(f"GPU admission failed; see {receipt_path}: {error}") from error
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    options = parser.parse_args()
    print(json.dumps(admit_gpu(options.receipt), sort_keys=True))
