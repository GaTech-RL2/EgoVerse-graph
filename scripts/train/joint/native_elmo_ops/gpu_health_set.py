"""Read-only admission of the exact requested CUDA-visible H100s in one run step."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from gpu_health_r3 import HealthError, _uuid, inspect_xml


def admit_gpu_set(receipt_path, expected):
    if expected not in (2, 4):
        raise HealthError("Expected two/four dedicated H100s")
    import torch
    receipt_path = Path(receipt_path)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    if receipt_path.exists():
        raise HealthError("Refusing to overwrite GPU-set admission")
    record = {"schema": "joint-h100-admission-v1", "status": "FAIL", "devices": [],
              "checked_at_utc": datetime.now(timezone.utc).isoformat()}
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != expected:
            raise HealthError("The exact requested CUDA-visible GPU count is required")
        for ordinal in range(expected):
            properties = torch.cuda.get_device_properties(ordinal)
            assigned_uuid = _uuid(properties.uuid)
            result = subprocess.run(["nvidia-smi", "-q", "-x", "-i", assigned_uuid],
                                    check=True, capture_output=True, text=True, timeout=30)
            xml_path = receipt_path.with_suffix(f".gpu{ordinal}.xml")
            xml_path.write_text(result.stdout)
            checks = inspect_xml(result.stdout, assigned_uuid)
            record["devices"].append({"ordinal": ordinal, "uuid": assigned_uuid,
                                      "name": properties.name, "checks": checks,
                                      "xml": str(xml_path),
                                      "xml_sha256": hashlib.sha256(result.stdout.encode()).hexdigest()})
        if len({device["uuid"] for device in record["devices"]}) != expected:
            raise HealthError("GPU UUIDs must be distinct")
        record["status"] = "PASS"
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        receipt_path.write_text(json.dumps(record, indent=2) + "\n")
        raise
    receipt_path.write_text(json.dumps(record, indent=2) + "\n")
    return record
