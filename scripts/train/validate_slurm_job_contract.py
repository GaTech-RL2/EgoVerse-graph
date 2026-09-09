#!/usr/bin/env python3
"""Validate an exact one- or two-GPU Slurm allocation from scontrol output."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
PASS_STATUS = "SLURM_JOB_CONTRACT_VALIDATED"
FAIL_STATUS = "SLURM_JOB_CONTRACT_FAILED"
ALLOWED_CONSTRAINTS = frozenset({"H100", "H200", "H100|H200"})
_FIELD_RE = re.compile(
    r"(?:^|\s)([A-Za-z][A-Za-z0-9_/:.-]*)=(.*?)"
    r"(?=(?:\s+[A-Za-z][A-Za-z0-9_/:.-]*=)|$)"
)
_SIZE_RE = re.compile(r"^([0-9]+)([KMGTPE]?)$", re.IGNORECASE)
_SIZE_FACTORS = {
    "K": 1024,
    "M": 1024**2,
    "G": 1024**3,
    "T": 1024**4,
    "P": 1024**5,
    "E": 1024**6,
}


class ContractError(RuntimeError):
    """Raised when captured scheduler evidence is malformed or ambiguous."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _constraint(value: str) -> str:
    if value not in ALLOWED_CONSTRAINTS:
        raise argparse.ArgumentTypeError(
            "must be one of the exact case-sensitive constraints "
            f"{sorted(ALLOWED_CONSTRAINTS)!r}"
        )
    return value


def parse_scontrol_record(text: str) -> dict[str, str]:
    """Parse one ``scontrol show job -dd -o`` key/value record."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ContractError(
            "scontrol capture must contain exactly one non-empty one-line record; "
            "capture with `scontrol show job -dd -o \"$SLURM_JOB_ID\"`"
        )
    line = lines[0]
    matches = list(_FIELD_RE.finditer(line))
    if not matches or matches[0].start() != 0:
        raise ContractError("scontrol record does not start with a key=value field")
    fields: dict[str, str] = {}
    cursor = 0
    for match in matches:
        if line[cursor : match.start()].strip():
            raise ContractError("scontrol record contains unparsed text")
        key = match.group(1)
        value = match.group(2).strip()
        if key in fields:
            raise ContractError(f"scontrol record repeats field {key}")
        fields[key] = value
        cursor = match.end()
    if line[cursor:].strip():
        raise ContractError("scontrol record has an unparsed suffix")
    return fields


def parse_tres(value: str) -> dict[str, str]:
    """Parse a comma-delimited Slurm TRES map and reject ambiguity."""

    parsed: dict[str, str] = {}
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item or "=" not in item:
            raise ContractError(f"malformed ReqTRES item: {raw_item!r}")
        key, raw_value = item.split("=", 1)
        if not key or not raw_value or key in parsed:
            raise ContractError(f"ambiguous ReqTRES item: {raw_item!r}")
        parsed[key] = raw_value
    return parsed


def parse_slurm_memory(value: str) -> int:
    """Convert Slurm memory syntax to bytes (a bare value means MiB)."""

    match = _SIZE_RE.fullmatch(value)
    if match is None:
        raise ContractError(f"invalid Slurm memory value: {value!r}")
    count = int(match.group(1))
    unit = match.group(2).upper()
    if count <= 0:
        raise ContractError("Slurm memory must be positive")
    factor = 1024**2 if not unit else _SIZE_FACTORS[unit]
    return count * factor


def parse_slurm_duration(value: str) -> int:
    """Convert a finite Slurm duration into seconds."""

    if value.upper() in {"UNLIMITED", "INFINITE", "N/A", "NOT_SET"}:
        raise ContractError(f"wall time must be finite, got {value!r}")
    days = 0
    clock = value
    if "-" in value:
        raw_days, clock = value.split("-", 1)
        if not raw_days.isdigit():
            raise ContractError(f"invalid Slurm duration: {value!r}")
        days = int(raw_days)
    parts = clock.split(":")
    if not all(part.isdigit() for part in parts):
        raise ContractError(f"invalid Slurm duration: {value!r}")
    numbers = [int(part) for part in parts]
    if days:
        if not 1 <= len(numbers) <= 3:
            raise ContractError(f"invalid Slurm duration: {value!r}")
        hours = numbers[0]
        minutes = numbers[1] if len(numbers) >= 2 else 0
        seconds = numbers[2] if len(numbers) == 3 else 0
    elif len(numbers) == 3:
        hours, minutes, seconds = numbers
    elif len(numbers) == 2:
        hours = 0
        minutes, seconds = numbers
    elif len(numbers) == 1:
        hours = 0
        minutes = numbers[0]
        seconds = 0
    else:
        raise ContractError(f"invalid Slurm duration: {value!r}")
    if hours < 0 or minutes < 0 or seconds < 0 or minutes >= 60 or seconds >= 60:
        raise ContractError(f"invalid Slurm duration: {value!r}")
    if days and hours >= 24:
        raise ContractError(f"day-qualified Slurm duration has hour >= 24: {value!r}")
    total = ((days * 24 + hours) * 60 + minutes) * 60 + seconds
    if total <= 0:
        raise ContractError("wall time must be positive")
    return total


def _integer_field(fields: dict[str, str], key: str) -> int:
    raw = fields.get(key)
    if raw is None or not raw.isdigit():
        raise ContractError(f"required scontrol field {key} is absent or non-integer")
    return int(raw)


def _required_field(fields: dict[str, str], key: str) -> str:
    try:
        value = fields[key]
    except KeyError as exc:
        raise ContractError(f"required scontrol field {key} is absent") from exc
    if not value:
        raise ContractError(f"required scontrol field {key} is empty")
    return value


def evaluate_contract(
    fields: dict[str, str],
    *,
    expected_job_id: str,
    expected_account: str,
    expected_partition: str,
    expected_qos: str,
    expected_cpus: int,
    expected_tasks: int = 1,
    expected_gpus: int = 1,
    expected_memory: str,
    expected_time_limit: str,
    expected_constraint: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Return expected values, observed values, and every failed comparison."""

    if expected_constraint not in ALLOWED_CONSTRAINTS:
        raise ContractError(
            "constraint must be one of "
            f"{sorted(ALLOWED_CONSTRAINTS)!r}, got {expected_constraint!r}"
        )
    if not expected_job_id or any(character.isspace() for character in expected_job_id):
        raise ContractError("expected job ID must be a non-empty token")
    if expected_cpus <= 0:
        raise ContractError("expected CPUs must be positive")
    if expected_tasks not in (1, 2):
        raise ContractError("expected tasks must be 1 or 2")
    if expected_gpus != expected_tasks:
        raise ContractError("expected GPUs must equal expected tasks")
    if expected_cpus % expected_tasks:
        raise ContractError("expected CPUs must divide evenly across tasks")
    expected_cpus_per_task = expected_cpus // expected_tasks
    expected_memory_bytes = parse_slurm_memory(expected_memory)
    expected_time_limit_seconds = parse_slurm_duration(expected_time_limit)
    requested_tres = parse_tres(_required_field(fields, "ReqTRES"))
    typed_gpu_tres = sorted(
        key for key in requested_tres if key.startswith("gres/gpu:")
    )

    observed = {
        "job_id": _required_field(fields, "JobId"),
        "account": _required_field(fields, "Account"),
        "partition": _required_field(fields, "Partition"),
        "qos": _required_field(fields, "QOS"),
        "nodes": _integer_field(fields, "NumNodes"),
        "tasks": _integer_field(fields, "NumTasks"),
        "cpus": _integer_field(fields, "NumCPUs"),
        "cpus_per_task": _integer_field(fields, "CPUs/Task"),
        "minimum_cpus_per_node": _integer_field(fields, "MinCPUsNode"),
        "memory_raw": _required_field(fields, "MinMemoryNode"),
        "memory_bytes": parse_slurm_memory(_required_field(fields, "MinMemoryNode")),
        "constraint": _required_field(fields, "Features"),
        "time_limit_raw": _required_field(fields, "TimeLimit"),
        "time_limit_seconds": parse_slurm_duration(
            _required_field(fields, "TimeLimit")
        ),
        "requested_tres": requested_tres,
        "typed_gpu_tres": typed_gpu_tres,
    }
    try:
        observed["requested_tres_nodes"] = int(requested_tres["node"])
        observed["requested_tres_cpus"] = int(requested_tres["cpu"])
        observed["requested_tres_memory_bytes"] = parse_slurm_memory(
            requested_tres["mem"]
        )
        observed["requested_tres_gpus"] = int(requested_tres["gres/gpu"])
    except (KeyError, ValueError) as exc:
        raise ContractError(
            "ReqTRES must contain integer node, cpu, and generic gres/gpu values "
            "plus a valid mem value"
        ) from exc

    expected = {
        "job_id": expected_job_id,
        "account": expected_account,
        "partition": expected_partition,
        "qos": expected_qos,
        "nodes": 1,
        "tasks": expected_tasks,
        "gpus": expected_gpus,
        "cpus": expected_cpus,
        "cpus_per_task": expected_cpus_per_task,
        "minimum_cpus_per_node": expected_cpus,
        "memory_raw": expected_memory,
        "memory_bytes": expected_memory_bytes,
        "memory_mode": "per_node",
        "constraint": expected_constraint,
        "time_limit_raw": expected_time_limit,
        "time_limit_seconds": expected_time_limit_seconds,
        "generic_gpu_request_only": True,
    }
    comparisons = [
        ("job_id", expected_job_id, observed["job_id"]),
        ("account", expected_account, observed["account"]),
        ("partition", expected_partition, observed["partition"]),
        ("qos", expected_qos, observed["qos"]),
        ("nodes", 1, observed["nodes"]),
        ("tasks", expected_tasks, observed["tasks"]),
        ("cpus", expected_cpus, observed["cpus"]),
        ("cpus_per_task", expected_cpus_per_task, observed["cpus_per_task"]),
        (
            "minimum_cpus_per_node",
            expected_cpus,
            observed["minimum_cpus_per_node"],
        ),
        ("memory_bytes", expected_memory_bytes, observed["memory_bytes"]),
        ("constraint", expected_constraint, observed["constraint"]),
        (
            "time_limit_seconds",
            expected_time_limit_seconds,
            observed["time_limit_seconds"],
        ),
        ("requested_tres_nodes", 1, observed["requested_tres_nodes"]),
        ("requested_tres_cpus", expected_cpus, observed["requested_tres_cpus"]),
        (
            "requested_tres_memory_bytes",
            expected_memory_bytes,
            observed["requested_tres_memory_bytes"],
        ),
        ("requested_tres_gpus", expected_gpus, observed["requested_tres_gpus"]),
        ("typed_gpu_tres", [], observed["typed_gpu_tres"]),
    ]
    failures = [
        {"field": field, "expected": expected_value, "observed": observed_value}
        for field, expected_value, observed_value in comparisons
        if expected_value != observed_value
    ]
    return expected, observed, failures


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Slurm contract evidence: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--expected-job-id", required=True)
    parser.add_argument("--expected-account", required=True)
    parser.add_argument("--expected-partition", required=True)
    parser.add_argument("--expected-qos", required=True)
    parser.add_argument("--expected-cpus", required=True, type=_positive_int)
    parser.add_argument("--expected-tasks", type=_positive_int, default=1)
    parser.add_argument("--expected-gpus", type=_positive_int, default=1)
    parser.add_argument("--expected-memory", required=True)
    parser.add_argument("--expected-time-limit", required=True)
    parser.add_argument(
        "--expected-constraint", required=True, type=_constraint, metavar="GPU_CONSTRAINT"
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    record_path = args.record.expanduser().resolve(strict=True)
    record_bytes = record_path.read_bytes()
    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": FAIL_STATUS,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "validator": str(Path(__file__).resolve()),
        "validator_sha256": _sha256(Path(__file__).read_bytes()),
        "record": str(record_path),
        "record_sha256": _sha256(record_bytes),
    }
    exit_code = 1
    try:
        fields = parse_scontrol_record(record_bytes.decode("utf-8", errors="strict"))
        expected, observed, failures = evaluate_contract(
            fields,
            expected_job_id=args.expected_job_id,
            expected_account=args.expected_account,
            expected_partition=args.expected_partition,
            expected_qos=args.expected_qos,
            expected_cpus=args.expected_cpus,
            expected_tasks=args.expected_tasks,
            expected_gpus=args.expected_gpus,
            expected_memory=args.expected_memory,
            expected_time_limit=args.expected_time_limit,
            expected_constraint=args.expected_constraint,
        )
        evidence.update(
            {
                "expected": expected,
                "observed": observed,
                "failures": failures,
            }
        )
        if not failures:
            evidence["status"] = PASS_STATUS
            exit_code = 0
    except (ContractError, UnicodeDecodeError) as exc:
        evidence["failures"] = [{"field": "record", "error": str(exc)}]
    _atomic_json(output, evidence)
    print(json.dumps(evidence, sort_keys=True, separators=(",", ":")))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
