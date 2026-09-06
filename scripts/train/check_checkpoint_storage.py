#!/usr/bin/env python3
"""Fail closed unless a target filesystem can retain every planned checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
PASS_STATUS = "CHECKPOINT_STORAGE_VALIDATED"
FAIL_STATUS = "CHECKPOINT_STORAGE_FAILED"


class StorageError(RuntimeError):
    """Raised when filesystem evidence cannot prove the storage contract."""


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _optional_checkpoint_size(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive byte count")
    return parsed


def _reserve(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("safety reserve must be positive")
    return parsed


def storage_evidence(
    target: Path,
    *,
    planned_checkpoint_count: int,
    measured_checkpoint_bytes: int | None,
    expected_checkpoint_bytes: int | None,
    safety_reserve_bytes: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measure a filesystem and calculate a conservative retention budget."""

    resolved = target.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise StorageError(f"filesystem target is not a directory: {resolved}")
    if not os.access(resolved, os.W_OK | os.X_OK):
        raise StorageError(f"filesystem target is not writable/searchable: {resolved}")
    if planned_checkpoint_count <= 0:
        raise StorageError("planned checkpoint count must be positive")
    if measured_checkpoint_bytes is None and expected_checkpoint_bytes is None:
        raise StorageError(
            "at least one of measured or expected checkpoint bytes is required"
        )
    checkpoint_sizes = [
        size
        for size in (measured_checkpoint_bytes, expected_checkpoint_bytes)
        if size is not None
    ]
    if any(size <= 0 for size in checkpoint_sizes):
        raise StorageError("checkpoint byte counts must be positive")
    if safety_reserve_bytes <= 0:
        raise StorageError("safety reserve must be positive")

    stats = os.stat(resolved)
    filesystem = os.statvfs(resolved)
    fragment_size = int(filesystem.f_frsize)
    if fragment_size <= 0:
        raise StorageError("filesystem reported a non-positive fragment size")
    available_bytes = int(filesystem.f_bavail) * fragment_size
    free_bytes = int(filesystem.f_bfree) * fragment_size
    capacity_bytes = int(filesystem.f_blocks) * fragment_size
    if not 0 <= available_bytes <= free_bytes <= capacity_bytes:
        raise StorageError("filesystem reported inconsistent capacity counters")
    read_only_flag = int(getattr(os, "ST_RDONLY", 1))
    is_read_only = bool(int(filesystem.f_flag) & read_only_flag)

    bytes_per_checkpoint = max(checkpoint_sizes)
    checkpoint_budget_bytes = planned_checkpoint_count * bytes_per_checkpoint
    required_available_bytes = checkpoint_budget_bytes + safety_reserve_bytes
    headroom_after_checkpoints_bytes = available_bytes - checkpoint_budget_bytes
    report = {
        "target": str(resolved),
        "filesystem_device": int(stats.st_dev),
        "filesystem_fragment_size_bytes": fragment_size,
        "filesystem_capacity_bytes": capacity_bytes,
        "filesystem_free_bytes": free_bytes,
        "filesystem_available_bytes": available_bytes,
        "filesystem_read_only": is_read_only,
        "planned_checkpoint_count": planned_checkpoint_count,
        "measured_checkpoint_bytes": measured_checkpoint_bytes,
        "expected_checkpoint_bytes": expected_checkpoint_bytes,
        "bytes_per_checkpoint_budget": bytes_per_checkpoint,
        "checkpoint_budget_bytes": checkpoint_budget_bytes,
        "safety_reserve_bytes": safety_reserve_bytes,
        "required_available_bytes": required_available_bytes,
        "headroom_after_checkpoints_bytes": headroom_after_checkpoints_bytes,
    }
    failures = []
    if is_read_only:
        failures.append(
            {"field": "filesystem_read_only", "expected": False, "observed": True}
        )
    if available_bytes < required_available_bytes:
        failures.append(
            {
                "field": "filesystem_available_bytes",
                "expected_at_least": required_available_bytes,
                "observed": available_bytes,
                "shortfall_bytes": required_available_bytes - available_bytes,
            }
        )
    return report, failures


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite storage evidence: {path}")
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
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument(
        "--planned-checkpoint-count", required=True, type=_positive_int
    )
    parser.add_argument(
        "--measured-checkpoint-bytes", type=_optional_checkpoint_size
    )
    parser.add_argument(
        "--expected-checkpoint-bytes", type=_optional_checkpoint_size
    )
    parser.add_argument("--safety-reserve-bytes", required=True, type=_reserve)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if (
        args.measured_checkpoint_bytes is None
        and args.expected_checkpoint_bytes is None
    ):
        parser.error(
            "one or both of --measured-checkpoint-bytes and "
            "--expected-checkpoint-bytes is required"
        )
    return args


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": FAIL_STATUS,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "validator": str(Path(__file__).resolve()),
        "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "failures": [],
    }
    exit_code = 1
    try:
        report, failures = storage_evidence(
            args.target,
            planned_checkpoint_count=args.planned_checkpoint_count,
            measured_checkpoint_bytes=args.measured_checkpoint_bytes,
            expected_checkpoint_bytes=args.expected_checkpoint_bytes,
            safety_reserve_bytes=args.safety_reserve_bytes,
        )
        evidence.update({"storage": report, "failures": failures})
        if not failures:
            evidence["status"] = PASS_STATUS
            exit_code = 0
    except (OSError, StorageError) as exc:
        evidence["failures"] = [{"field": "storage", "error": str(exc)}]
    _atomic_json(output, evidence)
    print(json.dumps(evidence, sort_keys=True, separators=(",", ":")))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
