#!/usr/bin/env python3
"""Create immutable, content-signed records for the UNITE ICE cutover.

This tool deliberately does not query or mutate either cluster.  A CANDIDATE
record can only be derived from a byte-pinned strict source-validation envelope
and the corresponding signed Skynet-to-ICE staging proof.  An AUTHORIZED record
can only be derived from that signed CANDIDATE and a fresh, signed evidence file
whose raw ``sacct`` and ``squeue`` outputs prove that the exact source allocation
is terminal and absent from the active queue.  Between those two records, a
create-only READY_TO_CANCEL_SOURCE record binds the successor's post-candidate
hardware/full-state smoke.  The source evidence and final authorization are
both bound to that intermediate record, so allocation alone can never authorize
a source cancellation.

The candidate and authorization records use the schemas consumed by
``unite_ice_resume_child.py``.  Record and ``.sha256`` sidecar publication is
create-only: an existing path is never replaced, even when it contains
identical bytes.
"""

from __future__ import annotations

import argparse
import fcntl
import glob
import hashlib
import json
import math
import os
import re
import shlex
import stat
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

SHA256_RE = re.compile(r"[0-9a-f]{64}")
SAFE_TEXT_RE = re.compile(r"[^\t\r\n]+")
EXIT_CODE_RE = re.compile(r"[0-9]+:[0-9]+")
END_TIME_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:[+-][0-9]{2}:[0-9]{2}|Z)?"
)
UTC_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
SAFE_NODE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
SQUEUE_JOB_ID_RE = re.compile(r"[1-9][0-9]*[A-Za-z0-9_.+%\[\]-]*")
SQUEUE_STATE_RE = re.compile(r"[A-Z][A-Z_]*")
TERMINAL_SOURCE_STATES = {
    "CANCELLED",
    "COMPLETED",
    "PREEMPTED",
    "TIMEOUT",
    "NODE_FAIL",
}
MAX_ALLOWED_EVIDENCE_AGE_SECONDS = 600
MAX_MTIME_OBSERVATION_SKEW_NS = 5_000_000_000
MAX_COMMAND_DURATION_NS = 30_000_000_000
MAX_INTER_COMMAND_GAP_NS = 5_000_000_000
SAFE_REMOTE_RE = re.compile(r"[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+")
REMOTE_IDENTITY_MARKER = "__UNITE_CUTOVER_SKYNET_IDENTITY_V1__"
REMOTE_IDENTITY_PROLOGUE = (
    "set -eu; "
    "host=$(/bin/hostname -f); user=$(/usr/bin/id -un); "
    f'printf \'{REMOTE_IDENTITY_MARKER}|%s|%s\\n\' "$host" "$user"; '
    # A clean environment prevents SLURM_CLUSTERS/SLURM_CONF and client-specific
    # SACCT_*/SQUEUE_* variables inherited through SSH from redirecting or
    # altering an otherwise absolute client invocation.
    "exec /usr/bin/env -i LANG=C LC_ALL=C "
)

CANDIDATE_REQUIRED_KEYS = {
    "schema_version",
    "status",
    "source_job_id",
    "successor_job_id",
    "wandb_id",
    "checkpoint",
    "checkpoint_sha256",
    "checkpoint_config_role",
    "checkpoint_config_sha256",
    "config_sha256",
    "global_step",
    "verified_from",
}
CANDIDATE_OPTIONAL_KEYS = {"created_at"}
AUTHORIZATION_REQUIRED_KEYS = CANDIDATE_REQUIRED_KEYS | {
    "source_state",
    "active_queue_state",
    "candidate_record",
    "candidate_record_sha256",
    "ready_to_cancel_record",
    "ready_to_cancel_record_sha256",
    "source_terminal_evidence",
    "source_terminal_evidence_sha256",
    "source_exit_code",
    "source_end_time",
}
AUTHORIZATION_OPTIONAL_KEYS = {
    "created_at",
}

VALIDATION_ENVELOPE_KEYS = {
    "schema_version",
    "status",
    "checkpoint",
    "checkpoint_config",
    "source_checkout",
    "slurm",
    "validator",
}
VALIDATION_CHECKPOINT_KEYS = {
    "expected_global_step",
    "path",
    "sha256_after",
    "sha256_before",
    "stat_after",
    "stat_before",
    "unchanged",
}
VALIDATION_CONFIG_KEYS = {
    "path",
    "sha256_after",
    "sha256_before",
    "unchanged",
}
VALIDATION_CHECKOUT_KEYS = {
    "clean_after",
    "head_after",
    "head_before",
    "path",
    "unchanged",
}
VALIDATION_SLURM_KEYS = {"gpu_allocation", "job_id", "node_list", "node_name"}
VALIDATION_TOOL_KEYS = {
    "exit_code",
    "parse_error",
    "path",
    "regression_test_path",
    "regression_test_sha256",
    "result",
    "sha256",
    "stderr_path",
    "stderr_sha256",
    "stdout_path",
    "stdout_sha256",
}
STAGE_PROOF_KEYS = {
    "schema_version",
    "status",
    "task_id",
    "global_step",
    "source_run_root",
    "remote",
    "access_selection",
    "access_selection_path",
    "access_selection_sha256",
    "source_checkpoint",
    "source_checkpoint_stat_before",
    "source_checkpoint_stat_after",
    "checkpoint",
    "checkpoint_size_bytes",
    "checkpoint_sha256",
    "checkpoint_publication",
    "source_config",
    "source_config_stat_before",
    "source_config_stat_after",
    "config",
    "config_size_bytes",
    "config_sha256",
    "config_publication",
    "semantic_config_sha256",
    "source_validation_result_sha256",
    "checkpoint_transfer_seconds",
    "checkpoint_effective_mib_per_second",
    "config_transfer_seconds",
    "config_effective_mib_per_second",
    "manifest",
    "manifest_sha256",
    "ssh_wrapper",
    "ssh_wrapper_sha256",
    "stager",
    "stager_sha256",
    "authorized_root",
    "completed_at_unix",
}
REMOTE_STAT_KEYS = {
    "device",
    "inode",
    "size",
    "mtime_ns",
    "ctime_ns",
    "mode",
    "uid",
    "gid",
}
ACCESS_SELECTION_KEYS = {
    "schema_version",
    "status",
    "probe_context",
    "canonical_remote",
    "fallback_remote",
    "canonical_probe",
    "fallback_probe",
    "selected_remote",
    "selection_reason",
    "completed_at_utc",
}
ACCESS_PROBE_KEYS = {
    "remote",
    "attempted_at_utc",
    "attempt_count",
    "returncode",
    "hostname",
    "username",
    "stderr_sha256",
}
ACCESS_CONTEXT_KEYS = {
    "cluster",
    "slurm_job_id",
    "slurm_node",
    "account",
    "partition",
    "qos",
}
SOURCE_EVIDENCE_KEYS = {
    "schema_version",
    "status",
    "source_cluster",
    "source_job_id",
    "successor_job_id",
    "wandb_id",
    "candidate_record_sha256",
    "allocation_record_sha256",
    "ready_to_cancel_record_sha256",
    "observed_at_unix_ns",
    "terminal_state",
    "queue",
    "command_evidence",
    "collector",
}
TERMINAL_EVIDENCE_KEYS = {"state", "exit_code", "end_time"}
QUEUE_EVIDENCE_KEYS = {"job_id", "present", "state"}
COMMAND_EVIDENCE_KEYS = {
    "argv",
    "transport_argv",
    "returncode",
    "stdout",
    "stdout_sha256",
    "stderr",
    "stderr_sha256",
    "started_at_unix_ns",
    "completed_at_unix_ns",
}
COLLECTOR_EVIDENCE_KEYS = {
    "path",
    "sha256",
    "record_writer",
    "record_writer_sha256",
    "ssh_wrapper",
    "ssh_wrapper_sha256",
    "access_selection",
    "access_selection_sha256",
    "selected_remote",
    "remote_hostname",
    "remote_username",
    "execution_cluster",
    "slurm_job_id",
    "slurm_node",
}
ALLOCATION_RECORD_KEYS = {
    "schema_version",
    "status",
    "created_at",
    "job_id",
    "restart_count",
    "node_list",
    "host",
    "account",
    "partition",
    "qos",
    "sweep_task_id",
    "source_job_id",
    "wandb_id",
    "candidate_record_sha256",
    "checkpoint_validation",
    "hardware_validation",
    "tool_sha256",
}
READY_TO_CANCEL_KEYS = {
    "schema_version",
    "status",
    "task_id",
    "source_job_id",
    "successor_job_id",
    "wandb_id",
    "candidate_record",
    "candidate_record_sha256",
    "ready_record_sha256",
    "checkpoint",
    "checkpoint_sha256",
    "checkpoint_config_role",
    "checkpoint_config_sha256",
    "config_sha256",
    "global_step",
    "cutover_lease_sha256",
    "continuation_contract_sha256",
    "allocation_record_sha256",
    "cutover_tool_manifest_sha256",
}
CUTOVER_TOOL_MANIFEST_KEYS = {
    "schema_version",
    "status",
    "record_writer_path",
    "record_writer_sha256",
    "resume_child_path",
    "resume_child_sha256",
    "source_terminal_collector_path",
    "source_terminal_collector_sha256",
}
CANDIDATE_PROVENANCE_RE = re.compile(
    r"source_validation_sha256=(?P<source_validation>[0-9a-f]{64});"
    r"stage_proof_sha256=(?P<stage_proof>[0-9a-f]{64});"
    r"staged_baseline_sha256=(?P<staged_baseline>[0-9a-f]{64});"
    r"staged_baseline_step=(?P<staged_step>[1-9][0-9]*);"
    r"continuation_contract_sha256=(?P<contract>[0-9a-f]{64});"
    r"allocation_record_sha256=(?P<allocation>[0-9a-f]{64});"
    r"cutover_tool_manifest_sha256=(?P<tool_manifest>[0-9a-f]{64});"
    r"cutover_lease_sha256=(?P<lease>[0-9a-f]{64});"
    r"selected_remote=(?P<selected_remote>[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+);"
    r"access_selection_sha256=(?P<access_selection>[0-9a-f]{64});"
    r"ssh_wrapper_sha256=(?P<ssh_wrapper>[0-9a-f]{64});"
    r"record_writer_sha256=(?P<record_writer>[0-9a-f]{64});"
    r"source_terminal_collector_sha256=(?P<collector>[0-9a-f]{64})"
)
AUTHORIZATION_PROVENANCE_RE = re.compile(
    r"candidate_record_sha256=(?P<candidate>[0-9a-f]{64});"
    r"ready_to_cancel_record_sha256=(?P<ready>[0-9a-f]{64});"
    r"source_terminal_evidence_sha256=(?P<evidence>[0-9a-f]{64})"
)
CUTOVER_LEASE_KEYS = {
    "schema_version",
    "status",
    "task_id",
    "source_job_id",
    "successor_job_id",
    "wandb_id",
    "continuation_contract_sha256",
    "allocation_record_sha256",
    "cutover_tool_manifest_sha256",
}


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_bytes(raw: bytes, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not strict UTF-8 JSON") from exc


def stat_identity(path: Path) -> tuple[int, int, int, int, int, int]:
    value = os.lstat(path)
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_mode,
    )


def stable_bytes(path: Path, label: str) -> tuple[bytes, os.stat_result]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is not a regular non-symlink file: {path}")
    before = stat_identity(path)
    raw = path.read_bytes()
    after_stat = os.lstat(path)
    after = stat_identity(path)
    if before != after:
        raise RuntimeError(f"{label} changed while being read: {path}")
    return raw, after_stat


def stable_sha256(path: Path, label: str) -> tuple[str, int]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is not a regular non-symlink file: {path}")
    before = stat_identity(path)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    if before != stat_identity(path):
        raise RuntimeError(f"{label} changed while being hashed: {path}")
    return digest.hexdigest(), size


def canonical_json(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    if set(value) != expected:
        raise RuntimeError(
            f"{label} keys are not canonical: "
            f"missing={sorted(expected - set(value))} "
            f"unknown={sorted(set(value) - expected)}"
        )
    return value


def positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"{label} must be a positive JSON integer")
    return value


def nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{label} must be a nonnegative JSON integer")
    return value


def nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or SAFE_TEXT_RE.fullmatch(value) is None:
        raise RuntimeError(f"{label} must be a nonempty single-line JSON string")
    return value


def lowercase_sha256(value: Any, label: str) -> str:
    text = nonempty_string(value, label)
    if SHA256_RE.fullmatch(text) is None:
        raise RuntimeError(f"{label} must be lowercase 64-hex SHA-256")
    return text


def canonical_absolute_file(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise RuntimeError(f"{label} must be absolute")
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a regular non-symlink file: {path}")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise RuntimeError(f"{label} must be canonical: {path}")
    return resolved


def canonical_remote_path(value: Any, label: str) -> PurePosixPath:
    text = nonempty_string(value, label)
    path = PurePosixPath(text)
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError(f"{label} is not a canonical absolute POSIX path")
    if str(path) != text:
        raise RuntimeError(f"{label} is not lexically canonical")
    return path


def signed_json(
    path: Path,
    *,
    expected_sha256: str | None,
    sidecar_absolute_path: bool,
    label: str,
) -> tuple[dict[str, Any], str, os.stat_result]:
    path = canonical_absolute_file(path, label)
    raw, metadata = stable_bytes(path, label)
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None:
        if lowercase_sha256(expected_sha256, f"expected {label} SHA-256") != digest:
            raise RuntimeError(f"{label} SHA-256 mismatch")
    sidecar = Path(str(path) + ".sha256")
    sidecar_raw, _ = stable_bytes(sidecar, f"{label} SHA sidecar")
    named_path = str(path) if sidecar_absolute_path else path.name
    expected_sidecar = f"{digest}  {named_path}\n".encode("utf-8")
    if sidecar_raw != expected_sidecar:
        raise RuntimeError(f"{label} SHA sidecar is not canonical")
    value = strict_json_bytes(raw, label)
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    if raw != canonical_json(value):
        raise RuntimeError(f"{label} is not canonical sorted JSON")
    return value, digest, metadata


def unsigned_exact_json(
    path: Path,
    expected_sha256: str,
    label: str,
    *,
    require_canonical: bool = True,
) -> tuple[dict[str, Any], str]:
    path = canonical_absolute_file(path, label)
    raw, _ = stable_bytes(path, label)
    digest = hashlib.sha256(raw).hexdigest()
    if lowercase_sha256(expected_sha256, f"expected {label} SHA-256") != digest:
        raise RuntimeError(f"{label} SHA-256 mismatch")
    value = strict_json_bytes(raw, label)
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    if require_canonical and raw != canonical_json(value):
        raise RuntimeError(f"{label} is not canonical sorted JSON")
    return value, digest


def verify_allowed_checkpoint(path: Path, patterns: Sequence[str]) -> None:
    if not patterns:
        raise RuntimeError("at least one allowed checkpoint glob is required")
    allowed: set[Path] = set()
    for pattern in patterns:
        if not Path(pattern).is_absolute():
            raise RuntimeError("allowed checkpoint globs must be absolute")
        for raw in glob.glob(pattern, recursive=True):
            candidate = Path(raw)
            if candidate.is_file() and not candidate.is_symlink():
                allowed.add(candidate.resolve(strict=True))
    if path not in allowed:
        raise RuntimeError("candidate checkpoint is outside the allowed globs")


def authorized_root_from_baseline(staged_checkpoint: Path, task_id: str) -> Path:
    checkpoint_dir = staged_checkpoint.parent
    step_dir = checkpoint_dir.parent
    candidates_dir = step_dir.parent
    task_dir = candidates_dir.parent
    staging_dir = task_dir.parent
    if (
        checkpoint_dir.name != "checkpoints"
        or re.fullmatch(r"step-[1-9][0-9]*", step_dir.name) is None
        or candidates_dir.name != "candidates"
        or task_dir.name != task_id
        or staging_dir.name != "staging"
    ):
        raise RuntimeError(
            "staged baseline does not have canonical content-addressed "
            "ROOT/staging/TASK/candidates/step-N/checkpoints identity"
        )
    authorized_root = staging_dir.parent
    if len(authorized_root.parts) < 6:
        raise RuntimeError("derived ICE authorized root is not a specific path")
    return authorized_root


def content_addressed_cutover_tool_manifest_path(
    authorized_root: Path, manifest_sha256: str
) -> Path:
    """Return the immutable manifest path selected by its canonical byte hash."""

    if (
        not authorized_root.is_absolute()
        or authorized_root.is_symlink()
        or not authorized_root.is_dir()
        or authorized_root.resolve(strict=True) != authorized_root
    ):
        raise RuntimeError("cutover authorized root is not canonical")
    digest = lowercase_sha256(manifest_sha256, "cutover tool manifest SHA-256")
    return (
        authorized_root
        / "provenance/cutover-tools"
        / f"tool-manifest-{digest}.json"
    )


def validate_cutover_tool_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    tools = exact_keys(manifest, CUTOVER_TOOL_MANIFEST_KEYS, "cutover tool manifest")
    if tools.get("schema_version") != 1 or tools.get("status") != "PINNED":
        raise RuntimeError("cutover tool manifest is not schema-1 PINNED")
    expected_writer = Path(__file__).resolve(strict=True)
    expected_resume_child = expected_writer.with_name("unite_ice_resume_child.py")
    path_fields = (
        ("record_writer_path", "record_writer_sha256", expected_writer),
        ("resume_child_path", "resume_child_sha256", expected_resume_child),
        (
            "source_terminal_collector_path",
            "source_terminal_collector_sha256",
            expected_writer.with_name("unite_source_terminal_evidence.py"),
        ),
    )
    for path_key, sha_key, exact_path in path_fields:
        path = canonical_absolute_file(
            Path(nonempty_string(tools.get(path_key), f"cutover tool {path_key}")),
            f"cutover tool {path_key}",
        )
        if exact_path is not None and path != exact_path:
            raise RuntimeError(f"cutover tool manifest path mismatch for {path_key}")
        expected_sha = lowercase_sha256(tools.get(sha_key), f"cutover tool {sha_key}")
        actual_sha, _ = stable_sha256(path, f"cutover tool {path_key}")
        if actual_sha != expected_sha:
            raise RuntimeError(f"cutover tool manifest SHA-256 mismatch for {path_key}")
    return tools


def cutover_tool_manifest_payload(
    *, resume_child: Path, source_terminal_collector: Path
) -> dict[str, Any]:
    writer_path = canonical_absolute_file(
        Path(__file__).resolve(), "cutover record writer"
    )
    resume_child = canonical_absolute_file(resume_child, "resume child")
    expected_resume_child = writer_path.with_name("unite_ice_resume_child.py")
    if resume_child != expected_resume_child:
        raise RuntimeError("resume child must be the record writer's exact sibling")
    source_terminal_collector = canonical_absolute_file(
        source_terminal_collector, "source terminal collector"
    )
    payload = {
        "schema_version": 1,
        "status": "PINNED",
        "record_writer_path": str(writer_path),
        "record_writer_sha256": stable_sha256(writer_path, "cutover record writer")[0],
        "resume_child_path": str(resume_child),
        "resume_child_sha256": stable_sha256(resume_child, "resume child")[0],
        "source_terminal_collector_path": str(source_terminal_collector),
        "source_terminal_collector_sha256": stable_sha256(
            source_terminal_collector, "source terminal collector"
        )[0],
    }
    validate_cutover_tool_manifest(payload)
    return payload


def validate_continuation_contract(
    contract: dict[str, Any],
    *,
    task_id: str,
    source_run_root: Path,
    source_job_id: int,
    wandb_id: str,
    staged_checkpoint: Path,
    staged_checkpoint_sha256: str,
    staged_step: int,
    candidate_minimum_step: int,
    checkpoint_step_multiple: int,
) -> dict[str, Any]:
    if contract.get("schema_version") != 2:
        raise RuntimeError("continuation contract must use schema version 2")
    rows = contract.get("rows")
    if not isinstance(rows, dict) or not isinstance(rows.get(task_id), dict):
        raise RuntimeError("continuation contract has no exact task row")
    row = rows[task_id]
    expected_row = {
        "source_job_id": source_job_id,
        "source_run_root": str(source_run_root),
        "wandb_id": wandb_id,
    }
    for key, expected in expected_row.items():
        if row.get(key) != expected or type(row.get(key)) is not type(expected):
            raise RuntimeError(f"continuation contract row mismatch for {key}")
    baseline = row.get("allocation_baseline")
    if not isinstance(baseline, dict):
        raise RuntimeError("continuation contract has no allocation baseline")
    expected_baseline = {
        "path": str(staged_checkpoint),
        "global_step": staged_step,
        "sha256": staged_checkpoint_sha256,
        "eligible_for_training": False,
    }
    for key, expected in expected_baseline.items():
        if baseline.get(key) != expected or type(baseline.get(key)) is not type(
            expected
        ):
            raise RuntimeError(f"continuation allocation baseline mismatch for {key}")
    lowercase_sha256(baseline.get("config_sha256"), "contract baseline config SHA-256")

    candidate_contract = contract.get("candidate_checkpoint_contract")
    training_contract = contract.get("training_contract")
    if not isinstance(candidate_contract, dict) or not isinstance(
        training_contract, dict
    ):
        raise RuntimeError("continuation contract lacks checkpoint cadence")
    if candidate_contract.get("minimum_step") != candidate_minimum_step:
        raise RuntimeError("continuation candidate minimum step mismatch")
    if (
        training_contract.get("checkpoint_every_optimizer_steps")
        != checkpoint_step_multiple
    ):
        raise RuntimeError("continuation checkpoint step multiple mismatch")
    if training_contract.get("wandb_resume") != "must":
        raise RuntimeError("continuation W&B resume contract is not canonical")
    nonempty_string(training_contract.get("wandb_entity"), "continuation W&B entity")
    nonempty_string(training_contract.get("wandb_project"), "continuation W&B project")
    destination = contract.get("destination_resource")
    if not isinstance(destination, dict):
        raise RuntimeError("continuation destination resource is not authorized")
    world_size = destination.get("world_size")
    gpus_per_node = destination.get("gpus_per_node")
    if (
        type(world_size) is not int
        or world_size not in {1, 2}
        or gpus_per_node != world_size
        or type(gpus_per_node) is not int
        or not isinstance(destination.get("gpu_model"), str)
        or not destination.get("gpu_model")
    ):
        raise RuntimeError("continuation destination resource is not authorized")
    for key in ("account", "partition", "qos"):
        nonempty_string(destination.get(key), f"continuation destination {key}")
    return row


def validate_allocation_record(
    record: dict[str, Any],
    *,
    contract_path: Path,
    contract_sha256: str,
    contract: dict[str, Any],
    task_id: str,
    source_job_id: int,
    successor_job_id: int,
    wandb_id: str,
    staged_checkpoint: Path,
    staged_checkpoint_sha256: str,
    staged_step: int,
) -> None:
    exact_keys(record, ALLOCATION_RECORD_KEYS, "allocated-successor record")
    expected = {
        "schema_version": 2,
        "status": "STAGED_CHECKPOINT_STRICT_LOAD_PASS_WAITING_CANDIDATE",
        "job_id": successor_job_id,
        "source_job_id": source_job_id,
        "sweep_task_id": task_id,
        "wandb_id": wandb_id,
        "candidate_record_sha256": "none",
        "account": contract["destination_resource"]["account"],
        "partition": contract["destination_resource"]["partition"],
        "qos": contract["destination_resource"]["qos"],
    }
    for key, value in expected.items():
        if record.get(key) != value or type(record.get(key)) is not type(value):
            raise RuntimeError(f"allocated-successor record mismatch for {key}")
    nonnegative_integer(record.get("restart_count"), "successor restart count")
    for key in ("created_at", "node_list", "host"):
        nonempty_string(record.get(key), f"allocated-successor {key}")

    validation = record.get("checkpoint_validation")
    if not isinstance(validation, dict):
        raise RuntimeError("allocated-successor has no checkpoint validation")
    expected_validation = {
        "status": "passed",
        "checkpoint": str(staged_checkpoint),
        "checkpoint_sha256": staged_checkpoint_sha256,
        "global_step": staged_step,
        "wandb_id": wandb_id,
        "task_id": task_id,
        "checkpoint_config_role": "adjacent_origin_hydra",
        "strict_load": True,
        "full_state_verified": True,
        "strict_optimizer_load": True,
        "strict_scheduler_load": True,
        "tensor_finiteness": "complete",
    }
    for key, value in expected_validation.items():
        if validation.get(key) != value or type(validation.get(key)) is not type(value):
            raise RuntimeError(f"allocated baseline validation mismatch for {key}")
    baseline_config_sha = contract["rows"][task_id]["allocation_baseline"][
        "config_sha256"
    ]
    if validation.get("checkpoint_config_sha256") != baseline_config_sha:
        raise RuntimeError(
            "allocated baseline config differs from continuation contract"
        )

    destination = contract["destination_resource"]
    hardware = record.get("hardware_validation")
    if not isinstance(hardware, dict):
        raise RuntimeError("allocated-successor has no hardware validation")
    expected_hardware = {
        "schema_version": 1,
        "status": "passed",
        "expected_gpu_model": destination["gpu_model"],
        "expected_gpu_count": destination["gpus_per_node"],
    }
    for key, value in expected_hardware.items():
        if hardware.get(key) != value or type(hardware.get(key)) is not type(value):
            raise RuntimeError(f"allocated hardware validation mismatch for {key}")
    rank_probes = hardware.get("rank_probes")
    if (
        not isinstance(rank_probes, list)
        or len(rank_probes) != destination["world_size"]
    ):
        raise RuntimeError("allocated hardware validation rank count is wrong")

    tool_sha = record.get("tool_sha256")
    if not isinstance(tool_sha, dict) or not tool_sha:
        raise RuntimeError("allocated-successor has no tool identity map")
    for path, digest in tool_sha.items():
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise RuntimeError("allocated-successor tool path is not absolute")
        lowercase_sha256(digest, f"allocated tool SHA-256 for {path}")
    if tool_sha.get(str(contract_path)) != contract_sha256:
        raise RuntimeError("allocated-successor used a different continuation contract")
    launcher_sha = contract.get("source_identity", {}).get("portable_launcher_sha256")
    if lowercase_sha256(launcher_sha, "contract portable launcher SHA-256") not in set(
        tool_sha.values()
    ):
        raise RuntimeError(
            "allocated-successor did not use the pinned portable launcher"
        )


def validate_source_validation(
    envelope: dict[str, Any],
    *,
    validator_tool_sha256: str,
    source_run_root: Path,
    task_id: str,
    wandb_id: str,
) -> dict[str, Any]:
    exact_keys(envelope, VALIDATION_ENVELOPE_KEYS, "source validation envelope")
    if envelope["schema_version"] != 1 or envelope["status"] != "passed":
        raise RuntimeError("source validation envelope did not pass schema 1")

    checkpoint = exact_keys(
        envelope["checkpoint"], VALIDATION_CHECKPOINT_KEYS, "validated checkpoint"
    )
    config = exact_keys(
        envelope["checkpoint_config"],
        VALIDATION_CONFIG_KEYS,
        "validated checkpoint config",
    )
    checkout = exact_keys(
        envelope["source_checkout"],
        VALIDATION_CHECKOUT_KEYS,
        "source checkout evidence",
    )
    slurm = exact_keys(
        envelope["slurm"], VALIDATION_SLURM_KEYS, "validation Slurm evidence"
    )
    tool = exact_keys(envelope["validator"], VALIDATION_TOOL_KEYS, "validator evidence")

    source_run = canonical_remote_path(str(source_run_root), "source run root")
    source_checkpoint = canonical_remote_path(
        checkpoint["path"], "validated checkpoint.path"
    )
    source_config = canonical_remote_path(config["path"], "validated config.path")
    if source_checkpoint.parent != source_run / "checkpoints":
        raise RuntimeError("validated checkpoint is outside the bound source run")
    if source_checkpoint.suffix != ".ckpt":
        raise RuntimeError("validated checkpoint does not have a .ckpt leaf")
    if source_config != source_run / ".hydra/config.yaml":
        raise RuntimeError("validated config is not the bound source run config")
    expected_step = positive_integer(
        checkpoint["expected_global_step"], "validated expected_global_step"
    )
    checkpoint_sha = lowercase_sha256(
        checkpoint["sha256_before"], "validated checkpoint SHA-256"
    )
    if (
        checkpoint["unchanged"] is not True
        or checkpoint["sha256_after"] != checkpoint_sha
        or checkpoint["stat_before"] != checkpoint["stat_after"]
    ):
        raise RuntimeError("validated checkpoint was not stable")
    config_sha = lowercase_sha256(config["sha256_before"], "validated config SHA-256")
    if config["unchanged"] is not True or config["sha256_after"] != config_sha:
        raise RuntimeError("validated config was not stable")
    if (
        checkout["unchanged"] is not True
        or checkout["clean_after"] is not True
        or checkout["head_before"] != checkout["head_after"]
    ):
        raise RuntimeError("source checkout was not clean and stable")
    if (
        slurm["gpu_allocation"] is not False
        or positive_integer(slurm["job_id"], "validation Slurm job ID") < 1
    ):
        raise RuntimeError("source strict validation was not a CPU Slurm job")
    if (
        tool["exit_code"] != 0
        or tool["parse_error"] is not None
        or lowercase_sha256(tool["sha256"], "validator tool SHA-256")
        != lowercase_sha256(validator_tool_sha256, "expected validator tool SHA-256")
    ):
        raise RuntimeError("strict checkpoint validator identity/result is wrong")
    for key in ("regression_test_sha256", "stderr_sha256", "stdout_sha256"):
        lowercase_sha256(tool[key], f"validator {key}")

    metadata = tool["result"]
    if not isinstance(metadata, dict):
        raise RuntimeError("validator result metadata must be a JSON object")
    expected_exact = {
        "status": "passed",
        "checkpoint": str(source_checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "global_step": expected_step,
        "task_id": task_id,
        "wandb_id": wandb_id,
        "checkpoint_config_role": "adjacent_origin_hydra",
        "checkpoint_config_sha256": config_sha,
        "strict_load": True,
        "full_state_verified": True,
        "strict_optimizer_load": True,
        "strict_scheduler_load": True,
        "tensor_finiteness": "complete",
        "rebased_schedule_verified": True,
    }
    for key, expected in expected_exact.items():
        if metadata.get(key) != expected or type(metadata.get(key)) is not type(
            expected
        ):
            raise RuntimeError(
                f"validator metadata mismatch for {key}: "
                f"{metadata.get(key)!r} != {expected!r}"
            )
    config_identity_sha = lowercase_sha256(
        metadata.get("config_sha256"), "validator semantic config SHA-256"
    )
    if metadata.get("embedded_config_identity_sha256") != config_identity_sha:
        raise RuntimeError("validator semantic config identity aliases disagree")
    if metadata.get("config_path") != str(source_config):
        raise RuntimeError("validator config path differs from validation envelope")
    canonical_stdout = (json.dumps(metadata, sort_keys=True) + "\n").encode("utf-8")
    if hashlib.sha256(canonical_stdout).hexdigest() != tool["stdout_sha256"]:
        raise RuntimeError("validator stdout SHA-256 does not bind result metadata")
    return {
        "source_run_root": str(source_run),
        "source_checkpoint": str(source_checkpoint),
        "source_config": str(source_config),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_config_sha256": config_sha,
        "config_sha256": config_identity_sha,
        "global_step": expected_step,
        "validator_stdout_sha256": tool["stdout_sha256"],
        "validator_stderr_sha256": tool["stderr_sha256"],
    }


def validate_validator_transcripts(
    *,
    validation_envelope: dict[str, Any],
    validator_stdout: Path,
    validator_stdout_sha256: str,
    validator_stderr: Path,
    validator_stderr_sha256: str,
) -> None:
    """Verify the exact copied validator streams bound by the envelope.

    The strict validator legitimately emits pinned-library warnings on stderr.
    Those bytes are evidence, not a failure condition: the envelope's typed
    exit/status fields decide success, while both streams must match their
    explicit CLI identities and the envelope byte-for-byte.
    """

    tool = exact_keys(
        validation_envelope.get("validator"),
        VALIDATION_TOOL_KEYS,
        "validator transcript evidence",
    )
    stdout_expected = lowercase_sha256(
        validator_stdout_sha256, "expected validator stdout SHA-256"
    )
    stderr_expected = lowercase_sha256(
        validator_stderr_sha256, "expected validator stderr SHA-256"
    )
    if tool.get("stdout_sha256") != stdout_expected:
        raise RuntimeError("validator stdout CLI identity differs from envelope")
    if tool.get("stderr_sha256") != stderr_expected:
        raise RuntimeError("validator stderr CLI identity differs from envelope")

    stdout_path = canonical_absolute_file(validator_stdout, "validator stdout mirror")
    stderr_path = canonical_absolute_file(validator_stderr, "validator stderr mirror")
    stdout_raw, _ = stable_bytes(stdout_path, "validator stdout mirror")
    stderr_raw, _ = stable_bytes(stderr_path, "validator stderr mirror")
    if hashlib.sha256(stdout_raw).hexdigest() != stdout_expected:
        raise RuntimeError("validator stdout mirror SHA-256 mismatch")
    if hashlib.sha256(stderr_raw).hexdigest() != stderr_expected:
        raise RuntimeError("validator stderr mirror SHA-256 mismatch")
    metadata = tool.get("result")
    if not isinstance(metadata, dict):
        raise RuntimeError("validator result metadata must be a JSON object")
    canonical_stdout = (json.dumps(metadata, sort_keys=True) + "\n").encode("utf-8")
    if stdout_raw != canonical_stdout:
        raise RuntimeError("validator stdout mirror differs from exact result metadata")


def validate_access_probe(
    value: Any, *, remote: str, hostname: str, username: str
) -> dict[str, Any]:
    probe = exact_keys(value, ACCESS_PROBE_KEYS, f"access probe for {remote}")
    if probe.get("remote") != remote:
        raise RuntimeError(f"access probe remote mismatch for {remote}")
    positive_integer(probe.get("attempt_count"), f"access attempts for {remote}")
    returncode = nonnegative_integer(
        probe.get("returncode"), f"access return code for {remote}"
    )
    if returncode > 255:
        raise RuntimeError(f"access return code is invalid for {remote}")
    lowercase_sha256(probe.get("stderr_sha256"), f"access stderr SHA-256 for {remote}")
    attempted_at = nonempty_string(
        probe.get("attempted_at_utc"), f"access probe attempted_at_utc for {remote}"
    )
    if UTC_RE.fullmatch(attempted_at) is None:
        raise RuntimeError(f"access probe timestamp is invalid for {remote}")
    for key in ("hostname", "username"):
        if probe.get(key) is not None and not isinstance(probe.get(key), str):
            raise RuntimeError(f"access probe {key} is invalid for {remote}")
    if returncode == 0 and (
        probe.get("hostname") != hostname or probe.get("username") != username
    ):
        raise RuntimeError(f"successful access identity mismatch for {remote}")
    return probe


def validate_access_selection(value: Any) -> dict[str, Any]:
    selection = exact_keys(value, ACCESS_SELECTION_KEYS, "ICE Skynet access selection")
    if selection.get("schema_version") != 1 or selection.get("status") != "READY":
        raise RuntimeError("ICE Skynet access selection is not schema-1 READY")
    context = exact_keys(
        selection.get("probe_context"), ACCESS_CONTEXT_KEYS, "access probe context"
    )
    if context.get("cluster") != "PACE ICE" or context.get("partition") != "ice-cpu":
        raise RuntimeError("access probe context is not an ICE CPU allocation")
    for key in ("account", "qos"):
        nonempty_string(context.get(key), f"access probe context {key}")
    job_id = context.get("slurm_job_id")
    if not isinstance(job_id, str) or re.fullmatch(r"[1-9][0-9]*", job_id) is None:
        raise RuntimeError("access selection Slurm job ID is invalid")
    node = nonempty_string(context.get("slurm_node"), "access selection Slurm node")
    if SAFE_NODE_RE.fullmatch(node) is None:
        raise RuntimeError("access selection Slurm node is invalid")
    canonical_remote = nonempty_string(
        selection.get("canonical_remote"), "canonical remote"
    )
    fallback_remote = nonempty_string(selection.get("fallback_remote"), "fallback remote")
    if (
        SAFE_REMOTE_RE.fullmatch(canonical_remote) is None
        or SAFE_REMOTE_RE.fullmatch(fallback_remote) is None
        or canonical_remote == fallback_remote
    ):
        raise RuntimeError("access selection remotes are invalid")
    canonical_user, canonical_hostname = canonical_remote.split("@", 1)
    fallback_user, fallback_hostname = fallback_remote.split("@", 1)
    if canonical_user != fallback_user:
        raise RuntimeError("access selection remotes must use the same user")
    canonical = validate_access_probe(
        selection.get("canonical_probe"),
        remote=canonical_remote,
        hostname=canonical_hostname,
        username=canonical_user,
    )
    if canonical["returncode"] == 0:
        if (
            selection.get("fallback_probe") is not None
            or selection.get("selected_remote") != canonical_remote
            or selection.get("selection_reason") != "canonical_probe_succeeded"
        ):
            raise RuntimeError("successful canonical access selection is inconsistent")
    else:
        if canonical["attempt_count"] < 2:
            raise RuntimeError("fallback access requires two failed canonical probes")
        fallback = validate_access_probe(
            selection.get("fallback_probe"),
            remote=fallback_remote,
            hostname=fallback_hostname,
            username=fallback_user,
        )
        if (
            fallback["returncode"] != 0
            or selection.get("selected_remote") != fallback_remote
            or selection.get("selection_reason")
            != "canonical_probe_failed_fallback_succeeded"
        ):
            raise RuntimeError("fallback access selection is inconsistent")
    completed_at = nonempty_string(
        selection.get("completed_at_utc"), "access selection completion time"
    )
    if UTC_RE.fullmatch(completed_at) is None:
        raise RuntimeError("access selection completion timestamp is invalid")
    return selection


def validate_stage_provenance(
    proof: dict[str, Any], proof_path: Path, expected_authorized_root: Path
) -> None:
    authorized_root = Path(
        nonempty_string(proof.get("authorized_root"), "stage authorized root")
    )
    if not authorized_root.is_absolute():
        raise RuntimeError("stage authorized root is not absolute")
    if authorized_root != expected_authorized_root:
        raise RuntimeError(
            "stage authorized root differs from continuation baseline root"
        )
    expected_proof_parent = authorized_root / "provenance/candidate-transfers"
    if proof_path.parent != expected_proof_parent:
        raise RuntimeError("candidate stage proof is outside its authorized root")

    selection = validate_access_selection(proof.get("access_selection"))
    if proof.get("remote") != selection["selected_remote"]:
        raise RuntimeError("stage remote differs from the selected Skynet remote")
    artifacts = (
        ("access_selection_path", "access_selection_sha256", "access selection"),
        ("manifest", "manifest_sha256", "stage manifest"),
        ("ssh_wrapper", "ssh_wrapper_sha256", "SSH wrapper"),
        ("stager", "stager_sha256", "candidate stager"),
    )
    artifact_bytes: dict[str, bytes] = {}
    for path_key, sha_key, label in artifacts:
        path = canonical_absolute_file(
            Path(nonempty_string(proof.get(path_key), f"stage {path_key}")), label
        )
        expected_sha = lowercase_sha256(proof.get(sha_key), f"stage {sha_key}")
        raw, _ = stable_bytes(path, label)
        actual_sha = hashlib.sha256(raw).hexdigest()
        if actual_sha != expected_sha:
            raise RuntimeError(f"stage provenance SHA-256 mismatch for {label}")
        artifact_bytes[path_key] = raw
        if path_key == "access_selection_path":
            embedded = strict_json_bytes(raw, "access selection artifact")
            if embedded != selection:
                raise RuntimeError(
                    "embedded access selection differs from its artifact"
                )

    manifest = strict_json_bytes(artifact_bytes["manifest"], "stage manifest")
    exact_keys(
        manifest,
        {"schema_version", "access_selection_sha256", "destination_root", "rows"},
        "stage manifest",
    )
    if manifest.get("schema_version") != 2:
        raise RuntimeError("stage manifest is not schema version 2")
    if manifest.get("destination_root") != str(authorized_root / "staging"):
        raise RuntimeError("stage manifest destination root is wrong")
    if manifest.get("access_selection_sha256") != proof.get("access_selection_sha256"):
        raise RuntimeError("stage manifest access-selection identity differs")
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("stage manifest rows are invalid")
    matching_rows = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("task_id") == proof.get("task_id")
    ]
    if len(matching_rows) != 1:
        raise RuntimeError("stage manifest does not contain one exact task row")
    manifest_row = matching_rows[0]
    expected_row = {
        "task_id": proof.get("task_id"),
        "global_step": proof.get("global_step"),
        "source_run_root": proof.get("source_run_root"),
        "source_checkpoint": proof.get("source_checkpoint"),
        "checkpoint_size_bytes": proof.get("checkpoint_size_bytes"),
        "checkpoint_sha256": proof.get("checkpoint_sha256"),
        "source_config": proof.get("source_config"),
        "config_size_bytes": proof.get("config_size_bytes"),
        "config_sha256": proof.get("config_sha256"),
        "semantic_config_sha256": proof.get("semantic_config_sha256"),
        "source_validation_result_sha256": proof.get("source_validation_result_sha256"),
    }
    if manifest_row != expected_row:
        raise RuntimeError("stage proof differs from its exact manifest row")

    task_id = nonempty_string(proof.get("task_id"), "stage task_id")
    global_step = positive_integer(proof.get("global_step"), "stage global_step")
    manifest_sha256 = lowercase_sha256(
        proof.get("manifest_sha256"), "stage manifest_sha256"
    )
    allowed_proof_names = {
        f"{task_id}-step-{global_step}.json",
        f"{task_id}-step-{global_step}-manifest-{manifest_sha256}.json",
    }
    if proof_path.name not in allowed_proof_names:
        raise RuntimeError("candidate stage proof filename identity is wrong")

    for prefix, expected_size in (
        ("source_checkpoint", proof.get("checkpoint_size_bytes")),
        ("source_config", proof.get("config_size_bytes")),
    ):
        before = proof.get(f"{prefix}_stat_before")
        after = proof.get(f"{prefix}_stat_after")
        if (
            not isinstance(before, dict)
            or set(before) != REMOTE_STAT_KEYS
            or before != after
            or before.get("size") != expected_size
            or any(type(item) is not int or item < 0 for item in before.values())
            or not stat.S_ISREG(before["mode"])
        ):
            raise RuntimeError(f"candidate stage proof {prefix} stat is invalid")

    for prefix in ("checkpoint", "config"):
        action = proof.get(f"{prefix}_publication")
        seconds = proof.get(f"{prefix}_transfer_seconds")
        throughput = proof.get(f"{prefix}_effective_mib_per_second")
        if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
            raise RuntimeError(f"candidate stage proof {prefix} duration is invalid")
        if not math.isfinite(seconds) or seconds < 0:
            raise RuntimeError(f"candidate stage proof {prefix} duration is invalid")
        if action == "reused":
            if seconds != 0 or throughput is not None:
                raise RuntimeError(f"reused {prefix} has transfer metrics")
        elif (
            seconds <= 0
            or not isinstance(throughput, (int, float))
            or isinstance(throughput, bool)
            or not math.isfinite(throughput)
            or throughput <= 0
        ):
            raise RuntimeError(f"transferred {prefix} metrics are invalid")
    completed_unix = proof.get("completed_at_unix")
    if (
        not isinstance(completed_unix, (int, float))
        or isinstance(completed_unix, bool)
        or not math.isfinite(completed_unix)
        or completed_unix <= 0
    ):
        raise RuntimeError("candidate stage proof completion time is invalid")


def validate_stage_proof(
    proof: dict[str, Any],
    *,
    proof_path: Path,
    validation_sha256: str,
    validation: dict[str, Any],
    candidate_checkpoint: Path,
    task_id: str,
    authorized_root: Path,
) -> dict[str, Any]:
    exact_keys(proof, STAGE_PROOF_KEYS, "candidate stage proof")
    validate_stage_provenance(proof, proof_path, authorized_root)
    expected_values = {
        "schema_version": 2,
        "status": "STAGED_SHA256_VERIFIED_NO_OVERWRITE",
        "task_id": task_id,
        "global_step": validation["global_step"],
        "source_run_root": validation["source_run_root"],
        "source_checkpoint": validation["source_checkpoint"],
        "source_config": validation["source_config"],
        "checkpoint": str(candidate_checkpoint),
        "checkpoint_sha256": validation["checkpoint_sha256"],
        "config_sha256": validation["checkpoint_config_sha256"],
        "semantic_config_sha256": validation["config_sha256"],
        "source_validation_result_sha256": validation_sha256,
    }
    for key, expected in expected_values.items():
        if proof.get(key) != expected or type(proof.get(key)) is not type(expected):
            raise RuntimeError(
                f"candidate stage proof mismatch for {key}: "
                f"{proof.get(key)!r} != {expected!r}"
            )
    expected_candidate_parent = (
        authorized_root
        / "staging"
        / task_id
        / "candidates"
        / f"step-{validation['global_step']}"
        / "checkpoints"
    )
    if candidate_checkpoint.parent != expected_candidate_parent:
        raise RuntimeError("candidate checkpoint is outside the continuation ICE root")
    if proof.get("checkpoint_publication") not in {
        "published",
        "reused",
        "reused_concurrent",
    }:
        raise RuntimeError("candidate stage proof checkpoint publication is invalid")
    if proof.get("config_publication") not in {
        "published",
        "reused",
        "reused_concurrent",
    }:
        raise RuntimeError("candidate stage proof config publication is invalid")
    for key in (
        "access_selection_sha256",
        "manifest_sha256",
        "ssh_wrapper_sha256",
        "stager_sha256",
    ):
        lowercase_sha256(proof.get(key), f"candidate stage proof {key}")
    candidate_size = positive_integer(
        proof.get("checkpoint_size_bytes"), "candidate checkpoint size"
    )
    candidate_sha, actual_size = stable_sha256(
        candidate_checkpoint, "staged candidate checkpoint"
    )
    if (
        candidate_sha != validation["checkpoint_sha256"]
        or actual_size != candidate_size
    ):
        raise RuntimeError("staged candidate checkpoint bytes differ from proof")

    config_path = Path(
        nonempty_string(proof.get("config"), "candidate stage proof config")
    )
    config_path = canonical_absolute_file(config_path, "staged candidate config")
    expected_config_path = candidate_checkpoint.parent.parent / ".hydra/config.yaml"
    if config_path != expected_config_path:
        raise RuntimeError("staged candidate config is not adjacent to checkpoint")
    config_size = positive_integer(
        proof.get("config_size_bytes"), "candidate config size"
    )
    actual_config_sha, actual_config_size = stable_sha256(
        config_path, "staged candidate config"
    )
    if (
        actual_config_sha != validation["checkpoint_config_sha256"]
        or actual_config_size != config_size
    ):
        raise RuntimeError("staged candidate config bytes differ from proof")
    return {
        "checkpoint": candidate_checkpoint,
        "checkpoint_sha256": candidate_sha,
        "checkpoint_config_sha256": actual_config_sha,
        "config_sha256": validation["config_sha256"],
        "global_step": validation["global_step"],
    }


def candidate_payload(
    *,
    cutover_tool_manifest: Path,
    cutover_tool_manifest_sha256: str,
    continuation_contract: Path,
    continuation_contract_sha256: str,
    allocation_record: Path,
    allocation_record_sha256: str,
    source_validation_result: Path,
    source_validation_result_sha256: str,
    validator_tool_sha256: str,
    validator_stdout: Path,
    validator_stdout_sha256: str,
    validator_stderr: Path,
    validator_stderr_sha256: str,
    stage_proof: Path,
    stage_proof_sha256: str,
    task_id: str,
    source_run_root: Path,
    source_job_id: int,
    successor_job_id: int,
    wandb_id: str,
    candidate_checkpoint: Path,
    staged_checkpoint: Path,
    staged_checkpoint_sha256: str,
    staged_step: int,
    candidate_minimum_step: int,
    checkpoint_step_multiple: int,
    allowed_checkpoint_globs: Sequence[str],
) -> dict[str, Any]:
    source_job_id = positive_integer(source_job_id, "source job ID")
    successor_job_id = positive_integer(successor_job_id, "successor job ID")
    staged_step = positive_integer(staged_step, "staged baseline step")
    candidate_minimum_step = positive_integer(
        candidate_minimum_step, "candidate minimum step"
    )
    checkpoint_step_multiple = positive_integer(
        checkpoint_step_multiple, "checkpoint step multiple"
    )
    task_id = nonempty_string(task_id, "task ID")
    wandb_id = nonempty_string(wandb_id, "W&B ID")
    candidate_checkpoint = canonical_absolute_file(
        candidate_checkpoint, "staged candidate checkpoint"
    )
    staged_checkpoint = canonical_absolute_file(
        staged_checkpoint, "staged allocation baseline checkpoint"
    )
    expected_staged_sha = lowercase_sha256(
        staged_checkpoint_sha256, "staged baseline SHA-256"
    )
    actual_staged_sha, _ = stable_sha256(
        staged_checkpoint, "staged allocation baseline checkpoint"
    )
    if actual_staged_sha != expected_staged_sha:
        raise RuntimeError("staged allocation baseline checkpoint SHA-256 mismatch")
    authorized_root = authorized_root_from_baseline(staged_checkpoint, task_id)
    cutover_tool_manifest = canonical_absolute_file(
        cutover_tool_manifest, "cutover tool manifest"
    )
    expected_tool_manifest = content_addressed_cutover_tool_manifest_path(
        authorized_root, cutover_tool_manifest_sha256
    )
    if cutover_tool_manifest != expected_tool_manifest:
        raise RuntimeError("cutover tool manifest path is not canonical")
    cutover_tools, cutover_tool_manifest_sha, _ = signed_json(
        cutover_tool_manifest,
        expected_sha256=cutover_tool_manifest_sha256,
        sidecar_absolute_path=True,
        label="cutover tool manifest",
    )
    cutover_tools = validate_cutover_tool_manifest(cutover_tools)
    source_validation_result = canonical_absolute_file(
        source_validation_result, "source validation result"
    )
    continuation_contract = canonical_absolute_file(
        continuation_contract, "continuation contract"
    )
    contract, contract_sha = unsigned_exact_json(
        continuation_contract,
        continuation_contract_sha256,
        "continuation contract",
        require_canonical=False,
    )
    validate_continuation_contract(
        contract,
        task_id=task_id,
        source_run_root=source_run_root,
        source_job_id=source_job_id,
        wandb_id=wandb_id,
        staged_checkpoint=staged_checkpoint,
        staged_checkpoint_sha256=staged_checkpoint_sha256,
        staged_step=staged_step,
        candidate_minimum_step=candidate_minimum_step,
        checkpoint_step_multiple=checkpoint_step_multiple,
    )
    allocation_record = canonical_absolute_file(
        allocation_record, "allocated-successor record"
    )
    allocation, allocation_sha, _ = signed_json(
        allocation_record,
        expected_sha256=allocation_record_sha256,
        sidecar_absolute_path=True,
        label="allocated-successor record",
    )
    validate_allocation_record(
        allocation,
        contract_path=continuation_contract,
        contract_sha256=contract_sha,
        contract=contract,
        task_id=task_id,
        source_job_id=source_job_id,
        successor_job_id=successor_job_id,
        wandb_id=wandb_id,
        staged_checkpoint=staged_checkpoint,
        staged_checkpoint_sha256=staged_checkpoint_sha256,
        staged_step=staged_step,
    )
    validation_envelope, validation_sha = unsigned_exact_json(
        source_validation_result,
        source_validation_result_sha256,
        "source validation result",
    )
    validation = validate_source_validation(
        validation_envelope,
        validator_tool_sha256=validator_tool_sha256,
        source_run_root=source_run_root,
        task_id=task_id,
        wandb_id=wandb_id,
    )
    validate_validator_transcripts(
        validation_envelope=validation_envelope,
        validator_stdout=validator_stdout,
        validator_stdout_sha256=validator_stdout_sha256,
        validator_stderr=validator_stderr,
        validator_stderr_sha256=validator_stderr_sha256,
    )
    stage_proof = canonical_absolute_file(stage_proof, "candidate stage proof")
    proof, proof_sha, _ = signed_json(
        stage_proof,
        expected_sha256=stage_proof_sha256,
        sidecar_absolute_path=False,
        label="candidate stage proof",
    )
    verify_allowed_checkpoint(candidate_checkpoint, allowed_checkpoint_globs)
    staged = validate_stage_proof(
        proof,
        proof_path=stage_proof,
        validation_sha256=validation_sha,
        validation=validation,
        candidate_checkpoint=candidate_checkpoint,
        task_id=task_id,
        authorized_root=authorized_root,
    )
    step = staged["global_step"]
    if step < staged_step:
        raise RuntimeError("candidate checkpoint is older than staged baseline")
    if step == staged_step and staged["checkpoint_sha256"] != expected_staged_sha:
        raise RuntimeError("equal-step candidate differs from staged baseline")
    if step < candidate_minimum_step or step % checkpoint_step_multiple != 0:
        raise RuntimeError("candidate step violates minimum/periodic contract")

    selected_remote = nonempty_string(proof["remote"], "selected Skynet remote")
    access_selection_sha = lowercase_sha256(
        proof["access_selection_sha256"], "access selection SHA-256"
    )
    ssh_wrapper_sha = lowercase_sha256(
        proof["ssh_wrapper_sha256"], "SSH wrapper SHA-256"
    )
    collector_sha = lowercase_sha256(
        cutover_tools["source_terminal_collector_sha256"],
        "source terminal collector SHA-256",
    )
    record_writer_sha = lowercase_sha256(
        cutover_tools["record_writer_sha256"], "cutover record writer SHA-256"
    )

    lease_dir = authorized_root / "provenance/cutover-leases"
    provenance_dir = lease_dir.parent
    if (
        provenance_dir.is_symlink()
        or not provenance_dir.is_dir()
        or provenance_dir.resolve(strict=True) != provenance_dir
    ):
        raise RuntimeError("ICE provenance directory is not canonical")
    try:
        lease_dir.mkdir(mode=0o700)
    except FileExistsError:
        pass
    if (
        lease_dir.is_symlink()
        or not lease_dir.is_dir()
        or lease_dir.resolve(strict=True) != lease_dir
    ):
        raise RuntimeError("cutover lease directory is not canonical")
    lease_path = lease_dir / f"source-job_{source_job_id}.json"
    lease_payload = {
        "schema_version": 1,
        "status": "SUCCESSOR_CLAIMED_FOR_SOURCE_CUTOVER",
        "task_id": task_id,
        "source_job_id": source_job_id,
        "successor_job_id": successor_job_id,
        "wandb_id": wandb_id,
        "continuation_contract_sha256": contract_sha,
        "allocation_record_sha256": allocation_sha,
        "cutover_tool_manifest_sha256": cutover_tool_manifest_sha,
    }
    if set(lease_payload) != CUTOVER_LEASE_KEYS:
        raise AssertionError("internal cutover lease schema drift")
    lease_sha = publish_signed_no_overwrite(lease_path, lease_payload)

    return {
        "schema_version": 1,
        "status": "CANDIDATE",
        "source_job_id": source_job_id,
        "successor_job_id": successor_job_id,
        "wandb_id": wandb_id,
        "checkpoint": str(staged["checkpoint"]),
        "checkpoint_sha256": staged["checkpoint_sha256"],
        "checkpoint_config_role": "adjacent_origin_hydra",
        "checkpoint_config_sha256": staged["checkpoint_config_sha256"],
        "config_sha256": staged["config_sha256"],
        "global_step": step,
        "verified_from": (
            f"source_validation_sha256={validation_sha};"
            f"stage_proof_sha256={proof_sha};"
            f"staged_baseline_sha256={expected_staged_sha};"
            f"staged_baseline_step={staged_step};"
            f"continuation_contract_sha256={contract_sha};"
            f"allocation_record_sha256={allocation_sha};"
            f"cutover_tool_manifest_sha256={cutover_tool_manifest_sha};"
            f"cutover_lease_sha256={lease_sha};"
            f"selected_remote={selected_remote};"
            f"access_selection_sha256={access_selection_sha};"
            f"ssh_wrapper_sha256={ssh_wrapper_sha};"
            f"record_writer_sha256={record_writer_sha};"
            f"source_terminal_collector_sha256={collector_sha}"
        ),
    }


def validate_candidate_transition_record(
    candidate: dict[str, Any],
    *,
    candidate_sha256: str,
    task_id: str,
    source_job_id: int,
    successor_job_id: int,
    wandb_id: str,
    staged_checkpoint: Path,
    staged_checkpoint_sha256: str,
    staged_step: int,
    candidate_minimum_step: int,
    checkpoint_step_multiple: int,
    allowed_checkpoint_globs: Sequence[str],
    continuation_contract_sha256: str,
    allocation_record_sha256: str,
    cutover_tool_manifest_sha256: str,
) -> tuple[dict[str, Any], re.Match[str], Path]:
    keys = set(candidate)
    if not CANDIDATE_REQUIRED_KEYS.issubset(keys) or not keys.issubset(
        CANDIDATE_REQUIRED_KEYS | CANDIDATE_OPTIONAL_KEYS
    ):
        raise RuntimeError("candidate record keys do not match resume-child schema")
    for key in CANDIDATE_OPTIONAL_KEYS.intersection(candidate):
        nonempty_string(candidate[key], f"candidate optional field {key}")
    expected = {
        "schema_version": 1,
        "status": "CANDIDATE",
        "source_job_id": source_job_id,
        "successor_job_id": successor_job_id,
        "wandb_id": wandb_id,
        "checkpoint_config_role": "adjacent_origin_hydra",
    }
    for key, value in expected.items():
        if candidate.get(key) != value or type(candidate.get(key)) is not type(value):
            raise RuntimeError(f"candidate record mismatch for {key}")
    checkpoint = canonical_absolute_file(
        Path(nonempty_string(candidate.get("checkpoint"), "candidate checkpoint")),
        "candidate checkpoint",
    )
    checkpoint_sha = lowercase_sha256(
        candidate.get("checkpoint_sha256"), "candidate checkpoint SHA-256"
    )
    actual_checkpoint_sha, _ = stable_sha256(checkpoint, "candidate checkpoint")
    if actual_checkpoint_sha != checkpoint_sha:
        raise RuntimeError("candidate checkpoint changed after candidate publication")
    config_sha = lowercase_sha256(
        candidate.get("checkpoint_config_sha256"), "candidate config SHA-256"
    )
    config = canonical_absolute_file(
        checkpoint.parent.parent / ".hydra/config.yaml",
        "candidate adjacent Hydra config",
    )
    actual_config_sha, _ = stable_sha256(config, "candidate adjacent Hydra config")
    if actual_config_sha != config_sha:
        raise RuntimeError("candidate adjacent Hydra config changed after publication")
    lowercase_sha256(
        candidate.get("config_sha256"), "candidate semantic config SHA-256"
    )
    step = positive_integer(candidate.get("global_step"), "candidate global step")
    staged_step = positive_integer(staged_step, "staged baseline step")
    candidate_minimum_step = positive_integer(
        candidate_minimum_step, "candidate minimum step"
    )
    checkpoint_step_multiple = positive_integer(
        checkpoint_step_multiple, "checkpoint step multiple"
    )
    staged_checkpoint = canonical_absolute_file(
        staged_checkpoint, "staged allocation baseline checkpoint"
    )
    staged_sha = lowercase_sha256(staged_checkpoint_sha256, "staged baseline SHA-256")
    actual_staged_sha, _ = stable_sha256(
        staged_checkpoint, "staged allocation baseline checkpoint"
    )
    if actual_staged_sha != staged_sha:
        raise RuntimeError("staged allocation baseline checkpoint SHA-256 mismatch")
    if step < staged_step or step < candidate_minimum_step:
        raise RuntimeError("candidate checkpoint violates the minimum step contract")
    if step % checkpoint_step_multiple != 0:
        raise RuntimeError("candidate checkpoint violates the periodic step contract")
    if step == staged_step and checkpoint_sha != staged_sha:
        raise RuntimeError("equal-step candidate differs from staged baseline")
    verify_allowed_checkpoint(checkpoint, allowed_checkpoint_globs)
    authorized_root = authorized_root_from_baseline(staged_checkpoint, task_id)
    expected_parent = (
        authorized_root
        / "staging"
        / task_id
        / "candidates"
        / f"step-{step}"
        / "checkpoints"
    )
    if checkpoint.parent != expected_parent:
        raise RuntimeError("candidate checkpoint has the wrong task/root identity")
    provenance = nonempty_string(candidate.get("verified_from"), "candidate provenance")
    match = CANDIDATE_PROVENANCE_RE.fullmatch(provenance)
    if match is None:
        raise RuntimeError("candidate writer provenance binding is not canonical")
    exact_provenance = {
        "staged_baseline": staged_sha,
        "staged_step": str(staged_step),
        "contract": lowercase_sha256(
            continuation_contract_sha256, "continuation contract SHA-256"
        ),
        "allocation": lowercase_sha256(
            allocation_record_sha256, "allocated-successor record SHA-256"
        ),
        "tool_manifest": lowercase_sha256(
            cutover_tool_manifest_sha256, "cutover tool manifest SHA-256"
        ),
    }
    for key, value in exact_provenance.items():
        if match.group(key) != value:
            raise RuntimeError(f"candidate provenance mismatch for {key}")

    lease_path = (
        authorized_root
        / "provenance/cutover-leases"
        / f"source-job_{source_job_id}.json"
    )
    lease, lease_sha, _ = signed_json(
        lease_path,
        expected_sha256=match.group("lease"),
        sidecar_absolute_path=True,
        label="cutover lease",
    )
    exact_keys(lease, CUTOVER_LEASE_KEYS, "cutover lease")
    expected_lease = {
        "schema_version": 1,
        "status": "SUCCESSOR_CLAIMED_FOR_SOURCE_CUTOVER",
        "task_id": task_id,
        "source_job_id": source_job_id,
        "successor_job_id": successor_job_id,
        "wandb_id": wandb_id,
        "continuation_contract_sha256": match.group("contract"),
        "allocation_record_sha256": match.group("allocation"),
        "cutover_tool_manifest_sha256": match.group("tool_manifest"),
    }
    for key, value in expected_lease.items():
        if lease.get(key) != value or type(lease.get(key)) is not type(value):
            raise RuntimeError(f"cutover lease mismatch for {key}")
    if lease_sha != match.group("lease"):
        raise RuntimeError("cutover lease SHA-256 mismatch")
    normalized = dict(candidate)
    normalized["checkpoint"] = str(checkpoint)
    normalized["record_sha256"] = lowercase_sha256(
        candidate_sha256, "candidate record SHA-256"
    )
    return normalized, match, authorized_root


def validate_candidate_tool_manifest_binding(
    *, authorized_root: Path, provenance: re.Match[str]
) -> dict[str, Any]:
    manifest_path = content_addressed_cutover_tool_manifest_path(
        authorized_root, provenance.group("tool_manifest")
    )
    tools, manifest_sha, _ = signed_json(
        manifest_path,
        expected_sha256=provenance.group("tool_manifest"),
        sidecar_absolute_path=True,
        label="cutover tool manifest",
    )
    tools = validate_cutover_tool_manifest(tools)
    if manifest_sha != provenance.group("tool_manifest"):
        raise RuntimeError("candidate cutover tool manifest SHA-256 mismatch")
    expected = {
        "record_writer_sha256": provenance.group("record_writer"),
        "source_terminal_collector_sha256": provenance.group("collector"),
    }
    for key, value in expected.items():
        if tools.get(key) != value:
            raise RuntimeError(f"candidate cutover tool manifest mismatch for {key}")
    return tools


def validate_ready_successor_record(
    record: dict[str, Any],
    *,
    record_path: Path,
    contract_path: Path,
    contract_sha256: str,
    contract: dict[str, Any],
    cutover_tools: dict[str, Any],
    task_id: str,
    source_job_id: int,
    successor_job_id: int,
    wandb_id: str,
    candidate: dict[str, Any],
    candidate_record_sha256: str,
    authorized_root: Path,
) -> None:
    exact_keys(record, ALLOCATION_RECORD_KEYS, "ready-for-source-cancel record")
    try:
        record_path.relative_to(authorized_root)
    except ValueError as exc:
        raise RuntimeError(
            "ready-for-source-cancel record is outside authorized root"
        ) from exc
    if not record_path.name.endswith(".ready-for-source-cancel.json"):
        raise RuntimeError("ready-for-source-cancel record filename is not canonical")
    expected = {
        "schema_version": 2,
        "status": "READY_FOR_SOURCE_CANCEL",
        "job_id": successor_job_id,
        "source_job_id": source_job_id,
        "sweep_task_id": task_id,
        "wandb_id": wandb_id,
        "candidate_record_sha256": candidate_record_sha256,
        "account": contract["destination_resource"]["account"],
        "partition": contract["destination_resource"]["partition"],
        "qos": contract["destination_resource"]["qos"],
    }
    for key, value in expected.items():
        if record.get(key) != value or type(record.get(key)) is not type(value):
            raise RuntimeError(f"ready-for-source-cancel record mismatch for {key}")
    nonnegative_integer(record.get("restart_count"), "ready successor restart count")
    for key in ("created_at", "node_list", "host"):
        nonempty_string(record.get(key), f"ready successor {key}")

    validation = record.get("checkpoint_validation")
    if not isinstance(validation, dict):
        raise RuntimeError("ready successor has no checkpoint validation")
    expected_validation = {
        "status": "passed",
        "checkpoint": candidate["checkpoint"],
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "global_step": candidate["global_step"],
        "wandb_id": wandb_id,
        "task_id": task_id,
        "checkpoint_config_role": "adjacent_origin_hydra",
        "checkpoint_config_sha256": candidate["checkpoint_config_sha256"],
        "config_sha256": candidate["config_sha256"],
        "embedded_config_identity_sha256": candidate["config_sha256"],
        "strict_load": True,
        "full_state_verified": True,
        "strict_optimizer_load": True,
        "strict_scheduler_load": True,
        "tensor_finiteness": "complete",
        "rebased_schedule_verified": True,
    }
    for key, value in expected_validation.items():
        if validation.get(key) != value or type(validation.get(key)) is not type(value):
            raise RuntimeError(f"ready candidate validation mismatch for {key}")

    destination = contract["destination_resource"]
    hardware = record.get("hardware_validation")
    if not isinstance(hardware, dict):
        raise RuntimeError("ready successor has no hardware validation")
    expected_hardware = {
        "schema_version": 1,
        "status": "passed",
        "expected_gpu_model": destination["gpu_model"],
        "expected_gpu_count": destination["gpus_per_node"],
    }
    for key, value in expected_hardware.items():
        if hardware.get(key) != value or type(hardware.get(key)) is not type(value):
            raise RuntimeError(f"ready hardware validation mismatch for {key}")
    rank_probes = hardware.get("rank_probes")
    if (
        not isinstance(rank_probes, list)
        or len(rank_probes) != destination["world_size"]
    ):
        raise RuntimeError("ready hardware validation has wrong rank count")

    tool_sha = record.get("tool_sha256")
    if not isinstance(tool_sha, dict) or not tool_sha:
        raise RuntimeError("ready successor has no tool identity map")
    for path, digest in tool_sha.items():
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise RuntimeError("ready successor tool path is not absolute")
        lowercase_sha256(digest, f"ready successor tool SHA-256 for {path}")
    if tool_sha.get(str(contract_path)) != contract_sha256:
        raise RuntimeError("ready successor used a different continuation contract")
    if (
        tool_sha.get(cutover_tools["resume_child_path"])
        != cutover_tools["resume_child_sha256"]
    ):
        raise RuntimeError("ready successor used a different resume child")
    launcher_sha = lowercase_sha256(
        contract.get("source_identity", {}).get("portable_launcher_sha256"),
        "contract portable launcher SHA-256",
    )
    if launcher_sha not in set(tool_sha.values()):
        raise RuntimeError("ready successor did not use the pinned portable launcher")


def ready_to_cancel_payload(
    *,
    cutover_tool_manifest: Path,
    cutover_tool_manifest_sha256: str,
    continuation_contract: Path,
    continuation_contract_sha256: str,
    allocation_record: Path,
    allocation_record_sha256: str,
    candidate_record: Path,
    candidate_record_sha256: str,
    ready_record: Path,
    ready_record_sha256: str,
    task_id: str,
    source_run_root: Path,
    source_job_id: int,
    successor_job_id: int,
    wandb_id: str,
    staged_checkpoint: Path,
    staged_checkpoint_sha256: str,
    staged_step: int,
    candidate_minimum_step: int,
    checkpoint_step_multiple: int,
    allowed_checkpoint_globs: Sequence[str],
) -> dict[str, Any]:
    task_id = nonempty_string(task_id, "task ID")
    source_job_id = positive_integer(source_job_id, "source job ID")
    successor_job_id = positive_integer(successor_job_id, "successor job ID")
    wandb_id = nonempty_string(wandb_id, "W&B ID")
    staged_checkpoint = canonical_absolute_file(
        staged_checkpoint, "staged allocation baseline checkpoint"
    )
    authorized_root = authorized_root_from_baseline(staged_checkpoint, task_id)
    cutover_tool_manifest = canonical_absolute_file(
        cutover_tool_manifest, "cutover tool manifest"
    )
    expected_tool_manifest = content_addressed_cutover_tool_manifest_path(
        authorized_root, cutover_tool_manifest_sha256
    )
    if cutover_tool_manifest != expected_tool_manifest:
        raise RuntimeError("cutover tool manifest path is not canonical")
    cutover_tools, cutover_tool_manifest_sha, _ = signed_json(
        cutover_tool_manifest,
        expected_sha256=cutover_tool_manifest_sha256,
        sidecar_absolute_path=True,
        label="cutover tool manifest",
    )
    cutover_tools = validate_cutover_tool_manifest(cutover_tools)
    continuation_contract = canonical_absolute_file(
        continuation_contract, "continuation contract"
    )
    contract, contract_sha = unsigned_exact_json(
        continuation_contract,
        continuation_contract_sha256,
        "continuation contract",
        require_canonical=False,
    )
    validate_continuation_contract(
        contract,
        task_id=task_id,
        source_run_root=source_run_root,
        source_job_id=source_job_id,
        wandb_id=wandb_id,
        staged_checkpoint=staged_checkpoint,
        staged_checkpoint_sha256=staged_checkpoint_sha256,
        staged_step=staged_step,
        candidate_minimum_step=candidate_minimum_step,
        checkpoint_step_multiple=checkpoint_step_multiple,
    )
    allocation_record = canonical_absolute_file(
        allocation_record, "allocated-successor record"
    )
    allocation, allocation_sha, _ = signed_json(
        allocation_record,
        expected_sha256=allocation_record_sha256,
        sidecar_absolute_path=True,
        label="allocated-successor record",
    )
    validate_allocation_record(
        allocation,
        contract_path=continuation_contract,
        contract_sha256=contract_sha,
        contract=contract,
        task_id=task_id,
        source_job_id=source_job_id,
        successor_job_id=successor_job_id,
        wandb_id=wandb_id,
        staged_checkpoint=staged_checkpoint,
        staged_checkpoint_sha256=staged_checkpoint_sha256,
        staged_step=staged_step,
    )
    candidate_record = canonical_absolute_file(candidate_record, "candidate record")
    candidate, candidate_sha, _ = signed_json(
        candidate_record,
        expected_sha256=candidate_record_sha256,
        sidecar_absolute_path=True,
        label="candidate record",
    )
    candidate, provenance, authorized_root = validate_candidate_transition_record(
        candidate,
        candidate_sha256=candidate_sha,
        task_id=task_id,
        source_job_id=source_job_id,
        successor_job_id=successor_job_id,
        wandb_id=wandb_id,
        staged_checkpoint=staged_checkpoint,
        staged_checkpoint_sha256=staged_checkpoint_sha256,
        staged_step=staged_step,
        candidate_minimum_step=candidate_minimum_step,
        checkpoint_step_multiple=checkpoint_step_multiple,
        allowed_checkpoint_globs=allowed_checkpoint_globs,
        continuation_contract_sha256=contract_sha,
        allocation_record_sha256=allocation_sha,
        cutover_tool_manifest_sha256=cutover_tool_manifest_sha,
    )
    ready_record = canonical_absolute_file(
        ready_record, "ready-for-source-cancel record"
    )
    ready, ready_sha, _ = signed_json(
        ready_record,
        expected_sha256=ready_record_sha256,
        sidecar_absolute_path=True,
        label="ready-for-source-cancel record",
    )
    validate_ready_successor_record(
        ready,
        record_path=ready_record,
        contract_path=continuation_contract,
        contract_sha256=contract_sha,
        contract=contract,
        cutover_tools=cutover_tools,
        task_id=task_id,
        source_job_id=source_job_id,
        successor_job_id=successor_job_id,
        wandb_id=wandb_id,
        candidate=candidate,
        candidate_record_sha256=candidate_sha,
        authorized_root=authorized_root,
    )
    payload = {
        "schema_version": 1,
        "status": "READY_TO_CANCEL_SOURCE",
        "task_id": task_id,
        "source_job_id": source_job_id,
        "successor_job_id": successor_job_id,
        "wandb_id": wandb_id,
        "candidate_record": str(candidate_record),
        "candidate_record_sha256": candidate_sha,
        "ready_record_sha256": ready_sha,
        "checkpoint": candidate["checkpoint"],
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "checkpoint_config_role": "adjacent_origin_hydra",
        "checkpoint_config_sha256": candidate["checkpoint_config_sha256"],
        "config_sha256": candidate["config_sha256"],
        "global_step": candidate["global_step"],
        "cutover_lease_sha256": provenance.group("lease"),
        "continuation_contract_sha256": contract_sha,
        "allocation_record_sha256": allocation_sha,
        "cutover_tool_manifest_sha256": cutover_tool_manifest_sha,
    }
    if set(payload) != READY_TO_CANCEL_KEYS:
        raise AssertionError("internal READY_TO_CANCEL_SOURCE schema drift")
    return payload


def normalize_sacct_state(value: str) -> str:
    match = re.fullmatch(
        r"(CANCELLED|COMPLETED|PREEMPTED|TIMEOUT|NODE_FAIL)(?:\+| by [0-9]+)?",
        value,
    )
    if match is None:
        raise RuntimeError(f"source state is not terminal: {value}")
    return match.group(1)


def remote_identity_command(argv: Sequence[str]) -> str:
    if not argv or any(not isinstance(item, str) or not item for item in argv):
        raise RuntimeError("remote query argv is invalid")
    return REMOTE_IDENTITY_PROLOGUE + shlex.join(argv)


def split_remote_identity_stdout(
    stdout: str, *, expected_hostname: str, expected_username: str, label: str
) -> str:
    if not isinstance(stdout, str):
        raise RuntimeError(f"{label} stdout is not text")
    header, separator, query_stdout = stdout.partition("\n")
    expected_header = (
        f"{REMOTE_IDENTITY_MARKER}|{expected_hostname}|{expected_username}"
    )
    if separator != "\n" or header != expected_header:
        raise RuntimeError(f"{label} same-session remote identity is wrong")
    return query_stdout


def validate_command_record(
    value: Any,
    *,
    label: str,
    expected_argv_tail: list[str],
    expected_executable: str,
    expected_ssh_wrapper: str,
    expected_remote: str,
    expected_remote_hostname: str,
    expected_remote_username: str,
) -> dict[str, Any]:
    record = exact_keys(value, COMMAND_EVIDENCE_KEYS, f"{label} command evidence")
    argv = record["argv"]
    if not isinstance(argv, list) or any(
        not isinstance(item, str) or not item for item in argv
    ):
        raise RuntimeError(f"{label} argv must be a nonempty string array")
    executable = argv[0] if argv else ""
    executable_path = PurePosixPath(executable)
    if (
        not executable.startswith("/")
        or executable_path.name != expected_executable
        or str(executable_path) != executable
        or argv[1:] != expected_argv_tail
    ):
        raise RuntimeError(f"{label} argv is not the exact allowlisted shape")
    expected_transport = [
        expected_ssh_wrapper,
        expected_remote,
        remote_identity_command(argv),
    ]
    if record["transport_argv"] != expected_transport:
        raise RuntimeError(f"{label} transport argv is not the exact pinned shape")
    if record["returncode"] != 0 or type(record["returncode"]) is not int:
        raise RuntimeError(f"{label} command did not exit zero")
    stdout = record["stdout"]
    stderr = record["stderr"]
    if not isinstance(stdout, str) or not isinstance(stderr, str):
        raise RuntimeError(f"{label} stdout/stderr must be strings")
    stdout_sha = lowercase_sha256(record["stdout_sha256"], f"{label} stdout SHA-256")
    stderr_sha = lowercase_sha256(record["stderr_sha256"], f"{label} stderr SHA-256")
    if hashlib.sha256(stdout.encode("utf-8")).hexdigest() != stdout_sha:
        raise RuntimeError(f"{label} stdout SHA-256 mismatch")
    if hashlib.sha256(stderr.encode("utf-8")).hexdigest() != stderr_sha:
        raise RuntimeError(f"{label} stderr SHA-256 mismatch")
    if stderr != "":
        raise RuntimeError(f"{label} command emitted stderr")
    started = positive_integer(
        record["started_at_unix_ns"], f"{label} command start time"
    )
    completed = positive_integer(
        record["completed_at_unix_ns"], f"{label} command completion time"
    )
    if completed < started or completed - started > MAX_COMMAND_DURATION_NS:
        raise RuntimeError(f"{label} command timing is invalid")
    query_stdout = split_remote_identity_stdout(
        stdout,
        expected_hostname=expected_remote_hostname,
        expected_username=expected_remote_username,
        label=label,
    )
    return {**record, "query_stdout": query_stdout}


def validate_collector_evidence(
    value: Any,
    *,
    expected_remote: str,
    expected_access_selection_sha256: str,
    expected_ssh_wrapper_sha256: str,
    expected_record_writer_sha256: str,
    expected_collector_sha256: str,
) -> dict[str, Any]:
    collector = exact_keys(value, COLLECTOR_EVIDENCE_KEYS, "source evidence collector")
    if collector.get("execution_cluster") != "PACE ICE":
        raise RuntimeError("source evidence collector was not run on PACE ICE")
    job_id = collector.get("slurm_job_id")
    if not isinstance(job_id, str) or re.fullmatch(r"[1-9][0-9]*", job_id) is None:
        raise RuntimeError("source evidence collector Slurm job ID is invalid")
    node = nonempty_string(collector.get("slurm_node"), "source collector Slurm node")
    if SAFE_NODE_RE.fullmatch(node) is None:
        raise RuntimeError("source evidence collector Slurm node is invalid")

    collector_path = canonical_absolute_file(
        Path(nonempty_string(collector.get("path"), "source collector path")),
        "source terminal collector",
    )
    collector_sha = lowercase_sha256(
        collector.get("sha256"), "source collector SHA-256"
    )
    if collector_sha != expected_collector_sha256:
        raise RuntimeError(
            "source evidence collector differs from candidate provenance"
        )
    actual_collector_sha, _ = stable_sha256(collector_path, "source terminal collector")
    if actual_collector_sha != collector_sha:
        raise RuntimeError("source terminal collector SHA-256 mismatch")

    record_writer_path = canonical_absolute_file(
        Path(
            nonempty_string(collector.get("record_writer"), "collector record writer")
        ),
        "collector record writer",
    )
    record_writer_sha = lowercase_sha256(
        collector.get("record_writer_sha256"), "collector record writer SHA-256"
    )
    if record_writer_sha != expected_record_writer_sha256:
        raise RuntimeError("collector record writer differs from candidate provenance")
    actual_record_writer_sha, _ = stable_sha256(
        record_writer_path, "collector record writer"
    )
    if actual_record_writer_sha != record_writer_sha:
        raise RuntimeError("collector record writer SHA-256 mismatch")

    wrapper_path = canonical_absolute_file(
        Path(
            nonempty_string(
                collector.get("ssh_wrapper"), "source collector SSH wrapper"
            )
        ),
        "source collector SSH wrapper",
    )
    wrapper_sha = lowercase_sha256(
        collector.get("ssh_wrapper_sha256"), "source collector SSH wrapper SHA-256"
    )
    if wrapper_sha != expected_ssh_wrapper_sha256:
        raise RuntimeError(
            "source evidence SSH wrapper differs from candidate provenance"
        )
    actual_wrapper_sha, _ = stable_sha256(wrapper_path, "source collector SSH wrapper")
    if actual_wrapper_sha != wrapper_sha:
        raise RuntimeError("source collector SSH wrapper SHA-256 mismatch")

    selection_path = canonical_absolute_file(
        Path(
            nonempty_string(
                collector.get("access_selection"), "source collector access selection"
            )
        ),
        "source collector access selection",
    )
    selection_sha = lowercase_sha256(
        collector.get("access_selection_sha256"),
        "source collector access selection SHA-256",
    )
    if selection_sha != expected_access_selection_sha256:
        raise RuntimeError("source access selection differs from candidate provenance")
    selection, actual_selection_sha = unsigned_exact_json(
        selection_path,
        selection_sha,
        "source collector access selection",
    )
    if actual_selection_sha != selection_sha:
        raise RuntimeError("source collector access selection SHA-256 mismatch")
    selection = validate_access_selection(selection)
    if (
        collector.get("selected_remote") != expected_remote
        or selection.get("selected_remote") != expected_remote
    ):
        raise RuntimeError("source collector selected remote differs from candidate")
    successful_probe = (
        selection["canonical_probe"]
        if expected_remote == selection["canonical_remote"]
        else selection["fallback_probe"]
    )
    if not isinstance(successful_probe, dict):
        raise RuntimeError("source collector has no successful selected probe")
    if collector.get("remote_hostname") != successful_probe.get(
        "hostname"
    ) or collector.get("remote_username") != successful_probe.get("username"):
        raise RuntimeError("source collector remote identity differs from access probe")
    return {
        **collector,
        "path": str(collector_path),
        "record_writer": str(record_writer_path),
        "ssh_wrapper": str(wrapper_path),
        "access_selection": str(selection_path),
    }


def validate_source_evidence(
    evidence: dict[str, Any],
    *,
    evidence_stat: os.stat_result,
    source_job_id: int,
    successor_job_id: int,
    wandb_id: str,
    candidate_record_sha256: str,
    allocation_record_sha256: str,
    ready_to_cancel_record_sha256: str,
    candidate_provenance: re.Match[str],
    max_age_seconds: int,
    now_unix_ns: int,
    enforce_file_freshness: bool = True,
) -> dict[str, str]:
    exact_keys(evidence, SOURCE_EVIDENCE_KEYS, "source terminal evidence")
    source_job_id = positive_integer(source_job_id, "source job ID")
    successor_job_id = positive_integer(successor_job_id, "successor job ID")
    wandb_id = nonempty_string(wandb_id, "W&B ID")
    candidate_record_sha256 = lowercase_sha256(
        candidate_record_sha256, "candidate record SHA-256"
    )
    allocation_record_sha256 = lowercase_sha256(
        allocation_record_sha256, "allocation record SHA-256"
    )
    ready_to_cancel_record_sha256 = lowercase_sha256(
        ready_to_cancel_record_sha256, "ready-to-cancel record SHA-256"
    )
    max_age_seconds = positive_integer(max_age_seconds, "maximum evidence age")
    if max_age_seconds > MAX_ALLOWED_EVIDENCE_AGE_SECONDS:
        raise RuntimeError(
            f"maximum evidence age may not exceed {MAX_ALLOWED_EVIDENCE_AGE_SECONDS} seconds"
        )
    now_unix_ns = positive_integer(now_unix_ns, "current Unix time")
    expected = {
        "schema_version": 1,
        "status": "SOURCE_TERMINAL_AND_ABSENT",
        "source_cluster": "skynet",
        "source_job_id": source_job_id,
        "successor_job_id": successor_job_id,
        "wandb_id": wandb_id,
        "candidate_record_sha256": candidate_record_sha256,
        "allocation_record_sha256": allocation_record_sha256,
        "ready_to_cancel_record_sha256": ready_to_cancel_record_sha256,
    }
    for key, value in expected.items():
        if evidence.get(key) != value or type(evidence.get(key)) is not type(value):
            raise RuntimeError(f"source terminal evidence mismatch for {key}")
    observed = positive_integer(
        evidence["observed_at_unix_ns"], "source evidence observation time"
    )
    maximum_age_ns = max_age_seconds * 1_000_000_000
    for timestamp, label in (
        (observed, "source evidence observation time"),
        (evidence_stat.st_mtime_ns, "source evidence file mtime"),
    ):
        if timestamp > now_unix_ns:
            raise RuntimeError(f"{label} is in the future")
        if enforce_file_freshness and now_unix_ns - timestamp > maximum_age_ns:
            raise RuntimeError(f"{label} is stale")
    if evidence_stat.st_mtime_ns + MAX_MTIME_OBSERVATION_SKEW_NS < observed:
        raise RuntimeError("source evidence file predates its observation")

    collector = validate_collector_evidence(
        evidence["collector"],
        expected_remote=candidate_provenance.group("selected_remote"),
        expected_access_selection_sha256=candidate_provenance.group("access_selection"),
        expected_ssh_wrapper_sha256=candidate_provenance.group("ssh_wrapper"),
        expected_record_writer_sha256=candidate_provenance.group("record_writer"),
        expected_collector_sha256=candidate_provenance.group("collector"),
    )

    job = str(source_job_id)
    commands = exact_keys(
        evidence["command_evidence"], {"sacct", "squeue"}, "source command evidence"
    )
    sacct = validate_command_record(
        commands["sacct"],
        label="sacct",
        expected_executable="sacct",
        expected_argv_tail=[
            "--noheader",
            "--parsable2",
            "--allocations",
            f"--jobs={job}",
            "--format=JobIDRaw,State,ExitCode,End",
        ],
        expected_ssh_wrapper=collector["ssh_wrapper"],
        expected_remote=collector["selected_remote"],
        expected_remote_hostname=collector["remote_hostname"],
        expected_remote_username=collector["remote_username"],
    )
    squeue = validate_command_record(
        commands["squeue"],
        label="squeue",
        expected_executable="squeue",
        expected_argv_tail=[
            "--noheader",
            "--array",
            f"--user={collector['remote_username']}",
            "--format=%i|%T",
        ],
        expected_ssh_wrapper=collector["ssh_wrapper"],
        expected_remote=collector["selected_remote"],
        expected_remote_hostname=collector["remote_hostname"],
        expected_remote_username=collector["remote_username"],
    )
    sacct_started = sacct["started_at_unix_ns"]
    sacct_completed = sacct["completed_at_unix_ns"]
    squeue_started = squeue["started_at_unix_ns"]
    squeue_completed = squeue["completed_at_unix_ns"]
    if squeue_started < sacct_completed:
        raise RuntimeError("source evidence commands are not ordered sacct then squeue")
    if squeue_started - sacct_completed > MAX_INTER_COMMAND_GAP_NS:
        raise RuntimeError("source evidence command gap is too large")
    if (
        observed < squeue_completed
        or observed - squeue_completed > MAX_INTER_COMMAND_GAP_NS
    ):
        raise RuntimeError("source evidence observation is not adjacent to commands")
    if enforce_file_freshness and now_unix_ns - sacct_started > maximum_age_ns:
        raise RuntimeError("source evidence command start is stale")
    if any(
        timestamp > now_unix_ns
        for timestamp in (
            sacct_started,
            sacct_completed,
            squeue_started,
            squeue_completed,
        )
    ):
        raise RuntimeError("source evidence command timing is in the future")
    sacct_lines = sacct["query_stdout"].splitlines()
    if len(sacct_lines) != 1:
        raise RuntimeError("sacct evidence must contain exactly one allocation row")
    fields = sacct_lines[0].split("|")
    if len(fields) != 4 or fields[0] != job:
        raise RuntimeError("sacct evidence does not name the exact source allocation")
    source_state = normalize_sacct_state(fields[1])
    source_exit_code = fields[2]
    source_end_time = fields[3]
    if EXIT_CODE_RE.fullmatch(source_exit_code) is None:
        raise RuntimeError("sacct source exit code is invalid")
    if END_TIME_RE.fullmatch(source_end_time) is None:
        raise RuntimeError("sacct source end time is invalid")

    terminal = exact_keys(
        evidence["terminal_state"], TERMINAL_EVIDENCE_KEYS, "terminal state evidence"
    )
    if terminal != {
        "state": source_state,
        "exit_code": source_exit_code,
        "end_time": source_end_time,
    }:
        raise RuntimeError("parsed sacct row differs from terminal state evidence")

    queue = exact_keys(evidence["queue"], QUEUE_EVIDENCE_KEYS, "queue evidence")
    for line in squeue["query_stdout"].splitlines():
        fields = [field.strip(" ") for field in line.split("|")]
        if (
            len(fields) != 2
            or SQUEUE_JOB_ID_RE.fullmatch(fields[0]) is None
            or SQUEUE_STATE_RE.fullmatch(fields[1]) is None
        ):
            raise RuntimeError("squeue active snapshot contains a malformed row")
        if fields[0] == job:
            raise RuntimeError("squeue evidence shows the source job is still active")
    if queue != {"job_id": source_job_id, "present": False, "state": None}:
        raise RuntimeError("queue evidence is not exact source-job absence")
    return {
        "source_state": source_state,
        "source_exit_code": source_exit_code,
        "source_end_time": source_end_time,
    }


def authorization_payload(
    *,
    candidate_record: Path,
    candidate_record_sha256: str,
    ready_to_cancel_record: Path,
    ready_to_cancel_record_sha256: str,
    source_evidence: Path,
    source_evidence_sha256: str,
    source_job_id: int,
    successor_job_id: int,
    wandb_id: str,
    max_evidence_age_seconds: int,
    now_unix_ns: int | None = None,
) -> dict[str, Any]:
    source_job_id = positive_integer(source_job_id, "source job ID")
    successor_job_id = positive_integer(successor_job_id, "successor job ID")
    wandb_id = nonempty_string(wandb_id, "W&B ID")
    candidate, candidate_sha, _ = signed_json(
        candidate_record,
        expected_sha256=candidate_record_sha256,
        sidecar_absolute_path=True,
        label="candidate record",
    )
    keys = set(candidate)
    if not CANDIDATE_REQUIRED_KEYS.issubset(keys) or not keys.issubset(
        CANDIDATE_REQUIRED_KEYS | CANDIDATE_OPTIONAL_KEYS
    ):
        raise RuntimeError("candidate record keys do not match resume-child schema")
    for key in CANDIDATE_OPTIONAL_KEYS.intersection(candidate):
        nonempty_string(candidate[key], f"candidate optional field {key}")
    expected_candidate = {
        "schema_version": 1,
        "status": "CANDIDATE",
        "source_job_id": source_job_id,
        "successor_job_id": successor_job_id,
        "wandb_id": wandb_id,
        "checkpoint_config_role": "adjacent_origin_hydra",
    }
    for key, expected in expected_candidate.items():
        if candidate.get(key) != expected or type(candidate.get(key)) is not type(
            expected
        ):
            raise RuntimeError(f"candidate record mismatch for {key}")
    checkpoint = canonical_absolute_file(
        Path(nonempty_string(candidate["checkpoint"], "candidate checkpoint")),
        "candidate checkpoint",
    )
    checkpoint_sha = lowercase_sha256(
        candidate["checkpoint_sha256"], "candidate checkpoint SHA-256"
    )
    actual_sha, _ = stable_sha256(checkpoint, "candidate checkpoint")
    if actual_sha != checkpoint_sha:
        raise RuntimeError("candidate checkpoint changed before authorization")
    candidate_config_sha = lowercase_sha256(
        candidate["checkpoint_config_sha256"], "candidate config SHA-256"
    )
    adjacent_config = canonical_absolute_file(
        checkpoint.parent.parent / ".hydra/config.yaml",
        "candidate adjacent Hydra config",
    )
    actual_config_sha, _ = stable_sha256(
        adjacent_config, "candidate adjacent Hydra config"
    )
    if actual_config_sha != candidate_config_sha:
        raise RuntimeError(
            "candidate adjacent Hydra config changed before authorization"
        )
    lowercase_sha256(candidate["config_sha256"], "candidate semantic config SHA-256")
    positive_integer(candidate["global_step"], "candidate global step")
    provenance = candidate.get("verified_from")
    if not isinstance(provenance, str):
        raise RuntimeError("candidate record has no writer provenance binding")
    match = CANDIDATE_PROVENANCE_RE.fullmatch(provenance)
    if match is None:
        raise RuntimeError("candidate writer provenance binding is not canonical")
    allocation_record_sha = match.group("allocation")

    ready_to_cancel_record = canonical_absolute_file(
        ready_to_cancel_record, "ready-to-cancel record"
    )
    ready_to_cancel, ready_to_cancel_sha, _ = signed_json(
        ready_to_cancel_record,
        expected_sha256=ready_to_cancel_record_sha256,
        sidecar_absolute_path=True,
        label="ready-to-cancel record",
    )
    exact_keys(ready_to_cancel, READY_TO_CANCEL_KEYS, "ready-to-cancel record")
    expected_ready = {
        "schema_version": 1,
        "status": "READY_TO_CANCEL_SOURCE",
        "source_job_id": source_job_id,
        "successor_job_id": successor_job_id,
        "wandb_id": wandb_id,
        "candidate_record": str(candidate_record),
        "candidate_record_sha256": candidate_sha,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_config_role": "adjacent_origin_hydra",
        "checkpoint_config_sha256": candidate_config_sha,
        "config_sha256": candidate["config_sha256"],
        "global_step": candidate["global_step"],
        "cutover_lease_sha256": match.group("lease"),
        "continuation_contract_sha256": match.group("contract"),
        "allocation_record_sha256": allocation_record_sha,
        "cutover_tool_manifest_sha256": match.group("tool_manifest"),
    }
    for key, value in expected_ready.items():
        if ready_to_cancel.get(key) != value or type(
            ready_to_cancel.get(key)
        ) is not type(value):
            raise RuntimeError(f"ready-to-cancel record mismatch for {key}")
    candidate_task_dir = checkpoint.parent.parent.parent.parent
    if (
        checkpoint.parent.name != "checkpoints"
        or checkpoint.parent.parent.parent.name != "candidates"
        or not candidate_task_dir.name
    ):
        raise RuntimeError("candidate checkpoint has no canonical task identity")
    if ready_to_cancel.get("task_id") != candidate_task_dir.name:
        raise RuntimeError("ready-to-cancel task ID differs from candidate")
    validate_candidate_tool_manifest_binding(
        authorized_root=candidate_task_dir.parent.parent,
        provenance=match,
    )
    lowercase_sha256(
        ready_to_cancel.get("ready_record_sha256"),
        "ready-for-source-cancel record SHA-256",
    )

    evidence, evidence_sha, evidence_stat = signed_json(
        source_evidence,
        expected_sha256=source_evidence_sha256,
        sidecar_absolute_path=True,
        label="source terminal evidence",
    )
    terminal = validate_source_evidence(
        evidence,
        evidence_stat=evidence_stat,
        source_job_id=source_job_id,
        successor_job_id=successor_job_id,
        wandb_id=wandb_id,
        candidate_record_sha256=candidate_sha,
        allocation_record_sha256=allocation_record_sha,
        ready_to_cancel_record_sha256=ready_to_cancel_sha,
        candidate_provenance=match,
        max_age_seconds=max_evidence_age_seconds,
        now_unix_ns=time.time_ns() if now_unix_ns is None else now_unix_ns,
    )
    return {
        **{key: value for key, value in candidate.items() if key != "verified_from"},
        "status": "AUTHORIZED",
        "source_state": terminal["source_state"],
        "active_queue_state": "absent",
        "candidate_record": str(candidate_record),
        "candidate_record_sha256": candidate_sha,
        "ready_to_cancel_record": str(ready_to_cancel_record),
        "ready_to_cancel_record_sha256": ready_to_cancel_sha,
        "source_terminal_evidence": str(source_evidence),
        "source_terminal_evidence_sha256": evidence_sha,
        "source_exit_code": terminal["source_exit_code"],
        "source_end_time": terminal["source_end_time"],
        "verified_from": (
            f"candidate_record_sha256={candidate_sha};"
            f"ready_to_cancel_record_sha256={ready_to_cancel_sha};"
            f"source_terminal_evidence_sha256={evidence_sha}"
        ),
    }


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_signed_no_overwrite(output: Path, payload: dict[str, Any]) -> str:
    if not output.is_absolute():
        raise RuntimeError("output record path must be absolute")
    if output.is_symlink():
        raise RuntimeError(f"output record path is a symlink: {output}")
    parent = output.parent.resolve(strict=True)
    if output.parent != parent:
        raise RuntimeError("output record path is not canonical")
    sidecar = Path(str(output) + ".sha256")
    reservation = output.with_name(f".{output.name}.publish-reservation")
    raw = canonical_json(payload)
    digest = hashlib.sha256(raw).hexdigest()
    sidecar_raw = f"{digest}  {output}\n".encode("utf-8")
    reservation_fd = os.open(
        reservation,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        if not stat.S_ISREG(os.fstat(reservation_fd).st_mode):
            raise RuntimeError("output publication reservation is not a regular file")
        fcntl.flock(reservation_fd, fcntl.LOCK_EX)

        def exists_as_exact_immutable(path: Path, expected: bytes, label: str) -> bool:
            if path.is_symlink():
                raise RuntimeError(f"{label} is a symlink: {path}")
            if not path.exists():
                return False
            actual, metadata = stable_bytes(path, label)
            if actual != expected:
                raise FileExistsError(
                    f"{label} already exists with different bytes: {path}"
                )
            if metadata.st_mode & 0o222:
                raise RuntimeError(f"{label} exact bytes are not immutable: {path}")
            return True

        def create_missing(path: Path, encoded: bytes, kind: str) -> None:
            descriptor, temporary_text = tempfile.mkstemp(
                prefix=f".{output.name}.{kind}.", dir=parent
            )
            temporary = Path(temporary_text)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.chmod(temporary, 0o444)
                os.link(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
            fsync_directory(parent)

        # The sidecar is the commit marker.  If a process dies after linking
        # the record but before linking the sidecar, an exact-byte retry safely
        # completes the pair without replacing either inode.
        record_exists = exists_as_exact_immutable(output, raw, "output record")
        sidecar_exists = exists_as_exact_immutable(
            sidecar, sidecar_raw, "output SHA sidecar"
        )
        if not record_exists:
            create_missing(output, raw, "record")
        if not sidecar_exists:
            create_missing(sidecar, sidecar_raw, "sidecar")
        if stable_bytes(output, "published record")[0] != raw:
            raise RuntimeError("published record verification failed")
        if stable_bytes(sidecar, "published record sidecar")[0] != sidecar_raw:
            raise RuntimeError("published sidecar verification failed")
    finally:
        try:
            fcntl.flock(reservation_fd, fcntl.LOCK_UN)
        finally:
            os.close(reservation_fd)
        # Keep the regular lock inode permanently.  Unlinking it would allow a
        # third publisher to lock a new inode while an older waiter still owns
        # the unlinked one.
        fsync_directory(parent)
    return digest


def positive_cli_integer(value: str) -> int:
    if re.fullmatch(r"[1-9][0-9]*", value) is None:
        raise argparse.ArgumentTypeError("must be a positive base-10 integer")
    return int(value)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    tool_manifest = commands.add_parser(
        "tool-manifest", help="write the signed cutover-tool companion manifest"
    )
    tool_manifest.add_argument("--output", type=Path, required=True)
    tool_manifest.add_argument("--resume-child", type=Path, required=True)
    tool_manifest.add_argument("--source-terminal-collector", type=Path, required=True)

    candidate = commands.add_parser("candidate", help="write a signed CANDIDATE")
    candidate.add_argument("--output", type=Path, required=True)
    candidate.add_argument("--cutover-tool-manifest", type=Path, required=True)
    candidate.add_argument("--cutover-tool-manifest-sha256", required=True)
    candidate.add_argument("--continuation-contract", type=Path, required=True)
    candidate.add_argument("--continuation-contract-sha256", required=True)
    candidate.add_argument("--allocation-record", type=Path, required=True)
    candidate.add_argument("--allocation-record-sha256", required=True)
    candidate.add_argument("--source-validation-result", type=Path, required=True)
    candidate.add_argument("--source-validation-result-sha256", required=True)
    candidate.add_argument("--validator-tool-sha256", required=True)
    candidate.add_argument("--validator-stdout", type=Path, required=True)
    candidate.add_argument("--validator-stdout-sha256", required=True)
    candidate.add_argument("--validator-stderr", type=Path, required=True)
    candidate.add_argument("--validator-stderr-sha256", required=True)
    candidate.add_argument("--stage-proof", type=Path, required=True)
    candidate.add_argument("--stage-proof-sha256", required=True)
    candidate.add_argument("--task-id", required=True)
    candidate.add_argument("--source-run-root", type=Path, required=True)
    candidate.add_argument("--source-job-id", type=positive_cli_integer, required=True)
    candidate.add_argument(
        "--successor-job-id", type=positive_cli_integer, required=True
    )
    candidate.add_argument("--wandb-id", required=True)
    candidate.add_argument("--candidate-checkpoint", type=Path, required=True)
    candidate.add_argument("--staged-checkpoint", type=Path, required=True)
    candidate.add_argument("--staged-checkpoint-sha256", required=True)
    candidate.add_argument("--staged-step", type=positive_cli_integer, required=True)
    candidate.add_argument(
        "--candidate-minimum-step", type=positive_cli_integer, required=True
    )
    candidate.add_argument(
        "--checkpoint-step-multiple", type=positive_cli_integer, required=True
    )
    candidate.add_argument("--allowed-checkpoint-glob", action="append", required=True)

    ready = commands.add_parser(
        "ready-to-cancel",
        help="write a signed READY_TO_CANCEL_SOURCE after successor hardware/full-state smoke",
    )
    ready.add_argument("--output", type=Path, required=True)
    ready.add_argument("--cutover-tool-manifest", type=Path, required=True)
    ready.add_argument("--cutover-tool-manifest-sha256", required=True)
    ready.add_argument("--continuation-contract", type=Path, required=True)
    ready.add_argument("--continuation-contract-sha256", required=True)
    ready.add_argument("--allocation-record", type=Path, required=True)
    ready.add_argument("--allocation-record-sha256", required=True)
    ready.add_argument("--candidate-record", type=Path, required=True)
    ready.add_argument("--candidate-record-sha256", required=True)
    ready.add_argument("--ready-record", type=Path, required=True)
    ready.add_argument("--ready-record-sha256", required=True)
    ready.add_argument("--task-id", required=True)
    ready.add_argument("--source-run-root", type=Path, required=True)
    ready.add_argument("--source-job-id", type=positive_cli_integer, required=True)
    ready.add_argument("--successor-job-id", type=positive_cli_integer, required=True)
    ready.add_argument("--wandb-id", required=True)
    ready.add_argument("--staged-checkpoint", type=Path, required=True)
    ready.add_argument("--staged-checkpoint-sha256", required=True)
    ready.add_argument("--staged-step", type=positive_cli_integer, required=True)
    ready.add_argument(
        "--candidate-minimum-step", type=positive_cli_integer, required=True
    )
    ready.add_argument(
        "--checkpoint-step-multiple", type=positive_cli_integer, required=True
    )
    ready.add_argument("--allowed-checkpoint-glob", action="append", required=True)

    authorization = commands.add_parser(
        "authorize",
        help="write a signed AUTHORIZED record",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The signed source-evidence JSON must contain the exact candidate-record SHA,
allocation-record SHA, successor job, W&B ID, and source job.  command_evidence
must retain raw UTF-8 stdout/stderr and their lowercase SHA-256 values.  Allowed
commands (JOB is the exact source allocation) are:

  sacct --noheader --parsable2 --allocations --jobs=JOB \\
    --format=JobIDRaw,State,ExitCode,End
  squeue --noheader --array --user=REMOTE_USER --format=%i|%T

The sacct query payload must be exactly one allocation row.  Every row in the
squeue active-user snapshot is parsed and the exact source allocation must be
absent. observed_at_unix_ns and the evidence file mtime must both be non-future
and no older than --max-evidence-age-seconds (hard maximum: 600 seconds).
""",
    )
    authorization.add_argument("--output", type=Path, required=True)
    authorization.add_argument("--candidate-record", type=Path, required=True)
    authorization.add_argument("--candidate-record-sha256", required=True)
    authorization.add_argument("--ready-to-cancel-record", type=Path, required=True)
    authorization.add_argument("--ready-to-cancel-record-sha256", required=True)
    authorization.add_argument("--source-evidence", type=Path, required=True)
    authorization.add_argument("--source-evidence-sha256", required=True)
    authorization.add_argument(
        "--source-job-id", type=positive_cli_integer, required=True
    )
    authorization.add_argument(
        "--successor-job-id", type=positive_cli_integer, required=True
    )
    authorization.add_argument("--wandb-id", required=True)
    authorization.add_argument(
        "--max-evidence-age-seconds",
        type=positive_cli_integer,
        default=120,
    )
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "tool-manifest":
        payload = cutover_tool_manifest_payload(
            resume_child=args.resume_child,
            source_terminal_collector=args.source_terminal_collector,
        )
    elif args.command == "candidate":
        payload = candidate_payload(
            cutover_tool_manifest=args.cutover_tool_manifest,
            cutover_tool_manifest_sha256=args.cutover_tool_manifest_sha256,
            continuation_contract=args.continuation_contract,
            continuation_contract_sha256=args.continuation_contract_sha256,
            allocation_record=args.allocation_record,
            allocation_record_sha256=args.allocation_record_sha256,
            source_validation_result=args.source_validation_result,
            source_validation_result_sha256=args.source_validation_result_sha256,
            validator_tool_sha256=args.validator_tool_sha256,
            validator_stdout=args.validator_stdout,
            validator_stdout_sha256=args.validator_stdout_sha256,
            validator_stderr=args.validator_stderr,
            validator_stderr_sha256=args.validator_stderr_sha256,
            stage_proof=args.stage_proof,
            stage_proof_sha256=args.stage_proof_sha256,
            task_id=args.task_id,
            source_run_root=args.source_run_root,
            source_job_id=args.source_job_id,
            successor_job_id=args.successor_job_id,
            wandb_id=args.wandb_id,
            candidate_checkpoint=args.candidate_checkpoint,
            staged_checkpoint=args.staged_checkpoint,
            staged_checkpoint_sha256=args.staged_checkpoint_sha256,
            staged_step=args.staged_step,
            candidate_minimum_step=args.candidate_minimum_step,
            checkpoint_step_multiple=args.checkpoint_step_multiple,
            allowed_checkpoint_globs=args.allowed_checkpoint_glob,
        )
    elif args.command == "ready-to-cancel":
        payload = ready_to_cancel_payload(
            cutover_tool_manifest=args.cutover_tool_manifest,
            cutover_tool_manifest_sha256=args.cutover_tool_manifest_sha256,
            continuation_contract=args.continuation_contract,
            continuation_contract_sha256=args.continuation_contract_sha256,
            allocation_record=args.allocation_record,
            allocation_record_sha256=args.allocation_record_sha256,
            candidate_record=args.candidate_record,
            candidate_record_sha256=args.candidate_record_sha256,
            ready_record=args.ready_record,
            ready_record_sha256=args.ready_record_sha256,
            task_id=args.task_id,
            source_run_root=args.source_run_root,
            source_job_id=args.source_job_id,
            successor_job_id=args.successor_job_id,
            wandb_id=args.wandb_id,
            staged_checkpoint=args.staged_checkpoint,
            staged_checkpoint_sha256=args.staged_checkpoint_sha256,
            staged_step=args.staged_step,
            candidate_minimum_step=args.candidate_minimum_step,
            checkpoint_step_multiple=args.checkpoint_step_multiple,
            allowed_checkpoint_globs=args.allowed_checkpoint_glob,
        )
    elif args.command == "authorize":
        payload = authorization_payload(
            candidate_record=args.candidate_record,
            candidate_record_sha256=args.candidate_record_sha256,
            ready_to_cancel_record=args.ready_to_cancel_record,
            ready_to_cancel_record_sha256=args.ready_to_cancel_record_sha256,
            source_evidence=args.source_evidence,
            source_evidence_sha256=args.source_evidence_sha256,
            source_job_id=args.source_job_id,
            successor_job_id=args.successor_job_id,
            wandb_id=args.wandb_id,
            max_evidence_age_seconds=args.max_evidence_age_seconds,
        )
    else:  # pragma: no cover - argparse owns this invariant.
        raise RuntimeError(f"unsupported command: {args.command}")
    digest = publish_signed_no_overwrite(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "record_sha256": digest,
                "status": payload["status"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"cutover record refused: {exc}", file=sys.stderr)
        raise SystemExit(1)
