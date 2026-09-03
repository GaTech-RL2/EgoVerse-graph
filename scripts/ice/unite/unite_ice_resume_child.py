#!/usr/bin/env python3
"""Bind the runner's exact checkpoint bytes to the immutable ICE launcher.

With no arguments this is the runner child.  The ``validate-candidate`` and
``validate-authorization`` subcommands are dependency-light, deterministic
validators for the two immutable cutover records; see ``--help`` for their
canonical JSON schemas.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any, Sequence

SHA256_RE = re.compile(r"[0-9a-f]{64}")
TERMINAL_SOURCE_STATES = {
    "CANCELLED",
    "COMPLETED",
    "PREEMPTED",
    "TIMEOUT",
    "NODE_FAIL",
}
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
CANDIDATE_OPTIONAL_STRING_KEYS = {"created_at"}
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
AUTHORIZATION_OPTIONAL_STRING_KEYS = {
    "created_at",
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


def required(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"missing required environment variable {name}")
    return value


def stat_signature(path: Path) -> tuple[int, int, int, int, int]:
    value = path.stat()
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def read_bytes_stable(path: Path, label: str) -> bytes:
    if not path.is_file():
        raise RuntimeError(f"{label} is not a file: {path}")
    before = stat_signature(path)
    raw = path.read_bytes()
    after = stat_signature(path)
    if before != after:
        raise RuntimeError(f"{label} changed while being read: {path}")
    return raw


def sha256_stable(path: Path) -> str:
    before = stat_signature(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    after = stat_signature(path)
    if before != after:
        raise RuntimeError("checkpoint changed while its SHA-256 was recomputed")
    return digest.hexdigest()


def _json_object(raw: bytes, label: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _canonical_json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _positive_json_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"{label} must be a positive JSON integer")
    return value


def _nonempty_json_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a nonempty JSON string")
    return value


def _check_exact_keys(
    record: dict[str, Any],
    required_keys: set[str],
    optional_string_keys: set[str],
    label: str,
) -> None:
    missing = required_keys - set(record)
    unknown = set(record) - required_keys - optional_string_keys
    if missing or unknown:
        raise RuntimeError(
            f"{label} keys are not canonical: "
            f"missing={sorted(missing)} unknown={sorted(unknown)}"
        )
    for key in optional_string_keys.intersection(record):
        _nonempty_json_string(record[key], f"{label}.{key}")


def verify_record_sidecar(
    record_path: Path, sidecar_path: Path
) -> tuple[dict[str, Any], str]:
    if not record_path.is_absolute() or not sidecar_path.is_absolute():
        raise RuntimeError("handshake record and sidecar paths must be absolute")
    if record_path.is_symlink() or sidecar_path.is_symlink():
        raise RuntimeError("handshake record and sidecar must not be symlinks")
    resolved_record = record_path.resolve(strict=True)
    resolved_sidecar = sidecar_path.resolve(strict=True)
    if resolved_record != record_path or resolved_sidecar != sidecar_path:
        raise RuntimeError("handshake record and sidecar paths must be canonical")
    record_path = resolved_record
    sidecar_path = resolved_sidecar
    raw = read_bytes_stable(record_path, "handshake record")
    sidecar_raw = read_bytes_stable(sidecar_path, "handshake SHA sidecar")
    try:
        sidecar = sidecar_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("handshake SHA sidecar is not UTF-8") from exc
    match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)\n", sidecar)
    if match is None:
        raise RuntimeError(
            "handshake SHA sidecar must be exactly '<sha256>  <absolute-path>\\n'"
        )
    sidecar_record = Path(match.group(2))
    if not sidecar_record.is_absolute() or sidecar_record.resolve() != record_path:
        raise RuntimeError("handshake SHA sidecar names the wrong record")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != match.group(1):
        raise RuntimeError("handshake record SHA-256 mismatch")
    record = _json_object(raw, "handshake record")
    if raw != _canonical_json_bytes(record):
        raise RuntimeError("handshake record is not canonical sorted JSON")
    return record, digest


def load_pinned_cutover_writer(expected_sha256: str) -> Any:
    """Execute only the exact sibling writer bytes named by candidate provenance."""

    if SHA256_RE.fullmatch(expected_sha256) is None:
        raise RuntimeError("candidate provenance record-writer SHA-256 is invalid")
    path = (
        Path(__file__).resolve(strict=True).with_name("unite_cutover_record_writer.py")
    )
    if path.is_symlink() or not path.is_file() or path.resolve(strict=True) != path:
        raise RuntimeError("cutover record writer is not a canonical sibling file")
    raw = read_bytes_stable(path, "cutover record writer")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RuntimeError("cutover record writer differs from candidate provenance")
    module = types.ModuleType("_pinned_unite_cutover_record_writer")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def validate_pinned_cutover_tool_manifest(
    *, authorized_root: Path, provenance_match: re.Match[str]
) -> Any:
    """Load the exact writer and verify its signed companion tool manifest."""

    if (
        not authorized_root.is_absolute()
        or authorized_root.is_symlink()
        or not authorized_root.is_dir()
        or authorized_root.resolve(strict=True) != authorized_root
    ):
        raise RuntimeError("cutover authorized root is not canonical")
    cutover_writer = load_pinned_cutover_writer(provenance_match.group("record_writer"))
    manifest_path = cutover_writer.content_addressed_cutover_tool_manifest_path(
        authorized_root, provenance_match.group("tool_manifest")
    )
    manifest, manifest_sha, _ = cutover_writer.signed_json(
        manifest_path,
        expected_sha256=provenance_match.group("tool_manifest"),
        sidecar_absolute_path=True,
        label="cutover tool manifest",
    )
    tools = cutover_writer.validate_cutover_tool_manifest(manifest)
    if manifest_sha != provenance_match.group("tool_manifest"):
        raise RuntimeError("cutover tool manifest SHA-256 mismatch")
    expected = {
        "record_writer_sha256": provenance_match.group("record_writer"),
        "source_terminal_collector_sha256": provenance_match.group("collector"),
        "resume_child_path": str(Path(__file__).resolve(strict=True)),
        "resume_child_sha256": hashlib.sha256(
            read_bytes_stable(Path(__file__).resolve(strict=True), "resume child")
        ).hexdigest(),
    }
    for key, value in expected.items():
        if tools.get(key) != value:
            raise RuntimeError(f"cutover tool manifest mismatch for {key}")
    return cutover_writer


def staged_checkpoint_root_and_task(staged_checkpoint: Path) -> tuple[Path, str]:
    """Derive the ICE root and task from a canonical content-addressed stage."""

    if (
        not staged_checkpoint.is_absolute()
        or staged_checkpoint.is_symlink()
        or not staged_checkpoint.is_file()
        or staged_checkpoint.resolve(strict=True) != staged_checkpoint
    ):
        raise RuntimeError(
            "staged checkpoint must be an absolute canonical non-symlink file"
        )
    checkpoint_dir = staged_checkpoint.parent
    step_dir = checkpoint_dir.parent
    candidates_dir = step_dir.parent
    task_dir = candidates_dir.parent
    staging_dir = task_dir.parent
    if (
        checkpoint_dir.name != "checkpoints"
        or re.fullmatch(r"step-[1-9][0-9]*", step_dir.name) is None
        or candidates_dir.name != "candidates"
        or not task_dir.name
        or staging_dir.name != "staging"
    ):
        raise RuntimeError(
            "staged checkpoint has no canonical content-addressed task/root identity"
        )
    authorized_root = staging_dir.parent
    if len(authorized_root.parts) < 6:
        raise RuntimeError("derived ICE authorized root is not a specific path")
    return authorized_root, task_dir.name


def validate_cutover_lease(
    *,
    staged_checkpoint: Path,
    candidate_checkpoint: Path,
    candidate_step: int,
    source_job_id: int,
    successor_job_id: int,
    wandb_id: str,
    continuation_contract_sha256: str,
    allocation_record_sha256: str,
    cutover_tool_manifest_sha256: str,
    cutover_lease_sha256: str,
) -> dict[str, Any]:
    authorized_root, task_id = staged_checkpoint_root_and_task(staged_checkpoint)
    staging_dir = authorized_root / "staging"
    expected_candidate_parent = (
        staging_dir / task_id / "candidates" / f"step-{candidate_step}" / "checkpoints"
    )
    if candidate_checkpoint.parent != expected_candidate_parent:
        raise RuntimeError(
            "candidate checkpoint differs from cutover task/root identity"
        )
    lease_path = (
        authorized_root
        / "provenance/cutover-leases"
        / f"source-job_{source_job_id}.json"
    )
    lease, lease_sha = verify_record_sidecar(
        lease_path, Path(str(lease_path) + ".sha256")
    )
    if lease_sha != cutover_lease_sha256:
        raise RuntimeError("candidate provenance cutover lease SHA-256 mismatch")
    if set(lease) != CUTOVER_LEASE_KEYS:
        raise RuntimeError("cutover lease keys are not canonical")
    expected = {
        "schema_version": 1,
        "status": "SUCCESSOR_CLAIMED_FOR_SOURCE_CUTOVER",
        "task_id": task_id,
        "source_job_id": source_job_id,
        "successor_job_id": successor_job_id,
        "wandb_id": wandb_id,
        "continuation_contract_sha256": continuation_contract_sha256,
        "allocation_record_sha256": allocation_record_sha256,
        "cutover_tool_manifest_sha256": cutover_tool_manifest_sha256,
    }
    for key, value in expected.items():
        if lease.get(key) != value or type(lease.get(key)) is not type(value):
            raise RuntimeError(f"cutover lease mismatch for {key}")
    return lease


def validate_candidate_record(
    *,
    record_path: Path,
    sidecar_path: Path,
    source_job_id: int,
    successor_job_id: int,
    wandb_id: str,
    staged_checkpoint: Path,
    staged_sha256: str,
    staged_step: int,
    candidate_minimum_step: int,
    checkpoint_step_multiple: int,
    allowed_checkpoint_globs: Sequence[str],
) -> dict[str, Any]:
    source_job_id = _positive_json_integer(source_job_id, "expected source job ID")
    successor_job_id = _positive_json_integer(
        successor_job_id, "expected successor job ID"
    )
    staged_step = _positive_json_integer(staged_step, "expected staged step")
    candidate_minimum_step = _positive_json_integer(
        candidate_minimum_step, "candidate minimum step"
    )
    checkpoint_step_multiple = _positive_json_integer(
        checkpoint_step_multiple, "checkpoint step multiple"
    )
    if not wandb_id:
        raise RuntimeError("expected W&B ID must be nonempty")
    record, record_sha = verify_record_sidecar(record_path, sidecar_path)
    _check_exact_keys(
        record,
        CANDIDATE_REQUIRED_KEYS,
        CANDIDATE_OPTIONAL_STRING_KEYS,
        "candidate",
    )
    if (
        _positive_json_integer(record["schema_version"], "candidate.schema_version")
        != 1
    ):
        raise RuntimeError("candidate.schema_version must be 1")
    if _nonempty_json_string(record["status"], "candidate.status") != "CANDIDATE":
        raise RuntimeError("candidate.status must be CANDIDATE")
    if (
        _positive_json_integer(record["source_job_id"], "candidate.source_job_id")
        != source_job_id
    ):
        raise RuntimeError("candidate source job is wrong")
    if (
        _positive_json_integer(record["successor_job_id"], "candidate.successor_job_id")
        != successor_job_id
    ):
        raise RuntimeError("candidate successor job is wrong")
    if _nonempty_json_string(record["wandb_id"], "candidate.wandb_id") != wandb_id:
        raise RuntimeError("candidate W&B ID is wrong")
    provenance = _nonempty_json_string(
        record["verified_from"], "candidate.verified_from"
    )
    provenance_match = CANDIDATE_PROVENANCE_RE.fullmatch(provenance)
    if provenance_match is None:
        raise RuntimeError("candidate verified_from provenance is not canonical")
    checkpoint_text = _nonempty_json_string(
        record["checkpoint"], "candidate.checkpoint"
    )
    checkpoint = Path(checkpoint_text)
    if (
        not checkpoint.is_absolute()
        or checkpoint.is_symlink()
        or not checkpoint.is_file()
    ):
        raise RuntimeError("candidate checkpoint is not an absolute non-symlink file")
    resolved_checkpoint = checkpoint.resolve(strict=True)
    if checkpoint != resolved_checkpoint:
        raise RuntimeError("candidate checkpoint path is not canonical")
    checkpoint = resolved_checkpoint
    allowed: set[Path] = set()
    for pattern in allowed_checkpoint_globs:
        if not Path(pattern).is_absolute():
            raise RuntimeError("allowed checkpoint globs must be absolute")
        allowed.update(
            Path(raw).resolve(strict=True)
            for raw in glob.glob(pattern, recursive=True)
            if Path(raw).is_file() and not Path(raw).is_symlink()
        )
    if checkpoint not in allowed:
        raise RuntimeError("candidate checkpoint is outside the allowed globs")
    checkpoint_sha = _nonempty_json_string(
        record["checkpoint_sha256"], "candidate.checkpoint_sha256"
    )
    if SHA256_RE.fullmatch(checkpoint_sha) is None:
        raise RuntimeError("candidate checkpoint SHA-256 is invalid")
    config_sha = _nonempty_json_string(
        record["checkpoint_config_sha256"],
        "candidate.checkpoint_config_sha256",
    )
    if SHA256_RE.fullmatch(config_sha) is None:
        raise RuntimeError("candidate checkpoint config SHA-256 is invalid")
    if (
        _nonempty_json_string(
            record["checkpoint_config_role"],
            "candidate.checkpoint_config_role",
        )
        != "adjacent_origin_hydra"
    ):
        raise RuntimeError(
            "candidate checkpoint_config_role must be adjacent_origin_hydra"
        )
    config_identity_sha = _nonempty_json_string(
        record["config_sha256"], "candidate.config_sha256"
    )
    if SHA256_RE.fullmatch(config_identity_sha) is None:
        raise RuntimeError("candidate embedded config identity SHA-256 is invalid")
    step = _positive_json_integer(record["global_step"], "candidate.global_step")
    if step < staged_step:
        raise RuntimeError("candidate checkpoint is older than the staged baseline")
    if step == staged_step and checkpoint_sha != staged_sha256:
        raise RuntimeError("equal-step candidate must be the exact staged SHA")
    if step < candidate_minimum_step or step % checkpoint_step_multiple != 0:
        raise RuntimeError(
            "candidate step violates the minimum/periodic checkpoint contract"
        )
    if SHA256_RE.fullmatch(staged_sha256) is None:
        raise RuntimeError("staged checkpoint SHA-256 is invalid")
    if provenance_match.group("staged_baseline") != staged_sha256:
        raise RuntimeError("candidate provenance staged SHA differs from baseline")
    if int(provenance_match.group("staged_step")) != staged_step:
        raise RuntimeError("candidate provenance staged step differs from baseline")
    authorized_root, _ = staged_checkpoint_root_and_task(staged_checkpoint)
    validate_pinned_cutover_tool_manifest(
        authorized_root=authorized_root,
        provenance_match=provenance_match,
    )
    validate_cutover_lease(
        staged_checkpoint=staged_checkpoint,
        candidate_checkpoint=checkpoint,
        candidate_step=step,
        source_job_id=source_job_id,
        successor_job_id=successor_job_id,
        wandb_id=wandb_id,
        continuation_contract_sha256=provenance_match.group("contract"),
        allocation_record_sha256=provenance_match.group("allocation"),
        cutover_tool_manifest_sha256=provenance_match.group("tool_manifest"),
        cutover_lease_sha256=provenance_match.group("lease"),
    )
    return {
        **record,
        "checkpoint": str(checkpoint),
        "record_sha256": record_sha,
    }


def validate_authorization_record(
    *,
    record_path: Path,
    sidecar_path: Path,
    source_job_id: int,
    successor_job_id: int,
    wandb_id: str,
    candidate_checkpoint: Path,
    candidate_sha256: str,
    candidate_config_sha256: str,
    candidate_config_identity_sha256: str,
    candidate_step: int,
    candidate_record_sha256: str,
    ready_to_cancel_record_sha256: str | None = None,
) -> dict[str, Any]:
    source_job_id = _positive_json_integer(source_job_id, "expected source job ID")
    successor_job_id = _positive_json_integer(
        successor_job_id, "expected successor job ID"
    )
    candidate_step = _positive_json_integer(candidate_step, "expected candidate step")
    if not wandb_id:
        raise RuntimeError("expected W&B ID must be nonempty")
    if (
        not candidate_checkpoint.is_absolute()
        or candidate_checkpoint.is_symlink()
        or not candidate_checkpoint.is_file()
        or candidate_checkpoint.resolve(strict=True) != candidate_checkpoint
    ):
        raise RuntimeError("candidate checkpoint is not canonical")
    if (
        candidate_checkpoint.parent.name != "checkpoints"
        or candidate_checkpoint.parent.parent.parent.name != "candidates"
        or candidate_checkpoint.parent.parent.parent.parent.parent.name != "staging"
    ):
        raise RuntimeError("candidate checkpoint has no canonical cutover root")
    authorized_root = candidate_checkpoint.parents[5]
    record, record_sha = verify_record_sidecar(record_path, sidecar_path)
    _check_exact_keys(
        record,
        AUTHORIZATION_REQUIRED_KEYS,
        AUTHORIZATION_OPTIONAL_STRING_KEYS,
        "authorization",
    )
    if (
        _positive_json_integer(record["schema_version"], "authorization.schema_version")
        != 1
    ):
        raise RuntimeError("authorization.schema_version must be 1")
    if _nonempty_json_string(record["status"], "authorization.status") != "AUTHORIZED":
        raise RuntimeError("authorization.status must be AUTHORIZED")
    if (
        _positive_json_integer(record["source_job_id"], "authorization.source_job_id")
        != source_job_id
    ):
        raise RuntimeError("authorization source job is wrong")
    if (
        _positive_json_integer(
            record["successor_job_id"], "authorization.successor_job_id"
        )
        != successor_job_id
    ):
        raise RuntimeError("authorization successor job is wrong")
    if _nonempty_json_string(record["wandb_id"], "authorization.wandb_id") != wandb_id:
        raise RuntimeError("authorization W&B ID is wrong")
    provenance = _nonempty_json_string(
        record["verified_from"], "authorization.verified_from"
    )
    provenance_match = AUTHORIZATION_PROVENANCE_RE.fullmatch(provenance)
    if provenance_match is None:
        raise RuntimeError("authorization verified_from provenance is not canonical")
    if SHA256_RE.fullmatch(candidate_record_sha256) is None:
        raise RuntimeError("candidate record SHA-256 is invalid")
    record_ready_sha = _nonempty_json_string(
        record["ready_to_cancel_record_sha256"],
        "authorization.ready_to_cancel_record_sha256",
    )
    if SHA256_RE.fullmatch(record_ready_sha) is None:
        raise RuntimeError("ready-to-cancel record SHA-256 is invalid")
    if ready_to_cancel_record_sha256 is not None:
        if SHA256_RE.fullmatch(ready_to_cancel_record_sha256) is None:
            raise RuntimeError("expected ready-to-cancel record SHA-256 is invalid")
        if ready_to_cancel_record_sha256 != record_ready_sha:
            raise RuntimeError("authorization ready-to-cancel expectation is wrong")
    ready_to_cancel_record_sha256 = record_ready_sha
    if provenance_match.group("candidate") != candidate_record_sha256:
        raise RuntimeError("authorization provenance does not bind the candidate")
    if provenance_match.group("ready") != ready_to_cancel_record_sha256:
        raise RuntimeError("authorization provenance does not bind ready-to-cancel")

    candidate_record_path = Path(
        _nonempty_json_string(
            record["candidate_record"], "authorization.candidate_record"
        )
    )
    original_candidate, original_candidate_sha = verify_record_sidecar(
        candidate_record_path, Path(str(candidate_record_path) + ".sha256")
    )
    if original_candidate_sha != candidate_record_sha256:
        raise RuntimeError("authorization candidate record content identity changed")
    _check_exact_keys(
        original_candidate,
        CANDIDATE_REQUIRED_KEYS,
        CANDIDATE_OPTIONAL_STRING_KEYS,
        "authorization original candidate",
    )
    candidate_provenance_text = _nonempty_json_string(
        original_candidate["verified_from"],
        "authorization original candidate provenance",
    )
    candidate_provenance = CANDIDATE_PROVENANCE_RE.fullmatch(candidate_provenance_text)
    if candidate_provenance is None:
        raise RuntimeError(
            "authorization original candidate provenance is not canonical"
        )
    cutover_writer = validate_pinned_cutover_tool_manifest(
        authorized_root=authorized_root,
        provenance_match=candidate_provenance,
    )
    for key in CANDIDATE_REQUIRED_KEYS - {"status", "verified_from"}:
        if original_candidate.get(key) != record.get(key) or type(
            original_candidate.get(key)
        ) is not type(record.get(key)):
            raise RuntimeError(
                f"authorization differs from original candidate for {key}"
            )
    if original_candidate.get("status") != "CANDIDATE":
        raise RuntimeError("authorization original candidate status is wrong")

    ready_path = Path(
        _nonempty_json_string(
            record["ready_to_cancel_record"], "authorization.ready_to_cancel_record"
        )
    )
    ready, ready_sha = verify_record_sidecar(
        ready_path, Path(str(ready_path) + ".sha256")
    )
    if ready_sha != ready_to_cancel_record_sha256:
        raise RuntimeError(
            "authorization ready-to-cancel record content identity changed"
        )
    if set(ready) != READY_TO_CANCEL_KEYS:
        raise RuntimeError(
            "authorization ready-to-cancel record keys are not canonical"
        )
    expected_ready = {
        "schema_version": 1,
        "status": "READY_TO_CANCEL_SOURCE",
        "source_job_id": source_job_id,
        "successor_job_id": successor_job_id,
        "wandb_id": wandb_id,
        "candidate_record": str(candidate_record_path),
        "candidate_record_sha256": candidate_record_sha256,
        "checkpoint": record["checkpoint"],
        "checkpoint_sha256": record["checkpoint_sha256"],
        "checkpoint_config_role": "adjacent_origin_hydra",
        "checkpoint_config_sha256": record["checkpoint_config_sha256"],
        "config_sha256": record["config_sha256"],
        "global_step": record["global_step"],
        "cutover_lease_sha256": candidate_provenance.group("lease"),
        "continuation_contract_sha256": candidate_provenance.group("contract"),
        "allocation_record_sha256": candidate_provenance.group("allocation"),
        "cutover_tool_manifest_sha256": candidate_provenance.group("tool_manifest"),
    }
    for key, value in expected_ready.items():
        if ready.get(key) != value or type(ready.get(key)) is not type(value):
            raise RuntimeError(f"authorization ready-to-cancel mismatch for {key}")
    _nonempty_json_string(ready.get("task_id"), "ready-to-cancel task ID")
    ready_successor_sha = ready.get("ready_record_sha256")
    if (
        not isinstance(ready_successor_sha, str)
        or SHA256_RE.fullmatch(ready_successor_sha) is None
    ):
        raise RuntimeError("ready-to-cancel has invalid successor-ready SHA-256")

    evidence_path = Path(
        _nonempty_json_string(
            record["source_terminal_evidence"],
            "authorization.source_terminal_evidence",
        )
    )
    evidence, evidence_sha = verify_record_sidecar(
        evidence_path, Path(str(evidence_path) + ".sha256")
    )
    if evidence_sha != record["source_terminal_evidence_sha256"]:
        raise RuntimeError("authorization source evidence field SHA-256 mismatch")
    if evidence_sha != provenance_match.group("evidence"):
        raise RuntimeError("authorization provenance does not bind source evidence")
    terminal_evidence = cutover_writer.validate_source_evidence(
        evidence,
        evidence_stat=os.lstat(evidence_path),
        source_job_id=source_job_id,
        successor_job_id=successor_job_id,
        wandb_id=wandb_id,
        candidate_record_sha256=candidate_record_sha256,
        allocation_record_sha256=candidate_provenance.group("allocation"),
        ready_to_cancel_record_sha256=ready_to_cancel_record_sha256,
        candidate_provenance=candidate_provenance,
        max_age_seconds=cutover_writer.MAX_ALLOWED_EVIDENCE_AGE_SECONDS,
        now_unix_ns=time.time_ns(),
        enforce_file_freshness=False,
    )
    source_state = _nonempty_json_string(
        record["source_state"], "authorization.source_state"
    )
    if source_state not in TERMINAL_SOURCE_STATES:
        raise RuntimeError("authorization source state is not terminal")
    if source_state != terminal_evidence["source_state"]:
        raise RuntimeError("authorization source state differs from terminal evidence")
    if record["active_queue_state"] != "absent":
        raise RuntimeError("authorization source job is still active")
    if record["candidate_record_sha256"] != candidate_record_sha256:
        raise RuntimeError("authorization does not bind the strict-loaded candidate")
    if record["ready_to_cancel_record_sha256"] != ready_to_cancel_record_sha256:
        raise RuntimeError("authorization does not bind ready-to-cancel")
    if record["source_terminal_evidence_sha256"] != evidence_sha:
        raise RuntimeError("authorization does not bind source terminal evidence")
    if record["source_exit_code"] != terminal_evidence["source_exit_code"]:
        raise RuntimeError("authorization source exit code differs from evidence")
    if record["source_end_time"] != terminal_evidence["source_end_time"]:
        raise RuntimeError("authorization source end time differs from evidence")
    checkpoint_text = _nonempty_json_string(
        record["checkpoint"], "authorization.checkpoint"
    )
    checkpoint = Path(checkpoint_text)
    if (
        not checkpoint.is_absolute()
        or checkpoint.is_symlink()
        or not checkpoint.is_file()
        or checkpoint.resolve(strict=True) != checkpoint
        or candidate_checkpoint.is_symlink()
        or not candidate_checkpoint.is_file()
        or candidate_checkpoint.resolve(strict=True) != candidate_checkpoint
        or checkpoint != candidate_checkpoint
    ):
        raise RuntimeError("authorization checkpoint path differs from candidate")
    if record["checkpoint_sha256"] != candidate_sha256:
        raise RuntimeError("authorization checkpoint SHA differs from candidate")
    if SHA256_RE.fullmatch(candidate_sha256) is None:
        raise RuntimeError("candidate checkpoint SHA-256 is invalid")
    if record["checkpoint_config_sha256"] != candidate_config_sha256:
        raise RuntimeError("authorization checkpoint config SHA differs from candidate")
    if SHA256_RE.fullmatch(candidate_config_sha256) is None:
        raise RuntimeError("candidate checkpoint config SHA-256 is invalid")
    if record["checkpoint_config_role"] != "adjacent_origin_hydra":
        raise RuntimeError(
            "authorization checkpoint_config_role must be adjacent_origin_hydra"
        )
    if record["config_sha256"] != candidate_config_identity_sha256:
        raise RuntimeError(
            "authorization embedded config identity SHA differs from candidate"
        )
    if SHA256_RE.fullmatch(candidate_config_identity_sha256) is None:
        raise RuntimeError("candidate embedded config identity SHA-256 is invalid")
    step = _positive_json_integer(record["global_step"], "authorization.global_step")
    if step != candidate_step:
        raise RuntimeError("authorization checkpoint step differs from candidate")
    task_dir = checkpoint.parent.parent.parent.parent
    if (
        checkpoint.parent.name != "checkpoints"
        or checkpoint.parent.parent.parent.name != "candidates"
        or ready.get("task_id") != task_dir.name
    ):
        raise RuntimeError("authorization ready-to-cancel task identity is wrong")
    return {
        **record,
        "checkpoint": str(checkpoint),
        "record_sha256": record_sha,
    }


def validate_runner_origin_config_sha256(
    runner_metadata: dict[str, Any], candidate_config_sha256: str
) -> str:
    """Bind the runner's freshly observed origin config to the cutover record."""

    candidate_config_sha256 = _nonempty_json_string(
        candidate_config_sha256, "candidate checkpoint config SHA-256"
    )
    if SHA256_RE.fullmatch(candidate_config_sha256) is None:
        raise RuntimeError("candidate checkpoint config SHA-256 is invalid")
    checkpoint_config_sha = str(runner_metadata.get("checkpoint_config_sha256", ""))
    if SHA256_RE.fullmatch(checkpoint_config_sha) is None:
        raise RuntimeError(
            "runner metadata has no exact checkpoint Hydra config SHA-256"
        )
    if checkpoint_config_sha != candidate_config_sha256:
        raise RuntimeError(
            "runner checkpoint origin config SHA differs from authorized candidate"
        )
    if runner_metadata.get("checkpoint_config_role") != "adjacent_origin_hydra":
        raise RuntimeError(
            "runner metadata does not identify the checkpoint config as origin Hydra"
        )
    return checkpoint_config_sha


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def parse_last_json(stdout: str, label: str) -> dict:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{label} emitted no JSON record")
    return _json_object(lines[-1].encode("utf-8"), f"{label} final line")


def validate_static_requeue_contract(launcher: Path) -> dict[str, Any]:
    expected_launcher_sha = required("EXPECTED_LAUNCHER_SHA")
    if SHA256_RE.fullmatch(expected_launcher_sha) is None:
        raise RuntimeError("invalid expected launcher SHA-256")
    if sha256_stable(launcher) != expected_launcher_sha:
        raise RuntimeError("launcher SHA-256 changed before child exec")
    contract_path = Path(required("CONTINUATION_CONTRACT")).resolve()
    expected_contract_sha = required("EXPECTED_CONTINUATION_CONTRACT_SHA256")
    if SHA256_RE.fullmatch(expected_contract_sha) is None:
        raise RuntimeError("invalid expected continuation-contract SHA-256")
    if sha256_stable(contract_path) != expected_contract_sha:
        raise RuntimeError("continuation-contract SHA-256 changed before child exec")
    contract = _json_object(
        read_bytes_stable(contract_path, "continuation contract"),
        "continuation contract",
    )
    if contract.get("schema_version") != 2:
        raise RuntimeError("continuation contract must use schema version 2")
    expected_requeue = {
        "owner": "runner",
        "save_signal": "SIGUSR2",
        "checkpoint_forwarding": "slurm-steps",
        "signal_checkpoint_dir": "${paths.output_dir}/checkpoints",
        "framework_auto_requeue": False,
        "fresh_checkpoint_required": True,
    }
    if contract.get("requeue_contract") != expected_requeue:
        raise RuntimeError("continuation requeue contract is not runner-owned USR2")
    return expected_requeue


def validate_runner_attempt(
    attempt_dir: Path,
    checkpoint: Path,
    checkpoint_sha: str,
    checkpoint_step: int,
    validator: Path,
    initial_checkpoint: Path,
    initial_checkpoint_sha: str,
    initial_checkpoint_step: int,
    expected_checkpoint_globs: Sequence[str],
    scancel: Path,
) -> dict[str, Any]:
    attempt_path = attempt_dir / "attempt.json"
    last_error: RuntimeError | None = None
    # The runner atomically advances STARTING -> RUNNING immediately after
    # spawning this child.  A replacement exactly between stat/read/stat is a
    # benign race; retry it, but keep the wait bounded and fail closed.
    for _ in range(50):
        try:
            attempt = _json_object(
                read_bytes_stable(attempt_path, "runner attempt record"),
                "runner attempt record",
            )
            break
        except RuntimeError as exc:
            last_error = exc
            time.sleep(0.02)
    else:
        raise RuntimeError("runner attempt record never became stable") from last_error
    if attempt.get("schema_version") != 2:
        raise RuntimeError("runner attempt record has the wrong schema")
    if attempt.get("status") not in {"STARTING", "RUNNING"}:
        raise RuntimeError("runner attempt is not in a child-start state")
    if attempt.get("requeue_owner") != "runner":
        raise RuntimeError("runner attempt does not own requeue")
    if attempt.get("checkpoint_signal") != "SIGUSR2":
        raise RuntimeError("runner attempt does not use the save-only SIGUSR2 path")
    if attempt.get("checkpoint_forwarding") != "slurm-steps":
        raise RuntimeError("runner attempt does not exclude the batch shell")
    if Path(str(attempt.get("scancel", ""))).resolve() != scancel:
        raise RuntimeError("runner attempt used a different scancel transport")
    if attempt.get("checkpoint_globs") != list(expected_checkpoint_globs):
        raise RuntimeError("runner attempt used different checkpoint roots")
    if Path(str(attempt.get("validator", ""))).resolve() != validator:
        raise RuntimeError("runner attempt used a different checkpoint validator")
    selected = attempt.get("checkpoint")
    if not isinstance(selected, dict):
        raise RuntimeError("runner attempt has no selected checkpoint")
    if Path(str(selected.get("path", ""))).resolve() != checkpoint:
        raise RuntimeError("runner attempt checkpoint path changed")
    if selected.get("sha256") != checkpoint_sha:
        raise RuntimeError("runner attempt checkpoint SHA-256 changed")
    if selected.get("global_step") != checkpoint_step:
        raise RuntimeError("runner attempt checkpoint step changed")
    initial = attempt.get("initial_checkpoint")
    if not isinstance(initial, dict):
        raise RuntimeError("runner attempt has no validated initial checkpoint")
    if Path(str(initial.get("path", ""))).resolve() != initial_checkpoint:
        raise RuntimeError("runner attempt initial checkpoint path changed")
    if initial.get("sha256") != initial_checkpoint_sha:
        raise RuntimeError("runner attempt initial checkpoint SHA-256 changed")
    if initial.get("global_step") != initial_checkpoint_step:
        raise RuntimeError("runner attempt initial checkpoint step changed")
    command = attempt.get("command")
    if not isinstance(command, list) or len(command) != 2:
        raise RuntimeError("runner attempt child command is not exact")
    if Path(str(command[0])).resolve() != Path(sys.executable).resolve():
        raise RuntimeError("runner attempt used a different Python interpreter")
    if Path(str(command[1])).resolve() != Path(__file__).resolve():
        raise RuntimeError("runner attempt used a different resume child")
    return attempt


def _positive_cli_integer(value: str) -> int:
    if re.fullmatch(r"[1-9][0-9]*", value) is None:
        raise argparse.ArgumentTypeError("must be a positive base-10 integer")
    return int(value)


def record_validator_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate an immutable ICE UNITE cutover record.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Canonical CANDIDATE JSON (unknown keys are rejected):
  required: schema_version=1, status="CANDIDATE", source_job_id:int,
            successor_job_id:int, wandb_id:str, checkpoint:absolute str,
            checkpoint_sha256:lowercase 64-hex,
            checkpoint_config_role="adjacent_origin_hydra",
            checkpoint_config_sha256:lowercase 64-hex byte identity of the
              adjacent source-run .hydra/config.yaml,
            config_sha256:lowercase 64-hex semantic identity computed from
              checkpoint hyper_parameters.config_tree,
            global_step:positive int, verified_from:canonical writer provenance
  optional: created_at:str

Canonical AUTHORIZED JSON (unknown keys are rejected):
  all CANDIDATE fields, with status="AUTHORIZED", plus
  source_state in {CANCELLED,COMPLETED,PREEMPTED,TIMEOUT,NODE_FAIL},
  active_queue_state="absent", candidate_record_sha256:lowercase 64-hex,
  ready/source-evidence absolute paths and lowercase 64-hex identities,
  source_exit_code:str, source_end_time:str, and
  verified_from:canonical candidate/ready/evidence provenance
  optional: created_at:str

Each sidecar is exactly: <sha256><two spaces><absolute record path><newline>.
The default output is one compact, sorted JSON object.  --format=tsv is only
for the shell handoff's internal field transport.
""",
    )
    commands = parser.add_subparsers(dest="record_command", required=True)

    candidate = commands.add_parser("validate-candidate")
    candidate.add_argument("--record", type=Path, required=True)
    candidate.add_argument("--sidecar", type=Path, required=True)
    candidate.add_argument("--source-job-id", type=_positive_cli_integer, required=True)
    candidate.add_argument(
        "--successor-job-id", type=_positive_cli_integer, required=True
    )
    candidate.add_argument("--wandb-id", required=True)
    candidate.add_argument("--staged-checkpoint", type=Path, required=True)
    candidate.add_argument("--staged-sha256", required=True)
    candidate.add_argument("--staged-step", type=_positive_cli_integer, required=True)
    candidate.add_argument(
        "--candidate-minimum-step", type=_positive_cli_integer, required=True
    )
    candidate.add_argument(
        "--checkpoint-step-multiple", type=_positive_cli_integer, required=True
    )
    candidate.add_argument("--allowed-checkpoint-glob", action="append", required=True)
    candidate.add_argument("--format", choices=("json", "tsv"), default="json")

    authorization = commands.add_parser("validate-authorization")
    authorization.add_argument("--record", type=Path, required=True)
    authorization.add_argument("--sidecar", type=Path, required=True)
    authorization.add_argument(
        "--source-job-id", type=_positive_cli_integer, required=True
    )
    authorization.add_argument(
        "--successor-job-id", type=_positive_cli_integer, required=True
    )
    authorization.add_argument("--wandb-id", required=True)
    authorization.add_argument("--candidate-checkpoint", type=Path, required=True)
    authorization.add_argument("--candidate-sha256", required=True)
    authorization.add_argument("--candidate-config-sha256", required=True)
    authorization.add_argument("--candidate-config-identity-sha256", required=True)
    authorization.add_argument(
        "--candidate-step", type=_positive_cli_integer, required=True
    )
    authorization.add_argument("--candidate-record-sha256", required=True)
    authorization.add_argument("--ready-to-cancel-record-sha256")
    authorization.add_argument("--format", choices=("json", "tsv"), default="json")
    return parser


def validate_record_cli(argv: Sequence[str]) -> int:
    args = record_validator_parser().parse_args(argv)
    if args.record_command == "validate-candidate":
        record = validate_candidate_record(
            record_path=args.record,
            sidecar_path=args.sidecar,
            source_job_id=args.source_job_id,
            successor_job_id=args.successor_job_id,
            wandb_id=args.wandb_id,
            staged_checkpoint=args.staged_checkpoint,
            staged_sha256=args.staged_sha256,
            staged_step=args.staged_step,
            candidate_minimum_step=args.candidate_minimum_step,
            checkpoint_step_multiple=args.checkpoint_step_multiple,
            allowed_checkpoint_globs=args.allowed_checkpoint_glob,
        )
        if args.format == "tsv":
            fields = (
                record["checkpoint"],
                record["checkpoint_sha256"],
                record["checkpoint_config_sha256"],
                record["config_sha256"],
                str(record["global_step"]),
                record["record_sha256"],
            )
            if any(
                any(character in value for character in "\t\r\n") for value in fields
            ):
                raise RuntimeError("candidate fields are unsafe for TSV transport")
            print("\t".join(fields))
        else:
            print(json.dumps(record, sort_keys=True, separators=(",", ":")))
        return 0
    if args.record_command == "validate-authorization":
        record = validate_authorization_record(
            record_path=args.record,
            sidecar_path=args.sidecar,
            source_job_id=args.source_job_id,
            successor_job_id=args.successor_job_id,
            wandb_id=args.wandb_id,
            candidate_checkpoint=args.candidate_checkpoint,
            candidate_sha256=args.candidate_sha256,
            candidate_config_sha256=args.candidate_config_sha256,
            candidate_config_identity_sha256=(args.candidate_config_identity_sha256),
            candidate_step=args.candidate_step,
            candidate_record_sha256=args.candidate_record_sha256,
            ready_to_cancel_record_sha256=args.ready_to_cancel_record_sha256,
        )
        if args.format == "tsv":
            print(
                record["record_sha256"] + "\t" + record["ready_to_cancel_record_sha256"]
            )
        else:
            print(json.dumps(record, sort_keys=True, separators=(",", ":")))
        return 0
    raise RuntimeError(f"unsupported record command: {args.record_command}")


def main() -> int:
    if required("ICE_REQUEUE_OWNER") != "runner":
        raise RuntimeError("the universal runner must be the sole requeue owner")
    if required("ICE_CHILD_REQUEUE_DISABLED") != "1":
        raise RuntimeError("framework/child auto-requeue is not confirmed disabled")
    checkpoint = Path(required("ICE_RESUME_CHECKPOINT")).resolve()
    validator = Path(required("ICE_CHECKPOINT_VALIDATOR")).resolve()
    launcher = Path(required("LAUNCHER_PATH")).resolve()
    attempt_dir = Path(required("ICE_ATTEMPT_DIR")).resolve()
    if not checkpoint.is_file() or not validator.is_file():
        raise RuntimeError("checkpoint and validator must be files")
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        raise RuntimeError("launcher must be an executable file")
    requeue_contract = validate_static_requeue_contract(launcher)

    runner_sha = required("ICE_RESUME_CHECKPOINT_SHA256")
    if not SHA256_RE.fullmatch(runner_sha):
        raise RuntimeError("runner supplied an invalid checkpoint SHA-256")
    runner_step_text = required("ICE_RESUME_GLOBAL_STEP")
    if not re.fullmatch(r"[1-9][0-9]*", runner_step_text):
        raise RuntimeError("runner supplied an invalid checkpoint global_step")
    runner_step = int(runner_step_text)
    runner_metadata = _json_object(
        required("ICE_RESUME_CHECKPOINT_METADATA_JSON").encode("utf-8"),
        "runner checkpoint metadata",
    )
    if runner_metadata.get("status") != "passed":
        raise RuntimeError("runner checkpoint metadata did not pass validation")
    runner_metadata_checkpoint = _nonempty_json_string(
        runner_metadata.get("checkpoint"), "runner metadata checkpoint"
    )
    if Path(runner_metadata_checkpoint).resolve() != checkpoint:
        raise RuntimeError("runner checkpoint path and metadata disagree")
    if (
        _positive_json_integer(
            runner_metadata.get("global_step"), "runner metadata global_step"
        )
        != runner_step
    ):
        raise RuntimeError("runner checkpoint step and metadata disagree")
    metadata_sha = str(runner_metadata.get("checkpoint_sha256", ""))
    if metadata_sha not in {"", "not_computed", runner_sha}:
        raise RuntimeError("runner checkpoint SHA and metadata disagree")

    # Recompute the runner-provided content identity before loading anything.
    if sha256_stable(checkpoint) != runner_sha:
        raise RuntimeError("runner-provided checkpoint SHA-256 mismatch")
    candidate_checkpoint = Path(required("CANDIDATE_CKPT")).resolve()
    candidate_sha = required("CANDIDATE_SHA")
    candidate_config_sha = required("CANDIDATE_CONFIG_SHA")
    candidate_config_identity_sha = required("CANDIDATE_CONFIG_IDENTITY_SHA")
    candidate_record_sha = required("CANDIDATE_RECORD_SHA")
    ready_to_cancel_record_sha = required("READY_TO_CANCEL_RECORD_SHA")
    candidate_step_text = required("CANDIDATE_STEP")
    if (
        SHA256_RE.fullmatch(candidate_sha) is None
        or SHA256_RE.fullmatch(candidate_config_sha) is None
        or SHA256_RE.fullmatch(candidate_config_identity_sha) is None
        or SHA256_RE.fullmatch(candidate_record_sha) is None
        or SHA256_RE.fullmatch(ready_to_cancel_record_sha) is None
        or re.fullmatch(r"[1-9][0-9]*", candidate_step_text) is None
    ):
        raise RuntimeError("invalid authorized-candidate identity")
    source_job_text = required("RESUME_SOURCE_JOB")
    successor_job_text = required("SLURM_JOB_ID")
    if (
        re.fullmatch(r"[1-9][0-9]*", source_job_text) is None
        or re.fullmatch(r"[1-9][0-9]*", successor_job_text) is None
    ):
        raise RuntimeError("invalid source/successor Slurm identity")
    authorization = validate_authorization_record(
        record_path=Path(required("RESUME_AUTHORIZATION")),
        sidecar_path=Path(required("RESUME_AUTHORIZATION") + ".sha256"),
        source_job_id=int(source_job_text),
        successor_job_id=int(successor_job_text),
        wandb_id=required("ICE_EXPECTED_WANDB_ID"),
        candidate_checkpoint=candidate_checkpoint,
        candidate_sha256=candidate_sha,
        candidate_config_sha256=candidate_config_sha,
        candidate_config_identity_sha256=candidate_config_identity_sha,
        candidate_step=int(candidate_step_text),
        candidate_record_sha256=candidate_record_sha,
        ready_to_cancel_record_sha256=ready_to_cancel_record_sha,
    )
    authorization_sha = required("RESUME_AUTHORIZATION_SHA256")
    if (
        SHA256_RE.fullmatch(authorization_sha) is None
        or authorization["record_sha256"] != authorization_sha
    ):
        raise RuntimeError("resume authorization SHA-256 changed before child load")
    expected_scancel = Path(required("ICE_EXPECTED_SCANCEL")).resolve()
    if not expected_scancel.is_file() or not os.access(expected_scancel, os.X_OK):
        raise RuntimeError("expected scancel transport is not executable")
    runner_attempt = validate_runner_attempt(
        attempt_dir,
        checkpoint,
        runner_sha,
        runner_step,
        validator,
        candidate_checkpoint,
        candidate_sha,
        int(candidate_step_text),
        (
            required("STAGING_CHECKPOINT_GLOB"),
            required("LIVE_CHECKPOINT_GLOB"),
        ),
        expected_scancel,
    )
    is_authorized_source_candidate = (
        checkpoint == candidate_checkpoint
        and runner_sha == candidate_sha
        and runner_step == int(candidate_step_text)
    )
    runtime_contract_verified = runner_metadata.get("runtime_requeue_contract_verified")
    if not isinstance(runtime_contract_verified, bool):
        raise RuntimeError("runner metadata has no typed runtime requeue proof")
    if not is_authorized_source_candidate and not runtime_contract_verified:
        raise RuntimeError("ICE checkpoint has no resolved save-only requeue contract")
    save_only_proof_verified = runner_metadata.get("save_only_signal_proof_verified")
    if not isinstance(save_only_proof_verified, bool):
        raise RuntimeError("runner metadata has no typed save-only proof result")
    if not is_authorized_source_candidate and not save_only_proof_verified:
        raise RuntimeError("ICE checkpoint has no exact save-only SIGUSR2 proof")
    if runtime_contract_verified:
        if runner_metadata.get("runtime_slurm_requeue_owner") != "runner":
            raise RuntimeError("checkpoint runtime requeue owner changed")
        if runner_metadata.get("runtime_slurm_save_signal") != "SIGUSR2":
            raise RuntimeError("checkpoint runtime save-only signal changed")

    validation_env = os.environ.copy()
    validation_env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "ICE_STRICT_LOAD": "1",
            "ICE_REQUIRE_REBASED_SCHEDULE": "1",
            "ICE_REQUIRE_RUNTIME_REQUEUE_CONTRACT": (
                "0" if is_authorized_source_candidate else "1"
            ),
            "ICE_EXPECTED_LR_START": required("RESUME_LR_START"),
            "ICE_EXPECTED_LR_FINAL": required("RESUME_LR_FINAL"),
            "ICE_EXPECTED_LR_START_STEP": required("RESUME_LR_START_STEP"),
            "ICE_EXPECTED_LR_END_STEP": required("RESUME_LR_END_STEP"),
            "ICE_EXPECTED_CHECKPOINT_SHA256": runner_sha,
            "ICE_EXPECTED_CHECKPOINT_STEP": str(runner_step),
        }
    )
    config_identity_sha = str(runner_metadata.get("config_sha256", ""))
    if not SHA256_RE.fullmatch(config_identity_sha):
        raise RuntimeError("runner metadata has no exact config identity SHA-256")
    if config_identity_sha != candidate_config_identity_sha:
        raise RuntimeError(
            "runner checkpoint embedded config identity differs from candidate"
        )
    if runner_metadata.get("embedded_config_identity_sha256") != config_identity_sha:
        raise RuntimeError(
            "runner metadata confuses embedded and origin config identities"
        )
    runner_checkpoint_config_sha = validate_runner_origin_config_sha256(
        runner_metadata, candidate_config_sha
    )
    validation_env["ICE_EXPECTED_CONFIG_IDENTITY_SHA256"] = config_identity_sha
    validation_env["ICE_EXPECTED_CHECKPOINT_CONFIG_SHA256"] = (
        runner_checkpoint_config_sha
    )
    validation = subprocess.run(
        [os.fspath(validator), os.fspath(checkpoint)],
        check=True,
        capture_output=True,
        text=True,
        env=validation_env,
    )
    record = parse_last_json(validation.stdout, "strict checkpoint validator")
    expected_wandb_id = required("ICE_EXPECTED_WANDB_ID")
    expected = {
        "status": "passed",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": runner_sha,
        "global_step": runner_step,
        "wandb_id": expected_wandb_id,
        "checkpoint_config_role": "adjacent_origin_hydra",
        "config_sha256": config_identity_sha,
        "embedded_config_identity_sha256": config_identity_sha,
        "checkpoint_config_sha256": candidate_config_sha,
        "strict_load": True,
        "full_state_verified": True,
        "strict_optimizer_load": True,
        "strict_scheduler_load": True,
        "tensor_finiteness": "complete",
        "rebased_schedule_verified": True,
        "runtime_requeue_contract_verified": runtime_contract_verified,
        "save_only_signal_proof_verified": save_only_proof_verified,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise RuntimeError(
                f"strict checkpoint record mismatch for {key}: "
                f"{record.get(key)!r} != {value!r}"
            )
    # The runner metadata is only a pre-exec assertion.  Rebind to the strict
    # validator's freshly observed adjacent origin config after checkpoint load
    # so a metadata/config swap in that interval cannot cross the cutover gate.
    checkpoint_config_sha = validate_runner_origin_config_sha256(
        record, candidate_config_sha
    )

    slurm_restart_text = os.environ.get("SLURM_RESTART_COUNT", "0")
    runner_restart_text = required("ICE_RESTART_COUNT")
    if (
        re.fullmatch(r"[0-9]+", slurm_restart_text) is None
        or re.fullmatch(r"[0-9]+", runner_restart_text) is None
    ):
        raise RuntimeError("restart counts must be base-10 nonnegative integers")
    restart_count = int(slurm_restart_text)
    if int(runner_restart_text) != restart_count:
        raise RuntimeError("runner and Slurm restart counts disagree")
    # This assertion is derived only after the exact checkpoint passed the
    # embedded optimizer/scheduler proof above; it is never trusted as input.
    already_rebased = "true"
    if os.environ.get("RESUME_LR_START") or os.environ.get("RESUME_LR_FINAL"):
        required("RESUME_LR_START")
        required("RESUME_LR_FINAL")
        required("RESUME_LR_START_STEP")
        required("RESUME_LR_END_STEP")

    child_env = os.environ.copy()
    child_env.update(
        {
            "RESUME_CKPT": str(checkpoint),
            "RESUME_EXPECTED_SHA256": runner_sha,
            "RESUME_EXPECTED_STEP": str(runner_step),
            "RESUME_WANDB_ID": expected_wandb_id,
            "RESUME_WANDB_MODE": "must",
            "RESUME_CHECKPOINT_ALREADY_REBASED": already_rebased,
        }
    )
    binding = {
        "schema_version": 2,
        "status": "strict_full_state_validated_and_bound",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": runner_sha,
        "global_step": runner_step,
        "checkpoint_origin_config_path": record["config_path"],
        "checkpoint_origin_config_role": "adjacent_origin_hydra",
        "checkpoint_origin_config_sha256": checkpoint_config_sha,
        "embedded_config_identity_sha256": config_identity_sha,
        "launcher": str(launcher),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_restart_count": restart_count,
        "wandb_id": expected_wandb_id,
        "resume_checkpoint_already_rebased": already_rebased == "true",
        "is_authorized_source_candidate": is_authorized_source_candidate,
        "requeue_contract": requeue_contract,
        "authorization": authorization,
        "runner_attempt": runner_attempt,
        "runner_metadata": runner_metadata,
        "strict_validation": record,
    }
    atomic_json(attempt_dir / "checkpoint_binding.json", binding)
    atomic_json(attempt_dir / "strict_checkpoint_validation.json", record)
    os.execve(str(launcher), [str(launcher)], child_env)
    return 127


if __name__ == "__main__":
    raise SystemExit(validate_record_cli(sys.argv[1:]) if len(sys.argv) > 1 else main())
