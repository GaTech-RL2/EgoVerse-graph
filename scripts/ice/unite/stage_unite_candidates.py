#!/usr/bin/env python3
"""Resume-capable, no-overwrite Skynet -> ICE checkpoint staging."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shlex
import stat
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

SHA256_RE = re.compile(r"[0-9a-f]{64}")
SAFE_REMOTE_RE = re.compile(r"[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+")
SAFE_PATH_RE = re.compile(r"/[A-Za-z0-9_./=-]+")
SAFE_TASK_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
SAFE_NODE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
UTC_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
CHECKPOINT_NAME_RE = re.compile(r"epoch-([0-9]+)-step-([0-9]+)\.ckpt")
SSH_TRANSPORT_ARGS = (
    "-T",
    "-o", "ClearAllForwardings=yes",
    "-o", "ForwardAgent=no",
    "-o", "ConnectTimeout=15",
    "-o", "ConnectionAttempts=1",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=2",
)
REMOTE_STAT_KEYS = {
    "device", "inode", "size", "mtime_ns", "ctime_ns", "mode", "uid", "gid"
}
PROOF_KEYS = {
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
COLLISION_VARIANT_KEYS = {
    "manifest",
    "manifest_sha256",
    "source_validation_result_sha256",
    "stager",
    "stager_sha256",
}


def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json_stable(path: Path) -> tuple[Any, str]:
    before = path.stat()
    encoded = path.read_bytes()
    after = path.stat()
    def identity(item: os.stat_result) -> tuple[int, int, int, int]:
        return (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
    if identity(before) != identity(after):
        raise RuntimeError(f"file changed while being read: {path}")
    digest = hashlib.sha256(encoded).hexdigest()
    payload = json.loads(
        encoded,
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON number: {value}")
        ),
    )
    return payload, digest


def exact_access_probe(
    value: Any, remote: str, hostname: str, expected_user: str
) -> dict[str, Any]:
    expected_keys = {
        "remote",
        "attempted_at_utc",
        "attempt_count",
        "returncode",
        "hostname",
        "username",
        "stderr_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError(f"access probe for {remote} is not canonical")
    if value["remote"] != remote:
        raise ValueError(f"access probe remote mismatch for {remote}")
    if UTC_RE.fullmatch(str(value["attempted_at_utc"])) is None:
        raise ValueError(f"access probe timestamp is invalid for {remote}")
    if type(value["attempt_count"]) is not int or value["attempt_count"] < 1:
        raise ValueError(f"access probe attempt count is invalid for {remote}")
    if type(value["returncode"]) is not int or not 0 <= value["returncode"] <= 255:
        raise ValueError(f"access probe returncode is invalid for {remote}")
    if SHA256_RE.fullmatch(str(value["stderr_sha256"])) is None:
        raise ValueError(f"access probe stderr SHA-256 is invalid for {remote}")
    for key in ("hostname", "username"):
        if value[key] is not None and not isinstance(value[key], str):
            raise ValueError(f"access probe {key} is invalid for {remote}")
    if value["returncode"] == 0:
        if value["hostname"] != hostname or value["username"] != expected_user:
            raise ValueError(f"successful access probe identity mismatch for {remote}")
    return value


def exact_access_selection(value: Any) -> dict[str, Any]:
    expected_keys = {
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
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("access-selection artifact keys are not canonical")
    if value["schema_version"] != 1:
        raise ValueError("unsupported access-selection schema")
    if value["status"] != "READY":
        raise ValueError("access-selection artifact is not READY")
    if UTC_RE.fullmatch(str(value["completed_at_utc"])) is None:
        raise ValueError("access-selection completion timestamp is invalid")
    context = value["probe_context"]
    context_keys = {
        "cluster", "slurm_job_id", "slurm_node", "account", "partition", "qos"
    }
    if not isinstance(context, dict) or set(context) != context_keys:
        raise ValueError("access-selection probe context is not canonical")
    if context["cluster"] != "PACE ICE":
        raise ValueError("access selection was not probed from PACE ICE")
    if (
        not isinstance(context["slurm_job_id"], str)
        or not context["slurm_job_id"].isdigit()
        or int(context["slurm_job_id"]) < 1
    ):
        raise ValueError("access-selection Slurm job id is invalid")
    if SAFE_NODE_RE.fullmatch(str(context["slurm_node"])) is None:
        raise ValueError("access-selection Slurm node is invalid")
    for key in ("account", "partition", "qos"):
        if not isinstance(context[key], str) or not context[key]:
            raise ValueError(f"access-selection {key} is invalid")
    if context["partition"] != "ice-cpu":
        raise ValueError("access selection was not probed in an ICE CPU allocation")
    canonical_remote = str(value["canonical_remote"])
    fallback_remote = str(value["fallback_remote"])
    if (
        SAFE_REMOTE_RE.fullmatch(canonical_remote) is None
        or SAFE_REMOTE_RE.fullmatch(fallback_remote) is None
        or canonical_remote == fallback_remote
    ):
        raise ValueError("access-selection remotes are invalid")
    canonical_user, canonical_hostname = canonical_remote.split("@", 1)
    fallback_user, fallback_hostname = fallback_remote.split("@", 1)
    if canonical_user != fallback_user:
        raise ValueError("canonical and fallback remotes must use the same user")
    canonical = exact_access_probe(
        value["canonical_probe"],
        canonical_remote,
        canonical_hostname,
        canonical_user,
    )
    fallback = value["fallback_probe"]
    if canonical["returncode"] == 0:
        if fallback is not None:
            raise ValueError("fallback probe must be absent after canonical success")
        if value["selected_remote"] != canonical_remote:
            raise ValueError("canonical success did not select sky1")
        if value["selection_reason"] != "canonical_probe_succeeded":
            raise ValueError("canonical selection reason is invalid")
    else:
        if canonical["attempt_count"] < 2:
            raise ValueError("sky1 fallback requires at least two failed canonical probes")
        fallback = exact_access_probe(
            fallback,
            fallback_remote,
            fallback_hostname,
            fallback_user,
        )
        if fallback["returncode"] != 0:
            raise ValueError("neither canonical nor fallback Skynet probe succeeded")
        if value["selected_remote"] != fallback_remote:
            raise ValueError("fallback success did not select sky2")
        if value["selection_reason"] != "canonical_probe_failed_fallback_succeeded":
            raise ValueError("fallback selection reason is invalid")
    return value


def stable_sha256(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    def identity(item: os.stat_result) -> tuple[int, int, int, int]:
        return (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
    if identity(before) != identity(after):
        raise RuntimeError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def canonical_existing_root(path: Path) -> Path:
    if not path.is_absolute() or len(path.parts) < 6:
        raise RuntimeError("authorized root is not a specific absolute path")
    canonical = path.resolve(strict=True)
    if canonical != path or not canonical.is_dir():
        raise RuntimeError("authorized root is not a canonical existing directory")
    return canonical


def ensure_directory_below(root: Path, path: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"directory escapes authorized root: {path}") from exc
    current = root
    for part in relative.parts:
        if part in {"", ".", ".."}:
            raise RuntimeError(f"directory path is not canonical: {path}")
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f"refusing symlink directory component: {current}")
        try:
            current.mkdir(mode=0o700)
        except FileExistsError:
            pass
        if current.is_symlink() or not current.is_dir():
            raise RuntimeError(f"directory component is not a real directory: {current}")


def fsync_file(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verified_existing(path: Path, expected_sha: str, size: int) -> bool:
    if path.is_symlink():
        raise RuntimeError(f"refusing symlink artifact: {path}")
    if not path.exists():
        return False
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != size:
        raise RuntimeError(f"existing destination has wrong identity: {path}")
    if stable_sha256(path) != expected_sha:
        raise RuntimeError(f"existing destination SHA-256 differs: {path}")
    return True


@contextmanager
def exclusive_row_lock(path: Path):
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError(f"row lock is not a regular file: {path}")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def publish_no_overwrite(
    partial: Path, final: Path, expected_sha: str, size: int
) -> str:
    if verified_existing(final, expected_sha, size):
        if partial.is_symlink():
            raise RuntimeError(f"refusing symlink partial: {partial}")
        if partial.exists():
            if not verified_existing(partial, expected_sha, size):
                raise RuntimeError(f"completed partial differs from destination: {partial}")
            partial.unlink()
            fsync_directory(final.parent)
        return "reused"
    if partial.is_symlink() or not verified_existing(partial, expected_sha, size):
        raise RuntimeError(f"downloaded partial failed size/SHA verification: {partial}")
    published = True
    try:
        os.link(partial, final)
    except FileExistsError:
        published = False
        if not verified_existing(final, expected_sha, size):
            raise RuntimeError(f"concurrent publisher installed different bytes: {final}")
    os.chmod(final, 0o444)
    fsync_file(final)
    partial.unlink()
    fsync_directory(final.parent)
    if not verified_existing(final, expected_sha, size):
        raise RuntimeError(f"published destination changed after verification: {final}")
    return "published" if published else "reused_concurrent"


def remote_stat(wrapper: Path, remote: str, path: str) -> dict[str, int]:
    program = (
        "import json, os, stat, sys\n"
        "value = os.lstat(sys.argv[1])\n"
        "if not stat.S_ISREG(value.st_mode):\n"
        "    raise SystemExit('source is not a regular file')\n"
        "print(json.dumps({\n"
        "    'device': value.st_dev, 'inode': value.st_ino,\n"
        "    'size': value.st_size, 'mtime_ns': value.st_mtime_ns,\n"
        "    'ctime_ns': value.st_ctime_ns, 'mode': value.st_mode,\n"
        "    'uid': value.st_uid, 'gid': value.st_gid,\n"
        "}, sort_keys=True, separators=(',', ':')))\n"
    )
    command = "python3 -c " + shlex.quote(program) + " " + shlex.quote(path)
    try:
        completed = subprocess.run(
            [str(wrapper), *SSH_TRANSPORT_ARGS, remote, command],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=45,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"remote stat timed out for {path}") from exc
    if completed.returncode != 0:
        raise RuntimeError(f"remote stat failed for {path}: {completed.stderr.strip()}")
    try:
        payload = json.loads(
            completed.stdout,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {value}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"remote stat returned malformed JSON for {path}") from exc
    if not isinstance(payload, dict) or set(payload) != REMOTE_STAT_KEYS:
        raise RuntimeError(f"remote stat returned malformed identity for {path}")
    if any(type(value) is not int or value < 0 for value in payload.values()):
        raise RuntimeError(f"remote stat returned invalid identity for {path}")
    if not stat.S_ISREG(payload["mode"]):
        raise RuntimeError(f"remote source is not a regular file: {path}")
    return payload


def rsync_file(
    wrapper: Path,
    remote: str,
    source: str,
    partial: Path,
) -> float:
    started = time.monotonic()
    completed = subprocess.run(
        [
            "rsync",
            "-a",
            "--partial",
            "--append-verify",
            "--protect-args",
            "--timeout=60",
            "-e",
            shlex.join([str(wrapper), *SSH_TRANSPORT_ARGS]),
            f"{remote}:{source}",
            str(partial),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"rsync failed for {source} with code {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    return elapsed


def exact_row(row: Any) -> dict[str, Any]:
    expected_keys = {
        "task_id",
        "global_step",
        "source_run_root",
        "source_checkpoint",
        "checkpoint_size_bytes",
        "checkpoint_sha256",
        "source_config",
        "config_size_bytes",
        "config_sha256",
        "semantic_config_sha256",
        "source_validation_result_sha256",
    }
    if not isinstance(row, dict) or set(row) != expected_keys:
        raise ValueError("candidate row keys are not canonical")
    task_id = str(row["task_id"])
    if SAFE_TASK_RE.fullmatch(task_id) is None:
        raise ValueError("invalid task_id")
    if type(row["global_step"]) is not int or row["global_step"] < 1:
        raise ValueError("global_step must be a positive integer")
    if row["global_step"] % 20_000:
        raise ValueError("candidate step is not a 20k boundary")
    for key in ("checkpoint_size_bytes", "config_size_bytes"):
        if type(row[key]) is not int or row[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in (
        "checkpoint_sha256",
        "config_sha256",
        "semantic_config_sha256",
        "source_validation_result_sha256",
    ):
        if SHA256_RE.fullmatch(str(row[key])) is None:
            raise ValueError(f"invalid {key}")
    run_root_text = str(row["source_run_root"])
    checkpoint_text = str(row["source_checkpoint"])
    config_text = str(row["source_config"])
    if SAFE_PATH_RE.fullmatch(run_root_text) is None:
        raise ValueError("unsafe remote path in source_run_root")
    if SAFE_PATH_RE.fullmatch(checkpoint_text) is None:
        raise ValueError("unsafe remote path in source_checkpoint")
    if SAFE_PATH_RE.fullmatch(config_text) is None:
        raise ValueError("unsafe remote path in source_config")
    run_root = PurePosixPath(run_root_text)
    checkpoint = PurePosixPath(checkpoint_text)
    config = PurePosixPath(config_text)
    if (
        str(run_root) != run_root_text
        or str(checkpoint) != checkpoint_text
        or str(config) != config_text
    ):
        raise ValueError("remote path is not canonical")
    match = CHECKPOINT_NAME_RE.fullmatch(checkpoint.name)
    if match is None or int(match.group(2)) != row["global_step"]:
        raise ValueError("checkpoint filename/global-step identity mismatch")
    if checkpoint.parent != run_root / "checkpoints":
        raise ValueError("checkpoint is outside the authorized source run")
    if config != run_root / ".hydra/config.yaml":
        raise ValueError("config is not the authorized adjacent run config")
    return row


def temporary_peer(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")


def ensure_proof_sidecar(path: Path, digest: str) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    encoded = f"{digest}  {path.name}\n".encode()
    if sidecar.is_symlink():
        raise RuntimeError(f"refusing symlink transfer sidecar: {sidecar}")
    if sidecar.exists():
        if not sidecar.is_file() or sidecar.read_bytes() != encoded:
            raise RuntimeError(f"existing transfer sidecar differs: {sidecar}")
        return
    temporary = temporary_peer(sidecar)
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, sidecar)
    except FileExistsError:
        if sidecar.is_symlink() or not sidecar.is_file() or sidecar.read_bytes() != encoded:
            raise RuntimeError(f"concurrent transfer sidecar differs: {sidecar}")
    finally:
        temporary.unlink(missing_ok=True)
    os.chmod(sidecar, 0o444)


def atomic_proof(path: Path, payload: dict[str, Any]) -> str:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    if path.is_symlink():
        raise RuntimeError(f"refusing symlink transfer proof: {path}")
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise RuntimeError(f"existing transfer proof differs: {path}")
    else:
        temporary = temporary_peer(path)
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
                raise RuntimeError(f"concurrent transfer proof differs: {path}")
        finally:
            temporary.unlink(missing_ok=True)
    os.chmod(path, 0o444)
    fsync_file(path)
    ensure_proof_sidecar(path, digest)
    fsync_directory(path.parent)
    return digest


def validate_completed_proof(
    path: Path,
    expected: dict[str, Any],
    checkpoint: Path,
    config: Path,
) -> str | None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if path.is_symlink() or sidecar.is_symlink():
        raise RuntimeError(f"refusing symlink transfer proof state for {path}")
    if not path.exists():
        if sidecar.exists():
            raise RuntimeError(f"transfer sidecar exists without proof: {sidecar}")
        return None
    if not path.is_file():
        raise RuntimeError(f"transfer proof is not a regular file: {path}")
    payload, digest = load_json_stable(path)
    canonical = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    if hashlib.sha256(canonical).hexdigest() != digest:
        raise RuntimeError(f"transfer proof is not canonical JSON: {path}")
    if not isinstance(payload, dict) or set(payload) != PROOF_KEYS:
        raise RuntimeError(f"transfer proof keys are not canonical: {path}")
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"transfer proof {key} mismatch: {path}")
    for prefix, expected_size in (
        ("source_checkpoint", expected["checkpoint_size_bytes"]),
        ("source_config", expected["config_size_bytes"]),
    ):
        before = payload[f"{prefix}_stat_before"]
        after = payload[f"{prefix}_stat_after"]
        if (
            not isinstance(before, dict)
            or set(before) != REMOTE_STAT_KEYS
            or before != after
            or before["size"] != expected_size
            or any(type(value) is not int or value < 0 for value in before.values())
            or not stat.S_ISREG(before["mode"])
        ):
            raise RuntimeError(f"transfer proof {prefix} stat mismatch: {path}")
    for prefix in ("checkpoint", "config"):
        action = payload[f"{prefix}_publication"]
        seconds = payload[f"{prefix}_transfer_seconds"]
        throughput = payload[f"{prefix}_effective_mib_per_second"]
        if action not in {"published", "reused", "reused_concurrent"}:
            raise RuntimeError(f"transfer proof {prefix} publication is invalid: {path}")
        if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
            raise RuntimeError(f"transfer proof {prefix} duration is invalid: {path}")
        if action == "reused":
            if seconds != 0 or throughput is not None:
                raise RuntimeError(f"reused {prefix} has transfer metrics: {path}")
        elif (
            seconds <= 0
            or not isinstance(throughput, (int, float))
            or not math.isfinite(throughput)
            or throughput <= 0
        ):
            raise RuntimeError(f"transferred {prefix} metrics are invalid: {path}")
    completed_at = payload["completed_at_unix"]
    if (
        not isinstance(completed_at, (int, float))
        or not math.isfinite(completed_at)
        or completed_at <= 0
    ):
        raise RuntimeError(f"transfer proof completion time is invalid: {path}")
    if not verified_existing(
        checkpoint, expected["checkpoint_sha256"], expected["checkpoint_size_bytes"]
    ):
        raise RuntimeError(f"transfer proof checkpoint is missing: {checkpoint}")
    if not verified_existing(
        config, expected["config_sha256"], expected["config_size_bytes"]
    ):
        raise RuntimeError(f"transfer proof config is missing: {config}")
    ensure_proof_sidecar(path, digest)
    os.chmod(path, 0o444)
    fsync_file(path)
    fsync_directory(path.parent)
    return digest


def manifest_addressed_proof_name(
    task_id: str, global_step: int, manifest_sha256: str
) -> str:
    if SAFE_TASK_RE.fullmatch(task_id) is None:
        raise ValueError("invalid task ID for transfer-proof name")
    if type(global_step) is not int or global_step < 1:
        raise ValueError("invalid global step for transfer-proof name")
    if SHA256_RE.fullmatch(manifest_sha256) is None:
        raise ValueError("invalid manifest SHA-256 for transfer-proof name")
    return f"{task_id}-step-{global_step}-manifest-{manifest_sha256}.json"


def validate_historical_manifest_binding(
    proof: dict[str, Any], authorized_root: Path
) -> None:
    manifest_text = proof.get("manifest")
    manifest_sha256 = proof.get("manifest_sha256")
    if not isinstance(manifest_text, str) or not manifest_text:
        raise RuntimeError("historical stage proof manifest path is invalid")
    if not isinstance(manifest_sha256, str) or SHA256_RE.fullmatch(manifest_sha256) is None:
        raise RuntimeError("historical stage proof manifest SHA-256 is invalid")
    manifest_path = Path(manifest_text)
    if not manifest_path.is_absolute() or manifest_path.is_symlink():
        raise RuntimeError("historical stage manifest is not a canonical file")
    try:
        canonical_manifest = manifest_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError("historical stage manifest is missing") from exc
    if canonical_manifest != manifest_path or not manifest_path.is_file():
        raise RuntimeError("historical stage manifest is not a canonical file")
    manifest, observed_sha256 = load_json_stable(manifest_path)
    if observed_sha256 != manifest_sha256:
        raise RuntimeError("historical stage manifest SHA-256 mismatch")
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version",
        "access_selection_sha256",
        "destination_root",
        "rows",
    }:
        raise RuntimeError("historical stage manifest keys are not canonical")
    if manifest["schema_version"] != 2:
        raise RuntimeError("historical stage manifest schema is invalid")
    if manifest["access_selection_sha256"] != proof.get("access_selection_sha256"):
        raise RuntimeError("historical stage manifest access identity mismatch")
    if manifest["destination_root"] != str(authorized_root / "staging"):
        raise RuntimeError("historical stage manifest destination root mismatch")
    rows = manifest["rows"]
    if not isinstance(rows, list):
        raise RuntimeError("historical stage manifest rows are invalid")
    matching = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("task_id") == proof.get("task_id")
    ]
    if len(matching) != 1:
        raise RuntimeError("historical stage manifest has no unique proof row")
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
        "source_validation_result_sha256": proof.get(
            "source_validation_result_sha256"
        ),
    }
    if matching[0] != expected_row:
        raise RuntimeError("historical stage proof differs from its manifest row")


def validate_historical_collision(
    path: Path,
    expected: dict[str, Any],
    checkpoint: Path,
    config: Path,
    authorized_root: Path,
) -> str:
    payload, _ = load_json_stable(path)
    if not isinstance(payload, dict) or set(payload) != PROOF_KEYS:
        raise RuntimeError(f"historical transfer proof keys are not canonical: {path}")
    if (
        payload.get("manifest_sha256") == expected["manifest_sha256"]
        or payload.get("source_validation_result_sha256")
        == expected["source_validation_result_sha256"]
    ):
        raise RuntimeError(
            "canonical transfer-proof collision is not a distinct manifest/result"
        )
    historical_expected = dict(expected)
    for key in COLLISION_VARIANT_KEYS:
        historical_expected[key] = payload.get(key)
    for key in ("manifest_sha256", "source_validation_result_sha256", "stager_sha256"):
        if not isinstance(historical_expected[key], str) or SHA256_RE.fullmatch(
            historical_expected[key]
        ) is None:
            raise RuntimeError(f"historical transfer proof {key} is invalid")
    stager_text = historical_expected["stager"]
    if not isinstance(stager_text, str) or not stager_text:
        raise RuntimeError("historical transfer proof stager path is invalid")
    historical_stager = Path(stager_text)
    if not historical_stager.is_absolute() or historical_stager.is_symlink():
        raise RuntimeError("historical candidate stager is not a canonical file")
    try:
        canonical_stager = historical_stager.resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError("historical candidate stager is missing") from exc
    if canonical_stager != historical_stager or not historical_stager.is_file():
        raise RuntimeError("historical candidate stager is not a canonical file")
    if stable_sha256(historical_stager) != historical_expected["stager_sha256"]:
        raise RuntimeError("historical candidate stager SHA-256 mismatch")
    digest = validate_completed_proof(
        path, historical_expected, checkpoint, config
    )
    if digest is None:
        raise RuntimeError("historical transfer proof disappeared")
    validate_historical_manifest_binding(payload, authorized_root)
    return digest


def select_completed_proof(
    legacy_path: Path,
    expected: dict[str, Any],
    checkpoint: Path,
    config: Path,
    authorized_root: Path,
) -> tuple[Path, str | None]:
    if not legacy_path.exists() and not legacy_path.is_symlink():
        return legacy_path, validate_completed_proof(
            legacy_path, expected, checkpoint, config
        )
    try:
        digest = validate_completed_proof(
            legacy_path, expected, checkpoint, config
        )
    except RuntimeError:
        validate_historical_collision(
            legacy_path, expected, checkpoint, config, authorized_root
        )
        addressed_path = legacy_path.with_name(
            manifest_addressed_proof_name(
                expected["task_id"],
                expected["global_step"],
                expected["manifest_sha256"],
            )
        )
        return addressed_path, validate_completed_proof(
            addressed_path, expected, checkpoint, config
        )
    return legacy_path, digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--access-selection", type=Path, required=True)
    parser.add_argument("--authorized-root", type=Path, required=True)
    parser.add_argument("--ssh-wrapper", type=Path, required=True)
    parser.add_argument("--proof-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve(strict=True)
    access_selection_path = args.access_selection.resolve(strict=True)
    wrapper = args.ssh_wrapper.resolve(strict=True)
    stager = Path(__file__).resolve(strict=True)
    if not wrapper.is_file() or not os.access(wrapper, os.X_OK):
        raise SystemExit("SSH wrapper is not executable")
    manifest, manifest_sha256 = load_json_stable(manifest_path)
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version", "access_selection_sha256", "destination_root", "rows"
    }:
        raise SystemExit("manifest keys are not canonical")
    if manifest["schema_version"] != 2:
        raise SystemExit("unsupported manifest schema")
    expected_access_sha256 = str(manifest["access_selection_sha256"])
    if SHA256_RE.fullmatch(expected_access_sha256) is None:
        raise SystemExit("manifest access-selection SHA-256 is invalid")
    access_selection, access_selection_sha256 = load_json_stable(
        access_selection_path
    )
    if access_selection_sha256 != expected_access_sha256:
        raise SystemExit("access-selection artifact SHA-256 mismatch")
    access_selection = exact_access_selection(access_selection)
    remote = str(access_selection["selected_remote"])
    if SAFE_REMOTE_RE.fullmatch(remote) is None:
        raise SystemExit("unsafe remote identity")
    wrapper_sha256 = stable_sha256(wrapper)
    stager_sha256 = stable_sha256(stager)
    authorized_root = canonical_existing_root(args.authorized_root)
    destination_root = Path(str(manifest["destination_root"]))
    if destination_root != authorized_root / "staging":
        raise SystemExit("destination_root is not the authorized ROOT/staging path")
    proof_dir = args.proof_dir
    if proof_dir != authorized_root / "provenance/candidate-transfers":
        raise SystemExit("proof-dir is not the authorized ROOT provenance path")
    if not isinstance(manifest["rows"], list) or not manifest["rows"]:
        raise SystemExit("manifest rows must be a non-empty list")
    rows = [exact_row(row) for row in manifest["rows"]]
    if len(rows) != len({row["task_id"] for row in rows}):
        raise SystemExit("duplicate task_id in manifest")
    ensure_directory_below(authorized_root, destination_root)
    ensure_directory_below(authorized_root, proof_dir)

    immutable_inputs = (
        (manifest_path, manifest_sha256, "manifest"),
        (access_selection_path, access_selection_sha256, "access-selection artifact"),
        (wrapper, wrapper_sha256, "SSH wrapper"),
        (stager, stager_sha256, "stager"),
    )

    def verify_immutable_inputs() -> None:
        for path, expected_sha, label in immutable_inputs:
            if stable_sha256(path) != expected_sha:
                raise RuntimeError(f"{label} changed during staging: {path}")

    for row in rows:
        task_id = row["task_id"]
        step = row["global_step"]
        destination = destination_root / task_id / "candidates" / f"step-{step}"
        checkpoint_dir = destination / "checkpoints"
        config_dir = destination / ".hydra"
        ensure_directory_below(authorized_root, checkpoint_dir)
        ensure_directory_below(authorized_root, config_dir)
        checkpoint = checkpoint_dir / Path(row["source_checkpoint"]).name
        config = config_dir / "config.yaml"
        checkpoint_partial = checkpoint.with_suffix(checkpoint.suffix + ".part")
        config_partial = config.with_suffix(config.suffix + ".part")
        legacy_proof_path = proof_dir / f"{task_id}-step-{step}.json"
        proof_expected = {
            "schema_version": 2,
            "status": "STAGED_SHA256_VERIFIED_NO_OVERWRITE",
            "task_id": task_id,
            "global_step": step,
            "source_run_root": row["source_run_root"],
            "remote": remote,
            "access_selection": access_selection,
            "access_selection_path": str(access_selection_path),
            "access_selection_sha256": access_selection_sha256,
            "source_checkpoint": row["source_checkpoint"],
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_size_bytes": row["checkpoint_size_bytes"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "source_config": row["source_config"],
            "config": str(config.resolve()),
            "config_size_bytes": row["config_size_bytes"],
            "config_sha256": row["config_sha256"],
            "semantic_config_sha256": row["semantic_config_sha256"],
            "source_validation_result_sha256": row[
                "source_validation_result_sha256"
            ],
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "ssh_wrapper": str(wrapper),
            "ssh_wrapper_sha256": wrapper_sha256,
            "stager": str(stager),
            "stager_sha256": stager_sha256,
            "authorized_root": str(authorized_root),
        }
        with exclusive_row_lock(destination / ".stage.lock"):
            verify_immutable_inputs()
            proof_path, digest = select_completed_proof(
                legacy_proof_path,
                proof_expected,
                checkpoint,
                config,
                authorized_root,
            )
            if digest is not None:
                print(
                    json.dumps(
                        {
                            "status": "already_staged",
                            "task_id": task_id,
                            "checkpoint": str(checkpoint.resolve()),
                            "checkpoint_sha256": row["checkpoint_sha256"],
                            "proof": str(proof_path),
                            "proof_sha256": digest,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                continue

            for partial in (checkpoint_partial, config_partial):
                if partial.is_symlink() or (partial.exists() and not partial.is_file()):
                    raise RuntimeError(f"refusing unsafe partial artifact: {partial}")

            checkpoint_source_before = remote_stat(
                wrapper, remote, row["source_checkpoint"]
            )
            config_source_before = remote_stat(wrapper, remote, row["source_config"])
            if checkpoint_source_before["size"] != row["checkpoint_size_bytes"]:
                raise RuntimeError(f"source checkpoint size changed for {task_id}")
            if config_source_before["size"] != row["config_size_bytes"]:
                raise RuntimeError(f"source config size changed for {task_id}")

            if verified_existing(
                checkpoint, row["checkpoint_sha256"], row["checkpoint_size_bytes"]
            ):
                checkpoint_seconds = 0.0
                checkpoint_publication = publish_no_overwrite(
                    checkpoint_partial,
                    checkpoint,
                    row["checkpoint_sha256"],
                    row["checkpoint_size_bytes"],
                )
            else:
                checkpoint_seconds = rsync_file(
                    wrapper, remote, row["source_checkpoint"], checkpoint_partial
                )
                checkpoint_publication = None
            if verified_existing(config, row["config_sha256"], row["config_size_bytes"]):
                config_seconds = 0.0
                config_publication = publish_no_overwrite(
                    config_partial,
                    config,
                    row["config_sha256"],
                    row["config_size_bytes"],
                )
            else:
                config_seconds = rsync_file(
                    wrapper, remote, row["source_config"], config_partial
                )
                config_publication = None

            checkpoint_source_after = remote_stat(
                wrapper, remote, row["source_checkpoint"]
            )
            config_source_after = remote_stat(wrapper, remote, row["source_config"])
            if checkpoint_source_before != checkpoint_source_after:
                raise RuntimeError(f"source checkpoint changed during transfer for {task_id}")
            if config_source_before != config_source_after:
                raise RuntimeError(f"source config changed during transfer for {task_id}")
            verify_immutable_inputs()

            if checkpoint_publication is None:
                checkpoint_publication = publish_no_overwrite(
                    checkpoint_partial,
                    checkpoint,
                    row["checkpoint_sha256"],
                    row["checkpoint_size_bytes"],
                )
            if config_publication is None:
                config_publication = publish_no_overwrite(
                    config_partial,
                    config,
                    row["config_sha256"],
                    row["config_size_bytes"],
                )

            checkpoint_throughput = (
                None
                if checkpoint_publication == "reused"
                else row["checkpoint_size_bytes"]
                / (1024 * 1024)
                / checkpoint_seconds
            )
            config_throughput = (
                None
                if config_publication == "reused"
                else row["config_size_bytes"] / (1024 * 1024) / config_seconds
            )
            proof = {
                **proof_expected,
                "source_checkpoint_stat_before": checkpoint_source_before,
                "source_checkpoint_stat_after": checkpoint_source_after,
                "checkpoint_publication": checkpoint_publication,
                "source_config_stat_before": config_source_before,
                "source_config_stat_after": config_source_after,
                "config_publication": config_publication,
                "checkpoint_transfer_seconds": checkpoint_seconds,
                "checkpoint_effective_mib_per_second": checkpoint_throughput,
                "config_transfer_seconds": config_seconds,
                "config_effective_mib_per_second": config_throughput,
                "completed_at_unix": time.time(),
            }
            if set(proof) != PROOF_KEYS:
                raise RuntimeError("internal transfer proof schema mismatch")
            digest = atomic_proof(proof_path, proof)
            print(
                json.dumps(
                    {
                        "status": "passed",
                        "task_id": task_id,
                        "checkpoint": str(checkpoint.resolve()),
                        "checkpoint_sha256": row["checkpoint_sha256"],
                        "proof": str(proof_path),
                        "proof_sha256": digest,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    verify_immutable_inputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
