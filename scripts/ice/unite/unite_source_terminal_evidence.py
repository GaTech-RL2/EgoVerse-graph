#!/usr/bin/env python3
"""Collect signed, fresh Skynet terminal/absent evidence from an ICE CPU job.

This collector is read-only.  It invokes only the exact allowlisted ``sacct``
and ``squeue`` queries through a SHA-pinned ICE-to-Skynet SSH wrapper.  It never
submits, requeues, or cancels a job.  The output is create-only and is consumed
by ``unite_cutover_record_writer.py authorize``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import types
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

SHA256_RE = re.compile(r"[0-9a-f]{64}")


def remote_executable(value: str, expected_name: str) -> str:
    path = PurePosixPath(value)
    if not value.startswith("/") or str(path) != value or path.name != expected_name:
        raise argparse.ArgumentTypeError(
            f"must be a canonical absolute remote {expected_name} path"
        )
    return value


def required_environment(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"missing required environment variable {name}")
    return value


def positive_cli_integer(value: str) -> int:
    if re.fullmatch(r"[1-9][0-9]*", value) is None:
        raise argparse.ArgumentTypeError("must be a positive base-10 integer")
    return int(value)


def stable_bytes(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file() or path.resolve(strict=True) != path:
        raise RuntimeError(f"{label} must be a canonical non-symlink file")
    before = os.lstat(path)
    raw = path.read_bytes()
    after = os.lstat(path)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise RuntimeError(f"{label} changed while being read")
    return raw


def load_pinned_writer(path: Path, expected_sha256: str) -> Any:
    if SHA256_RE.fullmatch(expected_sha256) is None:
        raise RuntimeError("record-writer SHA-256 is invalid")
    if not path.is_absolute() or path.is_symlink():
        raise RuntimeError("record-writer path must be absolute and non-symlink")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise RuntimeError("record-writer path must be canonical")
    path = resolved
    raw = stable_bytes(path, "cutover record writer")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RuntimeError("cutover record writer SHA-256 mismatch")
    module = types.ModuleType("_pinned_unite_cutover_record_writer")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def checked_sha(writer: Any, path: Path, expected: str, label: str) -> tuple[Path, str]:
    path = writer.canonical_absolute_file(path, label)
    expected = writer.lowercase_sha256(expected, f"expected {label} SHA-256")
    actual, _ = writer.stable_sha256(path, label)
    if actual != expected:
        raise RuntimeError(f"{label} SHA-256 mismatch")
    return path, actual


def run_remote(
    wrapper: Path, remote: str, argv: list[str], writer: Any
) -> dict[str, Any]:
    transport_argv = [
        str(wrapper),
        remote,
        writer.remote_identity_command(argv),
    ]
    started = time.time_ns()
    completed = subprocess.run(
        transport_argv,
        check=False,
        capture_output=True,
        text=False,
        timeout=25,
    )
    finished = time.time_ns()
    try:
        stdout = completed.stdout.decode("utf-8")
        stderr = completed.stderr.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("remote Slurm query output is not UTF-8") from exc
    return {
        "argv": argv,
        "transport_argv": transport_argv,
        "returncode": completed.returncode,
        "stdout": stdout,
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "stderr": stderr,
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        "started_at_unix_ns": started,
        "completed_at_unix_ns": finished,
    }


def build_payload(args: argparse.Namespace, writer: Any) -> dict[str, Any]:
    source_job_id = writer.positive_integer(args.source_job_id, "source job ID")
    successor_job_id = writer.positive_integer(
        args.successor_job_id, "successor job ID"
    )
    wandb_id = writer.nonempty_string(args.wandb_id, "W&B ID")
    candidate_sha = writer.lowercase_sha256(
        args.candidate_record_sha256, "candidate record SHA-256"
    )
    allocation_sha = writer.lowercase_sha256(
        args.allocation_record_sha256, "allocation record SHA-256"
    )
    ready_sha = writer.lowercase_sha256(
        args.ready_to_cancel_record_sha256, "ready-to-cancel record SHA-256"
    )
    wrapper, wrapper_sha = checked_sha(
        writer,
        args.ssh_wrapper, args.ssh_wrapper_sha256, "SSH wrapper"
    )
    collector = Path(__file__).resolve(strict=True)
    collector_sha, _ = writer.stable_sha256(collector, "source terminal collector")
    selection_path = writer.canonical_absolute_file(
        args.access_selection, "access selection"
    )
    selection, selection_sha = writer.unsigned_exact_json(
        selection_path,
        args.access_selection_sha256,
        "access selection",
    )
    selection = writer.validate_access_selection(selection)
    remote = selection["selected_remote"]
    probe = (
        selection["canonical_probe"]
        if remote == selection["canonical_remote"]
        else selection["fallback_probe"]
    )
    if not isinstance(probe, dict) or probe.get("returncode") != 0:
        raise RuntimeError("selected remote has no successful access probe")

    expected_context = selection["probe_context"]
    if (
        required_environment("SLURM_JOB_ACCOUNT"),
        required_environment("SLURM_JOB_PARTITION"),
        required_environment("SLURM_JOB_QOS"),
    ) != (
        expected_context["account"],
        expected_context["partition"],
        expected_context["qos"],
    ):
        raise RuntimeError("terminal collector is not in the required ICE CPU allocation")
    slurm_job_id = required_environment("SLURM_JOB_ID")
    if re.fullmatch(r"[1-9][0-9]*", slurm_job_id) is None:
        raise RuntimeError("terminal collector Slurm job ID is invalid")
    slurm_node = required_environment("SLURMD_NODENAME")
    if writer.SAFE_NODE_RE.fullmatch(slurm_node) is None:
        raise RuntimeError("terminal collector Slurm node is invalid")
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("terminal collector unexpectedly received GPU visibility")

    job = str(source_job_id)
    sacct_argv = [
        args.sacct,
        "--noheader",
        "--parsable2",
        "--allocations",
        f"--jobs={job}",
        "--format=JobIDRaw,State,ExitCode,End",
    ]
    expected_username = str(probe["username"])
    squeue_argv = [
        args.squeue,
        "--noheader",
        "--array",
        f"--user={expected_username}",
        "--format=%i|%T",
    ]
    sacct = run_remote(wrapper, remote, sacct_argv, writer)
    squeue = run_remote(wrapper, remote, squeue_argv, writer)
    observed = time.time_ns()

    # Parse before publication.  The writer repeats every check from raw bytes.
    if sacct["returncode"] != 0 or sacct["stderr"] != "":
        raise RuntimeError("remote sacct query failed or emitted stderr")
    if squeue["returncode"] != 0 or squeue["stderr"] != "":
        raise RuntimeError("remote squeue query failed or emitted stderr")
    expected_hostname = str(probe["hostname"])
    sacct_stdout = writer.split_remote_identity_stdout(
        sacct["stdout"],
        expected_hostname=expected_hostname,
        expected_username=expected_username,
        label="sacct",
    )
    squeue_stdout = writer.split_remote_identity_stdout(
        squeue["stdout"],
        expected_hostname=expected_hostname,
        expected_username=expected_username,
        label="squeue",
    )
    lines = sacct_stdout.splitlines()
    if len(lines) != 1:
        raise RuntimeError("remote sacct query did not return one allocation row")
    fields = lines[0].split("|")
    if len(fields) != 4 or fields[0] != job:
        raise RuntimeError("remote sacct row does not name the exact source allocation")
    state = writer.normalize_sacct_state(fields[1])
    if writer.EXIT_CODE_RE.fullmatch(fields[2]) is None:
        raise RuntimeError("remote sacct exit code is invalid")
    if writer.END_TIME_RE.fullmatch(fields[3]) is None:
        raise RuntimeError("remote sacct end time is invalid")
    for line in squeue_stdout.splitlines():
        queue_fields = [field.strip(" ") for field in line.split("|")]
        if (
            len(queue_fields) != 2
            or writer.SQUEUE_JOB_ID_RE.fullmatch(queue_fields[0]) is None
            or writer.SQUEUE_STATE_RE.fullmatch(queue_fields[1]) is None
        ):
            raise RuntimeError("remote squeue snapshot contains a malformed row")
        if queue_fields[0] == job:
            raise RuntimeError("source allocation is still present in squeue")

    payload = {
        "schema_version": 1,
        "status": "SOURCE_TERMINAL_AND_ABSENT",
        "source_cluster": "skynet",
        "source_job_id": source_job_id,
        "successor_job_id": successor_job_id,
        "wandb_id": wandb_id,
        "candidate_record_sha256": candidate_sha,
        "allocation_record_sha256": allocation_sha,
        "ready_to_cancel_record_sha256": ready_sha,
        "observed_at_unix_ns": observed,
        "terminal_state": {
            "state": state,
            "exit_code": fields[2],
            "end_time": fields[3],
        },
        "queue": {"job_id": source_job_id, "present": False, "state": None},
        "command_evidence": {"sacct": sacct, "squeue": squeue},
        "collector": {
            "path": str(collector),
            "sha256": collector_sha,
            "record_writer": str(args.record_writer),
            "record_writer_sha256": args.record_writer_sha256,
            "ssh_wrapper": str(wrapper),
            "ssh_wrapper_sha256": wrapper_sha,
            "access_selection": str(selection_path),
            "access_selection_sha256": selection_sha,
            "selected_remote": remote,
            "remote_hostname": expected_hostname,
            "remote_username": expected_username,
            "execution_cluster": "PACE ICE",
            "slurm_job_id": slurm_job_id,
            "slurm_node": slurm_node,
        },
    }
    if set(payload) != writer.SOURCE_EVIDENCE_KEYS:
        raise AssertionError("internal source terminal evidence schema drift")
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--record-writer", type=Path, required=True)
    result.add_argument("--record-writer-sha256", required=True)
    result.add_argument("--ssh-wrapper", type=Path, required=True)
    result.add_argument("--ssh-wrapper-sha256", required=True)
    result.add_argument("--access-selection", type=Path, required=True)
    result.add_argument("--access-selection-sha256", required=True)
    result.add_argument("--source-job-id", type=positive_cli_integer, required=True)
    result.add_argument("--successor-job-id", type=positive_cli_integer, required=True)
    result.add_argument("--wandb-id", required=True)
    result.add_argument("--candidate-record-sha256", required=True)
    result.add_argument("--allocation-record-sha256", required=True)
    result.add_argument("--ready-to-cancel-record-sha256", required=True)
    result.add_argument(
        "--sacct",
        type=lambda value: remote_executable(value, "sacct"),
        required=True,
    )
    result.add_argument(
        "--squeue",
        type=lambda value: remote_executable(value, "squeue"),
        required=True,
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    writer = load_pinned_writer(args.record_writer, args.record_writer_sha256)
    args.record_writer = Path(writer.__file__).resolve(strict=True)
    payload = build_payload(args, writer)
    digest = writer.publish_signed_no_overwrite(args.output, payload)
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
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"source terminal evidence refused: {exc}", file=sys.stderr)
        raise SystemExit(1)
