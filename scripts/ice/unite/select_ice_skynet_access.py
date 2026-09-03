#!/usr/bin/env python3
"""Select a canonical remote, or a proven fallback, from an ICE CPU node."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

SHA256_RE = re.compile(r"[0-9a-f]{64}")
NODE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
REMOTE_RE = re.compile(r"[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+")
SSH_TRANSPORT_ARGS = (
    "-T",
    "-o", "ClearAllForwardings=yes",
    "-o", "ForwardAgent=no",
    "-o", "ConnectTimeout=15",
    "-o", "ConnectionAttempts=1",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=2",
)


def required_environment(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stable_sha256(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    def identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
    if identity(before) != identity(after):
        raise RuntimeError(f"file changed while hashing: {path}")
    return digest.hexdigest()


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
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def probe(
    wrapper: Path,
    remote: str,
    attempts: int,
    *,
    expected_hostname: str,
    expected_user: str,
) -> dict[str, object]:
    attempted_at = utc_now()
    stderr_chunks: list[bytes] = []
    last_returncode = 255
    hostname: str | None = None
    username: str | None = None
    used = 0
    for index in range(attempts):
        used = index + 1
        hostname = username = None
        try:
            completed = subprocess.run(
                [
                    str(wrapper),
                    *SSH_TRANSPORT_ARGS,
                    remote,
                    "hostname -f; id -un",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=45,
            )
        except subprocess.TimeoutExpired as exc:
            timeout_stderr = exc.stderr or b""
            if isinstance(timeout_stderr, str):
                timeout_stderr = timeout_stderr.encode("utf-8", errors="replace")
            stderr_chunks.append(timeout_stderr + b"\nprobe_timeout\n")
            last_returncode = 124
            if index + 1 < attempts:
                time.sleep(2)
            continue
        stderr_chunks.append(completed.stderr)
        last_returncode = completed.returncode
        try:
            lines = completed.stdout.decode("utf-8").splitlines()
        except UnicodeDecodeError:
            lines = []
        if len(lines) == 2:
            hostname, username = lines
        else:
            hostname = username = None
        if (
            completed.returncode == 0
            and hostname == expected_hostname
            and username == expected_user
        ):
            last_returncode = 0
            break
        if completed.returncode == 0:
            # A successful SSH transport with a wrong remote identity is not a
            # successful access probe and must not silently select that host.
            last_returncode = 74
        if index + 1 < attempts:
            time.sleep(2)
    return {
        "remote": remote,
        "attempted_at_utc": attempted_at,
        "attempt_count": used,
        "returncode": max(0, min(int(last_returncode), 255)),
        "hostname": hostname,
        "username": username,
        "stderr_sha256": hashlib.sha256(b"".join(stderr_chunks)).hexdigest(),
    }


def publish_no_overwrite(
    path: Path, payload: dict[str, object], authorized_root: Path
) -> str:
    ensure_directory_below(authorized_root, path.parent)
    if path.is_symlink():
        raise RuntimeError(f"refusing symlink access-selection output: {path}")
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise RuntimeError(f"refusing to overwrite access selection: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    os.chmod(path, 0o444)
    fsync_file(path)
    sidecar = Path(str(path) + ".sha256")
    sidecar_text = f"{digest}  {path.name}\n"
    sidecar_temporary = sidecar.with_name(
        f".{sidecar.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    with sidecar_temporary.open("x", encoding="utf-8") as stream:
        stream.write(sidecar_text)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(sidecar_temporary, sidecar)
    except FileExistsError as exc:
        raise RuntimeError(f"refusing to overwrite access-selection sidecar: {sidecar}") from exc
    finally:
        sidecar_temporary.unlink(missing_ok=True)
    os.chmod(sidecar, 0o444)
    fsync_file(sidecar)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-wrapper", type=Path, required=True)
    parser.add_argument("--ssh-wrapper-sha256", required=True)
    parser.add_argument("--authorized-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canonical-remote", required=True)
    parser.add_argument("--canonical-hostname", required=True)
    parser.add_argument("--fallback-remote", required=True)
    parser.add_argument("--fallback-hostname", required=True)
    parser.add_argument("--expected-remote-user", required=True)
    parser.add_argument("--expected-account", required=True)
    parser.add_argument("--expected-partition", default="ice-cpu")
    parser.add_argument("--expected-qos", required=True)
    args = parser.parse_args()

    for label, remote in (
        ("canonical", args.canonical_remote),
        ("fallback", args.fallback_remote),
    ):
        if REMOTE_RE.fullmatch(remote) is None:
            raise SystemExit(f"{label} remote is not a safe user@host identity")
    if args.canonical_remote == args.fallback_remote:
        raise SystemExit("canonical and fallback remotes must differ")
    if NODE_RE.fullmatch(args.canonical_hostname) is None or NODE_RE.fullmatch(
        args.fallback_hostname
    ) is None:
        raise SystemExit("expected remote hostnames are invalid")
    if not args.expected_remote_user or "@" in args.expected_remote_user:
        raise SystemExit("expected remote user is invalid")

    wrapper = args.ssh_wrapper.resolve(strict=True)
    output = args.output
    authorized_root = canonical_existing_root(args.authorized_root)
    try:
        output.relative_to(authorized_root / "provenance")
    except ValueError as exc:
        raise SystemExit("output must be below authorized ROOT/provenance") from exc
    if output.suffix != ".json" or output.name.startswith("."):
        raise SystemExit("output must be a visible JSON artifact")
    ensure_directory_below(authorized_root, output.parent)
    sidecar = Path(str(output) + ".sha256")
    if output.exists() or output.is_symlink() or sidecar.exists() or sidecar.is_symlink():
        raise SystemExit("access-selection output or sidecar already exists")
    if not wrapper.is_file() or not os.access(wrapper, os.X_OK):
        raise SystemExit("SSH wrapper must be an executable file")
    if SHA256_RE.fullmatch(args.ssh_wrapper_sha256) is None:
        raise SystemExit("invalid expected SSH-wrapper SHA-256")
    wrapper_sha256 = stable_sha256(wrapper)
    if wrapper_sha256 != args.ssh_wrapper_sha256:
        raise SystemExit("SSH-wrapper SHA-256 mismatch")

    job_id = required_environment("SLURM_JOB_ID")
    node = required_environment("SLURMD_NODENAME")
    if not job_id.isdigit() or int(job_id) < 1 or NODE_RE.fullmatch(node) is None:
        raise RuntimeError("invalid Slurm job/node identity")
    expected_allocation = (
        args.expected_account,
        args.expected_partition,
        args.expected_qos,
    )
    if (
        required_environment("SLURM_JOB_ACCOUNT"),
        required_environment("SLURM_JOB_PARTITION"),
        required_environment("SLURM_JOB_QOS"),
    ) != expected_allocation:
        raise RuntimeError("access probe is not running in the required ICE CPU allocation")
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("access probe unexpectedly received GPU visibility")

    canonical = probe(
        wrapper,
        args.canonical_remote,
        attempts=2,
        expected_hostname=args.canonical_hostname,
        expected_user=args.expected_remote_user,
    )
    fallback: dict[str, object] | None = None
    if canonical["returncode"] == 0:
        selected = args.canonical_remote
        reason = "canonical_probe_succeeded"
    else:
        fallback = probe(
            wrapper,
            args.fallback_remote,
            attempts=1,
            expected_hostname=args.fallback_hostname,
            expected_user=args.expected_remote_user,
        )
        if fallback["returncode"] != 0:
            raise RuntimeError("neither sky1 nor sky2 is reachable from the ICE CPU node")
        selected = args.fallback_remote
        reason = "canonical_probe_failed_fallback_succeeded"
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "READY",
        "probe_context": {
            "cluster": "PACE ICE",
            "slurm_job_id": job_id,
            "slurm_node": node,
            "account": args.expected_account,
            "partition": args.expected_partition,
            "qos": args.expected_qos,
        },
        "canonical_remote": args.canonical_remote,
        "fallback_remote": args.fallback_remote,
        "canonical_probe": canonical,
        "fallback_probe": fallback,
        "selected_remote": selected,
        "selection_reason": reason,
        "completed_at_utc": utc_now(),
    }
    if stable_sha256(wrapper) != wrapper_sha256:
        raise RuntimeError("SSH wrapper changed during access selection")
    digest = publish_no_overwrite(output, payload, authorized_root)
    print(json.dumps({"status": "READY", "path": str(output), "sha256": digest, "selected_remote": selected}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
