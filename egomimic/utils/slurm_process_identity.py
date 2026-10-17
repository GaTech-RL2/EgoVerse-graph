"""Authenticate one direct-exec trainer in the runner's Linux PID namespace.

This stdlib-only boundary is shared by the generic runner and the Lightning
save callback. It never discovers PIDs, reads process environments, or signals
Slurm job steps. Different namespaces/nodes/cgroups fail closed.
"""

import hashlib
import json
import os
from pathlib import Path
import signal
import socket


def source_identity():
    root = Path(__file__).resolve().parents[2]
    files = ("egomimic/trainHydra.py", "egomimic/utils/slurm_requeue.py",
             "egomimic/utils/slurm_process_identity.py",
             "scripts/ice/ice_requeue_runner.py")
    return {"root": str(root), "sha256": {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}}


def process_identity(pid):
    if type(pid) is not int or pid < 1:
        raise ValueError("Invalid trainer PID")
    proc = Path("/proc") / str(pid)
    status = (proc / "status").read_text().splitlines()
    uid = next(line.split()[1:] for line in status if line.startswith("Uid:"))
    # /proc/<pid>/stat comm may contain spaces and parentheses.
    fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "uid": [int(value) for value in uid],
            "starttime": int(fields[19]), "node": socket.gethostname(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "pid_namespace": os.readlink(proc / "ns/pid"),
            "cgroup": (proc / "cgroup").read_text(),
            "exe": str((proc / "exe").resolve(strict=True)),
            "cwd": str((proc / "cwd").resolve(strict=True))}


def ready_record(nonce, job_id, restart):
    return {"schema": "egoverse-trainer-pid-ready-v1", "nonce": nonce,
            "job_id": job_id, "restart_count": restart,
            "handler_installed": True, "process": process_identity(os.getpid()),
            "source": source_identity()}


def signal_ready_trainer(path, *, pid, nonce, job_id, restart, parent_identity,
                         expected_source, expected_process, sig=signal.SIGUSR2):
    """Authenticate the known child and signal through a PID-reuse-safe fd."""
    if sig != signal.SIGUSR2:
        raise ValueError("Trainer PID transport only permits save-only SIGUSR2")
    path = Path(path)
    if path.is_symlink() or path.stat().st_uid != os.getuid():
        raise ValueError("Untrusted trainer readiness file ownership")
    record = json.loads(path.read_text())
    if (record.get("schema") != "egoverse-trainer-pid-ready-v1"
            or record.get("nonce") != nonce or record.get("job_id") != job_id
            or record.get("restart_count") != restart
            or record.get("handler_installed") is not True
            or record.get("source") != expected_source):
        raise ValueError("Trainer readiness authority mismatch")
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("Linux pidfd support is required; no unsafe PID fallback")
    descriptor = os.pidfd_open(pid)
    try:
        actual = process_identity(pid)
        if actual != record.get("process") or actual != expected_process:
            raise ValueError("Trainer process identity changed or was forged")
        for key in ("uid", "node", "boot_id", "pid_namespace", "cgroup", "exe", "cwd"):
            if actual[key] != parent_identity[key]:
                raise ValueError(f"Trainer differs from runner {key}")
        if actual["uid"] != [os.getuid()] * 4:
            raise ValueError("Trainer UID differs from current user")
        if actual["cwd"] != expected_source["root"]:
            raise ValueError("Trainer working directory differs from pinned source")
        if source_identity() != expected_source:
            raise ValueError("Signal-delivery source changed")
        signal.pidfd_send_signal(descriptor, sig)
        return {"transport": "trainer-pid", "pid": pid,
                "starttime": actual["starttime"], "signal": sig.name}
    finally:
        os.close(descriptor)
