#!/usr/bin/env python3
"""Single-rank save-only signal adapter; never signal a Slurm step or supervisor."""

from __future__ import annotations

import ctypes
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import time


def _linux_syscall(number, *args):
    assert (
        sys.platform == "linux" and os.uname().machine == "x86_64"
    ), "Only verified Linux x86_64 ABI supported"
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    value = libc.syscall(ctypes.c_long(number), *args)
    if value < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return value


def open_pidfd(pid):
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    return _linux_syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))


def send_pidfd(fd, sig):
    if hasattr(signal, "pidfd_send_signal"):
        return signal.pidfd_send_signal(fd, sig)
    return _linux_syscall(
        424,
        ctypes.c_int(fd),
        ctypes.c_int(sig),
        ctypes.c_void_p(None),
        ctypes.c_uint(0),
    )


def candidate(
    pid,
    *,
    repo,
    python,
    uid,
    namespace,
    job,
    proc=pathlib.Path("/proc"),
    native_steps=None,
):
    assert pid > 0
    p = proc / str(pid)
    status = dict(
        line.split(":", 1)
        for line in (p / "status").read_text().splitlines()
        if ":" in line
    )
    assert [int(x) for x in status["Uid"].split()][:2] == [uid, uid]
    assert (p / "ns/pid").stat().st_ino == namespace
    cgroup = (p / "cgroup").read_text()
    if re.search(r"(?:^|/)job_" + re.escape(job) + r"(?:/|$)", cgroup, re.M):
        membership = "job-id cgroup"
    else:
        # PACE cgroup v2 uses an allocation nonce, not job_JOBID. Require
        # independent native PID/step association AND exact process job/step
        # environment before accepting the observed cgroup layout.
        assert native_steps is not None and pid in native_steps
        native_step = str(native_steps[pid])
        assert native_step.isdecimal()
        pattern = (
            r"^0::/system\.slice/slurmstepd\.scope/[A-Za-z0-9_-]+/step_"
            + re.escape(native_step)
            + r"/user/task_[0-9]+$"
        )
        assert re.fullmatch(pattern, cgroup.strip())
        entries = (p / "environ").read_bytes().split(b"\0")
        env = {}
        for entry in entries:
            if b"=" not in entry:
                continue
            key, value = entry.split(b"=", 1)
            if key in (b"SLURM_JOB_ID", b"SLURM_STEP_ID"):
                assert key not in env, "Duplicate process Slurm identity"
                env[key] = value
        assert env.get(b"SLURM_JOB_ID") == job.encode()
        assert env.get(b"SLURM_STEP_ID") == native_step.encode()
        membership = "native scontrol PID/step + process job/step + PACE cgroup v2"

    argv = (p / "cmdline").read_bytes().split(b"\0")
    argv = [x.decode() for x in argv if x]
    assert any(
        argv[i : i + 2] == ["-m", "egomimic.trainHydra"] for i in range(len(argv) - 1)
    )
    assert (p / "cwd").resolve() == repo
    assert (p / "exe").resolve() == python
    bit = 1 << (signal.SIGUSR2 - 1)
    assert int(status["SigCgt"].strip(), 16) & bit
    assert not int(status["SigIgn"].strip(), 16) & bit
    fields = (p / "stat").read_text().rpartition(") ")[2].split()
    start = int(fields[19])
    assert start > 0
    return {
        "pid": pid,
        "parent_pid": int(fields[1]),
        "start_time_ticks": start,
        "uid": uid,
        "pid_namespace_inode": namespace,
        "source_repo": str(repo),
        "python": str(python),
        "job_id": job,
        "handler": "SIGUSR2 caught; not ignored",
        "command": argv,
        "job_membership": membership,
        "cgroup": cgroup.strip(),
    }


def select_trainer_root(candidates):
    """Forked data workers inherit argv/handlers; require one verified tree root."""
    by_pid = {item["pid"]: item for item in candidates}
    assert len(by_pid) == len(candidates), "Duplicate candidate PID"
    roots = [item for item in candidates if item["parent_pid"] not in by_pid]
    assert len(roots) == 1, ("Exactly one verified trainer tree required", len(roots))
    root = roots[0]
    for item in candidates:
        seen = set()
        current = item
        while current["pid"] != root["pid"]:
            assert current["pid"] not in seen, "Cyclic process ancestry"
            seen.add(current["pid"])
            assert current["parent_pid"] in by_pid, "Disconnected trainer candidate"
            parent = by_pid[current["parent_pid"]]
            assert parent["job_id"] == current["job_id"]
            assert parent["cgroup"] == current["cgroup"], "Cross-step trainer ancestry"
            current = parent
    return root


def main():
    audit = len(sys.argv) == 3 and sys.argv[1] == "--audit-only"
    assert audit or (
        len(sys.argv) == 3 and sys.argv[1] == "--signal=USR2"
    ), "Only exact save-only or audit invocation allowed"
    job = sys.argv[2]
    assert re.fullmatch(r"[0-9]+", job) and job == os.environ["SLURM_JOB_ID"]
    assert os.environ.get("SLURM_STEP_ID") is not None
    assert int(os.environ["SLURM_NTASKS"]) == 1, "Single-rank route only"
    repo = pathlib.Path(os.environ["ICE_TRAINER_REPO"]).resolve(strict=True)
    python = pathlib.Path(os.environ["ICE_TRAINER_PYTHON"]).resolve(strict=True)
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    assert head == os.environ["ICE_TRAINER_HEAD"]
    assert not subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).strip()
    clean = {k: v for k, v in os.environ.items() if k != "SLURM_CLUSTERS"}
    output = subprocess.check_output(
        ["scontrol", "listpids", job], text=True, env=clean
    )
    pids = []
    native_steps = {}
    for line in output.splitlines():
        row = line.split()
        if len(row) >= 3 and row[0].isdigit() and row[1] == job:
            pid = int(row[0])
            assert pid not in native_steps or native_steps[pid] == row[2]
            pids.append(pid)
            native_steps[pid] = row[2]
    assert pids, "Native scheduler returned no current job PIDs"
    context = dict(
        repo=repo,
        python=python,
        uid=os.getuid(),
        namespace=pathlib.Path("/proc/self/ns/pid").stat().st_ino,
        job=job,
        native_steps=native_steps,
    )
    candidates = []
    for pid in sorted(set(pids)):
        try:
            candidates.append(candidate(pid, **context))
        except (OSError, AssertionError, ValueError, KeyError, IndexError):
            pass
    proof = select_trainer_root(candidates)
    proof["source_head"] = head
    proof["checked_at_unix"] = time.time()
    if audit:
        proof["status"] = "TRAINER_PID_ROUTE_AUDITED"
    else:
        fd = open_pidfd(proof["pid"])
        try:
            current = candidate(proof["pid"], **context)
            assert all(
                current[k] == proof[k] for k in current
            ), "Trainer identity changed before signal"
            send_pidfd(fd, signal.SIGUSR2)
        finally:
            os.close(fd)
        proof["status"] = "SAVE_ONLY_TRAINER_PID_SIGNALED"
    print(json.dumps(proof, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
