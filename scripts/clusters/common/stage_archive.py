"""Stage a trusted immutable archive on allocation-local storage, with a receipt.

This boundary stages datasets and other regular-file artifacts. Call it inside
the scheduled allocation; keep training outputs on persistent storage.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile


def stage(archive, expected_sha256, scratch_root, reserve_bytes):
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
        raise RuntimeError("Archive staging requires a scheduled job step")
    archive, scratch_root = Path(archive).resolve(), Path(scratch_root).resolve()
    scratch_root.mkdir(parents=True, exist_ok=True)
    filesystem = subprocess.check_output(
        ["findmnt", "-n", "-o", "FSTYPE", "-T", str(scratch_root)], text=True).strip()
    if filesystem not in {"ext4", "xfs", "btrfs", "tmpfs", "overlay"}:
        raise ValueError(f"Refusing non-local staging filesystem: {filesystem}")
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected_sha256:
        raise ValueError("Archive checksum differs from approved input")
    target = scratch_root / digest
    with (scratch_root / ".stage.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt = target / ".stage-receipt.json"
        if target.exists():
            value = json.loads(receipt.read_text())
            if value["sha256"] != digest or not value["complete"]:
                raise ValueError("Existing staged directory lacks a matching receipt")
            return target
        with tarfile.open(archive) as bundle:
            members = bundle.getmembers()
            for member in members:
                name = PurePosixPath(member.name)
                if name.is_absolute() or ".." in name.parts or not (member.isfile() or member.isdir()):
                    raise ValueError(f"Unsafe archive member: {member.name}")
            unpacked = sum(member.size for member in members)
            free = shutil.disk_usage(scratch_root).free
            required = 2 * unpacked + reserve_bytes
            if free < required:
                raise RuntimeError(f"Local disk needs {required} bytes; only {free} free")
            temporary = Path(tempfile.mkdtemp(prefix=".staging-", dir=scratch_root))
            try:
                bundle.extractall(temporary, members=members, filter="data")
                value = {"complete": True, "sha256": digest, "archive": str(archive),
                         "unpacked_bytes": unpacked, "free_bytes_before": free,
                         "reserve_bytes": reserve_bytes, "filesystem": filesystem,
                         "job_id": os.environ["SLURM_JOB_ID"], "node": os.uname().nodename}
                (temporary / ".stage-receipt.json").write_text(json.dumps(value, indent=2) + "\n")
                temporary.rename(target)
            except BaseException:
                shutil.rmtree(temporary)
                raise
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--reserve-gib", type=int, default=100)
    args = parser.parse_args()
    if args.reserve_gib < 0:
        parser.error("reserve-gib must be nonnegative")
    print(stage(args.archive, args.sha256, args.scratch_root, args.reserve_gib * 1024**3))
