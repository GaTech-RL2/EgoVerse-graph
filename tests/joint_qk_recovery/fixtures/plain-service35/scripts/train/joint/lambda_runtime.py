"""Bind the maintained Lambda launcher to explicit runtime and scratch inputs."""

import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import sys
import tempfile


FIELDS = {
    "scratch_root": "JOINT_NODE_SCRATCH",
    "container_image": "JOINT_CONTAINER_IMAGE",
    "fa4_root": "JOINT_FA4_ROOT",
    "fa3_root": "JOINT_FA3_ROOT",
    "runtime_task": "JOINT_RUNTIME_TASK",
    "extra_pythonpath": "JOINT_EXTRA_PYTHONPATH",
    "actual_python": "JOINT_ACTUAL_PYTHON",
    "bootstrap_python": "JOINT_BOOTSTRAP_PYTHON",
}


def validate_policy(request, environment):
    runtime = request["lambda_runtime"]
    if (runtime.get("schema") != "joint-lambda-runtime-v1"
            or runtime["node"] != environment.get("SLURMD_NODENAME")
            or not environment.get("SLURM_JOB_ID", "").isdigit()
            or environment.get("SLURM_RESTART_COUNT", "0") != "0"):
        raise ValueError("Exact fresh Lambda node/allocation required")
    scratch = Path(runtime["scratch_root"])
    if (not scratch.is_absolute() or ".." in scratch.parts
            or scratch.parts[1:2] != ("ephemeral",) or len(scratch.parts) < 4):
        raise ValueError("Specific user-owned /ephemeral scratch required")
    for field in FIELDS.keys() - {"container_image"}:
        path = Path(runtime[field])
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("Absolute runtime path required: " + field)
    temporary = Path(runtime["temporary_root"])
    if (not temporary.is_absolute() or ".." in temporary.parts
            or temporary.parts[1:2] != ("ephemeral",) or len(temporary.parts) < 4
            or len(os.fsencode(temporary / ("j" + environment["SLURM_JOB_ID"]) / "tmp")) > 60):
        raise ValueError("Short explicit /ephemeral temporary root required for AF_UNIX sockets")
    if runtime["container_image"] != "nvcr.io#nvidia/pytorch:25.06-py3":
        raise ValueError("Changed pinned container recipe")
    if type(runtime["minimum_free_scratch_bytes"]) is not int or runtime["minimum_free_scratch_bytes"] < 10_000_000_000:
        raise ValueError("Explicit compiler/source scratch headroom required")
    return runtime


def socket_probe(directory):
    """Exercise the actual data-loader resource-sharer socket naming pattern."""
    with tempfile.TemporaryDirectory(prefix="pymp-", dir=directory) as child:
        address = str(Path(child) / "listener-0123456789abcdef")
        if len(os.fsencode(address)) >= 104:
            raise ValueError("Temporary path exceeds portable AF_UNIX socket budget")
        with socket.socket(socket.AF_UNIX) as listener:
            listener.bind(address)


def exports(request, environment, *, inside=False):
    runtime = validate_policy(request, environment)
    temporary = Path(runtime["temporary_root"]) / ("j" + environment["SLURM_JOB_ID"]) / "tmp"
    mounts = runtime["mounts"]
    path = Path(mounts["path"])
    if (not path.is_absolute()
            or hashlib.sha256(path.read_bytes()).hexdigest() != mounts["sha256"]):
        raise ValueError("Runtime mount manifest changed")
    if not inside:
        for item in path.read_text().strip().split(","):
            fields = item.split(":")
            if len(fields) != 3 or fields[2] != "ro" or not Path(fields[0]).is_dir():
                raise ValueError("Missing exact node-local runtime mount: " + item)
        scratch = Path(runtime["scratch_root"])
        scratch.mkdir(parents=True, exist_ok=True)
        if scratch.resolve() != scratch or scratch.stat().st_uid != os.getuid():
            raise ValueError("Scratch must be an owned physical directory")
        if shutil.disk_usage(scratch).free < runtime["minimum_free_scratch_bytes"]:
            raise ValueError("Insufficient node-local scratch")
        with tempfile.TemporaryFile(dir=scratch) as stream:
            stream.write(b"joint-runtime-write-check\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.mkdir(parents=True, exist_ok=True)
        if temporary.resolve() != temporary or temporary.stat().st_uid != os.getuid():
            raise ValueError("Temporary directory must be owned and physical")
        socket_probe(temporary)
    result = {name: runtime[field] for field, name in FIELDS.items()}
    result["JOINT_RUNTIME_MOUNTS"] = str(path)
    result["JOINT_JOB_TMP"] = str(temporary)
    return "\n".join("export " + name + "=" + shlex.quote(str(value))
                     for name, value in result.items())


def filesystem_recovery_reserve(request):
    """Check Lambda's actual persistent destination, without a fabricated Lustre quota."""
    quota = request["quota"]
    output, mount = Path(request["output"]), Path(quota["mount"])
    if (quota.get("kind") != "filesystem" or quota["uid"] != os.getuid()
            or quota["user"] != os.environ.get("SLURM_JOB_USER")
            or quota["node"] != os.environ.get("SLURMD_NODENAME")
            or not mount.is_absolute() or len(mount.parts) < 2
            or not output.is_relative_to(mount) or not output.parent.is_dir()):
        raise ValueError("Persistent recovery filesystem/owner binding changed")
    free = shutil.disk_usage(output.parent).free
    if free <= request["reserved_checkpoint_bytes"]:
        raise ValueError("Insufficient persistent recovery storage")
    return json.dumps(dict(kind="filesystem", user=quota["user"], uid=quota["uid"],
                           node=quota["node"], mount=str(mount), free_bytes=free))


def main():
    if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[2] not in ("--inside", "--check-only")):
        raise ValueError("Expected request [--inside|--check-only]")
    request = json.loads(Path(sys.argv[1]).read_text())
    result = exports(request, os.environ, inside=sys.argv[2:] == ["--inside"])
    if sys.argv[2:] == ["--check-only"]:
        scratch = Path(request["lambda_runtime"]["scratch_root"])
        print(json.dumps(dict(status="PASS", scratch_root=str(scratch), uid=os.getuid(),
                              node=os.environ["SLURMD_NODENAME"],
                              free_bytes=shutil.disk_usage(scratch).free,
                              temporary_root=str(Path(request["lambda_runtime"]["temporary_root"])),
                              runtime_mounts_verified=True, write_fsync_verified=True,
                              unix_socket_verified=True)))
    else:
        print(result)


if __name__ == "__main__":
    main()
