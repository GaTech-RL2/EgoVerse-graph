#!/usr/bin/env python3
"""Small positive base/CUDA/BF16/NCCL gate; actual model smoke stays downstream."""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from datetime import timedelta
from pathlib import Path


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def check_slurm_cgroup(host_text, inner_text, job_id, sluid=None, cluster=None):
    host_paths = [line.split(":", 2)[2] for line in host_text.splitlines()]
    inner_paths = [line.split(":", 2)[2] for line in inner_text.splitlines()]
    anchors = []
    for path in host_paths:
        parts = path.split("/")
        native_sluid = bool(sluid) and sluid in parts
        native_job = "job_" + job_id in parts or "job" + job_id in parts
        if native_sluid or (cluster != "phoenix-slurm" and native_job):
            anchors.append(path)
    require(anchors, "Slurm ancestor cgroup is not identified")
    require(
        all(
            any(
                inner == host or inner.startswith(host.rstrip("/") + "/")
                for inner in inner_paths
            )
            for host in anchors
        ),
        "Container escaped the Slurm ancestor cgroup",
    )


def cuda_uuid(properties):
    # PyTorch exposes CUDA's raw 16 UUID bytes. This avoids relying on the
    # formatting of the C++ _CUuuid.__str__ implementation across versions.
    return "GPU-" + str(uuid.UUID(bytes=bytes(properties.uuid.bytes)))


def validate_base(control):
    release = Path("/etc/os-release").read_text()
    require(
        "ID=ubuntu" in release or 'ID="ubuntu"' in release,
        "Container base is not Ubuntu",
    )
    require(sys.version_info[:2] == (3, 12), "Container Python is not 3.12")
    require(
        Path("/run/pace-proof/control.json").is_file(),
        "Positive container bind marker missing",
    )
    mount_namespace = os.readlink("/proc/self/ns/mnt")
    user_namespace = os.readlink("/proc/self/ns/user")
    require(
        mount_namespace != control["host_mount_namespace"],
        "Container mount namespace is unchanged",
    )
    require(
        user_namespace != control["host_user_namespace"],
        "Rootless container user namespace is unchanged",
    )
    check_slurm_cgroup(
        control["host_cgroup"],
        Path("/proc/self/cgroup").read_text(),
        control["attestation"]["job_id"],
        control["attestation"]["ownership"]["sluid"],
        control["attestation"]["cluster"],
    )
    return {
        "python": sys.version,
        "executable": sys.executable,
        "os_release": release,
        "mount_namespace": mount_namespace,
        "user_namespace": user_namespace,
        "host_cgroup": control["host_cgroup"],
        "container_cgroup": Path("/proc/self/cgroup").read_text(),
        "slurm_cgroup_preserved": True,
    }


def worker(control):
    # This entry is invoked through the frozen runtime_bootstrap.py prefix.
    import torch
    import torch.distributed as distributed

    rank = int(os.environ["LOCAL_RANK"])
    require(
        torch.__version__ == control["expected_torch"],
        "Scientific Torch version changed",
    )
    require(
        os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "0"
        and os.environ.get("NVIDIA_TF32_OVERRIDE") == "0",
        "Publisher TF32 override was not disabled",
    )
    require(
        not torch.backends.cuda.matmul.allow_tf32
        and not torch.backends.cudnn.allow_tf32,
        "Frozen disabled-TF32 policy was not applied",
    )
    require(
        torch.cuda.device_count() == 4,
        "Container does not expose exactly four CUDA GPUs",
    )
    require(torch.cuda.is_available(), "CUDA unavailable")
    torch.cuda.set_device(rank)
    properties = torch.cuda.get_device_properties(rank)
    require(properties.major >= 9, "CUDA GPU is below SM90")
    require(
        "H100" in properties.name or "H200" in properties.name,
        "CUDA GPU family changed",
    )
    # The CUDA runtime's UUID, rather than NVML's full host inventory, proves the
    # rank actually accesses the GPU admitted for this logical CUDA ordinal.
    actual_uuid = cuda_uuid(properties)
    require(
        actual_uuid.lower() == control["expected_uuids"][rank].lower(),
        "CUDA UUID differs from Slurm assignment",
    )
    require(torch.cuda.is_bf16_supported(), "BF16 unsupported")
    left = torch.full((64, 64), 0.5, dtype=torch.bfloat16, device="cuda")
    product = left @ left
    require(
        product.dtype == torch.bfloat16 and bool(torch.isfinite(product).all()),
        "BF16 matmul failed",
    )
    require(
        bool(
            torch.allclose(product.float(), torch.full((64, 64), 16.0, device="cuda"))
        ),
        "BF16 matmul gave wrong values",
    )
    distributed.init_process_group("nccl", timeout=timedelta(seconds=120))
    value = torch.tensor([float(rank + 1)], device="cuda")
    distributed.all_reduce(value)
    require(value.item() == 10.0, "Four-rank NCCL reduction failed")
    distributed.barrier()
    torch.cuda.synchronize()
    distributed.destroy_process_group()
    write_new(
        Path(control["receipt_directory"]) / f"gpu-rank-{rank}.json",
        {
            "status": "PASS",
            "rank": rank,
            "world_size": 4,
            "uuid": actual_uuid,
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "torch": torch.__version__,
            "torch_file": torch.__file__,
            "bf16": True,
            "nccl_all_reduce": 10.0,
        },
    )


def controller(control):
    base = validate_base(control)
    require(len(control["expected_uuids"]) == 4, "Four assigned UUIDs required")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    children = []
    started = time.monotonic()
    try:
        for rank in range(4):
            environment = dict(os.environ)
            environment.update(
                {
                    "RANK": str(rank),
                    "LOCAL_RANK": str(rank),
                    "WORLD_SIZE": "4",
                    "LOCAL_WORLD_SIZE": "4",
                    "MASTER_ADDR": "127.0.0.1",
                    "MASTER_PORT": str(port),
                    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
                }
            )
            command = [
                *control["python_prefix"],
                "/run/pace-boundary/gpu_admission.py",
                "--worker",
            ]
            children.append(subprocess.Popen(command, env=environment))
        while any(child.poll() is None for child in children):
            require(time.monotonic() - started < 300, "CUDA/NCCL admission timed out")
            require(
                all(child.poll() in (None, 0) for child in children),
                "CUDA/NCCL admission rank failed",
            )
            time.sleep(0.2)
        require(
            all(child.returncode == 0 for child in children),
            "CUDA/NCCL admission failed",
        )
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
    receipts = [
        json.loads(
            (Path(control["receipt_directory"]) / f"gpu-rank-{rank}.json").read_text()
        )
        for rank in range(4)
    ]
    require(
        [receipt["uuid"].lower() for receipt in receipts]
        == [uuid.lower() for uuid in control["expected_uuids"]],
        "Rank admission UUIDs differ",
    )
    admission = {
        "status": "PASS",
        "base": base,
        "ranks": receipts,
        "nccl_passed": True,
        "bf16_passed": True,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_new(Path(control["receipt_directory"]) / "gpu-admission.json", admission)
    attestation = dict(control["attestation"])
    attestation.update({"status": "PASS", "gpu_probe": admission})
    write_new(Path(control["receipt_directory"]) / "attestation.json", attestation)
    print(
        "PACE GPU ADMISSION PASS: exact base, four allocated UUIDs, BF16, four-rank NCCL",
        flush=True,
    )
    # Training command is opaque here; source/config/resume checks remain owned
    # by the original reviewed inner launcher.
    os.execvpe(control["inner_command"][0], control["inner_command"], os.environ)


def bootstrap(control):
    # The base interpreter only parses this control file. Every actual probe,
    # including the controller, runs through the frozen scientific bootstrap.
    prefix = control["python_prefix"]
    require(
        prefix and Path(prefix[0]).is_absolute(), "Scientific Python prefix is missing"
    )
    command = [*prefix, "/run/pace-boundary/gpu_admission.py", "--controller"]
    os.execvpe(command[0], command, os.environ)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--worker", action="store_true")
    choice.add_argument("--controller", action="store_true")
    choice.add_argument("--bootstrap", action="store_true")
    arguments = parser.parse_args()
    control_path = Path(
        os.environ.get("PHOENIX_BOUNDARY_CONTROL", "/run/pace-proof/control.json")
    )
    control = json.loads(control_path.read_text())
    if arguments.worker:
        worker(control)
    elif arguments.bootstrap:
        bootstrap(control)
    else:
        controller(control)


if __name__ == "__main__":
    main()
