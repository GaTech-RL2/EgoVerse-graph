#!/usr/bin/env python3
"""Run a hash-bound training command in a scheduled, rootless PACE container.

This is a hardware/container boundary, not a training implementation. No shared
runtime is extracted, no Slurm job is submitted, and no system configuration is
changed. The caller supplies the reviewed inner command and scientific runtime.
"""

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath

import gpu_ownership

SCHEMA = "pace-podman-boundary-v2"
GIB = 1024**3
GPU_UUID = re.compile(r"GPU-[0-9a-fA-F-]{36}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,95}\Z")
ENV_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class AdmissionError(RuntimeError):
    pass


class InterruptedLaunch(AdmissionError):
    def __init__(self, signum):
        super().__init__(f"Launch interrupted by signal {signum}; child stopped")
        self.signum = signum


def require(condition, message):
    if not condition:
        raise AdmissionError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verified_file(reference):
    require(isinstance(reference, dict), "File reference must be an object")
    path = Path(reference["path"])
    require(path.is_absolute(), "File reference must be absolute")
    require(not path.is_symlink() and path.is_file(), f"Not a regular input: {path}")
    require(bool(SHA256.fullmatch(reference.get("sha256", ""))), "Invalid SHA-256")
    if "size_bytes" in reference:
        require(path.stat().st_size == reference["size_bytes"], f"Size changed: {path}")
    require(digest(path) == reference["sha256"], f"SHA-256 changed: {path}")
    return path


def read_bound_json(reference):
    return json.loads(verified_file(reference).read_text())


def write_json_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def checked_tool(specification, name):
    path = verified_file(specification)
    require(os.access(path, os.X_OK), f"Tool is not executable: {name}")
    origin = specification.get("origin")
    require(origin in {"site", "official-nvidia"}, f"Unreviewed tool origin: {name}")
    if origin == "official-nvidia":
        proof = read_bound_json(specification["provenance"])
        serialized = json.dumps(proof)
        require(
            "https://github.com/NVIDIA/" in serialized
            or "https://nvidia.github.io/" in serialized,
            f"Missing official NVIDIA provenance: {name}",
        )
    return str(path)


def publisher_environment(receipt):
    require(receipt.get("status") == "PASS", "Base image receipt did not pass")
    reference = receipt.get("source_image_reference", "")
    image_digest = receipt.get("source_image_digest", "")
    require(
        bool(re.fullmatch(r"sha256:[0-9a-f]{64}", image_digest)), "Base digest missing"
    )
    if "@" in reference:
        require(
            receipt.get("source_tag") == "nvcr.io/nvidia/pytorch:25.06-py3",
            "Wrong base tag",
        )
        require(
            reference == "nvcr.io/nvidia/pytorch@" + image_digest,
            "Base reference/digest mismatch",
        )
    else:
        require(reference == "nvcr.io/nvidia/pytorch:25.06-py3", "Wrong base")
    require(
        receipt.get("platform") in (None, "linux/amd64", "linux-amd64")
        or receipt.get("platform") == {"os": "linux", "architecture": "amd64"},
        "Wrong platform",
    )
    result = {}
    for entry in receipt["config_env"]:
        require(isinstance(entry, str) and "=" in entry, "Invalid publisher ENV")
        key, value = entry.split("=", 1)
        require(bool(ENV_KEY.fullmatch(key)), "Invalid publisher ENV name")
        require(key not in result, "Duplicate publisher ENV")
        result[key] = value
    require("PATH" in result, "Publisher PATH missing")
    return result


def required_free_bytes(base_regular_bytes, payload_regular_bytes):
    require(
        isinstance(payload_regular_bytes, int) and payload_regular_bytes > 0,
        "Bad payload size",
    )
    if base_regular_bytes is None:
        return 200 * GIB
    require(
        isinstance(base_regular_bytes, int) and base_regular_bytes > 0, "Bad base size"
    )
    # One imported VFS layer plus its writable container copy; no uncompressed
    # tar intermediate is created. JIT/cache/log headroom is explicit.
    return 2 * base_regular_bytes + payload_regular_bytes + 16 * GIB


def local_workspace(parent, namespace, job_id, required):
    parent = Path(parent)
    require(parent.is_absolute() and parent.is_dir(), "Local parent is unavailable")
    resolved = parent.resolve()
    require(
        resolved != Path.home() and Path.home() not in resolved.parents,
        "Home cannot hold the runtime cache",
    )
    filesystem = local_filesystem(resolved, Path("/proc/self/mountinfo").read_text())
    require(
        filesystem in {"xfs", "ext2", "ext3", "ext4", "btrfs", "tmpfs"},
        "Runtime extraction must use allocated node-local temporary storage",
    )
    require(
        shutil.disk_usage(parent).free >= required, "Insufficient node-local capacity"
    )
    root = parent / f"pace-{namespace}-{job_id}"
    root.mkdir(mode=0o700)
    for child in ("payload", "storage", "run", "tmp", "cdi", "logs", "jit"):
        (root / child).mkdir(mode=0o700)
    return root


def local_filesystem(path, mountinfo):
    """Resolve the longest matching mount, rather than guessing from its name."""
    matches = []
    path = Path(path)
    for line in mountinfo.splitlines():
        before, separator, after = line.partition(" - ")
        require(bool(separator), "Malformed Linux mount information")
        fields = before.split()
        mount = Path(
            fields[4]
            .replace("\\040", " ")
            .replace("\\011", "\t")
            .replace("\\134", "\\")
        )
        if mount == path or mount in path.parents:
            matches.append((len(mount.parts), after.split()[0]))
    require(bool(matches), "Temporary filesystem type is unknown")
    return max(matches)[1]


def normalized_member(name):
    require(isinstance(name, str) and "\x00" not in name, "Invalid archive path")
    path = PurePosixPath(name)
    require(
        not path.is_absolute() and ".." not in path.parts,
        f"Escaping archive path: {name}",
    )
    return str(path)


def safe_extract_zstd(
    archive, destination, zstd, *, compression="zstd", regular_byte_limit=None
):
    """Extract without ever following an archive-created symlink on the host.

    Symlinks are installed last. The pinned venv's /usr/bin/python leaf link is
    retained literally; it must resolve against the container, never the host.
    Both reviewed collector formats use the identical streamed tar guards.
    """
    require(compression in {"zstd", "gzip"}, "Unsupported payload compression")
    if regular_byte_limit is not None:
        require(
            type(regular_byte_limit) is int and regular_byte_limit >= 0,
            "Invalid regular-byte budget",
        )
    destination = Path(destination)
    seen, deferred, directories = set(), [], []
    regular_bytes = 0
    producer = None
    if compression == "gzip":
        decompressed = gzip.open(archive, "rb")
    else:
        producer = subprocess.Popen(
            [zstd, "-dc", "--", str(archive)], stdout=subprocess.PIPE
        )
        decompressed = producer.stdout
    try:
        with tarfile.open(fileobj=decompressed, mode="r|") as stream:
            for member in stream:
                name = normalized_member(member.name)
                if name == ".":
                    require(member.isdir(), "Archive root must be a directory")
                    continue
                require(
                    name.startswith("workspace/"),
                    "Payload must contain only canonical workspace paths",
                )
                require(name not in seen, f"Duplicate archive member: {name}")
                seen.add(name)
                target = destination / name
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                # Any non-directory parent is rejected; symlinks are deferred.
                for parent in target.parents:
                    if parent == destination:
                        break
                    require(
                        not parent.is_symlink() and parent.is_dir(),
                        "Archive parent is not a directory",
                    )
                if member.isdir():
                    target.mkdir(mode=0o700, exist_ok=True)
                    directories.append((target, member.mode & 0o777, member.mtime))
                elif member.isreg():
                    require(member.size >= 0, "Negative archive member size")
                    if regular_byte_limit is not None:
                        require(
                            regular_bytes + member.size <= regular_byte_limit,
                            "Payload exceeded its immutable regular-byte budget",
                        )
                    with target.open("xb") as output:
                        source = stream.extractfile(member)
                        require(source is not None, "Missing archive file body")
                        shutil.copyfileobj(source, output, 8 * 1024 * 1024)
                    require(
                        target.stat().st_size == member.size, "Truncated archive file"
                    )
                    target.chmod(member.mode & 0o777)
                    os.utime(target, (member.mtime, member.mtime))
                    regular_bytes += member.size
                elif member.issym() or member.islnk():
                    deferred.append((member, target))
                else:
                    raise AdmissionError(f"Special archive member rejected: {name}")
        # Consume tar padding through the compression trailer. This verifies
        # gzip's CRC and avoids closing a zstd pipe before its final writes.
        while decompressed.read(1024 * 1024):
            pass
        decompressed.close()
        if producer is not None:
            require(producer.wait() == 0, "Zstd decompression failed")
        # Hard links only refer to already materialized regular files.
        for member, target in deferred:
            if member.islnk():
                source = destination / normalized_member(member.linkname)
                require(
                    source.is_file() and not source.is_symlink(),
                    "Unsafe archive hard link",
                )
                os.link(source, target)
        for member, target in deferred:
            if member.issym():
                require(
                    not target.exists() and not target.is_symlink(),
                    "Symlink collides with archive path",
                )
                target.symlink_to(member.linkname)
        for directory, mode, mtime in reversed(directories):
            directory.chmod(mode)
            os.utime(directory, (mtime, mtime))
        return {"archive_members": len(seen), "regular_bytes": regular_bytes}
    finally:
        decompressed.close()
        if producer is not None and producer.poll() is None:
            producer.kill()
            producer.wait()


def verify_payload_manifest(manifest, payload):
    count, seen = 0, set()
    for line in Path(manifest).read_text().splitlines():
        require(
            len(line) > 66 and line[64:66] in ("  ", " *"), "Invalid payload manifest"
        )
        expected, name = line[:64], line[66:]
        require(bool(SHA256.fullmatch(expected)), "Invalid payload manifest digest")
        name = normalized_member(
            name.lstrip("/") if name.startswith("/workspace/") else name
        )
        require(
            name.startswith("workspace/") and name not in seen,
            "Bad/duplicate manifest path",
        )
        seen.add(name)
        path = Path(payload) / name
        require(
            all(
                not parent.is_symlink()
                for parent in path.parents
                if parent != Path(payload)
            ),
            "Manifest traverses a symlink",
        )
        require(
            path.is_file() and not path.is_symlink(), f"Manifest file missing: {name}"
        )
        require(digest(path) == expected, f"Extracted payload hash changed: {name}")
        count += 1
    require(count > 0, "Empty payload manifest")
    return count


def parse_inventory(text):
    rows = []
    for row in csv.reader(text.splitlines(), skipinitialspace=True):
        require(len(row) == 5, "Unexpected GPU inventory columns")
        index, uuid, name, capability, memory = (value.strip() for value in row)
        require(
            index.isdigit() and bool(GPU_UUID.fullmatch(uuid)), "Invalid GPU identity"
        )
        rows.append(
            {
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "sm": float(capability),
                "memory_mib": int(memory),
            }
        )
    require(
        rows and len({row["uuid"] for row in rows}) == len(rows),
        "Invalid GPU inventory",
    )
    return rows


def assigned_gpus(evidence, expected_identity):
    # Slurm IDX and filtered NVML indices are separate namespaces. The current
    # cgroup, kernel device permissions and UUID/BDF/minor bijection prove ownership.
    try:
        return gpu_ownership.verify(evidence, expected_identity)
    except (ValueError, KeyError) as error:
        raise AdmissionError("GPU ownership: " + str(error)) from error


def gpu_health(xml_text, selected):
    tree = ET.fromstring(xml_text)
    nodes = {node.findtext("uuid"): node for node in tree.findall("gpu")}
    result = []
    for gpu in selected:
        require(gpu["uuid"] in nodes, "GPU health identity missing")
        node = nodes[gpu["uuid"]]
        checked = {}
        require(
            node.findtext("ecc_mode/current_ecc") == "Enabled", "GPU ECC is not enabled"
        )
        require(
            node.findtext("mig_mode/current_mig") == "Disabled",
            "MIG is outside this four-full-GPU contract",
        )
        for category in ("volatile", "aggregate"):
            group = node.find(f"ecc_errors/{category}")
            require(group is not None, "Uncorrectable ECC health is unavailable")
            values = [
                leaf.text
                for leaf in group.iter()
                if len(leaf) == 0 and "uncorrectable" in leaf.tag
            ]
            if not values:
                value = group.findtext("double_bit/total") or group.findtext(
                    "uncorrectable/total"
                )
                values = [value] if value is not None else []
            require(
                values and all(value and value.strip().isdigit() for value in values),
                "Uncorrectable ECC health is unavailable",
            )
            require(
                all(int(value) == 0 for value in values),
                "Uncorrectable ECC errors present",
            )
            checked[f"{category}_uncorrectable"] = [int(value) for value in values]
        recovery = node.findtext("gpu_recovery_action")
        if recovery is not None:
            require(recovery.strip().lower() == "none", "GPU recovery action requested")
            checked["gpu_recovery_action"] = recovery.strip()
        else:
            for key in ("reset_required", "drain_and_reset_recommended"):
                value = node.findtext(f"gpu_reset_status/{key}")
                require(
                    value is not None and value.strip().lower() == "no",
                    "GPU reset/repair requested or status unavailable",
                )
                checked[key] = value.strip()
        for key in ("reset_required", "drain_and_reset_recommended"):
            value = node.findtext(f"gpu_reset_status/{key}")
            if value is not None:
                require(
                    value.strip().lower() in {"no", "n/a"}, "GPU reset/repair requested"
                )
        for path in (
            "retired_pages/pending_retirement",
            "retired_pages/pending_blacklist",
            "remapped_rows/remapping_pending",
            "remapped_rows/remapping_failure_occurred",
            "remapped_rows/remapped_row_pending",
            "remapped_rows/remapped_row_failure",
            "ecc_errors/channel_repair_pending",
            "ecc_errors/tpc_repair_pending",
            "ecc_errors/aggregate/sram_threshold_exceeded",
        ):
            value = node.findtext(path)
            if value is not None:
                require(
                    value.strip().lower() in {"no", "n/a"},
                    "GPU page retirement/remapping needs attention",
                )
                checked[path] = value.strip()
        result.append({**gpu, "health": checked})
    return result


def restrict_cdi(specification, selected, hook_path):
    require(
        specification.get("kind") == "nvidia.com/gpu", "Unexpected CDI vendor/class"
    )
    by_name = {device["name"]: device for device in specification["devices"]}
    expected = [gpu["uuid"] for gpu in selected]
    require(all(uuid in by_name for uuid in expected), "CDI lacks assigned UUID names")
    result = dict(specification)
    result["devices"] = [by_name[uuid] for uuid in expected]
    indices = {gpu["minor"] for gpu in selected}
    for edits in [
        result.get("containerEdits", {}),
        *[device.get("containerEdits", {}) for device in result["devices"]],
    ]:
        for device in edits.get("deviceNodes", []):
            for key in ("path", "hostPath"):
                path = device.get(key, "")
                match = re.fullmatch(r"/dev/nvidia([0-9]+)", path)
                require(
                    not match or int(match.group(1)) in indices,
                    "CDI exposes an unallocated GPU",
                )
        for mount in edits.get("mounts", []):
            for key in ("hostPath", "containerPath"):
                path = mount.get(key, "")
                require(
                    path.rstrip("/") != "/dev",
                    "CDI mounts the entire host device directory",
                )
                match = re.fullmatch(r"/dev/nvidia([0-9]+)", path)
                require(
                    not match or int(match.group(1)) in indices,
                    "CDI mounts an unallocated GPU",
                )
        for hook in edits.get("hooks", []):
            require(hook.get("path") == hook_path, "CDI invokes an unverified hook")
    return result


def cdi_generation_command(ctk, hook, output, selected):
    command = [
        ctk,
        "cdi",
        "generate",
        "--format=json",
        "--device-name-strategy=uuid",
        "--no-all-device",
        "--nvidia-cdi-hook-path",
        hook,
        "--output",
        str(output),
    ]
    for gpu in selected:
        command += ["--device-id", gpu["uuid"]]
    return command


def podman_base(tool, local):
    return [
        tool,
        "--remote=false",
        "--root",
        str(local / "storage"),
        "--runroot",
        str(local / "run"),
        "--tmpdir",
        str(local / "tmp"),
        "--events-backend=file",
        "--storage-driver=vfs",
        "--storage-opt=vfs.ignore_chown_errors=true",
    ]


def check_podman_info(information, local):
    require(
        information["host"]["security"]["rootless"] is True, "Podman is not rootless"
    )
    require(
        information["store"]["graphDriverName"] == "vfs",
        "Unexpected Podman storage driver",
    )
    require(
        Path(information["store"]["graphRoot"]).resolve()
        == (local / "storage").resolve(),
        "Podman graph root escaped node-local storage",
    )
    require(
        Path(information["store"]["runRoot"]).resolve() == (local / "run").resolve(),
        "Podman run root escaped node-local storage",
    )


def reject_payload_shadowing(mounts, manifest):
    prefixes = [
        str(PurePosixPath(mount["container"])).rstrip("/") + "/" for mount in mounts
    ]
    for line in Path(manifest).read_text().splitlines():
        name = "/" + line[66:].lstrip("/")
        require(
            not any(
                name == prefix[:-1] or name.startswith(prefix) for prefix in prefixes
            ),
            "Persistent bind would shadow an immutable payload file",
        )


def prepare_mount_targets(payload, mounts):
    """Create new mount points before the containing workspace becomes read-only."""
    workspace = Path(payload) / "workspace"
    prepared = []
    for mount in mounts:
        checked_mount(mount)
        relative = PurePosixPath(mount["container"]).relative_to("/workspace")
        target = workspace.joinpath(*relative.parts)
        current = workspace
        for component in relative.parts[:-1]:
            current = current / component
            require(
                not current.is_symlink(),
                "Persistent mount target traverses an archive symlink",
            )
            current.mkdir(mode=0o700, exist_ok=True)
            require(current.is_dir(), "Persistent mount parent is not a directory")
        require(
            not target.is_symlink(), "Persistent mount target is an archive symlink"
        )
        if Path(mount["host"]).is_dir():
            target.mkdir(mode=0o700, exist_ok=True)
            require(target.is_dir(), "Directory bind target is not a directory")
        else:
            require(not target.exists(), "File bind would shadow an archived member")
            target.touch(mode=0o600, exist_ok=False)
        prepared.append(str(mount["container"]))
    return prepared


def allocation_environment(host, inner):
    result = dict(inner)
    for key, value in host.items():
        if key.startswith("SLURM_") or key == "SLURMD_NODENAME":
            result[key] = value
    require(host.get("SLURMD_NODENAME"), "Allocated node identity is missing")
    result["PIXEL_NODE"] = host["SLURMD_NODENAME"]
    return result


def apply_runtime_masks(environment, selected):
    result = dict(environment)
    result.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(gpu["uuid"] for gpu in selected),
            "NVIDIA_VISIBLE_DEVICES": "void",
            "PHOENIX_BOUNDARY_CONTROL": "/run/pace-proof/control.json",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
            "NVIDIA_TF32_OVERRIDE": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return result


def checked_mount(mount):
    host, container = Path(mount["host"]), PurePosixPath(mount["container"])
    require(
        host.is_absolute() and host.exists() and not host.is_symlink(),
        "Persistent mount is unavailable",
    )
    require(
        container.is_absolute() and ".." not in container.parts,
        "Invalid container mount target",
    )
    require(
        str(container).startswith("/workspace/") and str(container) != "/workspace/",
        "Persistent mounts must overlay explicit workspace subpaths",
    )
    require(isinstance(mount["read_only"], bool), "Mount mode must be explicit")
    require(
        "," not in str(host) and "," not in str(container), "Unsupported mount path"
    )
    return f"type=bind,src={host},dst={container}," + (
        "ro" if mount["read_only"] else "rw"
    )


def container_command(base, request, local, proofs, image, selected, environment):
    command = [
        *base,
        "--cdi-spec-dir",
        str(local / "cdi"),
        "run",
        "--rm",
        "--pull=never",
        "--network=host",
        "--ipc=host",
        "--cgroupns=host",
        "--group-add=keep-groups",
        "--security-opt=no-new-privileges",
        "--name",
        f"pace-{request['namespace']}-{os.environ['SLURM_JOB_ID']}",
    ]
    mounts = [
        f"type=bind,src={local / 'payload' / 'workspace'},dst=/workspace,ro",
        f"type=bind,src={Path(__file__).resolve().parent},dst=/run/pace-boundary,ro",
        f"type=bind,src={proofs},dst=/run/pace-proof,rw",
        f"type=bind,src={local / 'logs'},dst=/tmp/pixel-full-logs,rw",
        f"type=bind,src={local / 'jit'},dst=/run/pace-jit,rw",
    ]
    mounts.extend(
        checked_mount(mount) for mount in request.get("persistent_mounts", [])
    )
    credential = request["credential"]
    host = Path(credential["host"])
    require(
        host.is_absolute() and host.is_file() and not host.is_symlink(),
        "Credential binding is unavailable",
    )
    require(
        stat.S_IMODE(host.stat().st_mode) & 0o077 == 0, "Credential must be private"
    )
    require(
        credential["container"] == "/run/pixel-wandb.netrc",
        "Unexpected credential target",
    )
    mounts.append(f"type=bind,src={host},dst=/run/pixel-wandb.netrc,ro")
    for mount in mounts:
        command += ["--mount", mount]
    for gpu in selected:
        command += ["--device", f"nvidia.com/gpu={gpu['uuid']}"]
    for key, value in sorted(environment.items()):
        require(
            bool(ENV_KEY.fullmatch(key)) and isinstance(value, str),
            "Invalid container environment",
        )
        command += ["--env", f"{key}={value}"]
    command += [
        "--entrypoint",
        "/bin/bash",
        image,
        "/run/pace-boundary/inside_pace.sh",
    ]
    return command


def run_checked(command, environment, capture=True, timeout=None):
    child = subprocess.Popen(
        command,
        env=environment,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        start_new_session=True,
    )
    received = []
    previous = {}
    started = time.monotonic()

    def forward(signum, _frame):
        received.append((signum, time.monotonic()))
        if child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.signal(signum, forward)
    try:
        while True:
            try:
                stdout, stderr = child.communicate(timeout=1)
                break
            except subprocess.TimeoutExpired:
                if received and time.monotonic() - received[0][1] >= 30:
                    os.killpg(child.pid, signal.SIGKILL)
                    stdout, stderr = child.communicate(timeout=10)
                    break
                if timeout is not None and time.monotonic() - started >= timeout:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.communicate(timeout=10)
                    raise subprocess.TimeoutExpired(command, timeout)
        if received:
            raise InterruptedLaunch(received[0][0])
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=10)
    detail = (stderr or "")[-4000:] if capture else "See job stderr"
    require(
        child.returncode == 0,
        f"Boundary subprocess failed (exit {child.returncode}): {command[0]}\n{detail}",
    )
    return stdout if capture else None


def cleanup_container(base, name, environment):
    """Bounded cleanup of this request's named container before log mirroring."""
    result = []
    for arguments, timeout in (
        (["stop", "--time=20", name], 25),
        (["rm", "--force", name], 20),
    ):
        try:
            command = subprocess.run(
                [*base, *arguments],
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            result.append(
                {
                    "operation": arguments[0],
                    "exit_code": command.returncode,
                    "stderr_tail": command.stderr[-2000:],
                }
            )
        except subprocess.TimeoutExpired:
            result.append({"operation": arguments[0], "status": "TIMEOUT"})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--request-sha256", required=True)
    arguments = parser.parse_args()
    request_path = verified_file(
        {"path": arguments.request, "sha256": arguments.request_sha256}
    )
    request = json.loads(request_path.read_text())
    require(request.get("schema") == SCHEMA, "Wrong outer request schema")
    bound_helpers = {
        Path(item["path"]).name: verified_file(item)
        for item in request["boundary_files"]
    }
    require(
        set(bound_helpers)
        == {"pace_podman.py", "gpu_admission.py", "gpu_ownership.py", "inside_pace.sh"},
        "Boundary helper binding is incomplete",
    )
    require(
        all(
            path.resolve().parent == Path(__file__).resolve().parent
            for path in bound_helpers.values()
        ),
        "Boundary helper is outside the reviewed directory",
    )
    require(bool(NAME.fullmatch(request["namespace"])), "Invalid job namespace")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    require(
        job_id.isdigit() and os.environ.get("SLURMD_NODENAME"),
        "A scheduled compute allocation is required",
    )
    require(
        int(os.environ.get("SLURM_RESTART_COUNT", "0")) == 0,
        "Use a fresh immutable resume request after a restart",
    )
    if request.get("expected_node"):
        require(
            os.environ["SLURMD_NODENAME"] == request["expected_node"],
            "Wrong compute node",
        )
    require(
        request.get("inner_command")
        and all(isinstance(arg, str) for arg in request["inner_command"]),
        "Inner command is missing",
    )
    require(
        request.get("gpu_probe_python_prefix")
        and all(isinstance(arg, str) for arg in request["gpu_probe_python_prefix"]),
        "Scientific Python prefix is missing",
    )
    tools = {
        key: checked_tool(request["tools"][key], key)
        for key in ("podman", "zstd", "nvidia_ctk", "nvidia_cdi_hook")
    }
    base_receipt = read_bound_json(request["base_receipt"])
    image_environment = publisher_environment(base_receipt)
    base_archive = verified_file(base_receipt["rootfs_archive"])
    source_config = read_bound_json(base_receipt["source_config"])
    require(
        source_config.get("os") == "linux"
        and source_config.get("architecture") == "amd64",
        "OCI config has wrong platform",
    )
    require(
        source_config["config"]["Env"] == base_receipt["config_env"],
        "Publisher ENV differs from bound OCI config",
    )
    payload_archive = verified_file(request["payload"]["archive"])
    payload_compression = request["payload"]["archive"].get("compression")
    require(
        payload_compression in {"zstd", "gzip"},
        "Payload compression must be explicitly bound",
    )
    manifest = verified_file(request["payload"]["manifest"])
    required = required_free_bytes(
        request.get("base_uncompressed_bytes"), request["payload"]["regular_bytes"]
    )
    local = local_workspace(
        request.get("local_parent") or os.environ.get("TMPDIR", "/tmp"),
        request["namespace"],
        job_id,
        required,
    )
    proof_root = Path(request["proof_root"])
    require(
        proof_root.is_absolute()
        and proof_root.is_dir()
        and not proof_root.is_symlink(),
        "Persistent proof root is missing",
    )
    proofs = proof_root / f"job-{job_id}"
    proofs.mkdir(mode=0o700)
    record = {
        "schema": SCHEMA,
        "status": "STARTED",
        "job_id": job_id,
        "node": os.environ["SLURMD_NODENAME"],
        "request_sha256": arguments.request_sha256,
        "base_receipt_sha256": request["base_receipt"]["sha256"],
        "payload_sha256": request["payload"]["archive"]["sha256"],
        "required_local_free_bytes": required,
        "local_root": str(local),
        "started_unix": time.time(),
    }
    write_json_new(proofs / "started.json", record)
    container_started = False
    base = None
    try:
        host_env = dict(os.environ)
        host_env.update(
            {"XDG_RUNTIME_DIR": str(local / "run"), "TMPDIR": str(local / "tmp")}
        )
        ownership_evidence = gpu_ownership.collect()
        write_json_new(proofs / "ownership-evidence.json", ownership_evidence)
        ownership = assigned_gpus(ownership_evidence, request["expected_identity"])
        write_json_new(proofs / "gpu-ownership.json", ownership)
        selected = ownership["gpus"]
        health_xml = run_checked(
            ["nvidia-smi", "-q", "-x", "-i", ",".join(gpu["uuid"] for gpu in selected)],
            host_env,
            timeout=30,
        )
        health = gpu_health(health_xml, selected)
        write_json_new(proofs / "host-gpus.json", health)
        (proofs / "host-health.xml").write_text(health_xml)
        generated = local / "generated-cdi.json"
        run_checked(
            cdi_generation_command(
                tools["nvidia_ctk"], tools["nvidia_cdi_hook"], generated, selected
            ),
            host_env,
            timeout=120,
        )
        cdi = restrict_cdi(
            json.loads(generated.read_text()), selected, tools["nvidia_cdi_hook"]
        )
        write_json_new(local / "cdi" / "assigned-gpus.json", cdi)
        write_json_new(proofs / "assigned-cdi.json", cdi)
        extraction = safe_extract_zstd(
            payload_archive,
            local / "payload",
            tools["zstd"],
            compression=payload_compression,
            regular_byte_limit=request["payload"]["regular_bytes"],
        )
        require(
            extraction["regular_bytes"] <= request["payload"]["regular_bytes"],
            "Payload exceeded its immutable disk budget",
        )
        verified_count = verify_payload_manifest(manifest, local / "payload")
        reject_payload_shadowing(request.get("persistent_mounts", []), manifest)
        mount_targets = prepare_mount_targets(
            local / "payload", request.get("persistent_mounts", [])
        )
        base = podman_base(tools["podman"], local)
        podman_info = json.loads(
            run_checked([*base, "info", "--format=json"], host_env, timeout=120)
        )
        check_podman_info(podman_info, local)
        write_json_new(proofs / "podman-info.json", podman_info)
        image = run_checked(
            [
                *base,
                "import",
                "--quiet",
                "--arch=amd64",
                "--os=linux",
                str(base_archive),
            ],
            host_env,
            timeout=1800,
        ).strip()
        require(
            bool(re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", image)),
            "Podman import did not return an image digest",
        )
        inspect = json.loads(
            run_checked([*base, "image", "inspect", image], host_env, timeout=60)
        )[0]
        require(
            len(inspect.get("RootFS", {}).get("Layers", [])) == 1,
            "Base image is not a single flattened layer",
        )
        write_json_new(proofs / "imported-image.json", inspect)
        inner_env = request.get("inner_env", {})
        require(
            all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in inner_env.items()
            ),
            "Inner environment must contain strings",
        )
        require(
            "WANDB_API_KEY" not in inner_env,
            "Bind credentials separately from the public request",
        )
        image_environment.update(inner_env)
        # Publisher NVIDIA_VISIBLE_DEVICES=all must not activate a legacy hook
        # alongside explicit CDI. CUDA UUID masking is allocation-specific.
        image_environment = apply_runtime_masks(image_environment, selected)
        image_environment = allocation_environment(os.environ, image_environment)
        control = {
            "expected_uuids": [gpu["uuid"] for gpu in selected],
            "expected_torch": request.get("expected_torch", "2.9.1+cu126"),
            "python_prefix": request["gpu_probe_python_prefix"],
            "inner_command": request["inner_command"],
            "receipt_directory": "/run/pace-proof",
            "host_cgroup": Path("/proc/self/cgroup").read_text(),
            "host_mount_namespace": os.readlink("/proc/self/ns/mnt"),
            "host_user_namespace": os.readlink("/proc/self/ns/user"),
            "attestation": {
                "job_id": job_id,
                "node": os.environ["SLURMD_NODENAME"],
                "world_size": 4,
                "ownership": ownership,
                "cluster": ownership["cluster"],
                "payload_workspace": str(local / "payload" / "workspace"),
                "payload_archive_sha256": request["payload"]["archive"]["sha256"],
                "base_archive_sha256": base_receipt["rootfs_archive"]["sha256"],
                "source_image_digest": base_receipt["source_image_digest"],
                "gpus": health,
                "outer_request_sha256": arguments.request_sha256,
            },
        }
        write_json_new(proofs / "control.json", control)
        record.update(
            {
                "status": "HOST_ADMISSION_PASS",
                "gpu_uuids": control["expected_uuids"],
                "payload_verified_files": verified_count,
                "payload_extraction": extraction,
                "prepared_persistent_mount_targets": mount_targets,
                "imported_image": image,
            }
        )
        write_json_new(proofs / "host-admission.json", record)
        command = container_command(
            base, request, local, proofs, image, selected, image_environment
        )
        container_started = True
        run_checked(command, host_env, capture=False)
        require(
            (proofs / "gpu-admission.json").is_file(),
            "Container GPU admission receipt missing",
        )
        record.update({"status": "PASS", "finished_unix": time.time()})
        write_json_new(proofs / "completed.json", record)
    except BaseException as error:
        if container_started and base:
            record["container_cleanup"] = cleanup_container(
                base, f"pace-{request['namespace']}-{job_id}", host_env
            )
        record.update(
            {
                "status": "FAILED",
                "error_type": type(error).__name__,
                "error": str(error),
                "finished_unix": time.time(),
            }
        )
        write_json_new(proofs / "failed.json", record)
        raise


if __name__ == "__main__":
    try:
        main()
    except InterruptedLaunch as error:
        print(f"PHOENIX CONTAINER INTERRUPTED: {error}", file=sys.stderr)
        sys.exit(128 + error.signum)
    except (AdmissionError, KeyError, ValueError, OSError) as error:
        print(f"PHOENIX CONTAINER ADMISSION FAILED: {error}", file=sys.stderr)
        sys.exit(2)
