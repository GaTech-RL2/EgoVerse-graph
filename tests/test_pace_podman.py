"""Boundary checks use tiny archives and subprocess fixtures; no cluster jobs."""

import ast
import gzip
import importlib.util
import io
import json
import os
import signal
import sys
import tarfile
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1] / "scripts/clusters/pace"
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("pace_podman", ROOT / "pace_podman.py")
boundary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(boundary)
GPU_SPEC = importlib.util.spec_from_file_location(
    "gpu_admission", ROOT / "gpu_admission.py"
)
gpu = importlib.util.module_from_spec(GPU_SPEC)
GPU_SPEC.loader.exec_module(gpu)


def uuids():
    return [f"GPU-00000000-0000-0000-0000-{index:012d}" for index in range(8)]


def inventory():
    rows = boundary.parse_inventory(
        "\n".join(
            f"{index}, {uuid}, NVIDIA H100 80GB HBM3, 9.0, 81559"
            for index, uuid in enumerate(uuids())
        )
    )
    return [{**row, "minor": row["index"]} for row in rows]


def test_binding_rejects_changed_bytes(tmp_path):
    path = tmp_path / "receipt.json"
    path.write_text('{"status":"PASS"}')
    reference = {
        "path": str(path),
        "sha256": boundary.digest(path),
        "size_bytes": path.stat().st_size,
    }
    assert boundary.verified_file(reference) == path
    path.write_text('{"status":"FAIL"}')
    with pytest.raises(boundary.AdmissionError, match="SHA-256 changed"):
        boundary.verified_file(reference)


def test_binding_rejects_symlink(tmp_path):
    path = tmp_path / "real"
    path.write_text("immutable")
    alias = tmp_path / "alias"
    alias.symlink_to(path)
    with pytest.raises(boundary.AdmissionError, match="regular input"):
        boundary.verified_file({"path": str(alias), "sha256": boundary.digest(path)})


def test_publisher_env_preserves_values():
    receipt = {
        "status": "PASS",
        "source_image_reference": "nvcr.io/nvidia/pytorch:25.06-py3",
        "source_image_digest": "sha256:" + "a" * 64,
        "config_env": [
            "PATH=/usr/bin:/bin",
            "SHELL_VALUE=$(do not execute) `literal`",
            "EQUALS=a=b",
        ],
    }
    result = boundary.publisher_environment(receipt)
    assert result["EQUALS"] == "a=b"
    assert result["SHELL_VALUE"] == "$(do not execute) `literal`"


def test_space_budget_includes_both_vfs_copies():
    assert (
        boundary.required_free_bytes(30 * boundary.GIB, 25 * boundary.GIB)
        == 101 * boundary.GIB
    )
    assert boundary.required_free_bytes(None, 25 * boundary.GIB) == 200 * boundary.GIB


def test_actual_base_receipt_uses_pinned_reference_and_separate_tag():
    image_digest = "sha256:" + "3" * 64
    receipt = {
        "status": "PASS",
        "source_tag": "nvcr.io/nvidia/pytorch:25.06-py3",
        "source_image_reference": "nvcr.io/nvidia/pytorch@" + image_digest,
        "source_image_digest": image_digest,
        "platform": {"os": "linux", "architecture": "amd64"},
        "config_env": ["PATH=/usr/bin", "NVIDIA_PYTORCH_VERSION=25.06"],
    }
    assert boundary.publisher_environment(receipt)["NVIDIA_PYTORCH_VERSION"] == "25.06"
    receipt["source_image_digest"] = "sha256:" + "4" * 64
    with pytest.raises(boundary.AdmissionError, match="mismatch"):
        boundary.publisher_environment(receipt)


def test_actual_slurm_environment_overrides_historical_node():
    result = boundary.allocation_environment(
        {
            "SLURM_JOB_ID": "1416",
            "SLURMD_NODENAME": "phnx-allocated",
            "SLURM_JOB_GPUS": "4-7",
        },
        {"SLURM_JOB_ID": "411521", "SLURMD_NODENAME": "gpu19", "PIXEL_NODE": "gpu19"},
    )
    assert result["SLURM_JOB_ID"] == "1416"
    assert result["SLURMD_NODENAME"] == result["PIXEL_NODE"] == "phnx-allocated"


def test_runtime_masks_disable_publisher_tf32_and_all_gpu_defaults():
    result = boundary.apply_runtime_masks(
        {
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1",
            "NVIDIA_VISIBLE_DEVICES": "all",
            "PATH": "/publisher/bin",
        },
        inventory()[4:],
    )
    assert (
        result["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"]
        == result["NVIDIA_TF32_OVERRIDE"]
        == "0"
    )
    assert result["NVIDIA_VISIBLE_DEVICES"] == "void"
    assert result["CUDA_VISIBLE_DEVICES"] == ",".join(uuids()[4:])
    assert result["PATH"] == "/publisher/bin"


def test_local_filesystem_uses_longest_mount_prefix():
    mountinfo = "1 0 8:1 / / rw - xfs /dev/root rw\n2 1 0:1 / /scratch rw - nfs server:/scratch rw\n3 1 8:2 / /scratch/local-temp rw - ext4 /dev/nvme rw\n"
    assert (
        boundary.local_filesystem(Path("/scratch/local-temp/job123"), mountinfo)
        == "ext4"
    )
    assert boundary.local_filesystem(Path("/scratch/shared-temp"), mountinfo) == "nfs"


def xml_health(error="0", recovery="None"):
    return (
        "<nvidia_smi_log>"
        + "".join(
            f"<gpu><uuid>{uuid}</uuid><ecc_mode><current_ecc>Enabled</current_ecc></ecc_mode><mig_mode><current_mig>Disabled</current_mig></mig_mode><ecc_errors><volatile><sram_uncorrectable_parity>{error}</sram_uncorrectable_parity><sram_uncorrectable_secded>0</sram_uncorrectable_secded><dram_uncorrectable>0</dram_uncorrectable></volatile><aggregate><sram_uncorrectable_parity>0</sram_uncorrectable_parity><dram_uncorrectable>0</dram_uncorrectable></aggregate><channel_repair_pending>No</channel_repair_pending></ecc_errors><gpu_recovery_action>{recovery}</gpu_recovery_action><remapped_rows><remapped_row_pending>No</remapped_row_pending></remapped_rows></gpu>"
            for uuid in uuids()[:4]
        )
        + "</nvidia_smi_log>"
    )


def test_new_driver_ecc_schema_and_recovery():
    assert len(boundary.gpu_health(xml_health(), inventory()[:4])) == 4
    with pytest.raises(boundary.AdmissionError, match="Uncorrectable ECC"):
        boundary.gpu_health(xml_health(error="1"), inventory()[:4])
    with pytest.raises(boundary.AdmissionError, match="recovery action"):
        boundary.gpu_health(xml_health(recovery="Reset"), inventory()[:4])


def cdi_fixture():
    return {
        "cdiVersion": "0.8.0",
        "kind": "nvidia.com/gpu",
        "containerEdits": {
            "hooks": [
                {
                    "path": "/reviewed/nvidia-cdi-hook",
                    "args": ["nvidia-cdi-hook", "update-ldcache"],
                }
            ],
            "deviceNodes": [{"path": "/dev/nvidiactl"}],
        },
        "devices": [
            {
                "name": uuid,
                "containerEdits": {"deviceNodes": [{"path": f"/dev/nvidia{index}"}]},
            }
            for index, uuid in enumerate(uuids())
        ]
        + [
            {
                "name": "all",
                "containerEdits": {"deviceNodes": [{"path": "/dev/nvidia0"}]},
            }
        ],
    }


def test_cdi_retains_only_four_assigned_uuid_devices():
    restricted = boundary.restrict_cdi(
        cdi_fixture(), inventory()[4:], "/reviewed/nvidia-cdi-hook"
    )
    assert [device["name"] for device in restricted["devices"]] == uuids()[4:]
    assert "all" not in json.dumps([device["name"] for device in restricted["devices"]])


def test_cdi_generation_itself_requests_only_allocated_uuids():
    command = boundary.cdi_generation_command(
        "ctk", "hook", "output.json", inventory()[4:]
    )
    assert command.count("--device-id") == 4
    assert "--no-all-device" in command
    assert [
        command[index + 1]
        for index, item in enumerate(command)
        if item == "--device-id"
    ] == uuids()[4:]


def test_cdi_rejects_hidden_unallocated_gpu_and_hook():
    original = cdi_fixture()
    original["containerEdits"]["deviceNodes"].append({"path": "/dev/nvidia0"})
    with pytest.raises(boundary.AdmissionError, match="unallocated"):
        boundary.restrict_cdi(original, inventory()[4:], "/reviewed/nvidia-cdi-hook")
    original = cdi_fixture()
    with pytest.raises(boundary.AdmissionError, match="unverified hook"):
        boundary.restrict_cdi(original, inventory()[4:], "/wrong/hook")


@pytest.fixture
def fake_zstd(tmp_path):
    script = tmp_path / "zstd"
    script.write_text('#!/bin/sh\nexec cat "$3"\n')
    script.chmod(0o700)
    return str(script)


def make_tar(path, members):
    with tarfile.open(path, "w") as archive:
        for name, body, kind in members:
            item = tarfile.TarInfo(name)
            item.mode = 0o755 if kind == "directory" else 0o644
            if kind == "file":
                data = body.encode()
                item.size = len(data)
                archive.addfile(item, io.BytesIO(data))
            elif kind == "symlink":
                item.type = tarfile.SYMTYPE
                item.linkname = body
                archive.addfile(item)
            elif kind == "directory":
                item.type = tarfile.DIRTYPE
                archive.addfile(item)
            else:
                item.type = tarfile.FIFOTYPE
                archive.addfile(item)


def test_safe_extract_preserves_external_python_leaf_without_dereference(
    tmp_path, fake_zstd
):
    archive, output = tmp_path / "payload.tar", tmp_path / "output"
    output.mkdir()
    make_tar(
        archive,
        [
            ("workspace/runtime/bin/python", "/usr/bin/python", "symlink"),
            ("workspace/runtime/lib/content.py", "print('exact')\n", "file"),
        ],
    )
    result = boundary.safe_extract_zstd(archive, output, fake_zstd)
    assert result["archive_members"] == 2
    assert (output / "workspace/runtime/bin/python").readlink() == Path(
        "/usr/bin/python"
    )
    assert (
        output / "workspace/runtime/lib/content.py"
    ).read_text() == "print('exact')\n"


@pytest.mark.parametrize(
    "members",
    [
        [("../escape", "bad", "file")],
        [("/tmp/escape", "bad", "file")],
        [("not-workspace/file", "bad", "file")],
        [("workspace/pipe", "", "fifo")],
        [("workspace/a", "/tmp", "symlink"), ("workspace/a/pwn", "bad", "file")],
        [("workspace/a", "one", "file"), ("workspace/a", "two", "file")],
    ],
)
def test_safe_extract_rejects_unsafe_archive(tmp_path, fake_zstd, members):
    archive, output = tmp_path / "payload.tar", tmp_path / "output"
    output.mkdir()
    make_tar(archive, members)
    with pytest.raises((boundary.AdmissionError, FileExistsError, NotADirectoryError)):
        boundary.safe_extract_zstd(archive, output, fake_zstd)
    assert not (tmp_path / "escape").exists()


def test_gzip_payload_uses_same_safe_stream_and_preserves_python_link(
    tmp_path, monkeypatch
):
    archive, output = tmp_path / "payload.tar", tmp_path / "output"
    output.mkdir()
    make_tar(
        archive,
        [
            ("workspace/lib/value.py", "exact", "file"),
            ("workspace/bin/python", "/usr/bin/python", "symlink"),
        ],
    )
    compressed = tmp_path / "payload.tar.gz"
    compressed.write_bytes(gzip.compress(archive.read_bytes()))
    monkeypatch.setattr(
        boundary.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail(
            "Gzip extraction must use its stdlib stream"
        ),
    )
    result = boundary.safe_extract_zstd(
        compressed, output, "/unused/zstd", compression="gzip", regular_byte_limit=5
    )
    assert result["regular_bytes"] == 5
    assert (output / "workspace/lib/value.py").read_text() == "exact"
    assert (output / "workspace/bin/python").readlink() == Path("/usr/bin/python")


@pytest.mark.parametrize("name", ["../escape", "/tmp/escape", "workspace/../escape"])
def test_gzip_payload_rejects_traversal(tmp_path, name):
    archive, output = tmp_path / "payload.tar", tmp_path / "output"
    output.mkdir()
    make_tar(archive, [(name, "forbidden", "file")])
    compressed = tmp_path / "payload.tar.gz"
    compressed.write_bytes(gzip.compress(archive.read_bytes()))
    with pytest.raises(boundary.AdmissionError, match="Escaping"):
        boundary.safe_extract_zstd(
            compressed, output, "/unused/zstd", compression="gzip"
        )
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize("compression", ["zstd", "gzip"])
def test_runtime_byte_budget_stops_before_oversized_member_write(
    tmp_path, fake_zstd, compression
):
    archive, output = tmp_path / "payload.tar", tmp_path / "output"
    output.mkdir()
    make_tar(
        archive,
        [("workspace/first", "abc", "file"), ("workspace/second", "xyz", "file")],
    )
    if compression == "gzip":
        compressed = tmp_path / "payload.tar.gz"
        compressed.write_bytes(gzip.compress(archive.read_bytes()))
        archive = compressed
    with pytest.raises(boundary.AdmissionError, match="regular-byte budget"):
        boundary.safe_extract_zstd(
            archive, output, fake_zstd, compression=compression, regular_byte_limit=5
        )
    assert (output / "workspace/first").read_text() == "abc"
    assert not (output / "workspace/second").exists()


def test_gzip_crc_failure_is_not_ignored_after_tar_end(tmp_path):
    archive, output = tmp_path / "payload.tar", tmp_path / "output"
    output.mkdir()
    make_tar(archive, [("workspace/value", "exact", "file")])
    data = bytearray(gzip.compress(archive.read_bytes()))
    data[-8] ^= 1
    compressed = tmp_path / "bad.tar.gz"
    compressed.write_bytes(data)
    with pytest.raises(gzip.BadGzipFile):
        boundary.safe_extract_zstd(
            compressed, output, "/unused/zstd", compression="gzip"
        )


def test_only_reviewed_payload_compression_formats_are_allowed(tmp_path):
    with pytest.raises(
        boundary.AdmissionError, match="Unsupported payload compression"
    ):
        boundary.safe_extract_zstd(
            tmp_path / "payload", tmp_path, "zstd", compression="auto"
        )


def test_manifest_validates_extracted_bytes(tmp_path):
    payload = tmp_path / "payload"
    file = payload / "workspace/source/file.py"
    file.parent.mkdir(parents=True)
    file.write_text("exact\n")
    manifest = tmp_path / "manifest"
    manifest.write_text(f"{boundary.digest(file)}  workspace/source/file.py\n")
    assert boundary.verify_payload_manifest(manifest, payload) == 1
    file.write_text("changed\n")
    with pytest.raises(boundary.AdmissionError, match="hash changed"):
        boundary.verify_payload_manifest(manifest, payload)


def test_persistent_mount_cannot_mask_archived_source(tmp_path):
    manifest = tmp_path / "manifest"
    manifest.write_text("a" * 64 + "  workspace/source/train.py\n")
    boundary.reject_payload_shadowing(
        [{"container": "/workspace/operations"}], manifest
    )
    with pytest.raises(boundary.AdmissionError, match="shadow"):
        boundary.reject_payload_shadowing(
            [{"container": "/workspace/source"}], manifest
        )


def test_nested_operational_mount_target_is_created_before_workspace_ro(tmp_path):
    payload, persistent = tmp_path / "payload", tmp_path / "persistent"
    (payload / "workspace/users/user/tasks/model").mkdir(parents=True)
    persistent.mkdir()
    container = "/workspace/users/user/tasks/model/pace-operations-v1"
    result = boundary.prepare_mount_targets(
        payload, [{"host": str(persistent), "container": container, "read_only": False}]
    )
    assert result == [container]
    assert (payload / container.lstrip("/")).is_dir()


def test_operational_mount_cannot_traverse_archive_symlink(tmp_path):
    payload, persistent = tmp_path / "payload", tmp_path / "persistent"
    (payload / "workspace").mkdir(parents=True)
    persistent.mkdir()
    (payload / "workspace/alias").symlink_to(tmp_path)
    with pytest.raises(boundary.AdmissionError, match="symlink"):
        boundary.prepare_mount_targets(
            payload,
            [
                {
                    "host": str(persistent),
                    "container": "/workspace/alias/escape",
                    "read_only": False,
                }
            ],
        )
    assert not (tmp_path / "escape").exists()


def test_container_argv_has_four_cdi_devices_and_opaque_no_shell_environment(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    credential = tmp_path / "netrc"
    credential.write_text("private fixture")
    credential.chmod(0o600)
    request = {
        "namespace": "test-v1",
        "credential": {"host": str(credential), "container": "/run/pixel-wandb.netrc"},
        "persistent_mounts": [
            {
                "host": str(tmp_path),
                "container": "/workspace/task/operations",
                "read_only": False,
            }
        ],
    }
    command = boundary.container_command(
        ["podman"],
        request,
        tmp_path,
        tmp_path,
        "sha256:" + "a" * 64,
        inventory()[4:],
        {"LITERAL": "$(not-a-command) `literal`"},
    )
    assert command.count("--device") == 4
    assert [
        command[index + 1] for index, value in enumerate(command) if value == "--device"
    ] == [f"nvidia.com/gpu={uuid}" for uuid in uuids()[4:]]
    assert "--privileged" not in command
    assert not any("subuid" in argument for argument in command)
    assert "LITERAL=$(not-a-command) `literal`" in command
    assert "--pull=never" in command
    assert any("dst=/workspace,ro" in argument for argument in command)


def test_host_python_39_compatibility():
    for name in ("pace_podman.py", "gpu_admission.py"):
        ast.parse((ROOT / name).read_text(), feature_version=(3, 9))


def test_rootless_info_rejects_nonlocal_and_privileged_storage(tmp_path):
    information = {
        "host": {"security": {"rootless": True}},
        "store": {
            "graphDriverName": "vfs",
            "graphRoot": str(tmp_path / "storage"),
            "runRoot": str(tmp_path / "run"),
        },
    }
    boundary.check_podman_info(information, tmp_path)
    information["store"]["graphRoot"] = "/scratch/shared-storage"
    with pytest.raises(boundary.AdmissionError, match="graph root"):
        boundary.check_podman_info(information, tmp_path)
    information["host"]["security"]["rootless"] = False
    with pytest.raises(boundary.AdmissionError, match="not rootless"):
        boundary.check_podman_info(information, tmp_path)


def test_cuda_uuid_uses_raw_cuda_bytes():
    expected = uuids()[3]
    raw = uuid.UUID(expected[4:]).bytes
    assert (
        gpu.cuda_uuid(SimpleNamespace(uuid=SimpleNamespace(bytes=list(raw))))
        == expected
    )


def test_slurm_cgroup_requires_same_ancestor():
    host = "0::/slurm/uid42/job_123/step_batch\n"
    gpu.check_slurm_cgroup(
        host, "0::/slurm/uid42/job_123/step_batch/container\n", "123"
    )
    with pytest.raises(RuntimeError, match="escaped"):
        gpu.check_slurm_cgroup(host, "0::/user.slice/podman.scope\n", "123")
    with pytest.raises(RuntimeError, match="not identified"):
        gpu.check_slurm_cgroup("0::/\n", "0::/\n", "123")


def test_controller_attests_only_after_all_workers_use_scientific_prefix(
    tmp_path, monkeypatch
):
    prefix = ["/pinned/python", "/pinned/runtime_bootstrap.py"]
    control = {
        "expected_uuids": uuids()[:4],
        "python_prefix": prefix,
        "receipt_directory": str(tmp_path),
        "inner_command": ["/reviewed/inner", "--request", "opaque.json"],
        "attestation": {
            "job_id": "123",
            "world_size": 4,
            "payload_archive_sha256": "a" * 64,
        },
    }
    calls, executions = [], []

    def child(command, env):
        rank = int(env["RANK"])
        calls.append((command, env))
        gpu.write_new(
            tmp_path / f"gpu-rank-{rank}.json",
            {"status": "PASS", "uuid": uuids()[rank]},
        )
        return SimpleNamespace(poll=lambda: 0, returncode=0, wait=lambda timeout=10: 0)

    monkeypatch.setattr(gpu, "validate_base", lambda _: {"positive": True})
    monkeypatch.setattr(gpu.subprocess, "Popen", child)
    monkeypatch.setattr(gpu.os, "execvpe", lambda *args: executions.append(args))
    gpu.controller(control)
    assert len(calls) == 4
    assert all(command[:2] == prefix for command, _ in calls)
    assert [env["LOCAL_RANK"] for _, env in calls] == ["0", "1", "2", "3"]
    attestation = json.loads((tmp_path / "attestation.json").read_text())
    assert (
        attestation["status"] == "PASS"
        and attestation["gpu_probe"]["nccl_passed"] is True
    )
    assert executions[0][1] == control["inner_command"]


def test_controller_itself_uses_frozen_scientific_prefix(monkeypatch):
    executions = []
    monkeypatch.setattr(gpu.os, "execvpe", lambda *args: executions.append(args))
    prefix = ["/pinned/python", "/frozen/runtime_bootstrap.py"]
    gpu.bootstrap({"python_prefix": prefix})
    assert executions[0][1] == [
        *prefix,
        "/run/pace-boundary/gpu_admission.py",
        "--controller",
    ]


def test_controller_failed_worker_never_executes_inner(tmp_path, monkeypatch):
    control = {
        "expected_uuids": uuids()[:4],
        "python_prefix": ["/pinned/python"],
        "receipt_directory": str(tmp_path),
        "inner_command": ["/reviewed/inner"],
        "attestation": {"job_id": "123"},
    }
    executions = []
    monkeypatch.setattr(gpu, "validate_base", lambda _: {})
    monkeypatch.setattr(
        gpu.subprocess,
        "Popen",
        lambda *args, **kwargs: SimpleNamespace(
            poll=lambda: 1, returncode=1, wait=lambda timeout=10: 1
        ),
    )
    monkeypatch.setattr(gpu.os, "execvpe", lambda *args: executions.append(args))
    with pytest.raises(RuntimeError, match="admission failed"):
        gpu.controller(control)
    assert not executions and not (tmp_path / "attestation.json").exists()


def test_subprocess_error_has_bounded_diagnostics():
    with pytest.raises(boundary.AdmissionError, match="fixture failure"):
        boundary.run_checked(
            [
                sys.executable,
                "-c",
                "import sys; print('fixture failure',file=sys.stderr); sys.exit(7)",
            ],
            {},
        )


def test_term_is_forwarded_and_child_reaped_before_return(tmp_path):
    marker, pid_file = tmp_path / "terminated", tmp_path / "pid"
    previous = signal.getsignal(signal.SIGTERM)
    program = (
        "import os,signal,time,sys; from pathlib import Path; "
        + f"Path({str(pid_file)!r}).write_text(str(os.getpid())); "
        + f"signal.signal(signal.SIGTERM, lambda *args: (Path({str(marker)!r}).write_text('forwarded'), sys.exit(143))); "
        + "time.sleep(.15); os.kill(os.getppid(),signal.SIGTERM); time.sleep(30)"
    )
    with pytest.raises(boundary.InterruptedLaunch):
        boundary.run_checked([sys.executable, "-c", program], dict(os.environ))
    assert marker.read_text() == "forwarded"
    assert signal.getsignal(signal.SIGTERM) == previous
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)
