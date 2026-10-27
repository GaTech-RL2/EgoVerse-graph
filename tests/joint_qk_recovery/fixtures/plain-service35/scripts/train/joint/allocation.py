"""Explicit allocation admission shared by fresh joint smoke/full services.

Lambda retains its accepted node guard. PACE requires the current container's
kernel/UUID ownership proof and positive four-rank precision/collective receipt.
"""

import os
import json
import subprocess
import sys
from pathlib import Path

from scripts.train.joint.contract import bound, need, sha, write_once

PACE_ASSOCIATIONS = {
    "ice-slurm": (("coc", "coc-ice"), ("ece", "coe-ice")),
    "phoenix-slurm": (("gts-dxu345-rl2", "inferno"),),
}


def verify_pace(request, attestation, environment):
    policy = request["allocation"]
    cluster = policy["cluster"]
    world = request["world_size"]
    need(cluster in PACE_ASSOCIATIONS, "Unknown PACE cluster")
    need(
        (policy["account"], policy["qos"]) in PACE_ASSOCIATIONS[cluster],
        "Wrong cluster association",
    )
    need(
        world in (2, 4) and request["per_rank_batch"] * world == 32,
        "PACE requires the matched four-rank global32 recipe",
    )
    need(
        environment.get("SLURM_RESTART_COUNT", "0") == "0",
        "Fresh immutable allocation required",
    )
    need(
        attestation["status"] == "PASS"
        and attestation["world_size"] == world
        and attestation["job_id"] == environment["SLURM_JOB_ID"]
        and attestation["node"] == environment["SLURMD_NODENAME"]
        and attestation["cluster"] == cluster,
        "Wrong current container attestation",
    )
    ownership = attestation["ownership"]
    need(
        ownership["schema"] == "slurm-kernel-gpu-ownership-v3"
        and ownership["status"] == "PASS"
        and ownership["job_id"] == attestation["job_id"]
        and ownership["node"] == attestation["node"]
        and ownership["cluster"] == cluster,
        "Missing current kernel ownership proof",
    )
    need(
        all(ownership[key] == policy[key] for key in ("user", "uid", "account", "qos")),
        "Wrong current allocation owner/association",
    )
    resources = ownership["native_resources"]
    cpus = policy.get("cpus_per_task", 32)
    need(type(cpus) is int and (cpus == 32 or (cluster == "phoenix-slurm" and world == 2 and cpus == 16)),
         "Unsupported explicit CPU resource shape")
    need(
        resources["NumCPUs"] == str(policy.get("cpus_per_task", 32))
        and resources["MinMemoryNode"] == "192G"
        and resources["Requeue"] == "0",
        "Wrong dedicated CPU/memory/requeue contract",
    )
    partitions = set(resources["Partition"].split(","))
    expected_partitions = (
        ({"coe-gpu"} if policy["account"] == "ece" else {"ice-gpu"}) if cluster == "ice-slurm" else {"gpu-h200", "gpu-h100"}
    )
    need(
        partitions and partitions <= expected_partitions, "Unintended native partition"
    )
    limit = resources["TimeLimit"]
    days, clock = limit.split("-", 1) if "-" in limit else ("0", limit)
    hours, minutes, seconds = map(int, clock.split(":"))
    total_seconds = int(days) * 86400 + hours * 3600 + minutes * 60 + seconds
    need(
        0 < total_seconds <= (16 if cluster == "ice-slurm" else 24) * 3600,
        "Native wall time exceeds cluster policy",
    )
    probe = attestation["gpu_probe"]
    need(
        probe["status"] == "PASS" and probe["bf16_passed"] and probe["nccl_passed"],
        "Missing precision/collective admission",
    )
    gpus, ranks = ownership["gpus"], probe["ranks"]
    need(
        len(gpus) == len(ranks) == world
        and len({g["uuid"].lower() for g in gpus}) == world
        and len({g["minor"] for g in gpus}) == world
        and [r["uuid"].lower() for r in ranks] == [g["uuid"].lower() for g in gpus],
        "Wrong physical GPU/rank bijection",
    )
    for rank, record in enumerate(ranks):
        need(
            record["status"] == "PASS"
            and record["rank"] == rank
            and record["world_size"] == world
            and record["bf16"]
            and record["nccl_all_reduce"] == world*(world+1)/2
            and record["torch"] == "2.9.1+cu126"
            and record["compute_capability"][0] == 9,
            "Incomplete rank precision/collective proof",
        )
    if policy.get("kind") == "native_slurm":
        need(attestation["runtime_binding_sha256"] == policy["runtime_binding"]["sha256"],
             "Native runtime binding changed")
    else:
        need(
            attestation["payload_archive_sha256"] == policy["payload_archive_sha256"]
            and attestation["base_archive_sha256"] == policy["base_archive_sha256"],
            "Portable runtime/base binding changed",
        )
    return attestation


def admit_native(request, status_root, bootstrap):
    """Reuse exact ownership and real collective workers in an existing native runtime."""
    from scripts.clusters.pace import gpu_ownership
    policy = request["allocation"]
    need(status_root is not None and bootstrap is not None, "Native admission paths missing")
    bound(policy["runtime_binding"])
    world = request["world_size"]
    identity = {k: policy[k] for k in ("cluster", "user", "uid", "account", "qos")}
    identity["world_size"] = world
    ownership = gpu_ownership.verify(gpu_ownership.collect(), identity)
    directory = Path(status_root)/("native-admission-"+os.environ["SLURM_JOB_ID"])
    directory.mkdir()
    write_once(directory/"ownership.json", ownership)
    from scripts.clusters.pace.pace_podman import gpu_health
    xml = subprocess.run(["nvidia-smi", "-q", "-x", "-i",
                          ",".join(g["uuid"] for g in ownership["gpus"])],
                         check=True, capture_output=True, text=True, timeout=30).stdout
    (directory/"gpu-health.xml").write_text(xml)
    write_once(directory/"gpu-health.json", dict(status="PASS", devices=gpu_health(xml,ownership["gpus"])))
    uuids = [g["uuid"] for g in ownership["gpus"]]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(uuids)
    control = dict(expected_uuids=uuids, expected_torch="2.9.1+cu126",
                   receipt_directory=str(directory))
    control_path = directory/"control.json"
    write_once(control_path, control)
    environment = dict(os.environ, PHOENIX_BOUNDARY_CONTROL=str(control_path),
                       NVIDIA_TF32_OVERRIDE="0", TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="0")
    # Resolve the admitted operational module; old model snapshots need not
    # contain the current cluster boundary at all.
    from scripts.clusters.pace import gpu_admission
    worker = Path(gpu_admission.__file__).resolve()
    if request.get("operational_service"):
        need(worker.is_relative_to(Path(request["operational_service"]["root"]).resolve()),
             "GPU worker escaped the bound operational service")
    subprocess.run([sys.executable, str(bootstrap), "-m", "torch.distributed.run",
                    "--standalone", "--nnodes=1", "--nproc-per-node="+str(world),
                    "--max-restarts=0", str(bootstrap), str(worker), "--worker"],
                   check=True, env=environment)
    ranks = [json.loads((directory/f"gpu-rank-{i}.json").read_text()) for i in range(world)]
    attestation = dict(status="PASS", kind="native_slurm", world_size=world,
        job_id=os.environ["SLURM_JOB_ID"], node=os.environ["SLURMD_NODENAME"],
        cluster=policy["cluster"], ownership=ownership,
        runtime_binding_sha256=policy["runtime_binding"]["sha256"],
        gpu_probe=dict(status="PASS", bf16_passed=True, nccl_passed=True, ranks=ranks))
    verify_pace(request, attestation, os.environ)
    path = directory/"attestation.json"
    write_once(path, attestation)
    return dict(path=str(path), sha256=sha(path), job_id=attestation["job_id"],
                node=attestation["node"], cluster=attestation["cluster"])


def admit(request, status_root=None, bootstrap=None):
    policy = request.get("allocation")
    if policy is None:
        node = request.get("lambda_runtime", {}).get("node", "gpu21")
        need(
            os.environ["SLURMD_NODENAME"] == node
            and os.environ.get("SLURM_RESTART_COUNT", "0") == "0",
            "Wrong Lambda allocation",
        )
        return None
    if policy.get("kind") == "native_slurm":
        return admit_native(request, status_root, bootstrap)
    path = Path("/run/pace-proof/attestation.json")
    need(path.is_file() and not path.is_symlink(), "Missing current PACE attestation")
    attestation = bound(dict(path=str(path), sha256=sha(path)))
    verify_pace(request, attestation, os.environ)
    return dict(
        path=str(path),
        sha256=sha(path),
        job_id=attestation["job_id"],
        node=attestation["node"],
        cluster=attestation["cluster"],
    )
