"""Explicit allocation admission shared by fresh joint smoke/full services.

Lambda retains its accepted node guard. PACE requires the current container's
kernel/UUID ownership proof and positive four-rank precision/collective receipt.
"""

import os
from pathlib import Path

from scripts.train.joint.contract import bound, need, sha

PACE_ASSOCIATIONS = {
    "ice-slurm": ("coc", "coc-ice"),
    "phoenix-slurm": ("gts-dxu345-rl2", "inferno"),
}


def verify_pace(request, attestation, environment):
    policy = request["allocation"]
    cluster = policy["cluster"]
    need(cluster in PACE_ASSOCIATIONS, "Unknown PACE cluster")
    need(
        (policy["account"], policy["qos"]) == PACE_ASSOCIATIONS[cluster],
        "Wrong cluster association",
    )
    need(
        request["world_size"] == 4 and request["per_rank_batch"] == 8,
        "PACE requires the matched four-rank global32 recipe",
    )
    need(
        environment.get("SLURM_RESTART_COUNT", "0") == "0",
        "Fresh immutable allocation required",
    )
    need(
        attestation["status"] == "PASS"
        and attestation["world_size"] == 4
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
    need(
        resources["NumCPUs"] == "32"
        and resources["MinMemoryNode"] == "192G"
        and resources["Requeue"] == "0",
        "Wrong dedicated CPU/memory/requeue contract",
    )
    partitions = set(resources["Partition"].split(","))
    expected_partitions = (
        {"ice-gpu"} if cluster == "ice-slurm" else {"gpu-h200", "gpu-h100"}
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
        len(gpus) == len(ranks) == 4
        and len({g["uuid"].lower() for g in gpus}) == 4
        and len({g["minor"] for g in gpus}) == 4
        and [r["uuid"].lower() for r in ranks] == [g["uuid"].lower() for g in gpus],
        "Wrong physical GPU/rank bijection",
    )
    for rank, record in enumerate(ranks):
        need(
            record["status"] == "PASS"
            and record["rank"] == rank
            and record["world_size"] == 4
            and record["bf16"]
            and record["nccl_all_reduce"] == 10.0
            and record["torch"] == "2.9.1+cu126"
            and record["compute_capability"][0] == 9,
            "Incomplete rank precision/collective proof",
        )
    need(
        attestation["payload_archive_sha256"] == policy["payload_archive_sha256"]
        and attestation["base_archive_sha256"] == policy["base_archive_sha256"],
        "Portable runtime/base binding changed",
    )
    return attestation


def admit(request):
    policy = request.get("allocation")
    if policy is None:
        need(
            os.environ["SLURMD_NODENAME"] == "gpu21"
            and os.environ.get("SLURM_RESTART_COUNT", "0") == "0",
            "Wrong Lambda allocation",
        )
        return None
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
