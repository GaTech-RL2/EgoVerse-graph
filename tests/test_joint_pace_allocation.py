import pytest

from scripts.train.joint.allocation import verify_pace


def fixture():
    policy = dict(
        cluster="phoenix-slurm",
        account="gts-dxu345-rl2",
        qos="inferno",
        user="test",
        uid=42,
        payload_archive_sha256="a" * 64,
        base_archive_sha256="b" * 64,
    )
    uuids = [f"GPU-00000000-0000-0000-0000-{i:012d}" for i in range(4)]
    owner = dict(
        schema="slurm-kernel-gpu-ownership-v3",
        status="PASS",
        job_id="123",
        node="node1",
        **{k: policy[k] for k in ("cluster", "user", "uid", "account", "qos")},
        gpus=[dict(uuid=u, minor=i + 4) for i, u in enumerate(uuids)],
    )
    owner["native_resources"] = dict(
        NumCPUs="32",
        MinMemoryNode="192G",
        Requeue="0",
        Partition="gpu-h200,gpu-h100",
        TimeLimit="01:00:00",
    )
    ranks = [
        dict(
            status="PASS",
            rank=i,
            world_size=4,
            uuid=u,
            bf16=True,
            nccl_all_reduce=10.0,
            torch="2.9.1+cu126",
            compute_capability=[9, 0],
        )
        for i, u in enumerate(uuids)
    ]
    proof = dict(
        status="PASS",
        world_size=4,
        job_id="123",
        node="node1",
        cluster="phoenix-slurm",
        ownership=owner,
        gpu_probe=dict(status="PASS", bf16_passed=True, nccl_passed=True, ranks=ranks),
        payload_archive_sha256="a" * 64,
        base_archive_sha256="b" * 64,
    )
    return (
        dict(allocation=policy, world_size=4, per_rank_batch=8),
        proof,
        dict(SLURM_JOB_ID="123", SLURMD_NODENAME="node1"),
    )


def test_exact_current_container_is_admitted():
    request, proof, env = fixture()
    assert verify_pace(request, proof, env) is proof


@pytest.mark.parametrize(
    "damage",
    [
        "cluster",
        "account",
        "job",
        "node",
        "restart",
        "world",
        "owner",
        "uuid",
        "minor",
        "bf16",
        "nccl",
        "torch",
        "payload",
        "base",
        "cpu",
        "memory",
        "partition",
        "walltime",
    ],
)
def test_missing_or_changed_admission_is_rejected(damage):
    request, proof, env = fixture()
    if damage == "cluster":
        proof["cluster"] = "ice-slurm"
    elif damage == "account":
        request["allocation"]["account"] = "coc"
    elif damage == "job":
        env["SLURM_JOB_ID"] = "456"
    elif damage == "node":
        env["SLURMD_NODENAME"] = "node2"
    elif damage == "restart":
        env["SLURM_RESTART_COUNT"] = "1"
    elif damage == "world":
        request["world_size"] = 2
    elif damage == "owner":
        proof["ownership"]["uid"] = 0
    elif damage == "uuid":
        proof["gpu_probe"]["ranks"][0]["uuid"] = "GPU-other"
    elif damage == "minor":
        proof["ownership"]["gpus"][0]["minor"] = 5
    elif damage == "bf16":
        proof["gpu_probe"]["ranks"][0]["bf16"] = False
    elif damage == "nccl":
        proof["gpu_probe"]["nccl_passed"] = False
    elif damage == "torch":
        proof["gpu_probe"]["ranks"][0]["torch"] = "other"
    elif damage == "payload":
        proof["payload_archive_sha256"] = "c" * 64
    elif damage == "base":
        proof["base_archive_sha256"] = "c" * 64
    elif damage == "cpu":
        proof["ownership"]["native_resources"]["NumCPUs"] = "16"
    elif damage == "memory":
        proof["ownership"]["native_resources"]["MinMemoryNode"] = "128G"
    elif damage == "partition":
        proof["ownership"]["native_resources"]["Partition"] += ",gpu-v100"
    elif damage == "walltime":
        proof["ownership"]["native_resources"]["TimeLimit"] = "2-00:00:00"
    with pytest.raises(ValueError):
        verify_pace(request, proof, env)
