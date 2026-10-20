"""Synthetic namespace mismatch fixtures: no historical UUID is an allocation grant."""

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "scripts/clusters/pace"
sys.path.insert(0, str(ROOT))
admission = importlib.import_module("gpu_admission")
boundary = importlib.import_module("pace_podman")


def fixture(cluster="phoenix-slurm"):
    uuids = [f"GPU-00000000-0000-0000-0000-{i:012d}" for i in range(8)]
    minors = [1, 4, 5, 7]
    account, qos = (
        ("gts-dxu345-rl2", "inferno")
        if cluster == "phoenix-slurm"
        else ("coc", "coc-ice")
    )
    native = f"JobId=123 JobState=RUNNING UserId=test(42) Account={account} QOS={qos} NodeList=node1 BatchHost=node1 NumNodes=1 Restarts=0 SLUID=abc123 GRES=gpu:h200:4(IDX:2-3,5-6)"
    proc = {
        f"/proc/driver/nvidia/gpus/0000:01:{i:02x}.0/information": f"GPU UUID: {u}\nDevice Minor: {i}\nBus Location: 0000:01:{i:02x}.0\n"
        for i, u in enumerate(uuids)
    }
    xml = (
        "<nvidia_smi_log>"
        + "".join(
            f"<gpu><uuid>{uuids[i]}</uuid><minor_number>{i}</minor_number><pci><pci_bus_id>0000:01:{i:02x}.0</pci_bus_id></pci></gpu>"
            for i in minors
        )
        + "</nvidia_smi_log>"
    )
    evidence = dict(
        native=native,
        scheduler_config=f"ClusterName = {cluster}\n",
        environment=dict(
            SLURM_JOB_ID="123",
            SLURMD_NODENAME="node1",
            SLURM_JOB_GPUS="2,3,5,6",
            CUDA_VISIBLE_DEVICES="0,1,2,3",
        ),
        cgroup="0::/slurm/abc123/step_batch\n",
        inventory="\n".join(
            f"{r},{uuids[i]},NVIDIA H200,9.0,143771" for r, i in enumerate(minors)
        ),
        xml=xml,
        proc=proc,
        devices=[
            dict(
                minor=i,
                major=195,
                character_device=True,
                read_open_permitted=i in minors,
            )
            for i in range(8)
        ],
    )
    return evidence, dict(
        user="test", uid=42, account=account, qos=qos, cluster=cluster
    )


@pytest.mark.parametrize("cluster", ["phoenix-slurm", "ice-slurm"])
def test_current_allocation_proves_physical_ownership_despite_filtered_ordinals(
    cluster,
):
    e, r = fixture(cluster)
    proof = boundary.assigned_gpus(e, r)
    assert proof["native_indices"] == ["2", "3", "5", "6"]
    assert [gpu["index"] for gpu in proof["gpus"]] == [0, 1, 2, 3]
    assert [gpu["minor"] for gpu in proof["gpus"]] == [1, 4, 5, 7]


@pytest.mark.parametrize(
    "damage",
    [
        "owner",
        "job",
        "cgroup",
        "idx",
        "extra_device",
        "missing_owned",
        "xml_minor",
        "duplicate_uuid",
        "family",
        "missing_proc",
        "cluster",
        "uuid_env",
    ],
)
def test_ownership_mismatches_fail_closed(damage):
    e, r = fixture()
    if damage == "owner":
        r["uid"] = 0
    elif damage == "job":
        e["environment"]["SLURM_JOB_ID"] = "12"
    elif damage == "cgroup":
        e["cgroup"] = "0::/slurm/not-abc123/step_batch"
    elif damage == "idx":
        e["environment"]["SLURM_JOB_GPUS"] = "0-3"
    elif damage == "extra_device":
        e["devices"][0]["read_open_permitted"] = True
    elif damage == "missing_owned":
        e["devices"][1]["read_open_permitted"] = False
    elif damage == "xml_minor":
        e["xml"] = e["xml"].replace(
            "<minor_number>1</minor_number>", "<minor_number>0</minor_number>"
        )
    elif damage == "duplicate_uuid":
        e["inventory"] += "\n" + e["inventory"].splitlines()[0]
    elif damage == "family":
        e["inventory"] = e["inventory"].replace("H200", "V100")
    elif damage == "missing_proc":
        e["proc"].pop(next(iter(e["proc"])))
    elif damage == "cluster":
        e["scheduler_config"] = "ClusterName = ice-slurm\n"
    elif damage == "uuid_env":
        e["environment"]["SLURM_JOB_GPUS"] = ",".join(
            f"GPU-99999999-0000-0000-0000-{i:012d}" for i in range(4)
        )
    with pytest.raises((boundary.AdmissionError, ValueError, KeyError)):
        boundary.assigned_gpus(e, r)


def test_cdi_uses_owned_physical_minors_not_nvml_ordinals():
    e, r = fixture()
    selected = boundary.assigned_gpus(e, r)["gpus"]
    cdi = dict(
        kind="nvidia.com/gpu",
        devices=[
            dict(
                name=g["uuid"],
                containerEdits=dict(
                    deviceNodes=[dict(path=f"/dev/nvidia{g['minor']}")]
                ),
            )
            for g in selected
        ],
    )
    assert len(boundary.restrict_cdi(cdi, selected, "hook")["devices"]) == 4
    cdi["devices"][0]["containerEdits"]["deviceNodes"][0]["path"] = "/dev/nvidia0"
    with pytest.raises(boundary.AdmissionError, match="unallocated"):
        boundary.restrict_cdi(cdi, selected, "hook")


def test_native_sluid_cgroup_is_required_and_inherited():
    host = "0::/slurm/abc123/step_batch\n"
    admission.check_slurm_cgroup(
        host,
        "0::/slurm/abc123/step_batch/container\n",
        "123",
        "abc123",
        "phoenix-slurm",
    )
    with pytest.raises(RuntimeError, match="not identified"):
        admission.check_slurm_cgroup(
            "0::/slurm/job_123/step_batch\n",
            "0::/slurm/job_123/step_batch/container\n",
            "123",
            "abc123",
            "phoenix-slurm",
        )
    with pytest.raises(RuntimeError, match="escaped"):
        admission.check_slurm_cgroup(
            host,
            "0::/slurm/abc123/step_other/container\n",
            "123",
            "abc123",
            "phoenix-slurm",
        )
