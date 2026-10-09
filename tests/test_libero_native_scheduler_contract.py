import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "guard", Path(__file__).parents[1] / "scripts/train/validate_slurm_job_contract.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
PROFILE = "libero/action_flow_libero10_h240_euler50_dithalf_80k_s42"


def inputs():
    fields = m.parse_scontrol_record(
        (
            Path(__file__).parent / "fixtures/libero_native_scontrol_408787.txt"
        ).read_text()
    )
    args = dict(
        expected_job_id="408787",
        expected_account="loaner",
        expected_partition="rl2",
        expected_qos="loaner-normal",
        expected_cpus=8,
        expected_memory="128G",
        expected_time_limit="02:00:00",
        expected_constraint="(null)",
        native_profile=PROFILE,
        gpu_probe=dict(
            status="PASSED",
            gpu_name="NVIDIA H100 80GB HBM3",
            world_size=1,
            rank=0,
            local_rank=0,
            bf16_supported=True,
            bf16_forward_backward=dict(finite=True),
        ),
    )
    return fields, args


def test_native_actual_h100_no_constraint():
    f, a = inputs()
    assert not m.evaluate_contract(f, **a)[2]


@pytest.mark.parametrize(
    "change",
    [
        "wrongprofile",
        "missingprobe",
        "wronggpu",
        "nonfinite",
        "wrongallocation",
        "constraint",
        "planar",
    ],
)
def test_reject_contract_drift(change):
    f, a = inputs()
    if change == "wrongprofile":
        a["native_profile"] = "planar"
    if change == "missingprobe":
        a["gpu_probe"] = None
    if change == "wronggpu":
        a["gpu_probe"]["gpu_name"] = "NVIDIA H200"
    if change == "nonfinite":
        a["gpu_probe"]["bf16_forward_backward"]["finite"] = False
    if change == "wrongallocation":
        f["GRES"] = "gpu:a40:1(IDX:3)"
    if change == "constraint":
        f["Features"] = "H100"
    if change == "planar":
        a["native_profile"] = None
    try:
        failures = m.evaluate_contract(f, **a)[2]
    except m.ContractError:
        return
    assert failures


@pytest.mark.parametrize(
    "field,value",
    [
        ("JOB_GRES", "gpu:h100:2"),
        ("TresPerNode", "gres/gpu:h200:1"),
        ("Nodes", "gpu18"),
        ("GRES", "gpu:h100:1"),
        ("AllocTRES", "cpu=4,mem=128G,node=1"),
        ("ReqTRES", "cpu=8,mem=128G,node=1,gres/gpu=1"),
    ],
)
def test_actual_allocation_drift(field, value):
    f, a = inputs()
    f[field] = value
    try:
        failures = m.evaluate_contract(f, **a)[2]
    except m.ContractError:
        return
    assert failures
