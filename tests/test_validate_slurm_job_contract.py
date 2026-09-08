import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "train" / "validate_slurm_job_contract.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("validate_slurm_job_contract", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(**replacements):
    fields = {
        "JobId": "12345",
        "JobName": "action-flow",
        "Account": "gts-dxu345-rl2",
        "QOS": "inferno",
        "JobState": "RUNNING",
        "TimeLimit": "3-00:00:00",
        "Partition": "gpu-h200",
        "NumNodes": "1",
        "NumCPUs": "8",
        "NumTasks": "1",
        "CPUs/Task": "8",
        "ReqTRES": "cpu=8,mem=250G,node=1,billing=8,gres/gpu=1",
        "MinCPUsNode": "8",
        "MinMemoryNode": "250G",
        "Features": "H100|H200",
        "Command": "/task/launch.sh",
    }
    fields.update(replacements)
    return " ".join(f"{key}={value}" for key, value in fields.items()) + "\n"


def _expectations():
    return {
        "expected_job_id": "12345",
        "expected_account": "gts-dxu345-rl2",
        "expected_partition": "gpu-h200",
        "expected_qos": "inferno",
        "expected_cpus": 8,
        "expected_memory": "250G",
        "expected_time_limit": "3-00:00:00",
        "expected_constraint": "H100|H200",
    }


def test_contract_cli_writes_immutable_pass_evidence(tmp_path):
    record = tmp_path / "scontrol.txt"
    record.write_text(_record())
    output = tmp_path / "contract.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--record",
            str(record),
            "--expected-job-id",
            "12345",
            "--expected-account",
            "gts-dxu345-rl2",
            "--expected-partition",
            "gpu-h200",
            "--expected-qos",
            "inferno",
            "--expected-cpus",
            "8",
            "--expected-memory",
            "250G",
            "--expected-time-limit",
            "3-00:00:00",
            "--expected-constraint",
            "H100|H200",
            "--output",
            str(output),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    payload = json.loads(output.read_text())
    assert payload["status"] == "SLURM_JOB_CONTRACT_VALIDATED"
    assert payload["failures"] == []
    assert payload["observed"]["memory_bytes"] == 250 * 1024**3
    assert payload["observed"]["requested_tres_gpus"] == 1

    repeated = subprocess.run(
        completed.args,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert repeated.returncode != 0
    assert json.loads(output.read_text())["status"] == "SLURM_JOB_CONTRACT_VALIDATED"


@pytest.mark.parametrize(
    ("replacement", "failure_field"),
    [
        ({"JobId": "999"}, "job_id"),
        ({"Account": "coc"}, "account"),
        ({"Partition": "ice-gpu"}, "partition"),
        ({"QOS": "coc-ice"}, "qos"),
        ({"NumNodes": "2"}, "nodes"),
        ({"NumTasks": "2"}, "tasks"),
        ({"NumCPUs": "16"}, "cpus"),
        ({"CPUs/Task": "4"}, "cpus_per_task"),
        ({"MinCPUsNode": "4"}, "minimum_cpus_per_node"),
        ({"MinMemoryNode": "200G"}, "memory_bytes"),
        ({"Features": "H100"}, "constraint"),
        ({"TimeLimit": "2-00:00:00"}, "time_limit_seconds"),
        (
            {"ReqTRES": "cpu=8,mem=250G,node=2,billing=8,gres/gpu=1"},
            "requested_tres_nodes",
        ),
        (
            {"ReqTRES": "cpu=4,mem=250G,node=1,billing=8,gres/gpu=1"},
            "requested_tres_cpus",
        ),
        (
            {"ReqTRES": "cpu=8,mem=200G,node=1,billing=8,gres/gpu=1"},
            "requested_tres_memory_bytes",
        ),
        (
            {"ReqTRES": "cpu=8,mem=250G,node=1,billing=8,gres/gpu=2"},
            "requested_tres_gpus",
        ),
        (
            {
                "ReqTRES": (
                    "cpu=8,mem=250G,node=1,billing=8,"
                    "gres/gpu=1,gres/gpu:h100=1"
                )
            },
            "typed_gpu_tres",
        ),
    ],
)
def test_each_scheduler_contract_mismatch_fails(replacement, failure_field):
    module = _load_module()
    fields = module.parse_scontrol_record(_record(**replacement))
    _, _, failures = module.evaluate_contract(fields, **_expectations())
    assert failure_field in {failure["field"] for failure in failures}


def test_contract_cli_records_failure_and_exits_nonzero(tmp_path):
    record = tmp_path / "scontrol.txt"
    record.write_text(_record(Features="H200"))
    output = tmp_path / "contract.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--record",
            str(record),
            "--expected-job-id",
            "12345",
            "--expected-account",
            "gts-dxu345-rl2",
            "--expected-partition",
            "gpu-h200",
            "--expected-qos",
            "inferno",
            "--expected-cpus",
            "8",
            "--expected-memory",
            "250G",
            "--expected-time-limit",
            "3-00:00:00",
            "--expected-constraint",
            "H100|H200",
            "--output",
            str(output),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 1
    payload = json.loads(output.read_text())
    assert payload["status"] == "SLURM_JOB_CONTRACT_FAILED"
    assert payload["failures"] == [
        {"field": "constraint", "expected": "H100|H200", "observed": "H200"}
    ]


def test_parser_rejects_multiline_and_duplicate_fields():
    module = _load_module()
    with pytest.raises(module.ContractError, match="one-line"):
        module.parse_scontrol_record("JobId=1\nAccount=x\n")
    with pytest.raises(module.ContractError, match="repeats field JobId"):
        module.parse_scontrol_record("JobId=1 JobId=2")


def test_parser_accepts_empty_unrelated_slurm_fields():
    module = _load_module()
    fields = module.parse_scontrol_record(_record(StdErr=""))
    assert fields["StdErr"] == ""
    _, _, failures = module.evaluate_contract(fields, **_expectations())
    assert failures == []


def test_contract_still_rejects_empty_required_fields():
    module = _load_module()
    fields = module.parse_scontrol_record(_record(Account=""))
    with pytest.raises(module.ContractError, match="required scontrol field Account is empty"):
        module.evaluate_contract(fields, **_expectations())


@pytest.mark.parametrize(
    ("raw", "seconds"),
    [("3-00:00:00", 3 * 86400), ("16:00:00", 16 * 3600), ("30", 1800)],
)
def test_slurm_duration_forms(raw, seconds):
    module = _load_module()
    assert module.parse_slurm_duration(raw) == seconds


@pytest.mark.parametrize("constraint", ["A40", "A100-40GB", "A100-80GB", "L40S", "A40|L40S"])
def test_alternate_smoke_profile_requires_exact_allocation(constraint):
    module = _load_module()
    expected = {**_expectations(), "expected_constraint": constraint,
                "gpu_profile": "smoke-bf16", "run_kind": "smoke"}
    fields = module.parse_scontrol_record(_record(Features=constraint))
    accepted, _, failures = module.evaluate_contract(fields, **expected)
    assert failures == []
    assert accepted["gpu_profile"] == "smoke-bf16"
    assert accepted["run_kind"] == "smoke"
    with pytest.raises(module.ContractError, match="full allocations"):
        module.evaluate_contract(fields, **{**expected, "run_kind": "full"})
    with pytest.raises(module.ContractError, match="exactly"):
        module.evaluate_contract(fields, **{**expected, "gpu_profile": "h100-h200"})


@pytest.mark.parametrize("constraint", ["V100", "A40|V100", "A40|A40", "", "A100", "A40|"])
def test_alternate_smoke_profile_rejects_unknown_or_ambiguous_features(constraint):
    module = _load_module()
    with pytest.raises(module.ContractError):
        module.validate_gpu_profile("smoke-bf16", "smoke", constraint)
