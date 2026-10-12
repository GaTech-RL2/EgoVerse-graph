"""Render one data-gated, six-arm L40S pusher training/evaluation workflow."""
import argparse
import copy
import json
from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[2]


def workflow(commit, name, regime, seed):
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or not re.fullmatch(r"[a-z0-9-]{1,23}", name):
        raise ValueError("Expected a full source commit and short workflow name")
    spec = yaml.safe_load((ROOT / "egomimic/hydra_configs/benchmark/pusher_triangle_holdout.yaml").read_text())
    assert regime in spec["regimes"] and seed in spec["training_seeds"]
    r2 = {"grabber-arc-r2-20260916": {"R2_ACCESS_KEY_ID": "r2_access_key_id",
        "R2_SECRET_ACCESS_KEY": "r2_secret_access_key", "R2_ENDPOINT_URL": "r2_endpoint_url"}}
    gate_entry = """set -Eeuo pipefail
set +x
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
python -m venv /workspace/emimic
source /workspace/emimic/bin/activate
pip install --disable-pip-version-check boto3==1.43.98 numpy==2.3.5
python /tmp/wait_data.py /tmp/campaign.json "$REGIME" '{{output}}'
"""
    tasks = [{"name": "data-gate", "resource": "gate", "image": "docker.io/library/python:3.11-slim",
        "credentials": r2, "environment": {"SOURCE_COMMIT": commit, "REGIME": regime},
        "command": ["bash"], "args": ["/tmp/gate.sh"],
        "files": [{"path": "/tmp/gate.sh", "contents": gate_entry},
                  {"path": "/tmp/wait_data.py", "contents": (ROOT / "scripts/benchmarks/wait_pusher_dataset.py").read_text()},
                  {"path": "/tmp/campaign.json", "contents": json.dumps(spec)}]}]
    credentials = {**r2, "egoverse-github": {"GITHUB_TOKEN": "github_token"},
                   "egoverse-wandb": {"WANDB_API_KEY": "wandb_api_key"}}
    entry = (ROOT / "scripts/benchmarks/pusher_holdout_entry.sh").read_text()
    for arm in spec["arms"]:
        task_name = "train-" + arm.replace("_", "-")
        task = {"name": task_name, "resource": "training", "image": "nvcr.io/nvidia/pytorch:25.06-py3",
            "credentials": credentials, "inputs": [{"task": "data-gate"}],
            "environment": {"SOURCE_COMMIT": commit, "RUN_KIND": "train", "RUN_ID": name + "-" + arm,
                "ARM": arm, "REGIME": regime, "TRAINING_SEED": str(seed), "WORKFLOW_OUTPUT": "{{output}}"},
            "command": ["bash"], "args": ["/tmp/entry.sh"],
            "files": [{"path": "/tmp/entry.sh", "contents": entry.replace("set +x\n", "set +x\ncp '{{input:0}}/data-ready.json' /tmp/data-ready.json\n", 1)}]}
        tasks.append(task)
        for shard in range(3):
            evaluate = copy.deepcopy(task)
            evaluate.update(name=f"eval-{arm.replace('_', '-')}-s{shard}", resource="evaluation", inputs=[{"task": task_name}])
            evaluate["environment"].update(RUN_KIND="evaluation", EVAL_SHARD=str(shard))
            evaluate["files"] = [{"path": "/tmp/entry.sh", "contents": entry.replace("set +x\n", "set +x\ncp '{{input:0}}/training-complete.json' /tmp/training-complete.json\n", 1)}]
            tasks.append(evaluate)
    assert len(tasks) == 25
    return {"workflow": {"name": name, "tasks": tasks, "resources": {
        "gate": {"cpu": 2, "gpu": 0, "memory": "8Gi", "storage": "12Gi", "platform": "ovx-l40s"},
        "training": {"cpu": 32, "gpu": spec["gpus"], "memory": "192Gi", "storage": "500Gi", "platform": "ovx-l40s"},
        "evaluation": {"cpu": 8, "gpu": 1, "memory": "64Gi", "storage": "120Gi", "platform": "ovx-l40s"}},
        "timeout": {"queue_timeout": "48h", "exec_timeout": "96h"}}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--regime", choices=("cotrain", "clean"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(yaml.safe_dump(workflow(args.commit, args.name, args.regime, args.seed), sort_keys=False))
    print(args.output)
