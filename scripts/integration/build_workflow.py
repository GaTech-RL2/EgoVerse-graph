"""Render a source-pinned OSMO L40 workflow; rendering does not submit compute."""

import argparse
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def workflow(commit, name):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("OSMO requires an immutable full source commit")
    tasks = []
    for suite, resource in (
        ("stage", "cpu"),
        ("hpt", "single"),
        ("pi", "single"),
        ("ddp", "distributed"),
    ):
        task = {
            "name": "stage-inputs" if suite == "stage" else suite + "-gates",
            "image": "nvcr.io/nvidia/pytorch:25.06-py3",
            "resource": resource,
            "credentials": {"egoverse-github": {"GITHUB_TOKEN": "github_token"}},
            "environment": {
                "SOURCE_COMMIT": commit,
                "GATE_SUITE": suite,
                "GATE_OUTPUT": "{{output}}",
            },
            "command": ["bash"],
            "args": ["/tmp/entry.sh"],
            "files": [
                {
                    "path": "/tmp/entry.sh",
                    "contents": (
                        ROOT / "scripts/integration/osmo_entry.sh"
                    ).read_text(),
                }
            ],
        }
        if suite == "stage":
            task["credentials"]["grabber-arc-r2-20260916"] = {
                "R2_ACCESS_KEY_ID": "r2_access_key_id",
                "R2_SECRET_ACCESS_KEY": "r2_secret_access_key",
                "R2_ENDPOINT_URL": "r2_endpoint_url",
            }
        else:
            task["inputs"] = [{"task": "stage-inputs"}]
            if suite == "ddp":
                # Release the HPT GPU before scheduling two distributed workers.
                task["inputs"].append(
                    {"task": "hpt-gates", "regex": "^suite-progress\\.json$"}
                )
            task["environment"]["GATE_INPUT"] = "{{input:0}}"
            task["environment"]["ARTIFACT_PREFIX"] = (
                f"experiments/graph-integration-20260924/{name}/{commit}/{suite}"
            )
            task["credentials"]["grabber-arc-r2-20260916"] = {
                "R2_ACCESS_KEY_ID": "r2_access_key_id",
                "R2_SECRET_ACCESS_KEY": "r2_secret_access_key",
                "R2_ENDPOINT_URL": "r2_endpoint_url",
            }
        tasks.append(task)
    return {
        "workflow": {
            "name": name,
            "tasks": tasks,
            "resources": {
                "cpu": {
                    "gpu": 0,
                    "cpu": 4,
                    "memory": "24Gi",
                    "storage": "70Gi",
                    "platform": "ovx-l40",
                },
                "single": {
                    "gpu": 1,
                    "cpu": 8,
                    "memory": "96Gi",
                    "storage": "200Gi",
                    "platform": "ovx-l40",
                },
                "distributed": {
                    "gpu": 2,
                    "cpu": 12,
                    "memory": "64Gi",
                    "storage": "90Gi",
                    "platform": "ovx-l40",
                },
            },
            "timeout": {"queue_timeout": "1h", "exec_timeout": "3h"},
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x") as handle:
        yaml.safe_dump(workflow(args.source_commit, args.name), handle, sort_keys=False)
