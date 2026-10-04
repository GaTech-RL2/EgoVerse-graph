"""Replace terminal failed tasks without restarting healthy training lanes."""

import copy
import json
import re
from pathlib import Path

from egomimic.benchmarks.libero.arc_streams import campaign


def retry_downloads(task):
    entry = next(f for f in task["files"] if f["path"] == "/tmp/entry.sh")
    install = "uv pip install -e '.[oat,libero]'"
    if entry["contents"].count(install) != 1:
        raise ValueError("Unexpected dependency installation entrypoint")
    entry["contents"] = entry["contents"].replace(
        install,
        "\n".join(
            [
                "for download_attempt in 1 2 3; do",
                f"    if {install}; then break; fi",
                '    if [[ "$download_attempt" == 3 ]]; then exit 1; fi',
                "    sleep 10",
                "done",
            ]
        ),
    )


def failed_tasks(query, spec):
    body = spec["workflow"]
    if query["name"] != body["name"] + "-1" or not body["name"].startswith(
        "arc-sx-20261003-"
    ):
        raise ValueError("Repair is restricted to the submitted expansion workflow")
    live = {t["name"]: t for g in query["groups"] for t in g["tasks"]}
    if live.keys() != {t["name"] for t in body["tasks"]}:
        raise ValueError("Submitted tasks differ from the saved specification")
    failed = {
        n for n, t in live.items() if t["status"] in {"FAILED", "FAILED_UPSTREAM"}
    }
    if not failed or any(
        not re.fullmatch(r"(?:train|evaluate)-\d{2}", n) for n in failed
    ):
        raise ValueError("Repair requires failed policy tasks, not failed references")
    return failed


def require_empty_failed_prefixes(client, query, spec):
    """A task with existing artifacts requires explicit checkpoint recovery."""
    selected = failed_tasks(query, spec)
    for task in spec["workflow"]["tasks"]:
        if task["name"] in selected:
            run = task["environment"]["RUN_ID"]
            prefix = f"experiments/arc-oat-20260919/{run}/"
            if client.list_objects_v2(Bucket="rldb", Prefix=prefix, MaxKeys=1).get(
                "KeyCount"
            ):
                raise FileExistsError(
                    f"Failed run has artifacts; explicit recovery required: {run}"
                )
    return sorted(selected)


def repair_workflow(query, spec, name, evaluation_lanes=4):
    selected = failed_tasks(query, spec)
    if (
        not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", name)
        or type(evaluation_lanes) is not int
        or not 1 <= evaluation_lanes <= 4
    ):
        raise ValueError("Invalid repair workflow")
    result = copy.deepcopy(spec)
    body = result["workflow"]
    body["name"] = name
    tasks = {t["name"]: t for t in body["tasks"]}
    body["tasks"] = [tasks[f"reference-{mode}-ready"] for mode in ("stk", "dur")]
    repaired_training = {n for n in selected if n.startswith("train-")}
    for task_name in sorted(repaired_training):
        task = tasks[task_name]
        task["inputs"] = [
            i for i in task.get("inputs", []) if i["task"] in repaired_training
        ]
        retry_downloads(task)
        body["tasks"].append(task)
    previous = [None] * evaluation_lanes
    for index, task_name in enumerate(
        sorted(n for n in selected if n.startswith("evaluate-"))
    ):
        evaluation = tasks[task_name]
        training_name = evaluation["inputs"][0]["task"]
        training = tasks[training_name]
        env = training["environment"]
        mode = env["ARC_STREAM_MODE"]
        dependency = training_name
        if training_name not in repaired_training:
            dependency = training_name.replace("train-", "checkpoint-")
            waiter = copy.deepcopy(tasks[f"reference-{mode}-ready"])
            waiter["name"] = dependency
            identity = {
                "source_run": env["RUN_ID"],
                "source_commit": env["SOURCE_COMMIT"],
                "suite": env["SUITE"],
                "method": f"arc_{mode}",
                "arc_stream_variant": env["ARC_STREAM_VARIANT"],
                "total_optimizer_steps": campaign()["optimizer_steps"][env["SUITE"]],
                "stream_reference_run": env["ARC_STREAM_REFERENCE_RUN"],
            }
            waiter["args"][1] = (
                "python -m pip install --no-cache-dir boto3==1.43.98\n"
                "python /tmp/wait-checkpoint.py --identity /tmp/policy-identity.json --output '{{output}}'"
            )
            waiter["files"] = [
                {
                    "path": "/tmp/wait-checkpoint.py",
                    "contents": Path(__file__)
                    .with_name("wait_libero_stream_checkpoint.py")
                    .read_text(),
                },
                {
                    "path": "/tmp/policy-identity.json",
                    "contents": json.dumps(identity, indent=2) + "\n",
                },
            ]
            body["tasks"].append(waiter)
        evaluation["inputs"] = [
            {"task": dependency},
            {"task": f"reference-{mode}-ready"},
        ]
        lane = index % evaluation_lanes
        if previous[lane]:
            evaluation["inputs"].append({"task": previous[lane]})
        previous[lane] = task_name
        retry_downloads(evaluation)
        body["tasks"].append(evaluation)
    names = {t["name"] for t in body["tasks"]}
    if any(i["task"] not in names for t in body["tasks"] for i in t.get("inputs", [])):
        raise ValueError("Repair has a dependency outside its workflow")
    return result
