"""Move pending stream cells into parallel lanes, retaining running STK references."""

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

import yaml

from egomimic.benchmarks.libero.arc_streams import campaign, validate_replay_proof
from scripts.benchmarks.launch_libero_osmo import baseline_workflow


def pending_cells():
    spec = campaign()
    first = [("reference", "dur"), ("gripper", "stk"), ("component_time", "dur")]
    return first + [
        (variant, mode)
        for variant in spec["variants"]
        for mode in spec["modes"]
        if (variant, mode) not in [*first, ("reference", "stk")]
    ]


def parallel_workflow(commit, name, suite, *, replay_proof, reference_run, lanes=3):
    if (
        not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,27}", name)
        or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", reference_run)
        or not reference_run.endswith("-reference-stk-eval")
        or type(lanes) is not int
        or not 1 <= lanes <= 6
        or suite not in campaign()["suites"]
    ):
        raise ValueError("Invalid parallel stream workflow")
    validate_replay_proof(replay_proof, suite, commit)
    workflow = baseline_workflow(
        commit, name, suite, backbone="unet", mode="full", gpus=4, gpu_type="L40S"
    )
    body = workflow["workflow"]
    training, evaluation = body["tasks"]
    body["resources"]["evaluation"]["memory"] = "120Gi"
    body["resources"]["reference_wait"] = {
        "cpu": 1,
        "gpu": 0,
        "memory": "1Gi",
        "storage": "4Gi",
        "platform": "ovx-l40s",
    }
    waiter = Path(__file__).with_name("wait_libero_stream_reference.py").read_text()
    body["tasks"] = [
        {
            "name": "reference-stk-ready",
            "image": "docker.io/library/python:3.11-slim",
            "resource": "reference_wait",
            "credentials": {
                "grabber-arc-r2-20260916": copy.deepcopy(
                    training["credentials"]["grabber-arc-r2-20260916"]
                )
            },
            "command": ["bash"],
            "args": [
                "-ceu",
                "python -m pip install --no-cache-dir boto3==1.43.98\n"
                f"python /tmp/wait-reference.py --run-id {reference_run} --suite {suite} --commit {commit} --output '{{{{output}}}}/reference-ready.json'",
            ],
            "files": [{"path": "/tmp/wait-reference.py", "contents": waiter}],
        }
    ]
    last_training = [None] * lanes
    previous_evaluation = None
    for index, (variant, mode) in enumerate(pending_cells()):
        lane = index % lanes
        run_id = f"{name}-{variant.replace('_', '-')}-{mode}"
        if len(run_id + "-eval") > 63:
            raise ValueError("Parallel stream run ID is too long")
        task = copy.deepcopy(training)
        task["name"] = f"train-{index:02d}"
        reference = reference_run if mode == "stk" else f"{name}-reference-dur-eval"
        task["environment"].update(
            RUN_KIND="arc_stream_train",
            RUN_ID=run_id,
            ARC_STREAM_VARIANT=variant,
            ARC_STREAM_MODE=mode,
            ARC_STREAM_REFERENCE_RUN=reference,
            ARC_STREAM_OUTPUT="{{output}}",
        )
        task["files"].append(
            {
                "path": "/tmp/replay-completion.json",
                "contents": json.dumps(replay_proof, sort_keys=True),
            }
        )
        if last_training[lane] is not None:
            task["inputs"] = [{"task": last_training[lane]}]
        body["tasks"].append(task)
        last_training[lane] = task["name"]
        evaluate = copy.deepcopy(evaluation)
        evaluate["name"] = f"evaluate-{index:02d}"
        evaluate["environment"]["RUN_ID"] = run_id + "-eval"
        evaluate["inputs"] = [{"task": task["name"]}]
        if previous_evaluation is not None:
            evaluate["inputs"].append({"task": previous_evaluation})
        if mode == "stk":
            evaluate["inputs"].append({"task": "reference-stk-ready"})
        body["tasks"].append(evaluate)
        previous_evaluation = evaluate["name"]
    return workflow


def mark_pending_tail_moved(client, old_workflow, new_workflow_id, source_commit):
    """Reserve only the unstarted serial tail; its overwrite guard stops duplicates.

    OSMO cannot edit a submitted DAG. The first moved task will refuse this
    occupied artifact prefix and its dependents become FAILED_UPSTREAM. The
    running reference STK and its evaluation are independent of that tail.
    Call only after the replacement workflow was successfully submitted.
    """
    workflow_id = old_workflow["name"]
    if not re.fullmatch(r"arc-str-20261002r2-(spatial|object|goal|10)-1", workflow_id):
        raise ValueError("Handoff is restricted to this campaign's serial workflows")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", new_workflow_id):
        raise ValueError("Invalid replacement workflow ID")
    tasks = {t["name"]: t for g in old_workflow["groups"] for t in g["tasks"]}
    if len(tasks) != 45 or any(
        tasks.get(f"{kind}-{i:02d}", {}).get("status") != "WAITING"
        for i in range(1, 22)
        for kind in ("train", "evaluate")
    ):
        raise ValueError("A serial tail task has already started; refusing handoff")
    run_id = workflow_id.removesuffix("-1") + "-reference-dur"
    prefix = f"experiments/arc-oat-20260919/{run_id}/"
    if client.list_objects_v2(Bucket="rldb", Prefix=prefix, MaxKeys=1).get("KeyCount"):
        raise FileExistsError("Serial duration reference already has artifacts")
    receipt = {
        "operation": "pending_tail_moved_to_parallel_workflow",
        "old_workflow": workflow_id,
        "new_workflow": new_workflow_id,
        "source_commit": source_commit,
        "first_retired_task": "train-01",
        "preserved_tasks": ["replay", "train-00", "evaluate-00"],
        "retired_training_tasks": [f"train-{i:02d}" for i in range(1, 22)],
        "retired_evaluation_tasks": [f"evaluate-{i:02d}" for i in range(1, 22)],
        "expected_retirement": "ArtifactUploader rejects the occupied prefix before training; dependent tasks become FAILED_UPSTREAM.",
    }
    body = (json.dumps(receipt, indent=2) + "\n").encode()
    sha = hashlib.sha256(body).hexdigest()
    key = prefix + "orchestration-handoff.json"
    client.put_object(
        Bucket="rldb",
        Key=key,
        Body=body,
        ContentType="application/json",
        Metadata={"sha256": sha},
        IfNoneMatch="*",
    )
    return {**receipt, "uri": "s3://rldb/" + key, "sha256": sha}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--reference-prefix", required=True)
    parser.add_argument("--replay-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lanes", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for suite in campaign()["suites"]:
        suffix = suite.removeprefix("libero_")
        name = f"{args.prefix}-{suffix}"
        proof = json.loads((args.replay_directory / f"{suffix}.json").read_text())
        spec = parallel_workflow(
            args.commit,
            name,
            suite,
            replay_proof=proof,
            reference_run=f"{args.reference_prefix}-{suffix}-reference-stk-eval",
            lanes=args.lanes,
        )
        destination = args.output / f"{name}.yaml"
        destination.write_text(yaml.safe_dump(spec, sort_keys=False))
        print(destination)


if __name__ == "__main__":
    main()
