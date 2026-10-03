"""Generate parallel recovery without rerunning completed reference training."""

import argparse
import copy
import json
import re
from pathlib import Path

import yaml

from egomimic.benchmarks.libero.arc_streams import campaign
from egomimic.benchmarks.libero.evaluate import validate_request
from scripts.benchmarks.launch_libero_stream_parallel import (
    parallel_workflow,
    pending_cells,
)


def resume_request(run, artifacts, commit):
    directory = artifacts / run["training_run_id"]
    runtime = json.loads((directory / "runtime.json").read_text())
    if runtime["source_commit"] != commit:
        raise ValueError("Recovery cannot change the training source")
    state = json.loads((directory / "status.json").read_text())
    if state["state"] == "TRAINING_COMPLETE":
        raise ValueError("Completed policies must be evaluated, not retrained")
    method = f"arc_{run['mode']}"
    receipt = json.loads((directory / "checkpoint-receipts.json").read_text())[
        f"training/{method}/checkpoints/last.ckpt"
    ]
    request = {
        "source_run": run["training_run_id"],
        "source_commit": commit,
        "suite": run["suite"],
        "method": method,
        "arc_stream_variant": run["variant"],
        "epochs": 5001,
        "total_optimizer_steps": campaign()["optimizer_steps"][run["suite"]],
        "checkpoint": receipt,
    }
    validate_request(request)
    return request


def attach_resume(task, request):
    environment = task["environment"]
    if (request["suite"], request["method"], request["arc_stream_variant"]) != (
        environment["SUITE"],
        f"arc_{environment['ARC_STREAM_MODE']}",
        environment["ARC_STREAM_VARIANT"],
    ) or request["source_commit"] != environment["SOURCE_COMMIT"]:
        raise ValueError("Resume request differs from task recipe")
    entry = next(f for f in task["files"] if f["path"] == "/tmp/entry.sh")
    old = 'python -m egomimic.benchmarks.libero.arc_streams "${STREAM_ARGS[@]}"'
    if entry["contents"].count(old) != 1:
        raise ValueError("Unexpected stream entrypoint")
    entry["contents"] = entry["contents"].replace(
        old,
        'python /tmp/resume-libero-stream.py "${STREAM_ARGS[@]}" --resume-request /tmp/resume-request.json',
    )
    task["files"] += [
        {
            "path": "/tmp/resume-libero-stream.py",
            "contents": Path(__file__).with_name("resume_libero_stream.py").read_text(),
        },
        {
            "path": "/tmp/resume-request.json",
            "contents": json.dumps(request, indent=2) + "\n",
        },
    ]


def recovery_workflow(
    commit, name, suite, proof, previous, artifacts, completed_evaluation=None
):
    source_runs = {
        (r["variant"], r["mode"]): r for r in previous["runs"] if r["suite"] == suite
    }
    reference_run = completed_evaluation or f"{name}-reference-stk-eval"
    result = parallel_workflow(
        commit, name, suite, replay_proof=proof, reference_run=reference_run
    )
    body = result["workflow"]
    tasks = {t["name"]: t for t in body["tasks"]}
    for i, cell in enumerate(pending_cells()[:3]):
        attach_resume(
            tasks[f"train-{i:02d}"],
            resume_request(source_runs[cell], artifacts, commit),
        )
    reference = source_runs["reference", "stk"]
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", reference["training_run_id"]):
        raise ValueError("Invalid reference training run")
    if completed_evaluation:
        state = json.loads(
            (artifacts / reference["training_run_id"] / "status.json").read_text()
        )
        if state["state"] != "TRAINING_COMPLETE":
            raise ValueError(
                "External reference evaluation requires completed training"
            )
        tasks["reference-stk-ready"]["args"][1] += (
            f" --training-run {reference['training_run_id']}"
        )
    else:
        # The retained reference itself was preempted. Its evaluation becomes
        # the readiness dependency, alongside the three other training lanes.
        training = copy.deepcopy(tasks["train-00"])
        training["name"] = "reference-train"
        training["environment"].update(
            RUN_ID=f"{name}-reference-stk",
            ARC_STREAM_MODE="stk",
            ARC_STREAM_REFERENCE_RUN=reference_run,
        )
        training["files"] = [
            f
            for f in training["files"]
            if f["path"]
            not in ("/tmp/resume-libero-stream.py", "/tmp/resume-request.json")
        ]
        entry = next(f for f in training["files"] if f["path"] == "/tmp/entry.sh")
        entry["contents"] = entry["contents"].replace(
            'python /tmp/resume-libero-stream.py "${STREAM_ARGS[@]}" --resume-request /tmp/resume-request.json',
            'python -m egomimic.benchmarks.libero.arc_streams "${STREAM_ARGS[@]}"',
        )
        attach_resume(training, resume_request(reference, artifacts, commit))
        evaluation = copy.deepcopy(tasks["evaluate-00"])
        evaluation["name"] = "reference-stk-ready"
        evaluation["inputs"] = [{"task": "reference-train"}]
        evaluation["environment"]["RUN_ID"] = reference_run
        body["tasks"][0] = evaluation
        body["tasks"].insert(0, training)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--previous-manifest", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--replay-directory", type=Path, required=True)
    parser.add_argument("--completed-evaluations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    previous = json.loads(args.previous_manifest.read_text())
    evaluations = json.loads(args.completed_evaluations.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for suite in campaign()["suites"]:
        suffix = suite.removeprefix("libero_")
        proof = json.loads((args.replay_directory / f"{suffix}.json").read_text())
        name = f"{args.prefix}-{suffix}"
        spec = recovery_workflow(
            args.commit,
            name,
            suite,
            proof,
            previous,
            args.artifacts,
            completed_evaluation=evaluations.get(suite),
        )
        path = args.output / f"{name}.yaml"
        path.write_text(yaml.safe_dump(spec, sort_keys=False))
        print(path)


if __name__ == "__main__":
    main()
