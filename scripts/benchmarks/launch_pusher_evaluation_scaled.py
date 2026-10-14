"""Render extra L40S workers that resume disjoint frozen evaluation levels."""
import argparse
import copy
import json
from pathlib import Path

import yaml

from launch_pusher_evaluation import BENCHMARKS, workflow as recovery_workflow


def workflow(receipts, assignments, *, name, regime, seed, harness_commit):
    result = recovery_workflow(receipts, name=name, regime=regime, seed=seed,
        shards=[0], revision="r3", harness_commit=harness_commit)
    base = result["workflow"]
    templates = {}
    for task in base["tasks"]:
        files = {f["path"]: f["contents"] for f in task["files"]}
        training = json.loads(files["/tmp/training-complete.json"])
        templates[training["run_id"]] = task
    tasks = []
    identities = set()
    for assignment in assignments:
        assert assignment["run_id"] in templates
        identity = (assignment["run_id"], assignment["partition"])
        assert identity not in identities
        identities.add(identity)
        task = copy.deepcopy(templates[assignment["run_id"]])
        arm = assignment["arm"].replace("_", "-")
        task["name"] = f"eval-{arm}-p{assignment['partition']}"
        for file in task["files"]:
            if file["path"] == "/tmp/rebind_evaluation.py":
                file["contents"] = (BENCHMARKS / "repartition_pusher_evaluation.py").read_text()
        task["files"] += [
            {"path": "/tmp/evaluation-assignment.json", "contents": json.dumps(assignment)},
            {"path": "/tmp/planar_eval_resume.py", "contents": (BENCHMARKS / "planar_eval_resume.py").read_text()},
        ]
        tasks.append(task)
    assert tasks
    base["tasks"] = tasks
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-receipts", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--regime", choices=("cotrain", "clean"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--harness-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = workflow(json.loads(args.training_receipts.read_text()), json.loads(args.assignments.read_text()),
        name=args.name, regime=args.regime, seed=args.seed, harness_commit=args.harness_commit)
    args.output.write_text(yaml.safe_dump(result, sort_keys=False))
    print(args.output)
