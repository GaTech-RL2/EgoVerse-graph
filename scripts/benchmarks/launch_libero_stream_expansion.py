"""Expand the unstarted recovery tail while preserving all checkpoint resumes."""

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

import yaml

from egomimic.benchmarks.libero.arc_streams import campaign
from scripts.benchmarks.launch_libero_stream_parallel import (
    parallel_workflow,
    pending_cells,
)


def expanded_workflow(commit, name, suite, *, replay_proof, references, lanes=6):
    """Move only cells 3..20; both reference policies remain in their current jobs."""
    if set(references) != {"stk", "dur"}:
        raise ValueError("Both reference timing modes are required")
    for mode, reference in references.items():
        for key in ("training_run_id", "evaluation_run_id"):
            if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", reference[key]):
                raise ValueError("Invalid reference run ID")
        if not reference["evaluation_run_id"].endswith(f"-reference-{mode}-eval"):
            raise ValueError("Reference timing mode differs")
    result = parallel_workflow(
        commit,
        name,
        suite,
        replay_proof=replay_proof,
        reference_run=references["stk"]["evaluation_run_id"],
        lanes=lanes,
    )
    body = result["workflow"]
    template = body["tasks"][0]
    tasks = {task["name"]: task for task in body["tasks"]}
    body["tasks"] = []
    for mode, reference in references.items():
        waiter = copy.deepcopy(template)
        waiter["name"] = f"reference-{mode}-ready"
        waiter["args"][1] = (
            "python -m pip install --no-cache-dir boto3==1.43.98\n"
            f"python /tmp/wait-reference.py --mode {mode} "
            f"--run-id {reference['evaluation_run_id']} --suite {suite} --commit {commit} "
            f"--training-run {reference['training_run_id']} "
            "--output '{{output}}/reference-ready.json'"
        )
        body["tasks"].append(waiter)
    for index, (_, mode) in enumerate(pending_cells()[3:], start=3):
        train = tasks[f"train-{index:02d}"]
        train["environment"]["ARC_STREAM_REFERENCE_RUN"] = references[mode][
            "evaluation_run_id"
        ]
        train.pop("inputs", None)
        if index >= 3 + lanes:
            train["inputs"] = [{"task": f"train-{index - lanes:02d}"}]
        evaluate = tasks[f"evaluate-{index:02d}"]
        # Input zero must stay the training output: it supplies evaluation-request.json.
        evaluate["inputs"] = [
            {"task": train["name"]},
            {"task": f"reference-{mode}-ready"},
        ]
        if index > 3:
            evaluate["inputs"].append({"task": f"evaluate-{index - 1:02d}"})
        body["tasks"].extend((train, evaluate))
    return result


def mark_recovery_tail_moved(client, old_workflow, old_spec, new_workflow, new_spec):
    """Retire three WAITING lane heads only after their replacement is accepted.

    Existing jobs are immutable. Reserving these empty run prefixes makes their
    overwrite guards stop the old tail before training. All fourteen recoveries
    and their evaluations remain outside the retired dependency descendants.
    """
    old_id, new_id = old_workflow["name"], new_workflow["name"]
    match = re.fullmatch(r"arc-sr-20261003-(spatial|object|goal|10)-1", old_id)
    if not match or new_id != f"arc-sx-20261003-{match[1]}-1":
        raise ValueError("Handoff is restricted to this recovery expansion")
    if old_workflow["status"] != "RUNNING" or new_workflow["status"] not in {
        "PENDING",
        "RUNNING",
        "QUEUED",
        "INITIALIZING",
        "SCHEDULING",
        "PROCESSING",
    }:
        raise ValueError("Recovery must be live and replacement must be accepted")
    if any(
        spec["workflow"]["name"] + "-1" != query["name"]
        for spec, query in ((old_spec, old_workflow), (new_spec, new_workflow))
    ):
        raise ValueError("Submitted workflow identity differs from the specification")
    old_tasks = {t["name"]: t for t in old_spec["workflow"]["tasks"]}
    new_tasks = {t["name"]: t for t in new_spec["workflow"]["tasks"]}
    live = {t["name"]: t for g in old_workflow["groups"] for t in g["tasks"]}
    new_live = {t["name"]: t for g in new_workflow["groups"] for t in g["tasks"]}
    if live.keys() != old_tasks.keys() or new_live.keys() != new_tasks.keys():
        raise ValueError("Live workflow tasks differ from the specification")
    retired = {
        f"{kind}-{i:02d}" for i in range(3, 21) for kind in ("train", "evaluate")
    }
    if any(live.get(name, {}).get("status") != "WAITING" for name in retired):
        raise ValueError("A recovery tail task has started; refusing handoff")
    if new_tasks.keys() != retired | {"reference-stk-ready", "reference-dur-ready"}:
        raise ValueError("Replacement does not cover exactly the unstarted cells")
    heads = {f"train-{i:02d}" for i in range(3, 6)}
    descendants = set(heads)
    while True:
        expanded = descendants | {
            name
            for name, t in old_tasks.items()
            if any(i["task"] in descendants for i in t.get("inputs", []))
        }
        if expanded == descendants:
            break
        descendants = expanded
    if descendants != retired:
        raise ValueError("Retirement would affect preserved work or miss a moved cell")
    source = None
    for i, (variant, mode) in enumerate(pending_cells()[3:], start=3):
        a, b = (
            tasks[f"train-{i:02d}"]["environment"] for tasks in (old_tasks, new_tasks)
        )
        if (a["ARC_STREAM_VARIANT"], a["ARC_STREAM_MODE"], a["SUITE"]) != (
            variant,
            mode,
            "libero_" + match[1],
        ):
            raise ValueError("Old recipe differs from the campaign")
        for key in (
            "SOURCE_COMMIT",
            "SUITE",
            "RUN_MODE",
            "RUN_KIND",
            "ARC_STREAM_VARIANT",
            "ARC_STREAM_MODE",
            "ARC_STREAM_REFERENCE_RUN",
        ):
            if a[key] != b[key]:
                raise ValueError(f"Replacement recipe differs: {key}")
        expected_tail = f"-{variant.replace('_', '-')}-{mode}"
        if (
            a["RUN_ID"] != old_id.removesuffix("-1") + expected_tail
            or b["RUN_ID"] != new_id.removesuffix("-1") + expected_tail
        ):
            raise ValueError("Run prefix differs from its workflow")
        source = source or a["SOURCE_COMMIT"]
        if a["SOURCE_COMMIT"] != source:
            raise ValueError("Inconsistent training source")
    prefixes = {
        name: f"experiments/arc-oat-20260919/{old_tasks[name]['environment']['RUN_ID']}/"
        for name in sorted(heads)
    }
    # Validate all three before writing any marker; never replace existing data.
    for prefix in prefixes.values():
        if client.list_objects_v2(Bucket="rldb", Prefix=prefix, MaxKeys=1).get(
            "KeyCount"
        ):
            raise FileExistsError("Retired lane head already has artifacts")
    receipt = {
        "operation": "recovery_tail_moved_to_expanded_workflow",
        "old_workflow": old_id,
        "new_workflow": new_id,
        "source_commit": source,
        "preserved_tasks": sorted(old_tasks.keys() - retired),
        "retired_tasks": sorted(retired),
        "retired_lane_heads": sorted(heads),
        "expected_retirement": "ArtifactUploader rejects the occupied prefixes before training; only their descendants become FAILED_UPSTREAM.",
    }
    body = (json.dumps(receipt, indent=2) + "\n").encode()
    sha = hashlib.sha256(body).hexdigest()
    markers = []
    for prefix in prefixes.values():
        key = prefix + "orchestration-handoff.json"
        client.put_object(
            Bucket="rldb",
            Key=key,
            Body=body,
            ContentType="application/json",
            Metadata={"sha256": sha},
            IfNoneMatch="*",
        )
        markers.append({"uri": "s3://rldb/" + key, "sha256": sha})
    return {**receipt, "markers": markers}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--previous-manifest", type=Path, required=True)
    parser.add_argument("--replay-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lanes", type=int, default=6)
    args = parser.parse_args()
    previous = json.loads(args.previous_manifest.read_text())
    if previous["source_commit"] != args.commit:
        raise ValueError("Expansion must retain the pinned training source")
    args.output.mkdir(parents=True, exist_ok=True)
    for suite in campaign()["suites"]:
        suffix = suite.removeprefix("libero_")
        references = {
            r["mode"]: r
            for r in previous["runs"]
            if r["suite"] == suite and r["variant"] == "reference"
        }
        proof = json.loads((args.replay_directory / f"{suffix}.json").read_text())
        name = f"{args.prefix}-{suffix}"
        spec = expanded_workflow(
            args.commit,
            name,
            suite,
            replay_proof=proof,
            references=references,
            lanes=args.lanes,
        )
        path = args.output / f"{name}.yaml"
        path.write_text(yaml.safe_dump(spec, sort_keys=False))
        print(path)


if __name__ == "__main__":
    main()
