"""Render a fixed LIBERO stream sweep with bounded training/evaluation concurrency."""

import argparse
import copy
import re
from pathlib import Path

import yaml

from egomimic.benchmarks.libero.arc_streams import campaign
from scripts.benchmarks.launch_libero_osmo import baseline_workflow


def stream_workflow(commit, name, suite, *, gpus=4, mode="full", variants=None):
    spec = campaign()
    variants = spec["variants"] if variants is None else list(variants)
    if mode == "full" and (not variants or variants[0] != "reference"):
        raise ValueError(
            "Full sweeps must train both reference modes before comparisons"
        )
    if (
        not variants
        or len(set(variants)) != len(variants)
        or set(variants) - set(spec["variants"])
        or suite not in spec["suites"]
        or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,27}", name)
    ):
        raise ValueError("Invalid stream campaign name, suite or variants")
    workflow = baseline_workflow(
        commit,
        name,
        suite,
        backbone=spec["backbone"],
        mode=mode,
        gpus=gpus,
        gpu_type="L40S",
    )
    body = workflow["workflow"]
    training = body["tasks"][0]
    evaluation = body["tasks"][1] if mode == "full" else None
    body["tasks"] = []
    if mode == "full":
        replay = copy.deepcopy(training)
        replay.update(name="replay", resource="replay")
        replay["environment"].update(
            RUN_KIND="arc_stream_replay",
            RUN_ID=name + "-replay",
            TRAINING_GPUS="1",
            ARC_STREAM_OUTPUT="{{output}}",
        )
        body["tasks"].append(replay)
        body["resources"]["replay"] = {
            "cpu": spec["replay"]["workers"] + 3,
            "gpu": 1,
            "memory": "120Gi",
            "storage": "240Gi",
            "platform": "ovx-l40s",
        }
        body["resources"]["evaluation"]["memory"] = "120Gi"
    previous_training, previous_evaluation = None, None
    for index, (variant, arc_mode) in enumerate(
        (v, m) for v in variants for m in spec["modes"]
    ):
        task = copy.deepcopy(training)
        run_id = f"{name}-{variant.replace('_', '-')}-{arc_mode}"
        if len(run_id + "-eval") > 63:
            raise ValueError("Stream run name exceeds the artifact/DNS limit")
        task["name"] = f"train-{index:02d}"
        task["environment"].update(
            RUN_KIND="arc_stream_train",
            RUN_ID=run_id,
            ARC_STREAM_VARIANT=variant,
            ARC_STREAM_MODE=arc_mode,
            ARC_STREAM_REFERENCE_RUN=f"{name}-reference-{arc_mode}-eval",
            ARC_STREAM_OUTPUT="{{output}}",
        )
        inputs = [{"task": "replay"}] if mode == "full" else []
        if previous_training:
            inputs.append({"task": previous_training})
        if inputs:
            task["inputs"] = inputs
        if mode == "full":
            task["files"][0]["contents"] = task["files"][0]["contents"].replace(
                "set +x\n",
                "set +x\ncp '{{input:0}}/replay-completion.json' /tmp/replay-completion.json\n",
                1,
            )
        body["tasks"].append(task)
        previous_training = task["name"]
        if evaluation is not None:
            evaluate = copy.deepcopy(evaluation)
            evaluate["name"] = f"evaluate-{index:02d}"
            evaluate["environment"].update(RUN_ID=run_id + "-eval")
            evaluate["inputs"] = [{"task": task["name"]}]
            if previous_evaluation:
                evaluate["inputs"].append({"task": previous_evaluation})
            body["tasks"].append(evaluate)
            previous_evaluation = evaluate["name"]
    # One training task and one evaluation task per suite at a time. A failed
    # preflight stops downstream work; no workstation watcher submits jobs.
    return workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=campaign()["suites"],
        default=campaign()["suites"],
    )
    parser.add_argument("--variants", nargs="+", choices=campaign()["variants"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for suite in args.suites:
        suffix = suite.removeprefix("libero_")
        name = f"{args.prefix}-{suffix}"
        workflow = stream_workflow(
            args.commit,
            name,
            suite,
            gpus=args.gpus,
            mode=args.mode,
            variants=args.variants,
        )
        destination = args.output / f"{name}.yaml"
        destination.write_text(yaml.safe_dump(workflow, sort_keys=False))
        print(destination)


if __name__ == "__main__":
    main()
