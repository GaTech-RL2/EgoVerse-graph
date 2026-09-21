"""Validate complete, paired benchmark records before computing comparisons."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from egomimic.benchmarks.libero.catalog import (
    LIBERO_COMMIT,
    OAT_COMMIT,
    TASKS,
    get_tasks,
)
from egomimic.benchmarks.libero.rollout import RolloutSpec, rollout_plan


def read_run(directory):
    directory = Path(directory)
    protocol = json.loads((directory / "protocol.json").read_text())
    records = [
        json.loads(line)
        for line in (directory / "episodes.jsonl").read_text().splitlines()
        if line
    ]
    plan = [RolloutSpec(**entry) for entry in protocol["plan"]]

    def key(entry):
        return tuple(entry[field] for field in ("task", "repetition", "trial", "seed"))

    expected = {key(asdict(spec)) for spec in plan}
    actual = [key(record) for record in records]
    if (
        len(expected) != len(plan)
        or len(set(actual)) != len(actual)
        or set(actual) != expected
    ):
        raise ValueError("Missing, unexpected, or duplicate benchmark episodes")
    for record in records:
        if (
            not isinstance(record["success"], bool)
            or not 1 <= record["steps"] <= protocol["max_episode_steps"]
        ):
            raise ValueError("Invalid episode outcome")
        latency = np.asarray(record["inference_seconds"])
        if not len(latency) or not np.isfinite(latency).all() or (latency < 0).any():
            raise ValueError("Invalid inference latency")
        if not record.get("initial_state_sha256"):
            raise ValueError("Episode lacks initial-state identity")
    return protocol, {key(record): record for record in records}


def summarize(records):
    values = list(records.values())
    repetitions = sorted({entry["repetition"] for entry in values})
    scores = [
        np.mean(
            [entry["success"] for entry in values if entry["repetition"] == repetition]
        )
        for repetition in repetitions
    ]
    tasks = sorted({entry["task"] for entry in values})
    latencies = [latency for entry in values for latency in entry["inference_seconds"]]
    return {
        "episodes": len(values),
        "mean_success_rate": float(np.mean(scores)),
        "success_rate_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else None,
        "success_rate_stderr": float(np.std(scores, ddof=1) / np.sqrt(len(scores)))
        if len(scores) > 1
        else None,
        "per_repetition_success": [float(score) for score in scores],
        "per_task_success": {
            task: float(
                np.mean([entry["success"] for entry in values if entry["task"] == task])
            )
            for task in tasks
        },
        "inference_seconds_mean": float(np.mean(latencies)) if latencies else None,
        "inference_seconds_p95": float(np.percentile(latencies, 95))
        if latencies
        else None,
    }


def validate_full_protocol(protocol):
    """Require the released evaluation protocol, including every task and seed."""
    expected = [asdict(spec) for spec in rollout_plan(protocol["suite"])]
    if protocol["plan"] != expected or protocol["max_episode_steps"] != 550:
        raise ValueError(
            "Full benchmark requires 50 trials/task, five repeats, seed 1000 and 550 steps"
        )
    defaults = {
        "horizon": 32,
        "n_obs_steps": 2,
        "n_action_steps": 16,
        "oat_commit": OAT_COMMIT,
        "libero_commit": LIBERO_COMMIT,
        "use_ema": True,
    }
    if any(protocol[field] != value for field, value in defaults.items()):
        raise ValueError(
            "Full benchmark requires the released control protocol and EMA checkpoints"
        )


def compare_runs(arc_directory, oat_directory, *, require_full=True, arc_mode=None):
    arc_protocol, arc = read_run(arc_directory)
    oat_protocol, oat = read_run(oat_directory)
    common = (
        "suite",
        "max_episode_steps",
        "n_obs_steps",
        "n_action_steps",
        "horizon",
        "data_context",
        "observations_sha256",
        "use_ema",
        "oat_commit",
        "libero_commit",
        "plan",
    )
    for field in common:
        if field not in arc_protocol or arc_protocol[field] != oat_protocol.get(field):
            raise ValueError(f"ARC and OAT protocol mismatch: {field}")
    if arc_protocol.get("method") != "arc" or oat_protocol.get("method") != "oat":
        raise ValueError("Comparison requires native ARC and OAT policy records")
    if (
        arc_mode is not None
        and arc_protocol.get("representation", {}).get("mode") != arc_mode
    ):
        raise ValueError("ARC representation mode differs from requested comparison")
    suite = arc_protocol["suite"]
    if require_full:
        validate_full_protocol(arc_protocol)
    if set(arc) != set(oat):
        raise ValueError("Paired rollout identities differ")
    for key in arc:
        if arc[key]["initial_state_sha256"] != oat[key]["initial_state_sha256"]:
            raise ValueError(f"Initial states differ for {key}")
    arc_metrics, oat_metrics = summarize(arc), summarize(oat)
    differences = (
        np.asarray(arc_metrics["per_repetition_success"])
        - oat_metrics["per_repetition_success"]
    )
    return {
        "suite": suite,
        "complete_protocol": require_full,
        "arc": arc_metrics,
        "oat": oat_metrics,
        "paired_success_difference": float(differences.mean()),
        "paired_difference_stderr": float(
            differences.std(ddof=1) / np.sqrt(len(differences))
        )
        if len(differences) > 1
        else None,
        "checkpoints": {
            "arc": arc_protocol.get("checkpoint_sha256"),
            "oat": oat_protocol.get("checkpoint_sha256"),
        },
        "representations": {
            "arc": arc_protocol.get("representation"),
            "oat": oat_protocol.get("representation"),
        },
        "parameters": {
            "arc": arc_protocol.get("parameters"),
            "oat": oat_protocol.get("parameters"),
        },
    }


def compare_suite(arc_root, oat_root):
    results = {
        suite: compare_runs(Path(arc_root) / suite, Path(oat_root) / suite)
        for suite in TASKS
    }
    for suite, result in results.items():
        if get_tasks(result["suite"]) != get_tasks(suite):
            raise ValueError(f"Result folder {suite} contains a different task suite")
    return {
        "complete_benchmark_suite": True,
        "unique_tasks": sum(len(get_tasks(suite)) for suite in TASKS),
        "suites": results,
    }
