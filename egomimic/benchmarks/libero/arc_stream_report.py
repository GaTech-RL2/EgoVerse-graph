"""Pair different ARC representations without relaxing episode/data matching."""

import hashlib
import re
from pathlib import Path

from egomimic.benchmarks.libero.report import (
    read_run,
    summarize,
    validate_full_protocol,
)


def compare(candidate_directory, reference_directory):
    candidate, rows = read_run(candidate_directory)
    reference, old = read_run(reference_directory)
    for protocol in (candidate, reference):
        validate_full_protocol(protocol)
        if protocol.get("method") != "arc" or protocol.get("arc_backbone") != "unet":
            raise ValueError("Stream comparison requires its fixed ARC/U-Net policies")
    if reference.get("arc_stream_variant") != "reference":
        raise ValueError("Stream comparison requires the prespecified reference arm")
    for key in (
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
    ):
        if key not in candidate or candidate[key] != reference.get(key):
            raise ValueError(f"Stream comparison protocol differs: {key}")
    for key in ("mode", "waypoints", "dt", "max_translation", "max_rotation_degrees"):
        if candidate["representation"].get(key) != reference["representation"].get(key):
            raise ValueError(f"Stream comparison geometry differs: {key}")
    if rows.keys() != old.keys() or any(
        rows[k]["initial_state_sha256"] != old[k]["initial_state_sha256"] for k in rows
    ):
        raise ValueError(
            "Stream comparison episodes or initial simulator states differ"
        )
    score, control = summarize(rows), summarize(old)
    return {
        "candidate": score,
        "reference": control,
        "paired_episodes": len(rows),
        "initial_state_mismatches": 0,
        "complete_protocol": True,
        "candidate_variant": candidate["arc_stream_variant"],
        "candidate_representation": candidate["representation"],
        "reference_representation": reference["representation"],
        "candidate_parameters": candidate.get("parameters"),
        "reference_parameters": reference.get("parameters"),
        "success_delta_percentage_points": 100
        * (score["mean_success_rate"] - control["mean_success_rate"]),
        "candidate_only_successes": sum(
            bool(rows[k]["success"]) and not bool(old[k]["success"]) for k in rows
        ),
        "reference_only_successes": sum(
            bool(old[k]["success"]) and not bool(rows[k]["success"]) for k in rows
        ),
    }


def audit_reference(client, request, candidate_directory, evidence):
    reference = request["stream_reference_run"]
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", reference):
        raise ValueError("Invalid stream reference run")
    directory = Path(evidence) / "stream-reference"
    directory.mkdir(exist_ok=False)
    receipts = {}
    for name in ("protocol.json", "episodes.jsonl"):
        key = f"experiments/arc-oat-20260919/{reference}/{request['method']}/{request['suite']}/{name}"
        obj = client.get_object(Bucket="rldb", Key=key)
        body = obj["Body"].read()
        sha = hashlib.sha256(body).hexdigest()
        if obj.get("Metadata", {}).get("sha256") != sha:
            raise ValueError("Stream reference artifact checksum differs")
        (directory / name).write_bytes(body)
        receipts[name] = {"uri": "s3://rldb/" + key, "sha256": sha}
    return {**compare(candidate_directory, directory), "reference_artifacts": receipts}
