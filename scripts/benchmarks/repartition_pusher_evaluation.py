"""Bind a disjoint level assignment to the existing frozen pusher evaluation."""
import json
import os
from pathlib import Path

import boto3
import yaml


def rebind(config, assignment, training, bank, environ):
    assert config["contract"]["source_commit"] == training["source_commit"] == environ["SOURCE_COMMIT"]
    assert assignment["run_id"] == training["run_id"]
    levels = assignment["levels"]
    assert levels and levels == sorted(set(levels)) and set(levels) <= set(range(31))
    config["task_levels"] = levels
    config["run_id"] = assignment["evaluation_id"]
    config["output_prefix"] = assignment["output_prefix"]
    config["resume_sources"] = assignment["resume_sources"]
    config["verify_resume_only"] = assignment.get("verify_resume_only", False)
    config["artifact_policy"]["video_seeds_by_level"] = {
        str(level): [row["seed"] for row in bank["levels"][str(level)]["selected"][:3]]
        for level in range(31)}
    config["evaluation_revision"] = {"name": environ["EVAL_REVISION"],
        "harness_commit": environ["EVAL_HARNESS_COMMIT"],
        "harness_sha256": environ["EVAL_HARNESS_SHA256"],
        "training_source_commit": training["source_commit"],
        "partition": assignment["partition"]}
    return config


if __name__ == "__main__":
    import hashlib
    config_path = Path("/tmp/evaluation.yaml")
    config = yaml.safe_load(config_path.read_text())
    assignment = json.loads(Path("/tmp/evaluation-assignment.json").read_text())
    training = json.loads(Path("/tmp/training-complete.json").read_text())
    client = boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"], region_name="auto")
    bucket, key = config["seed_bank"]["uri"][5:].split("/", 1)
    raw = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    assert hashlib.sha256(raw).hexdigest() == config["seed_bank"]["sha256"]
    result = rebind(config, assignment, training, json.loads(raw), os.environ)
    config_path.write_text(yaml.safe_dump(result, sort_keys=False))
