"""Relay a completed stream policy's pinned evaluation request using only CPU."""

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

from botocore.exceptions import ClientError


def checkpoint_ready(client, identity):
    run = identity["source_run"]
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", run):
        raise ValueError("Invalid policy run ID")
    prefix = f"experiments/arc-oat-20260919/{run}/"
    receipts = {}

    def read(name):
        try:
            response = client.get_object(Bucket="rldb", Key=prefix + name)
        except ClientError as error:
            if error.response["Error"]["Code"] in {"NoSuchKey", "404", "NotFound"}:
                return None
            raise
        body = response["Body"].read()
        sha = hashlib.sha256(body).hexdigest()
        if response.get("Metadata", {}).get("sha256") != sha:
            raise ValueError(f"Policy artifact checksum differs: {name}")
        receipts[name] = sha
        return json.loads(body)

    status = read("status.json")
    if status is None or status.get("state") not in {"TRAINING_COMPLETE", "FAILED"}:
        return None
    if status["state"] == "FAILED":
        raise RuntimeError(f"Source training failed: {run}")
    expected = {
        "epochs_completed": 5001,
        "global_step": identity["total_optimizer_steps"],
        "ema_num_updates": identity["total_optimizer_steps"],
    }
    if any(status.get(k) != v for k, v in expected.items()):
        raise ValueError("Policy has not completed its exact optimizer/EMA budget")
    budget = status.get("training_budget", {})
    if any(
        budget.get(k) != v
        for k, v in {
            "epochs": 5001,
            "total_optimizer_steps": identity["total_optimizer_steps"],
            "global_batch_size": 1024,
            "world_size": 4,
        }.items()
    ):
        raise ValueError("Policy training layout differs")
    runtime = read("runtime.json")
    if runtime is None:
        return None
    mode = identity["method"].removeprefix("arc_")
    expected_runtime = {
        "run_kind": "arc_streams",
        "source_commit": identity["source_commit"],
        "suite": identity["suite"],
        "mode": "full",
        "operation": "train",
        "arc_stream_variant": identity["arc_stream_variant"],
        "arc_mode": mode,
    }
    if mode not in {"stk", "dur"} or any(
        runtime.get(k) != v for k, v in expected_runtime.items()
    ):
        raise ValueError("Policy identity differs")
    checkpoints = read("checkpoint-receipts.json")
    if checkpoints is None:
        return None
    checkpoint = checkpoints.get(f"training/{identity['method']}/checkpoints/last.ckpt")
    if checkpoint is None:
        return None
    sha = checkpoint.get("sha256", "")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", sha)
        or type(checkpoint.get("bytes")) is not int
        or checkpoint["bytes"] < 1
        or checkpoint.get("uri") != f"s3://rldb/{prefix}checkpoints/{sha}/last.ckpt"
    ):
        raise ValueError("Checkpoint is not pinned within its source run")
    # The native evaluator subsequently downloads and checks the full checkpoint,
    # exact optimizer/EMA budget, normalizer and representation before rollouts.
    return {**identity, "epochs": 5001, "checkpoint": checkpoint}, receipts


def main():
    import boto3
    from botocore.config import Config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    identity = json.loads(args.identity.read_text())
    client = boto3.client(
        "s3",
        region_name="auto",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(connect_timeout=10, read_timeout=30, retries={"max_attempts": 3}),
    )
    deadline = time.monotonic() + 7 * 86400
    while time.monotonic() < deadline:
        result = checkpoint_ready(client, identity)
        if result:
            request, receipts = result
            # Allow final uploads to settle, then re-read every checksum before
            # releasing the evaluation GPU. Incomplete payloads still fail the
            # evaluator's independent full-file and training-budget validation.
            time.sleep(60)
            confirmed = checkpoint_ready(client, identity)
            if confirmed != result:
                continue
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "evaluation-request.json").write_text(
                json.dumps(request, indent=2) + "\n"
            )
            (args.output / "checkpoint-ready.json").write_text(
                json.dumps(receipts, indent=2) + "\n"
            )
            print("CHECKPOINT_READY", identity["source_run"], flush=True)
            return
        print("WAITING_FOR_CHECKPOINT", identity["source_run"], flush=True)
        time.sleep(30)
    raise TimeoutError(f"Training did not complete: {identity['source_run']}")


if __name__ == "__main__":
    main()
