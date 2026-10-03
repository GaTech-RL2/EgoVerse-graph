"""Wait on CPU for a completed, checksummed reference policy evaluation."""

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

from botocore.exceptions import ClientError


def reference_ready(client, run_id, suite, commit, *, training_run=None, mode="stk"):
    if mode not in {"stk", "dur"}:
        raise ValueError("Invalid reference timing mode")
    prefix = f"experiments/arc-oat-20260919/{run_id}/"

    def read(name):
        response = client.get_object(Bucket="rldb", Key=prefix + name)
        body = response["Body"].read()
        sha = hashlib.sha256(body).hexdigest()
        if response.get("Metadata", {}).get("sha256") != sha:
            raise ValueError(f"Reference artifact checksum differs: {name}")
        return json.loads(body), sha

    try:
        status, status_sha = read("status.json")
    except ClientError as error:
        if error.response["Error"]["Code"] in {"NoSuchKey", "404", "NotFound"}:
            return None
        raise
    if status.get("state") == "FAILED":
        raise RuntimeError(f"Reference evaluation failed: {run_id}")
    if status.get("state") != "POLICY_EVALUATION_COMPLETE":
        return None
    if status.get("episodes") != 2500 or status.get("complete_protocol") is not True:
        raise ValueError("Reference evaluation did not complete its full protocol")
    runtime, runtime_sha = read("runtime.json")
    expected = {
        "source_commit": commit,
        "suite": suite,
        "method": f"arc_{mode}",
        "run_kind": "policy_evaluation",
        "mode": "full",
        "evaluate_from_run": training_run or run_id.removesuffix("-eval"),
    }
    if any(runtime.get(key) != value for key, value in expected.items()):
        raise ValueError("Reference evaluation source or protocol differs")
    return {
        "reference_run": run_id,
        "suite": suite,
        "source_commit": commit,
        "arc_mode": mode,
        "status_sha256": status_sha,
        "runtime_sha256": runtime_sha,
    }


def main():
    import boto3
    from botocore.config import Config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--training-run")
    parser.add_argument("--mode", choices=("stk", "dur"), default="stk")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=7 * 86400)
    args = parser.parse_args()
    if (
        not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", args.run_id)
        or not args.run_id.endswith(f"-reference-{args.mode}-eval")
        or args.suite
        not in {"libero_spatial", "libero_object", "libero_goal", "libero_10"}
        or not re.fullmatch(r"[0-9a-f]{40}", args.commit)
        or args.timeout_seconds <= 0
        or (
            args.training_run is not None
            and not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", args.training_run)
        )
    ):
        raise ValueError("Invalid reference readiness request")
    client = boto3.client(
        "s3",
        region_name="auto",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(connect_timeout=10, read_timeout=30, retries={"max_attempts": 3}),
    )
    deadline = time.monotonic() + args.timeout_seconds
    while time.monotonic() < deadline:
        receipt = reference_ready(
            client,
            args.run_id,
            args.suite,
            args.commit,
            training_run=args.training_run,
            mode=args.mode,
        )
        if receipt is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(receipt, indent=2) + "\n")
            print("REFERENCE_READY", args.run_id, flush=True)
            return
        print("WAITING_FOR_REFERENCE", args.run_id, flush=True)
        time.sleep(30)
    raise TimeoutError(f"Reference evaluation did not complete: {args.run_id}")


if __name__ == "__main__":
    main()
