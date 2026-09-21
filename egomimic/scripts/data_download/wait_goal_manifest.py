"""CPU readiness gate for an existing data-generation workflow; never writes data."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import yaml

from egomimic.scripts.data_download.generate_goal_data import client


def validate_manifest(raw, metadata, cfg):
    digest = hashlib.sha256(raw).hexdigest()
    if digest != metadata.get("sha256"):
        raise ValueError("dataset manifest SHA-256 does not match its receipt")
    manifest = json.loads(raw)
    if (manifest["status"] != "READY" or manifest["generator_sha256"] != cfg["generator_sha256"]
            or manifest["transitions"] != cfg["expected_transitions"]):
        raise ValueError("dataset manifest does not satisfy the requested data contract")
    shards = manifest["shards"]
    if (len(shards) != cfg["shards"]
            or len({row["sha256"] for row in shards}) != len(shards)
            or len({row["array_sha256"] for row in shards}) != len(shards)
            or sum(row["rows"] - row["episodes"] for row in shards) != cfg["expected_transitions"]):
        raise ValueError("dataset shard count, uniqueness, or transition count mismatch")
    return {"state": "READY", "manifest_sha256": digest,
            "transitions": manifest["transitions"], "shards": len(shards)}


def wait(cfg):
    s3 = client()
    start = time.monotonic()
    while time.monotonic() - start < cfg["timeout_seconds"]:
        try:
            response = s3.get_object(Bucket=cfg["bucket"], Key=cfg["manifest_key"])
        except s3.exceptions.ClientError as error:
            if error.response["Error"]["Code"] not in {"404", "NoSuchKey"}:
                raise
            print(json.dumps({"state": "WAITING_FOR_EXISTING_DATA", "manifest": cfg["manifest_key"],
                              "elapsed_seconds": round(time.monotonic() - start)}), flush=True)
            time.sleep(cfg["poll_seconds"])
            continue
        receipt = validate_manifest(response["Body"].read(), response["Metadata"], cfg)
        receipt["manifest_uri"] = f"s3://{cfg['bucket']}/{cfg['manifest_key']}"
        raw = (json.dumps(receipt, indent=2) + "\n").encode()
        s3.put_object(Bucket=cfg["receipt_bucket"], Key=cfg["receipt_key"], Body=raw,
                      Metadata={"sha256": hashlib.sha256(raw).hexdigest()})
        print(json.dumps(receipt), flush=True)
        return receipt
    raise TimeoutError("existing dataset generation did not publish a verified manifest before timeout")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    wait(yaml.safe_load(Path(args.config).read_text()))
