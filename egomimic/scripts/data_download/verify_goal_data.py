"""Verify every generated shard before releasing a dataset to RL workers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import yaml

from egomimic.scripts.data_download.generate_goal_data import client


def verify(cfg):
    s3 = client()
    def check(seed):
        key = cfg["prefix"].rstrip("/") + f"/receipts/{seed:03d}.json"
        receipt = json.loads(s3.get_object(Bucket=cfg["bucket"], Key=key)["Body"].read())
        assert receipt["seed"] == seed and receipt["generator_sha256"] == cfg["generator_sha256"]
        train, validation = receipt["artifacts"]
        assert train["episodes"] == cfg["episodes_per_shard"]
        assert validation["episodes"] == cfg["episodes_per_shard"] // 10
        for role in [train, validation]:
            head = s3.head_object(Bucket=cfg["bucket"], Key=role["uri"].split("/", 3)[3])
            assert head["Metadata"]["sha256"] == role["sha256"] and head["ContentLength"] == role["bytes"]
        return {"url": train["uri"], "sha256": train["sha256"],
                "array_sha256": train["array_sha256"],
                "validation_url": validation["uri"], "validation_sha256": validation["sha256"],
                "rows": train["rows"], "episodes": train["episodes"]}
    with ThreadPoolExecutor(max_workers=16) as pool:
        shards = list(pool.map(check, cfg.get("seeds", list(range(cfg["shards"])))))
    assert len(shards) == cfg["shards"]
    # Distinct shard files are required; a repeated upload cannot fake scale.
    assert len({row["sha256"] for row in shards}) == len(shards)
    assert len({row["array_sha256"] for row in shards}) == len(shards)
    transitions = sum(row["rows"] - row["episodes"] for row in shards)
    assert transitions == cfg["expected_transitions"]
    manifest = {"status": "READY", "shards": shards, "transitions": transitions,
                "generator_sha256": cfg["generator_sha256"]}
    raw = (json.dumps(manifest, indent=2) + "\n").encode()
    sha = hashlib.sha256(raw).hexdigest()
    s3.put_object(Bucket=cfg["bucket"], Key=cfg["prefix"].rstrip("/") + "/manifest.json",
                  Body=raw, Metadata={"sha256": sha})
    print(json.dumps({"status": "READY", "transitions": transitions, "sha256": sha}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    verify(yaml.safe_load(Path(args.config).read_text()))
