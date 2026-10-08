"""Bind the previous frozen rollout protocol to this run's final checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import boto3
from botocore.config import Config
import yaml


def main(args):
    root = Path(__file__).resolve().parents[2]
    base = yaml.safe_load((root / "egomimic/hydra_configs/benchmark/planar_stream_evaluation.yaml").read_text())
    campaign = yaml.safe_load((root / "egomimic/hydra_configs/benchmark/planar_streams.yaml").read_text())
    complete = json.loads(args.training.read_text())
    assert complete["source_commit"] == os.environ["SOURCE_COMMIT"]
    assert complete["global_step"] == campaign["updates"]
    assert complete["arm"] in campaign["arms"]
    client = boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto", config=Config(retries={"max_attempts": 8}))
    receipt = base["seed_bank"]
    response = client.get_object(Bucket="rldb", Key=receipt["uri"].removeprefix("s3://rldb/"))
    raw = response["Body"].read()
    assert hashlib.sha256(raw).hexdigest() == receipt["sha256"]
    bank = json.loads(raw)
    assert bank["contract"] == base["seed_bank_contract"]
    levels = list(range(31))[args.shard::args.shards]
    run_id = complete["run_id"] + f"-{args.domain}-s{args.shard}"
    config = {k: v for k, v in base.items() if k not in ("domains", "video_previews_per_level")}
    config.update(base["domains"][args.domain])
    config["contract"]["source_commit"] = complete["source_commit"]
    config.update(run_id=run_id, output_prefix=campaign["artifact_prefix"] + "evaluation/" + run_id + "/",
        task_levels=levels, data_manifest_sha256=complete["data_manifest_sha256"],
        arm={"arm": complete["arm"], "restore": {"checkpoint.ckpt": complete["checkpoint"],
            "config.yaml": complete["config"], "normalizer.json": complete["normalizer"]}},
        artifact_policy={"video_seeds_by_level": {str(level): [r["seed"] for r in bank["levels"][str(level)]["selected"][:base["video_previews_per_level"]]] for level in levels},
            "policy": "first three frozen seeds per level; all episode scores and action traces"})
    args.output.write_text(yaml.safe_dump(config, sort_keys=False))
    print("EVALUATION_BOUND", run_id, len(levels)*50, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--domain", choices=("usocket", "chaingripper"), required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("/tmp/evaluation.yaml"))
    main(parser.parse_args())
