"""Bind the shared triangle holdout protocol to a final pusher checkpoint."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import boto3
from botocore.config import Config
import yaml


def bind(base, campaign, complete, bank, shard, shards):
    assert 0 <= shard < shards
    assert complete["global_step"] == campaign["optimizer_updates"]
    assert complete["arm"] in campaign["arms"]
    assert complete["seed"] in campaign["training_seeds"]
    assert complete["regime"] in campaign["regimes"]
    assert bank["source_selection_sha256"] == campaign["source_selection_sha256"]
    assert bank["simulator"] == complete["evaluation"]["simulator"]
    levels = list(range(31))[shard::shards]
    config = copy.deepcopy(base)
    config["contract"]["source_commit"] = complete["source_commit"]
    config["contract"].update(bank["contract"])
    run_id = complete["run_id"] + f"-triangle-s{shard}"
    config.update(complete["evaluation"])
    config.update(run_id=run_id, output_prefix=campaign["artifact_prefix"] + "evaluation/" + run_id + "/",
        task_levels=levels, data_manifest_sha256=complete["data_manifest_sha256"],
        source_selection_sha256=campaign["source_selection_sha256"], seed_bank_contract=bank["contract"],
        training_seed=complete["seed"], regime=complete["regime"], parent_sweep=campaign["artifact_prefix"],
        arm={"arm": complete["arm"], "restore": {"checkpoint.ckpt": complete["checkpoint"],
            "config.yaml": complete["config"], "normalizer.json": complete["normalizer"]}},
        artifact_policy={"video_seeds_by_level": {str(level): [r["seed"] for r in bank["levels"][str(level)]["selected"][:base["video_previews_per_level"]]] for level in levels},
            "policy": "first three frozen seeds per level; all episode scores and action traces"})
    return config


def main(args):
    root = Path(__file__).resolve().parents[2]
    base = yaml.safe_load((root / "egomimic/hydra_configs/benchmark/pusher_holdout_evaluation.yaml").read_text())
    campaign = yaml.safe_load((root / "egomimic/hydra_configs/benchmark/pusher_triangle_holdout.yaml").read_text())
    complete = json.loads(args.training.read_text())
    assert complete["source_commit"] == os.environ["SOURCE_COMMIT"]
    client = boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto", config=Config(retries={"max_attempts": 8}))
    receipt = complete["evaluation"]["seed_bank"]
    raw = client.get_object(Bucket="rldb", Key=receipt["uri"].removeprefix("s3://rldb/"))["Body"].read()
    assert hashlib.sha256(raw).hexdigest() == receipt["sha256"]
    config = bind(base, campaign, complete, json.loads(raw), args.shard, args.shards)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False))
    print("EVALUATION_BOUND", config["run_id"], len(config["task_levels"])*50, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("/tmp/evaluation.yaml"))
    main(parser.parse_args())
