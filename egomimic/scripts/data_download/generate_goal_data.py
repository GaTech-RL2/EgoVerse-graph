"""Run a pinned benchmark generator and publish independent, hashed shards.

Configuration supplies the generator source/hash, environment, arguments and
seed range. Seeds initialize NumPy and the first Gymnasium reset; the upstream
manipulation generator declares --seed but does not use it itself.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import urllib.request

import yaml


def client():
    import boto3
    from botocore.config import Config
    return boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"], region_name="auto",
        config=Config(retries={"max_attempts": 8}))


def run_shard(work):
    cfg, seed = work
    s3 = client()
    receipt_key = cfg["prefix"].rstrip("/") + f"/receipts/{seed:03d}.json"
    try:
        receipt = json.loads(s3.get_object(Bucket=cfg["bucket"], Key=receipt_key)["Body"].read())
    except s3.exceptions.ClientError as error:
        if error.response["Error"]["Code"] not in {"404", "NoSuchKey"}:
            raise
    else:
        if receipt["generator_sha256"] != cfg["generator_sha256"]:
            raise RuntimeError("existing shard was produced by a different generator")
        for item in receipt["artifacts"]:
            key = item["uri"].split("/", 3)[3]
            head = s3.head_object(Bucket=cfg["bucket"], Key=key)
            assert head["Metadata"]["sha256"] == item["sha256"]
            assert head["ContentLength"] == item["bytes"]
        return receipt
    import gymnasium
    import numpy as np
    from absl import flags
    directory = Path(cfg["output_dir"]) / str(seed)
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "upstream_generator.py"
    with urllib.request.urlopen(cfg["generator_url"], timeout=120) as response:
        source = response.read()
    if hashlib.sha256(source).hexdigest() != cfg["generator_sha256"]:
        raise ValueError("generator source hash mismatch")
    script.write_bytes(source)
    np.random.seed(seed)
    original_make = gymnasium.make

    def seeded_make(*args, **kwargs):
        env = original_make(*args, **kwargs)
        original_reset = env.reset
        first = [True]
        def reset(*args, **kwargs):
            if first[0]:
                kwargs["seed"] = seed
                first[0] = False
            return original_reset(*args, **kwargs)
        env.reset = reset
        return env

    gymnasium.make = seeded_make
    name = cfg["file_stem"] + f"-{seed:03d}.npz"
    path = directory / name
    sys.argv = [str(script), f"--seed={seed}", f"--save_path={path}"]
    sys.argv.extend(f"--{key}={value}" for key, value in cfg["generator_args"].items())
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as error:
        if error.code not in [0, None]:
            raise
    finally:
        gymnasium.make = original_make
        # Each process runs one shard (max_tasks_per_child=1), keeping absl flags
        # and MuJoCo state isolated rather than redefining them for later seeds.
    receipts = []
    for data_path in [path, path.with_name(path.stem + "-val.npz")]:
        sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
        arrays = np.load(data_path)
        semantic = hashlib.sha256()
        for field in sorted(arrays.files):
            array = arrays[field]
            semantic.update(field.encode())
            semantic.update(str((array.shape, array.dtype.str)).encode())
            semantic.update(array.tobytes())
        semantic_sha = semantic.hexdigest()
        key = cfg["prefix"].rstrip("/") + "/" + data_path.name
        # Unique campaign prefixes; completed shards are immutable.
        try:
            existing = s3.head_object(Bucket=cfg["bucket"], Key=key)
        except s3.exceptions.ClientError as error:
            if error.response["Error"]["Code"] not in {"404", "NoSuchKey"}:
                raise
            existing = None
        if existing and existing.get("Metadata", {}).get("array_sha256") != semantic_sha:
            raise RuntimeError("refusing to replace a different existing shard")
        if not existing:
            s3.upload_file(str(data_path), cfg["bucket"], key,
                ExtraArgs={"Metadata": {"sha256": sha, "array_sha256": semantic_sha}})
        head = s3.head_object(Bucket=cfg["bucket"], Key=key)
        assert head["Metadata"]["array_sha256"] == semantic_sha
        receipts.append({"uri": f"s3://{cfg['bucket']}/{key}", "sha256": head["Metadata"]["sha256"],
                         "array_sha256": semantic_sha,
                         "rows": int(len(arrays["actions"])),
                         "episodes": int(arrays["terminals"].sum()), "bytes": head["ContentLength"]})
    report = {"seed": seed, "generator_sha256": cfg["generator_sha256"],
              "generator_args": cfg["generator_args"], "artifacts": receipts}
    s3.put_object(Bucket=cfg["bucket"], Key=receipt_key, Body=json.dumps(report).encode())
    print(json.dumps(report), flush=True)
    # Remove only this worker's uploaded scratch files, never source datasets.
    for data_path in [path, path.with_name(path.stem + "-val.npz")]:
        data_path.unlink()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    with ProcessPoolExecutor(max_workers=cfg["workers"], max_tasks_per_child=1) as pool:
        list(pool.map(run_shard, [(cfg, seed) for seed in cfg["seeds"]]))
