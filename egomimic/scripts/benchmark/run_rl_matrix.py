"""Run a configured seed matrix with durable resume and separate processes."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf

from egomimic.rldb.goal_replay import download_verified
from egomimic.utils.experiment_artifacts import ArtifactWriter


def read_optional(client, bucket, key):
    try:
        return json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read())
    except client.exceptions.ClientError as error:
        if error.response["Error"]["Code"] not in {"404", "NoSuchKey"}:
            raise
        return None


def run_matrix(matrix):
    writer = ArtifactWriter(matrix.output_dir, bucket=matrix.bucket, prefix=matrix.prefix)
    template = OmegaConf.create(OmegaConf.to_container(matrix.run, resolve=False))
    if matrix.get("dataset_manifest"):
        manifest_path = writer.directory / "dataset-manifest.json"
        download_verified(matrix.dataset_manifest, manifest_path)
        manifest = json.loads(manifest_path.read_text())
        if manifest["status"] != "READY":
            raise ValueError("dataset manifest is incomplete")
        template.shards = manifest["shards"]
    for seed in matrix.seeds:
        cfg = OmegaConf.create(OmegaConf.to_container(template, resolve=False))
        cfg.seed = int(seed)
        cfg.run_id = f"{matrix.name}-s{seed}"
        cfg.output_dir = str(writer.directory / cfg.run_id)
        cfg.artifacts.prefix = matrix.prefix.rstrip("/") + "/" + cfg.run_id
        status_key = cfg.artifacts.prefix + "/status.json"
        status = read_optional(writer.client, matrix.bucket, status_key)
        if status and status.get("state") == "COMPLETED" and status["step"] == cfg.steps:
            print(json.dumps({"skip_completed": cfg.run_id, "step": status["step"]}), flush=True)
            continue
        checkpoint = read_optional(writer.client, matrix.bucket, cfg.artifacts.prefix + "/latest-checkpoint.json")
        if checkpoint:
            resume = writer.directory / "resume" / (cfg.run_id + "-" + checkpoint["sha256"][:16] + ".ckpt")
            download_verified(checkpoint["uri"], resume, checkpoint["sha256"])
            cfg.resume = str(resume)
        path = writer.directory / (cfg.run_id + ".yaml")
        OmegaConf.save(cfg, path)
        writer.publish(path)
        subprocess.run([sys.executable, "-m", "egomimic.trainRL", "--config", str(path)], check=True)
    writer.json("matrix-status.json", {"state": "COMPLETED", "seeds": list(matrix.seeds)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_matrix(OmegaConf.load(args.config))
