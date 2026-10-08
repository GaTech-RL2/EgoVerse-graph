"""Train a pinned four-source DP/ARC arm and publish full-state checkpoints."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import threading
import time

import boto3
from botocore.config import Config
import numpy as np
import torch
import yaml

ROOT = Path("/workspace/cotrain")
SOURCE = ROOT / "source"
OUT = ROOT / "evidence"
RUN = ROOT / "run"
DATA = ROOT / "data"
NORMALIZERS = ROOT / "normalization"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def main(args):
    for folder in (OUT, RUN, DATA, NORMALIZERS):
        folder.mkdir(parents=True, exist_ok=True)
    spec = yaml.safe_load((SOURCE / "egomimic/hydra_configs/benchmark/planar_streams.yaml").read_text())
    arm = spec["arms"][args.arm]
    destination = spec["artifact_prefix"] + "runs/" + args.run_id + "/"
    client = boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto", config=Config(max_pool_connections=32, retries={"max_attempts": 8}))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip()
    assert commit == os.environ["SOURCE_COMMIT"]
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=SOURCE, text=True).strip()
    assert torch.cuda.device_count() == spec["gpus"]
    assert all("L40S" in torch.cuda.get_device_name(i) for i in range(spec["gpus"]))

    def read(key):
        try:
            response = client.get_object(Bucket="rldb", Key=key)
        except client.exceptions.NoSuchKey:
            return None
        raw = response["Body"].read()
        if response.get("Metadata", {}).get("sha256"):
            assert hashlib.sha256(raw).hexdigest() == response["Metadata"]["sha256"]
        return json.loads(raw)

    def upload(path, key=None):
        key = key or destination + path.relative_to(OUT).as_posix()
        if path.stat().st_size < 64 * 1024 * 1024:
            raw = path.read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            client.put_object(Bucket="rldb", Key=key, Body=raw, Metadata={"sha256": sha})
            return {"uri": "s3://rldb/" + key, "sha256": sha, "bytes": len(raw)}
        sha = digest(path)
        client.upload_file(str(path), "rldb", key, ExtraArgs={"Metadata": {"sha256": sha}})
        return {"uri": "s3://rldb/" + key, "sha256": sha, "bytes": path.stat().st_size}

    def write(name, value):
        p = OUT / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        return upload(p)

    def download(receipt, path):
        key = receipt.get("key") or receipt["uri"].removeprefix("s3://rldb/")
        path.parent.mkdir(parents=True, exist_ok=True)
        client.download_file("rldb", key, str(path))
        assert digest(path) == receipt["sha256"], (key, "checksum")
        return path

    completed = read(destination + "TRAINING_COMPLETE.json")
    if completed:
        assert completed["source_commit"] == commit and completed["global_step"] == spec["updates"]
        (args.output / "training-complete.json").write_text(json.dumps(completed))
        print("ALREADY_COMPLETE", args.run_id, flush=True)
        return
    prior = read(destination + "runtime.json")
    if prior:
        assert prior["source_commit"] == commit and prior["arm"] == args.arm
    ready = json.loads(Path("/tmp/data-ready.json").read_text())
    assert ready["status"] == "PASS" and ready["counts"] == spec["expected_counts"]
    manifest_path = download(ready["manifest"], DATA / "manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest_hash = digest(manifest_path)
    if prior:
        assert prior["data_manifest_sha256"] == manifest_hash
    write("runtime.json", {"source_commit": commit, "arm": args.arm, "data_manifest_sha256": manifest_hash,
        "gpus": [torch.cuda.get_device_name(i) for i in range(spec["gpus"])], "python": sys.version,
        "torch": torch.__version__, "uv_lock_sha256": digest(SOURCE / "uv.lock"), "protocol": spec})
    write("data-ready.json", ready)

    def stage_shard(shard):
        path = ROOT / Path(shard["key"]).name
        download(shard, path)
        with tarfile.open(path) as stream:
            for member in stream.getmembers():
                if member.issym() or member.islnk() or member.name.startswith("/") or ".." in Path(member.name).parts:
                    raise ValueError("Unsafe cache archive member")
            stream.extractall(DATA, filter="data")
        path.unlink()

    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        for i, _ in enumerate(pool.map(stage_shard, manifest["shards"]), 1):
            if i % 10 == 0:
                print("STAGED_CACHE_SHARDS", i, len(manifest["shards"]), flush=True)
    assert {p.name for p in DATA.iterdir() if p.is_dir()} == {e["episode_id"] for e in manifest["episodes"]}
    write("CACHE_STAGED.json", {"episodes": len(manifest["episodes"]), "manifest_sha256": manifest_hash})

    entry = [sys.executable, str(SOURCE / "egomimic/trainHydra.py")]

    def common(output):
        return ["+experiment=pusht/" + arm["experiment"], f"hydra.run.dir={output}", f"paths.output_dir={output}",
            f"paths.work_dir={SOURCE}", f"planar.cotrain_data_root={DATA}", f"planar.data_manifest_sha256={manifest_hash}",
            "seed=42", "ckpt_path=null", f"launch_params.gpus_per_node={spec['gpus']}", "launch_params.nodes=1",
            "trainer.limit_val_batches=0", "trainer.limit_train_batches=1.0", "trainer.num_sanity_val_steps=0",
            "trainer.accumulate_grad_batches=1", "++trainer.enable_progress_bar=false", "runtime.slurm_requeue_owner=none",
            "norm_stats.norm_mode=quantile", "norm_stats.sample_frac=0.05", "norm_stats.num_workers=8",
            f"norm_stats.save_cache_dir={NORMALIZERS}"]

    def execute(command, log_name):
        print("EXECUTE", log_name, flush=True)
        with (OUT / log_name).open("a") as stream:
            proc = subprocess.Popen(command, cwd=SOURCE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                stream.write(line); stream.flush(); print(line, end="", flush=True)
            code = proc.wait()
        upload(OUT / log_name)
        if code:
            raise RuntimeError(f"{log_name}: exit {code}")

    full_args = common(RUN) + [f"trainer.max_steps={spec['updates']}", "norm_stats.save_cache_dir=null",
        f"norm_stats.precomputed_norm_path={NORMALIZERS}/norm_stats", f"logger.wandb.id={args.run_id}",
        f"++logger.wandb.name={args.run_id}", "logger.wandb.group=obstacle-streams-20261007",
        "++logger.wandb.resume=allow", "callbacks.model_checkpoint.every_n_epochs=0",
        "++callbacks.model_checkpoint.every_n_train_steps=10000", "++callbacks.model_checkpoint.save_on_train_epoch_end=false",
        "callbacks.model_checkpoint.filename='step_{step:09d}'", f"callbacks.run_progress.path={OUT}/training-progress.json"]
    resolved = subprocess.check_output(entry + full_args + ["--cfg", "job", "--resolve"], cwd=SOURCE, text=True)
    config = yaml.safe_load(resolved)
    assert config["trainer"]["max_steps"] == config["model"]["scheduler"]["max_steps"] == spec["updates"]
    assert config["planar"]["batch_size"] * spec["gpus"] * 2 == spec["global_batch_size"]
    assert config["model"]["pipeline"]["homogeneous_training"] is True
    config_path = OUT / "resolved-training-config.yaml"
    config_path.write_text(resolved)
    config_receipt = upload(config_path)

    # Representation checks on actual held-out trajectories precede GPU smoke.
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    codec_summary = []
    for source in spec["expected_counts"]:
        selected = [e for e in manifest["episodes"] if e["source"] == source and e["split"] == "valid"][:8]
        if not selected:
            selected = [e for e in manifest["episodes"] if e["source"] == source][:8]
        for row in selected:
            native = np.load(DATA / row["episode_id"] / "actions.npy", mmap_mode="r")
            horizon = config["planar"]["raw_action_horizon"]
            offset = row["action_target_offset_obs2"]
            for start in (0, len(native)//2, len(native)-2):
                indices = np.minimum(np.arange(start, start+horizon+offset), len(native)-1)
                raw = np.asarray(native[indices]).copy()
                transform_cfg = config["data"]["train_datasets"][row["embodiment"]]["resolver"]["transform_list"]
                transforms = instantiate(OmegaConf.create(dict(transform_cfg, action_target_offset=offset)))
                sample = {"actions": raw.copy()}
                for transform in transforms:
                    sample = transform.transform(sample)
                decoder = instantiate(OmegaConf.create(config["evaluator"]["native_decoders"][row["embodiment"]]))
                decoded = np.asarray(decoder(sample["actions"])).reshape(horizon, native.shape[1])
                assert np.isfinite(decoded).all()
                error = decoded - raw[offset:]
                error[:, 2] = np.arctan2(np.sin(error[:, 2]), np.cos(error[:, 2]))
                assert np.abs(error[0]).max() < 1e-4
                codec_summary.append({"episode": row["episode_id"], "split": row["split"], "start": start,
                    "rmse": np.sqrt(np.mean(error**2, axis=0)).tolist()})
    write("codec-audit.json", {"arm": args.arm, "windows": codec_summary, "is_policy_score": False})

    records = read(destination + "checkpoint-receipts.json") or []
    stop = threading.Event()
    uploaded = {}
    checkpoint_seen = set()
    uploader_errors = []

    def sync(final=False):
        for path in OUT.rglob("*"):
            if path.is_file() and path.suffix != ".tmp" and path.stat().st_size < 64*1024*1024:
                sha = digest(path)
                if uploaded.get(str(path)) != sha:
                    upload(path); uploaded[str(path)] = sha
        for path in sorted((RUN / "checkpoints").glob("*.ckpt")):
            if path.name == "last.ckpt" and not final:
                continue
            identity = (path.name, path.stat().st_size, path.stat().st_mtime_ns)
            if identity in checkpoint_seen or (not final and time.time() - path.stat().st_mtime < 60):
                continue
            staged = ROOT / "checkpoint-snapshot.ckpt"
            shutil.copyfile(path, staged)
            if identity != (path.name, path.stat().st_size, path.stat().st_mtime_ns):
                continue
            ckpt = torch.load(staged, map_location="cpu", mmap=True, weights_only=False)
            step, ema = int(ckpt["global_step"]), int(ckpt["ema_num_updates"])
            assert step == ema, (step, ema)
            del ckpt
            sha = digest(staged)
            receipt = upload(staged, destination + f"checkpoints/{sha}/{path.name}")
            records.append({**receipt, "global_step": step, "ema_num_updates": ema, "source_commit": commit})
            checkpoint_seen.add(identity)
            write("checkpoint-receipts.json", records)
            staged.unlink()
            # Immutable remote checkpoint is durable; bound local disk usage.
            if path.name != "last.ckpt":
                path.unlink()
            print("CHECKPOINT_UPLOADED", step, sha, flush=True)

    def loop():
        while not stop.wait(60):
            try:
                sync()
            except Exception as e:
                uploader_errors.append(type(e).__name__)
                print("ARTIFACT_RETRY", type(e).__name__, flush=True)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    try:
        norm = NORMALIZERS / "norm_stats/norm_stats.json"
        prior_norm = read(destination + "normalizer-receipt.json")
        if prior_norm:
            assert prior_norm["data_manifest_sha256"] == manifest_hash
            download(prior_norm["artifact"], norm)
        else:
            execute(entry + common(ROOT / "normalization-log") + ["norm_stats_only=true", "norm_stats.precomputed_norm_path=null", "~logger"], "normalization.log")
            assert "normalizer_state" in json.loads(norm.read_text())
            norm_receipt = upload(norm, destination + "normalizer.json")
            write("normalizer-receipt.json", {"artifact": norm_receipt, "data_manifest_sha256": manifest_hash, "train_only": True})
        norm_receipt = upload(norm, destination + "normalizer.json")
        execute(entry + common(ROOT / "smoke") + ["trainer.max_steps=4", "~logger", "~callbacks.model_checkpoint",
            "++trainer.enable_checkpointing=false", "norm_stats.save_cache_dir=null",
            f"norm_stats.precomputed_norm_path={NORMALIZERS}/norm_stats",
            f"callbacks.run_progress.path={OUT}/smoke-progress.json"], "four-gpu-smoke.log")
        smoke = json.loads((OUT / "smoke-progress.json").read_text())
        assert smoke["global_step"] == 4 and smoke["batch_layout"]["fused"] and np.isfinite(smoke["loss"])
        write("SMOKE_PASSED.json", {"steps": 4, "gpus": spec["gpus"], "fresh_full_run": not records,
            "source_commit": commit, "trainable_parameters": smoke["trainable_parameters"]})
        if records:
            latest = max(records, key=lambda r: r["global_step"])
            assert latest["source_commit"] == commit and latest["global_step"] <= spec["updates"]
            path = download(latest, ROOT / "resume.ckpt")
            full_args = [arg for arg in full_args if not arg.startswith("ckpt_path=")] + [f"ckpt_path={path}"]
            write("RESUMED.json", {"checkpoint": latest, "normalizer": norm_receipt, "data_manifest_sha256": manifest_hash})
        elif prior and read(destination + "training-progress.json"):
            raise RuntimeError("Previous attempt has no durable checkpoint; refusing an implicit fresh restart")
        write("RUN_STATUS.json", {"status": "TRAINING", "arm": args.arm, "updates": spec["updates"],
                                  "resume_step": max((r["global_step"] for r in records), default=0)})
        execute(entry + full_args, "training.log")
        stop.set(); thread.join()
        sync(final=True)
        final = max(records, key=lambda r: r["global_step"])
        assert final["global_step"] == spec["updates"]
        complete = {"arm": args.arm, "run_id": args.run_id, "source_commit": commit, "global_step": spec["updates"],
            "data_manifest_sha256": manifest_hash, "checkpoint": final, "config": config_receipt,
            "normalizer": norm_receipt, "trainable_parameters": smoke["trainable_parameters"],
            "uploader_transient_errors": uploader_errors}
        write("TRAINING_COMPLETE.json", complete)
        (args.output / "training-complete.json").write_text(json.dumps(complete))
    except BaseException as e:
        write("FAILED.json", {"type": type(e).__name__, "message": str(e), "time": time.time()})
        raise
    finally:
        stop.set(); thread.join(); sync()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    main(arguments)
