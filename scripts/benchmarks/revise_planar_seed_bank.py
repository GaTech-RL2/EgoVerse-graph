"""Freeze paired replacement resets when a new corpus overlaps an older bank."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tarfile
import traceback

import boto3
import numpy as np

ROOT = Path("/workspace/seed-revision")
CONFIG = json.loads(Path("/tmp/seed-revision.json").read_text())
CLIENT = boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT_URL"],
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"], region_name="auto")


def download(receipt, path):
    key = receipt.get("key") or receipt["uri"].removeprefix("s3://rldb/")
    path.parent.mkdir(parents=True, exist_ok=True)
    CLIENT.download_file("rldb", key, str(path))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["sha256"]


def write(name, value):
    raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    key = CONFIG["output_prefix"] + name
    sha = hashlib.sha256(raw).hexdigest()
    CLIENT.put_object(Bucket="rldb", Key=key, Body=raw, Metadata={"sha256": sha})
    return {"uri": "s3://rldb/" + key, "sha256": sha, "bytes": len(raw)}


def revise_level(level):
    sys.path.insert(0, str(ROOT / "simulator"))
    from Tsimulation.sim_v2.pushshapes.env import PushShapesEnv
    from planar_seed_rules import paired_candidate, pose_vector, pose_matches
    bank = json.loads((ROOT / "old-bank.json").read_text())
    corpus = json.loads((ROOT / "manifest.json").read_text())["episodes"]
    contract = CONFIG["contract"]
    # Exclude both training and validation data, across all layouts.
    poses = np.asarray([pose_vector(e["episode_init"]) for e in corpus])
    known_seeds = {int(e["episode_init"]["reset_seed"]) for e in corpus
        if e["obstacle_level"] == level and e["episode_init"].get("reset_seed") is not None}
    original = bank["levels"][str(level)]
    kept, removed = [], []
    for row in original["selected"]:
        overlap = int(row["seed"]) in known_seeds or any(
            pose_matches(pose_vector(init), poses, contract["initial_pose_tolerance"])
            for init in row["initial_states"].values())
        (removed if overlap else kept).append(row)
    if not removed:
        return level, dict(original, replacement_audit={"removed": [], "added": [], "attempted": 0})
    environments = [PushShapesEnv(object_shape="T", pusher_shape=p, obstacle_level=level, image_size=8)
                    for p in ("chain_gripper", "u_socket")]
    for env in environments:
        env._skip_obs_render = True
    added, attempted = [], 0
    vectors = [pose_vector(row["initial_states"]["chaingripper"]) for row in kept]
    first = max(int(row["seed"]) for row in original["selected"]) + 1
    try:
        for seed in range(first, contract["seed_stop_exclusive"]):
            if seed in known_seeds:
                continue
            attempted += 1
            row = paired_candidate(*environments, seed, contract, poses)
            if row is None:
                continue
            vector = pose_vector(row["initial_states"]["chaingripper"])
            if pose_matches(vector, vectors, contract["initial_pose_tolerance"]):
                continue
            added.append(row); vectors.append(vector)
            if len(added) == len(removed):
                break
        assert len(added) == len(removed), (level, len(added), len(removed))
        selected = kept + added
        assert len(selected) == 50 and len({r["seed"] for r in selected}) == 50
        return level, {**original, "selected": selected, "replacement_audit": {
            "removed": [r["seed"] for r in removed], "added": [r["seed"] for r in added],
            "attempted": attempted, "seed_start": first}}
    finally:
        for env in environments:
            env.close()


def main():
    assert not CLIENT.list_objects_v2(Bucket="rldb", Prefix=CONFIG["output_prefix"], MaxKeys=1).get("KeyCount")
    versions = {p: importlib.metadata.version(p) for p in CONFIG["contract"]["simulator_dependencies"]}
    assert versions == CONFIG["contract"]["simulator_dependencies"]
    ROOT.mkdir(parents=True, exist_ok=True)
    download(CONFIG["simulator"], ROOT / "simulator.tar")
    with tarfile.open(ROOT / "simulator.tar") as stream:
        stream.extractall(ROOT / "simulator", filter="data")
    assert hashlib.sha256((ROOT / "simulator/Tsimulation/sim_v2/pushshapes/obstacles.py").read_bytes()).hexdigest() == CONFIG["contract"]["obstacles_sha256"]
    download(CONFIG["old_seed_bank"], ROOT / "old-bank.json")
    download(CONFIG["manifest"], ROOT / "manifest.json")
    write("STARTED.json", {"config": CONFIG, "versions": versions,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    levels = {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(revise_level, level) for level in range(31)]):
            level, result = future.result(); levels[str(level)] = result
            write(f"level-{level:02d}.json", result)
            print("LEVEL_REVISED", level, result["replacement_audit"], flush=True)
    bank = json.loads((ROOT / "old-bank.json").read_text())
    bank.update(name=CONFIG["contract"]["name"], contract=CONFIG["contract"],
        created_at_utc=datetime.now(timezone.utc).isoformat(), versions=versions,
        source_seed_bank=CONFIG["old_seed_bank"], excluded_corpus=CONFIG["manifest"],
        levels={str(level): levels[str(level)] for level in range(31)})
    receipt = write("seed-bank.json", bank)
    write("READY.json", {"status": "PASS", "seed_bank": receipt, "contract": CONFIG["contract"],
        "removed_seed_pairs": sum(len(level["replacement_audit"]["removed"]) for level in levels.values()),
        "paired_resets": 1550, "excludes_train_and_validation": True})


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        write("FAILED.json", {"traceback": traceback.format_exc()})
        raise
