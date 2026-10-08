"""Release GPU work only after a pinned dataset and reset-overlap audit pass."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import boto3
from botocore.config import Config
import numpy as np


def main(spec, evaluation, output):
    client = boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto", config=Config(retries={"max_attempts": 8}))

    def read(key, sha=None):
        try:
            response = client.get_object(Bucket="rldb", Key=key)
        except client.exceptions.NoSuchKey:
            return None
        raw = response["Body"].read()
        expected = sha or response.get("Metadata", {}).get("sha256")
        if expected and hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Artifact hash mismatch: " + key)
        return json.loads(raw)

    data_prefix = spec["data_ready_key"].rsplit("/", 1)[0] + "/"
    deadline = time.monotonic() + 12*3600
    while time.monotonic() < deadline:
        failed = read(data_prefix + "FAILED.json")
        if failed:
            raise RuntimeError("Dataset preparation failed: " + json.dumps(failed))
        ready = read(spec["data_ready_key"])
        if ready:
            break
        print("WAITING_FOR_DATA", spec["data_ready_key"], flush=True)
        time.sleep(30)
    else:
        raise TimeoutError("No dataset completion receipt after 12 hours")
    started = read(data_prefix + "STARTED.json")
    assert started["script_sha256"] == spec["data_preparation_sha256"]
    assert ready["status"] == "PASS" and ready["counts"] == spec["expected_counts"]
    manifest = read(ready["manifest"]["key"], ready["manifest"]["sha256"])
    assert manifest["counts"] == ready["counts"]
    bank = read(evaluation["seed_bank"]["uri"].removeprefix("s3://rldb/"), evaluation["seed_bank"]["sha256"])
    assert bank["contract"] == evaluation["seed_bank_contract"]
    assert sorted(map(int, bank["levels"])) == list(range(31))
    overlaps, seed_overlaps = [], []
    for label, domain in evaluation["domains"].items():
        name = domain["domain"]["embodiment"]
        training = [e for e in manifest["episodes"] if e["embodiment"] == name and e["split"] == "train"]
        poses = np.asarray([e["episode_init"]["object_pose"] + e["episode_init"]["goal_pose"] for e in training])
        known_seeds = {(e["obstacle_level"], e["episode_init"].get("reset_seed")) for e in training
                       if e["episode_init"].get("reset_seed") is not None}
        for level, rows in bank["levels"].items():
            assert len(rows["selected"]) == 50
            for row in rows["selected"]:
                init = row["initial_states"][label]
                delta = poses - np.asarray(init["object_pose"] + init["goal_pose"])
                delta[:, [2, 5]] = (delta[:, [2, 5]] + np.pi) % (2*np.pi) - np.pi
                if np.any(np.max(np.abs(delta), axis=1) <= bank["contract"]["initial_pose_tolerance"]):
                    overlaps.append([label, int(level), row["seed"]])
                if (int(level), int(row["seed"])) in known_seeds:
                    seed_overlaps.append([label, int(level), row["seed"]])
    audit = {"source_commit": os.environ["SOURCE_COMMIT"], "data_manifest_sha256": ready["manifest"]["sha256"],
        "seed_bank_sha256": evaluation["seed_bank"]["sha256"], "initial_pose_overlaps": overlaps,
        "known_seed_overlaps": seed_overlaps, "missing_training_seed_metadata_is_not_evidence_of_disjoint_seeds": True}
    raw = (json.dumps(audit, indent=2) + "\n").encode()
    client.put_object(Bucket="rldb", Key=spec["artifact_prefix"] + "preflight/reset-overlap-audit.json",
        Body=raw, Metadata={"sha256": hashlib.sha256(raw).hexdigest()})
    if overlaps or seed_overlaps:
        raise RuntimeError("Frozen evaluation resets overlap the new training corpus")
    output.mkdir(parents=True, exist_ok=True)
    (output / "data-ready.json").write_text(json.dumps(ready))
    (output / "reset-overlap-audit.json").write_bytes(raw)
    print("DATA_GATE_PASSED", ready["episodes"], flush=True)


if __name__ == "__main__":
    main(json.loads(Path(sys.argv[1]).read_text()), json.loads(Path(sys.argv[2]).read_text()), Path(sys.argv[3]))
