"""Release GPU jobs after the pusher cache, family and reset audits pass."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import boto3
from botocore.config import Config
import numpy as np


def main(spec, regime, output):
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
        assert expected and hashlib.sha256(raw).hexdigest() == expected, key
        return json.loads(raw)

    def asset(receipt):
        return read(receipt.get("key") or receipt["uri"].removeprefix("s3://rldb/"), receipt["sha256"])

    prefix = spec["data_release_prefix"]
    deadline = time.monotonic() + 18*3600
    while time.monotonic() < deadline:
        failed = read(prefix + "FAILED.json")
        if failed:
            raise RuntimeError("Dataset preparation failed: " + json.dumps(failed))
        complete = read(prefix + "COMPLETE.json")
        if complete:
            break
        print("WAITING_FOR_DATA", regime, flush=True)
        time.sleep(30)
    else:
        raise TimeoutError("No completed data release after 18 hours")
    started = read(prefix + "STARTED.json")
    assert started["script_sha256"] == spec["data_finalizer_sha256"]
    assert started["config"]["selection_sha256"] == spec["source_selection_sha256"]
    assert complete["status"] == "PASS"
    ready = asset(complete["releases"][regime])
    assert ready["status"] == "PASS" and ready["regime"] == regime
    assert ready["episodes"] == spec["regimes"][regime]["episodes"]
    manifest = asset(ready["manifest"])
    assert manifest["source_selection_sha256"] == spec["source_selection_sha256"]
    assert manifest["counts"] == ready["counts"]
    rows = manifest["episodes"]
    assert len(rows) == ready["episodes"] == len({r["episode_id"] for r in rows})
    family_splits = {}
    for row in rows:
        tool = row["embodiment"].removeprefix("pushshapes_sim_")
        assert row["source"] in spec["sources"][tool]
        assert regime != "clean" or row["collection"] == "clean"
        family_splits.setdefault(row["family_sha256"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in family_splits.values())
    bank = asset(ready["seed_bank"])
    assert bank["source_selection_sha256"] == spec["source_selection_sha256"]
    assert bank["independent_of_policy_scores"] is True
    assert sorted(map(int, bank["levels"])) == list(range(31))
    poses = np.asarray([r["episode_init"]["object_pose"] + r["episode_init"]["goal_pose"] for r in rows])
    seeds = {(r["obstacle_level"], r["episode_init"].get("reset_seed")) for r in rows}
    for level, values in bank["levels"].items():
        assert len(values["selected"]) == 50
        assert len({r["seed"] for r in values["selected"]}) == 50
        for row in values["selected"]:
            init = row["initial_states"]["triangle"]
            assert row["known_source_overlap"] is False and (int(level), row["seed"]) not in seeds
            delta = poses - np.asarray(init["object_pose"] + init["goal_pose"])
            delta[:, [2, 5]] = (delta[:, [2, 5]] + np.pi) % (2*np.pi) - np.pi
            assert not np.any(np.max(np.abs(delta), axis=1) <= bank["contract"]["initial_pose_tolerance"])
    budgets = asset(ready["budgets"])
    assert budgets["source_selection_sha256"] == spec["source_selection_sha256"]
    assert sorted(map(int, budgets["by_level"])) == list(range(31))
    assert all(b["budget"] > 0 and b["source_train_episodes"] > 0 for b in budgets["by_level"].values())
    output.mkdir(parents=True, exist_ok=True)
    (output / "data-ready.json").write_text(json.dumps(ready))
    print("DATA_GATE_PASSED", regime, len(rows), ready["manifest"]["sha256"], flush=True)


if __name__ == "__main__":
    main(json.loads(Path(sys.argv[1]).read_text()), sys.argv[2], Path(sys.argv[3]))
