import hashlib
import io
import json
import subprocess

import pytest
from botocore.exceptions import ClientError

from egomimic.benchmarks.libero.arc_streams import candidates, source_receipt
from egomimic.benchmarks.libero.evaluate import validate_request
from scripts.benchmarks.launch_libero_stream_expansion import expanded_workflow
from scripts.benchmarks.repair_libero_stream_tasks import (
    repair_workflow,
    require_empty_failed_prefixes,
)
from scripts.benchmarks.wait_libero_stream_checkpoint import checkpoint_ready

COMMIT = "a" * 40


def original():
    references = {
        m: {
            "training_run_id": f"existing-reference-{m}",
            "evaluation_run_id": f"existing-reference-{m}-eval",
        }
        for m in ("stk", "dur")
    }
    proof = {
        "source_commit": COMMIT,
        "suite": "libero_spatial",
        "sources": source_receipt(),
        "candidates": candidates(),
        "controls_passed": True,
        "episodes": 30,
    }
    spec = expanded_workflow(
        COMMIT,
        "arc-sx-20261003-spatial",
        "libero_spatial",
        replay_proof=proof,
        references=references,
    )
    failed = {"train-05", "train-11", "train-17"} | {
        f"evaluate-{i:02d}" for i in range(5, 21)
    }
    query = {
        "name": "arc-sx-20261003-spatial-1",
        "groups": [
            {
                "tasks": [
                    {
                        "name": t["name"],
                        "status": "FAILED"
                        if t["name"] == "train-05"
                        else "FAILED_UPSTREAM"
                        if t["name"] in failed
                        else "RUNNING",
                    }
                    for t in spec["workflow"]["tasks"]
                ]
            }
        ],
    }
    return query, spec, failed


class EmptyStorage:
    def list_objects_v2(self, **kwargs):
        return {"KeyCount": 0}


def test_repair_keeps_healthy_training_and_reconnects_all_failed_evaluations():
    query, old, failed = original()
    assert set(require_empty_failed_prefixes(EmptyStorage(), query, old)) == failed
    new = repair_workflow(query, old, "repair-spatial")["workflow"]
    tasks = {t["name"]: t for t in new["tasks"]}
    originals = {t["name"]: t for t in old["workflow"]["tasks"]}
    assert len(tasks) == 34
    assert {n for n in tasks if n.startswith("train-")} == {
        "train-05",
        "train-11",
        "train-17",
    }
    assert len([n for n in tasks if n.startswith("checkpoint-")]) == 13
    assert len([n for n in tasks if n.startswith("evaluate-")]) == 16
    assert new["resources"]["default"]["gpu"] == 4
    assert new["resources"]["reference_wait"]["gpu"] == 0
    for name in failed:
        assert tasks[name]["environment"] == originals[name]["environment"]
        entry = next(
            f["contents"] for f in tasks[name]["files"] if f["path"] == "/tmp/entry.sh"
        )
        subprocess.run(["bash", "-n"], input=entry, text=True, check=True)
        assert "for download_attempt in 1 2 3" in entry
    assert tasks["train-05"]["inputs"] == []
    assert tasks["train-11"]["inputs"] == [{"task": "train-05"}]
    assert tasks["train-17"]["inputs"] == [{"task": "train-11"}]
    for i in range(5, 21):
        name = f"evaluate-{i:02d}"
        expected = f"train-{i:02d}" if i in (5, 11, 17) else f"checkpoint-{i:02d}"
        assert tasks[name]["inputs"][0] == {"task": expected}
        assert sum(
            dep["task"].startswith("evaluate-") for dep in tasks[name]["inputs"]
        ) == (i >= 9)
    assert all(
        dep["task"] in tasks for t in tasks.values() for dep in t.get("inputs", [])
    )


@pytest.mark.parametrize(
    "problem", ["checkpoint", "wrong_workflow", "reference_failure"]
)
def test_repair_refuses_existing_artifacts_or_unrelated_tasks(problem):
    query, spec, _ = original()
    client = EmptyStorage()
    if problem == "checkpoint":
        client.list_objects_v2 = lambda **kw: {"KeyCount": 1}
    elif problem == "wrong_workflow":
        query["name"] = "other-agent-1"
    else:
        next(
            t for t in query["groups"][0]["tasks"] if t["name"] == "reference-stk-ready"
        )["status"] = "FAILED"
    with pytest.raises((ValueError, FileExistsError)):
        require_empty_failed_prefixes(client, query, spec)


def policy():
    identity = {
        "source_run": "existing-gripper-stk",
        "source_commit": COMMIT,
        "suite": "libero_spatial",
        "method": "arc_stk",
        "arc_stream_variant": "gripper",
        "total_optimizer_steps": 270054,
        "stream_reference_run": "existing-reference-stk-eval",
    }
    sha = "b" * 64
    objects = {
        "status.json": {
            "state": "TRAINING_COMPLETE",
            "epochs_completed": 5001,
            "global_step": 270054,
            "ema_num_updates": 270054,
            "training_budget": {
                "epochs": 5001,
                "total_optimizer_steps": 270054,
                "global_batch_size": 1024,
                "world_size": 4,
            },
        },
        "runtime.json": {
            "run_kind": "arc_streams",
            "source_commit": COMMIT,
            "suite": "libero_spatial",
            "mode": "full",
            "operation": "train",
            "arc_stream_variant": "gripper",
            "arc_mode": "stk",
        },
        "checkpoint-receipts.json": {
            "training/arc_stk/checkpoints/last.ckpt": {
                "sha256": sha,
                "bytes": 100,
                "uri": f"s3://rldb/experiments/arc-oat-20260919/existing-gripper-stk/checkpoints/{sha}/last.ckpt",
            }
        },
    }

    class Storage:
        corrupt = False

        def get_object(self, *, Key, **kwargs):
            name = Key.rsplit("/", 1)[-1]
            if name not in objects:
                raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
            raw = json.dumps(objects[name]).encode()
            return {
                "Body": io.BytesIO(raw),
                "Metadata": {
                    "sha256": "bad" if self.corrupt else hashlib.sha256(raw).hexdigest()
                },
            }

    return identity, objects, Storage()


def test_checkpoint_waiter_delays_until_complete_and_relays_native_request():
    identity, objects, client = policy()
    request, hashes = checkpoint_ready(client, identity)
    validate_request(request)
    assert request["stream_reference_run"] == identity["stream_reference_run"]
    assert set(hashes) == set(objects)
    objects["status.json"]["state"] = "TRAINING"
    assert checkpoint_ready(client, identity) is None
    del objects["status.json"]
    assert checkpoint_ready(client, identity) is None


@pytest.mark.parametrize(
    "problem",
    ["partial", "ema", "layout", "identity", "checkpoint", "checksum", "failed"],
)
def test_checkpoint_waiter_rejects_wrong_policy_or_incomplete_budget(problem):
    identity, objects, client = policy()
    if problem == "partial":
        objects["status.json"]["global_step"] -= 1
    elif problem == "ema":
        objects["status.json"]["ema_num_updates"] -= 1
    elif problem == "layout":
        objects["status.json"]["training_budget"]["world_size"] = 8
    elif problem == "identity":
        objects["runtime.json"]["arc_stream_variant"] = "reference"
    elif problem == "checkpoint":
        objects["checkpoint-receipts.json"]["training/arc_stk/checkpoints/last.ckpt"][
            "uri"
        ] = "s3://rldb/other/last.ckpt"
    elif problem == "checksum":
        client.corrupt = True
    else:
        objects["status.json"]["state"] = "FAILED"
    with pytest.raises((ValueError, RuntimeError)):
        checkpoint_ready(client, identity)
