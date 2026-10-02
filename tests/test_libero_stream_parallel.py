import hashlib
import io
import json

import pytest
from botocore.exceptions import ClientError

from egomimic.benchmarks.libero.arc_streams import campaign, candidates, source_receipt
from scripts.benchmarks.launch_libero_stream_parallel import (
    mark_pending_tail_moved,
    parallel_workflow,
    pending_cells,
)
from scripts.benchmarks.launch_libero_streams import stream_workflow
from scripts.benchmarks.wait_libero_stream_reference import reference_ready

COMMIT = "a" * 40


def replay_proof(suite):
    return {
        "source_commit": COMMIT,
        "suite": suite,
        "sources": source_receipt(),
        "candidates": candidates(),
        "controls_passed": True,
        "episodes": 30,
    }


@pytest.mark.parametrize("suite", campaign()["suites"])
def test_parallel_lanes_preserve_all_cells_and_never_wait_for_evaluation(suite):
    cells = pending_cells()
    assert len(cells) == len(set(cells)) == 21
    assert set(cells) | {("reference", "stk")} == {
        (v, m) for v in campaign()["variants"] for m in campaign()["modes"]
    }
    proof = replay_proof(suite)
    reference = "existing-reference-stk-eval"
    spec = parallel_workflow(
        COMMIT, "parallel-test", suite, replay_proof=proof, reference_run=reference
    )["workflow"]
    tasks = {t["name"]: t for t in spec["tasks"]}
    assert len(tasks) == 43
    assert spec["resources"]["default"]["gpu"] == 4
    assert spec["resources"]["reference_wait"]["gpu"] == 0
    for i, (variant, mode) in enumerate(cells):
        train, evaluate = tasks[f"train-{i:02d}"], tasks[f"evaluate-{i:02d}"]
        assert train.get("inputs", []) == (
            [] if i < 3 else [{"task": f"train-{i - 3:02d}"}]
        )
        assert train["environment"]["SOURCE_COMMIT"] == COMMIT
        assert train["environment"]["ARC_STREAM_VARIANT"] == variant
        assert train["environment"]["ARC_STREAM_MODE"] == mode
        assert train["environment"]["RUN_MODE"] == "full"
        receipt_file = next(
            f for f in train["files"] if f["path"] == "/tmp/replay-completion.json"
        )
        assert json.loads(receipt_file["contents"]) == proof
        assert evaluate["inputs"][0] == {"task": train["name"]}
        if i:
            assert {"task": f"evaluate-{i - 1:02d}"} in evaluate["inputs"]
        assert ({"task": "reference-stk-ready"} in evaluate["inputs"]) == (
            mode == "stk"
        )
    # Waiting on reference scores never holds GPUs or blocks a training lane.
    assert tasks["reference-stk-ready"].get("inputs", []) == []
    assert cells[0] == ("reference", "dur")


@pytest.mark.parametrize(
    "field,value",
    [("source_commit", "b" * 40), ("episodes", 29), ("controls_passed", False)],
)
def test_parallel_launch_rejects_incompatible_replay(field, value):
    proof = replay_proof("libero_spatial")
    proof[field] = value
    with pytest.raises(ValueError):
        parallel_workflow(
            COMMIT,
            "parallel-test",
            "libero_spatial",
            replay_proof=proof,
            reference_run="existing-reference-stk-eval",
        )


class Storage:
    def __init__(self, objects=None):
        self.objects = objects or {}
        self.puts = []

    def get_object(self, *, Bucket, Key):
        name = Key.rsplit("/", 1)[-1]
        if name not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        body = json.dumps(self.objects[name]).encode()
        return {
            "Body": io.BytesIO(body),
            "Metadata": {"sha256": hashlib.sha256(body).hexdigest()},
        }

    def list_objects_v2(self, **kwargs):
        return {"KeyCount": len(self.objects)}

    def put_object(self, **kwargs):
        self.puts.append(kwargs)


def test_reference_wait_requires_completed_full_evaluation_and_matching_source():
    client = Storage()
    args = (client, "old-reference-stk-eval", "libero_spatial", COMMIT)
    assert reference_ready(*args) is None
    client.objects["status.json"] = {"state": "ROLLOUTS"}
    assert reference_ready(*args) is None
    client.objects["status.json"] = {
        "state": "POLICY_EVALUATION_COMPLETE",
        "episodes": 2500,
        "complete_protocol": True,
    }
    client.objects["runtime.json"] = {
        "source_commit": COMMIT,
        "suite": "libero_spatial",
        "method": "arc_stk",
        "run_kind": "policy_evaluation",
        "mode": "full",
        "evaluate_from_run": "old-reference-stk",
    }
    assert reference_ready(*args)["reference_run"] == args[1]
    client.objects["runtime.json"]["source_commit"] = "b" * 40
    with pytest.raises(ValueError, match="source"):
        reference_ready(*args)
    client.objects["status.json"] = {"state": "FAILED"}
    with pytest.raises(RuntimeError, match="failed"):
        reference_ready(*args)


def old_query():
    tasks = [{"name": "replay", "status": "COMPLETED"}]
    tasks += [
        {
            "name": f"{kind}-{i:02d}",
            "status": "RUNNING" if kind == "train" and i == 0 else "WAITING",
        }
        for i in range(22)
        for kind in ("train", "evaluate")
    ]
    return {"name": "arc-str-20261002r2-spatial-1", "groups": [{"tasks": tasks}]}


def test_handoff_reserves_only_unstarted_tail_without_touching_current_reference(
    monkeypatch, tmp_path
):
    from egomimic.benchmarks.libero.cluster import ArtifactUploader

    client = Storage()
    receipt = mark_pending_tail_moved(client, old_query(), "parallel-test-1", COMMIT)
    assert receipt["preserved_tasks"] == ["replay", "train-00", "evaluate-00"]
    (put,) = client.puts
    assert put["IfNoneMatch"] == "*"
    assert put["Key"].endswith("-reference-dur/orchestration-handoff.json")
    assert hashlib.sha256(put["Body"]).hexdigest() == put["Metadata"]["sha256"]
    client.objects["orchestration-handoff.json"] = receipt
    monkeypatch.setattr("boto3.client", lambda *args, **kwargs: client)
    for key in ["R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"]:
        monkeypatch.setenv(key, "test")
    with pytest.raises(FileExistsError, match="existing run prefix"):
        ArtifactUploader(tmp_path, "arc-str-20261002r2-spatial-reference-dur")
    # The retired tail is not an ancestor of either preserved reference task.
    spec = stream_workflow(COMMIT, "old-stream", "libero_spatial")["workflow"]
    descendants = {"train-01"}
    for task in spec["tasks"]:
        if any(i["task"] in descendants for i in task.get("inputs", [])):
            descendants.add(task["name"])
    assert "train-00" not in descendants and "evaluate-00" not in descendants
    assert all(f"train-{i:02d}" in descendants for i in range(1, 22))


@pytest.mark.parametrize("problem", ["started", "artifacts", "other_workflow"])
def test_handoff_refuses_to_replace_started_or_unrelated_work(problem):
    client, query = Storage(), old_query()
    if problem == "started":
        next(t for t in query["groups"][0]["tasks"] if t["name"] == "train-01")[
            "status"
        ] = "RUNNING"
    elif problem == "artifacts":
        client.objects["checkpoint"] = "exists"
    else:
        query["name"] = "someone-elses-workflow"
    with pytest.raises((ValueError, FileExistsError)):
        mark_pending_tail_moved(client, query, "parallel-test-1", COMMIT)
    assert not client.puts
