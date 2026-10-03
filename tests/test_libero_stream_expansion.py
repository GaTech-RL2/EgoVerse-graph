import copy
import hashlib
import io
import json

import pytest

from egomimic.benchmarks.libero.arc_streams import campaign, candidates, source_receipt
from scripts.benchmarks.launch_libero_stream_expansion import (
    expanded_workflow,
    mark_recovery_tail_moved,
)
from scripts.benchmarks.launch_libero_stream_parallel import (
    parallel_workflow,
    pending_cells,
)
from scripts.benchmarks.wait_libero_stream_reference import reference_ready

COMMIT = "a" * 40


def fixtures(suite="libero_spatial"):
    suffix = suite.removeprefix("libero_")
    old_name, new_name = f"arc-sr-20261003-{suffix}", f"arc-sx-20261003-{suffix}"
    proof = {
        "source_commit": COMMIT,
        "suite": suite,
        "sources": source_receipt(),
        "candidates": candidates(),
        "controls_passed": True,
        "episodes": 30,
    }
    refs = {
        mode: {
            "training_run_id": f"{old_name}-reference-{mode}",
            "evaluation_run_id": f"{old_name}-reference-{mode}-eval",
        }
        for mode in ("stk", "dur")
    }
    if suffix in ("spatial", "goal"):
        refs["stk"] = {
            "training_run_id": f"arc-str-20261002r2-{suffix}-reference-stk",
            "evaluation_run_id": f"arc-se-20261003-{suffix}-reference-stk-eval",
        }
    old = parallel_workflow(
        COMMIT,
        old_name,
        suite,
        replay_proof=proof,
        reference_run=refs["stk"]["evaluation_run_id"],
    )
    if suffix in ("object", "10"):
        training = copy.deepcopy(old["workflow"]["tasks"][1])
        training["name"] = "reference-train"
        old["workflow"]["tasks"][0]["inputs"] = [{"task": "reference-train"}]
        old["workflow"]["tasks"].insert(0, training)
    new = expanded_workflow(
        COMMIT, new_name, suite, replay_proof=proof, references=refs
    )

    def query(spec, old=False):
        return {
            "name": spec["workflow"]["name"] + "-1",
            "status": "RUNNING",
            "groups": [
                {
                    "tasks": [
                        {
                            "name": t["name"],
                            "status": "RUNNING"
                            if old
                            and t["name"]
                            in {"train-00", "train-01", "train-02", "reference-train"}
                            else "WAITING",
                        }
                        for t in spec["workflow"]["tasks"]
                    ]
                }
            ],
        }

    return old, new, query(old, True), query(new), refs


class Storage:
    def __init__(self):
        self.objects, self.puts = {}, []

    def list_objects_v2(self, *, Prefix, **kwargs):
        return {"KeyCount": sum(k.startswith(Prefix) for k in self.objects)}

    def put_object(self, **kwargs):
        assert kwargs["IfNoneMatch"] == "*" and kwargs["Key"] not in self.objects
        self.puts.append(kwargs)
        self.objects[kwargs["Key"]] = kwargs["Body"]


@pytest.mark.parametrize("suite", campaign()["suites"])
def test_six_additional_lanes_cover_only_unstarted_cells_and_preserve_recoveries(
    suite, monkeypatch, tmp_path
):
    from egomimic.benchmarks.libero.cluster import ArtifactUploader

    old, new, old_query, new_query, refs = fixtures(suite)
    body = new["workflow"]
    tasks = {t["name"]: t for t in body["tasks"]}
    trains = [t for t in tasks.values() if t["name"].startswith("train-")]
    assert len(trains) == 18
    assert len([t for t in trains if not t.get("inputs")]) == 6
    assert body["resources"]["default"]["gpu"] == 4
    assert body["resources"]["reference_wait"]["gpu"] == 0
    assert body["resources"]["evaluation"]["gpu"] == 1
    assert {
        (t["environment"]["ARC_STREAM_VARIANT"], t["environment"]["ARC_STREAM_MODE"])
        for t in trains
    } == set(pending_cells()[3:])
    for i, (_, mode) in enumerate(pending_cells()[3:], start=3):
        train, evaluate = tasks[f"train-{i:02d}"], tasks[f"evaluate-{i:02d}"]
        assert train.get("inputs", []) == (
            [] if i < 9 else [{"task": f"train-{i - 6:02d}"}]
        )
        assert train["environment"]["SOURCE_COMMIT"] == COMMIT
        assert (
            train["environment"]["ARC_STREAM_REFERENCE_RUN"]
            == refs[mode]["evaluation_run_id"]
        )
        assert evaluate["inputs"][0] == {"task": train["name"]}
        assert {"task": f"reference-{mode}-ready"} in evaluate["inputs"]
        assert all(dep["task"] in tasks for dep in evaluate["inputs"])
        assert not any("resume" in f["path"] for f in train["files"])
    for mode in ("stk", "dur"):
        waiter = tasks[f"reference-{mode}-ready"]
        assert not waiter.get("inputs")
        assert f"--mode {mode}" in waiter["args"][1]
        assert f"--training-run {refs[mode]['training_run_id']}" in waiter["args"][1]
        assert waiter["image"] == "docker.io/library/python:3.11-slim"
    storage = Storage()
    receipt = mark_recovery_tail_moved(storage, old_query, old, new_query, new)
    assert len(storage.puts) == len(receipt["markers"]) == 3
    assert len(receipt["retired_tasks"]) == 36
    assert {f"{k}-{i:02d}" for i in range(3) for k in ("train", "evaluate")} <= set(
        receipt["preserved_tasks"]
    )
    monkeypatch.setattr("boto3.client", lambda *a, **kw: storage)
    for key in ("R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(key, "test")
    old_tasks = {t["name"]: t for t in old["workflow"]["tasks"]}
    for name, put in zip(receipt["retired_lane_heads"], storage.puts):
        assert hashlib.sha256(put["Body"]).hexdigest() == put["Metadata"]["sha256"]
        assert put["Key"].endswith("orchestration-handoff.json")
        with pytest.raises(FileExistsError, match="existing run prefix"):
            ArtifactUploader(tmp_path, old_tasks[name]["environment"]["RUN_ID"])


@pytest.mark.parametrize(
    "problem",
    [
        "started",
        "artifacts",
        "unaccepted",
        "other_workflow",
        "missing_cell",
        "recipe",
        "prefix",
        "preserved_descendant",
    ],
)
def test_expansion_handoff_refuses_unsafe_or_incomplete_replacement(problem):
    old, new, oq, nq, _ = fixtures()
    storage = Storage()
    ot = {t["name"]: t for t in old["workflow"]["tasks"]}
    nt = {t["name"]: t for t in new["workflow"]["tasks"]}
    if problem == "started":
        next(t for t in oq["groups"][0]["tasks"] if t["name"] == "train-11")[
            "status"
        ] = "INITIALIZING"
    elif problem == "artifacts":
        storage.objects[
            f"experiments/arc-oat-20260919/{ot['train-05']['environment']['RUN_ID']}/checkpoint"
        ] = b"existing"
    elif problem == "unaccepted":
        nq["status"] = "FAILED"
    elif problem == "other_workflow":
        oq["name"] = "other-agent-1"
    elif problem == "missing_cell":
        new["workflow"]["tasks"].remove(nt["train-07"])
    elif problem == "recipe":
        nt["train-03"]["environment"]["SOURCE_COMMIT"] = "b" * 40
    elif problem == "prefix":
        nt["train-03"]["environment"]["RUN_ID"] = ot["train-03"]["environment"][
            "RUN_ID"
        ]
    else:
        ot["evaluate-00"]["inputs"].append({"task": "train-03"})
    with pytest.raises((ValueError, FileExistsError)):
        mark_recovery_tail_moved(storage, oq, old, nq, new)
    assert not storage.puts


def test_duration_waiter_rejects_stk_or_different_training_source():
    objects = {
        "status.json": {
            "state": "POLICY_EVALUATION_COMPLETE",
            "episodes": 2500,
            "complete_protocol": True,
        },
        "runtime.json": {
            "source_commit": COMMIT,
            "suite": "libero_spatial",
            "method": "arc_dur",
            "run_kind": "policy_evaluation",
            "mode": "full",
            "evaluate_from_run": "old-reference-dur",
        },
    }

    class Client:
        def get_object(self, *, Key, **kwargs):
            body = json.dumps(objects[Key.rsplit("/", 1)[-1]]).encode()
            return {
                "Body": io.BytesIO(body),
                "Metadata": {"sha256": hashlib.sha256(body).hexdigest()},
            }

    args = (Client(), "new-reference-dur-eval", "libero_spatial", COMMIT)
    assert (
        reference_ready(*args, training_run="old-reference-dur", mode="dur")["arc_mode"]
        == "dur"
    )
    with pytest.raises(ValueError, match="source"):
        reference_ready(*args, mode="dur")
    with pytest.raises(ValueError, match="source"):
        reference_ready(*args, training_run="old-reference-dur", mode="stk")
    objects["status.json"]["episodes"] = 2499
    with pytest.raises(ValueError, match="full protocol"):
        reference_ready(*args, training_run="old-reference-dur", mode="dur")
