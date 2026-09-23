"""Completed-checkpoint isolation and exact full-protocol evaluation routing."""

import copy
import io
import json
import shutil

import pytest
import torch

from egomimic.benchmarks.libero.cluster import digest
from egomimic.benchmarks.libero.evaluate import (
    checkpoint_completion,
    restore_policy,
    validate_request,
)
from scripts.benchmarks.launch_libero_osmo import evaluation_workflow


@pytest.fixture
def checkpoint(tmp_path):
    payload = {
        "global_step": 270054,
        "ema_num_updates": 270054,
        "ema_state_dict": {"weight": torch.ones(1)},
        "normalizer_state": {},
        "oat_tokenizer_config": {"_target_": "native.OAT"},
        "benchmark_data_context": {"suite": "libero_spatial"},
        "hyper_parameters": {
            "config_tree": {
                "model": {"benchmark_protocol": {"suite": "libero_spatial"}},
            }
        },
        "loops": {"fit_loop": {"epoch_progress": {"current": {"completed": 5001}}}},
        "training_budget": {
            "epochs": 5001,
            "global_batch_size": 1024,
            "total_optimizer_steps": 270054,
        },
    }
    path = tmp_path / "source.ckpt"
    torch.save(payload, path)
    sha = digest(path)
    request = {
        "source_run": "oat-source",
        "source_commit": "a" * 40,
        "method": "oat",
        "suite": "libero_spatial",
        "epochs": 5001,
        "total_optimizer_steps": 270054,
        "checkpoint": {
            "sha256": sha,
            "bytes": path.stat().st_size,
            "uri": f"s3://rldb/experiments/arc-oat-20260919/oat-source/checkpoints/{sha}/last.ckpt",
        },
    }
    return payload, path, request


@pytest.mark.parametrize("partial", ["epochs", "optimizer"])
def test_completion_waits_for_both_final_epoch_and_optimizer(checkpoint, partial):
    payload, _, request = checkpoint
    if partial == "epochs":
        payload["loops"]["fit_loop"]["epoch_progress"]["current"]["completed"] = 5000
    else:
        payload["global_step"] -= 1
    assert checkpoint_completion(payload, request) is None


@pytest.mark.parametrize(
    "invalid",
    ["ema", "weights", "budget", "suite", "tokenizer", "normalizer", "overrun"],
)
def test_completion_rejects_incompatible_checkpoint(checkpoint, invalid):
    payload, _, request = checkpoint
    if invalid == "ema":
        payload["ema_num_updates"] -= 1
    elif invalid == "weights":
        del payload["ema_state_dict"]
    elif invalid == "budget":
        payload["training_budget"]["global_batch_size"] = 256
    elif invalid == "suite":
        payload["benchmark_data_context"]["suite"] = "libero_object"
    elif invalid == "tokenizer":
        del payload["oat_tokenizer_config"]
    elif invalid == "normalizer":
        del payload["normalizer_state"]
    else:
        payload["global_step"] += 1
    with pytest.raises(ValueError):
        checkpoint_completion(payload, request)


@pytest.mark.parametrize("invalid", [None, "hash", "runtime", "partial"])
def test_restores_only_pinned_complete_policy_without_training_data(
    tmp_path, checkpoint, invalid
):
    payload, source, request = checkpoint
    runtime = {
        "source_commit": "a" * 40,
        "suite": "libero_spatial",
        "epochs": 5001,
        "global_batch_size": 1024,
        "mode": "full",
    }
    if invalid == "hash":
        source.write_bytes(b"corrupt checkpoint")
    elif invalid == "runtime":
        runtime["source_commit"] = "b" * 40
    elif invalid == "partial":
        payload["global_step"] -= 1
        torch.save(payload, source)
        request["checkpoint"].update(sha256=digest(source), bytes=source.stat().st_size)
        request["checkpoint"]["uri"] = request["checkpoint"]["uri"].replace(
            request["checkpoint"]["uri"].split("/")[-2], digest(source)
        )

    class Storage:
        def get_object(self, *, Bucket, Key):
            assert Key == "experiments/arc-oat-20260919/oat-source/runtime.json"
            return {"Body": io.BytesIO(json.dumps(runtime).encode())}

        def download_file(self, bucket, key, destination):
            assert key == request["checkpoint"]["uri"].removeprefix("s3://rldb/")
            shutil.copyfile(source, destination)

    root, evidence = tmp_path / "worker", tmp_path / "worker/evidence"
    if invalid:
        with pytest.raises(ValueError):
            restore_policy(Storage(), request, root, evidence)
    else:
        path = restore_policy(Storage(), request, root, evidence)
        assert digest(path) == request["checkpoint"]["sha256"]
        proof = json.loads((evidence / "recovered-training.json").read_text())
        assert proof["global_step"] == proof["ema_num_updates"] == 270054
        assert not (root / "data").exists()
        assert not list(evidence.rglob("*.ckpt"))


def test_request_cannot_point_to_other_run_or_unpinned_checkpoint(checkpoint):
    _, _, request = checkpoint
    for uri in (
        "s3://rldb/another/checkpoint.ckpt",
        request["checkpoint"]["uri"].replace("oat-source", "other-run"),
    ):
        invalid = copy.deepcopy(request)
        invalid["checkpoint"]["uri"] = uri
        with pytest.raises(ValueError):
            validate_request(invalid)


def test_evaluation_routes_directly_to_native_rollouts_on_one_gpu(checkpoint):
    _, _, request = checkpoint
    result = evaluation_workflow("b" * 40, "oat-evaluation", request)["workflow"]
    assert result["resources"]["default"]["gpu"] == 1
    task = result["tasks"][0]
    assert task["environment"]["RUN_KIND"] == "policy_evaluation"
    assert task["environment"]["SOURCE_COMMIT"] == "b" * 40
    files = {item["path"]: item["contents"] for item in task["files"]}
    assert json.loads(files["/tmp/evaluation-request.json"]) == request
    entry = files["/tmp/entry.sh"]
    branch = entry.split("== policy_evaluation ]]; then")[1].split("elif")[0]
    assert "egomimic.benchmarks.libero.evaluate" in branch
    assert "trainHydra" not in branch and "stage_dataset" not in branch
