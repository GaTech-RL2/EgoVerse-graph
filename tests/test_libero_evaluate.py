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


@pytest.mark.parametrize("invalid", [None, "hash", "checkpoint", "duplicate", "seed"])
@pytest.mark.parametrize("source_kind", ["standalone", "inline", "wrong_inline_budget"])
def test_restore_evaluation_keeps_only_verified_same_policy_records(
    tmp_path, checkpoint, invalid, source_kind
):
    import hashlib
    from dataclasses import asdict

    from botocore.exceptions import ClientError

    from egomimic.benchmarks.libero.catalog import LIBERO_COMMIT, OAT_COMMIT
    from egomimic.benchmarks.libero.evaluate import restore_evaluation
    from egomimic.benchmarks.libero.rollout import rollout_plan

    _, _, request = checkpoint
    if source_kind != "standalone":
        request["checkpoint"]["uri"] = request["checkpoint"]["uri"].replace(
            "/oat-source/", "/previous-evaluation/"
        )
        request["source_run"] = "previous-evaluation"
    plan = [asdict(s) for s in rollout_plan("libero_spatial", repetition_index=0)]
    protocol = {
        "suite": "libero_spatial",
        "method": "oat",
        "evaluation_repetition": 0,
        "plan": plan,
        "checkpoint_sha256": request["checkpoint"]["sha256"],
        "max_episode_steps": 550,
        "horizon": 32,
        "n_obs_steps": 2,
        "n_action_steps": 16,
        "oat_commit": OAT_COMMIT,
        "libero_commit": LIBERO_COMMIT,
        "use_ema": True,
    }
    records = [
        {
            **s,
            "success": True,
            "steps": 3,
            "inference_seconds": [0.1],
            "initial_state_sha256": "same-state",
        }
        for s in plan[:2]
    ]
    records[0]["video"] = "rollout_000000.mp4"
    if invalid == "checkpoint":
        protocol["checkpoint_sha256"] = "b" * 64
    elif invalid == "duplicate":
        records.append(records[0])
    elif invalid == "seed":
        records[0]["seed"] += 1
    prefix = "experiments/arc-oat-20260919/previous-evaluation/"
    relative = "repetitions/0/oat/libero_spatial/"
    artifacts = {
        prefix + "evaluation-request.json": json.dumps(request).encode(),
        prefix + relative + "protocol.json": json.dumps(protocol).encode(),
        prefix + relative + "episodes.jsonl": b"".join(
            (json.dumps(r) + "\n").encode() for r in records
        ),
        prefix + relative + "rollout_000000.mp4": b"saved-video",
    }
    if source_kind != "standalone":
        del artifacts[prefix + "evaluation-request.json"]
        artifacts.update(
            {
                prefix + "runtime.json": json.dumps(
                    {
                        "source_commit": request["source_commit"],
                        "suite": request["suite"],
                        "epochs": 5001,
                        "global_batch_size": 1024,
                        "mode": "full",
                    }
                ).encode(),
                prefix + "training/oat/training-budget.json": json.dumps(
                    {
                        "epochs": 5001,
                        "global_batch_size": (
                            512 if source_kind == "wrong_inline_budget" else 1024
                        ),
                        "total_optimizer_steps": request["total_optimizer_steps"],
                    }
                ).encode(),
                prefix + "checkpoint-receipts.json": json.dumps(
                    {"training/oat/checkpoints/last.ckpt": request["checkpoint"]}
                ).encode(),
            }
        )

    class Storage:
        def get_object(self, *, Bucket, Key):
            if Key not in artifacts:
                raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
            body = artifacts[Key]
            sha = "bad" if invalid == "hash" else hashlib.sha256(body).hexdigest()
            return {"Body": io.BytesIO(body), "Metadata": {"sha256": sha}}

    evidence = tmp_path / "recovery"
    if invalid or source_kind == "wrong_inline_budget":
        with pytest.raises(ValueError):
            restore_evaluation(Storage(), "previous-evaluation", request, evidence)
    else:
        proof = restore_evaluation(Storage(), "previous-evaluation", request, evidence)
        assert proof["episodes"] == 2 and proof["per_repetition"] == [2, 0, 0, 0, 0]
        assert (evidence / relative / "episodes.jsonl").read_bytes() == artifacts[
            prefix + relative + "episodes.jsonl"
        ]
        assert (
            evidence / relative / "rollout_000000.mp4"
        ).read_bytes() == b"saved-video"
        spec = evaluation_workflow(
            "b" * 40,
            "resumed-evaluation",
            request,
            workers=5,
            resume_evaluation_from="previous-evaluation",
        )
        assert (
            spec["workflow"]["tasks"][0]["environment"]["RESUME_EVALUATION_FROM"]
            == "previous-evaluation"
        )


def test_parallel_repetitions_keep_every_original_seed_once():
    from egomimic.benchmarks.libero.rollout import rollout_plan

    complete = rollout_plan("libero_spatial")
    shards = [rollout_plan("libero_spatial", repetition_index=i) for i in range(5)]
    assert [entry for shard in shards for entry in shard] == complete
    assert all(len(shard) == 500 for shard in shards)
    with pytest.raises(ValueError, match="outside"):
        rollout_plan("libero_spatial", repetition_index=5)


@pytest.mark.parametrize(
    "invalid", [None, "missing", "duplicate", "checkpoint", "repetition", "seed"]
)
def test_parallel_merge_requires_complete_identical_policy_protocol(tmp_path, invalid):
    from dataclasses import asdict

    from egomimic.benchmarks.libero.catalog import LIBERO_COMMIT, OAT_COMMIT
    from egomimic.benchmarks.libero.evaluate import merge_repetitions
    from egomimic.benchmarks.libero.report import (
        read_run,
        summarize,
        validate_full_protocol,
    )
    from egomimic.benchmarks.libero.rollout import rollout_plan

    directories = []
    for index in range(5):
        directory = tmp_path / str(index)
        directory.mkdir()
        directories.append(directory)
        plan = [
            asdict(s) for s in rollout_plan("libero_spatial", repetition_index=index)
        ]
        protocol = {
            "plan": plan,
            "suite": "libero_spatial",
            "horizon": 32,
            "n_obs_steps": 2,
            "n_action_steps": 16,
            "max_episode_steps": 550,
            "oat_commit": OAT_COMMIT,
            "libero_commit": LIBERO_COMMIT,
            "use_ema": True,
            "checkpoint_sha256": "a" * 64,
            "evaluation_repetition": index,
        }
        if index == 2 and invalid == "checkpoint":
            protocol["checkpoint_sha256"] = "b" * 64
        if index == 2 and invalid == "repetition":
            protocol["evaluation_repetition"] = 1
        if index == 2 and invalid == "seed":
            plan[0]["seed"] += 1
        records = [
            {
                **s,
                "success": True,
                "steps": 1,
                "initial_state_sha256": str(s["seed"]),
                "inference_seconds": [0.01],
            }
            for s in plan
        ]
        if index == 2 and invalid == "missing":
            records.pop()
        if index == 2 and invalid == "duplicate":
            records.append(records[0])
        if index == 0:
            (directory / "rollout_000000.mp4").write_bytes(b"video")
            records[0]["video"] = "rollout_000000.mp4"
        (directory / "protocol.json").write_text(json.dumps(protocol))
        (directory / "episodes.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in records)
        )
    output = tmp_path / "combined"
    if invalid:
        with pytest.raises(ValueError):
            merge_repetitions(directories, output)
        assert not output.exists()
    else:
        merge_repetitions(directories, output)
        protocol, records = read_run(output)
        validate_full_protocol(protocol)
        result = summarize(records)
        assert result["episodes"] == 2500 and result["mean_success_rate"] == 1
        assert len(result["per_task_success"]) == 10
        assert len(result["per_repetition_success"]) == 5
        assert (output / "repetition_0_rollout_000000.mp4").read_bytes() == b"video"


def test_parallel_workflow_reserves_cpus_and_still_uses_one_gpu(checkpoint):
    _, _, request = checkpoint
    result = evaluation_workflow("b" * 40, "parallel-evaluation", request, workers=5)[
        "workflow"
    ]
    assert result["resources"]["default"]["gpu"] == 1
    assert result["resources"]["default"]["cpu"] == 15
    assert result["tasks"][0]["environment"]["EVALUATION_WORKERS"] == "5"
    with pytest.raises(ValueError):
        evaluation_workflow("b" * 40, "invalid-workers", request, workers=8)
