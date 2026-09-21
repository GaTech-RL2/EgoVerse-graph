import importlib.util
import json
import zipfile
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from egomimic.benchmarks.libero.cluster import (
    ArtifactUploader,
    CalibrationPending,
    digest,
    extract_replay,
    load_arc_calibration,
    publish_campaign,
    restore_checkpoints,
    training_arguments,
)


@pytest.mark.parametrize("mode", ["smoke", "full"])
@pytest.mark.parametrize("method", ["tokenizer", "oat", "arc"])
def test_cluster_arguments_compose_native_graph(mode, method, tmp_path):
    args = training_arguments(
        method, "libero_10", tmp_path / "replay.zarr", tmp_path, mode, 5001
    )
    config_dir = Path(__file__).parents[1] / "egomimic/hydra_configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="train_zarr_cartesian", overrides=args[3:])
    assert cfg.model.pipeline._target_ == "egomimic.pipeline.algo.PipelineAlgo"
    assert cfg.trainer.devices == 1 and cfg.trainer.precision == "bf16-mixed"
    assert cfg.benchmark.batch_size == (4 if mode == "smoke" else 256)
    assert cfg.trainer.max_epochs == (1 if mode == "smoke" else 5001)
    if mode == "full":
        assert cfg.logger.csv._target_ == "lightning.pytorch.loggers.CSVLogger"
        assert cfg.trainer.accumulate_grad_batches == 4
        assert cfg.callbacks.batch_budget.global_batch_size == 1024
    else:
        assert cfg.trainer.accumulate_grad_batches == 1
        assert cfg.callbacks.batch_budget is None
    assert cfg.callbacks.ema.final_checkpoint_path.endswith("checkpoints/last.ckpt")
    if method == "oat":
        assert cfg.benchmark.tokenizer_checkpoint.endswith(
            "tokenizer/checkpoints/last.ckpt"
        )


def test_workflow_pins_source_and_requests_one_gpu():
    path = Path(__file__).parents[1] / "scripts/benchmarks/launch_libero_osmo.py"
    spec = importlib.util.spec_from_file_location("libero_osmo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    workflow = module.workflow("a" * 40, "libero-smoke-test", "libero_10")["workflow"]
    assert workflow["resources"]["default"]["gpu"] == 1
    assert workflow["tasks"][0]["environment"]["SOURCE_COMMIT"] == "a" * 40
    with pytest.raises(ValueError, match="immutable"):
        module.workflow("main", "libero-smoke-test", "libero_10")
    full = module.workflow(
        "a" * 40, "study-libero-10", "libero_10", mode="full", campaign_id="study"
    )["workflow"]
    assert full["timeout"]["exec_timeout"] == "60d"
    assert full["tasks"][0]["environment"]["CAMPAIGN_ID"] == "study"
    replay = module.workflow(
        "a" * 40,
        "replay-refine",
        "libero_90",
        replay=True,
        replay_spec="libero_arc_replay_refine",
    )["workflow"]
    assert replay["resources"]["default"]["cpu"] >= 32
    assert replay["resources"]["default"]["gpu"] == 1
    with pytest.raises(ValueError, match="Campaign"):
        module.workflow(
            "a" * 40, "unrelated-run", "libero_10", mode="full", campaign_id="study"
        )


@pytest.mark.parametrize(
    "condition", ["complete", "missing", "running", "source", "budget"]
)
def test_campaign_only_publishes_complete_matching_results(condition):
    import io

    from botocore.exceptions import ClientError

    from egomimic.benchmarks.libero.catalog import TASKS

    class Storage:
        report = None

        def get_object(self, Bucket, Key):
            suite = next(
                suite for suite in TASKS if f"study-{suite.replace('_', '-')}/" in Key
            )
            if suite == "libero_90" and condition == "missing":
                raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
            payload = {
                "status.json": {
                    "state": "TRAINING" if condition == "running" else "SUITE_COMPLETE"
                },
                "runtime.json": {
                    "source_commit": "b" * 40 if condition == "source" else "a" * 40,
                    "mode": "full",
                    "epochs": 1 if condition == "budget" else 5001,
                    "global_batch_size": 1024,
                    "suite": suite,
                },
                "comparison.json": {"suite": suite, "complete_protocol": True},
            }[Key.rsplit("/", 1)[1]]
            return {"Body": io.BytesIO(json.dumps(payload).encode())}

        def put_object(self, **kwargs):
            self.report = json.loads(kwargs["Body"])

    storage = Storage()
    if condition in {"source", "budget"}:
        with pytest.raises(ValueError, match="Campaign"):
            publish_campaign(storage, "study", commit="a" * 40, epochs=5001)
    else:
        assert publish_campaign(storage, "study", commit="a" * 40, epochs=5001) == (
            condition == "complete"
        )
    if condition == "complete":
        assert storage.report["unique_tasks"] == 130
        assert set(storage.report["suites"]) == set(TASKS)
        assert storage.report["full_released_training_budget"] is True
    else:
        assert storage.report is None


def test_released_replay_archive_detection_and_traversal_rejection(tmp_path):
    archive = tmp_path / "replay.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("libero10_N500.zarr/.zgroup", "{}")
        handle.writestr("libero10_N500.zarr/meta/episode_ends/.zarray", "{}")
    assert (
        extract_replay(archive, tmp_path / "out") == tmp_path / "out/libero10_N500.zarr"
    )
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../outside", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        extract_replay(archive, tmp_path / "bad")
    assert not (tmp_path / "outside").exists()


def test_checkpoint_snapshots_are_content_addressed_and_receipted(
    tmp_path, monkeypatch
):
    import boto3

    class Storage:
        def __init__(self):
            self.objects = {}

        def list_objects_v2(self, **kwargs):
            return {"KeyCount": 0}

        def upload_file(self, source, bucket, key, ExtraArgs):
            self.objects[key] = Path(source).read_bytes()
            assert ExtraArgs["Metadata"]["sha256"] == digest(source)

    storage = Storage()
    monkeypatch.setattr(boto3, "client", lambda *args, **kwargs: storage)
    for name in ("R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(name, "test")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    path = evidence / "last.ckpt"
    path.write_bytes(b"first checkpoint")
    uploader = ArtifactUploader(evidence, "test")
    uploader.upload(final=True)
    first_hash = digest(path)
    path.write_bytes(b"final checkpoint")
    uploader.upload(final=True)
    uploader.upload(final=True)
    receipts = json.loads((evidence / "checkpoint-receipts.json").read_text())
    assert receipts["last.ckpt"]["sha256"] == digest(path)
    assert any(first_hash in key for key in storage.objects)
    assert any(digest(path) in key for key in storage.objects)


@pytest.mark.parametrize("invalid", [None, "budget", "hash", "incomplete", "outside"])
def test_recovery_verifies_checkpoints_and_complete_training(tmp_path, invalid):
    import io
    import shutil

    import torch

    source = tmp_path / "source.ckpt"
    torch.save(
        {
            "global_step": 2,
            "loops": {
                "fit_loop": {
                    "epoch_progress": {
                        "current": {"completed": 0 if invalid == "incomplete" else 1}
                    }
                }
            },
        },
        source,
    )
    prefix = "experiments/arc-oat-20260919/source-run/"
    runtime = {
        "suite": "libero_10",
        "mode": "full" if invalid == "budget" else "smoke",
        "epochs": 1,
    }
    receipt = {
        "sha256": "bad" if invalid == "hash" else digest(source),
        "bytes": source.stat().st_size,
        "uri": "s3://rldb/"
        + ("another-run/" if invalid == "outside" else prefix + "checkpoints/")
        + "last.ckpt",
    }
    receipts = {
        f"training/{method}/checkpoints/last.ckpt": receipt
        for method in ("tokenizer", "oat", "arc")
    }

    class Storage:
        def get_object(self, Bucket, Key):
            value = runtime if Key.endswith("runtime.json") else receipts
            return {"Body": io.BytesIO(json.dumps(value).encode())}

        def download_file(self, bucket, key, destination):
            shutil.copyfile(source, destination)

    evidence = tmp_path / "evidence"
    if invalid is not None:
        with pytest.raises(ValueError):
            restore_checkpoints(
                Storage(),
                "source-run",
                evidence,
                suite="libero_10",
                mode="smoke",
                epochs=5001,
            )
    else:
        restore_checkpoints(
            Storage(),
            "source-run",
            evidence,
            suite="libero_10",
            mode="smoke",
            epochs=5001,
        )
        assert (
            len(
                json.loads((evidence / "recovered-training.json").read_text())[
                    "checkpoints"
                ]
            )
            == 3
        )


@pytest.mark.parametrize("invalid", [None, "optimizer", "normalizer", "budget"])
def test_partial_resume_preserves_only_existing_verified_checkpoints(tmp_path, invalid):
    import io
    import shutil

    import torch

    payload = {
        "global_step": 140,
        "loops": {"fit_loop": {"epoch_progress": {"current": {"completed": 2}}}},
        "training_budget": {
            "global_batch_size": 256 if invalid == "budget" else 1024,
            "epochs": 5001,
        },
        "optimizer_states": [{"state": {}}],
        "normalizer_state": {},
    }
    if invalid == "optimizer":
        del payload["optimizer_states"]
    if invalid == "normalizer":
        del payload["normalizer_state"]
    source = tmp_path / "partial.ckpt"
    torch.save(payload, source)
    runtime = {
        "suite": "libero_10",
        "mode": "full",
        "epochs": 5001,
        "global_batch_size": 1024,
    }
    receipts = {
        "training/tokenizer/checkpoints/last.ckpt": {
            "sha256": digest(source),
            "bytes": source.stat().st_size,
            "uri": "s3://rldb/experiments/arc-oat-20260919/source-run/checkpoints/partial.ckpt",
        }
    }

    class Storage:
        def get_object(self, Bucket, Key):
            value = runtime if Key.endswith("runtime.json") else receipts
            return {"Body": io.BytesIO(json.dumps(value).encode())}

        def download_file(self, bucket, key, destination):
            shutil.copyfile(source, destination)

    if invalid:
        with pytest.raises(ValueError, match="Partial checkpoint"):
            restore_checkpoints(
                Storage(),
                "source-run",
                tmp_path / "evidence",
                suite="libero_10",
                mode="full",
                epochs=5001,
                allow_partial=True,
            )
    else:
        restored = restore_checkpoints(
            Storage(),
            "source-run",
            tmp_path / "evidence",
            suite="libero_10",
            mode="full",
            epochs=5001,
            allow_partial=True,
        )
        assert set(restored) == {"tokenizer"}
        assert restored["tokenizer"]["global_step"] == 140
        assert restored["tokenizer"]["complete"] is False
        assert not (tmp_path / "evidence/training/oat").exists()


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        "pending",
        "suite",
        "protocol",
        "spec",
        "codec",
        "retention",
        "incomplete",
        "evaluated",
        "unpermitted_gap",
    ],
)
def test_arc_training_requires_matching_confirmed_replay(
    tmp_path, monkeypatch, invalid
):
    import hashlib
    import io
    import subprocess

    import yaml

    from egomimic.benchmarks.libero.replay import candidate_id

    root = Path(__file__).parents[1]
    spec = yaml.safe_load(
        (root / "egomimic/hydra_configs/benchmark/libero_arc_replay.yaml").read_text()
    )
    if invalid in {"evaluated", "unpermitted_gap"}:
        spec.update(
            selection_objective="success_then_tokens",
            maximum_success_rate_drop=1.0,
            allow_reference_gap=invalid == "evaluated",
        )
    codec = {"num_waypoints": 16, "max_translation": 0.2, "max_rotation_degrees": 48}
    selected = candidate_id(codec)
    result = {
        "confirmed": invalid not in {"pending", "evaluated", "unpermitted_gap"},
        "confirmation_complete": True,
        "suite": "libero_10",
        "codec": codec,
        "candidate_id": selected,
        "selected_before_confirmation": True,
        "spec_sha256": hashlib.sha256(
            (json.dumps(spec, indent=2) + "\n").encode()
        ).hexdigest(),
        "confirmation": {
            "raw": {"success_rate": 1, "episodes": 50},
            "raw_repeat": {"max_state_l2_vs_raw_mean": 0, "episodes": 50},
            "dense": {"retention": 1, "max_action_mse": 0, "episodes": 50},
            selected: {
                "retention": 0.9 if invalid == "retention" else 1,
                "success_rate": 0.9
                if invalid in {"retention", "evaluated", "unpermitted_gap"}
                else 1,
                "raw_successes": 50,
                "episodes": 50,
                "successes": 50,
                "action_mse": 0.1,
            },
        },
    }
    runtime = {
        "suite": "libero_goal" if invalid == "suite" else "libero_10",
        "source_commit": "a" * 40,
    }
    if invalid == "protocol":
        spec["execute_steps"] = 8
    if invalid == "spec":
        result["spec_sha256"] = "wrong"
    if invalid == "incomplete":
        result["confirmation"][selected]["episodes"] = 49
    content = (root / "egomimic/rldb/zarr/libero_arc.py").read_bytes()
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *args, **kwargs: b"different codec" if invalid == "codec" else content,
    )

    class Storage:
        def get_object(self, Bucket, Key):
            value = {
                "status.json": {
                    "state": "REPLAY_EVALUATED"
                    if invalid in {"evaluated", "unpermitted_gap"}
                    else "CONFIRMED"
                },
                "result.json": result,
                "runtime.json": runtime,
                "spec.json": spec,
            }[Key.rsplit("/", 1)[-1]]
            return {"Body": io.BytesIO(json.dumps(value).encode())}

    if invalid and invalid != "evaluated":
        with pytest.raises((ValueError, CalibrationPending)):
            load_arc_calibration(Storage(), "replay-run", tmp_path, suite="libero_10")
        assert not (tmp_path / "arc-calibration.json").exists()
    else:
        overrides = load_arc_calibration(
            Storage(), "replay-run", tmp_path, suite="libero_10"
        )
        assert overrides == {
            "arc_waypoints": 16,
            "arc_max_translation": 0.2,
            "arc_max_rotation_degrees": 48,
        }
        assert (
            json.loads((tmp_path / "arc-calibration.json").read_text())[
                "verified_codec_sha256"
            ]
            == hashlib.sha256(content).hexdigest()
        )
    with pytest.raises(CalibrationPending, match="requires"):
        load_arc_calibration(Storage(), None, tmp_path, suite="libero_10")
