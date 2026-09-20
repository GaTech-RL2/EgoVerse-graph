import importlib.util
import json
import zipfile
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from egomimic.benchmarks.libero.cluster import (
    ArtifactUploader,
    digest,
    extract_replay,
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
