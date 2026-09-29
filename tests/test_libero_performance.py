"""Performance paths must preserve exact targets, data and training budgets."""

import importlib.util
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from egomimic.benchmarks.libero.cluster import (
    arc_checkpoint_settings,
    digest,
    restore_checkpoints,
    training_arguments,
    training_layout,
)
from egomimic.pipeline.stages_libero_arc import LiberoArcStage


@pytest.mark.parametrize("mode", ["joint_dur", "stk", "dur"])
def test_target_cache_is_exact_bounded_and_invalidated_by_codec_changes(mode):
    options = dict(
        arc_mode=mode,
        num_waypoints=24,
        max_translation=0.8,
        max_rotation_degrees=192,
        velocity_norm_bound=3**0.5,
    )
    cached = LiberoArcStage(**options, encode_cache_size=2)
    reference = LiberoArcStage(**options)
    actions = torch.from_numpy(
        np.random.default_rng(5).uniform(-1, 1, (3, 32, 7)).astype(np.float32)
    )
    actions[0, 4:12, :6] = 0  # Exercise duration dwells and velocity limitations.
    calls = []
    original = cached.codec.encode

    def count(row):
        calls.append(row.copy())
        return original(row)

    cached.codec.encode = count
    for indices, count_expected in [([0, 1, 0], 2), ([1, 0], 2), ([2, 0], 3), ([1], 4)]:
        batch = actions[indices]
        expected = reference.execute({"actions": batch}, mode="train")["target"]
        actual = cached.execute({"actions": batch}, mode="train")["target"]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert len(calls) == count_expected
        assert len(cached._encode_cache) <= 2
        actual.zero_()  # Returned values must never alias cached physical tokens.
    cached.codec.max_translation = reference.codec.max_translation = 0.01
    actual = cached.execute({"actions": actions[1:2]}, mode="train")["target"]
    expected = reference.execute({"actions": actions[1:2]}, mode="train")["target"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(calls) == 5
    assert cached.state_dict().keys() == reference.state_dict().keys()
    with pytest.raises(ValueError, match="batch"):
        cached._encode(actions[1:2].numpy().reshape(1, 7, 32))


def test_target_cache_preserves_normalizer_changes_and_rejects_nonfinite():
    cached = LiberoArcStage(arc_mode="stk", encode_cache_size=8)
    reference = LiberoArcStage(arc_mode="stk")
    actions = torch.ones(1, 32, 7) * 0.25
    cached.execute({"actions": actions}, mode="train")
    for stage in (cached, reference):
        stage.action_scale.fill_(2)
        stage.action_offset.fill_(0.1)
    torch.testing.assert_close(
        cached.execute({"actions": actions}, mode="train")["target"],
        reference.execute({"actions": actions}, mode="train")["target"],
        rtol=0,
        atol=0,
    )
    assert len(cached._encode_cache) == 2
    with pytest.raises(ValueError, match="finite"):
        cached.execute(
            {"actions": torch.full_like(actions, float("nan"))}, mode="train"
        )


def test_replay_array_handles_survive_repeated_reads_and_reset_on_spawn(
    tmp_path, monkeypatch
):
    import zarr

    from egomimic.rldb.zarr.libero_dataset import LiberoReplayResolver, keymap
    from tests.test_libero_benchmark import make_replay

    path = tmp_path / "replay.zarr"
    make_replay(path)
    leaf = next(
        iter(LiberoReplayResolver(path, keymap(), "libero10").resolve().values())
    )
    expected = leaf[0]
    original = zarr.Group.__getitem__
    calls = []

    def tracked(group, key):
        calls.append(key)
        return original(group, key)

    monkeypatch.setattr(zarr.Group, "__getitem__", tracked)
    actual = leaf[0]
    assert calls == []
    restored = pickle.loads(pickle.dumps(leaf))
    assert restored._arrays is None
    restored_sample = restored[0]
    assert calls
    for key, value in expected.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
            torch.testing.assert_close(restored_sample[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("gpus", [1, 2, 4, 8])
@pytest.mark.parametrize("mode", ["full", "smoke"])
def test_distributed_arguments_preserve_global_batch(gpus, mode, tmp_path):
    args = training_arguments(
        "arc_stk", "libero_10", tmp_path / "replay", tmp_path, mode, 5001, gpus=gpus
    )
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).parents[1] / "egomimic/hydra_configs"),
    ):
        cfg = compose(config_name="train_zarr_cartesian", overrides=args[3:])
    assert cfg.trainer.devices == gpus
    effective = cfg.benchmark.batch_size * gpus * cfg.trainer.accumulate_grad_batches
    assert effective == (1024 if mode == "full" else 4 * gpus)
    if mode == "full":
        assert cfg.callbacks.batch_budget.global_batch_size == effective
        assert cfg.trainer.max_epochs == 5001
    if gpus > 1:
        assert cfg.trainer.strategy == "ddp_find_unused_parameters_true"
    with pytest.raises(ValueError, match="GPUs"):
        training_layout(3, mode)


def test_sweep_resume_reuses_confirmed_replay_and_resources(tmp_path, monkeypatch):
    from egomimic.benchmarks.libero import arc_sweep, cluster

    path = Path(__file__).parents[1] / "scripts/benchmarks/launch_libero_osmo.py"
    spec = importlib.util.spec_from_file_location("ddp_launcher", path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    args = dict(
        commit="a" * 40,
        run_id="ddp-stk1",
        suite="libero_10",
        mode="full",
        arc_modes=["stk"],
        arc_profile="stk_1",
        resume_from_run="old-stk1",
        oat_reference_run="baseline",
        gpus=4,
    )
    with pytest.raises(ValueError, match="completed mode-specific replay"):
        launcher.workflow(**args)
    workflow = launcher.workflow(**args, arc_replay_runs={"stk": "confirmed-stk1"})[
        "workflow"
    ]
    assert workflow["resources"]["default"]["gpu"] == 4
    assert workflow["resources"]["default"]["cpu"] == 48
    assert workflow["resources"]["default"]["memory"] == "256Gi"
    env = workflow["tasks"][0]["environment"]
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert env["TRAINING_GPUS"] == "4"
    calls = []
    monkeypatch.setattr(cluster, "execute", lambda argv, log: calls.append(argv))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sweep",
            "--root",
            str(tmp_path),
            "--suite",
            "libero_10",
            "--run-id",
            "ddp-stk1",
            "--profile",
            "stk_1",
            "--arc-mode",
            "stk",
        ],
    )
    arc_sweep.main()
    assert len(calls) == 1 and calls[0][2].endswith(".cluster")
    assert "--arc-only" in calls[0]
    assert os.environ["ARC_REPLAY_RUNS_JSON"] == '{"stk": "confirmed-stk1"}'


def test_resume_changes_world_size_without_resetting_optimizer_or_ema(tmp_path):
    script = Path(__file__).with_name("libero_ddp_resume_worker.py")
    env = dict(
        os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
    )
    for suffix in ([], ["--resume"]):
        result = subprocess.run(
            [sys.executable, str(script), "--root", str(tmp_path), *suffix],
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stdout[-8000:] + result.stderr[-8000:]
    initial = json.loads((tmp_path / "initial.json").read_text())
    resumed = json.loads((tmp_path / "resumed.json").read_text())
    assert initial["global_step"] == initial["ema_num_updates"] == 11
    assert resumed["global_step"] == resumed["ema_num_updates"] == 22
    assert resumed["optimizer_states"]
    budget = resumed["training_budget"]
    assert budget["world_size"] == budget["gradient_accumulation"] == 2
    assert budget["global_batch_size"] == 8
    assert budget["optimizer_steps_per_epoch"] == 11


@pytest.mark.parametrize("method", ["arc", "arc_stk", "arc_dur"])
@pytest.mark.parametrize("backbone", ["unet", "oat_dp"])
def test_arc_resume_reads_actual_saved_graph_config(tmp_path, method, backbone):
    import io
    import shutil

    from egomimic.trainHydra import _build_model_config_tree

    args = training_arguments(
        method,
        "libero_10",
        tmp_path,
        tmp_path,
        "full",
        5001,
        arc_backbone=backbone,
    )
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).parents[1] / "egomimic/hydra_configs"),
    ):
        cfg = compose(config_name="train_zarr_cartesian", overrides=args[3:])
    config = _build_model_config_tree(cfg)
    assert "benchmark" not in config  # Actual trainHydra checkpoint format.
    source = tmp_path / "source.ckpt"
    torch.save(
        {
            "global_step": 140,
            "loops": {"fit_loop": {"epoch_progress": {"current": {"completed": 2}}}},
            "training_budget": {"global_batch_size": 1024, "epochs": 5001},
            "optimizer_states": [{"state": {}}],
            "normalizer_state": {},
            "hyper_parameters": {"config_tree": config},
        },
        source,
    )
    runtime = {
        "suite": "libero_10",
        "mode": "full",
        "epochs": 5001,
        "global_batch_size": 1024,
    }
    receipts = {
        f"training/{method}/checkpoints/last.ckpt": {
            "sha256": digest(source),
            "bytes": source.stat().st_size,
            "uri": "s3://rldb/experiments/arc-oat-20260919/source/checkpoints/last.ckpt",
        }
    }

    class Storage:
        def get_object(self, Bucket, Key):
            value = runtime if Key.endswith("runtime.json") else receipts
            return {"Body": io.BytesIO(json.dumps(value).encode())}

        def download_file(self, bucket, key, destination):
            shutil.copyfile(source, destination)

    restored = restore_checkpoints(
        Storage(),
        "source",
        tmp_path / "evidence",
        suite="libero_10",
        mode="full",
        epochs=5001,
        allow_partial=True,
        methods=[method],
    )[method]
    assert restored["global_step"] == 140
    assert restored["benchmark"].get("arc_backbone", "unet") == backbone
    for key, value in restored["benchmark"].items():
        assert value == cfg.benchmark[key]
    with pytest.raises(ValueError, match="protocol"):
        arc_checkpoint_settings(config, suite="libero_spatial")
    from omegaconf import open_dict

    with open_dict(config.model.benchmark_protocol):
        config.model.benchmark_protocol.arc_backbone = "unknown"
    with pytest.raises(ValueError, match="unknown backbone"):
        arc_checkpoint_settings(config, suite="libero_10")
    config.model.benchmark_protocol.arc_backbone = backbone
    config.model.pipeline.stages[-1].num_waypoints += 1
    with pytest.raises(ValueError, match="parameters differ"):
        arc_checkpoint_settings(config, suite="libero_10")
