"""Hermetic BSP parity, temporal semantics, normalization and checkpoint tests."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from egomimic.models.bspline import BSplineCodec


def test_encoding_and_span_decoding_match_pinned_upstream():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/bspline_reference.json").read_text()
    )
    codec = BSplineCodec()
    assert codec.context()["upstream_commit"] == fixture["upstream_commit"]
    trajectory = codec.fit(fixture["actions"])
    assert trajectory.max_error < codec.max_error
    for case in fixture["cases"]:
        actual = codec.encode(trajectory, case["frame_index"])
        np.testing.assert_allclose(actual, case["parameters"], atol=2e-6)
        spline = codec.spline(actual)
        times = np.linspace(spline.t[3], spline.t[-4], 9)
        np.testing.assert_allclose(
            spline(times), case["uniform_span_decode"], atol=2e-6
        )


@pytest.mark.parametrize("length", [1, 2, 3, 4, 40])
def test_constant_short_episode_padding_and_terminal_gripper(length):
    codec = BSplineCodec()
    commands = np.tile([0.1, 0, 0, 0, 0, 0, -1], (length, 1))
    fitted = codec.fit(commands)
    for t in (0, length - 1):
        actual = codec.decode(codec.encode(fitted, t), horizon=32)
        np.testing.assert_allclose(actual, np.tile(commands[0], (32, 1)), atol=2e-6)


def test_sampling_retains_observation_origin_clock_speed_and_terminal_hold():
    codec = BSplineCodec()
    t = np.arange(80)
    actions = np.column_stack([t / 100, t / 200, np.zeros((80, 4)), -np.ones(80)])
    params = codec.encode(codec.fit(actions), 11)
    np.testing.assert_allclose(
        codec.decode(params, horizon=16), actions[11:27], atol=1e-6
    )
    np.testing.assert_allclose(
        codec.decode(params, horizon=16, speed=2), actions[11:43:2], atol=1e-6
    )
    tail = codec.decode(codec.encode(codec.fit(actions), 78), horizon=5)
    np.testing.assert_allclose(tail, actions[[78, 79, 79, 79, 79]], atol=1e-6)


def test_knot_projection_keeps_control_points_order_and_handles_degenerate_predictions():
    codec = BSplineCodec()
    knots = np.array([0, 0, 0, 0, 3, 2, 1, 7, 8, 9, 10, 11, 12, 12, 12, 12.0])
    projected = codec.project_knots(knots)
    assert np.all(np.diff(projected) >= 0)
    np.testing.assert_array_equal(projected[:4], knots[:4])
    assert projected[5] == pytest.approx(3 + 1e-6)
    constant = np.zeros((16, 8))
    constant[:, 1:] = 0.25
    np.testing.assert_allclose(codec.decode(constant), 0.25, atol=1e-6)
    for invalid in (np.full((16, 8), np.nan), np.zeros((15, 8))):
        with pytest.raises(ValueError):
            codec.decode(invalid)


def test_absolute_action_alignment_finds_matching_phase():
    codec = BSplineCodec()
    actions = np.arange(40, dtype=float)[:, None] * np.ones((1, 7))
    params = codec.encode(codec.fit(actions), 0)
    assert codec.alignment_time(
        params, actions[7], inference_latency=8
    ) == pytest.approx(7, abs=1e-4)
    assert codec.alignment_time(params, actions[7], inference_latency=0) == 0
    # Padded terminal knots must not invent a zero-valued alignment candidate.
    shifted = codec.encode(codec.fit(actions + 2), 0)
    assert codec.alignment_time(shifted, np.zeros(7), inference_latency=80) == 0


def test_dataset_cache_split_stats_and_decode_normalization(tmp_path, monkeypatch):
    from egomimic.pipeline.stages_bspline import BSplineDecodeStage
    from egomimic.rldb.zarr.libero_bspline import (
        LiberoBSplineNormalizer,
        LiberoBSplineResolver,
    )
    from egomimic.rldb.zarr.libero_dataset import (
        EMBODIMENT,
        LiberoDataset,
        LiberoNormalizer,
        keymap,
    )
    from tests.test_libero_benchmark import make_replay

    path = tmp_path / "replay.zarr"
    make_replay(path, episode_length=8)
    resolver = LiberoBSplineResolver(
        path, keymap(2, 32), "libero_10", decoded_cache=True
    )
    train = LiberoDataset._from_resolver(resolver)
    valid = LiberoDataset._from_resolver(resolver, mode="valid")
    assert set(train.datasets).isdisjoint(valid.datasets)
    norm = LiberoBSplineNormalizer()
    norm.populate_from_datasets({"libero_panda": train})
    norm.infer_norm_from_dataset(train, "libero_panda")
    train.set_norm_stats_from(norm)
    valid.set_norm_stats_from(norm)
    low = np.full(8, np.inf)
    high = np.full(8, -np.inf)
    for ep in train.datasets.values():
        values = np.stack([ep.codec.encode(ep.trajectory, t) for t in range(len(ep))])
        low = np.minimum(low, values.min(axis=(0, 1)))
        high = np.maximum(high, values.max(axis=(0, 1)))
    np.testing.assert_allclose(norm.norm_stats[EMBODIMENT]["bspline"]["min"], low)
    np.testing.assert_allclose(norm.norm_stats[EMBODIMENT]["bspline"]["max"], high)
    restored_norm = LiberoNormalizer(state=norm.to_state())
    stage = BSplineDecodeStage()
    stage.bind_data_context(normalizer=restored_norm)
    sample = train[0]
    predicted = stage({"pred_action": sample["bspline"].unsqueeze(0)})["pred_action"]
    native = restored_norm.unnormalize({"actions": predicted}, EMBODIMENT)["actions"][
        0
    ].numpy()
    name, index = train.index_map[0]
    ep = train.datasets[name]
    expected = ep.codec.decode(ep.codec.encode(ep.trajectory, index))
    np.testing.assert_allclose(native, expected, atol=2e-4)

    # A new resolver must reuse disk fits, without fitting again or reading images.
    def fail_fit(*args, **kwargs):
        raise AssertionError("cache hit attempted another fit")

    monkeypatch.setattr(BSplineCodec, "fit", fail_fit)
    again = LiberoDataset._from_resolver(
        LiberoBSplineResolver(path, keymap(), "libero_10")
    )
    assert again[0]["bspline"].shape == (16, 8)
    incompatible = BSplineDecodeStage(codec={"max_error": 0.01})
    with pytest.raises(ValueError, match="representation differ"):
        incompatible.bind_data_context(normalizer=restored_norm)


@pytest.mark.parametrize(
    "backbone,method", [("unet", "bspline_unet"), ("oat_dp", "bspline_oat")]
)
def test_recipe_workflow_and_protocol_are_distinct_from_raw_dp(
    tmp_path, backbone, method
):
    from egomimic.benchmarks.libero.bspline import validate_bspline_config
    from egomimic.benchmarks.libero.cluster import training_arguments
    from scripts.benchmarks.launch_libero_osmo import (
        baseline_workflow,
        evaluation_workflow,
    )
    from tests.test_oat_training import config_for

    cfg = config_for(f"libero_bspline_{backbone}_policy", "unused.zarr", tmp_path)
    cfg.benchmark.horizon, cfg.benchmark.n_action_steps = 32, 16
    resolved = OmegaConf.to_container(cfg, resolve=True)
    validate_bspline_config(resolved, method)
    for field, value in [
        ("segment_alignment", True),
        ("execution_speed", 2),
        ("action_representation", "raw_actions"),
    ]:
        invalid = copy.deepcopy(resolved)
        invalid["model"]["benchmark_protocol"][field] = value
        with pytest.raises(ValueError):
            validate_bspline_config(invalid, method)
    spec = baseline_workflow(
        "a" * 40, "bsp-test", "libero_10", backbone=backbone, representation="bspline"
    )
    train, evaluate = spec["workflow"]["tasks"]
    assert train["environment"]["DP_REPRESENTATION"] == "bspline"
    assert evaluate["inputs"] == [{"task": "train"}]
    assert all(
        r["platform"] == "ovx-l40s" for r in spec["workflow"]["resources"].values()
    )
    command = training_arguments(method, "libero_10", "dataset", tmp_path, "full", 5001)
    assert f"+experiment=oat/libero_bspline_{backbone}_policy" in command
    request = {
        "source_run": "bsp-test",
        "source_commit": "a" * 40,
        "suite": "libero_10",
        "method": method,
        "epochs": 5001,
        "total_optimizer_steps": 5001,
        "checkpoint": {
            "sha256": "b" * 64,
            "bytes": 1,
            "uri": "s3://rldb/experiments/arc-oat-20260919/bsp-test/checkpoints/"
            + "b" * 64
            + "/last.ckpt",
        },
    }
    recovery = evaluation_workflow("a" * 40, "bsp-test-eval", request)
    task = recovery["workflow"]["tasks"][0]
    assert task["environment"]["DP_REPRESENTATION"] == "bspline"
    # Preserve the dependency entrypoint used by existing ARC recovery tooling.
    assert "uv pip install -e '.[oat,libero]'" in task["files"][0]["contents"]


@pytest.mark.parametrize(
    "backbone,method", [("unet", "bspline_unet"), ("oat_dp", "bspline_oat")]
)
def test_native_training_resume_checkpoint_and_target_free_rollout(
    tmp_path, backbone, method
):
    from egomimic.benchmarks.libero.baseline import verify_checkpoint
    from egomimic.benchmarks.libero.cli import policy_method
    from egomimic.benchmarks.libero.rollout import load_policy
    from egomimic.trainHydra import train
    from tests.test_libero_benchmark import make_replay
    from tests.test_oat_training import config_for

    path = tmp_path / "replay.zarr"
    make_replay(path)
    cfg = config_for(f"libero_bspline_{backbone}_policy", path, tmp_path / "training")
    cfg.benchmark.suite = "libero_10"
    cfg.benchmark.horizon, cfg.benchmark.n_action_steps = 32, 16
    cfg.trainer.precision = "bf16-mixed"
    net = cfg.model.pipeline.stages[3].policy.model
    if backbone == "unet":
        net.down_dims, net.diffusion_step_embed_dim = [16, 32], 32
    else:
        net.n_emb, net.n_layer = 32, 1
    _, objects = train(cfg)
    checkpoint = tmp_path / "training/checkpoints/last.ckpt"
    proof = verify_checkpoint(
        checkpoint, suite="libero_10", method=method, mode="smoke"
    )
    assert proof["global_step"] == proof["ema_num_updates"] == 2
    cfg.ckpt_path, cfg.trainer.max_epochs = str(checkpoint), 2
    _, resumed = train(cfg)
    assert resumed["trainer"].global_step == 4
    restored, protocol = load_policy(checkpoint, device="cpu")
    assert policy_method(restored.algo.pipeline.stages, protocol) == method
    # Evaluation is independent of the source dataset and derived fit cache.
    import shutil

    shutil.rmtree(path)
    shutil.rmtree(str(path) + ".bspline")
    observation = {
        "agentview_rgb": np.zeros((128, 128, 3), dtype=np.float32),
        "robot0_eye_in_hand_rgb": np.zeros((128, 128, 3), dtype=np.float32),
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([0, 0, 0, 1]),
        "robot0_gripper_qpos": np.zeros(2),
        "task_uid": np.array([30]),
    }
    restored.reset(observation)
    assert restored.predict().shape == (16, 7)
