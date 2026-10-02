import copy

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from egomimic.benchmarks.libero.arc_streams import (
    ROOT,
    campaign,
    candidates,
    source_receipt,
    stream_training_arguments,
    validate_replay_proof,
    validate_stream_config,
    verify_checkpoint,
)
from scripts.benchmarks.launch_libero_streams import stream_workflow


def experiment(variant, mode, suite="libero_spatial", train_mode="full", gpus=4):
    arguments = stream_training_arguments(
        suite, "data.zarr", "out", train_mode, variant=variant, arc_mode=mode, gpus=gpus
    )
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "egomimic/hydra_configs")
    ):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=arguments[3:] + ["norm_stats.save_cache_dir=out/norm_stats"],
        )
    from egomimic.trainHydra import _build_model_config_tree

    tree = OmegaConf.to_container(_build_model_config_tree(cfg), resolve=True)
    assert "benchmark" not in tree and "seed" not in tree
    return cfg, tree


@pytest.mark.parametrize("variant", campaign()["variants"])
@pytest.mark.parametrize("mode", ["stk", "dur"])
@pytest.mark.parametrize("suite", campaign()["suites"])
def test_all_88_cells_use_the_same_geometry_optimizer_budget_and_backbone(
    variant, mode, suite
):
    cfg, tree = experiment(variant, mode, suite)
    validate_stream_config(tree, suite=suite, variant=variant, arc_mode=mode)
    assert (
        cfg.trainer.devices
        * cfg.benchmark.batch_size
        * cfg.trainer.accumulate_grad_batches
        == 1024
    )
    assert cfg.trainer.max_epochs == 5001
    assert cfg.seed == 42
    assert cfg.benchmark.arc_waypoints == 36
    assert cfg.benchmark.arc_max_translation == 1.6
    assert cfg.benchmark.arc_max_rotation_degrees == 384
    assert cfg.benchmark.n_action_steps == 16


@pytest.mark.parametrize(
    "corruption", ["clock", "encoder", "budget", "width", "mode", "suite", "seed"]
)
def test_checkpoint_recipe_mismatches_fail_closed(corruption):
    _, tree = experiment("component_time", "dur")
    protocol = tree["model"]["benchmark_protocol"]
    if corruption == "clock":
        protocol["arc_stream_spec"]["clocks"] = "group"
    elif corruption == "encoder":
        tree["model"]["pipeline"]["stages"][1]["stream_spec"]["gripper"] = "translation"
    elif corruption == "budget":
        protocol["arc_max_translation"] = 0.8
    elif corruption == "width":
        tree["model"]["pipeline"]["stages"][3]["policy"]["model"]["down_dims"] = [
            128,
            256,
            512,
        ]
    elif corruption == "mode":
        protocol["arc_mode"] = "stk"
    elif corruption == "suite":
        protocol["suite"] = "libero_object"
    else:
        protocol["seed"] = 1
    with pytest.raises(ValueError):
        validate_stream_config(
            tree, suite="libero_spatial", variant="component_time", arc_mode="dur"
        )


def test_full_size_smoke_really_uses_full_microbatch_and_only_two_updates():
    cfg, _ = experiment("component_time", "dur", train_mode="smoke")
    assert cfg.benchmark.batch_size == 256
    assert cfg.trainer.limit_train_batches == 2
    assert cfg.trainer.max_epochs == 1
    assert cfg.callbacks.batch_budget is None


@pytest.mark.parametrize("suite", campaign()["suites"])
def test_replay_precedes_all_training_and_concurrency_is_bounded(suite):
    spec = stream_workflow("a" * 40, "stream-test", suite)["workflow"]
    tasks = {t["name"]: t for t in spec["tasks"]}
    assert len(tasks) == 45
    assert spec["resources"]["default"]["gpu"] == 4
    assert spec["resources"]["evaluation"]["gpu"] == 1
    assert all(r["platform"] == "ovx-l40s" for r in spec["resources"].values())
    assert tasks["replay"]["resource"] == "replay"
    for i in range(22):
        train, evaluation = tasks[f"train-{i:02d}"], tasks[f"evaluate-{i:02d}"]
        assert train["inputs"][0] == {"task": "replay"}
        assert evaluation["inputs"][0] == {"task": train["name"]}
        if i:
            assert train["inputs"][1] == {"task": f"train-{i - 1:02d}"}
            assert evaluation["inputs"][1] == {"task": f"evaluate-{i - 1:02d}"}
        assert (
            "cp '{{input:0}}/replay-completion.json'" in train["files"][0]["contents"]
        )
        assert (
            "cp '{{input:0}}/evaluation-request.json'"
            in evaluation["files"][0]["contents"]
        )
        assert evaluation["environment"]["EVALUATION_WORKERS"] == "5"


def test_replay_receipt_rejects_a_previous_source_or_different_arms():
    proof = {
        "source_commit": "a" * 40,
        "suite": "libero_spatial",
        "sources": source_receipt(),
        "candidates": candidates(),
        "controls_passed": True,
        "episodes": 30,
    }
    validate_replay_proof(proof, "libero_spatial", "a" * 40)
    for field, value in [
        ("source_commit", "b" * 40),
        ("candidates", {}),
        ("episodes", 20),
        ("controls_passed", False),
    ]:
        bad = copy.deepcopy(proof)
        bad[field] = value
        with pytest.raises(ValueError):
            validate_replay_proof(bad, "libero_spatial", "a" * 40)


def test_replay_initializes_dataset_directory_before_simulator_configuration(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from egomimic.benchmarks.libero import arc_streams
    from egomimic.benchmarks.libero import replay

    class StopBeforeNetwork(Exception):
        pass

    def configure(root):
        assert (root / "data").is_dir()

    def stage(*args):
        raise StopBeforeNetwork

    monkeypatch.setattr(arc_streams, "configure_simulator", configure)
    monkeypatch.setattr(replay, "stage_raw_dataset", stage)
    args = SimpleNamespace(root=tmp_path, suite="libero_spatial")
    with pytest.raises(StopBeforeNetwork):
        arc_streams.run_replay(args, tmp_path, "a" * 40)


def test_replay_sends_float32_actions_matching_graph_policy():
    from egomimic.benchmarks.libero.replay import reconstruct_episode
    from egomimic.benchmarks.libero.arc_streams import variant_settings

    settings = variant_settings("component_time", "dur")
    settings.pop("horizon")
    settings.pop("dt")
    actions = np.tile(
        np.array([0.2, -0.1, 0.1, 0.1, 0.1, 0.1, -1], dtype=np.float32), (47, 1)
    )
    spec = {
        "horizon": 32,
        "dt": 0.05,
        "execute_steps": 16,
        "decoded_action_dtype": "float32",
    }
    decoded, metrics = reconstruct_episode(actions, settings, spec)
    assert decoded.dtype == np.float32
    assert metrics["decoded_action_dtype"] == "float32"
    np.testing.assert_allclose(decoded, actions, atol=4e-6)


@pytest.mark.parametrize("mismatch", [None, "initial_state", "data_context", "mode"])
def test_cross_representation_comparison_keeps_exact_pairing(tmp_path, mismatch):
    import json
    from dataclasses import asdict
    from egomimic.benchmarks.libero.arc_stream_report import compare
    from egomimic.benchmarks.libero.catalog import LIBERO_COMMIT, OAT_COMMIT
    from egomimic.benchmarks.libero.rollout import rollout_plan

    protocol = {
        "suite": "libero_spatial",
        "method": "arc",
        "max_episode_steps": 550,
        "horizon": 32,
        "n_obs_steps": 2,
        "n_action_steps": 16,
        "use_ema": True,
        "oat_commit": OAT_COMMIT,
        "libero_commit": LIBERO_COMMIT,
        "data_context": {"id": "fixed"},
        "observations_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "arc_backbone": "unet",
        "arc_stream_variant": "reference",
        "representation": {
            "mode": "dur",
            "waypoints": 36,
            "dt": 0.05,
            "max_translation": 1.6,
            "max_rotation_degrees": 384.0,
            "float32_channels": 12,
        },
        "plan": [asdict(row) for row in rollout_plan("libero_spatial")],
    }
    rows = [
        {
            **row,
            "success": True,
            "initial_state_sha256": "c" * 64,
            "steps": 20,
            "inference_seconds": [0.1],
        }
        for row in protocol["plan"]
    ]
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    for path in (reference, candidate):
        path.mkdir()
    (reference / "protocol.json").write_text(json.dumps(protocol))
    (reference / "episodes.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    protocol["arc_stream_variant"] = "component_time"
    protocol["representation"]["float32_channels"] = 17
    protocol["checkpoint_sha256"] = "d" * 64
    rows[0]["success"] = False
    if mismatch == "initial_state":
        rows[0]["initial_state_sha256"] = "e" * 64
    elif mismatch == "data_context":
        protocol["data_context"] = {"id": "different"}
    elif mismatch == "mode":
        protocol["representation"]["mode"] = "stk"
    (candidate / "protocol.json").write_text(json.dumps(protocol))
    (candidate / "episodes.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    if mismatch:
        with pytest.raises(ValueError):
            compare(candidate, reference)
    else:
        result = compare(candidate, reference)
        assert result["paired_episodes"] == 2500
        assert result["reference_only_successes"] == 1
        assert result["success_delta_percentage_points"] == pytest.approx(-0.04)


@pytest.mark.parametrize(
    "variant,mode",
    [
        ("reference", "stk"),
        ("gripper", "stk"),
        ("component_time", "dur"),
        ("all_scalar", "stk"),
    ],
)
def test_real_training_checkpoint_ema_reload_and_target_free_inference(
    tmp_path, variant, mode
):
    import torch

    from egomimic.benchmarks.libero.rollout import load_policy
    from egomimic.trainHydra import train
    from tests.test_libero_benchmark import make_replay

    path = tmp_path / "data.zarr"
    make_replay(path)
    arguments = stream_training_arguments(
        "libero_10", path, tmp_path, "smoke", variant=variant, arc_mode=mode, gpus=1
    )
    overrides = [
        x
        for x in arguments[3:]
        if not x.startswith(("benchmark.batch_size=", "trainer.precision="))
    ]
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "egomimic/hydra_configs")
    ):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=overrides
            + [
                "benchmark.batch_size=2",
                "trainer.accelerator=cpu",
                "trainer.precision=32-true",
                f"norm_stats.save_cache_dir={tmp_path}",
                f"paths.work_dir={tmp_path}",
            ],
        )
    if variant != "reference":
        cfg.model.pipeline.stages[3].policy.model.down_dims = [16, 32]
        cfg.model.pipeline.stages[3].policy.model.diffusion_step_embed_dim = 16
        cfg.model.pipeline.stages[3].policy.num_inference_steps = 2
    _, objects = train(cfg)
    assert objects["trainer"].global_step == 2
    checkpoint = tmp_path / f"training/arc_{mode}/checkpoints/last.ckpt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["ema_num_updates"] == 2
    assert payload["optimizer_states"]
    if variant == "reference":
        # Exercise the production verifier on a real, full-width checkpoint,
        # including the model-only config tree and Lightning's epoch counters.
        proof = verify_checkpoint(
            checkpoint, suite="libero_10", variant=variant, arc_mode=mode, mode="smoke"
        )
        assert proof["epochs_completed"] == 1
        assert proof["global_step"] == proof["ema_num_updates"] == 2
    policy, protocol = load_policy(checkpoint, device="cpu")
    assert protocol["arc_stream_variant"] == variant
    observation = {
        "agentview_rgb": np.zeros((128, 128, 3), dtype=np.uint8),
        "robot0_eye_in_hand_rgb": np.zeros((128, 128, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([0, 0, 0, 1]),
        "robot0_gripper_qpos": np.zeros(2),
        "task_uid": np.array([30]),
    }
    policy.reset(observation)
    actual = policy.predict()
    assert actual.shape == (16, 7) and np.isfinite(actual).all()
    # Do not accumulate multi-hundred-MB synthetic checkpoints between cases.
    checkpoint.unlink()
