import math
from pathlib import Path

import hydra
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from egomimic.rldb.embodiment.yam import Yam
from egomimic.trainHydra import _instantiate_model_wrapper
from scripts.data.measure_abc_control_distance import TASKS, window_distances

ROOT = Path(__file__).resolve().parents[1]
SINGLE_TASK_VISUAL = [
    ("robot_bc/stationery_rl2_hpt300_visual_baseline_openloop", "yam_bimanual", False),
    ("robot_bc/stationery_rl2_hpt300_visual_hybrid_openloop", "yam_bimanual", True),
    ("human_bc/mecka_fold_clothes_40h_human_visual_baseline_openloop", "human_bimanual", False),
    ("human_bc/mecka_fold_clothes_40h_human_visual_hybrid_openloop", "human_bimanual", True),
]
EXPERIMENTS = [
    ("robot_bc/abc_towels_hpt180_baseline_visual_openloop", False, 640, 4),
    ("robot_bc/abc_towels_hpt180_hybrid_visual_openloop", True, 640, 4),
    ("robot_bc/abc_multitask4_hpt300_baseline_visual_openloop", False, 840, 16),
    ("robot_bc/abc_multitask4_hpt300_hybrid_visual_openloop", True, 840, 16),
]


def config(name):
    overrides = [f"+experiment=abc_arc/{name}", "hydra/launcher=submitit_pace_h100"]
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "egomimic/hydra_configs")
    ):
        return compose(config_name="train_zarr_cartesian", overrides=overrides)


@pytest.mark.parametrize("name,embodiment,hybrid", SINGLE_TASK_VISUAL)
def test_single_task_visual_launch_contract(name, embodiment, hybrid, monkeypatch):
    import torch
    monkeypatch.setenv("EGOVERSE_ABC_DATASET_DIR", "/tmp/unused-abc-data")
    cfg = config(name)
    assert "qwen" not in OmegaConf.to_yaml(cfg.model, resolve=True).lower()
    assert cfg.hpt.action_horizon == (200 if hybrid else 100)
    assert cfg.norm_stats.sample_frac == .20
    assert cfg.evaluator.distance_dtw_enabled
    assert cfg.evaluator.velocity_mode == "per_waypoint"
    assert cfg.evaluator.action_mode == ("arc" if hybrid else "baseline")
    for split in ("train_datasets", "valid_datasets"):
        dataset = cfg.data[split][embodiment]
        assert list(dataset.batch_keys) == ["observations.state.ee_pose", "actions_cartesian"]
        assert dataset.resolver.key_map.annotation_key is None
        assert dataset.valid_ratio == .2
    targets = [stage for stage in cfg.model.pipeline.stages
               if stage._target_.endswith("ActionTargetBuilder")]
    stage = hydra.utils.instantiate(targets[0])
    actions = torch.zeros((1, cfg.hpt.action_horizon, 14))
    assert stage({"actions_cartesian": actions})["target"] is actions


@pytest.mark.parametrize("name,hybrid,width,episodes", EXPERIMENTS)
def test_visual_config_contract(name, hybrid, width, episodes, monkeypatch):
    monkeypatch.setenv("EGOVERSE_ABC_DATASET_DIR", "/tmp/unused-abc-data")
    cfg = config(name)
    assert cfg.hpt.embed_dim == width
    assert cfg.hpt.action_horizon == (200 if hybrid else 100)
    assert cfg.norm_stats.sample_frac == 0.20
    targets = [stage for stage in cfg.model.pipeline.stages
               if stage._target_.endswith("ActionTargetBuilder")]
    assert len(targets) == 1
    assert targets[0].action_key == "actions_cartesian"
    assert cfg.evaluator.execute_fraction == 0.30
    assert cfg.evaluator.arc_execution_cap_mode == "waypoints"
    assert cfg.evaluator.limit_val_episodes == episodes
    model = OmegaConf.to_yaml(cfg.model, resolve=True).lower()
    assert "qwen" not in model and "annotation" not in model and "prompt" not in model
    for split in ("train_datasets", "valid_datasets"):
        data = cfg.data[split].yam_bimanual
        assert "annotations" not in data.batch_keys
        assert data.resolver.key_map.annotation_key is None
        assert data.split_seed == 42 and data.valid_ratio == 0.2
        keymap = hydra.utils.instantiate(data.resolver.key_map)
        assert not any(k.get("key_type") == "annotation_keys" for k in keymap.values())
        transform = data.resolver.transform_list
        if hybrid:
            assert transform.action_mode == "hybrid_arc_tokenizer_cartesian"
            assert transform.velocity_mode == "per_waypoint"
            assert transform.rotation_distance_unit == pytest.approx(math.radians(24))
            assert transform.min_distance_unit == cfg.evaluator.min_distance_unit
            for key in (
                "left.cmd_ee_pose",
                "right.cmd_ee_pose",
                "left.cmd_gripper",
                "right.cmd_gripper",
            ):
                assert keymap[key]["horizon"]["distance"] == cfg.abc.arc_distance
                assert (
                    keymap[key]["horizon"]["rotation_distance"]
                    == cfg.abc.arc_rotation_distance
                )
        else:
            assert keymap["left.cmd_ee_pose"]["horizon"] == 100
        predicate = hydra.utils.instantiate(data.filters)
        for task in TASKS:
            row = dict(
                lab="abc",
                embodiment="yam_bimanual",
                task=task,
                zarr_processed_path="s3://example/episode.zarr",
                is_deleted=False,
            )
            assert predicate.matches(row) == ("multitask4" in name or task == TASKS[0])
            assert not predicate.matches(dict(row, lab="rl2"))
            assert not predicate.matches(dict(row, is_deleted=True))


@pytest.mark.parametrize("name,hybrid,width,episodes", EXPERIMENTS)
def test_visual_model_instantiates_without_text_encoder(name, hybrid, width, episodes):
    model = _instantiate_model_wrapper(config(name))
    count = sum(p.numel() for p in model.parameters())
    print(f"{name}: total_parameters={count}", flush=True)
    assert 100_000_000 < count < 350_000_000
    assert not any(
        "qwen" in type(module).__name__.lower() for module in model.modules()
    )


def test_distance_is_joint_sum_not_mean_or_endpoint():
    t = np.arange(101)
    left = np.column_stack((t * 0.001, np.zeros(101), np.zeros(101)))
    right = left * 2
    assert np.allclose(window_distances(left, right), 0.297)
    assert len(window_distances(left, right)) == 2


def test_keymap_override_does_not_mutate_default():
    configured = Yam.get_keymap(
        "hybrid_arc_tokenizer_cartesian",
        min_distance_unit=0.71,
        rotation_distance_unit=0.8,
    )
    assert configured["left.cmd_ee_pose"]["horizon"]["distance"] == 0.71
    default = Yam.get_keymap("hybrid_arc_tokenizer_cartesian")
    assert default["left.cmd_ee_pose"]["horizon"]["distance"] == 0.4
