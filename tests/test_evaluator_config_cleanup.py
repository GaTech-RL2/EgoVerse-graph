"""Evaluator names describe predictions/frames, not a policy architecture."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

CONFIGS = Path(__file__).resolve().parents[1] / "egomimic/hydra_configs"


def evaluator_config(name):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        cfg = compose(config_name=None, overrides=[f"+evaluator={name}"])
    with open_dict(cfg):
        cfg.paths = {"output_dir": "/tmp/evaluator-config-test"}
        cfg.model = {
            "pipeline": {"stages": [{"policy": {"config": {"reverse_kl_samples": 8}}}]}
        }
    return cfg.evaluator


def test_model_named_evaluator_and_renderer_aliases_are_removed():
    root = CONFIGS / "evaluator"
    assert not list(root.glob("eval_hpt*.yaml"))
    assert not list(root.glob("eval_pi*.yaml"))
    assert not list((root / "pi05").glob("*.yaml"))
    assert not list((root / "viz").glob("pi_*.yaml"))
    for name in ("cartesian_wrist", "cartesian_wrist_lang", "cotrain_lang"):
        assert not (root / "viz" / f"{name}.yaml").exists()


def test_open_loop_and_chunk_metrics_share_native_pose_bindings():
    execution = evaluator_config("eval_open_loop_sim")
    chunks = evaluator_config("eval_bimanual_cartesian")
    for key in (
        "action_key",
        "obs_pose_key",
        "image_key",
        "revert_transforms",
        "viz_func",
        "video_chunk_frames",
        "max_episode_frames",
    ):
        assert OmegaConf.to_container(OmegaConf.create({key: execution[key]})) == (
            OmegaConf.to_container(OmegaConf.create({key: chunks[key]}))
        )
    evaluator = instantiate(execution)
    assert evaluator.distance_dtw_enabled
    assert evaluator.execute_steps == 30
    assert evaluator.data_requirements().complete_episodes
    assert evaluator.data_requirements().ordered
    assert evaluator.arc_execution_cap_mode == "waypoints"


@pytest.mark.parametrize(
    "name,suffix,max_batches",
    [
        (
            "eval_cartesian_wrist_prompt",
            "cartesian_revert_eef_frame_transform_list",
            None,
        ),
        ("eval_cartesian_camera_6d_prompt", "cartesian_revert_6d_transform_list", None),
        (
            "eval_cartesian_wrist_6d_prompt",
            "cartesian_revert_6d_wristframe_transform_list",
            10,
        ),
        (
            "eval_cartesian_wrist_6d_gripper_prompt",
            "cartesian_revert_6d_wristframe_grip_transform_list",
            10,
        ),
    ],
)
def test_prompt_presets_preserve_frame_and_sampling_contract(name, suffix, max_batches):
    cfg = evaluator_config(name)
    assert cfg.revert_transforms.human_bimanual._target_.endswith(suffix)
    assert cfg.viz_every_n_epochs == 100
    assert cfg.get("viz_max_batches") == max_batches
    assert cfg.rkl_samples == 8
    assert cfg.group_options.train_viz.limit_batches == 50
    assert cfg.group_options.train_viz.rkl_samples == 1
    assert cfg.viz_func.human_bimanual.annotation_key == "sampled_prompt"
    assert cfg.viz_func.eva_bimanual.fallback_intrinsics[0][0] == 436.26
    instantiate(cfg)
