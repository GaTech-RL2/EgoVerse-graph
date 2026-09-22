"""Composition checks for the stationary HPT180 ARC/non-ARC pair."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir


_REPO = Path(__file__).resolve().parents[1]
_CONFIGS = _REPO / "egomimic/hydra_configs"


def _compose(experiment: str):
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIGS)):
        return compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment=abc_arc/{experiment}", "++paths.root_dir=."],
        )


@pytest.mark.parametrize(
    ("experiment", "is_arc", "expected_horizon"),
    [
        ("stationery_rl2_hpt180_baseline_openloop", False, 100),
        ("stationery_rl2_hpt180_arc_D40_M100_openloop", True, 200),
    ],
)
def test_stationery_hpt180_pair_preserves_run_contract(
    experiment, is_arc, expected_horizon
):
    cfg = _compose(experiment)

    # h640t8_d384x10 rollout bundle architecture.
    assert cfg.hpt.embed_dim == 640
    assert cfg.hpt.num_blocks == 19
    assert cfg.hpt.num_heads == 8
    assert cfg.hpt.stem_specs.cross_attn.crossattn_heads == 8
    assert cfg.hpt.stem_specs.cross_attn.crossattn_dim_head == 80

    stages = cfg.model.pipeline.stages
    assert stages[1].stems["observations.images.front_img_1"].output_dim == 640
    assert stages[2].trunk.embed_dim == 640
    assert stages[2].trunk.attn_target.num_heads == 8
    flow = stages[5].model
    assert flow.hidden_dim == 384
    assert flow.nblocks == 10
    assert flow.act_seq == expected_horizon

    # The 300M pair's run contract is deliberately unchanged.
    assert cfg.hpt.action_horizon == expected_horizon
    assert cfg.trainer.val_check_interval == 10000
    assert cfg.norm_stats.sample_frac == 0.20
    assert cfg.evaluator.limit_val_episodes == 4
    assert cfg.evaluator.requires_ordered_validation is True
    assert cfg.evaluator.control_horizon == 100
    assert cfg.evaluator.execute_fraction == 0.30
    assert cfg.evaluator.action_mode == ("arc" if is_arc else "baseline")

    transform_list = cfg.data.train_datasets.yam_bimanual.resolver.transform_list
    if is_arc:
        assert cfg.abc.arc_token_rows == 200
        assert transform_list.action_mode == "arc_tokenizer_cartesian"
        assert transform_list.min_distance_unit == 0.40
        assert transform_list.resampled_vector_length == 100
    else:
        assert transform_list.action_mode == "cartesian"


@pytest.mark.parametrize(
    "config_name",
    [
        "stationery_rl2_hpt_arc_D40_M100.yaml",
        "stationery_rl2_hpt_baseline.yaml",
        "stationery_rl2_hpt180_arc_D40_M100.yaml",
        "stationery_rl2_hpt180_baseline.yaml",
    ],
)
def test_stationery_rl2_filters_use_catalog_task_name(config_name):
    config = (
        _CONFIGS / "data" / "abc_arc" / config_name
    ).read_text(encoding="utf-8")
    expected = "row['task'] == 'organize_stationary_updated'"
    stale = "row['task'] == 'organize_stationary'"
    assert config.count(expected) == 2
    assert stale not in config
