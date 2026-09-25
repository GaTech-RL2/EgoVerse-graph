"""Composition checks for the HPT180 ABC towel ARC/baseline pair."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir


_CONFIGS = Path(__file__).resolve().parents[1] / "egomimic/hydra_configs"


def _compose(experiment: str):
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIGS)):
        return compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment=abc_arc/{experiment}", "++paths.root_dir=."],
        )


@pytest.mark.parametrize(
    ("experiment", "is_arc", "expected_horizon"),
    [
        ("robot_bc/abc_towels_hpt180_baseline_visual_openloop", False, 100),
        ("robot_bc/abc_towels_hpt180_hybrid_visual_openloop", True, 200),
    ],
)
def test_abc_towel_pair_preserves_hpt180_contract(experiment, is_arc, expected_horizon):
    cfg = _compose(experiment)
    assert cfg.hpt.embed_dim == 640
    assert cfg.hpt.num_blocks == 19
    assert cfg.hpt.num_heads == 8
    assert cfg.hpt.action_horizon == expected_horizon
    assert cfg.trainer.val_check_interval == 10000
    assert cfg.norm_stats.sample_frac == 0.10
    assert cfg.evaluator.limit_val_episodes == 4
    assert cfg.evaluator.requires_ordered_validation is True
    assert cfg.evaluator.control_horizon == 100
    assert cfg.evaluator.execute_fraction == 0.30
    assert cfg.evaluator.action_mode == ("arc" if is_arc else "baseline")

    train_filter = cfg.data.train_datasets.yam_bimanual.filters.filter_lambdas[0]
    valid_filter = cfg.data.valid_datasets.yam_bimanual.filters.filter_lambdas[0]
    for expression in (train_filter, valid_filter):
        assert "row['lab'] == 'abc'" in expression
        assert "row['task'] == 'fold and stack the towels'" in expression

    transform = cfg.data.train_datasets.yam_bimanual.resolver.transform_list
    assert transform.action_mode == (
        "hybrid_arc_tokenizer_cartesian" if is_arc else "cartesian"
    )
