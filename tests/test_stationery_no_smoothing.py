"""Keep the matched earbuds recipe while removing only velocity averaging."""

import numpy as np
from hydra.utils import instantiate

from egomimic.rldb.zarr.e1_arc_tokenizer import chunk_speed
from scripts.e1.validate_stattempo_configs import compose_recipe


def test_no_smoothing_preserves_a_speed_transition():
    positions = np.zeros((20, 3))
    steps = np.r_[np.full(9, 0.001), np.full(10, 0.01)]
    positions[1:, 0] = np.cumsum(steps)
    expected = np.r_[steps, steps[-1]] * 30
    np.testing.assert_allclose(chunk_speed(positions, 1 / 30, 1), expected)
    assert not np.allclose(chunk_speed(positions, 1 / 30, 7), expected)


def test_arcvel_defaults_to_no_smoothing():
    cfg = compose_recipe("bc_rl2_stattempo_slowpace366_arcvel_m200_d033")
    for split in ("train_datasets", "valid_datasets"):
        transforms = instantiate(cfg.data[split].yam_bimanual.resolver.transform_list)
        assert transforms[-1].velocity_mode == "profile"
        assert transforms[-1].speed_smooth_frames == 1


def test_stationery_recipe_drops_only_speed_smoothing():
    cfg = compose_recipe("bc_rl2_stattempo_slowpace366_arcvel_nosmooth_m200_d033")
    assert cfg.evaluator.token_layout == "e1_profile"
    for split in ("train_datasets", "valid_datasets"):
        transforms = instantiate(cfg.data[split].yam_bimanual.resolver.transform_list)
        assert transforms[-1].speed_smooth_frames == 1
        assert transforms[-1].velocity_mode == "profile"
