"""Keep the matched earbuds recipe while removing only velocity averaging."""
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from egomimic.pipeline.inference_config import inference_contract_sha256, model_pipeline_sha256
from egomimic.rldb.zarr.e1_arc_tokenizer import chunk_speed
from scripts.e1.validate_stattempo_configs import compose_recipe


def test_no_smoothing_preserves_a_speed_transition():
    positions = np.zeros((20, 3))
    steps = np.r_[np.full(9, 0.001), np.full(10, 0.01)]
    positions[1:, 0] = np.cumsum(steps)
    expected = np.r_[steps, steps[-1]] * 30
    np.testing.assert_allclose(chunk_speed(positions, 1 / 30, 1), expected)
    assert not np.allclose(chunk_speed(positions, 1 / 30, 7), expected)


def test_recipe_keeps_frozen_split_model_and_budget():
    old = compose_recipe("earbuds_updated_arcvel_hpt300")
    new = compose_recipe("earbuds_updated_arcvel_nosmooth_hpt300")
    a, b = [OmegaConf.to_container(c, resolve=True) for c in (old, new)]
    for key in ("trainer", "callbacks", "norm_stats", "hpt", "ft", "seed", "init_weights_from"):
        assert a[key] == b[key], key
    assert model_pipeline_sha256(old) == model_pipeline_sha256(new)
    assert inference_contract_sha256(old) != inference_contract_sha256(new)
    for split in ("train_datasets", "valid_datasets"):
        before = a["data"][split]["yam_bimanual"]
        after = b["data"][split]["yam_bimanual"]
        assert after["resolver"]["transform_list"].pop("speed_smooth_frames") == 1
        assert before == after
        transforms = instantiate(new.data[split].yam_bimanual.resolver.transform_list)
        assert transforms[-1].speed_smooth_frames == 1
        assert transforms[-1].progress_smooth_hz is None


def test_stationery_recipe_drops_only_speed_smoothing():
    cfg = compose_recipe("bc_rl2_stattempo_slowpace366_arcvel_nosmooth")
    assert cfg.evaluator.token_layout == "e1_profile"
    for split in ("train_datasets", "valid_datasets"):
        transforms = instantiate(cfg.data[split].yam_bimanual.resolver.transform_list)
        assert transforms[-1].speed_smooth_frames == 1
        assert transforms[-1].velocity_mode == "profile"
