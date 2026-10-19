"""The HPT-300M towels / earbuds M 200 / D 0.33 leaves thread M and D everywhere they are read."""

import numpy as np
import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from scripts.e1.validate_stattempo_configs import compose_recipe

LEAVES = {
    "earbuds_updated_arcdur_m200_d033_hpt300": ("arcdur", "dur"),
    "earbuds_updated_arcvel_m200_d033_hpt300": ("arcvel", "profile"),
    "earbuds_updated_arcdur_abstime_m200_d033_hpt300": ("arcdur", "dur"),
    "earbuds_updated_arcvel_nosmooth_m200_d033_hpt300": ("arcvel", "profile"),
    "towels_since0924plus74_arcdur_m200_d033_hpt300": ("arcdur", "dur"),
    "towels_since0924plus74_arcvel_m200_d033_hpt300": ("arcvel", "profile"),
    "towels_new290plus74_arcdur_m200_d033_hpt300": ("arcdur", "dur"),
    "towels_new290plus74_arcvel_m200_d033_hpt300": ("arcvel", "profile"),
}


def flat(d, p=""):
    if isinstance(d, dict):
        return {
            k2: v2
            for k, v in d.items()
            for k2, v2 in flat(v, f"{p}.{k}" if p else str(k)).items()
        }
    if isinstance(d, list):
        return {
            k2: v2 for i, v in enumerate(d) for k2, v2 in flat(v, f"{p}[{i}]").items()
        }
    return {p: d}


@pytest.mark.parametrize("name", LEAVES)
def test_leaf_changes_only_m_and_d_against_its_m100_self(name):
    old = compose_recipe(name, overrides=["e1.M=100", "e1.D=0.4"])
    new = compose_recipe(name)
    a, b = [flat(OmegaConf.to_container(c, resolve=True)) for c in (old, new)]
    for k in {k for k in set(a) | set(b) if a.get(k) != b.get(k)}:
        assert (a.get(k), b.get(k)) in ((100, 200), (0.4, 0.33)), (
            k,
            a.get(k),
            b.get(k),
        )


@pytest.mark.parametrize("name,expected", LEAVES.items())
def test_codec_evaluator_and_decoder_agree_on_m_and_d(name, expected):
    variant, mode = expected
    cfg = compose_recipe(name)
    M, D = int(cfg.e1.M), float(cfg.e1.D)
    assert (M, D) == (200, 0.33) and cfg.e1.variant == variant
    assert cfg.hpt.embed_dim == 840 and cfg.hpt.num_blocks == 19
    for split in ("train_datasets", "valid_datasets"):
        codec = instantiate(cfg.data[split].yam_bimanual.resolver.transform_list)[-1]
        assert codec.M == M and codec.tokenizer.config.min_distance_unit == D
        assert codec.velocity_mode == mode
        if "abstime" in name:
            assert codec.duration_timing == "absolute"
        if "nosmooth" in name:
            assert codec.speed_smooth_frames == 1
    evaluator = instantiate(cfg.evaluator)
    assert (evaluator.metric_options["M"], evaluator.metric_options["D"]) == (M, D)
    decoder = instantiate(cfg.model.inference.profiles.default.adapter.decoder)
    assert (
        decoder.codec.M == M and decoder.codec.tokenizer.config.min_distance_unit == D
    )
    assert decoder.first_stream == 1 and decoder.execute_percent == 50
    chunk = np.zeros((100, 14))
    for o in (0, 7):  # 0.5 m of path per arm: longer than D
        chunk[:, o] = 0.5 * np.linspace(0, 1, 100) ** 1.4
        chunk[:, o + 6] = np.linspace(0, 0.08, 100)
    tokens = codec.transform({"actions_cartesian": chunk})["actions_cartesian"]
    assert tokens.shape == (M, 16)
    for o in (0, 7):
        path = np.linalg.norm(np.diff(tokens[:, o : o + 3], axis=0), axis=1).sum()
        np.testing.assert_allclose(path, D, rtol=1e-3)
    decoder.first_stream = 0
    np.testing.assert_allclose(decoder(tokens)[0], codec.detokenize(tokens, 100))
    assert int(cfg.e1.time_rows) == 100
    assert int(cfg.model.action_horizon) == int(cfg.hpt.action_horizon) == M


@pytest.mark.parametrize(
    "name", ["earbuds_updated_time_hpt300", "towels_since0924plus74_time_hpt300"]
)
def test_time_leaves_keep_the_hpt300_recipe_without_arc_settings(name):
    cfg = compose_recipe(name)
    assert cfg.e1.variant == "time" and cfg.e1.action_dim == 14
    assert cfg.hpt.embed_dim == 840 and int(cfg.model.action_horizon) == 100
    decoder = instantiate(cfg.model.inference.profiles.default.adapter.decoder)
    assert type(decoder).__name__ == "TimeChunkRetimer"
