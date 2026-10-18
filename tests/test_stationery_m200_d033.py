"""The six slow-pace 366 M 200 / D 0.33 leaves change only M and D against their parents, everywhere they are read."""

import numpy as np
import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from scripts.e1.validate_stattempo_configs import compose_recipe

P = "bc_rl2_stattempo_slowpace366_"
VARIANTS = (
    "arcdur",
    "arcdur_abstime",
    "arcvel",
    "arcvel_nosmooth",
    "arc_projection_dur",
    "arc_projection_vel",
)


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


@pytest.mark.parametrize("variant", VARIANTS)
def test_leaf_changes_only_m_and_d(variant):
    old, new = compose_recipe(P + variant), compose_recipe(P + variant + "_m200_d033")
    a, b = [flat(OmegaConf.to_container(c, resolve=True)) for c in (old, new)]
    for k in {k for k in set(a) | set(b) if a.get(k) != b.get(k)}:
        if (
            k.startswith(("logger.wandb", "name", "description"))
            or k == "evaluator.tick_prefix"
        ):
            continue
        assert (a.get(k), b.get(k)) in ((100, 200), (0.4, 0.33)), (
            k,
            a.get(k),
            b.get(k),
        )


@pytest.mark.parametrize("variant", VARIANTS)
def test_targets_evaluator_and_decoder_agree(variant):
    cfg = compose_recipe(P + variant + "_m200_d033")
    M, D = int(cfg.e1.M), float(cfg.e1.D)
    assert (M, D) == (200, 0.33)
    for split in ("train_datasets", "valid_datasets"):
        codec = instantiate(cfg.data[split].yam_bimanual.resolver.transform_list)[-1]
        assert codec.M == M and codec.tokenizer.config.min_distance_unit == D
    evaluator = instantiate(cfg.evaluator)
    assert (evaluator.resampled_vector_length, evaluator.min_distance_unit) == (M, D)
    assert evaluator.tick_prefix
    decoder = instantiate(cfg.model.inference.profiles.default.adapter.decoder)
    assert (
        decoder.codec.M == M and decoder.codec.tokenizer.config.min_distance_unit == D
    )
    assert int(cfg.model.action_horizon) == int(cfg.hpt.action_horizon) == M
    chunk = np.zeros((100, 14))
    for o in (0, 7):  # 0.5 m of path per arm: longer than D
        chunk[:, o] = 0.5 * np.linspace(0, 1, 100) ** 1.4
        chunk[:, o + 6] = np.linspace(0, 0.08, 100)
    tokens = codec.transform({"actions_cartesian": chunk})["actions_cartesian"]
    assert tokens.shape == (M, 16)
    if (
        "projection" not in variant
    ):  # projection spends D on the offset point, not on xyz
        for o in (0, 7):
            path = np.linalg.norm(np.diff(tokens[:, o : o + 3], axis=0), axis=1).sum()
            np.testing.assert_allclose(path, D, rtol=1e-3)
    np.testing.assert_allclose(decoder(tokens)[0], codec.detokenize(tokens, 100))
    assert decoder(tokens)[0].shape[0] == int(cfg.e1.time_rows) == 100
