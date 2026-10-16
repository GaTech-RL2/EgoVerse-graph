"""The d03 leaf changes exactly one tokenizer setting against the earbuds absolute-time recipe, everywhere it is read."""
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from scripts.e1.validate_stattempo_configs import compose_recipe

KEY, VALUE = "min_distance_unit", 0.3


def flat(d, p=""):
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            out.update(flat(v, f"{p}.{k}" if p else str(k)))
        return out
    if isinstance(d, list):
        out = {}
        for i, v in enumerate(d):
            out.update(flat(v, f"{p}[{i}]"))
        return out
    return {p: d}


def test_leaf_changes_only_the_tokenizer_setting():
    old = compose_recipe("earbuds_updated_arcdur_abstime_hpt300")
    new = compose_recipe("earbuds_updated_arcdur_abstime_d03_hpt300")
    a, b = [flat(OmegaConf.to_container(c, resolve=True)) for c in (old, new)]
    changed = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    expected_old = 100 if KEY == "resampled_vector_length" else 0.4
    for k in changed:
        if k.startswith(("logger.wandb", "name")) or "abstime_hpt300" in str(a.get(k)):  # run-name strings
            continue
        assert a.get(k) in (expected_old, None) and b.get(k) == VALUE, (k, a.get(k), b.get(k))
    for split in ("train_datasets", "valid_datasets"):
        assert new.data[split].yam_bimanual.resolver.transform_list[KEY] == VALUE
    assert new.arc_tokenizer[KEY] == VALUE


def test_targets_evaluator_and_decoder_agree():
    cfg = compose_recipe("earbuds_updated_arcdur_abstime_d03_hpt300")
    M, D = int(cfg.e1.M), float(cfg.e1.D)
    for split in ("train_datasets", "valid_datasets"):
        codec = instantiate(cfg.data[split].yam_bimanual.resolver.transform_list)[-1]
        assert codec.M == M and codec.tokenizer.config.min_distance_unit == D
        assert codec.duration_timing == "absolute"
    evaluator = instantiate(cfg.evaluator)
    assert (evaluator.metric_options["M"], evaluator.metric_options["D"]) == (M, D)
    decoder = instantiate(cfg.model.inference.profiles.default.adapter.decoder)
    assert decoder.codec.M == M and decoder.codec.tokenizer.config.min_distance_unit == D
    chunk = np.zeros((100, 14))
    for o in (0, 7):  # 0.5 m of path per arm: longer than either D
        chunk[:, o] = 0.5 * np.linspace(0, 1, 100) ** 1.4
        chunk[:, o + 6] = np.linspace(0, 0.08, 100)
    tokens = codec.transform({"actions_cartesian": chunk})["actions_cartesian"]
    assert tokens.shape == (M, 16)
    for o in (0, 7):
        path = np.linalg.norm(np.diff(tokens[:, o:o + 3], axis=0), axis=1).sum()
        np.testing.assert_allclose(path, D, rtol=1e-3)
    np.testing.assert_allclose(decoder(tokens)[0], codec.detokenize(tokens, 100))
    assert decoder(tokens)[0].shape[0] == int(cfg.e1.time_rows) == 100
    assert int(cfg.model.action_horizon) == int(cfg.hpt.action_horizon) == M
