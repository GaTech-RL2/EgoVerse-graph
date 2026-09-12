from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from egomimic.robot.arc_decoder import ARC_TOKEN_LAYOUTS, BimanualArcDecoder


@pytest.mark.parametrize("layout", ARC_TOKEN_LAYOUTS)
def test_decoder_matches_source_codec(layout):
    decoder = BimanualArcDecoder(layout, resampled_vector_length=20, action_horizon=40)
    values = np.zeros(decoder.shape)
    t = np.linspace(0, 1, 20)
    values[:20, 0] = 0.3 * t
    values[:20, 7] = 0.2 * t
    if layout == "lab":
        values[-1, [0, 7]] = [0.3, 0.2]
    elif layout == "e1_dur":
        values[1:, 14:] = 1/30
    elif layout == "e1_logdur":
        values[:, 14:] = np.log(1/30)
    else:
        values[:, 14:] = 0.3
    expected = decoder.codec.detokenize(values, action_horizon=40)
    np.testing.assert_array_equal(decoder(values)[0], expected)
    with pytest.raises(ValueError):
        decoder(values[:-1])
    values[0, 0] = np.nan
    with pytest.raises(ValueError):
        decoder(values)


@pytest.mark.parametrize("experiment", ["abc_stationery_bc", "abc_stationery_arc_bc", "shorts_extreme_bc", "shorts_extreme_arc_bc", "shorts_extreme_arcdur"])
def test_robot_campaign_composes_and_preserves_horizons(experiment):
    root = Path(__file__).parents[1]/"egomimic/hydra_configs"
    with initialize_config_dir(version_base=None, config_dir=str(root)):
        cfg = compose(config_name="train_zarr_cartesian", overrides=[
            f"+experiment=abc_arc/{experiment}",
            "paths.output_dir=/tmp/robot-campaign"])
    instantiate(cfg.evaluator)
    expected = (100, 16) if "arcdur" in experiment else (101, 14) if "arc_bc" in experiment else (100, 14)
    assert (cfg.e1.action_horizon, cfg.e1.action_dim) == expected
    for group in [cfg.data.train_datasets, cfg.data.valid_datasets]:
        for ds in group.values():
            instantiate(ds.resolver.key_map)
            instantiate(ds.resolver.transform_list)
            assert ds.resolver.embodiment_override == "yam_bimanual"
    # Explicit disjoint shorts lists are 200 train episodes and 30 validation episodes.
    if experiment.startswith("shorts"):
        import ast
        sets = []
        for group in [cfg.data.train_datasets, cfg.data.valid_datasets]:
            expr = next(iter(group.values())).filters.filter_lambdas[0]
            tree = ast.parse(expr, mode="eval")
            sets.append(ast.literal_eval(tree.body.args.defaults[0].args[0]))
        assert len(sets[0]) == 200 and len(sets[1]) == 30
        assert sets[0].isdisjoint(sets[1])
