from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from egomimic.rldb.zarr.e1_arc_tokenizer import speed_columns_to_durations
from egomimic.robot.arc_decoder import ARC_TOKEN_LAYOUTS, BimanualArcDecoder


@pytest.mark.parametrize("layout", ARC_TOKEN_LAYOUTS)
def test_decoder_matches_source_codec(layout):
    # Whole-token decode: the codec comparison below is not about chunk termination.
    decoder = BimanualArcDecoder(
        layout, resampled_vector_length=20, action_horizon=40, first_stream=0
    )
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
    elif layout == "e1_profile":
        values[:, 14:] = 0.3
    elif layout == "cartesian_per_waypoint":
        values[20:, [0, 7]] = 0.3
        values[20:, [6, 13]] = 0.0
    elif layout == "cartesian_duration":
        values[20:, [0, 7]] = 1 / 30
    source = values
    if layout in ("e1_profhyb", "e1_proftri"):
        # Speed columns: the decoder converts them to durations on the whole token
        # first, and its codec runs in the matching duration mode.
        source = speed_columns_to_durations(
            values,
            tri=layout == "e1_proftri",
            hold_time=39 / 30,
            eps=decoder.codec.tokenizer.config.zero_dist_epsilon,
        )
    expected = decoder.codec.detokenize(source, action_horizon=40)
    np.testing.assert_array_equal(decoder(values)[0], expected)
    with pytest.raises(ValueError):
        decoder(values[:-1])
    values[0, 0] = np.nan
    with pytest.raises(ValueError):
        decoder(values)


@pytest.mark.parametrize("chunking_mode", ["joint_distance", "race", "multistream"])
def test_hybrid_decoder_supports_pr177_chunking_modes(chunking_mode):
    decoder = BimanualArcDecoder(
        "cartesian_per_waypoint",
        min_distance_unit=0.4,
        resampled_vector_length=20,
        dt=1 / 30,
        action_horizon=40,
        rotation_distance_unit=0.4,
        arc_chunking_mode=chunking_mode,
    )
    values = np.zeros(decoder.shape)
    t = np.linspace(0, 1, 20)
    values[:20, 0] = 0.4 * t
    values[:20, 7] = 0.2 * t
    values[20:, 0] = 0.4 * 19 / 30
    values[20:, 7] = 0.2 * 19 / 30
    result = decoder(values)
    assert result.shape == (1, 40, 14)
    assert np.isfinite(result).all()



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
