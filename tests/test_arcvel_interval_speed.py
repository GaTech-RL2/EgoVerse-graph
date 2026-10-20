"""arcvel's interval speed encoding returns arcdur's clock exactly; the point encoding stays decodable."""

import numpy as np
import pytest
from hydra.utils import instantiate

from egomimic.rldb.zarr.e1_arc_tokenizer import TokenizeBimanualArcLengthE1
from egomimic.robot.arc_decoder import BimanualArcDecoder
from scripts.e1.validate_stattempo_configs import compose_recipe


def codec(**kwargs):
    return TokenizeBimanualArcLengthE1(
        min_distance_unit=0.33, resampled_vector_length=200, dt=1 / 30, **kwargs
    )


def chunk():
    t = np.arange(100) / 30
    actions = np.zeros((100, 14))
    rng = np.random.default_rng(0)
    for off in (0, 7):  # move, a jittered pause, move, then a still tail
        actions[:, off] = 0.08 * np.minimum(t, 0.8) + 0.08 * np.clip(t - 1.8, 0, 0.6)
        actions[:, off + 1] = 2e-4 * rng.standard_normal(100)
        actions[:, off + 3] = 0.4 * np.minimum(t, 1.0)
        actions[:, off + 6] = np.clip(t - 0.9, 0, 1)
    return actions


def token(c):
    return c.transform({"actions_cartesian": chunk()})["actions_cartesian"]


def test_interval_speeds_decode_to_the_duration_clock():
    vel, dur = codec(velocity_mode="profile"), codec(velocity_mode="dur")
    assert vel.speed_encoding == "interval"
    tv, td = token(vel), token(dur)
    np.testing.assert_array_equal(tv[:, :14], td[:, :14])
    for a, b in zip(vel.clock_at_waypoints(tv), dur.clock_at_waypoints(td)):
        np.testing.assert_allclose(a, b, atol=1e-12)
    np.testing.assert_allclose(
        vel.detokenize(tv, 100), dur.detokenize(td, 100), atol=1e-10
    )
    point = codec(velocity_mode="profile", speed_encoding="point")
    lost = (
        dur.clock_at_waypoints(td)[0][-1]
        - point.clock_at_waypoints(token(point))[0][-1]
    )
    assert lost > 0.1  # the point encoding drops pause time


def test_new_recipes_declare_interval_and_old_bundles_decode_as_point():
    cfg = compose_recipe("bc_rl2_stattempo_slowpace366_arcvel_m200_d033")
    training = instantiate(
        cfg.data.train_datasets.yam_bimanual.resolver.transform_list
    )[-1]
    decoder = instantiate(cfg.model.inference.profiles.default.adapter.decoder)
    evaluator = instantiate(cfg.evaluator)
    assert training.speed_encoding == "interval"
    assert decoder.codec.speed_encoding == "interval"
    assert (
        evaluator._e1_tokenizer(evaluator.token_layout, None).speed_encoding
        == "interval"
    )
    legacy = BimanualArcDecoder("e1_profile", 0.33, 200, 1 / 30, 100)
    assert legacy.codec.speed_encoding == "point"
    with pytest.raises(ValueError, match="speed_encoding"):
        codec(velocity_mode="profile", speed_encoding="mean")
