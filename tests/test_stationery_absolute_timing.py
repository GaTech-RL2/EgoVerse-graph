"""Absolute waypoint clocks must agree across targets, evaluation and rollout."""

import numpy as np
import pytest
from hydra.utils import instantiate

from egomimic.rldb.zarr.e1_arc_tokenizer import (
    TokenizeBimanualArcLengthE1,
    absolute_times_to_clock,
)
from scripts.e1.validate_stattempo_configs import compose_recipe


def codec(**kwargs):
    return TokenizeBimanualArcLengthE1(
        velocity_mode="dur",
        min_distance_unit=0.4,
        resampled_vector_length=100,
        dt=1 / 30,
        **kwargs,
    )


@pytest.mark.parametrize("fixed_spacing", [False, True])
@pytest.mark.parametrize("idle_right", [False, True])
def test_true_tokens_preserve_delta_clock_and_decoded_motion(fixed_spacing, idle_right):
    chunk = np.zeros((100, 14))
    for offset in (0, 7):
        if offset == 7 and idle_right:
            continue
        chunk[:, offset] = 0.3 * np.linspace(0, 1, 100) ** 1.4
        chunk[:, offset + 3] = np.linspace(0, 0.2, 100)
        chunk[:, offset + 6] = np.linspace(0, 0.08, 100)
    old = codec(fixed_spacing=fixed_spacing)
    new = codec(fixed_spacing=fixed_spacing, duration_timing="absolute")
    a = old.transform({"actions_cartesian": chunk.copy()})["actions_cartesian"]
    b = new.transform({"actions_cartesian": chunk.copy()})["actions_cartesian"]
    np.testing.assert_array_equal(a[:, :14], b[:, :14])
    np.testing.assert_allclose(b[0, 14:], 0)
    np.testing.assert_allclose(
        old.clock_at_waypoints(a), new.clock_at_waypoints(b), atol=1e-12
    )
    np.testing.assert_allclose(
        old.detokenize(a, 100), new.detokenize(b, 100), atol=1e-10
    )


def test_local_time_error_does_not_accumulate_and_padding_is_ignored():
    arc = np.linspace(0, 0.4, 5)
    truth = np.arange(5) * 0.1
    predicted = truth.copy()
    predicted[2] += 0.03
    error = absolute_times_to_clock(predicted, arc) - truth
    np.testing.assert_allclose(error, [0, 0, 0.03, 0, 0], atol=1e-12)
    np.testing.assert_allclose(
        absolute_times_to_clock([5, 0.1, -1, 0.3, 99], [0, 0.1, 0.2, 0.3, 0.3]),
        [0, 0.02, 0.04, 0.3, 0.3],
    )
    # Average the backwards pair's slack above the fastest possible clock.
    # The repaired waypoints retain enough elapsed time to traverse 10 cm.
    np.testing.assert_allclose(
        absolute_times_to_clock([0, 0.1, 0.35, 0.3, 0.4], arc),
        [0, 0.1, 0.315, 0.335, 0.4],
    )
    with pytest.raises(ValueError, match="finite"):
        absolute_times_to_clock([0, np.nan], [0, 0.1])


@pytest.mark.parametrize(
    "predicted",
    [
        [0, 0.1, 0.35, 0.3, 0.4],  # positive backwards pair
        [0, -0.1, -0.2, 0.3, 0.4],  # previously skipped two supports at t=0
        [0, 0, 0, 0, 0],  # previously started at the final pose
    ],
)
def test_repaired_absolute_clocks_preserve_anchor_and_bound_decoded_motion(predicted):
    c = codec(duration_timing="absolute")
    values = np.zeros((5, 16))
    values[:, 0] = np.arange(5) * 0.1
    values[:, 14] = predicted
    clock = c.clock_at_waypoints(values)[0]
    np.testing.assert_allclose(clock[0], 0.0)
    assert np.all(np.diff(clock) >= 0.1 / 5.0 - 1e-14)

    # Check control-rate output and a dense grid around the former 0.325 s
    # discontinuity. Checking only monotonic clock values misses teleportation.
    times = np.unique(
        np.r_[np.arange(16) / 30, np.arange(0.30, 0.351, 0.001), 0.324999, 0.325]
    )
    decoded = c.detokenize(values, len(times), times=times)
    np.testing.assert_array_equal(decoded[0], values[0, :14])
    assert np.all(np.diff(decoded[:, 0]) >= -1e-14)
    assert np.all(np.diff(decoded[:, 0]) <= 5.0 * np.diff(times) + 1e-12)
    np.testing.assert_allclose(decoded[-1, 0], 0.4)
    np.testing.assert_array_equal(decoded[:, 7:], 0.0)


def test_absolute_total_duration_bound_preserves_local_speed_and_padding():
    # Uniformly shrinking this clock to its 20 s total limit would turn the
    # first 10 cm / 20 ms interval into a 25 m/s segment.
    clock = absolute_times_to_clock(
        [8, 0.02, 100, 999], [0, 0.1, 0.2, 0.2]
    )
    np.testing.assert_allclose(clock, [0, 0.02, 20, 20])
    np.testing.assert_allclose(
        absolute_times_to_clock([8, 0.1, 999, 0.3, 999], [0, 0.1, 0.1, 0.2, 0.2]),
        [0, 0.1, 0.1, 0.3, 0.3],
    )
    np.testing.assert_array_equal(absolute_times_to_clock([9, 10], [0, 0]), [0, 0])


@pytest.mark.parametrize(
    "arc,kwargs",
    [
        ([1, 2], {}),
        ([0, -1], {}),
        ([0, 1], {"min_speed": 0}),
        ([0, 1], {"max_speed": 0}),
        ([0, 1], {"min_speed": 6, "max_speed": 5}),
        ([0, 1], {"max_speed": np.inf}),
    ],
)
def test_absolute_clock_rejects_invalid_geometry_and_speed_bounds(arc, kwargs):
    with pytest.raises(ValueError):
        absolute_times_to_clock([0, 1], arc, **kwargs)


def test_invalid_mode_cannot_silently_use_absolute_times():
    with pytest.raises(ValueError, match="requires velocity_mode"):
        TokenizeBimanualArcLengthE1(velocity_mode="profile", duration_timing="absolute")
    with pytest.raises(ValueError, match="duration_timing"):
        codec(duration_timing="typo")


def test_stationery_open_loop_evaluator_decodes_absolute_times():
    from egomimic.eval.open_loop_sim import truncate_e1_wide_token

    cfg = compose_recipe("bc_rl2_stattempo_slowpace366_arcdur_abstime_m200_d033")
    for split in ("train_datasets", "valid_datasets"):
        transforms = instantiate(cfg.data[split].yam_bimanual.resolver.transform_list)
        assert transforms[-1].duration_timing == "absolute"
    evaluator = instantiate(cfg.evaluator)
    chunk = np.zeros((100, 14))
    chunk[:, 0] = 0.3 * np.linspace(0, 1, 100) ** 1.4
    chunk[:, 7] = np.linspace(0, 0.2, 100)
    tokens = transforms[-1].transform({"actions_cartesian": chunk})["actions_cartesian"]
    decoded, steps = evaluator._decode_prediction_with_steps(tokens)
    assert evaluator._arc_tokenizer.duration_timing == "absolute"
    expected = transforms[-1].detokenize(
        truncate_e1_wide_token(tokens, evaluator.execute_fraction), action_horizon=steps
    )
    np.testing.assert_allclose(decoded, expected)
    decoder = instantiate(cfg.model.inference.profiles.default.adapter.decoder)
    assert decoder.codec.duration_timing == "absolute"
    with pytest.raises(Exception, match="e1_dur"):  # Hydra wraps the ValueError
        instantiate(cfg.evaluator, token_layout="e1_profile")
