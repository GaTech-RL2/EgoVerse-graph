"""Compatibility checks for the E1 codecs carried onto the graph ARC stack."""

import numpy as np
import pytest

from egomimic.rldb.embodiment.e1_fold import get_transform_list
from egomimic.rldb.zarr.arc_length_tokenizer import TokenizeBimanualArcLengthCartesian
from egomimic.rldb.zarr.e1_arc_tokenizer import (
    TokenizeBimanualArcLengthE1,
    durations_to_clock_abs,
)


def _codec(mode, **kwargs):
    return TokenizeBimanualArcLengthE1(
        min_distance_unit=0.4,
        resampled_vector_length=61,
        dt=1 / 30,
        velocity_mode=mode,
        **kwargs,
    )


def _chunk(*, curved=False, right_idle=False):
    t = np.arange(181) / 30
    chunk = np.zeros((len(t), 14))
    for offset in (0, 7):
        if offset == 7 and right_idle:
            continue
        chunk[:, offset] = 0.1 * t
        chunk[:, offset + 6] = 0.05 * t
        if curved:
            chunk[:, offset + 1] = 0.04 * np.sin(t)
            chunk[:, offset + 3] = 0.08 * t
    return chunk


@pytest.mark.parametrize("mode", ["mean", "profile", "logdur", "dur"])
def test_e1_layouts_recover_a_constant_speed_trajectory(mode):
    codec = _codec(mode)
    chunk = _chunk()
    token = codec.transform({"actions_cartesian": chunk.copy()})["actions_cartesian"]
    assert token.shape == ((62, 14) if mode == "mean" else (61, 16))
    decoded = codec.detokenize(token, action_horizon=40)
    np.testing.assert_allclose(decoded, chunk[:40], atol=1e-10, rtol=0)


def test_e1_chord_tokens_and_decoder_match_the_current_lab_codec():
    parent = TokenizeBimanualArcLengthCartesian(
        min_distance_unit=0.4, resampled_vector_length=61, dt=1 / 30
    )
    e1 = _codec("mean", velocity_norm="chord")
    chunk = _chunk(curved=True)
    reference = parent.transform({"actions_cartesian": chunk.copy()})[
        "actions_cartesian"
    ]
    token = e1.transform({"actions_cartesian": chunk.copy()})["actions_cartesian"]
    np.testing.assert_allclose(token, reference, atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        e1.detokenize(token, action_horizon=40),
        parent.detokenize(reference, action_horizon=40),
        atol=1e-10,
        rtol=0,
    )


@pytest.mark.parametrize("fixed_spacing", [False, True])
@pytest.mark.parametrize("right_idle", [False, True])
def test_absolute_and_log_durations_preserve_the_same_clock(fixed_spacing, right_idle):
    # Vary tempo along a straight path: its resampled polyline has exactly the
    # source span, so this isolates the timing representation from geometric
    # discretization error on curved paths.
    chunk = _chunk(right_idle=right_idle)
    for offset in (0, 7):
        if offset != 7 or not right_idle:
            chunk[:, offset] = 0.6 * np.linspace(0, 1, len(chunk)) ** 1.25
    decoders = [_codec(mode, fixed_spacing=fixed_spacing) for mode in ("dur", "logdur")]
    tokens = [
        c.transform({"actions_cartesian": chunk.copy()})["actions_cartesian"]
        for c in decoders
    ]
    decoded = [c.detokenize(t, action_horizon=40) for c, t in zip(decoders, tokens)]
    np.testing.assert_allclose(decoded[0], decoded[1], atol=1e-10, rtol=0)
    assert all(np.isfinite(t).all() for t in tokens)
    if right_idle:
        np.testing.assert_array_equal(decoded[0][:, 7:], 0.0)


def test_absolute_clock_ignores_padding_and_negative_durations():
    clock = durations_to_clock_abs(
        np.array([0.1, 0.2, -0.5, 999.0]), np.array([0.0, 0.1, 0.2, 0.2])
    )
    np.testing.assert_allclose(clock, [0.0, 0.1, 0.3, 0.3], atol=1e-12)


@pytest.mark.parametrize("embodiment", ["human", "yam"])
@pytest.mark.parametrize(
    "variant", ["time", "arcmean", "arcvel", "arclogdur", "arcdur"]
)
def test_e1_transforms_compose_with_the_current_embodiments(embodiment, variant):
    transforms = get_transform_list(variant, chunk_length=200, embodiment=embodiment)
    codecs = [t for t in transforms if isinstance(t, TokenizeBimanualArcLengthE1)]
    assert len(codecs) == (0 if variant == "time" else 1)
    copy = [t for t in transforms if getattr(t, "dst", None) == "actions_time"]
    assert len(copy) == 1
    assert copy[0].n_rows == 100
