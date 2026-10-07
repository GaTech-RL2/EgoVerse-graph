"""Hardware-free validation of declared shapes against decoder implementation."""

import pytest

from egomimic.robot.arc_decoder import (
    BimanualArcDecoder,
    BimanualIntervalArcDecoder,
    FirstStreamArcDecoder,
    TimeChunkRetimer,
)

M, H, DT = 100, 100, 1 / 30


@pytest.mark.parametrize(
    "make, native",
    [
        (lambda: BimanualArcDecoder("e1_dur"), [M, 16]),
        (lambda: BimanualArcDecoder("e1_durhyb", rotation_distance_unit=0.6), [M, 18]),
        (lambda: BimanualArcDecoder("e1_durtri", rotation_distance_unit=0.6), [M, 20]),
        (
            lambda: BimanualIntervalArcDecoder(
                "per_waypoint", 0.4, M, DT, H, velocity_layout="stacked"
            ),
            [2 * M, 14],
        ),
        (
            lambda: BimanualIntervalArcDecoder(
                "per_waypoint", 0.4, M, DT, H, velocity_layout="wide"
            ),
            [M, 28],
        ),
        (
            lambda: BimanualIntervalArcDecoder(
                "duration",
                0.4,
                M,
                DT,
                H,
                velocity_layout="clock",
                rotation_distance_unit=0.6,
                arc_chunking_mode="multistream",
            ),
            [M, 18],
        ),
        (
            lambda: FirstStreamArcDecoder("per_waypoint", "stacked", 0.4, 0.6),
            [2 * M, 14],
        ),
        (lambda: FirstStreamArcDecoder("per_waypoint", "wide", 0.4, 0.6), [M, 28]),
        (lambda: FirstStreamArcDecoder("duration", "clock", 0.4, 0.6), [M, 18]),
        (lambda: TimeChunkRetimer(), [H, 14]),
    ],
)
def test_contract_preflight_matches_implemented_tensor_shapes_without_mutation(
    make, native
):
    decoder = make()
    before = dict(decoder.__dict__)
    assert decoder.validate_inference_contract(native, [H, 14]) is None
    assert decoder.__dict__ == before
    wrong = [native[0] + 1, native[1]]
    with pytest.raises(ValueError, match="native_shape"):
        decoder.validate_inference_contract(wrong, [H, 14])
    for wrong_output in ([H + 1, 14], [H, 18]):
        with pytest.raises(ValueError, match="canonical_shape"):
            decoder.validate_inference_contract(native, wrong_output)
    assert decoder.__dict__ == before


@pytest.mark.parametrize("mode", ["mean", "per_waypoint"])
def test_clock_width_cannot_be_declared_for_wrong_timing_mode(mode):
    with pytest.raises(ValueError):
        BimanualIntervalArcDecoder(mode, 0.4, M, DT, H, velocity_layout="clock")


@pytest.mark.parametrize("wrong_native", [[M, 18], [2 * M, 14]])
def test_e1_interval_decoder_rejects_shape_that_belongs_to_another_codec(wrong_native):
    decoder = BimanualArcDecoder("e1_dur")
    with pytest.raises(ValueError, match="native_shape"):
        decoder.validate_inference_contract(wrong_native, [H, 14])


@pytest.mark.parametrize("bad", [None, [100], [100, True], [0, 14], [100.0, 14]])
def test_contract_dimensions_must_be_two_positive_integers(bad):
    decoder = TimeChunkRetimer()
    with pytest.raises(ValueError):
        decoder.validate_inference_contract(bad, [H, 14])


@pytest.mark.parametrize("version", ["canonical200", "m28_99be4af0"])
@pytest.mark.parametrize("first_stream", [0, 1])
def test_first_stream_uses_explicit_training_codec_for_asymmetric_arm_rotation(
    version, first_stream
):
    import numpy as np

    from egomimic.rldb.zarr import arc_length_tokenizer as canonical
    from egomimic.rldb.zarr import arc_length_tokenizer_m28 as frozen

    selected = canonical if version == "canonical200" else frozen
    t = np.arange(200)[:, None] * DT
    trajectory = np.zeros((200, 14))
    trajectory[:, 0:3] = t * np.array([0.1, 0.03, 0.0])
    trajectory[:, 7:10] = t * np.array([0.0, 0.06, 0.0])
    # Left remains belowR while right reachesR: no shared rotation budget/clock.
    trajectory[:, 3:6] = t * np.array([0.12, 0.0, 0.0])
    trajectory[:, 10:13] = t * np.array([0.3, 0.0, 0.0])
    trajectory[:, [6, 13]] = 0.5
    encoder = selected.TokenizeBimanualArcLengthCartesian(
        min_distance_unit=0.4,
        rotation_distance_unit=1.0,
        resampled_vector_length=M,
        dt=DT,
        velocity_mode="per_waypoint",
        velocity_layout="wide",
        arc_chunking_mode="multistream",
    )
    token = encoder.transform({"actions_cartesian": trajectory})["actions_cartesian"]
    decoder = FirstStreamArcDecoder(
        "per_waypoint",
        "wide",
        0.4,
        1.0,
        M,
        DT,
        H,
        first_stream=first_stream,
        execute_percent=50,
        codec_version=version,
    )
    assert decoder.codec_module is selected
    assert decoder.codec.__class__.__module__ == selected.__name__
    decoded = decoder(token)[0]
    expected = encoder.detokenize(token, H)
    n = decoder.execution_steps() or H
    np.testing.assert_allclose(decoded[:n], expected[:n], rtol=0, atol=1e-12)
    if not first_stream:
        np.testing.assert_allclose(decoded[:, 3], 0.12 * np.arange(H) * DT, atol=1e-12)
        np.testing.assert_allclose(decoded[:, 10], 0.3 * np.arange(H) * DT, atol=1e-12)


def test_current_first_stream_contract_does_not_implicitly_choose_frozen_m28():
    from egomimic.rldb.zarr import arc_length_tokenizer as canonical

    decoder = FirstStreamArcDecoder("per_waypoint", "wide", 0.4, 0.6)
    assert decoder.codec_version == "canonical200" and decoder.codec_module is canonical
    with pytest.raises(ValueError, match="codec_version"):
        FirstStreamArcDecoder(
            "per_waypoint", "wide", 0.4, 0.6, codec_version="by_shape"
        )


@pytest.mark.parametrize(
    "layout,version",
    [
        ("e1_durhyb", "m28_99be4af0"),
        ("cartesian_duration", "e1_8ff"),
        ("lab_pw_stacked", "canonical200"),
        ("e1_durtri", "e1_8ff"),
    ],
)
def test_shape_colliding_codec_versions_are_incompatible_with_named_layout(
    layout, version
):
    with pytest.raises(ValueError, match="incompatible with token_layout"):
        BimanualArcDecoder(layout, codec_version=version)


@pytest.mark.parametrize("chunking", ["joint_distance", "race", "multistream"])
@pytest.mark.parametrize("version", ["canonical200", "m28_99be4af0"])
def test_cartesian_interval_decoder_uses_declared_current_or_historical_codec(
    chunking, version
):
    import numpy as np

    from egomimic.rldb.zarr import arc_length_tokenizer as canonical
    from egomimic.rldb.zarr import arc_length_tokenizer_m28 as frozen

    selected = canonical if version == "canonical200" else frozen
    t = np.arange(200)[:, None] * DT
    trajectory = np.zeros((200, 14))
    trajectory[:, 0:3] = t * np.array([0.1, 0.03, 0.0])
    trajectory[:, 7:10] = t * np.array([0.0, 0.06, 0.0])
    trajectory[:, 3:6] = t * np.array([0.12, 0.0, 0.0])
    trajectory[:, 10:13] = t * np.array([0.3, 0.0, 0.0])
    trajectory[:, [6, 13]] = 0.5
    encoder = selected.TokenizeBimanualArcLengthCartesian(
        min_distance_unit=0.4,
        rotation_distance_unit=1.0,
        resampled_vector_length=M,
        dt=DT,
        velocity_mode="per_waypoint",
        velocity_layout="stacked",
        arc_chunking_mode=chunking,
    )
    token = encoder.transform({"actions_cartesian": trajectory})["actions_cartesian"]
    decoder = BimanualArcDecoder(
        "cartesian_per_waypoint",
        0.4,
        M,
        DT,
        H,
        rotation_distance_unit=1.0,
        arc_chunking_mode=chunking,
        velocity_layout="stacked",
        codec_version=version,
    )
    assert decoder.codec.__class__.__module__ == selected.__name__
    np.testing.assert_array_equal(decoder(token)[0], encoder.detokenize(token, H))
    assert decoder.last_stats["codec_version"] == version


def test_e1_and_pr193_compatibility_have_distinct_explicit_identities():
    assert BimanualArcDecoder("e1_durhyb").codec_version == "e1_8ff"
    assert BimanualArcDecoder("e1_durtri").codec_version == "e1_tri_b1ecba63"
    decoder = BimanualArcDecoder("lab_pw_stacked", codec_version="pr193")
    assert decoder.codec_version == "pr193" and decoder.shape == (2 * M, 14)
