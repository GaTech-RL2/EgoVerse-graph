"""open_loop_sim on the E1 hybrid (M, 18) tokens: the executed prefix keeps each stream's start delay."""

import numpy as np

from egomimic.eval.open_loop_sim import truncate_e1_wide_token
from egomimic.rldb.zarr.e1_arc_tokenizer import TokenizeBimanualArcLengthE1

DT = 1 / 30


def _slow_chunk():
    """Both arms translate 0.06 m slowly over 100 frames; the left wrist starts yawing 30 deg at frame 10."""
    T = 100
    c = np.zeros((T, 14))
    c[:, 0] = c[:, 7] = 0.06 * np.arange(T) / (T - 1)
    c[:, 3] = np.deg2rad(30) * np.clip((np.arange(T) - 10) / 80.0, 0, 1)
    c[:, 6] = np.clip(np.arange(T) / 50.0, 0, 1)
    return c


def test_hybrid_prefix_keeps_start_delay_and_matches_the_full_decode():
    for mode in ("durhyb", "profhyb"):
        tok = TokenizeBimanualArcLengthE1(
            action_key="a",
            output_action_key="a",
            min_distance_unit=0.40,
            resampled_vector_length=100,
            dt=DT,
            velocity_norm="path",
            velocity_mode=mode,
        )
        token = tok.transform({"a": _slow_chunk()})["a"]
        part = truncate_e1_wide_token(token, 0.25, hybrid=True)
        assert part.shape == (25, 18)
        assert np.allclose(part[24, 14:], token[99, 14:])  # start delays carried over
        full = tok.detokenize(token, action_horizon=100)[:25]
        prefix = tok.detokenize(part, action_horizon=25)
        # A slow token: 25 executed steps stay inside the first 25 % of each stream, so eval decode = deploy decode.
        assert np.abs(prefix[:, :3] - full[:, :3]).max() < 1e-6
        assert np.abs(prefix[:, 3] - full[:, 3]).max() < np.deg2rad(0.1)
