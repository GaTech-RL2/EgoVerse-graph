"""The shared metric/video adapter must never treat timing as geometry."""

import numpy as np
import pytest
import torch

from egomimic.eval.arc_bimanual_cartesian_eval import ArcBimanualCartesianEval


@pytest.mark.parametrize("clock", ["velocity", "duration"])
@pytest.mark.parametrize(
    "streams,dim", [("full_separation", 20), ("translation_rotation", 18)]
)
def test_overlay_uses_exact_shared_codec(clock, streams, dim):
    evaluator = ArcBimanualCartesianEval(
        clock_mode=clock,
        streams=streams,
        min_distance_unit=0.4,
        arcmatch_distance=0.4,
        resampled_vector_length=20,
        action_horizon=50,
    )
    raw = np.zeros((120, 14))
    t = np.arange(120) / 30
    raw[:, 0], raw[:, 7] = 0.2 * t, 0.1 * t
    raw[:, 3], raw[:, 10] = 0.1 * t, 0.2 * t
    token = evaluator._tokenizer.transform({"actions_cartesian": raw})[
        "actions_cartesian"
    ]
    assert token.shape == (20, dim)
    actual = evaluator._viz_source(torch.tensor(token[None]), 7)
    expected = evaluator._tokenizer.detokenize(token, 50)
    assert actual.shape == (1, 50, 14)
    np.testing.assert_allclose(actual.numpy()[0], expected)
    baseline = torch.tensor(raw[None])
    assert not evaluator._is_arc(baseline)
    np.testing.assert_array_equal(evaluator._viz_source(baseline, 7), baseline)
    with pytest.raises(ValueError, match="token shape"):
        evaluator._viz_source(torch.zeros(1, 20, 28), 7)
