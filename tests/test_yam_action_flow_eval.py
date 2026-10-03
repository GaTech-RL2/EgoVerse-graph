import math

import pytest
import torch

from egomimic.eval.yam_action_flow_eval import (
    YAM_ENERGY_DISTANCE_METADATA,
    YAM_NATIVE_ERROR_CONTRACT,
    YamCartesianActionFlowEval,
    yam_native_residual,
)


class IdentityNormalizer:
    def unnormalize(self, values, embodiment_id):
        assert embodiment_id == 7
        assert set(values) == {"actions_cartesian"}
        return values


def evaluator_for_synthetic_chunk():
    evaluator = object.__new__(YamCartesianActionFlowEval)
    evaluator.normalizer = IdentityNormalizer()
    evaluator.action_key = "actions_cartesian"
    return evaluator


def test_yam_evaluator_rejects_other_action_interfaces_and_records_distance():
    evaluator = YamCartesianActionFlowEval(energy_score_enabled=False)
    assert evaluator.action_key == "actions_cartesian"
    assert evaluator.energy_score_distance_metadata == YAM_ENERGY_DISTANCE_METADATA
    with pytest.raises(ValueError, match="actions_cartesian"):
        YamCartesianActionFlowEval(energy_score_enabled=False, action_key="actions")
    with pytest.raises(ValueError, match="private decoder"):
        YamCartesianActionFlowEval(energy_score_enabled=False, native_decoder=object())


def test_yam_native_residual_wraps_both_arms_and_broadcasts():
    prediction = torch.zeros(32, 1, 2, 100, 14)
    target = torch.zeros(1, 32, 2, 100, 14)
    prediction[..., 3] = math.pi - 0.01
    target[..., 3] = -math.pi + 0.01
    prediction[..., 10] = -math.pi + 0.02
    target[..., 10] = math.pi - 0.02
    residual = yam_native_residual(prediction, target)
    assert residual.shape == (32, 32, 2, 100, 14)
    torch.testing.assert_close(residual[..., 3], torch.full_like(residual[..., 3], -0.02), atol=1e-5, rtol=0)
    torch.testing.assert_close(residual[..., 10], torch.full_like(residual[..., 10], 0.04), atol=1e-5, rtol=0)


def test_yam_energy_score_requires_complete_32_by_100_by_14_chunks():
    evaluator = evaluator_for_synthetic_chunk()
    target = torch.zeros(2, 100, 14)
    samples = target.unsqueeze(0).expand(32, -1, -1, -1).clone()
    values = evaluator._energy_values(samples, target, embodiment_id=7, label="yam_bimanual")
    assert values["score"].item() == pytest.approx(0.0)
    assert values["accuracy"].item() == pytest.approx(0.0)
    assert values["diversity"].item() == pytest.approx(0.0)

    samples[..., 0] = 1.0
    values = evaluator._energy_values(samples, target, embodiment_id=7, label="yam_bimanual")
    assert values["accuracy"].item() > 0
    assert values["diversity"].item() == pytest.approx(0.0)
    with pytest.raises(ValueError, match="32"):
        evaluator._energy_values(samples[:31], target, embodiment_id=7)
    with pytest.raises(ValueError, match="complete"):
        evaluator._energy_values(samples[..., :13], target, embodiment_id=7)


def test_yam_action_flow_diagnostic_native_error_wraps_after_unnormalize():
    evaluator = evaluator_for_synthetic_chunk()
    evaluator._action_flow_diagnostics = type(
        "Runner", (), {"native_error_enabled": True, "native_error": YAM_NATIVE_ERROR_CONTRACT}
    )()
    functions = evaluator._action_flow_native_error_fns(
        {"yam_bimanual": {"embodiment": torch.tensor([7])}}
    )
    target = torch.zeros(1, 100, 14)
    prediction = target.clone()
    prediction[..., 3] = math.pi - 0.01
    target[..., 3] = -math.pi + 0.01
    value = functions["yam_bimanual"](prediction, target)
    assert value.shape == (1,)
    assert value.item() == pytest.approx((0.02**2) / 14, abs=1e-6)
