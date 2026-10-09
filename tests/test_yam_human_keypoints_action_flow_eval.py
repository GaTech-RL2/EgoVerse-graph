"""Fixed YAM/human validation contract checks without a training allocation."""

import torch

from egomimic.eval.yam_human_keypoints_action_flow_eval import (
    HUMAN_ACTION_DIM,
    HUMAN_WRIST_ANGLE_INDICES,
    YamHumanKeypointsActionFlowEval,
    human_native_residual,
)


class _IdentityNormalizer:
    def __init__(self):
        self.keys = []

    def unnormalize(self, batch, embodiment_id):
        self.keys.append((int(embodiment_id), tuple(batch)))
        return batch


def test_action_keys_are_private_and_fail_closed():
    evaluator = YamHumanKeypointsActionFlowEval(energy_score_enabled=False)
    normalizer = _IdentityNormalizer()
    evaluator.bind_data_context(normalizer=normalizer)
    assert evaluator._action_key_for_id(7) == "actions_cartesian"
    assert evaluator._action_key_for_id(3) == "actions_keypoints"
    assert evaluator._native(torch.zeros(1, 100, 14), 7, None).shape == (1, 100, 14)
    assert evaluator._native(torch.zeros(1, 100, 138), 3, None).shape == (1, 100, 138)
    assert normalizer.keys == [(7, ("actions_cartesian",)), (3, ("actions_keypoints",))]


def test_human_native_wrist_angles_wrap_without_changing_keypoint_residual():
    target = torch.zeros(1, 100, HUMAN_ACTION_DIM)
    prediction = target.clone()
    prediction[..., list(HUMAN_WRIST_ANGLE_INDICES)] = 2 * torch.pi - 0.1
    prediction[..., 6] = 2.0
    residual = human_native_residual(prediction, target)
    assert torch.allclose(residual[..., list(HUMAN_WRIST_ANGLE_INDICES)], torch.full((1, 100, 6), -0.1), atol=1e-5)
    assert torch.all(residual[..., 6] == 2.0)


def test_energy_score_32_is_finite_for_both_native_action_layouts():
    evaluator = YamHumanKeypointsActionFlowEval(energy_score_enabled=False)
    evaluator.bind_data_context(normalizer=_IdentityNormalizer())
    for label, embodiment_id, width in (("yam_bimanual", 7, 14), ("human_bimanual", 3, 138)):
        target = torch.zeros(2, 100, width)
        samples = torch.linspace(-0.1, 0.1, 32).view(32, 1, 1, 1).expand(32, 2, 100, width).clone()
        values = evaluator._energy_values(samples, target, embodiment_id, label)
        assert all(torch.isfinite(value).all() for value in values.values())
        assert values["score_by_condition"].shape == (2,)


def test_validation_groups_have_distinct_artifact_paths_and_diagnostic_budgets(tmp_path):
    class Diagnostic:
        def __init__(self):
            self.artifact_root = tmp_path / "diag"
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1

    evaluator = YamHumanKeypointsActionFlowEval(energy_score_enabled=False)
    diagnostic = Diagnostic()
    evaluator._co_train_artifact_root = tmp_path / "energy"
    evaluator._co_train_diagnostic_root = diagnostic.artifact_root
    evaluator._action_flow_diagnostics = diagnostic

    evaluator.set_validation_group("yam")
    yam_energy = evaluator.artifact_root
    yam_diagnostic = diagnostic.artifact_root
    assert (yam_energy, yam_diagnostic) == (tmp_path / "energy/yam", tmp_path / "diag/yam")
    assert diagnostic.reset_count == 1
    evaluator.set_validation_group("yam")
    assert diagnostic.reset_count == 1

    evaluator.set_validation_group("human")
    assert evaluator.artifact_root == tmp_path / "energy/human"
    assert diagnostic.artifact_root == tmp_path / "diag/human"
    assert evaluator.artifact_root != yam_energy
    assert diagnostic.artifact_root != yam_diagnostic
    assert diagnostic.reset_count == 2
