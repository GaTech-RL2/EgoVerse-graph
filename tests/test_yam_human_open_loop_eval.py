from types import SimpleNamespace

import pytest
import torch

from egomimic.eval.yam_human_open_loop_eval import YamHumanOpenLoopEval


class IdentityNormalizer:
    def unnormalize(self, batch, embodiment_id):
        return batch


def evaluator():
    value = YamHumanOpenLoopEval(limit_val_batches=1.0, energy_score_enabled=False)
    value.bind_data_context(normalizer=IdentityNormalizer())
    value.on_validation_start()
    value.set_validation_group("yam")
    return value


def batch(width=14, embodiment=7):
    action_key = "actions_cartesian" if width == 14 else "actions_keypoints"
    return {
        "source": {
            "embodiment": torch.full((6,), embodiment),
            action_key: torch.zeros(6, 100, width),
            "episode_hash": ["episode-a"] * 6,
            "frame_index": torch.arange(6),
            "retiming_rate": torch.ones(6),
        }
    }


def test_reuses_predictions_and_scores_robot_episode_with_existing_implementation():
    value = evaluator()
    value._collect_validation_predictions(
        batch(), {"source": {"pred_action": torch.ones(6, 100, 14)}}
    )
    result = value.on_validation_end()
    assert result["execute_control_steps"] == 25
    assert result["executed_control_steps"] == 6
    assert result["per_group"]["yam"]["micro"]["mse"] == 1.0
    assert result["episodes"] == 1


def test_human_resampled_clock_is_not_scored_as_robot_control_frequency():
    value = evaluator()
    value.set_validation_group("human")
    value._collect_validation_predictions(
        batch(138, 3), {"source": {"pred_action": torch.ones(6, 100, 138)}}
    )
    assert value.robot_open_loop._records == []


@pytest.mark.parametrize("limit", [1, 8, None, 0.5])
def test_rejects_truncated_evaluator_validation(limit):
    with pytest.raises(ValueError, match="limit_val_batches"):
        YamHumanOpenLoopEval(limit_val_batches=limit, energy_score_enabled=False)


def test_rejects_trainer_truncation_and_missing_episode_frames():
    value = evaluator()
    value.trainer = SimpleNamespace(limit_val_batches=256)
    with pytest.raises(ValueError, match="trainer.limit_val_batches"):
        value.on_validation_start()
    value.trainer = None
    source = batch()
    source["source"]["frame_index"] = torch.tensor([0, 1, 2, 4, 5, 6])
    value._collect_validation_predictions(
        source, {"source": {"pred_action": torch.ones(6, 100, 14)}}
    )
    with pytest.raises(RuntimeError, match="missing 1 frames"):
        value.on_validation_end()


def test_rejects_nonidentity_retiming_for_robot():
    value = evaluator()
    source = batch()
    source["source"]["retiming_rate"] = torch.full((6,), 0.2)
    with pytest.raises(ValueError, match="retiming_rate"):
        value._collect_validation_predictions(
            source, {"source": {"pred_action": torch.ones(6, 100, 14)}}
        )
