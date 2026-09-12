"""Regression coverage for the graph adapter and the original E1 metric values."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from egomimic.eval.e1_fold_tempo_eval import E1FoldTempoEval
from egomimic.rldb.zarr.e1_arc_tokenizer import TokenizeBimanualArcLengthE1


def example(variant="time", perturbation="none"):
    progress = np.linspace(0, 0.6, 100)
    truth = np.zeros((1, 100, 14))
    for offset in (0, 7):
        truth[0, :, offset] = progress
        truth[0, :, offset + 1] = 0.04 * np.sin(progress * 8)
        truth[0, :, offset + 6] = progress / 2
    pred = truth.copy()
    if perturbation == "short":
        pred *= 0.45
    elif perturbation == "offset":
        pred[:, :, 1] += 0.07
    if variant != "time":
        codec = TokenizeBimanualArcLengthE1(
            min_distance_unit=0.4,
            resampled_vector_length=100,
            dt=1 / 30,
            velocity_norm="path",
            velocity_mode={
                "arcmean": "mean",
                "arcvel": "profile",
                "arclogdur": "logdur",
                "arcdur": "dur",
            }[variant],
        )
        pred = codec.transform({"actions_cartesian": pred[0]})["actions_cartesian"][
            None
        ]
    return torch.tensor(pred, dtype=torch.float32), torch.tensor(
        truth, dtype=torch.float32
    )


class Normalizer:
    def __init__(self, scale=1):
        self.scale = scale

    def unnormalize(self, values, _embodiment):
        return {key: value * self.scale for key, value in values.items()}


def evaluator(pred, variant="time", **kwargs):
    result = E1FoldTempoEval(variant=variant, **kwargs)
    result.model = SimpleNamespace(
        forward_eval=lambda batch: {source: {"pred_action": pred} for source in batch}
    )
    result.bind_data_context(normalizer=Normalizer())
    return result


def batch(truth):
    return {"opaque-source": {"embodiment": torch.tensor([7]), "actions_time": truth}}


@pytest.mark.parametrize(
    "case",
    [
        "time-none",
        "time-short",
        "time-offset",
        "arcmean-none",
        "arclogdur-none",
        "arcdur-none",
    ],
)
def test_graph_evaluation_matches_frozen_source_campaign_values(case):
    variant, perturbation = case.split("-")
    pred, truth = example(variant, perturbation)
    obj = evaluator(pred, variant)
    obj.on_validation_step(batch(truth), 0)
    actual = obj.on_validation_end()
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/e1_source_metrics.json").read_text()
    )
    expected = fixture["cases"][case]
    for key, value in expected.items():
        if isinstance(value, (int, float)):
            assert actual[key] == pytest.approx(value, rel=1e-9, abs=1e-12), key
        else:
            assert actual[key] == value, key


def test_graph_predictions_and_time_targets_are_unnormalized_once(tmp_path):
    pred, truth = example()
    obj = evaluator(pred / 2, results_path=tmp_path / "result.json")
    obj.bind_data_context(normalizer=Normalizer(2))
    metrics = obj.on_validation_step(batch(truth / 2), 0)
    result = obj.on_validation_end()
    assert result["paired_mse"] == 0
    assert metrics["Valid/E1/E_time/yam_bimanual"].device == pred.device
    assert json.loads((tmp_path / "result.json").read_text())["n_chunks"] == 1


def test_validation_groups_keep_independent_accumulators():
    pred, truth = example()
    obj = evaluator(pred)
    obj.set_validation_group("perfect")
    obj.on_validation_step(batch(truth), 0)
    obj.set_validation_group("offset")
    obj.on_validation_step(batch(truth + 0.1), 0)
    result = obj.on_validation_end()["groups"]
    assert result["perfect"]["yam_bimanual"]["paired_mse"] == 0
    assert result["offset"]["yam_bimanual"]["paired_mse"] > 0
    assert result["perfect"]["yam_bimanual"]["n_chunks"] == 1


def test_distributed_results_merge_counts_and_sums(monkeypatch):
    from egomimic.eval import e1_fold_tempo_eval as module

    pred, truth = example("time", "offset")
    obj = evaluator(pred)
    obj.on_validation_step(batch(truth), 0)
    local = obj.on_validation_end()
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(
        module.dist,
        "all_gather_object",
        lambda output, value: output.__setitem__(slice(None), [value, value]),
    )
    merged = obj.on_validation_end()
    assert merged["n_chunks"] == 2
    assert merged["paired_mse"] == pytest.approx(local["paired_mse"])
    assert merged["e_time_p90"] == pytest.approx(local["e_time_p90"])


def test_missing_time_targets_fail_instead_of_writing_empty_metrics():
    pred, _ = example()
    obj = evaluator(pred)
    with pytest.raises(KeyError, match="actions_time"):
        obj.on_validation_step({"source": {"embodiment": torch.tensor([7])}}, 0)
    with pytest.raises(RuntimeError, match="no prediction"):
        obj.on_validation_end()


def test_validation_logs_through_lightning_and_writes_rescore_schema(tmp_path):
    pred, truth = example()
    obj = evaluator(pred, results_path=tmp_path / "e1_tempo.json")
    logged = []
    obj.trainer = SimpleNamespace(
        is_global_zero=True,
        lightning_module=SimpleNamespace(log_dict=lambda values, **kwargs: logged.append(values)),
    )
    obj.on_validation_step(batch(truth), 0)
    obj.on_validation_end()
    assert len(logged) == 1
    assert logged[0]["Valid/E1/E_time/yam_bimanual"] == 0
    result = json.loads((tmp_path / "eval_metrics.json").read_text())
    assert result["results"][0]["Valid/E1/paired_mse/yam_bimanual"] == 0
