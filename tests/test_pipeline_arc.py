"""Arc tokenize / detokenize graph nodes and the inference-only restriction."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from egomimic.pipeline.core import Pipeline, Stage
from egomimic.pipeline.stages_arc import ArcDetokenizeStage, ArcTokenizeStage
from egomimic.rldb.zarr.planar_arc import (
    PLANAR_ACTION_DIM,
    TokenizePlanarArcLength,
)

_REPO = Path(__file__).resolve().parents[1]
_EXPERIMENT = (
    _REPO / "egomimic/hydra_configs/experiment/pusht/planar_v2_usocket_arc_graph_tok.yaml"
)
_SPEC = importlib.util.spec_from_file_location(
    "config_graph_arc", _REPO / "tools/config_graph.py"
)
config_graph = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(config_graph)

_M = 8
_DT = 1.0 / 30.0


def _straight_line(steps: int = 20, span: float = 100.0) -> torch.Tensor:
    actions = torch.zeros(2, steps, 4, dtype=torch.float64)
    actions[:, :, 0] = torch.linspace(0.0, span, steps)
    return actions


def _tokenizer(**kwargs) -> ArcTokenizeStage:
    params = dict(min_distance_unit=50.0, resampled_vector_length=_M, dt=_DT)
    params.update(kwargs)
    return ArcTokenizeStage(**params)


def _detokenizer(**kwargs) -> ArcDetokenizeStage:
    params = dict(resampled_vector_length=_M, action_horizon=16, dt=_DT)
    params.update(kwargs)
    return ArcDetokenizeStage(**params)


# -- contracts --------------------------------------------------------------


def test_tokenize_writes_target_and_is_the_only_writer_of_it():
    stage = _tokenizer()
    assert stage.contract("train") == (("actions",), ("target",))


def test_tokenize_reads_the_configured_action_key():
    assert _tokenizer(action_key="actions_native").reads == ("actions_native",)


def test_tokenize_is_dropped_from_the_inference_graph():
    _, excluded = Pipeline([_tokenizer()]).plan(["actions"], mode="inference")
    assert excluded[0][1] == ["<train-only>"]


def test_detokenize_is_dropped_from_the_train_graph():
    _, excluded = Pipeline([_detokenizer()]).plan(["pred_action"], mode="train")
    assert excluded[0][1] == ["<inference-only>"]


def test_an_inference_only_stage_does_not_block_a_train_graph():
    # The runner raises on a blocked stage, so this is what the flag buys.
    pipeline = Pipeline([_detokenizer()])
    assert pipeline.execute({"unrelated": 1}, mode="train") == {"unrelated": 1}


def test_explain_names_the_inference_only_exclusion():
    text = Pipeline([_detokenizer()]).explain(["pred_action"], mode="train")
    assert "EXCLUDED: inference-only" in text


def test_a_genuinely_blocked_stage_still_raises():
    class NeedsMissing(Stage):
        reads = ("absent",)
        writes = ("out",)

        def forward(self, batch):  # pragma: no cover - never runs
            return batch

    with pytest.raises(RuntimeError, match="blocked stages"):
        Pipeline([NeedsMissing()]).execute({"present": 1}, mode="train")


# -- tokenize ---------------------------------------------------------------


def test_tokenize_emits_m_waypoints_plus_one_timing_row():
    out = _tokenizer().forward({"actions": _straight_line()})
    assert out["target"].shape == (2, _M + 1, PLANAR_ACTION_DIM)


def test_tokenize_consumes_the_action_key_it_read():
    out = _tokenizer().forward({"actions": _straight_line()})
    assert "actions" not in out


def test_tokenize_matches_the_loader_side_transform_exactly():
    # A graph-side and a dataset-side tokenizer configured alike must agree,
    # or switching a run between them silently changes the target.
    actions = _straight_line()
    graph = _tokenizer().forward({"actions": actions})["target"][0].numpy()
    loader = TokenizePlanarArcLength(
        min_distance_unit=50.0, resampled_vector_length=_M, dt=_DT
    ).tokenize(actions[0].numpy().astype(np.float64))
    np.testing.assert_allclose(graph, loader)


def test_tokenize_preserves_dtype_and_device_of_its_input():
    actions = _straight_line().to(torch.float32)
    out = _tokenizer().forward({"actions": actions})["target"]
    assert out.dtype == torch.float32 and out.device == actions.device


def test_tokenize_anchors_the_first_waypoint_on_the_current_step():
    actions = _straight_line()
    token = _tokenizer().forward({"actions": actions})["target"]
    np.testing.assert_allclose(token[0, 0, :2].numpy(), actions[0, 0, :2].numpy())


def test_tokenize_rejects_an_unbatched_chunk():
    with pytest.raises(ValueError, match=r"\(B, T, D\)"):
        _tokenizer().forward({"actions": torch.zeros(20, 4)})


def test_tokenize_rejects_a_non_tensor():
    with pytest.raises(TypeError, match="must be a tensor"):
        _tokenizer().forward({"actions": np.zeros((2, 20, 4))})


# -- detokenize -------------------------------------------------------------


def test_detokenize_returns_the_requested_horizon_and_native_width():
    token = _tokenizer().forward({"actions": _straight_line()})["target"]
    out = _detokenizer(action_horizon=12, native_action_dim=3).forward(
        {"pred_action": token}
    )
    assert out["pred_action_native"].shape == (2, 12, 3)


def test_detokenize_recovers_uniform_spacing_on_a_straight_line():
    # The 50-unit window is traversed at 150 units/s, i.e. 5 units per step, so
    # 11 steps land exactly on the end and every step is the same size.
    token = _tokenizer().forward({"actions": _straight_line()})["target"]
    native = _detokenizer(action_horizon=11).forward({"pred_action": token})[
        "pred_action_native"
    ]
    steps = torch.diff(native[0, :, 0])
    assert torch.allclose(steps, steps[0].expand_as(steps), atol=1e-6)
    assert steps[0] > 0
    assert native[0, -1, 0].item() == pytest.approx(50.0)


def test_detokenize_step_size_is_speed_times_dt():
    token = _tokenizer().forward({"actions": _straight_line()})["target"]
    speed = token[0, _M, 0].item()
    native = _detokenizer().forward({"pred_action": token})["pred_action_native"]
    assert torch.diff(native[0, :, 0])[0].item() == pytest.approx(speed * _DT, rel=1e-6)


def test_detokenize_holds_position_for_a_zero_speed_token():
    token = torch.zeros(1, _M + 1, PLANAR_ACTION_DIM, dtype=torch.float64)
    token[0, :_M, 0] = torch.linspace(0.0, 5.0, _M)
    token[0, :_M, 2] = 1.0  # cos(theta) = 1
    token[0, _M, 0] = 0.0  # mean arc speed
    native = _detokenizer().forward({"pred_action": token})["pred_action_native"]
    assert torch.allclose(native[0, :, 0], torch.zeros(16, dtype=torch.float64))


def test_detokenize_clamps_at_the_end_of_the_polyline():
    # A horizon long enough to outrun the window must saturate, not extrapolate.
    token = _tokenizer().forward({"actions": _straight_line()})["target"]
    native = _detokenizer(action_horizon=400).forward({"pred_action": token})[
        "pred_action_native"
    ]
    final = native[0, -1, 0]
    assert torch.allclose(native[0, -5:, 0], final.expand(5))


def test_detokenize_recovers_heading_through_the_pi_seam():
    # cos/sin interpolation exists so a chunk that WRAPS past +pi does not
    # unwind through zero. The headings advance monotonically past the seam:
    # pi - 0.2 ... pi + 0.2, which is -pi + 0.2 once wrapped.
    token = torch.zeros(1, _M + 1, PLANAR_ACTION_DIM, dtype=torch.float64)
    token[0, :_M, 0] = torch.linspace(0.0, 10.0, _M)
    thetas = torch.linspace(math.pi - 0.2, math.pi + 0.2, _M, dtype=torch.float64)
    token[0, :_M, 2] = torch.cos(thetas)
    token[0, :_M, 3] = torch.sin(thetas)
    token[0, _M, 0] = 30.0
    native = _detokenizer().forward({"pred_action": token})["pred_action_native"]
    # Every decoded heading stays near the seam; none swings back through 0.
    assert torch.all(native[0, :, 2].abs() > math.pi - 0.3)


def test_detokenize_logs_the_window_it_decoded():
    token = _tokenizer().forward({"actions": _straight_line()})["target"]
    out = _detokenizer().forward({"pred_action": token})
    assert out["log/ArcSpeed"].ndim == 0 and out["log/ArcChunkDistance"].ndim == 0


def test_detokenize_rejects_a_token_of_the_wrong_width():
    with pytest.raises(ValueError, match="ArcDetokenizeStage expects"):
        _detokenizer().forward({"pred_action": torch.zeros(2, _M, PLANAR_ACTION_DIM)})


@pytest.mark.parametrize("kwargs", [{"resampled_vector_length": 1}, {"action_horizon": 0}, {"native_action_dim": 7}])
def test_detokenize_rejects_impossible_settings(kwargs):
    with pytest.raises(ValueError):
        _detokenizer(**kwargs)


# -- the configured graph ---------------------------------------------------


@pytest.mark.parametrize("mode", ["train", "inference"])
def test_arc_graph_experiment_lints_clean(mode):
    assert config_graph.build_graph(_EXPERIMENT, mode=mode)["lint"] == []


def test_train_graph_tokenizes_and_omits_the_detokenizer():
    graph = config_graph.build_graph(_EXPERIMENT, mode="train")
    names = [node["t"] for node in graph["nodes"]]
    assert "ArcTokenizeStage" in names and "ArcDetokenizeStage" not in names
    assert {"t": "ArcDetokenizeStage"}.items() <= next(
        entry for entry in graph["skipped_stages"] if entry["t"] == "ArcDetokenizeStage"
    ).items()


def test_inference_graph_detokenizes_and_omits_the_tokenizer():
    graph = config_graph.build_graph(_EXPERIMENT, mode="inference")
    names = [node["t"] for node in graph["nodes"]]
    assert "ArcDetokenizeStage" in names and "ArcTokenizeStage" not in names
    assert "pred_action_native" in {
        key for node in graph["nodes"] for key in node["out"]
    }


def test_tokenizer_replaces_the_plain_target_builder_rather_than_following_it():
    graph = config_graph.build_graph(_EXPERIMENT, mode="train")
    names = [node["t"] for node in graph["nodes"]]
    assert "ActionTargetBuilder" not in names
    writers = [n["t"] for n in graph["nodes"] if "target" in n["out"]]
    assert writers == ["ArcTokenizeStage"]
