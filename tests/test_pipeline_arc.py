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
    _REPO
    / "egomimic/hydra_configs/experiment/pusht/planar_v2_usocket_arc_graph_tok.yaml"
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


@pytest.mark.parametrize(
    "kwargs",
    [{"resampled_vector_length": 1}, {"action_horizon": 0}, {"native_action_dim": 7}],
)
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


# -- the token's speed is an SE(2) rate, not a translational one -------------
#
# Ported from the main repo's `fix(arc): detokenize walked arc length at a
# chord rate`. There the token carried a chord rate walked against arc
# positions; here the token carries an SE(2) rate (translation + lambda *
# rotation) and the decoder must walk the same metric. Same defect class:
# advancing a rate defined in one metric against positions accumulated in
# another, which silently rescales how fast the chunk replays.

_RR = 30.0


def _rotating_window(steps: int = 40, span: float = 60.0, sweep: float = 2.5):
    actions = torch.zeros(1, steps, 4, dtype=torch.float64)
    actions[0, :, 0] = torch.linspace(0.0, span, steps)
    actions[0, :, 2] = torch.linspace(0.0, sweep, steps)
    return actions


def _rotating_token(waypoints: int = 100):
    stage = ArcTokenizeStage(
        min_distance_unit=200.0,
        resampled_vector_length=waypoints,
        rotation_radius=_RR,
        dt=_DT,
    )
    return stage.forward({"actions": _rotating_window()})["target"]


def _saturation_step(native: torch.Tensor) -> int:
    moved = torch.diff(native[0, :, 0]).abs() > 1e-9
    return int(moved.sum().item())


def test_detokenize_walks_the_se2_metric_the_token_speed_is_expressed_in():
    # The 40-step window is 39 intervals; the reconstruction should need about
    # that many steps, not a third of them.
    token = _rotating_token()
    native = ArcDetokenizeStage(
        resampled_vector_length=100, action_horizon=60, dt=_DT, rotation_radius=_RR
    ).forward({"pred_action": token})["pred_action_native"]
    assert _saturation_step(native) == pytest.approx(39, abs=2)


def test_ignoring_the_rotation_term_replays_a_rotating_path_too_fast():
    # Pins the defect itself: at radius 0 the decoder measures translation only,
    # so it outruns an SE(2) rate and saturates early.
    token = _rotating_token()
    fast = ArcDetokenizeStage(
        resampled_vector_length=100, action_horizon=60, dt=_DT, rotation_radius=0.0
    ).forward({"pred_action": token})["pred_action_native"]
    correct = ArcDetokenizeStage(
        resampled_vector_length=100, action_horizon=60, dt=_DT, rotation_radius=_RR
    ).forward({"pred_action": token})["pred_action_native"]
    assert _saturation_step(fast) < _saturation_step(correct) / 2


def test_translation_only_and_se2_agree_when_there_is_no_rotation_weight():
    # lambda == 0 makes the two metrics identical, so a straight-line token
    # must decode the same either way.
    token = _tokenizer().forward({"actions": _straight_line()})["target"]
    a = _detokenizer(rotation_radius=0.0).forward({"pred_action": token})
    b = _detokenizer(rotation_radius=_RR).forward({"pred_action": token})
    torch.testing.assert_close(a["pred_action_native"], b["pred_action_native"])


def test_arc_positions_match_the_tokenizers_own_step_metric():
    from egomimic.rldb.zarr.planar_arc import lambda_for_radius, planar_step_distance

    token = _rotating_token()
    stage = ArcDetokenizeStage(
        resampled_vector_length=100, action_horizon=60, dt=_DT, rotation_radius=_RR
    )
    waypoints = token[:, :100]
    decoded = stage._arc_positions(waypoints)[0, -1].item()
    xy = waypoints[0, :, :2].numpy()
    theta = np.arctan2(waypoints[0, :, 3].numpy(), waypoints[0, :, 2].numpy())
    expected = planar_step_distance(xy, theta, lambda_for_radius(_RR)).sum()
    assert decoded == pytest.approx(float(expected), rel=1e-9)


def test_detokenize_rejects_a_negative_rotation_radius():
    with pytest.raises(ValueError, match="rotation_radius must be non-negative"):
        _detokenizer(rotation_radius=-1.0)


def test_configured_graph_gives_both_arc_nodes_the_same_rotation_radius():
    # A tokenizer and decoder that disagree on lambda is the whole defect, and
    # nothing in the shapes would reveal it.
    graph = config_graph.build_graph(_EXPERIMENT, mode="train")
    tokenize = next(n for n in graph["nodes"] if n["t"] == "ArcTokenizeStage")
    decode = next(
        entry for entry in graph["skipped_stages"] if entry["t"] == "ArcDetokenizeStage"
    )
    assert decode["reason"] == "inference-only"
    inference = config_graph.build_graph(_EXPERIMENT, mode="inference")
    decode_node = next(n for n in inference["nodes"] if n["t"] == "ArcDetokenizeStage")
    assert tokenize["p"]["rotation_radius"] == decode_node["p"]["rotation_radius"]
    assert tokenize["p"]["dt"] == decode_node["p"]["dt"]
    assert (
        tokenize["p"]["resampled_vector_length"]
        == decode_node["p"]["resampled_vector_length"]
    )


# -- the graph tokenizes the RAW window, never a pre-decimated one -----------
#
# Ported from `feat(arc): tokenize the raw window directly, stop
# pre-decimating it`. The main repo interpolated a 200-frame window down to 100
# rows before tokenizing, so arc length read short and the hardcoded dt was 2x
# too large. Tokenizing in the graph structurally avoids that -- the loader's
# dense transform only pads -- and these pin it.


def _experiment_config():
    config, _ = config_graph._load_selected(_EXPERIMENT)
    return config


def test_the_loader_hands_over_the_full_raw_window():
    from omegaconf import OmegaConf

    config = _experiment_config()
    raw = OmegaConf.select(config, "planar.raw_action_horizon")
    for dataset in OmegaConf.select(config, "data.train_datasets").values():
        assert OmegaConf.select(dataset, "resolver.key_map.action_horizon") == raw


def test_the_dense_transform_list_does_not_resample_before_tokenizing():
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from egomimic.rldb.zarr.action_chunk_transforms import (
        InterpolateLinear,
        InterpolatePose,
    )

    config = _experiment_config()
    dataset = next(iter(OmegaConf.select(config, "data.train_datasets").values()))
    transforms = instantiate(OmegaConf.select(dataset, "resolver.transform_list"))
    assert transforms
    for transform in transforms:
        assert not isinstance(transform, (InterpolatePose, InterpolateLinear))
    # And no tokenizer either: tokenization is the graph's job in this config.
    assert not any(isinstance(t, TokenizePlanarArcLength) for t in transforms)


def test_dt_matches_the_raw_capture_rate():
    from omegaconf import OmegaConf

    config = _experiment_config()
    # Rows reaching the tokenizer are real 30 Hz frames, so dt is 1/30. If the
    # loader ever resampled, this constant would silently be wrong.
    assert OmegaConf.select(config, "planar.arc_dt") == pytest.approx(1.0 / 30.0)


# -- velocity_mode: granular per-waypoint rates ------------------------------
#
# `mean` carries one arc speed for the whole chunk, which is exact only when
# the chunk is traversed at constant speed. `per_waypoint` carries a local rate
# per interval, recovered from the bracketing source frames, so a chunk that
# accelerates, decelerates or dwells replays at its original pace.

from egomimic.rldb.zarr.planar_arc import (  # noqa: E402
    VELOCITY_MODES,
    arc_token_rows,
    validate_velocity_mode,
)


def _shaped(kind: str, steps: int = 120) -> torch.Tensor:
    """(1, T, 3) native planar actions with a chosen speed profile."""
    t = np.linspace(0.0, 1.0, steps)
    a = np.zeros((steps, 3))
    if kind == "constant":
        a[:, 0] = t * 150.0
    elif kind == "accelerating":
        a[:, 0] = (t**2) * 150.0
    elif kind == "decelerating":
        a[:, 0] = np.sqrt(t) * 150.0
    elif kind == "dwell_then_move":
        a[:, 0] = np.clip((t - 0.5) / 0.5, 0.0, 1.0) * 150.0
    else:
        raise ValueError(kind)
    a[:, 2] = t * 1.2
    return torch.from_numpy(a[None])


def _round_trip(kind: str, mode: str, *, horizon: int = 16):
    tok = ArcTokenizeStage(
        min_distance_unit=200.0,
        resampled_vector_length=32,
        dt=_DT,
        rotation_radius=_RR,
        velocity_mode=mode,
    )
    det = ArcDetokenizeStage(
        resampled_vector_length=32,
        action_horizon=horizon,
        dt=_DT,
        rotation_radius=_RR,
        native_action_dim=3,
        velocity_mode=mode,
    )
    actions = _shaped(kind)
    token = tok.forward({"actions": actions.clone()})["target"]
    decoded = det.forward({"pred_action": token})["pred_action_native"][0].numpy()
    return token, decoded, actions[0, :horizon].numpy()


def _errors(decoded, truth):
    xy = float(np.linalg.norm(decoded[:, :2] - truth[:, :2], axis=-1).mean())
    d = decoded[:, 2] - truth[:, 2]
    ang = float(np.abs(np.arctan2(np.sin(d), np.cos(d))).mean())
    return xy, ang


def test_arc_token_rows_is_the_single_source_of_truth():
    assert arc_token_rows(32, "mean") == 33
    assert arc_token_rows(32, "per_waypoint") == 64
    assert arc_token_rows(32, "duration") == 64


def test_arc_token_rows_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="velocity_mode must be one of"):
        arc_token_rows(32, "per_step")


def test_validate_velocity_mode_accepts_every_declared_mode():
    for mode in VELOCITY_MODES:
        assert validate_velocity_mode(mode) == mode


def test_mean_stays_the_default_so_existing_runs_are_untouched():
    assert ArcTokenizeStage().velocity_mode == "mean"
    assert ArcDetokenizeStage().velocity_mode == "mean"
    token, _, _ = _round_trip("constant", "mean")
    assert token.shape[1] == arc_token_rows(32, "mean")


@pytest.mark.parametrize("mode", VELOCITY_MODES)
def test_token_width_follows_the_mode(mode):
    token, _, _ = _round_trip("constant", mode)
    assert tuple(token.shape[1:]) == (arc_token_rows(32, mode), PLANAR_ACTION_DIM)


def test_both_modes_are_exact_on_constant_speed():
    # A single mean rate is a complete description here, so the granular mode
    # must not be WORSE -- that would mean its rate recovery is broken.
    _, mean_out, truth = _round_trip("constant", "mean")
    _, gran_out, _ = _round_trip("constant", "per_waypoint")
    _, dur_out, _ = _round_trip("constant", "duration")
    mean_xy, _ = _errors(mean_out, truth)
    gran_xy, _ = _errors(gran_out, truth)
    dur_xy, _ = _errors(dur_out, truth)
    # Tolerance is relative to the 150-unit path: both are exact to ~1e-5.
    assert mean_xy < 1e-3 and gran_xy < 1e-3 and dur_xy < 1e-3


@pytest.mark.parametrize("kind", ["accelerating", "decelerating", "dwell_then_move"])
@pytest.mark.parametrize("granular_mode", ["per_waypoint", "duration"])
def test_granular_beats_mean_on_non_uniform_motion(kind, granular_mode):
    _, mean_out, truth = _round_trip(kind, "mean")
    _, gran_out, _ = _round_trip(kind, granular_mode)
    mean_xy, mean_ang = _errors(mean_out, truth)
    gran_xy, gran_ang = _errors(gran_out, truth)
    assert gran_xy < mean_xy, f"{kind}/{granular_mode}: {gran_xy} !< {mean_xy}"
    assert gran_ang <= mean_ang + 1e-6


def test_granular_recovers_a_decelerating_chunk_the_mean_mode_mangles():
    # The widest gap measured: one mean rate overshoots badly when the chunk
    # front-loads its travel.
    _, mean_out, truth = _round_trip("decelerating", "mean")
    _, gran_out, _ = _round_trip("decelerating", "per_waypoint")
    _, dur_out, _ = _round_trip("decelerating", "duration")
    mean_xy, _ = _errors(mean_out, truth)
    gran_xy, _ = _errors(gran_out, truth)
    dur_xy, _ = _errors(dur_out, truth)
    assert mean_xy > 5.0
    assert gran_xy < 1.0
    assert dur_xy < 1.0


def test_granular_rate_rows_are_populated_and_non_negative():
    token, _, _ = _round_trip("accelerating", "per_waypoint")
    rates = token[0, 32:, 0].numpy()
    assert (rates >= 0).all()
    assert (rates > 0).any()
    # Reserved columns stay zero so the block keeps the waypoint width.
    assert np.allclose(token[0, 32:, 1:].numpy(), 0.0)


def test_duration_rows_are_populated_and_non_negative():
    token, _, _ = _round_trip("accelerating", "duration")
    durations = token[0, 32:, 0].numpy()
    assert (durations >= 0).all()
    assert (durations > 0).any()
    assert np.allclose(token[0, 32:, 1:].numpy(), 0.0)


def test_duration_and_per_waypoint_agree_on_round_trip():
    # Same source-frame times; rate vs Δt are inverse encodings of the same clock.
    _, rate_out, truth = _round_trip("decelerating", "per_waypoint")
    _, dur_out, _ = _round_trip("decelerating", "duration")
    del truth
    assert np.allclose(rate_out, dur_out, atol=1e-5)


def test_granular_rates_track_the_speed_profile():
    accel = _round_trip("accelerating", "per_waypoint")[0][0, 32:, 0].numpy()
    decel = _round_trip("decelerating", "per_waypoint")[0][0, 32:, 0].numpy()
    # Accelerating chunks end faster than they start; decelerating, the reverse.
    assert accel[-2] > accel[0]
    assert decel[-2] < decel[0]


def test_a_stalled_interval_cannot_traverse_the_path():
    # A nonzero-distance interval with a zero predicted rate gets more time than
    # the whole horizon, so sampling stays inside that first interval. It creeps
    # rather than freezing -- what the guard rules out is the jump to the end.
    det = ArcDetokenizeStage(
        resampled_vector_length=32,
        action_horizon=16,
        dt=_DT,
        rotation_radius=0.0,
        native_action_dim=3,
        velocity_mode="per_waypoint",
    )
    token = torch.zeros(1, 64, PLANAR_ACTION_DIM, dtype=torch.float64)
    token[0, :32, 0] = torch.linspace(0.0, 50.0, 32)  # real travel
    token[0, :32, 2] = 1.0  # cos(theta)=1
    token[0, 32:, 0] = 0.0  # every rate zero
    out = det.forward({"pred_action": token})["pred_action_native"][0].numpy()
    travelled = float(out[-1, 0] - out[0, 0])
    one_interval = 50.0 / 31.0
    assert 0.0 <= travelled <= one_interval, travelled
    assert travelled < 0.05 * 50.0  # a few percent of the path, not all of it


def test_duration_stalled_interval_cannot_traverse_the_path():
    det = ArcDetokenizeStage(
        resampled_vector_length=32,
        action_horizon=16,
        dt=_DT,
        rotation_radius=0.0,
        native_action_dim=3,
        velocity_mode="duration",
    )
    token = torch.zeros(1, 64, PLANAR_ACTION_DIM, dtype=torch.float64)
    token[0, :32, 0] = torch.linspace(0.0, 50.0, 32)
    token[0, :32, 2] = 1.0
    token[0, 32:, 0] = 0.0  # every duration zero
    out = det.forward({"pred_action": token})["pred_action_native"][0].numpy()
    travelled = float(out[-1, 0] - out[0, 0])
    one_interval = 50.0 / 31.0
    assert 0.0 <= travelled <= one_interval, travelled
    assert travelled < 0.05 * 50.0


def test_detokenize_rejects_the_other_modes_token_width():
    mean_det = ArcDetokenizeStage(resampled_vector_length=32, action_horizon=16)
    with pytest.raises(ValueError, match="velocity_mode='mean'"):
        mean_det.forward({"pred_action": torch.zeros(1, 64, PLANAR_ACTION_DIM)})
    gran_det = ArcDetokenizeStage(
        resampled_vector_length=32, action_horizon=16, velocity_mode="per_waypoint"
    )
    with pytest.raises(ValueError, match="velocity_mode='per_waypoint'"):
        gran_det.forward({"pred_action": torch.zeros(1, 33, PLANAR_ACTION_DIM)})


@pytest.mark.parametrize("mode", VELOCITY_MODES)
def test_native_decoder_sizes_itself_from_the_mode(mode):
    from egomimic.pipeline.pushshapes import PlanarArcWaypointZeroNativeDecoder

    decoder = PlanarArcWaypointZeroNativeDecoder(
        resampled_vector_length=32, native_action_dim=3, velocity_mode=mode
    )
    rows = arc_token_rows(32, mode)
    assert decoder.decode(torch.zeros(2, rows, PLANAR_ACTION_DIM)).shape == (2, 1, 3)
    with pytest.raises(ValueError, match="velocity_mode"):
        decoder.decode(torch.zeros(2, rows + 1, PLANAR_ACTION_DIM))


def test_experiment_token_rows_match_its_declared_velocity_mode():
    """`planar.arc_token_rows` must equal what the mode implies.

    The diffusion stages validate the target's width against this value, and
    the repo registers no arithmetic resolver, so the row count is written out
    by hand. This is the guard that catches it drifting from the mode -- a
    mismatch otherwise surfaces as a shape error on the first training batch.
    """
    from omegaconf import OmegaConf

    config, _ = config_graph._load_selected(_EXPERIMENT)
    planar = OmegaConf.select(config, "planar")
    assert planar.arc_token_rows == arc_token_rows(
        planar.arc_waypoints, planar.arc_velocity_mode
    )


def test_experiment_wires_one_velocity_mode_everywhere():
    """Tokenizer, decoder and eval decoder must read the same knob."""
    from omegaconf import OmegaConf

    config, _ = config_graph._load_selected(_EXPERIMENT)
    mode = OmegaConf.select(config, "planar.arc_velocity_mode")
    stages = OmegaConf.select(config, "model.pipeline.stages")
    seen = [s.velocity_mode for s in stages if "velocity_mode" in s]
    assert len(seen) == 2, "expected the tokenize and detokenize nodes"
    assert set(seen) == {mode}
    assert OmegaConf.select(config, "planar.eval_native_decoder.velocity_mode") == mode
