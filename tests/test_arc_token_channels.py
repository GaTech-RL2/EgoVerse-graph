"""Trailing timing features preserve ARC geometry, clocks and execution caps."""

from pathlib import Path

import hydra
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from egomimic.eval.arc_bimanual_cartesian_eval import ArcBimanualCartesianEval
from egomimic.eval.open_loop_sim import (
    OpenLoopSimEval,
    arc_execution_prefix,
    arc_prefix_control_steps,
)
from egomimic.pipeline.core import Pipeline
from egomimic.rldb.zarr.arc_length_tokenizer import (
    TokenizeBimanualArcLengthCartesian,
    bimanual_arc_to_rows,
    bimanual_arc_token_shape,
)


def _source():
    time = np.arange(211) / 30.0
    raw = np.zeros((len(time), 14))
    raw[:, 0] = 0.2 * time
    raw[:, 7] = 0.1 * time
    raw[:, [3, 10]] = 0.05 * time[:, None]
    raw[:, [6, 13]] = 0.1 * time[:, None]
    return raw


def _codec(layout, **kwargs):
    options = dict(
        min_distance_unit=0.4,
        rotation_distance_unit=0.6,
        resampled_vector_length=20,
        velocity_mode="per_waypoint",
        arc_chunking_mode="joint_distance",
        token_layout=layout,
    )
    return TokenizeBimanualArcLengthCartesian(**(options | kwargs))


@pytest.mark.parametrize("mode", ["race", "multistream", "joint_distance"])
@pytest.mark.parametrize(
    "motion", ["both", "stationary_arm", "stationary", "short", "rotation_only"]
)
def test_trailing_clock_columns_preserve_all_streams_and_terminal_holds(mode, motion):
    raw = _source()
    if motion == "stationary_arm":
        raw[:, 7:10] = [0.2, 0.3, 0.4]
    elif motion == "stationary":
        raw[:] = raw[0]
    elif motion == "short":
        raw[:, 0] = np.minimum(raw[:, 0], 0.04)  # 1/10 of D
    elif motion == "rotation_only":
        raw[:, [0, 7]] = 0
    old = _codec("rows", arc_chunking_mode=mode)
    new = _codec("channels", arc_chunking_mode=mode)
    before = old.transform({"actions_cartesian": raw})["actions_cartesian"]
    after = new.transform({"actions_cartesian": raw})["actions_cartesian"]
    assert before.shape == (40, 14)
    assert after.shape == (20, 28)
    np.testing.assert_array_equal(after[:, :14], before[:20])
    np.testing.assert_array_equal(after[:, 14:], before[20:])
    decoded = new.detokenize(after, 211)
    np.testing.assert_array_equal(decoded, old.detokenize(before, 211))
    assert np.isfinite(decoded).all()
    if motion == "short":
        np.testing.assert_allclose(decoded[30:, 0], 0.04, atol=1e-9)
        assert np.max(decoded[:, 0]) <= 0.04 + 1e-9
    if motion != "stationary":
        # The rotation clock still finishes after the translation clock.
        np.testing.assert_allclose(decoded[180:, [3, 10]], 0.3, atol=1e-9)


@pytest.mark.parametrize("timing", ["per_waypoint", "duration"])
def test_nonhybrid_timing_packs_losslessly_and_rejects_the_wrong_layout(timing):
    options = dict(
        rotation_distance_unit=None, arc_chunking_mode=None, velocity_mode=timing
    )
    old, new = _codec("rows", **options), _codec("channels", **options)
    rows = old.transform({"actions_cartesian": _source()})["actions_cartesian"]
    token = new.transform({"actions_cartesian": _source()})["actions_cartesian"]
    np.testing.assert_array_equal(bimanual_arc_to_rows(token, timing, "channels"), rows)
    np.testing.assert_array_equal(new.detokenize(token, 100), old.detokenize(rows, 100))
    with pytest.raises(ValueError, match="token_layout"):
        old.detokenize(token, 100)
    with pytest.raises(ValueError, match="token_layout"):
        new.detokenize(rows, 100)


def test_invalid_layout_and_unrepresentable_mean_timing_fail_at_construction():
    with pytest.raises(ValueError, match="token_layout"):
        _codec("guess")
    with pytest.raises(ValueError, match="per_waypoint or duration"):
        _codec("channels", velocity_mode="mean")
    assert bimanual_arc_token_shape(20, "mean") == (21, 14)


@pytest.mark.parametrize("mode", ["race", "multistream", "joint_distance"])
@pytest.mark.parametrize("cap", ["waypoints", "distance"])
def test_execution_prefix_and_replanning_are_layout_independent(mode, cap):
    old, new = (
        _codec("rows", arc_chunking_mode=mode),
        _codec("channels", arc_chunking_mode=mode),
    )
    rows = old.transform({"actions_cartesian": _source()})["actions_cartesian"]
    token = new.transform({"actions_cartesian": _source()})["actions_cartesian"]
    kwargs = dict(arc_chunking_mode=mode, rotation_distance_unit=0.6)
    before = arc_execution_prefix(rows, 0.3, "per_waypoint", 0.4, cap, **kwargs)
    after = arc_execution_prefix(
        token, 0.3, "per_waypoint", 0.4, cap, token_layout="channels", **kwargs
    )
    np.testing.assert_array_equal(
        bimanual_arc_to_rows(after, "per_waypoint", "channels"), before
    )
    before_steps = arc_prefix_control_steps(
        rows, 0.3, "per_waypoint", 1 / 30, 0.4, arc_execution_cap_mode=cap, **kwargs
    )
    after_steps = arc_prefix_control_steps(
        token,
        0.3,
        "per_waypoint",
        1 / 30,
        0.4,
        arc_execution_cap_mode=cap,
        token_layout="channels",
        **kwargs,
    )
    assert before_steps == after_steps
    evaluators = [
        OpenLoopSimEval(
            action_mode="arc",
            resampled_vector_length=20,
            min_distance_unit=0.4,
            execute_fraction=0.3,
            arc_execution_cap_mode=cap,
            token_layout=layout,
            **kwargs,
        )
        for layout in ("rows", "channels")
    ]
    np.testing.assert_array_equal(
        evaluators[0]._decode_prediction(rows), evaluators[1]._decode_prediction(token)
    )


@pytest.mark.parametrize("mode", ["race", "multistream", "joint_distance"])
def test_cartesian_visualization_and_shape_metrics_never_treat_clock_columns_as_pose(
    mode,
):
    codec = _codec("channels", arc_chunking_mode=mode)
    token = codec.transform({"actions_cartesian": _source()})["actions_cartesian"]
    evaluator = ArcBimanualCartesianEval(
        min_distance_unit=0.4,
        rotation_distance_unit=0.6,
        resampled_vector_length=20,
        token_layout="channels",
        velocity_mode="per_waypoint",
        arc_chunking_mode=mode,
    )
    assert evaluator._tokenizer.arc_chunking_mode == mode
    evaluator._native = lambda value, embodiment: value
    prediction = torch.from_numpy(token[None])
    decoded = evaluator._viz_source(prediction, 0)
    assert decoded.shape == (1, 100, 14)
    np.testing.assert_array_equal(decoded[0].numpy(), codec.detokenize(token, 100))
    geometry = evaluator._arc_pred_for_arcmatch(prediction, [_source()], 0)
    np.testing.assert_array_equal(geometry[0], token[:, :14])
    legacy = bimanual_arc_to_rows(token, "per_waypoint", "channels")
    with pytest.raises(ValueError, match="legacy timing rows"):
        evaluator._viz_source(torch.from_numpy(legacy[None]), 0)


def test_visual_channel_targets_train_and_sample_through_configured_flow():
    config_root = Path(__file__).resolve().parents[1] / "egomimic/hydra_configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_root)):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=[
                "+experiment=abc_arc/robot_bc/stationery_rl2_hpt300_visual_hybrid_openloop",
                "abc.arc_waypoints=20",
                "hpt.embed_dim=32",
                "hpt.flow_blocks=1",
                "hpt.flow_hidden_dim=32",
            ],
        )
    assert cfg.hpt.action_horizon == 20 and cfg.hpt.action_dim == 28
    proprio = cfg.model.pipeline.stages[0].domain_stems.yam_bimanual[
        "observations.state.ee_pose"
    ]
    assert proprio.input_dim == 14
    stages = [hydra.utils.instantiate(spec) for spec in cfg.model.pipeline.stages[3:]]
    stages[1].num_inference_steps = 2
    pipeline = Pipeline(stages)
    token = _codec("channels").transform({"actions_cartesian": _source()})[
        "actions_cartesian"
    ]
    target = torch.from_numpy(token).float().unsqueeze(0).repeat(2, 1, 1)
    batch = {"condition": torch.randn(2, 32), "target": target}
    trained = pipeline.execute(batch, mode="train")
    loss = trained["loss/flow_velocity"]
    assert torch.isfinite(loss)
    loss.backward()
    output_grad = stages[1].model.proj_d.weight.grad
    assert torch.isfinite(output_grad).all()
    assert output_grad[:14].abs().sum() > 0
    assert output_grad[14:].abs().sum() > 0
    sampled = pipeline.execute({"condition": batch["condition"]}, mode="inference")[
        "pred_action"
    ]
    assert sampled.shape == (2, 20, 28) and torch.isfinite(sampled).all()
