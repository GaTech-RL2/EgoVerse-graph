"""Supported Cartesian ARC contract, shared by training, evaluation and rollout."""

import numpy as np
import pytest

from egomimic.rldb.zarr.arc_length_tokenizer import (
    LOG_CLOCK_EPS,
    TokenizeBimanualArcLengthCartesian,
    bimanual_arc_token_shape,
    clock_token_seconds,
)


def codec(clock="velocity", streams="full_separation", **kwargs):
    options = dict(
        clock_mode=clock,
        streams=streams,
        action_key="actions",
        output_action_key="actions",
        resampled_vector_length=10,
        min_distance_unit=0.09,
        rotation_distance_unit=0.9,
        dt=0.1,
        gripper_stream_mode="distance",
        gripper_distance_unit=0.9,
    )
    options.update(kwargs)
    return TokenizeBimanualArcLengthCartesian(**options)


def trajectory():
    frames = np.arange(30)
    raw = np.zeros((30, 14))
    for col, delay, step in (
        (0, 3, 0.01),
        (7, 5, 0.01),
        (3, 4, 0.1),
        (10, 2, 0.1),
        (6, 6, 0.1),
        (13, 1, 0.1),
    ):
        raw[:, col] = np.clip(frames - delay, 0, 9) * step
    return raw


@pytest.mark.parametrize("clock", ["velocity", "duration"])
@pytest.mark.parametrize(
    "streams,dim", [("full_separation", 20), ("translation_rotation", 18)]
)
def test_contract_shape_and_first_delay(clock, streams, dim):
    c = codec(clock, streams)
    token = c.transform({"actions": trajectory()})["actions"]
    assert token.shape == bimanual_arc_token_shape(10, clock, streams) == (10, dim)
    seconds = clock_token_seconds(token, clock)
    expected = [0.3, 0.5, 0.4, 0.2, 0.6, 0.1][: dim - 14]
    np.testing.assert_allclose(seconds[0, 14:], expected, atol=1e-10)
    interval = [0.1, 0.1, 1.0, 1.0] if clock == "velocity" else [0.1] * 4
    np.testing.assert_allclose(
        seconds[1:, 14:18], np.tile(interval, (9, 1)), atol=1e-10
    )


@pytest.mark.parametrize("streams", ["full_separation", "translation_rotation"])
def test_velocity_and_duration_reconstruct_same_moving_intervals(streams):
    raw = trajectory()
    a, b = codec("velocity", streams), codec("duration", streams)
    va = a.transform({"actions": raw})["actions"]
    vb = b.transform({"actions": raw})["actions"]
    np.testing.assert_allclose(a.detokenize(va, 30), b.detokenize(vb, 30), atol=1e-9)
    assert a.control_steps(va) == b.control_steps(vb)


@pytest.mark.parametrize("clock", ["velocity", "duration"])
def test_four_stream_gripper_follows_translation_not_an_independent_clock(clock):
    c = codec(clock, "translation_rotation")
    raw = trajectory()
    token = c.transform({"actions": raw})["actions"]
    result = c.detokenize(token, 30)
    # Each gripper is linearly interpolated on that arm's translation supports.
    for off in (0, 7):
        x, g = token[:, off], token[:, off + 6]
        np.testing.assert_allclose(
            result[:, off + 6], np.interp(result[:, off], x, g), atol=1e-9
        )
    # Independent gripper motion cannot become a fifth/sixth termination racer.
    changed = token.copy()
    changed[:, 6] = np.linspace(0, 100, 10)
    assert c.control_steps(changed) == c.control_steps(token)


@pytest.mark.parametrize("clock", ["velocity", "duration"])
@pytest.mark.parametrize("streams", ["full_separation", "translation_rotation"])
def test_full_hold_has_constant_waypoints_and_configured_duration(clock, streams):
    c = codec(clock, streams, hold_window_frames=400)
    raw = np.tile(np.arange(14) * 0.01, (400, 1))
    token = c.transform({"actions": raw})["actions"]
    np.testing.assert_allclose(token[:, :14], np.tile(raw[0], (10, 1)))
    timing = clock_token_seconds(token, clock)[:, 14:]
    np.testing.assert_allclose(timing[0], 0, atol=1e-12)
    if clock == "velocity":
        np.testing.assert_array_equal(timing[1:], 0)
    else:
        np.testing.assert_allclose(timing[1:].sum(axis=0), 40)
    assert c.control_steps(token) == 400
    np.testing.assert_allclose(c.detokenize(token, 401), np.tile(raw[0], (401, 1)))


@pytest.mark.parametrize("clock", ["velocity", "duration"])
@pytest.mark.parametrize("streams", ["full_separation", "translation_rotation"])
@pytest.mark.parametrize("fraction", [0.5, 1.0])
def test_eval_rollout_parity_all_contracts(clock, streams, fraction):
    from egomimic.eval.open_loop_sim import OpenLoopSimEval
    from egomimic.robot.arc_decoder import FirstStreamArcDecoder

    c = codec(clock, streams)
    token = c.transform({"actions": trajectory()})["actions"]
    shared = dict(
        clock_mode=clock,
        streams=streams,
        min_distance_unit=0.09,
        rotation_distance_unit=0.9,
        resampled_vector_length=10,
        gripper_stream_mode="distance",
        gripper_distance_unit=0.9,
    )
    e = OpenLoopSimEval(
        **shared, control_dt=0.1, action_mode="arc", execute_fraction=fraction
    )
    r = FirstStreamArcDecoder(
        **shared, dt=0.1, execute_percent=int(fraction * 100), action_horizon=30
    )
    expected, steps = e._decode_prediction_with_steps(token, max_steps=30)
    actual = r(token)[0]
    assert r.execution_steps() == steps
    np.testing.assert_allclose(actual[:steps], expected, atol=1e-10)
    np.testing.assert_allclose(
        actual[steps:], np.tile(actual[steps - 1], (30 - steps, 1))
    )


@pytest.mark.parametrize(
    "option,value",
    [
        ("velocity_mode", "mean"),
        ("velocity_mode", "per_waypoint"),
        ("velocity_layout", "wide"),
        ("velocity_layout", "log_clock"),
        ("velocity_norm", "chord"),
        ("velocity_norm", "path"),
        ("arc_chunking_mode", "race"),
    ],
)
def test_old_options_fail_instead_of_reinterpreting_checkpoints(option, value):
    with pytest.raises(TypeError, match=option):
        codec(**{option: value})


@pytest.mark.parametrize(
    "clock", ["mean", "per_waypoint", "log_clock", "absolute", None]
)
def test_unknown_clock_rejected(clock):
    with pytest.raises(ValueError, match="clock_mode"):
        codec(clock)


@pytest.mark.parametrize("streams", ["wide", "stacked", "e1_durtri", None])
def test_unknown_streams_rejected(streams):
    with pytest.raises(ValueError, match="streams"):
        codec(streams=streams)


@pytest.mark.parametrize("clock", ["velocity", "duration"])
def test_moving_interval_with_unusable_clock_cannot_win_race(clock):
    c = codec(clock)
    token = c.transform({"actions": trajectory()})["actions"]
    token[1:, 14:] = 0 if clock == "velocity" else np.log(LOG_CLOCK_EPS)
    with pytest.raises(ValueError, match="finite replan"):
        c.control_steps(token)
    assert c.control_steps(token, max_steps=30) == 30


@pytest.mark.parametrize("clock", ["velocity", "duration"])
@pytest.mark.parametrize("streams", ["full_separation", "translation_rotation"])
def test_preserved_ground_truth_is_not_reconstructed_from_tokens(clock, streams):
    c = codec(clock, streams, preserve_action_key="raw", preserve_action_rows=40)
    raw = trajectory()
    batch = c.transform({"actions": raw.copy()})
    np.testing.assert_array_equal(batch["raw"][:30], raw)
    np.testing.assert_array_equal(batch["raw"][30:], np.tile(raw[-1], (10, 1)))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_source_and_tokens_rejected(bad):
    c = codec()
    raw = trajectory()
    raw[3, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        c.transform({"actions": raw})
    token = np.zeros((10, 20))
    token[2, 14] = bad
    with pytest.raises(ValueError, match="finite"):
        c.detokenize(token, 20)


@pytest.mark.parametrize("clock", ["velocity", "duration"])
@pytest.mark.parametrize(
    "streams,dim", [("full_separation", 20), ("translation_rotation", 18)]
)
def test_recipe_data_evaluator_and_declared_rollout_share_contract(clock, streams, dim):
    from hydra.utils import instantiate

    from egomimic.pipeline.inference_config import build_inference_config
    from scripts.audit_hydra_configs import CONFIGS, compose_for_audit

    recipe = (
        CONFIGS
        / "experiment/abc_arc/cotrain"
        / f"organize_clock_{clock}_{streams}.yaml"
    )
    with compose_for_audit(recipe) as cfg:
        transforms = instantiate(
            cfg.data.train_datasets.yam_bimanual.resolver.transform_list
        )
        c = next(
            t for t in transforms if isinstance(t, TokenizeBimanualArcLengthCartesian)
        )
        assert c.clock_mode == clock and c.streams == streams
        assert c.shape == (100, dim)
        e = instantiate(cfg.evaluator)
        assert e.clock_mode == clock and e.streams == streams
        assert e._codec().shape == c.shape
        artifact = build_inference_config(cfg)
        assert artifact["status"] == "ready"
        profile = artifact["inference_graph"]["profiles"]["default"]
        assert profile["native_shape"] == [100, dim]
        decoder = instantiate(profile["adapter"]["decoder"])
        assert decoder.codec.shape == c.shape
        assert decoder.codec.clock_mode == clock
        assert decoder.codec.streams == streams


@pytest.mark.parametrize(
    "change",
    [
        {},
        {"clock_mode": "duration"},
        {"streams": "translation_rotation"},
        {"timing_layout": "legacy_outgoing"},
        {"hold_window_frames": 100},
    ],
)
def test_normalization_cache_requires_identical_clock_contract(tmp_path, change):
    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

    def normalizer():
        return MultiDataset(
            state={
                "norm_mode": "quantile",
                "embodiments": [0],
                "key_types": {0: {"actions_cartesian": "action_keys"}},
                "zarr_keys": {0: {"actions_cartesian": "actions_cartesian"}},
            }
        )

    contract = dict(
        representation="hybrid_arc_tokenizer_cartesian",
        arc_chunking_mode="multistream",
        clock_mode="velocity",
        streams="full_separation",
        timing_layout="first_delay_incoming_v1",
        hold_window_frames=400,
    )
    source = normalizer()
    source.infer_norm_from_dataset(
        [{"actions_cartesian": np.ones((10, 20))}],
        dataset_name=0,
        num_workers=0,
        sample_frac=1.0,
        action_contract=contract,
    )
    source.cache_stats(str(tmp_path))
    options = dict(
        dataset_name=0,
        precomputed_norm_path=str(tmp_path / "norm_stats/norm_stats.json"),
        action_contract={**contract, **change},
    )
    if change:
        with pytest.raises(ValueError, match="action contract"):
            normalizer().infer_norm_from_dataset([], **options)
    else:
        normalizer().infer_norm_from_dataset([], **options)
