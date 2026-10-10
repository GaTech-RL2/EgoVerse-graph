"""ARC extensions obey main's model-owned, hardware-free inference contract."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from egomimic.pipeline.action_adapter import execution_prefix
from egomimic.pipeline.inference_config import build_inference_config
from egomimic.pipeline.inference_controls import (
    apply_control_bindings,
    configure_profile_controls,
)
from egomimic.robot.arc_codecs import m28
from egomimic.robot.arc_decoder import (
    BimanualArcDecoder,
    BimanualIntervalArcDecoder,
    FirstStreamArcDecoder,
)
from tests.test_inference_config import training_config

ROOT = Path(__file__).resolve().parents[1]


def declared_profile():
    config = training_config(horizon=100, action_dim=16)
    config.model.inference = OmegaConf.load(
        ROOT / "egomimic/hydra_configs/model/inference/station_arc_duration.yaml"
    )
    return config


def bindings(config, decoder):
    stage = SimpleNamespace(num_inference_steps=50)
    graph = SimpleNamespace(pipeline=SimpleNamespace(stage_by_id=lambda _: stage))
    controls = configure_profile_controls(
        graph,
        config,
        OmegaConf.to_container(config.model.inference.profiles, resolve=True),
        decoder=decoder,
    )
    return {control.name: control for control in controls}, stage


def test_yaml_owns_decoder_controls_and_export_stays_schema2():
    config = declared_profile()
    artifact = build_inference_config(config)
    assert artifact["schema_version"] == 2 and artifact["status"] == "ready"
    decoder = instantiate(config.model.inference.profiles.default.adapter.decoder)
    controls, stage = bindings(config, decoder)
    policy = SimpleNamespace(replan_every=30)
    assert stage.num_inference_steps == 20
    assert decoder.first_stream == 1 and decoder.execute_percent == 50
    apply_control_bindings(
        controls,
        {"arc_speed_percent": 200, "multistream_fastest_stream": 0},
        policy=policy,
    )
    assert decoder.speed == 2.0 and decoder.first_stream == 0
    assert controls["arc_speed_percent"].public()["value"] == 200


def test_no_decoder_controls_are_injected_into_models_that_do_not_declare_them():
    config = declared_profile()
    config.model.inference.profiles.default.overrides = {}
    decoder = instantiate(config.model.inference.profiles.default.adapter.decoder)
    controls, _ = bindings(config, decoder)
    assert controls == {}


def test_missing_decoder_fails_before_mutating_sampler_defaults():
    config = declared_profile()
    stage = SimpleNamespace(num_inference_steps=50)
    graph = SimpleNamespace(pipeline=SimpleNamespace(stage_by_id=lambda _: stage))
    with pytest.raises(ValueError, match="configured decoder"):
        configure_profile_controls(
            graph,
            config,
            OmegaConf.to_container(config.model.inference.profiles, resolve=True),
        )
    assert stage.num_inference_steps == 50


def test_invalid_decoder_property_rolls_back_entire_control_update():
    config = declared_profile()
    decoder = BimanualArcDecoder("e1_dur", resampled_vector_length=50)
    controls, stage = bindings(config, decoder)
    # 33% is within the example's M=100 bounds, but invalid for an M=50 decoder.
    # Property-level failure must not leave the sampler partly changed.
    with pytest.raises(ValueError, match="whole prefix"):
        apply_control_bindings(
            controls, {"inference_steps": 40, "execute_waypoint_percent": 33}
        )
    assert stage.num_inference_steps == 20 and decoder.execute_percent == 50
    assert controls["inference_steps"].value == 20


@pytest.mark.parametrize(
    "mode,layout",
    [("per_waypoint", "wide"), ("per_waypoint", "stacked"), ("duration", "clock")],
)
def test_fastest_stream_prefix_is_shared_by_tensor_and_robot_consumers(mode, layout):
    dt, horizon, waypoints = 1 / 30, 100, 100
    t = np.arange(400)[:, None] * dt
    actions = np.zeros((400, 14))
    actions[:, 0] = t[:, 0] * 0.5
    actions[:, 8] = t[:, 0] * 0.1
    actions[:, 3] = actions[:, 10] = t[:, 0] * 0.2
    actions[:, [6, 13]] = 0.5
    codec = m28.TokenizeBimanualArcLengthCartesian(
        min_distance_unit=0.7,
        rotation_distance_unit=2.0,
        resampled_vector_length=waypoints,
        dt=dt,
        velocity_mode=mode,
        velocity_layout=layout,
        arc_chunking_mode="multistream",
    )
    native = codec.transform({"actions_cartesian": actions})["actions_cartesian"]
    decoder = FirstStreamArcDecoder(
        mode,
        layout,
        0.7,
        2.0,
        waypoints,
        dt,
        horizon,
        execute_percent=30,
        first_stream=1,
    )
    prediction = decoder(native)
    n = decoder.last_stats["replan_steps"]
    moving = [d for d in decoder.last_stats["stream_durations_s"] if d > 1e-9]
    assert n == min(horizon, int(np.ceil(min(moving) / dt - 1e-9)))
    robot_plan = execution_prefix(prediction[0], decoder=decoder, replan_every=7)
    tensor_plan = execution_prefix(
        torch.from_numpy(prediction), decoder=decoder, replan_every=7, time_axis=1
    )
    assert len(robot_plan) == n and tensor_plan.shape == (1, n, 14)
    np.testing.assert_array_equal(robot_plan, tensor_plan.numpy()[0])
    assert np.all(prediction[0, n:] == prediction[0, n - 1])
    decoder.first_stream = 0
    prediction = decoder(native)
    assert len(execution_prefix(prediction[0], decoder=decoder, replan_every=7)) == 7


def test_default_decoder_is_duration_without_implicitly_enabling_fastest_stream():
    decoder = BimanualArcDecoder()
    assert decoder.token_layout == "e1_dur" and decoder.first_stream == 0


@pytest.mark.parametrize("mode", ["per_waypoint", "duration"])
def test_main_interval_decoder_keeps_current_training_codec(mode):
    from egomimic.rldb.zarr.arc_length_tokenizer import (
        TokenizeBimanualArcLengthCartesian,
    )

    decoder = BimanualIntervalArcDecoder(mode, 0.4, 20, 1 / 30, 40)
    assert type(decoder.codec) is TokenizeBimanualArcLengthCartesian
    source = np.zeros((80, 14))
    source[:, 0] = np.linspace(0, 0.5, len(source))
    source[:, 7] = np.linspace(0, 0.2, len(source))
    source[:, [6, 13]] = 0.5
    tokens = decoder.codec.transform({"actions_cartesian": source})["actions_cartesian"]
    expected = decoder.codec.detokenize(tokens, action_horizon=40)
    np.testing.assert_array_equal(decoder(tokens[None])[0], expected)
