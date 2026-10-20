"""Physical geometry and training/evaluation/deployment parity for ARC Projection."""

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from egomimic.pipeline.inference_config import (
    build_inference_config,
    inference_contract_sha256,
)
from egomimic.rldb.zarr.arc_length_tokenizer import cumulative_arc_length
from egomimic.rldb.zarr.arc_projection_tokenizer import TokenizeBimanualArcProjection
from egomimic.rldb.zarr.e1_arc_tokenizer import TokenizeBimanualArcLengthE1
from egomimic.robot.arc_decoder import BimanualArcDecoder
from scripts.e1.validate_stattempo_configs import compose_recipe


def codec(mode, **kwargs):
    return TokenizeBimanualArcProjection(
        velocity_mode=mode,
        resampled_vector_length=100,
        dt=1 / 30,
        **kwargs,
    )


def token(c, actions):
    return c.transform({"actions_cartesian": actions.copy()})["actions_cartesian"]


def rotation_chunk(rate=0.1):
    t = np.arange(100) / 30
    actions = np.zeros((100, 14))
    actions[:, 3] = t * rate
    actions[:, 6] = t / t[-1]
    # Second arm translates on its own stream at the same elapsed times.
    actions[:, 7] = t * 0.06
    actions[:, 13] = 1 - t / t[-1]
    return actions


def test_offset_is_in_local_axes_and_two_centimetres_per_component():
    c = codec("dur")
    pos = np.array([[1, 2, 3], [1, 2, 3]], dtype=float)
    ypr = np.array([[0, 0, 0], [np.pi / 2, 0, 0]])
    np.testing.assert_allclose(
        c.progress_coordinates(pos, ypr),
        pos + [[0.02, 0.02, 0.02], [-0.02, 0.02, 0.02]],
        atol=1e-14,
    )


@pytest.mark.parametrize("mode", ["dur", "profile"])
def test_pure_rotation_drives_clock_and_keeps_gripper_synchronized(mode):
    c = codec(mode, speed_smooth_frames=1)
    actions = rotation_chunk()
    values = token(c, actions)
    assert values.shape == (100, 16)
    # Stored positions are the stationary EEF, not the projected point.
    np.testing.assert_array_equal(values[:, :3], 0)
    assert np.max(values[:, 14]) > 0
    np.testing.assert_allclose(
        c.clock_at_waypoints(values)[0], np.arange(100) / 30, atol=1e-9
    )
    np.testing.assert_allclose(c.detokenize(values, 100), actions, atol=1e-9)
    # Legacy xyz-only ARC cannot see the turn at all.
    old = TokenizeBimanualArcLengthE1(velocity_mode=mode, resampled_vector_length=100)
    np.testing.assert_array_equal(old.detokenize(token(old, actions), 100)[:, 3], 0)


@pytest.mark.parametrize("mode", ["dur", "profile"])
def test_translation_only_matches_existing_codec(mode):
    actions = rotation_chunk()
    actions[:, 0] = np.linspace(0, 0.25, 100)
    actions[:, 3:6] = [0.4, -0.2, 0.3]
    c = codec(mode)
    old = TokenizeBimanualArcLengthE1(
        velocity_mode=mode, resampled_vector_length=100, min_speed=1e-6
    )
    values, expected = token(c, actions), token(old, actions)
    np.testing.assert_allclose(values, expected, atol=1e-12)
    np.testing.assert_allclose(
        c.detokenize(values, 100), old.detokenize(expected, 100), atol=1e-10
    )


@pytest.mark.parametrize("mode", ["dur", "profile"])
def test_projected_distance_budget_cuts_rotation_and_holds_endpoint(mode):
    c = codec(mode, min_distance_unit=0.02, speed_smooth_frames=1)
    actions = rotation_chunk(rate=1)
    values = token(c, actions)
    # A z turn sweeps a point at radius sqrt(.02**2 + .02**2).
    expected_angle = 0.02 / np.hypot(0.02, 0.02)
    assert values[-1, 3] == pytest.approx(expected_angle, abs=4e-5)
    # The sampled raw path and decoded waypoint path use polygonal chords at
    # different resolutions. Both approximate the circular arc within 0.1 ms.
    assert c.clock_at_waypoints(values)[0][-1] == pytest.approx(
        expected_angle, abs=1e-4
    )
    decoded = c.detokenize(values, 100)
    np.testing.assert_allclose(
        decoded[30:, :7], np.repeat(values[-1:, :7], 70, axis=0), atol=1e-9
    )


def test_mixed_pose_path_sets_supports_using_rotation_and_translation():
    c = codec("dur", min_distance_unit=0.06)
    actions = rotation_chunk(rate=0.7)
    actions[:, 0] = np.linspace(0, 0.08, 100)
    actions[:, 1] = 0.03 * np.sin(np.linspace(0, 2, 100))
    q = c.progress_coordinates(actions[:, :3], actions[:, 3:6])
    cum = cumulative_arc_length(q)
    frames = np.interp(np.linspace(0, min(0.06, cum[-1]), 100), cum, np.arange(100))
    values = token(c, actions)
    np.testing.assert_allclose(
        values[:, 0], np.interp(frames, np.arange(100), actions[:, 0]), atol=1e-12
    )
    np.testing.assert_allclose(values[:, 3], frames * 0.7 / 30, atol=1e-12)
    np.testing.assert_allclose(values[:, 6], frames / 99, atol=1e-12)
    np.testing.assert_allclose(values[:-1, 14], np.diff(frames) / 30, atol=1e-12)


def test_progress_is_invariant_under_rigid_change_of_reference_frame():
    c = codec("dur")
    actions = rotation_chunk(rate=0.7)
    actions[:, 0] = np.linspace(0, 0.2, 100)
    rot = Rotation.from_euler("ZYX", actions[:, 3:6])
    frame = Rotation.from_euler("ZYX", [0.7, -0.4, 0.2])
    q = c.progress_coordinates(actions[:, :3], actions[:, 3:6])
    transformed = c.progress_coordinates(
        frame.apply(actions[:, :3]) + [1, 2, 3], (frame * rot).as_euler("ZYX")
    )
    np.testing.assert_allclose(transformed, frame.apply(q) + [1, 2, 3], atol=1e-14)
    np.testing.assert_allclose(
        cumulative_arc_length(q), cumulative_arc_length(transformed), atol=1e-13
    )


@pytest.mark.parametrize("mode", ["dur", "profile"])
def test_idle_and_rotation_about_offset_are_finite_holds(mode):
    c = codec(mode)
    actions = np.zeros((100, 14))
    axis = np.ones(3) / np.sqrt(3)
    actions[:, 3:6] = Rotation.from_rotvec(
        np.linspace(0, 1, 100)[:, None] * axis
    ).as_euler("ZYX")
    q = c.progress_coordinates(actions[:, :3], actions[:, 3:6])
    np.testing.assert_allclose(q, np.full((100, 3), 0.02), atol=1e-14)
    decoded = c.detokenize(token(c, actions), 100)
    assert np.isfinite(decoded).all()
    np.testing.assert_array_equal(decoded, 0)


@pytest.mark.parametrize(
    "offset",
    [[0, 0, 0], [0.02, 0.02], [0.02, float("nan"), 0.02], [0, 0, float("inf")]],
)
def test_invalid_projection_is_rejected(offset):
    with pytest.raises(ValueError, match="projection_offset_m"):
        codec("dur", projection_offset_m=offset)


@pytest.mark.parametrize("mode", ["mean", "durhyb", "durtri", "logdur"])
def test_projection_rejects_unimplemented_timing(mode):
    with pytest.raises(ValueError, match="supports only"):
        codec(mode)


@pytest.mark.parametrize("variant,mode", [("dur", "dur"), ("vel", "profile")])
def test_recipe_training_evaluator_and_rollout_use_identical_projection(variant, mode):
    cfg = compose_recipe(
        f"bc_rl2_stattempo_slowpace366_arc_projection_{variant}_m200_d033"
    )
    OmegaConf.to_container(cfg, resolve=True)
    artifact = build_inference_config(cfg)
    assert artifact["status"] == "ready", artifact.get("reason")
    assert cfg.model.action_token_dim == 16
    assert cfg.model.embed_dim == 640
    assert cfg.model.trunk_blocks == 8
    assert cfg.model.flow_blocks == 10
    assert cfg.model.flow_hidden_dim == 384
    decoder = instantiate(cfg.model.inference.profiles.default.adapter.decoder)
    evaluator = instantiate(cfg.evaluator)
    assert isinstance(decoder, BimanualArcDecoder)
    # Declared with fastest-stream termination on; the whole-token comparison
    # below turns it off.
    assert decoder.first_stream == 1
    decoder.first_stream = 0
    assert evaluator.tick_prefix
    assert evaluator.arc_video_trajectory_cap_mode == "execution_horizon"
    actions = rotation_chunk(rate=0.7)
    for split in (cfg.data.train_datasets, cfg.data.valid_datasets):
        transforms = instantiate(split.yam_bimanual.resolver.transform_list)
        training = transforms[-1]
        assert isinstance(training, TokenizeBimanualArcProjection)
        assert training.velocity_mode == mode
        values = token(training, actions)
        expected = training.detokenize(values, 100)
        np.testing.assert_allclose(decoder(values)[0], expected, atol=1e-12)
        np.testing.assert_allclose(
            evaluator._decode_tick_prefix(values, 25), expected[:25], atol=1e-12
        )
        decoded, steps = evaluator._decode_prediction_with_steps(values)
        assert steps == 25
        np.testing.assert_allclose(decoded, expected[:25], atol=1e-12)
    evaluator.distance_dtw_enabled = False
    result = evaluator._compute_results(
        [
            dict(
                source="yam_bimanual",
                label="yam_bimanual",
                episode="projection",
                frame=0,
                prediction=values,
                ground_truth=expected,
            )
        ]
    )
    assert result["execute_control_steps"] == 25
    assert result["arc_execution_cap_mode"] is None
    assert result["replan_stride_mode"] == "fixed_control_frames"
    assert result["projection"]["offset_m"] == [0.02] * 3
    assert result["projection"]["min_speed_m_s"] == 1e-6
    assert (
        list(cfg.model.inference.compatibility.tokenizer.projection_offset_m)
        == [0.02] * 3
    )


def test_offset_override_is_shared_by_data_inference_and_evaluation():
    cfg = compose_recipe("bc_rl2_stattempo_slowpace366_arc_projection_dur_m200_d033")
    original_contract = inference_contract_sha256(cfg)
    cfg.model.projection_offset_m = [0.01, -0.03, 0.04]
    assert inference_contract_sha256(cfg) != original_contract
    training = instantiate(
        cfg.data.train_datasets.yam_bimanual.resolver.transform_list
    )[-1]
    decoder = instantiate(cfg.model.inference.profiles.default.adapter.decoder)
    evaluator = instantiate(cfg.evaluator)
    for c in (training, decoder.codec, evaluator._tick_tokenizer):
        np.testing.assert_array_equal(c.projection_offset_m, [0.01, -0.03, 0.04])
    artifact = build_inference_config(cfg)
    assert artifact["status"] == "ready"


def test_duration_and_velocity_have_distinct_checkpoint_contracts():
    configs = [
        compose_recipe(f"bc_rl2_stattempo_slowpace366_arc_projection_{v}_m200_d033")
        for v in ("dur", "vel")
    ]
    assert inference_contract_sha256(configs[0]) != inference_contract_sha256(
        configs[1]
    )


def test_plain_decoder_rejects_projection_parameters():
    with pytest.raises(ValueError, match="Projection layout"):
        BimanualArcDecoder(token_layout="e1_dur", projection_offset_m=[0.02] * 3)


@pytest.mark.parametrize("variant", ["dur", "vel"])
def test_projection_graph_can_train_and_sample(variant):
    torch.set_num_threads(1)
    cfg = compose_recipe(
        f"bc_rl2_stattempo_slowpace366_arc_projection_{variant}_m200_d033"
    )
    for name, value in dict(
        embed_dim=32,
        attention_heads=4,
        attention_head_dim=8,
        trunk_blocks=1,
        flow_blocks=1,
        flow_hidden_dim=32,
    ).items():
        cfg.model[name] = value
    cfg.model.graph.sampler.num_inference_steps = 2
    for stems in (
        cfg.model.graph.stem.stems,
        cfg.model.graph.stem.domain_stems.yam_bimanual,
    ):
        for stem in stems.values():
            if "weights" in stem:
                stem.weights = None
    training = instantiate(
        cfg.data.train_datasets.yam_bimanual.resolver.transform_list
    )[-1]
    actions = torch.tensor(token(training, rotation_chunk()), dtype=torch.float32)[
        None
    ].repeat(2, 1, 1)
    model = instantiate(cfg.model.pipeline, device="cpu")
    row = {
        "embodiment": torch.tensor([7, 7]),
        "observations.state.ee_pose": torch.randn(2, 1, 14),
        "actions_cartesian": actions,
    }
    for camera in ("front_img_1", "left_wrist_img", "right_wrist_img"):
        row["observations.images." + camera] = torch.rand(2, 1, 3, 32, 32)
    batch = {"opaque": row}
    result = model.forward_training(batch)
    loss = model.compute_losses(result, batch)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    model.nets.eval()
    with torch.no_grad():
        prediction = model.forward_eval(batch)["opaque"]["pred_action"]
    assert prediction.shape == (2, int(cfg.model.action_horizon), 16) == (2, 200, 16)
    assert torch.isfinite(prediction).all()
    decoder = instantiate(cfg.model.inference.profiles.default.adapter.decoder)
    decoder.first_stream = 0  # whole-token decode for the shape check below
    decoded = decoder(prediction)
    assert decoded.shape == (2, 100, 14)
    assert np.isfinite(decoded).all()


def test_projection_duration_defaults_to_absolute_isotonic_waypoint_times():
    cfg = compose_recipe("bc_rl2_stattempo_slowpace366_arc_projection_dur_m200_d033")
    evaluator = instantiate(cfg.evaluator)
    decoder = instantiate(cfg.model.inference.profiles.default.adapter.decoder)
    codecs = [
        instantiate(split.yam_bimanual.resolver.transform_list)[-1]
        for split in (cfg.data.train_datasets, cfg.data.valid_datasets)
    ] + [evaluator._e1_tokenizer(evaluator.token_layout, None), decoder.codec]
    assert {(c.velocity_mode, c.duration_timing) for c in codecs} == {
        ("dur", "absolute")
    }
    assert cfg.model.inference.native_output.timing.kind == "per_waypoint_absolute_time"
    assert cfg.model.inference.compatibility.tokenizer.duration_timing == "absolute"
    actions = rotation_chunk(rate=0.7)
    values = token(codecs[0], actions)
    np.testing.assert_allclose(values[1:, 14:], np.maximum.accumulate(values[1:, 14:]))
    delta = TokenizeBimanualArcProjection(
        velocity_mode="dur",
        min_distance_unit=0.33,
        resampled_vector_length=200,
        dt=1 / 30,
    )
    decoder.first_stream = 0  # compare whole tokens, as the identity test above does
    np.testing.assert_allclose(
        decoder(values)[0], delta.detokenize(token(delta, actions), 100), atol=1e-10
    )
    vel = compose_recipe("bc_rl2_stattempo_slowpace366_arc_projection_vel_m200_d033")
    assert vel.evaluator.duration_timing == "delta"
