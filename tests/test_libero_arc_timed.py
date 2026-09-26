import numpy as np
import pytest

from egomimic.rldb.zarr.libero_arc_timed import LiberoArcTimedCodec, _support_frames


@pytest.mark.parametrize("mode", ["stk", "dur"])
@pytest.mark.parametrize("motion", ["translation", "rotation", "mixed", "stationary"])
def test_constant_controls_use_independent_stream_clocks(mode, motion):
    actions = np.zeros((32, 7))
    actions[:, 6] = -1
    if motion in {"translation", "mixed"}:
        actions[:, :3] = [0.2, -0.1, 0.05]
    if motion in {"rotation", "mixed"}:
        actions[:, 5] = 0.25  # Crosses pi in world-frame accumulated SO(3).
    codec = LiberoArcTimedCodec(mode=mode, num_waypoints=16)
    tokens = codec.encode(actions)
    assert tokens.shape == (16, 12)
    np.testing.assert_allclose(codec.decode(tokens), actions, atol=3e-6)
    assert codec.represented_seconds(tokens) == pytest.approx(1.6, abs=1e-6)


@pytest.mark.parametrize("mode", ["stk", "dur"])
def test_rotation_budget_does_not_shorten_translation_or_grip(mode):
    actions = np.zeros((32, 7))
    actions[:, 0] = 0.2
    actions[:, 5] = np.deg2rad(3) / 0.5
    actions[:, 6] = -1
    codec = LiberoArcTimedCodec(
        mode=mode, num_waypoints=16, max_translation=0.2, max_rotation_degrees=12
    )
    tokens = codec.encode(actions)
    t_clock, r_clock = codec.clocks(tokens)
    assert t_clock[-1] == pytest.approx(1.0, abs=1e-6)
    assert r_clock[-1] == pytest.approx(0.2, abs=1e-6)
    decoded = codec.decode(tokens)
    np.testing.assert_allclose(decoded[:20, 0], 0.2, atol=3e-6)
    np.testing.assert_allclose(decoded[20:, 0], 0, atol=3e-6)
    np.testing.assert_allclose(decoded[:4, 5], actions[:4, 5], atol=3e-6)
    np.testing.assert_allclose(decoded[4:, 5], 0, atol=3e-6)


@pytest.mark.parametrize("mode", ["stk", "dur"])
def test_noncommuting_rotations_use_world_frame_osc_convention(mode):
    # Equal angular increments make the uniform arc supports coincide with
    # control frames; changing axes detects reversed composition conventions.
    actions = np.zeros((16, 7))
    actions[:, 0] = 0.1
    for index in range(16):
        actions[index, 3 + index % 3] = 0.2
    actions[:, 6] = -1
    actions[8:, 6] = 1
    codec = LiberoArcTimedCodec(mode=mode, horizon=16, num_waypoints=17)
    np.testing.assert_allclose(codec.decode(codec.encode(actions)), actions, atol=3e-6)


@pytest.mark.parametrize("mode", ["stk", "dur"])
def test_each_budget_preserves_its_fractional_final_command(mode):
    actions = np.zeros((32, 7))
    actions[:, 0] = 0.2
    actions[:, 5] = np.deg2rad(3) / 0.5
    codec = LiberoArcTimedCodec(
        mode=mode, num_waypoints=16, max_translation=0.055, max_rotation_degrees=13.5
    )
    decoded = codec.decode(codec.encode(actions))
    np.testing.assert_allclose(decoded[:5, 0], 0.2, atol=3e-6)
    assert decoded[5, 0] == pytest.approx(0.1, abs=3e-6)
    np.testing.assert_allclose(decoded[6:, 0], 0, atol=3e-6)
    np.testing.assert_allclose(decoded[:4, 5], actions[:4, 5], atol=3e-6)
    assert decoded[4, 5] == pytest.approx(actions[4, 5] / 2, abs=3e-6)
    np.testing.assert_allclose(decoded[5:, 5], 0, atol=3e-6)


def test_duration_preserves_independent_stop_and_go_intervals():
    actions = np.zeros((32, 7))
    actions[:, 6] = -1
    actions[8:16, 0] = 0.2
    actions[24:28, 0] = 0.2
    actions[4:12, 5] = 0.2
    actions[20:24, 5] = -0.2
    codec = LiberoArcTimedCodec(mode="dur", num_waypoints=16)
    np.testing.assert_allclose(codec.decode(codec.encode(actions)), actions, atol=3e-6)


def test_velocity_cannot_encode_grip_only_dwell_but_duration_can():
    actions = np.zeros((32, 7))
    actions[:, 6] = -1
    actions[16:, 6] = 1
    duration = LiberoArcTimedCodec(mode="dur", num_waypoints=33)
    velocity = LiberoArcTimedCodec(mode="stk", num_waypoints=33)
    np.testing.assert_allclose(
        duration.decode(duration.encode(actions)), actions, atol=3e-6
    )
    tokens = velocity.encode(actions)
    assert velocity.represented_seconds(tokens) == 0
    assert np.mean(velocity.decode(tokens)[:, 6] != actions[:, 6]) > 0


@pytest.mark.parametrize("mode,scale", [("dur", 2.0), ("stk", 0.5)])
def test_time_scaling_changes_only_timing_channels(mode, scale):
    actions = np.random.default_rng(33).uniform(-0.2, 0.2, (32, 7))
    fast = LiberoArcTimedCodec(mode=mode, dt=0.05)
    slow = LiberoArcTimedCodec(mode=mode, dt=0.1)
    a, b = fast.encode(actions), slow.encode(actions)
    np.testing.assert_allclose(
        b[:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 11]], a[:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 11]]
    )
    np.testing.assert_allclose(b[:, [3, 10]], a[:, [3, 10]] * scale)
    np.testing.assert_allclose(slow.decode(b), fast.decode(a), atol=3e-6)


def test_uniform_sampling_and_longest_dwell_reservation_match_native_contract():
    arc = np.array([0, 0, 0, 1, 2, 3, 3, 3, 3, 4], dtype=float)
    np.testing.assert_allclose(_support_frames(arc, 4, 5, "stk"), [0, 3, 4, 5, 9])
    # Four supports reserve the longest hold; five can retain both holds.
    np.testing.assert_allclose(_support_frames(arc, 4, 4, "dur"), [0, 5, 8, 9])
    np.testing.assert_allclose(_support_frames(arc, 4, 5, "dur"), [0, 2, 5, 8, 9])


@pytest.mark.parametrize("mode", ["stk", "dur"])
def test_degenerate_predicted_tokens_stall_instead_of_teleport(mode):
    codec = LiberoArcTimedCodec(mode=mode, num_waypoints=4)
    tokens = np.zeros((4, 12))
    tokens[1:, 0] = 1
    decoded = codec.decode(tokens)
    assert np.isfinite(decoded).all()
    # A missing clock is longer than the output horizon, not an immediate jump.
    assert abs(decoded[0, 0]) < 1
    assert decoded[:, 0].sum() * codec.translation_scale < 1


@pytest.mark.parametrize("mode", ["stk", "dur"])
def test_native_graph_normalizes_each_variant_and_decodes_bfloat16(mode):
    import torch

    from egomimic.pipeline.stages_libero_arc import LiberoArcStage

    stage = LiberoArcStage(arc_mode=mode, num_waypoints=16, reconstruction=True)
    stage.action_scale.copy_(torch.arange(1, 8))
    stage.action_offset.copy_(torch.arange(7) / 5)
    native = torch.zeros(2, 32, 7)
    native[:, :, 0] = 0.1
    native[:, :, 5] = 0.2
    native[:, :, 6] = -1
    normalized = native * stage.action_scale + stage.action_offset
    result = stage.execute({"actions": normalized}, mode="inference")
    torch.testing.assert_close(result["pred_action"], normalized, atol=1e-5, rtol=1e-5)
    predicted = result["target"].to(torch.bfloat16)
    decoder = LiberoArcStage(arc_mode=mode, num_waypoints=16, operation="decode")
    output = decoder.execute({"pred_arc": predicted}, mode="inference")["pred_action"]
    assert output.shape == (2, 32, 7)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("mode", ["stk", "dur"])
def test_mode_specific_replay_grid_and_policy_shape(mode):
    import torch

    from egomimic.benchmarks.libero.replay import (
        candidates_from_spec,
        reconstruct_episode,
    )
    from egomimic.models.denoising_nets import ConditionalUnet1D

    spec = dict(
        arc_mode=mode,
        horizon=32,
        execute_steps=16,
        dt=0.05,
        rotation_degrees=[12],
        translation_metres=[0.4],
        waypoints=[4, 8, 16, 24, 32, 36],
    )
    candidates = candidates_from_spec(spec)
    model = ConditionalUnet1D(
        input_dim=12,
        cond_dim=276,
        ac_latent_seq=1,
        diffusion_step_embed_dim=16,
        down_dims=[8, 16, 32],
    )
    for name, candidate in candidates.items():
        assert name.startswith(mode + "_")
        inputs = torch.randn(2, candidate["num_waypoints"], 12, requires_grad=True)
        output = model(inputs, torch.tensor([1, 3]), torch.randn(2, 276))
        assert output.shape == inputs.shape
        output.square().mean().backward()
        assert torch.isfinite(inputs.grad).all()
    actions = np.zeros((73, 7))
    actions[:, 0] = 0.1
    decoded, metrics = reconstruct_episode(actions, candidate, spec)
    np.testing.assert_allclose(decoded, actions, atol=3e-6)
    assert metrics["execution_coverage"] == pytest.approx(1)


def test_velocity_targets_fit_diffusion_clipping_for_diagonal_osc_commands():
    import torch

    from egomimic.pipeline.stages_libero_arc import LiberoArcStage

    actions = torch.ones(1, 32, 7)
    stage = LiberoArcStage(
        arc_mode="stk", num_waypoints=33, velocity_norm_bound=np.sqrt(3)
    )
    tokens = stage.execute({"actions": actions}, mode="train")["target"]
    assert float(tokens.abs().max()) <= 1 + 1e-6
    # Clipping to the scheduler's [-1,1] range must retain all legal diagonal
    # speeds. A per-component scale instead clips these magnitudes by sqrt(3).
    decoder = LiberoArcStage(
        arc_mode="stk",
        num_waypoints=33,
        operation="decode",
        velocity_norm_bound=np.sqrt(3),
    )
    decoded = decoder.execute({"pred_arc": tokens.clamp(-1, 1)}, mode="inference")
    torch.testing.assert_close(decoded["pred_action"], actions, atol=5e-6, rtol=5e-6)
