import numpy as np
import pytest

from egomimic.rldb.zarr.libero_arc import LiberoArcCodec


@pytest.mark.parametrize(
    "motion", ["translation", "rotation", "gripper", "mixed", "hold", "loop"]
)
def test_dense_arc_roundtrip_all_seven_control_channels(motion):
    n = 32
    codec = LiberoArcCodec(num_waypoints=n + 1, horizon=n)
    actions = np.zeros((n, 7))
    actions[:, 6] = -1
    if motion in ("translation", "mixed"):
        actions[:, :3] = np.random.default_rng(3).uniform(-0.3, 0.3, (n, 3))
    if motion in ("rotation", "mixed"):
        actions[:, 3:6] = np.random.default_rng(4).uniform(-0.5, 0.5, (n, 3))
    if motion in ("gripper", "mixed"):
        actions[12:, 6] = 1
    if motion == "loop":
        phase = np.linspace(0, 2 * np.pi, n + 1)
        actions[:, :2] = np.diff(
            np.column_stack((np.cos(phase), np.sin(phase))), axis=0
        )
    tokens = codec.encode(actions)
    assert np.isfinite(tokens).all()
    assert tokens[:-1, 10].sum() == pytest.approx(n * codec.dt)
    np.testing.assert_allclose(codec.decode(tokens), actions, atol=8e-6)


def test_holds_keep_duration_and_zero_command_intervals():
    codec = LiberoArcCodec(num_waypoints=8)
    actions = np.zeros((32, 7))
    actions[:, 6] = -1
    actions[8:16, 0] = 0.1
    actions[24:28, 0] = 0.1
    decoded = codec.decode(codec.encode(actions))
    np.testing.assert_allclose(decoded[:8, :6], 0, atol=2e-7)
    np.testing.assert_allclose(decoded[16:24, :6], 0, atol=2e-7)
    np.testing.assert_allclose(decoded[28:, :6], 0, atol=2e-7)
    np.testing.assert_allclose(decoded[:, 0].sum(), actions[:, 0].sum(), atol=2e-6)


def test_pure_rotation_crosses_pi_and_preserves_direction():
    codec = LiberoArcCodec(num_waypoints=16)
    actions = np.zeros((32, 7))
    actions[:, 5] = 0.25
    decoded = codec.decode(codec.encode(actions))
    np.testing.assert_allclose(decoded[:, :3], 0, atol=1e-7)
    np.testing.assert_allclose(decoded[:, 3:6], actions[:, 3:6], atol=2e-6)


def test_time_scale_changes_duration_without_changing_arc_geometry():
    actions = np.random.default_rng(71).uniform(-0.2, 0.2, (32, 7))
    actions[:, 6] = -1
    actions[16:, 6] = 1
    original = LiberoArcCodec(num_waypoints=16, dt=0.05)
    slower = LiberoArcCodec(num_waypoints=16, dt=0.1)
    tokens, slow_tokens = original.encode(actions), slower.encode(actions)
    np.testing.assert_allclose(slow_tokens[:, :10], tokens[:, :10], atol=1e-7)
    np.testing.assert_allclose(slow_tokens[:, 10], 2 * tokens[:, 10], atol=1e-7)
    np.testing.assert_allclose(
        slower.decode(slow_tokens), original.decode(tokens), atol=1e-6
    )


def test_gripper_only_transitions_and_short_prefixes_are_distinct():
    codec = LiberoArcCodec(num_waypoints=8)
    actions = np.zeros((32, 7))
    actions[:, 6] = -1
    actions[10:, 6] = 1
    np.testing.assert_allclose(codec.decode(codec.encode(actions)), actions, atol=2e-6)


def test_predicted_degenerate_rotation_and_negative_duration_are_finite():
    codec = LiberoArcCodec(num_waypoints=4)
    tokens = np.zeros((4, 11))
    tokens[:, 10] = [-1, 0, 0.2, 0]
    assert np.isfinite(codec.decode(tokens)).all()


def test_arc_stage_decodes_bfloat16_predictions():
    import torch

    from egomimic.pipeline.stages_libero_arc import LiberoArcStage

    stage = LiberoArcStage(operation="decode", num_waypoints=4)
    predicted = torch.zeros(2, 4, 11, dtype=torch.bfloat16)
    predicted[:, :, 10] = 0.25
    result = stage.execute({"pred_arc": predicted}, mode="inference")
    assert result["pred_action"].shape == (2, 32, 7)
    assert torch.isfinite(result["pred_action"]).all()


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_bad_commands_fail_without_replacement(bad):
    actions = np.zeros((32, 7))
    actions[2, 3] = bad
    with pytest.raises(ValueError, match="finite"):
        LiberoArcCodec().encode(actions)


def test_batch_stage_unnormalizes_once_and_restores_normalized_actions():
    from types import SimpleNamespace

    import torch

    from egomimic.pipeline.algo import PipelineAlgo
    from egomimic.pipeline.stages_libero_arc import LiberoArcStage
    from egomimic.rldb.zarr.libero_dataset import EMBODIMENT

    scale, offset = (
        np.arange(1, 8, dtype=np.float32),
        np.arange(7, dtype=np.float32) / 5,
    )
    norm = SimpleNamespace(
        norm_stats={EMBODIMENT: {"actions": {"scale": scale, "offset": offset}}},
        to_state=lambda: {},
        tokenizer_context=lambda: {},
    )
    algo = PipelineAlgo(
        [LiberoArcStage(num_waypoints=33, reconstruction=True)], device="cpu"
    )
    algo.bind_data_context(normalizer=norm)
    native = torch.randn(2, 32, 7) / 5
    normalized = native * torch.tensor(scale) + torch.tensor(offset)
    result = algo.forward_eval({"libero_panda": {"actions": normalized}})
    torch.testing.assert_close(
        result["libero_panda"]["pred_action"], normalized, atol=1e-5, rtol=1e-5
    )


def test_distance_cap_keeps_original_clock_and_fractional_final_command():
    codec = LiberoArcCodec(num_waypoints=36, max_translation=0.055)
    actions = np.zeros((32, 7))
    actions[:, 0] = 0.2  # 1 cm per control interval.
    actions[:, 6] = -1
    tokens = codec.encode(actions)
    assert tokens[:, 10].sum() == pytest.approx(0.275)
    decoded = codec.decode(tokens)
    np.testing.assert_allclose(decoded[:5, 0], 0.2, atol=1e-6)
    assert decoded[5, 0] == pytest.approx(0.1, abs=1e-6)
    np.testing.assert_allclose(decoded[6:, :6], 0, atol=1e-6)
    np.testing.assert_allclose(decoded[:, 6], -1)


def test_rotation_horizon_is_degrees_and_works_without_translation():
    actions = np.zeros((32, 7))
    actions[:, 5] = np.deg2rad(3) / 0.5
    codec = LiberoArcCodec(num_waypoints=36, max_rotation_degrees=12)
    tokens = codec.encode(actions)
    assert tokens[:, 10].sum() == pytest.approx(4 * codec.dt)
    decoded = codec.decode(tokens)
    np.testing.assert_allclose(decoded[:4, :6], actions[:4, :6], atol=2e-6)
    np.testing.assert_allclose(decoded[4:, :6], 0, atol=2e-6)


def test_first_budget_crossing_preserves_leading_hold():
    actions = np.zeros((32, 7))
    actions[8:, 0] = 0.2
    actions[8:, 5] = np.deg2rad(4) / 0.5
    codec = LiberoArcCodec(
        num_waypoints=36, max_translation=0.02, max_rotation_degrees=12
    )
    tokens = codec.encode(actions)
    assert tokens[:, 10].sum() == pytest.approx(10 * codec.dt)
    decoded = codec.decode(tokens)
    np.testing.assert_allclose(decoded[:10, :6], actions[:10, :6], atol=2e-6)
    np.testing.assert_allclose(decoded[10:, :6], 0, atol=2e-6)


@pytest.mark.parametrize("parameter", ["max_translation", "max_rotation_degrees"])
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_invalid_horizon_budgets_rejected(parameter, value):
    with pytest.raises(ValueError, match="budgets"):
        LiberoArcCodec(**{parameter: value})
