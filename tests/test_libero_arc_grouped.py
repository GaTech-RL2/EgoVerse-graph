import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from egomimic.rldb.zarr.libero_arc_grouped import LiberoArcGroupedCodec
from egomimic.rldb.zarr.libero_arc_timed import LiberoArcTimedCodec

SPECS = [
    {},
    {"gripper": "separate"},
    {"gripper": "separate", "translation_groups": [[0], [1, 2]]},
    {"gripper": "separate", "translation_groups": [[0], [1], [2]]},
    {"gripper": "separate", "timing": "component"},
    {"gripper": "separate", "timing": "component", "clocks": "component"},
    {"gripper": "separate", "rotation_representation": "angular_driver"},
    {
        "gripper": "separate",
        "rotation_representation": "angular_driver",
        "translation_groups": [[0], [1], [2]],
        "rotation_groups": [[0], [1], [2]],
    },
]


@pytest.mark.parametrize("mode", ["stk", "dur"])
def test_reference_is_legacy_codec_with_only_a_column_permutation(mode):
    actions = np.random.default_rng(14).uniform(-0.8, 0.8, (32, 7))
    actions[:, 6] = np.where(actions[:, 6] > 0, 1, -1)
    legacy = LiberoArcTimedCodec(
        mode=mode, num_waypoints=36, max_translation=0.8, max_rotation_degrees=192
    )
    new = LiberoArcGroupedCodec(
        {}, mode=mode, num_waypoints=36, max_translation=0.8, max_rotation_degrees=192
    )
    old_tokens = legacy.encode(actions)
    new_tokens = new.encode(actions)
    permutation = [0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 3, 10]
    np.testing.assert_allclose(new_tokens, old_tokens[:, permutation], atol=2e-7)
    np.testing.assert_allclose(
        new.decode(new_tokens), legacy.decode(old_tokens), atol=2e-6
    )


@pytest.mark.parametrize("mode", ["stk", "dur"])
@pytest.mark.parametrize("spec", SPECS)
def test_constant_diagonal_commands_and_rotation_beyond_pi(spec, mode):
    codec = LiberoArcGroupedCodec(spec, mode=mode, num_waypoints=36)
    actions = np.tile([0.2, -0.3, 0.1, 0.3, -0.2, 0.4, -1], (32, 1))
    tokens = codec.encode(actions)
    assert tokens.shape == (36, codec.action_dim)
    np.testing.assert_allclose(codec.decode(tokens), actions, atol=8e-6)
    assert codec.represented_seconds(tokens) == pytest.approx(1.6, abs=2e-6)


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("mode", ["stk", "dur"])
def test_constant_null_action_holds_the_gripper_without_movement(spec, mode):
    actions = np.zeros((32, 7))
    actions[:, 6] = -1
    codec = LiberoArcGroupedCodec(spec, mode=mode, num_waypoints=36)
    np.testing.assert_allclose(codec.decode(codec.encode(actions)), actions, atol=1e-7)


@pytest.mark.parametrize("mode", ["stk", "dur"])
def test_gripper_event_distance_preserves_stationary_switch_times(mode):
    actions = np.zeros((32, 7))
    actions[:, 6] = -1
    actions[11:23, 6] = 1
    codec = LiberoArcGroupedCodec({"gripper": "separate"}, mode=mode, num_waypoints=36)
    np.testing.assert_allclose(codec.decode(codec.encode(actions)), actions, atol=1e-7)


def test_gripper_events_exceeding_capacity_are_reportable_loss_not_hidden_time():
    actions = np.zeros((32, 7))
    actions[:, 6] = np.where(np.arange(32) % 2, 1, -1)
    codec = LiberoArcGroupedCodec({"gripper": "separate"}, mode="dur", num_waypoints=4)
    tokens = codec.encode(actions)
    assert tokens.shape == (4, 13)
    assert np.isfinite(codec.decode(tokens)).all()
    assert np.count_nonzero(codec.decode(tokens)[:, 6] != actions[:, 6]) > 0


def test_small_rotation_is_preserved_below_R_in_each_replanned_window():
    actions = np.zeros((32, 7))
    actions[:, 0] = 0.2
    actions[:, 5] = 1e-3
    for mode in ("stk", "dur"):
        codec = LiberoArcGroupedCodec(
            {}, mode=mode, num_waypoints=36, max_rotation_degrees=384
        )
        decoded = codec.decode(codec.encode(actions))
        np.testing.assert_allclose(decoded[:16, 3:6], actions[:16, 3:6], atol=1e-7)
        assert decoded[:16, 5].sum() == pytest.approx(0.016, abs=1e-7)


def test_duration_component_clocks_preserve_noncommuting_rotation():
    # Equal arc steps put supports exactly on input frames. Holding XYZ does
    # not hold the wrist, and independent angular clocks must compose in order.
    actions = np.zeros((32, 7))
    actions[:, :3] = [0.1, -0.1, 0.1]
    actions[np.arange(32), 3 + np.arange(32) % 3] = 0.3
    actions[:, 6] = -1
    spec = {"gripper": "separate", "timing": "component", "clocks": "component"}
    codec = LiberoArcGroupedCodec(spec, mode="dur", num_waypoints=33)
    decoded = codec.decode(codec.encode(actions))
    np.testing.assert_allclose(decoded, actions, atol=3e-6)
    wrong = Rotation.from_rotvec(actions[:, 3:6].sum(0) * 0.5)
    right = Rotation.identity()
    for action in decoded:
        right = Rotation.from_rotvec(action[3:6] * 0.5) * right
    assert (right * wrong.inv()).magnitude() > 0.05


def test_per_component_time_changes_only_the_addressed_shape_component():
    spec = {"gripper": "separate", "timing": "component", "clocks": "component"}
    codec = LiberoArcGroupedCodec(spec, mode="dur", num_waypoints=36)
    actions = np.tile([0.2, -0.1, 0.1, 0, 0, 0, -1], (32, 1))
    tokens = codec.encode(actions)
    tokens[:, codec.timing_slices["xyz0"].start] *= 2
    actual = codec.decode(tokens)
    np.testing.assert_allclose(actual[:, 0], 0.1, atol=3e-6)
    np.testing.assert_allclose(actual[:, 1:], actions[:, 1:], atol=3e-6)


def test_duration_keeps_component_dwells_but_velocity_has_no_hidden_time():
    spec = {"gripper": "separate", "timing": "component", "clocks": "component"}
    actions = np.zeros((32, 7))
    actions[:, 1] = 0.2
    actions[8:24, 0] = 0.2
    dur = LiberoArcGroupedCodec(spec, mode="dur", num_waypoints=36)
    stk = LiberoArcGroupedCodec(spec, mode="stk", num_waypoints=36)
    # Group support sampling is still shared. Independent STK timing loses
    # the initial x-only dwell even though the y stream keeps moving.
    assert dur.clocks(dur.encode(actions))["xyz0"][-1, 0] == pytest.approx(1.6)
    assert stk.clocks(stk.encode(actions))["xyz0"][-1, 0] < 1.0


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("mode", ["stk", "dur"])
def test_normalization_contains_all_legal_diagonal_commands(spec, mode):
    codec = LiberoArcGroupedCodec(spec, mode=mode, num_waypoints=36)
    actions = np.ones((32, 7))
    actions[:, 6] = np.where(np.arange(32) % 2, 1, -1)
    normalized = codec.encode(actions) / codec.token_scale()
    assert np.abs(normalized).max() <= 1 + 2e-6


@pytest.mark.parametrize(
    "spec",
    [
        {"translation_groups": [[0, 1], [1, 2]]},
        {"rotation_groups": [[0], [1], [2]]},
        {"timing": "group", "clocks": "component"},
        {"carry": True},
    ],
)
def test_ambiguous_or_invalid_geometry_is_rejected(spec):
    with pytest.raises(ValueError):
        LiberoArcGroupedCodec(spec)


def test_graph_cache_and_representation_record_the_stream_layout():
    import torch

    from egomimic.pipeline.stages_libero_arc import LiberoArcStage

    stage = LiberoArcStage(
        arc_mode="dur",
        stream_spec={"gripper": "separate"},
        num_waypoints=36,
        encode_cache_size=4,
        reconstruction=True,
    )
    actions = torch.zeros((2, 32, 7))
    actions[:, :, 6] = -1
    actions[:, 12:, 6] = 1
    output = stage.execute({"actions": actions}, mode="inference")
    torch.testing.assert_close(output["pred_action"], actions)
    assert output["target"].shape == (2, 36, 13)
    assert len(stage._encode_cache) == 1
    assert stage.representation_context()["layout"] == "shape_then_timing"
