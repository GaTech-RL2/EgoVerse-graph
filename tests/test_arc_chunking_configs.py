"""Source-window utilities and supported Cartesian recipe integration."""

import numpy as np
import pytest
from hydra.utils import instantiate

from egomimic.rldb.embodiment.human import Human
from egomimic.rldb.embodiment.yam import Yam
from egomimic.rldb.zarr.arc_length_tokenizer import TokenizeBimanualArcLengthCartesian
from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset
from scripts.audit_hydra_configs import CONFIGS, compose_for_audit

MODES = ("race", "multistream", "joint_distance")


def _dataset(total=90, right_stationary=False):
    frame = np.arange(total)
    poses = {}
    for side, step in (("left", 0.06), ("right", 0.02)):
        pose = np.zeros((total, 7))
        pose[:, 0] = frame * step
        if side == "right" and right_stationary:
            pose[:, 0] = 0.0
        pose[:, 3] = np.cos(frame * 0.05 / 2)
        pose[:, 6] = np.sin(frame * 0.05 / 2)
        poses[side] = pose

    class Reader:
        def __init__(self):
            self.ranges = []

        def read(self, ranges):
            self.ranges.append(ranges)
            return {key: poses[key][start:end] for key, (start, end) in ranges.items()}

    dataset = ZarrDataset.__new__(ZarrDataset)
    dataset.total_frames = total
    dataset.episode_reader = Reader()
    return dataset


def _spec(mode, rotation=0.19, **kwargs):
    return dict(
        type="arc_hybrid",
        distance=0.39,
        rotation_distance=rotation,
        source_buffer_frames=600,
        pose_zarr_keys=["left", "right"],
        arc_chunking_mode=mode,
        **kwargs,
    )


@pytest.mark.parametrize(
    "mode,expected", [("race", 8), ("multistream", 21), ("joint_distance", 6)]
)
def test_source_horizon_uses_selected_translation_clock(mode, expected):
    assert _dataset()._resolve_dynamic_horizon(0, _spec(mode)) == expected


@pytest.mark.parametrize(
    "mode,expected", [("race", 19), ("multistream", 21), ("joint_distance", 19)]
)
def test_source_horizon_covers_rotation_after_translation_cutoff(mode, expected):
    assert (
        _dataset()._resolve_dynamic_horizon(0, _spec(mode, rotation=0.89)) == expected
    )


@pytest.mark.parametrize(
    "mode,expected", [("race", 8), ("multistream", 90), ("joint_distance", 8)]
)
def test_stationary_arm_needs_bounded_fallback_only_for_multistream(mode, expected):
    assert (
        _dataset(right_stationary=True)._resolve_dynamic_horizon(0, _spec(mode))
        == expected
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "total,start,buffer,expected",
    [
        (900, 0, 600, 600),
        (90, 0, 12, 12),
        (14, 10, 600, 4),
        (14, 13, 600, 2),
        (14, 14, 600, 0),
    ],
)
def test_source_horizon_respects_buffer_episode_and_repeat_last_bounds(
    mode, total, start, buffer, expected
):
    dataset = _dataset(total)
    spec = _spec(mode, rotation=1000.0)
    spec["source_buffer_frames"] = buffer
    assert dataset._resolve_dynamic_horizon(start, spec) == expected
    for ranges in dataset.episode_reader.ranges:
        assert all(
            lo == start and hi <= min(total, start + buffer)
            for lo, hi in ranges.values()
        )


@pytest.mark.parametrize(
    "mode,expected", [("race", 8), ("multistream", 21), ("joint_distance", 6)]
)
def test_source_horizon_keeps_legacy_no_rotation_calls_available(mode, expected):
    spec = _spec(mode)
    spec["type"] = "arc_distance"
    del spec["rotation_distance"]
    assert _dataset()._resolve_dynamic_horizon(0, spec) == expected


def test_source_horizon_rejects_unknown_mode():
    with pytest.raises(ValueError, match="chunking_mode"):
        _dataset()._resolve_dynamic_horizon(0, _spec("unknown"))


def test_legacy_mode_omission_preserves_source_semantics():
    spec = _spec(None)
    assert _dataset()._resolve_dynamic_horizon(0, spec) == 6
    spec["type"] = "arc_distance"
    del spec["rotation_distance"]
    assert _dataset()._resolve_dynamic_horizon(0, spec) == 21
    spec["require_all_arms"] = False
    assert _dataset()._resolve_dynamic_horizon(0, spec) == 8
    assert (
        Human.get_keymap("arc_tokenizer_cartesian")["left.action_ee_pose"]["horizon"]
        == 600
    )
    assert (
        Yam.get_keymap("arc_tokenizer_cartesian", gripper_stream_mode="distance")[
            "left.cmd_ee_pose"
        ]["horizon"]["type"]
        == "arc_distance"
    )


RECIPES = sorted((CONFIGS / "experiment/abc_arc").glob("*/*.yaml"))


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda p: p.stem)
def test_retained_recipe_shapes_and_clocks_agree(recipe):
    with compose_for_audit(recipe) as cfg:
        evaluator = instantiate(cfg.evaluator)
        for dataset in cfg.data.train_datasets.values():
            transforms = instantiate(dataset.resolver.transform_list)
            codecs = [
                t
                for t in transforms
                if isinstance(t, TokenizeBimanualArcLengthCartesian)
            ]
            for codec in codecs:
                assert codec.shape[0] == cfg.abc.arc_waypoints
                assert codec.shape[1] == cfg.abc.arc_token_dim
                assert evaluator._codec().shape == codec.shape
                assert evaluator.clock_mode == codec.clock_mode
                assert evaluator.streams == codec.streams
                contract = cfg.run_provenance.action_contract
                assert contract.clock_mode == codec.clock_mode
                assert contract.streams == codec.streams
                assert contract.timing_layout == "first_delay_incoming_v1"
