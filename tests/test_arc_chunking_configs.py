"""ARC representation selection from Hydra through embodiment/source loading."""

from pathlib import Path

import hydra
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.embodiment.human import Human
from egomimic.rldb.embodiment.yam import Yam
from egomimic.rldb.zarr.action_chunk_transforms import InterpolatePose
from egomimic.rldb.zarr.arc_length_tokenizer import TokenizeBimanualArcLengthCartesian
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset, ZarrDataset


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "egomimic/hydra_configs"
MODES = ("race", "multistream", "joint_distance")
RECIPES = sorted(
    str(path.relative_to(CONFIG_ROOT / "experiment").with_suffix(""))
    for population in ("robot_bc", "human_bc")
    for path in (CONFIG_ROOT / "experiment/abc_arc" / population).glob("*.yaml")
)


def _compose(recipe, mode=None):
    overrides = [f"+experiment={recipe}"]
    if mode is not None:
        overrides.append(f"abc.arc_chunking_mode={mode}")
    if recipe.endswith("abc_visual_hpt300_base"):
        overrides.append("abc.task_predicate=True")
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
        return compose(config_name="train_zarr_cartesian", overrides=overrides)


@pytest.mark.parametrize("recipe", RECIPES)
@pytest.mark.parametrize("mode", MODES)
def test_retained_visual_recipes_forward_mode_and_caps(recipe, mode, monkeypatch):
    monkeypatch.setenv("EGOVERSE_ABC_DATASET_DIR", "/tmp/unused-arc-config-data")
    cfg = _compose(recipe, mode)
    assert cfg.abc.arc_chunking_mode == mode
    assert cfg.evaluator.arc_chunking_mode == mode
    for split in ("train_datasets", "valid_datasets"):
        for dataset in cfg.data[split].values():
            resolver = dataset.resolver
            keymap = hydra.utils.instantiate(resolver.key_map)
            transforms = hydra.utils.instantiate(resolver.transform_list)
            tokenizers = [t for t in transforms if isinstance(t, TokenizeBimanualArcLengthCartesian)]
            if "hybrid" not in recipe:
                assert not tokenizers
                assert all(not isinstance(s.get("horizon"), dict) for s in keymap.values())
                continue
            assert len(tokenizers) == 1
            tokenizer = tokenizers[0]
            assert tokenizer.arc_chunking_mode == mode
            assert tokenizer.velocity_mode == "per_waypoint"
            assert cfg.abc.arc_rotation_distance > 0
            assert tokenizer.rotation_distance_unit == pytest.approx(cfg.abc.arc_rotation_distance)
            for spec in keymap.values():
                if spec.get("key_type") == "action_keys":
                    assert spec["horizon"]["arc_chunking_mode"] == mode
                    assert spec["horizon"]["distance"] == cfg.abc.arc_distance
                    assert spec["horizon"]["rotation_distance"] == cfg.abc.arc_rotation_distance
                    assert spec["horizon"]["source_buffer_frames"] == 600


@pytest.mark.parametrize("recipe", RECIPES)
def test_visual_recipes_default_to_joint_distance(recipe):
    assert _compose(recipe).abc.arc_chunking_mode == "joint_distance"


@pytest.mark.parametrize("embodiment", (Yam, Human, Eva))
@pytest.mark.parametrize("mode", MODES)
def test_explicit_embodiment_mode_keeps_native_source_rows_and_rotation(embodiment, mode):
    kwargs = dict(
        min_distance_unit=0.39,
        rotation_distance_unit=0.89,
        arc_chunking_mode=mode,
    )
    keymap = embodiment.get_keymap("hybrid_arc_tokenizer_cartesian", **kwargs)
    transforms = embodiment.get_transform_list(
        action_mode="hybrid_arc_tokenizer_cartesian", coord_frame="eef_frame",
        velocity_mode="per_waypoint", **kwargs,
    )
    tokenizer = next(t for t in transforms if isinstance(t, TokenizeBimanualArcLengthCartesian))
    assert tokenizer.arc_chunking_mode == mode
    assert tokenizer.rotation_distance_unit == 0.89
    assert tokenizer.tokenizer.config.dt == pytest.approx(1 / 30)
    # A dynamic source window must reach the tokenizer at native cadence.
    raw = np.zeros((23, 7))
    raw[:, 3] = 1.0
    raw[:, 0] = np.arange(23) * 0.01
    for transform in transforms:
        if isinstance(transform, InterpolatePose):
            result = transform.transform({transform.action_key: raw.copy()})
            np.testing.assert_array_equal(result[transform.output_action_key], raw)
    horizons = [s["horizon"] for s in keymap.values() if s.get("key_type") == "action_keys"]
    assert horizons
    assert all(s["arc_chunking_mode"] == mode and s["rotation_distance"] == 0.89 for s in horizons)


@pytest.mark.parametrize("embodiment", (Yam, Human, Eva))
def test_legacy_embodiment_omission_reaches_codec_as_none(embodiment):
    action_mode = (
        "arc_tokenizer_cartesian_gripper_padded"
        if embodiment is Human else "arc_tokenizer_cartesian"
    )
    transforms = embodiment.get_transform_list(action_mode=action_mode)
    tokenizer = next(t for t in transforms if isinstance(t, TokenizeBimanualArcLengthCartesian))
    # The codec rejects ANY explicit mode without R, so successful construction
    # proves the helper preserved None instead of forwarding inferred multistream.
    assert tokenizer.rotation_distance_unit is None
    assert tokenizer.arc_chunking_mode == "multistream"


@pytest.mark.parametrize("embodiment", (Yam, Human, Eva))
@pytest.mark.parametrize("mode", MODES)
def test_explicit_embodiment_modes_require_rotation(embodiment, mode):
    action_mode = (
        "arc_tokenizer_cartesian_gripper_padded"
        if embodiment is Human else "arc_tokenizer_cartesian"
    )
    with pytest.raises(ValueError, match="rotation_distance_unit"):
        embodiment.get_transform_list(
            action_mode=action_mode, arc_chunking_mode=mode, velocity_mode="per_waypoint"
        )


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
        type="arc_hybrid", distance=0.39, rotation_distance=rotation,
        source_buffer_frames=600, pose_zarr_keys=["left", "right"],
        arc_chunking_mode=mode, **kwargs,
    )


@pytest.mark.parametrize("mode,expected", [("race", 8), ("multistream", 21), ("joint_distance", 6)])
def test_source_horizon_uses_selected_translation_clock(mode, expected):
    assert _dataset()._resolve_dynamic_horizon(0, _spec(mode)) == expected


@pytest.mark.parametrize("mode,expected", [("race", 10), ("multistream", 21), ("joint_distance", 10)])
def test_source_horizon_covers_rotation_after_translation_cutoff(mode, expected):
    assert _dataset()._resolve_dynamic_horizon(0, _spec(mode, rotation=0.89)) == expected


@pytest.mark.parametrize("mode,expected", [("race", 8), ("multistream", 90), ("joint_distance", 8)])
def test_stationary_arm_needs_bounded_fallback_only_for_multistream(mode, expected):
    assert _dataset(right_stationary=True)._resolve_dynamic_horizon(0, _spec(mode)) == expected


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("total,start,buffer,expected", [(900, 0, 600, 600), (90, 0, 12, 12), (14, 10, 600, 4), (14, 13, 600, 2), (14, 14, 600, 0)])
def test_source_horizon_respects_buffer_episode_and_repeat_last_bounds(mode, total, start, buffer, expected):
    dataset = _dataset(total)
    spec = _spec(mode, rotation=1000.0)
    spec["source_buffer_frames"] = buffer
    assert dataset._resolve_dynamic_horizon(start, spec) == expected
    for ranges in dataset.episode_reader.ranges:
        assert all(lo == start and hi <= min(total, start + buffer) for lo, hi in ranges.values())


@pytest.mark.parametrize("mode,expected", [("race", 8), ("multistream", 21), ("joint_distance", 6)])
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
    assert Human.get_keymap("arc_tokenizer_cartesian")["left.action_ee_pose"]["horizon"] == 600
    assert Yam.get_keymap("arc_tokenizer_cartesian")["left.cmd_ee_pose"]["horizon"]["type"] == "arc_distance"


def test_saved_checkpoint_config_retains_representation_selection(monkeypatch):
    from egomimic.trainHydra import _build_model_config_tree

    monkeypatch.setenv("EGOVERSE_ABC_DATASET_DIR", "/tmp/unused-arc-config-data")
    recipe = "abc_arc/robot_bc/abc_multitask4_hpt300_hybrid_visual_openloop"
    cfg = _compose(recipe, "race")
    # Training/checkpoint serialization must keep the representation with the
    # source config; dimensions alone cannot identify an ARC representation.
    saved = OmegaConf.create(OmegaConf.to_yaml(cfg, resolve=False))
    assert saved.abc.arc_chunking_mode == "race"
    assert saved.data.train_datasets.yam_bimanual.resolver.key_map.arc_chunking_mode == "race"
    assert saved.data.train_datasets.yam_bimanual.resolver.transform_list.arc_chunking_mode == "race"
    assert saved.evaluator.arc_chunking_mode == "race"
    checkpoint_config = _build_model_config_tree(cfg)
    contract = checkpoint_config.run_provenance.action_contract
    assert contract.representation == "hybrid_arc_tokenizer_cartesian"
    assert contract.arc_chunking_mode == "race"
    assert contract.translation_distance_m == 0.81
    assert contract.rotation_distance_radians == pytest.approx(np.deg2rad(24))
    assert contract.velocity_mode == "per_waypoint"


def _norm_contract(mode, distance=0.4):
    return dict(
        representation="hybrid_arc_tokenizer_cartesian", arc_chunking_mode=mode,
        translation_distance_m=distance, rotation_distance_radians=0.42,
        waypoints=100, velocity_mode="per_waypoint", control_dt=1 / 30,
    )


def _normalizer():
    return MultiDataset(state={
        "norm_mode": "quantile", "embodiments": [0],
        "key_types": {0: {"actions_cartesian": "action_keys"}},
        "zarr_keys": {0: {"actions_cartesian": "actions_cartesian"}},
    })


def _write_norm_cache(directory, mode, source_sampling=None):
    normalizer = _normalizer()
    contract = None if mode is None else _norm_contract(mode)
    if contract is not None and source_sampling is not None:
        contract["source_sampling"] = source_sampling
    normalizer.infer_norm_from_dataset(
        [{"actions_cartesian": np.array([[1.0, 2.0], [3.0, 4.0]])}],
        dataset_name=0, num_workers=0, sample_frac=1.0,
        action_contract=contract,
    )
    normalizer.cache_stats(str(directory))
    return normalizer, str(directory / "norm_stats/norm_stats.json")


@pytest.mark.parametrize("cached_mode", [None, *MODES])
@pytest.mark.parametrize("requested_mode", MODES)
def test_norm_cache_requires_matching_mode_except_legacy_joint(tmp_path, cached_mode, requested_mode):
    _, path = _write_norm_cache(tmp_path, cached_mode)
    consumer = _normalizer()
    accepted = cached_mode == requested_mode or (cached_mode is None and requested_mode == "joint_distance")
    if accepted:
        consumer.infer_norm_from_dataset(
            [], dataset_name=0, precomputed_norm_path=path,
            action_contract=_norm_contract(requested_mode),
        )
        np.testing.assert_array_equal(
            consumer.norm_stats[0]["actions_cartesian"]["mean"], [[1.0, 2.0], [3.0, 4.0]]
        )
    else:
        with pytest.raises(ValueError, match="action contract|arc_chunking_mode"):
            consumer.infer_norm_from_dataset(
                [], dataset_name=0, precomputed_norm_path=path,
                action_contract=_norm_contract(requested_mode),
            )
        assert not consumer.norm_stats[0]


def test_norm_cache_rejects_same_mode_different_cap(tmp_path):
    _, path = _write_norm_cache(tmp_path, "joint_distance")
    with pytest.raises(ValueError, match="action contract"):
        _normalizer().infer_norm_from_dataset(
            [], dataset_name=0, precomputed_norm_path=path,
            action_contract=_norm_contract("joint_distance", distance=0.81),
        )


@pytest.mark.parametrize("tagged", [False, True])
def test_changed_source_sampling_requires_tagged_cache_even_for_joint(tmp_path, tagged):
    _, path = _write_norm_cache(
        tmp_path, "joint_distance" if tagged else None,
        source_sampling="native_30hz_v1" if tagged else None,
    )
    contract = _norm_contract("joint_distance")
    contract["source_sampling"] = "native_30hz_v1"
    consumer = _normalizer()
    if tagged:
        consumer.infer_norm_from_dataset(
            [], dataset_name=0, precomputed_norm_path=path, action_contract=contract,
        )
        assert consumer.norm_stats[0]["actions_cartesian"]
    else:
        with pytest.raises(ValueError, match="action contract"):
            consumer.infer_norm_from_dataset(
                [], dataset_name=0, precomputed_norm_path=path, action_contract=contract,
            )


def test_human_arc_config_records_changed_native_source_sampling():
    from egomimic.trainHydra import _arc_normalization_contract

    cfg = _compose("abc_arc/human_bc/mecka_fold_clothes_40h_human_visual_hybrid_openloop")
    assert _arc_normalization_contract(cfg)["source_sampling"] == "native_30hz_v1"


def test_norm_contract_survives_state_roundtrip_and_recache(tmp_path):
    original, _ = _write_norm_cache(tmp_path / "original", "race")
    restored = MultiDataset.from_state(original.to_state())
    restored.cache_stats(str(tmp_path / "restored"))
    path = str(tmp_path / "restored/norm_stats/norm_stats.json")
    with pytest.raises(ValueError, match="action contract"):
        _normalizer().infer_norm_from_dataset(
            [], dataset_name=0, precomputed_norm_path=path,
            action_contract=_norm_contract("joint_distance"),
        )


@pytest.mark.parametrize("recipe", [name for name in RECIPES if "hybrid" in name])
def test_train_normalization_uses_resolved_arc_contract(recipe):
    from egomimic.trainHydra import _arc_normalization_contract

    contract = _arc_normalization_contract(_compose(recipe, "multistream"))
    assert contract["arc_chunking_mode"] == "multistream"
    assert contract["rotation_distance_radians"] > 0
    assert contract["velocity_mode"] == "per_waypoint"


def test_train_normalization_preserves_baseline_and_legacy_configs():
    from egomimic.trainHydra import _arc_normalization_contract

    assert _arc_normalization_contract(OmegaConf.create({})) is None
    cfg = _compose("abc_arc/robot_bc/abc_multitask4_hpt300_baseline_visual_openloop", "race")
    assert _arc_normalization_contract(cfg) is None
