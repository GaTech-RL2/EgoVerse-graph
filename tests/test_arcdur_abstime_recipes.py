"""Aidan's absolute-duration single-clock ARC entry points keep the 2026-10-20 run protocol."""

import re

import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from egomimic.pipeline.inference_config import build_inference_config
from egomimic.rldb.zarr.arc_length_tokenizer import TokenizeBimanualArcLengthCartesian
from scripts.audit_hydra_configs import CONFIGS, compose_for_audit

FOLDER = "experiment/real/arcdur_abstime"
LEAVES = {
    "organize_stationary": 0.22,
    "wireless_earbuds": 0.22,
    "fold_and_stack_towels": 0.33,
}


def _hashes(lam):
    return re.findall(r"'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{6})'", lam)


def test_folder_holds_exactly_the_three_tasks():
    names = {p.name for p in (CONFIGS / FOLDER).glob("*.yaml")}
    assert names == {f"{task}.yaml" for task in LEAVES}


@pytest.mark.parametrize("task,distance", LEAVES.items())
def test_protocol_values_reach_data_evaluator_and_rollout(task, distance):
    with compose_for_audit(CONFIGS / f"{FOLDER}/{task}.yaml") as cfg:
        abc = cfg.abc
        assert (abc.arc_clock_mode, abc.arc_streams) == ("duration", "translation")
        assert (abc.arc_token_dim, abc.arc_waypoints) == (16, 200)
        assert abc.arc_distance == distance
        assert abc.arc_source_buffer_frames == 100
        assert cfg.logger.wandb.id == f"{cfg.name}_s{cfg.seed}"
        assert (
            cfg.run_provenance.action_contract.timing_layout
            == "first_delay_absolute_v1"
        )
        evaluator = instantiate(cfg.evaluator)
        assert type(evaluator).__name__ == "OpenLoopSimEval"
        assert (evaluator.clock_mode, evaluator.streams) == ("duration", "translation")
        assert evaluator.execute_fraction == 0.25
        assert evaluator.min_distance_unit == distance
        assert evaluator.resampled_vector_length == 200
        for split in ("train_datasets", "valid_datasets"):
            dataset = cfg.data[split].yam_bimanual
            assert dataset.resolver.transform_list.time_rows == 100
            assert dataset.resolver.key_map.yam_source_frames == 100
            codecs = [
                t
                for t in instantiate(dataset.resolver.transform_list)
                if isinstance(t, TokenizeBimanualArcLengthCartesian)
            ]
            assert len(codecs) == 1
            codec = codecs[0]
            assert (codec.clock_mode, codec.streams) == ("duration", "translation")
            assert codec.shape == (200, 16)
            assert codec.tokenizer.config.min_distance_unit == distance
            assert evaluator._codec().shape == codec.shape
            instantiate(dataset.resolver.key_map)
        flow = [
            stage
            for stage in cfg.model.pipeline.stages
            if stage._target_.rsplit(".", 1)[-1]
            in ("FlowNoisingStage", "FlowDenoiserStage")
        ]
        assert len(flow) == 2
        assert all(s.action_dim == 16 and s.action_horizon == 200 for s in flow)
        contract = build_inference_config(cfg)
        assert contract["status"] == "ready"
        profile = contract["inference_graph"]["profiles"]["default"]
        assert profile["native_shape"] == [200, 16]
        decoder = instantiate(profile["adapter"]["decoder"])
        assert (decoder.codec.clock_mode, decoder.codec.streams) == (
            "duration",
            "translation",
        )
        assert (decoder.execute_percent, decoder.first_stream) == (50, 1)
        assert decoder.codec.shape == codec.shape


def test_stationery_pool_is_slowpace366_plus_elmo51_with_frozen_validation():
    new = OmegaConf.load(CONFIGS / "data/abc_arc/stationery_sp366e51_bc_time.yaml")
    old = OmegaConf.load(CONFIGS / "data/abc_arc/stationery_slowpace366_bc_time.yaml")
    train = _hashes(new.train_datasets.yam_bimanual.filters.filter_lambdas[0])
    base = _hashes(old.train_datasets.yam_bimanual.filters.filter_lambdas[0])
    assert len(train) == len(set(train)) == 417 and len(base) == 366
    added = sorted(set(train) - set(base))
    assert len(added) == 51 and all(h.startswith("2026-10-20-") for h in added)
    assert added[0] == "2026-10-20-01-47-13-426000"
    assert added[-1] == "2026-10-20-03-12-10-879000"
    with compose_for_audit(CONFIGS / f"{FOLDER}/organize_stationary.yaml") as cfg:
        filters = cfg.data.train_datasets.yam_bimanual.filters
        assert len(_hashes(filters.filter_lambdas[0])) == 417
        assert cfg.data.train_datasets.yam_bimanual.expected_train_episode_count == 417
        assert cfg.data.valid_datasets.yam_bimanual.expected_train_episode_count == 24
        val = _hashes(cfg.data.valid_datasets.yam_bimanual.filters.filter_lambdas[0])
        assert len(val) == 24
        assert val == _hashes(old.valid_datasets.yam_bimanual.filters.filter_lambdas[0])


def test_earbuds_pool_is_updated_plus_49_takes_with_frozen_validation():
    old = OmegaConf.load(CONFIGS / "data/abc_arc/earbuds_updated.yaml")
    with compose_for_audit(CONFIGS / f"{FOLDER}/wireless_earbuds.yaml") as cfg:
        train = list(cfg.data.train_datasets.yam_bimanual.filters.episode_hashes)
        base = list(old.train_datasets.yam_bimanual.filters.episode_hashes)
        assert len(train) == len(set(train)) == 319 and len(base) == 270
        assert cfg.data.train_datasets.yam_bimanual.expected_train_episode_count == 319
        assert cfg.data.valid_datasets.yam_bimanual.expected_train_episode_count == 15
        added = sorted(set(train) - set(base))
        assert len(added) == 49 and all(h.startswith("2026-10-20-") for h in added)
        val = list(cfg.data.valid_datasets.yam_bimanual.filters.episode_hashes)
        assert len(val) == 15
        assert val == list(old.valid_datasets.yam_bimanual.filters.episode_hashes)


def test_towels_pool_is_the_frozen_since0924plus74_split():
    with compose_for_audit(CONFIGS / f"{FOLDER}/fold_and_stack_towels.yaml") as cfg:
        assert len(cfg.data.train_datasets.yam_bimanual.filters.episode_hashes) == 443
        assert len(cfg.data.valid_datasets.yam_bimanual.filters.episode_hashes) == 17
        assert cfg.towels_pool == "since0924plus74"
