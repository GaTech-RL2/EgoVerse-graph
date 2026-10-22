"""The nine real task leaves: Aidan's frozen pools, HPT-300M graph and single-clock ARC protocol."""

import re

import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from egomimic.pipeline.inference_config import build_inference_config
from egomimic.rldb.zarr.arc_length_tokenizer import TokenizeBimanualArcLengthCartesian
from scripts.audit_hydra_configs import CONFIGS, compose_for_audit

REAL = CONFIGS / "experiment/real"
TASKS = {"stationary": 0.22, "earbuds": 0.22, "towels": 0.33}
MODES = ("time", "duration", "velocity")
POOLS = {"stationary": (417, 24), "earbuds": (339, 15), "towels": (462, 17)}
ARC = [(f"{task}_{mode}", task, mode) for task in TASKS for mode in MODES[1:]]


def _hashes(lam):
    return re.findall(r"'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{6})'", lam)


def test_real_holds_exactly_the_nine_leaves():
    assert {p.name for p in REAL.glob("*.yaml")} == {
        f"{task}_{mode}.yaml" for task in TASKS for mode in MODES
    }


@pytest.mark.parametrize("task", TASKS)
def test_the_three_leaves_of_a_task_share_graph_widths_pool_and_schedule(task):
    seen = []
    for mode in MODES:
        with compose_for_audit(REAL / f"{task}_{mode}.yaml") as cfg:
            assert cfg.name == f"{task}_{mode}"
            assert cfg.logger.wandb.id == f"{cfg.name}_s{cfg.seed}"
            assert (cfg.hpt.embed_dim, cfg.hpt.num_blocks, cfg.hpt.num_heads) == (
                840,
                19,
                10,
            )
            assert cfg.trainer.max_steps == 240000
            assert cfg.trainer.limit_val_batches == 1.0
            assert type(instantiate(cfg.evaluator)).__name__ == "OpenLoopSimEval"
            assert cfg.evaluator.execute_fraction == 0.25
            train = cfg.data.train_datasets.yam_bimanual
            valid = cfg.data.valid_datasets.yam_bimanual
            assert (
                train.expected_train_episode_count,
                valid.expected_train_episode_count,
            ) == POOLS[task]
            seen.append(
                OmegaConf.to_container(
                    OmegaConf.create({"train": train.filters, "valid": valid.filters}),
                    resolve=True,
                )
            )
    assert seen[0] == seen[1] == seen[2]


@pytest.mark.parametrize("task", TASKS)
def test_time_leaf_is_the_frozen_yam_time_baseline(task):
    with compose_for_audit(REAL / f"{task}_time.yaml") as cfg:
        assert cfg.frozen_yam.variant == "time"
        assert "abc" not in cfg
        assert cfg.evaluator.action_mode == "baseline"
        for split in ("train_datasets", "valid_datasets"):
            transforms = instantiate(
                cfg.data[split].yam_bimanual.resolver.transform_list
            )
            assert not any(
                isinstance(t, TokenizeBimanualArcLengthCartesian) for t in transforms
            )
        artifact = build_inference_config(cfg)
        assert artifact["status"] == "ready"
        profile = artifact["inference_graph"]["profiles"]["default"]
        assert profile["native_shape"] == [100, 14]
        assert type(instantiate(profile["adapter"]["decoder"])).__name__ == (
            "TimeChunkRetimer"
        )


@pytest.mark.parametrize("leaf,task,mode", ARC)
def test_arc_protocol_reaches_data_evaluator_and_rollout(leaf, task, mode):
    distance = TASKS[task]
    with compose_for_audit(REAL / f"{leaf}.yaml") as cfg:
        abc = cfg.abc
        assert (abc.arc_clock_mode, abc.arc_streams) == (mode, "translation")
        assert (abc.arc_token_dim, abc.arc_waypoints) == (16, 200)
        assert abc.arc_distance == distance
        assert abc.arc_source_buffer_frames == 100
        assert cfg.run_provenance.action_contract.timing_layout == (
            "first_delay_absolute_v1"
            if mode == "duration"
            else "first_delay_incoming_v1"
        )
        evaluator = instantiate(cfg.evaluator)
        assert (evaluator.clock_mode, evaluator.streams) == (mode, "translation")
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
            assert (codec.clock_mode, codec.streams) == (mode, "translation")
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
            mode,
            "translation",
        )
        assert (decoder.execute_percent, decoder.first_stream) == (50, 1)
        assert decoder.codec.shape == codec.shape


def test_stream_override_widens_the_token_everywhere():
    with compose_for_audit(REAL / "stationary_duration.yaml") as cfg:
        cfg.abc.arc_streams = "full_separation"
        assert cfg.abc.arc_token_dim == 20
        codec = next(
            t
            for t in instantiate(
                cfg.data.train_datasets.yam_bimanual.resolver.transform_list
            )
            if isinstance(t, TokenizeBimanualArcLengthCartesian)
        )
        assert codec.shape == (200, 20)
        assert instantiate(cfg.evaluator)._codec().shape == (200, 20)
        profile = build_inference_config(cfg)["inference_graph"]["profiles"]["default"]
        assert profile["native_shape"] == [200, 20]


@pytest.mark.parametrize(
    "pool,task,counts",
    [
        ("stationary", "organize_stationary", (417, 24)),
        ("earbuds", "wireless_earbuds", (339, 15)),
        ("towels", "fold_towels", (462, 17)),
    ],
)
def test_pools_point_at_their_sql_task_with_a_frozen_validation_list(
    pool, task, counts
):
    cfg = OmegaConf.load(CONFIGS / f"data/abc_arc/{pool}.yaml")
    train = cfg.train_datasets.yam_bimanual
    valid = cfg.valid_datasets.yam_bimanual
    (train_rule,) = train.filters.filter_lambdas
    (valid_rule,) = valid.filters.filter_lambdas
    for rule in (train_rule, valid_rule):
        assert f"row['task'] == '{task}'" in rule
        assert "lambda row, V=frozenset(" in rule
    assert "not in V" in train_rule and " in V" in valid_rule.replace("not in V", "")
    assert (
        "episode_hashes" not in train.filters and "episode_hashes" not in valid.filters
    )
    val = _hashes(valid_rule)
    assert len(val) == len(set(val)) == counts[1]
    assert (
        _hashes(train_rule) == val
    )  # the train rule lists nothing but the validation set
    assert (
        train.expected_train_episode_count,
        valid.expected_train_episode_count,
    ) == counts
    keep = eval(train_rule)
    row = dict(
        embodiment="yam_bimanual",
        lab="rl2",
        task=task,
        rig_name="rl2_abc",
        zarr_processed_path="s3://x",
        zarr_processing_error="",
        is_deleted=False,
        is_eval=False,
        num_frames=100,
    )
    assert keep({**row, "episode_hash": "2099-01-01-00-00-00-000000"})
    assert not keep({**row, "episode_hash": val[0]})
    assert eval(valid_rule)({**row, "episode_hash": val[0]})
    assert not keep(
        {**row, "task": f"{task}_retired", "episode_hash": "2099-01-01-00-00-00-000000"}
    )
