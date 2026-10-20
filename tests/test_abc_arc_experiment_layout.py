"""Task-first recipes share one clock contract without changing frozen pools."""

import json
from pathlib import Path

import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from egomimic.pipeline.inference_config import build_inference_config
from egomimic.rldb.zarr.arc_length_tokenizer import TokenizeBimanualArcLengthCartesian
from scripts.audit_hydra_configs import CONFIGS, compose_for_audit

ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    "real/organize_stationary",
    "real/fold_and_stack_towels",
    "real/wireless_earbuds",
    "real/multitask",
)


@pytest.mark.parametrize("task", TASKS)
def test_three_public_entries_per_task(task):
    assert {p.name for p in (CONFIGS / "experiment" / task).glob("*.yaml")} == {
        "arc_vel.yaml",
        "arc_dur.yaml",
        "baseline.yaml",
    }


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize(
    "clock,leaf", [("velocity", "arc_vel"), ("duration", "arc_dur")]
)
@pytest.mark.parametrize(
    "streams,width", [("full_separation", 20), ("translation_rotation", 18)]
)
def test_task_clocks_match_data_eval_and_rollout(task, clock, leaf, streams, width):
    with compose_for_audit(CONFIGS / f"experiment/{task}/{leaf}.yaml") as cfg:
        cfg.abc.arc_streams = streams
        assert cfg.abc.arc_token_dim == width
        evaluator = instantiate(cfg.evaluator)
        assert evaluator.clock_mode == clock
        assert evaluator.streams == streams
        flow_stages = [
            stage
            for stage in cfg.model.pipeline.stages
            if stage._target_.rsplit(".", 1)[-1]
            in ("FlowNoisingStage", "FlowDenoiserStage")
        ]
        assert len(flow_stages) == 2
        for stage in flow_stages:
            assert stage.action_dim == width
            assert stage.action_horizon == cfg.abc.arc_waypoints
        for split in ("train_datasets", "valid_datasets"):
            for dataset in cfg.data[split].values():
                transforms = instantiate(dataset.resolver.transform_list)
                codecs = [
                    t
                    for t in transforms
                    if isinstance(t, TokenizeBimanualArcLengthCartesian)
                ]
                assert len(codecs) == 1
                codec = codecs[0]
                assert codec.clock_mode == clock
                assert codec.streams == streams
                assert codec.shape == (cfg.abc.arc_waypoints, width)
                assert evaluator._codec().shape == codec.shape
                instantiate(dataset.resolver.key_map)
        contract = build_inference_config(cfg)
        assert contract["status"] == "ready"
        profile = contract["inference_graph"]["profiles"]["default"]
        assert profile["native_shape"] == [cfg.abc.arc_waypoints, width]
        decoder = instantiate(profile["adapter"]["decoder"])
        assert decoder.codec.clock_mode == clock
        assert decoder.codec.streams == streams
        assert decoder.codec.shape == evaluator._codec().shape
        assert cfg.run_provenance.action_contract.clock_mode == clock


@pytest.mark.parametrize("task", TASKS)
def test_task_variants_preserve_identical_episode_selection(task):
    selections = []
    for leaf in ("baseline", "arc_vel", "arc_dur"):
        with compose_for_audit(CONFIGS / f"experiment/{task}/{leaf}.yaml") as cfg:
            selections.append(
                {
                    (split, name): {
                        key: OmegaConf.to_container(value, resolve=True)
                        if OmegaConf.is_config(value)
                        else value
                        for key, value in dataset.items()
                        if key in ("filters", "mode", "valid_ratio", "split_seed")
                        or key.startswith("expected_")
                    }
                    for split in ("train_datasets", "valid_datasets")
                    for name, dataset in cfg.data[split].items()
                }
            )
    assert selections[0] == selections[1] == selections[2]


@pytest.mark.parametrize("task", TASKS)
def test_baseline_remains_time_indexed_and_rollout_ready(task):
    with compose_for_audit(CONFIGS / f"experiment/{task}/baseline.yaml") as cfg:
        for split in ("train_datasets", "valid_datasets"):
            for dataset in cfg.data[split].values():
                assert not any(
                    isinstance(t, TokenizeBimanualArcLengthCartesian)
                    for t in instantiate(dataset.resolver.transform_list)
                )
        artifact = build_inference_config(cfg)
        assert artifact["status"] == "ready"
        assert artifact["inference_graph"]["profiles"]["default"]["native_shape"] == [
            100,
            14,
        ]


def test_migration_targets_exist_and_retired_folders_are_empty():
    mapping = json.loads((ROOT / "docs/experiment_config_migration.json").read_text())
    assert all(
        any(
            (CONFIGS / f"experiment/{target}{suffix}").is_file()
            for suffix in (".yaml", ".json")
        )
        for target in mapping.values()
    )
    for folder in ("abc_arc/cotrain", "abc_arc/robot_bc", "yam_arc_grid"):
        assert not list((CONFIGS / "experiment" / folder).rglob("*.yaml"))


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("leaf", ("baseline", "arc_vel", "arc_dur"))
def test_public_recipe_is_a_complete_run_entrypoint(task, leaf):
    with compose_for_audit(CONFIGS / f"experiment/{task}/{leaf}.yaml") as cfg:
        for section in ("model", "data", "evaluator", "trainer", "callbacks", "logger"):
            assert cfg[section] is not None
            OmegaConf.to_container(cfg[section], resolve=True, throw_on_missing=True)
        assert cfg.model.pipeline.stages
        assert cfg.model.optimizer._target_
        assert cfg.data.train_datasets
        assert cfg.data.valid_datasets
        assert cfg.evaluator._target_
        assert cfg.callbacks.model_checkpoint._target_
        assert cfg.logger.wandb._target_
