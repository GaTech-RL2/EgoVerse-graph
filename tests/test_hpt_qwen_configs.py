"""Compose the language HPT experiments without downloading Qwen.

``tools/config_graph.py`` instantiates every stage, which would pull
Qwen/Qwen3-Embedding-0.6B. These tests only compose and inspect the yaml.
"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

_REPO = Path(__file__).resolve().parents[1]
_CONFIGS = _REPO / "egomimic/hydra_configs"

_LANG_EXPERIMENTS = (
    "abc_arc/abc_lang_fstshirt_mecka_freefold_cotrain_baseline",
    "abc_arc/abc_lang_fstshirt_mecka_freefold_cotrain_arcD40M100",
    "abc_arc/abc_lang_mecka_fold_multitask_cotrain_baseline",
    "abc_arc/abc_lang_mecka_fold_multitask_cotrain_arcD40M100",
)
_MIXTURE_EXPERIMENT = (
    "abc_arc/mixture_hpt_abc_fstshirt_mecka_freefold_cotrain_arcD40M100"
)
_CLOCK_EXPERIMENT = "abc_arc/clock_head_abc_fstshirt_mecka_freefold_cotrain_arcD40M100"

_QWEN_STEM = "egomimic.models.stems.text_encoders.QwenPooledEncoder"
_PROMPT_STAGE = "egomimic.pipeline.stages_hpt.AnnotationPromptStage"


def _compose(experiment: str):
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIGS)):
        return compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment={experiment}", "++paths.root_dir=."],
        )


@pytest.mark.parametrize("experiment", _LANG_EXPERIMENTS)
def test_lang_experiments_compose(experiment):
    cfg = _compose(experiment)
    stages = cfg.model.pipeline.stages
    assert stages[0]._target_ == _PROMPT_STAGE
    stems = stages[1].stems
    assert stems["observations.annotation"]._target_ == _QWEN_STEM
    assert cfg.hpt.embed_dim == 840
    assert cfg.hpt.qwen.model_name == "Qwen/Qwen3-Embedding-0.6B"
    assert (
        cfg.data.train_datasets.yam_bimanual.resolver.key_map.annotation_key
        == "annotations"
    )
    assert "annotations" in cfg.data.train_datasets.yam_bimanual.batch_keys


def test_mixture_experiment_seeds_annotation_prompt_stage():
    cfg = _compose(_MIXTURE_EXPERIMENT)
    assert cfg.model.pipeline.stages[0]._target_ == (
        "egomimic.pipeline.stages_hpt.AnnotationPromptStage"
    )
    for split in ("train_datasets", "valid_datasets"):
        for embodiment in ("human_bimanual", "yam_bimanual"):
            dataset = cfg.data[split][embodiment]
            assert "annotations" in dataset.batch_keys
            assert dataset.resolver.key_map.annotation_key == "annotations"


@pytest.mark.parametrize("experiment", (_CLOCK_EXPERIMENT, _MIXTURE_EXPERIMENT))
def test_dual_head_experiments_compose_with_split_flow(experiment):
    cfg = _compose(experiment)
    targets = [stage._target_ for stage in cfg.model.pipeline.stages]
    assert "egomimic.pipeline.stages_dual_flow.DualFlowNoisingStage" in targets
    assert "egomimic.pipeline.stages_dual_flow.DualFlowDenoiserStage" in targets
    assert "egomimic.pipeline.stages_dual_flow.DualFlowVelocityLossStage" in targets
    assert cfg.abc.arc_token_rows == 2 * cfg.abc.arc_waypoints


def test_baseline_horizon_is_time_indexed():
    cfg = _compose("abc_arc/abc_lang_fstshirt_mecka_freefold_cotrain_baseline")
    assert cfg.hpt.action_horizon == 100
    assert cfg.model.pipeline.stages[4].action_horizon == 100
    assert cfg.model.pipeline.stages[5].model.act_seq == 100


def test_arc_horizon_follows_per_waypoint_token_rows():
    cfg = _compose("abc_arc/abc_lang_fstshirt_mecka_freefold_cotrain_arcD40M100")
    assert cfg.abc.arc_token_rows == 200
    assert cfg.hpt.action_horizon == 200
    assert cfg.model.pipeline.stages[4].action_horizon == 200
    assert cfg.model.pipeline.stages[5].model.act_seq == 200


def test_baseline_and_arc_share_the_qwen_architecture():
    base = _compose("abc_arc/abc_lang_mecka_fold_multitask_cotrain_baseline")
    arc = _compose("abc_arc/abc_lang_mecka_fold_multitask_cotrain_arcD40M100")
    assert (
        base.model.pipeline.stages[1].stems["observations.annotation"]._target_
        == arc.model.pipeline.stages[1].stems["observations.annotation"]._target_
    )
    assert base.hpt.embed_dim == arc.hpt.embed_dim == 840
    assert base.hpt.num_blocks == arc.hpt.num_blocks == 19
