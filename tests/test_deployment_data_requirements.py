"""Selected data must agree with model-owned frame, time and tokenizer semantics."""

from copy import deepcopy

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from egomimic.pipeline.inference_config import (
    build_inference_config,
    validate_model_data_context,
)
from egomimic.pl_utils.data_context import DataContext
from egomimic.rldb.zarr.data_module import _preprocessing_contract
from scripts.audit_hydra_configs import CONFIGS, compose_for_audit

PI_CASES = [
    ("pi0.5_bc_eva", "eva_pi_lang"),
    ("pi0.5_bc_abc_eva_6d", "pi05/abc_eva_pi_6d"),
    ("pi0.5_ft_abc_eva_6d", "pi05/abc_eva_pi_6d"),
    ("pi0.5_bc_mecka_6d", "pi05/mecka_all_pi_6d"),
    ("pi0.5_bc_mecka_abc_6d", "pi05/mecka_abc_pi_6d"),
    ("pi0.5_cotrain_eva_aria_6d", "pi05/cotrain_pi_lang_6d"),
    ("pi0.5_ft_aria_fold_grip", "pi05/aria_fold_pi_6d_grip"),
]
EXPERIMENTS = [
    "abc/yam_fstshirt_dp",
    "abc/yam_fstshirt_hpt",
    "abc_arc/abc_fstshirt_arc_bc",
    "abc_arc/abc_lang_fstshirt_mecka_freefold_cotrain_baseline",
    "abc_arc/abc_lang_fstshirt_mecka_freefold_cotrain_arcD40M100",
    "e1/fold_time",
    "e1/fold_arcvel",
    "e1/fold_arcdur",
    "e1/fold_arclogdur",
    "e1/abc_time",
    "e1/abc_arcdur",
    "e1/abcs_time",
    "e1/abcs_arclogdur",
]
SOURCE_IDENTITIES = {"human_bimanual": "3", "eva_bimanual": "6", "yam_bimanual": "7"}


def declared_data_context(cfg):
    """Read data-side configuration, independently of the model requirements.

    No normalization or training claim: actual transforms are covered by the
    source fixtures and real dataset/optimizer tests. This gate catches YAML
    composition mismatches before requiring any cloud episode.
    """
    state = {
        "preprocessing": {
            SOURCE_IDENTITIES[name]: _preprocessing_contract(
                OmegaConf.to_container(ds, resolve=True), cfg.data.source_fps
            )
            for name, ds in cfg.data.train_datasets.items()
            if ds is not None
        }
    }
    return DataContext(None, {}, (), state)


def check_selected_contract(cfg):
    context = declared_data_context(cfg)
    validate_model_data_context(cfg, context)
    artifact = build_inference_config(cfg)
    assert artifact["status"] == "ready"
    graph = artifact["inference_graph"]
    assert graph["native_output"]["frame"] == graph["output"]["frame"]
    for source in context.state["preprocessing"]:
        wrong = deepcopy(context.state)
        wrong["preprocessing"][source]["source_fps"] = 15
        with pytest.raises(ValueError, match="source_fps"):
            validate_model_data_context(cfg, DataContext(None, {}, (), wrong))
        wrong = deepcopy(context.state)
        transform = wrong["preprocessing"][source]["transforms"]
        # Timing variants may have equal tensor dimensions and still differ.
        key = "variant" if "variant" in transform else "coord_frame"
        transform[key] = "incompatible"
        with pytest.raises(ValueError, match=key):
            validate_model_data_context(cfg, DataContext(None, {}, (), wrong))


@pytest.mark.parametrize("model,data", PI_CASES)
def test_pi_selected_frame_rotation_and_stride(model, data):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        cfg = compose(
            config_name="train_zarr_cartesian_pi",
            overrides=[
                f"model=pi05/{model}",
                f"data={data}",
                "pi05.pretrained_weights=/unused/no-weight-download",
            ],
        )
    check_selected_contract(cfg)


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_yam_arc_and_tempo_selected_contracts(experiment):
    with compose_for_audit(CONFIGS / f"experiment/{experiment}.yaml") as cfg:
        check_selected_contract(cfg)


def test_every_shipped_ready_model_declares_data_semantics():
    for path in sorted((CONFIGS / "model").rglob("*.yaml")):
        with compose_for_audit(path) as cfg:
            if build_inference_config(cfg)["status"] == "ready":
                assert OmegaConf.select(cfg, "model.data_requirements"), path
                graph = build_inference_config(cfg)["inference_graph"]
                assert graph["native_output"]["frame"] in ("camframe", "eef_frame")
                assert graph["output"]["frame"] == graph["native_output"]["frame"]


def test_changing_data_alone_cannot_rewrite_pi_model_frame():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        cfg = compose(
            config_name="train_zarr_cartesian_pi",
            overrides=[
                "pi05.pretrained_weights=/unused/no-weight-download",
                "data.train_datasets.eva_bimanual.resolver.transform_list.coord_frame=camframe",
            ],
        )
    assert cfg.model.coordinate_frame == "eef_frame"
    with pytest.raises(ValueError, match="coord_frame"):
        validate_model_data_context(cfg, declared_data_context(cfg))
    # An intentional new protocol can choose the matching model-owned frame.
    cfg.model.coordinate_frame = "camframe"
    validate_model_data_context(cfg, declared_data_context(cfg))
    graph = build_inference_config(cfg)["inference_graph"]
    assert graph["profiles"]["default"]["adapter"]["action_frame"] == "model_frame"
