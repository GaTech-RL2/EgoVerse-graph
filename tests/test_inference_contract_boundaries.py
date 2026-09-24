"""Deployment declarations preserve duration, source width and typed settings."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from egomimic.pipeline.inference_config import (
    build_inference_config,
    validate_input_constants,
)
from egomimic.robot.graph_policy import configure_profile_controls

CONFIGS = Path(__file__).parents[1] / "egomimic/hydra_configs"


def compose_recipe(**kwargs):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        return compose(**kwargs)


def test_resampled_yam_contract_restores_native_duration():
    cfg = compose_recipe(
        config_name="train_zarr_cartesian",
        overrides=["+experiment=abc/yam_fstshirt_dp"],
    )
    artifact = build_inference_config(cfg)
    assert artifact["status"] == "ready"
    graph = artifact["inference_graph"]
    assert graph["native_output"]["shape"] == [100, 14]
    assert graph["output"]["shape"] == [45, 14]
    decoder = instantiate(graph["profiles"]["default"]["adapter"]["decoder"])
    actions = np.zeros((1, 100, 14))
    actions[0, :, 0] = np.linspace(0, 44, 100)
    np.testing.assert_allclose(decoder(actions)[0, :, 0], np.arange(45), atol=1e-10)
    with pytest.raises(ValueError, match="input profile"):
        validate_input_constants(graph, {"embodiment": 6})


def test_chunk_mean_timing_is_non_deployable_even_with_same_model_shape():
    cfg = compose_recipe(
        config_name="train_zarr_cartesian",
        overrides=[
            "+experiment=abc_arc/abc_fstshirt_arc_bc",
            "abc.arc_velocity_mode=mean",
        ],
    )
    artifact = build_inference_config(cfg)
    assert artifact["status"] == "unsupported"
    assert "timing cannot reconstruct intervals" in artifact["reason"]


@pytest.mark.parametrize(
    "domain, native_width, output_shape",
    [
        ("human_bimanual", 18, [30, 12]),
        ("eva_bimanual", 20, [45, 14]),
    ],
)
def test_pi_mixed_source_contract_selects_exact_width_and_time(
    domain, native_width, output_shape
):
    cfg = compose_recipe(
        config_name="train_zarr_cartesian_pi",
        overrides=[
            "model=pi05/pi0.5_bc_mecka_abc_6d",
            "pi05.pretrained_weights=/unused/weights",
            f"model.deployment_domain={domain}",
        ],
    )
    artifact = build_inference_config(cfg)
    assert artifact["status"] == "ready"
    graph = artifact["inference_graph"]
    assert graph["native_output"]["shape"] == [100, native_width]
    assert graph["output"]["shape"] == output_shape
    decoder = instantiate(graph["profiles"]["default"]["adapter"]["decoder"])
    native = np.zeros((1, 100, native_width))
    for offset in (0, native_width // 2):
        native[..., offset + 3] = 1
        native[..., offset + 7] = 1
    native[0, :, 0] = np.linspace(0, output_shape[0] - 1, 100)
    decoded = decoder(native)
    assert list(decoded.shape[1:]) == output_shape
    np.testing.assert_allclose(decoded[0, :, 0], np.arange(output_shape[0]), atol=1e-10)


def test_boolean_and_enum_settings_bind_atomically():
    owner = SimpleNamespace(enabled=False, method="first")
    graph = SimpleNamespace(pipeline=SimpleNamespace(stage_by_id=lambda _: owner))
    training = OmegaConf.create(
        {"model": {"pipeline": {"stages": [{}], "stage_ids": {"backend": 0}}}}
    )
    controls = {
        "enabled": {
            "label": "Enabled",
            "type": "boolean",
            "default": True,
            "target": {
                "kind": "stage_attribute",
                "stage_id": "backend",
                "attribute_path": "enabled",
            },
        },
        "method": {
            "label": "Method",
            "type": "enum",
            "choices": ["first", "second"],
            "default": "second",
            "target": {
                "kind": "stage_attribute",
                "stage_id": "backend",
                "attribute_path": "method",
            },
        },
    }
    profile = {"default": {"stage_id": "backend", "overrides": controls}}
    bad = deepcopy(profile)
    bad["default"]["overrides"]["method"]["target"]["attribute_path"] = "absent"
    with pytest.raises(ValueError, match="does not exist"):
        configure_profile_controls(graph, training, bad)
    assert owner.enabled is False and owner.method == "first"
    bindings = configure_profile_controls(graph, training, profile)
    assert owner.enabled is True and owner.method == "second"
    with pytest.raises(ValueError, match="boolean"):
        bindings[0].validate(1)
    with pytest.raises(ValueError, match="one of"):
        bindings[1].validate("third")
