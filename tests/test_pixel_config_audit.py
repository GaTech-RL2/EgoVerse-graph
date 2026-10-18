"""Offline composition supplies inputs without relaxing production contracts."""

import socket

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import MissingMandatoryValue, OmegaConf

from scripts.audit_hydra_configs import CONFIGS, compose_for_audit

PREPARED_CONTEXTS = [
    "data/world_model/prepared.yaml",
    "experiment/world_model/continuous.yaml",
    "experiment/world_model/discrete.yaml",
    "model/world_model/base.yaml",
    "data/dino_proprio/prepared.yaml",
    "data/pixel_proprio/prepared.yaml",
    "experiment/dino_proprio/direct_action.yaml",
    "experiment/dino_proprio/visual_state.yaml",
    "experiment/pixel_proprio/video_continuous.yaml",
    "experiment/pixel_proprio/video_discrete.yaml",
    "experiment/pixel_proprio/video_only.yaml",
    "experiment/pixel_proprio/readout_continuous.yaml",
    "experiment/pixel_proprio/readout_discrete.yaml",
    "model/dino_proprio/base.yaml",
    "model/pixel_proprio/base.yaml",
    "model/pixel_proprio/readout.yaml",
    "model/pixel_proprio/video.yaml",
]


@pytest.mark.parametrize("name", PREPARED_CONTEXTS)
def test_prepared_recipes_and_fragments_resolve_without_external_artifacts(
    name, monkeypatch
):
    def no_network(*args, **kwargs):
        pytest.fail("Offline configuration audit attempted network access")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    previous_hydra = HydraConfig.instance().cfg
    with compose_for_audit(CONFIGS / name) as cfg:
        # Exercise the same strict recursive resolution as the shipped auditor.
        resolved = {
            key: OmegaConf.to_container(cfg[key], resolve=True, throw_on_missing=True)
            if OmegaConf.is_config(cfg[key])
            else cfg[key]
            for key in cfg
            if key != "hydra"
        }
        OmegaConf.to_container(cfg.hydra.launcher, resolve=True, throw_on_missing=True)
        assert resolved["data"]["manifest_path"].startswith(
            "/tmp/egoverse-config-audit/"
        )
        assert resolved["data"]["mode"] in {"continuous", "discrete"}
        if "pixel_readout" in resolved:
            binding = resolved["model"]["pipeline"]["initialization"][0]
            assert binding["source"] == resolved["pixel_readout"]["initial_weights"]
            assert binding["sha256"] == "0" * 64
            assert resolved["trainer"]["max_steps"] == 1000
        if resolved["data"].get("task") == "direct_action":
            assert resolved["data"]["action_manifest_path"].endswith(
                "/dino-actions.json"
            )
    assert HydraConfig.instance().cfg is previous_hydra


@pytest.mark.parametrize(
    "experiment",
    [
        "world_model/continuous",
        "world_model/discrete",
        "dino_proprio/direct_action",
        "dino_proprio/visual_state",
        "pixel_proprio/video_continuous",
        "pixel_proprio/readout_continuous",
        "pixel_proprio/readout_discrete",
    ],
)
def test_audit_bindings_do_not_escape_into_fresh_production_composition(experiment):
    with compose_for_audit(CONFIGS / f"experiment/{experiment}.yaml"):
        pass
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment={experiment}", "hydra/launcher=basic"],
        )
    assert OmegaConf.is_missing(cfg.data, "manifest_path")
    with pytest.raises(MissingMandatoryValue):
        OmegaConf.to_container(cfg.data, resolve=True, throw_on_missing=True)
    if "pixel_readout" in cfg:
        assert OmegaConf.is_missing(cfg.pixel_readout, "initial_weights")
        assert OmegaConf.is_missing(cfg.pixel_readout, "initial_weights_sha256")
    if experiment == "dino_proprio/direct_action":
        assert cfg.data.action_manifest_path is None


def test_world_model_pose_schema_remains_required_outside_audit():
    with compose_for_audit(CONFIGS / "model/world_model/base.yaml") as cfg:
        assert cfg.model.pipeline.stages[0].model.pose_dims == {
            "audit_measured_pose": 9
        }
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=["+experiment=world_model/continuous", "hydra/launcher=basic"],
        )
    assert OmegaConf.is_missing(cfg.model.pipeline.stages[0].model, "pose_dims")
