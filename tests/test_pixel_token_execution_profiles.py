"""Packed execution is opt-in and remains bound across readout initialization."""

import copy
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts.world_model.initialize_pixel_readout import (
    resolve_model_config,
    validate_transition,
)

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "egomimic/hydra_configs"


def configuration(stage, arm, profile=None):
    overrides = [
        f"+experiment=pixel_proprio/{stage}_{arm}",
        "hydra/launcher=basic",
        "data.manifest_path=/fixture/manifest.json",
        "paths.output_dir=/fixture/output",
        "paths.work_dir=/fixture/source",
        # Compare whole configs without wall-clock run-name differences between
        # independent Hydra compositions. Keep all scientific fields asserted.
        "logger.wandb.id=fixture-run",
        "++logger.wandb.name=fixture-run",
        "+pixel_execution=ddp4_prefetch",
    ]
    if profile is not None:
        overrides.append(f"+world_model_execution={profile}")
    if stage == "readout":
        overrides.extend(
            [
                "pixel_readout.initial_weights=/fixture/expanded.pt",
                "pixel_readout.initial_weights_sha256=" + "a" * 64,
            ]
        )
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        config = compose(config_name="train_zarr_cartesian", overrides=overrides)
    return OmegaConf.to_container(config, resolve=True)


def denoiser(config):
    pipeline = config["model"]["pipeline"]
    return pipeline["stages"][pipeline["stage_ids"]["pixel_denoiser"]]["model"]


@pytest.mark.parametrize("stage", ["video", "readout"])
@pytest.mark.parametrize("arm", ["continuous", "discrete"])
@pytest.mark.parametrize("profile", [None, "batched_eager", "regional_compile"])
def test_existing_profiles_keep_padded_token_execution(stage, arm, profile):
    config = configuration(stage, arm, profile)
    assert denoiser(config)["token_execution"] == "padded"


@pytest.mark.parametrize("stage", ["video", "readout"])
@pytest.mark.parametrize("arm", ["continuous", "discrete"])
def test_packed_profile_changes_only_token_execution(stage, arm):
    baseline = configuration(stage, arm, "regional_compile")
    packed = configuration(stage, arm, "packed_regional_compile")
    assert denoiser(packed)["token_execution"] == "packed_valid"
    assert packed["world_model_execution"]["token_execution"] == "packed_valid"
    assert packed["world_model_execution"]["compile_mode"] == "default"
    assert packed["world_model_execution"]["compile_dynamic"] is True
    assert packed["world_model_execution"]["loss_reduction"] == "mean_per_example"
    normalized = copy.deepcopy(packed)
    denoiser(normalized)["token_execution"] = "padded"
    normalized["world_model_execution"].pop("token_execution")
    assert normalized == baseline


@pytest.mark.parametrize("arm", ["continuous", "discrete"])
def test_old_checkpoint_config_omission_resolves_to_padded(arm):
    source = denoiser(configuration("video", arm, "regional_compile"))
    target = denoiser(configuration("readout", arm, "regional_compile"))
    del source["token_execution"]
    assert resolve_model_config(source)["token_execution"] == "padded"
    resolved_source, resolved_target = validate_transition(source, target)
    assert (
        resolved_source["token_execution"]
        == resolved_target["token_execution"]
        == "padded"
    )


@pytest.mark.parametrize("arm", ["continuous", "discrete"])
def test_readout_initialization_requires_matching_token_execution(arm):
    source = denoiser(configuration("video", arm, "packed_regional_compile"))
    target = denoiser(configuration("readout", arm, "packed_regional_compile"))
    left, right = validate_transition(source, target)
    assert left["token_execution"] == right["token_execution"] == "packed_valid"
    for changed in (source, target):
        changed["token_execution"] = "padded"
        with pytest.raises(ValueError, match="configuration mismatch.*token_execution"):
            validate_transition(source, target)
        changed["token_execution"] = "packed_valid"


@pytest.mark.parametrize("arm", ["continuous", "discrete"])
def test_deferred_readout_keeps_its_short_budget_and_validation_schedule(arm):
    video = configuration("video", arm)
    readout = configuration("readout", arm)
    assert video["trainer"]["max_steps"] == 300000
    assert video["trainer"]["val_check_interval"] == 20000
    assert video["callbacks"]["model_checkpoint"]["every_n_train_steps"] == 20000
    assert video["callbacks"]["model_checkpoint"]["save_last"] == "link"
    assert readout["trainer"]["max_steps"] == 1000
    assert readout["trainer"]["val_check_interval"] == 250
    assert readout["callbacks"]["model_checkpoint"]["every_n_train_steps"] == 250
    assert readout["callbacks"]["model_checkpoint"]["save_last"] is True
