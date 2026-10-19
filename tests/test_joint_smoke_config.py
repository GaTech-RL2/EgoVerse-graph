from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from egomimic.pl_utils.joint_smoke import load_smoke_configs
from egomimic.trainHydra import _build_model_config_tree


def snapshots(root, arm="continuous"):
    with initialize_config_dir(
        config_dir=str(Path("egomimic/hydra_configs").resolve()), version_base=None
    ):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=[
                f"+experiment=pixel_proprio/joint_{arm}",
                "+joint_execution=smoke",
                "hydra/launcher=basic",
                "logger=[csv]",
                "~logger.wandb",
                f"paths.output_dir={root}",
                "data.manifest_path=/fixture/manifest.json",
                "+run_provenance={source_revision:fixture-source,data_context_sha256:fixture-context}",
            ],
        )
    minimal = _build_model_config_tree(cfg)
    OmegaConf.set_struct(cfg, False)
    assert "data" not in minimal and "trainer" not in minimal
    (root / ".hydra").mkdir()
    OmegaConf.save(cfg, root / ".hydra/config.yaml")
    OmegaConf.save(minimal, root / "smoke-config.yaml")
    return cfg, minimal


@pytest.mark.parametrize("arm", ["discrete", "continuous"])
def test_postflight_reads_full_hydra_and_preserves_model_only_checkpoint_tree(
    tmp_path, arm
):
    cfg, minimal = snapshots(tmp_path, arm)
    loaded, model_config = load_smoke_configs(tmp_path)
    assert loaded.data.couple_state_observations
    assert loaded.trainer.devices == (2 if arm == "discrete" else 4)
    assert loaded.model.scheduler.warmup_steps == 3000
    assert model_config == minimal
    assert OmegaConf.to_container(loaded.model, resolve=True) == OmegaConf.to_container(
        cfg.model, resolve=True
    )


@pytest.mark.parametrize("damage", ["model", "provenance", "output", "missing_data"])
def test_postflight_rejects_mismatched_full_run_snapshot(tmp_path, damage):
    cfg, _ = snapshots(tmp_path)
    if damage == "model":
        cfg.model.optimizer.lr = 1e-3
    elif damage == "provenance":
        cfg.run_provenance.source_revision = "another-source"
    elif damage == "output":
        cfg.paths.output_dir = str(tmp_path / "another-output")
    else:
        del cfg["data"]
    OmegaConf.save(cfg, tmp_path / ".hydra/config.yaml")
    with pytest.raises(ValueError):
        load_smoke_configs(tmp_path)


def test_postflight_does_not_fall_back_to_partial_config(tmp_path):
    snapshots(tmp_path)
    (tmp_path / ".hydra/config.yaml").unlink()
    with pytest.raises(FileNotFoundError):
        load_smoke_configs(tmp_path)
