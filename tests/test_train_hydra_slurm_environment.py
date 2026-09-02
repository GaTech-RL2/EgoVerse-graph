import signal
from pathlib import Path

import pytest
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import OmegaConf

import egomimic.trainHydra as train_hydra
from egomimic.utils.slurm_requeue import SaveOnlySignalCheckpoint

_OWNERSHIP_VARIABLES = ("ICE_REQUEUE_OWNER", "ICE_CHILD_REQUEUE_DISABLED")


def _clear_ownership(monkeypatch):
    for name in _OWNERSHIP_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def _cfg(owner="lightning"):
    return OmegaConf.create(
        {
            "mmap_checkpoint": False,
            "runtime": {"slurm_requeue_owner": owner},
        }
    )


def test_base_config_records_lightning_as_default_requeue_owner():
    config_path = (
        Path(__file__).parents[1]
        / "egomimic"
        / "hydra_configs"
        / "train_zarr_cartesian.yaml"
    )
    cfg = OmegaConf.load(config_path)
    cfg.paths = {"output_dir": "/tmp/egomimic-test-output"}

    assert cfg.runtime.slurm_requeue_owner == "lightning"
    assert cfg.runtime.slurm_save_signal == "SIGUSR2"
    assert (
        cfg.runtime.slurm_signal_checkpoint_dir
        == "/tmp/egomimic-test-output/checkpoints"
    )


def test_skynet_default_keeps_lightning_auto_requeue(monkeypatch):
    _clear_ownership(monkeypatch)
    plugin = train_hydra._slurm_environment(_cfg())

    assert isinstance(plugin, SLURMEnvironment)
    assert plugin.auto_requeue is True
    assert plugin.requeue_signal == signal.SIGUSR1


def test_runner_owned_job_builds_one_non_requeueing_slurm_environment(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("ICE_REQUEUE_OWNER", "runner")
    monkeypatch.setenv("ICE_CHILD_REQUEUE_DISABLED", "1")

    plugins = train_hydra._instantiate_trainer_plugins(_cfg("runner"))

    slurm_plugins = [
        plugin for plugin in plugins if isinstance(plugin, SLURMEnvironment)
    ]
    assert len(slurm_plugins) == 1
    assert slurm_plugins[0].auto_requeue is False
    assert slurm_plugins[0].requeue_signal == signal.SIGUSR1


def test_runner_owned_job_adds_one_save_only_sigusr2_callback(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("ICE_REQUEUE_OWNER", "runner")
    monkeypatch.setenv("ICE_CHILD_REQUEUE_DISABLED", "1")
    cfg = _cfg("runner")
    cfg.runtime.slurm_save_signal = "SIGUSR2"
    cfg.runtime.slurm_signal_checkpoint_dir = str(tmp_path / "checkpoints")

    callbacks = train_hydra._instantiate_slurm_callbacks(cfg)

    assert len(callbacks) == 1
    assert isinstance(callbacks[0], SaveOnlySignalCheckpoint)
    assert callbacks[0].save_signal == signal.SIGUSR2
    assert callbacks[0].checkpoint_dir == (tmp_path / "checkpoints").resolve()


def test_child_owned_job_keeps_lightning_auto_requeue(monkeypatch):
    monkeypatch.setenv("ICE_REQUEUE_OWNER", "child")
    monkeypatch.setenv("ICE_CHILD_REQUEUE_DISABLED", "0")

    plugin = train_hydra._slurm_environment(_cfg())

    assert plugin.auto_requeue is True


@pytest.mark.parametrize(
    ("configured_owner", "owner", "disabled"),
    [
        ("runner", "runner", None),
        ("runner", "runner", "0"),
        ("runner", None, None),
        ("lightning", "runner", "1"),
        ("lightning", "child", None),
        ("lightning", "child", "1"),
        ("lightning", None, "1"),
        ("unknown", "runner", "1"),
    ],
)
def test_inconsistent_requeue_ownership_fails_closed(
    monkeypatch, configured_owner, owner, disabled
):
    _clear_ownership(monkeypatch)
    if owner is not None:
        monkeypatch.setenv("ICE_REQUEUE_OWNER", owner)
    if disabled is not None:
        monkeypatch.setenv("ICE_CHILD_REQUEUE_DISABLED", disabled)

    with pytest.raises(RuntimeError, match="inconsistent Slurm requeue ownership"):
        train_hydra._slurm_environment(_cfg(configured_owner))
