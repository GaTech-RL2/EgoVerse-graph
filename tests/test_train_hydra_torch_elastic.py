"""Explicit torchrun policy must supersede inherited one-task Slurm metadata."""

import pytest
from lightning.pytorch.plugins.environments import (
    SLURMEnvironment,
    TorchElasticEnvironment,
)
from omegaconf import OmegaConf

from egomimic.trainHydra import (
    _instantiate_slurm_callbacks,
    _instantiate_trainer_plugins,
)


@pytest.fixture
def elastic(monkeypatch):
    for key in ("ICE_REQUEUE_OWNER", "ICE_CHILD_REQUEUE_DISABLED"):
        monkeypatch.delenv(key, raising=False)
    for key, value in {
        "TORCHELASTIC_RUN_ID": "separate-run-rendezvous",
        "RANK": "2", "LOCAL_RANK": "2", "WORLD_SIZE": "4",
        "LOCAL_WORLD_SIZE": "4", "GROUP_RANK": "0",
        "MASTER_ADDR": "localhost", "MASTER_PORT": "29451",
        "SLURM_JOB_ID": "1234", "SLURM_NTASKS": "1",
        "SLURM_PROCID": "0", "SLURM_LOCALID": "0",
    }.items():
        monkeypatch.setenv(key, value)
    return OmegaConf.create({
        "mmap_checkpoint": False,
        "runtime": {"cluster_environment": "torch_elastic",
                    "slurm_requeue_owner": "none"},
    })


def test_torchrun_overrides_one_task_slurm_without_mutating_provenance(elastic):
    import os

    before = dict(os.environ)
    plugins = _instantiate_trainer_plugins(elastic)
    assert len(plugins) == 1
    plugin = plugins[0]
    assert isinstance(plugin, TorchElasticEnvironment)
    assert not isinstance(plugin, SLURMEnvironment)
    assert (plugin.world_size(), plugin.global_rank(), plugin.local_rank()) == (4, 2, 2)
    assert plugin.main_port == 29451
    plugin.validate_settings(num_devices=4, num_nodes=1)
    assert _instantiate_slurm_callbacks(elastic) == []
    assert dict(os.environ) == before


@pytest.mark.parametrize("key,value", [
    ("RANK", None), ("LOCAL_RANK", "four"), ("WORLD_SIZE", "0"),
    ("LOCAL_WORLD_SIZE", "3"), ("RANK", "4"), ("RANK", "-1"),
    ("LOCAL_RANK", "4"), ("GROUP_RANK", "1"),
    ("MASTER_ADDR", ""), ("MASTER_PORT", "0"),
    ("MASTER_PORT", "65536"), ("MASTER_PORT", "invalid"),
])
def test_malformed_torchrun_environment_fails(elastic, monkeypatch, key, value):
    if value is None:
        monkeypatch.delenv(key, raising=False)
    else:
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="torch_elastic"):
        _instantiate_trainer_plugins(elastic)


def test_explicit_elastic_requires_actual_torchrun(elastic, monkeypatch):
    monkeypatch.delenv("TORCHELASTIC_RUN_ID")
    with pytest.raises(RuntimeError, match="launched by torchrun"):
        _instantiate_trainer_plugins(elastic)


@pytest.mark.parametrize("owner", ["lightning", "runner"])
def test_elastic_requires_no_slurm_requeue_owner(elastic, owner):
    elastic.runtime.slurm_requeue_owner = owner
    with pytest.raises(ValueError, match="slurm_requeue_owner=none"):
        _instantiate_trainer_plugins(elastic)


def test_elastic_rejects_inherited_runner_ownership(elastic, monkeypatch):
    monkeypatch.setenv("ICE_REQUEUE_OWNER", "runner")
    with pytest.raises(RuntimeError, match="inconsistent Slurm requeue ownership"):
        _instantiate_trainer_plugins(elastic)


def test_unknown_environment_fails(elastic):
    elastic.runtime.cluster_environment = "torchrun_typo"
    with pytest.raises(ValueError, match="cluster_environment"):
        _instantiate_trainer_plugins(elastic)


def test_default_still_uses_slurm(elastic):
    del elastic.runtime.cluster_environment
    plugins = _instantiate_trainer_plugins(elastic)
    assert len(plugins) == 1 and isinstance(plugins[0], SLURMEnvironment)
    assert plugins[0].world_size() == 1
