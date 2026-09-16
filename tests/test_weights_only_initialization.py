from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

import egomimic.trainHydra as train_hydra
from egomimic.pipeline.algo import PipelineAlgo


class _StateDictOnlyCheckpoint(dict):
    """Fail if the initialization path reads non-model checkpoint state."""

    def __init__(self, state_dict):
        super().__init__(
            state_dict=state_dict,
            optimizer_states="must not be restored",
            lr_schedulers="must not be restored",
            global_step=210000,
            callbacks="must not be restored",
            loops="must not be restored",
        )
        self.read_keys = []

    def __getitem__(self, key):
        self.read_keys.append(key)
        if key != "state_dict":
            raise AssertionError(f"weights-only initialization read {key!r}")
        return super().__getitem__(key)


def _pipeline_with_one_parameter():
    algo = PipelineAlgo(stages=[], device="cpu")
    algo.nets["tiny"] = torch.nn.Linear(2, 1, bias=False)
    return algo


def _cfg(path: Path, *, ckpt_path=None):
    return OmegaConf.create(
        {
            "ckpt_path": ckpt_path,
            "weights_only_init": {"path": str(path), "use_ema": False},
        }
    )


def test_weights_only_initialization_reads_only_pipeline_weights(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "source.ckpt"
    checkpoint_path.touch()
    algo = _pipeline_with_one_parameter()
    source_weight = torch.tensor([[2.0, -3.0]])
    checkpoint = _StateDictOnlyCheckpoint({"nets.tiny.weight": source_weight.clone()})
    load_calls = []

    def load_checkpoint(_self, path, *, map_location, weights_only):
        load_calls.append((path, map_location, weights_only))
        return checkpoint

    monkeypatch.setattr(
        train_hydra.MmapCheckpointIO, "load_checkpoint", load_checkpoint
    )

    loaded_path = train_hydra._apply_weights_only_initialization(
        SimpleNamespace(model=algo), _cfg(checkpoint_path)
    )

    assert loaded_path == str(checkpoint_path.resolve())
    assert checkpoint.read_keys == ["state_dict"]
    assert load_calls == [(str(checkpoint_path.resolve()), "cpu", False)]
    assert torch.equal(algo.nets["tiny"].weight.detach(), source_weight)


def test_weights_only_initialization_rejects_full_state_resume(tmp_path):
    checkpoint_path = tmp_path / "source.ckpt"
    checkpoint_path.touch()

    with pytest.raises(ValueError, match="mutually exclusive"):
        train_hydra._weights_only_initialization_spec(
            _cfg(checkpoint_path, ckpt_path="/tmp/full-resume.ckpt")
        )


def test_weights_only_initialization_rejects_incompatible_weights_before_mutation(
    monkeypatch, tmp_path
):
    checkpoint_path = tmp_path / "source.ckpt"
    checkpoint_path.touch()
    algo = _pipeline_with_one_parameter()
    before = algo.nets["tiny"].weight.detach().clone()
    checkpoint = _StateDictOnlyCheckpoint({"nets.tiny.weight": torch.ones((2, 2))})

    monkeypatch.setattr(
        train_hydra.MmapCheckpointIO,
        "load_checkpoint",
        lambda *_args, **_kwargs: checkpoint,
    )

    with pytest.raises(ValueError, match="tensor mismatch"):
        train_hydra._apply_weights_only_initialization(
            SimpleNamespace(model=algo), _cfg(checkpoint_path)
        )

    assert torch.equal(algo.nets["tiny"].weight.detach(), before)


def test_base_config_disables_weights_only_initialization():
    config_path = (
        Path(__file__).parents[1]
        / "egomimic"
        / "hydra_configs"
        / "train_zarr_cartesian.yaml"
    )
    cfg = OmegaConf.load(config_path)

    assert cfg.weights_only_init.path is None
    assert cfg.weights_only_init.use_ema is False
