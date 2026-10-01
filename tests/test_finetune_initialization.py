"""Weights-only transfer preserves new action heads and cannot override resume."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from egomimic.eval.checkpoint_loading import init_pipeline_weights_from_checkpoint
from egomimic.trainHydra import _initialize_finetune_weights


def _model(width):
    return SimpleNamespace(nets=nn.Sequential(nn.BatchNorm1d(3), nn.Linear(3, width)))


def _checkpoint(model):
    return {
        "state_dict": {
            "nets." + k: v.clone() for k, v in model.nets.state_dict().items()
        },
        "ema_state_dict": {
            "nets." + k: torch.full_like(v, 2) for k, v in model.nets.named_parameters()
        },
        "global_step": 210000,
        "optimizer_states": [{"unrelated": "must not load"}],
    }


def test_finetune_transfers_ema_and_online_buffers_without_resuming():
    source, target = _model(14), _model(16)
    source.nets[0].running_mean.fill_(7)
    before = {k: v.clone() for k, v in target.nets.state_dict().items()}
    optimizer = torch.optim.Adam(target.nets.parameters())
    _, changed = init_pipeline_weights_from_checkpoint(
        target, _checkpoint(source), use_ema=True
    )
    assert {row[0] for row in changed} == {"1.weight", "1.bias"}
    assert changed[0][1:] == ((14, 3), (16, 3))
    assert torch.equal(target.nets[1].weight, before["1.weight"])
    assert torch.equal(target.nets[1].bias, before["1.bias"])
    assert torch.equal(target.nets[0].weight, torch.full((3,), 2.0))
    assert torch.equal(target.nets[0].running_mean, torch.full((3,), 7.0))
    assert optimizer.state == {}


@pytest.mark.parametrize("invalid", ["keys", "nan"])
def test_failed_transfer_does_not_partially_mutate_model(invalid):
    model = _model(14)
    before = {k: v.clone() for k, v in model.nets.state_dict().items()}
    checkpoint = _checkpoint(_model(14))
    if invalid == "keys":
        checkpoint["state_dict"]["nets.extra"] = torch.zeros(1)
    else:
        checkpoint["state_dict"]["nets.1.bias"][0] = float("nan")
    with pytest.raises(ValueError):
        init_pipeline_weights_from_checkpoint(model, checkpoint)
    for key, value in model.nets.state_dict().items():
        assert torch.equal(value, before[key])


@pytest.mark.parametrize("resume,restart", [("last.ckpt", "0"), (None, "2")])
def test_resume_and_requeue_never_open_initialization_checkpoint(
    monkeypatch, resume, restart
):
    monkeypatch.setenv("SLURM_RESTART_COUNT", restart)

    def forbidden(*args, **kwargs):
        pytest.fail("resume must not load the fine-tuning checkpoint")

    monkeypatch.setattr(torch, "load", forbidden)
    _initialize_finetune_weights(
        OmegaConf.create({"init_weights_from": "unreadable.ckpt", "ckpt_path": resume}),
        SimpleNamespace(model=_model(16)),
    )


def test_first_finetune_logs_changed_heads(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("SLURM_RESTART_COUNT", "0")
    checkpoint = tmp_path / "init.ckpt"
    torch.save(_checkpoint(_model(14)), checkpoint)
    _initialize_finetune_weights(
        OmegaConf.create({"init_weights_from": str(checkpoint)}),
        SimpleNamespace(model=_model(16)),
    )
    assert "1.weight" in caplog.text and "1.bias" in caplog.text
