import copy
from functools import partial

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from egomimic.pl_utils.pl_model_unite_released import ReleasedUniteModelWrapper
from egomimic.trainHydra import _resolve_model_wrapper_class
from egomimic.utils.unite_optim import (
    ReleasedUniteCompositeOptimizer,
    VendoredTorchMuon,
    partition_released_unite_parameters,
    released_unite_two_stage_scheduler,
)


class _TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        visual = nn.Module()
        visual.encoder = nn.Module()
        visual.encoder.img_encoders = nn.ModuleDict(
            {"front_img_1": nn.Sequential(nn.Conv2d(3, 2, 1), nn.Flatten())}
        )
        generative = nn.Module()
        generative.blocks = nn.ModuleList([nn.Linear(2, 2)])
        generative.content_projection = nn.Linear(2, 2)
        generative.condition_projection = nn.Linear(2, 2)
        generative.final_layer = nn.Linear(2, 2)
        generative.action_projection = nn.Linear(2, 2)
        self.stages = nn.ModuleList([visual, generative])


class _TinyAlgo:
    def __init__(self):
        self.nets = nn.ModuleDict({"policy": _TinyPolicy()})

    def process_batch_for_training(self, batch):
        return batch

    def forward_training(self, batch):
        return batch

    def compute_losses(self, predictions, batch):
        del predictions, batch
        parameter = self.nets["policy"].stages[1].blocks[0].weight
        return {
            "0_loss_unite_reconstruction": parameter.square().mean(),
            "0_loss_unite_latent": (parameter - 1.0).square().mean(),
        }

    def log_info(self, info):
        assert "action_loss" in info["losses"]
        return {}


def _optimizer(model) -> ReleasedUniteCompositeOptimizer:
    return ReleasedUniteCompositeOptimizer(
        model.named_parameters(),
        lr=1.0e-3,
        adamw_weight_decay=0.0,
        muon_weight_decay=0.0,
    )


def _quadratic_loss(model):
    return sum(parameter.square().sum() for parameter in model.parameters())


def test_parameter_partition_is_complete_disjoint_and_name_stable():
    model = _TinyAlgo().nets
    named = tuple(model.named_parameters(prefix="nets", remove_duplicate=True))
    adamw, muon = partition_released_unite_parameters(named)
    adamw_names = {name for name, _ in adamw}
    muon_names = {name for name, _ in muon}
    assert adamw_names.isdisjoint(muon_names)
    assert adamw_names | muon_names == {name for name, _ in named}
    assert len(named) == len({id(parameter) for _, parameter in named})
    assert any("img_encoders" in name for name in adamw_names)
    assert any("content_projection" in name for name in adamw_names)
    assert any("condition_projection" in name for name in adamw_names)
    assert any("blocks.0.weight" in name for name in muon_names)
    assert any("action_projection.weight" in name for name in muon_names)


def test_composite_optimizer_step_and_state_round_trip():
    torch.manual_seed(7)
    model_a = _TinyAlgo().nets
    optimizer_a = _optimizer(model_a)
    _quadratic_loss(model_a).backward()
    optimizer_a.step()
    optimizer_a.zero_grad()
    assert isinstance(optimizer_a.muon, VendoredTorchMuon)
    assert optimizer_a.muon.state
    state = copy.deepcopy(optimizer_a.state_dict())

    model_b = _TinyAlgo().nets
    model_b.load_state_dict(model_a.state_dict())
    optimizer_b = _optimizer(model_b)
    optimizer_b.load_state_dict(state)
    _quadratic_loss(model_a).backward()
    _quadratic_loss(model_b).backward()
    optimizer_a.step()
    optimizer_b.step()
    for left, right in zip(model_a.parameters(), model_b.parameters()):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)


def test_released_scheduler_intentionally_keeps_phase_two_beyond_run():
    parameter = nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([parameter], lr=1.0e-4)
    scheduler = released_unite_two_stage_scheduler(
        optimizer,
        warmup_steps=8_000,
        decay_start_1_steps=12_000,
        decay_end_1_steps=20_000,
        decay_start_2_steps=1_200_000,
        decay_end_2_steps=1_200_000,
        base_lr_1=1.0e-4,
        base_lr_2=5.0e-5,
        final_lr=1.0e-5,
    )
    multiplier = scheduler.lr_lambdas[0]
    assert multiplier(0) == 0.0
    assert multiplier(8_000) == 1.0
    assert multiplier(12_000) == 1.0
    assert multiplier(20_000) == 0.5
    assert multiplier(240_000) == 0.5
    assert multiplier(1_200_000) == pytest.approx(0.1)


def test_released_wrapper_uses_registered_nets_and_saves_topology(monkeypatch):
    wrapper = ReleasedUniteModelWrapper(
        robomimic_model=_TinyAlgo(),
        optimizer=partial(ReleasedUniteCompositeOptimizer, lr=1.0e-3),
        scheduler=None,
        share_encoder_denoiser=True,
    )
    configured = wrapper.configure_optimizers()
    optimizer = configured["optimizer"]
    expected = tuple(
        wrapper.nets.named_parameters(prefix="nets", remove_duplicate=True)
    )
    grouped_names = set(optimizer.group_manifest["adamw_parameter_names"]) | set(
        optimizer.group_manifest["muon_parameter_names"]
    )
    assert grouped_names == {name for name, _ in expected}

    checkpoint = {}
    wrapper.on_save_checkpoint(checkpoint)
    assert checkpoint["hyper_parameters"]["share_encoder_denoiser"] is True

    logged = {}
    monkeypatch.setattr(
        wrapper,
        "log",
        lambda name, value, **_kwargs: logged.setdefault(name, value),
    )
    loss = wrapper.training_step({}, 0)
    assert bool(torch.isfinite(loss))
    loss.backward()
    assert "Train/UNITE/ReconstructionLoss" in logged
    assert "Train/UNITE/FlowLoss" in logged


def test_training_entry_resolves_only_model_wrapper_subclasses():
    cfg = OmegaConf.create(
        {
            "model": {
                "_target_": (
                    "egomimic.pl_utils.pl_model_unite_released."
                    "ReleasedUniteModelWrapper"
                )
            }
        }
    )
    assert _resolve_model_wrapper_class(cfg) is ReleasedUniteModelWrapper
    cfg.model._target_ = "torch.nn.Linear"
    with pytest.raises(TypeError, match="ModelWrapper subclass"):
        _resolve_model_wrapper_class(cfg)
