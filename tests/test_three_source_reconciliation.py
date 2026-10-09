"""Behavior contracts required by the stationary/LIBERO shared-source merge."""

import pytest
import torch
from torch import nn

from egomimic.pipeline.stages_action_flow import (
    ActionFlowObjectiveStage,
    ConditionalVelocityStage,
    ContentDecoderStage,
    ContentEncoderStage,
)


class Field(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.4, dtype=torch.float64))
        self.calls = 0

    def forward(self, state, time, condition, *, condition_drop_mask):
        self.calls += 1
        return self.scale * state.tanh() + condition[:, None] + time[:, None, None]


def execute(mode):
    state = torch.linspace(-1, 1, 8, dtype=torch.float64).reshape(2, 2, 2).requires_grad_()
    condition = torch.ones(2, 2, dtype=torch.float64, requires_grad=True)
    field = Field()
    stage = ConditionalVelocityStage(
        field, flow_clean_gradient_mode="all_stopgrad", fm_field_execution=mode,
    )
    batch = stage({
        "action_flow/state": state, "action_flow/time": torch.ones(2, dtype=torch.float64),
        "action_flow/condition": condition, "action_flow/condition_drop_mask": torch.zeros(2, dtype=torch.bool),
        "action_flow/target_velocity": -state,
    })
    fm = batch["action_flow/fm_velocity_residual"].square().mean()
    action = batch["action_flow/velocity_residual"].square().mean()
    fm_state = torch.autograd.grad(fm, state, retain_graph=True, allow_unused=True)[0]
    assert fm_state is None or torch.count_nonzero(fm_state) == 0
    gradients = torch.autograd.grad(fm + action, (state, condition, field.scale))
    return field.calls, fm.detach(), action.detach(), gradients


def test_explicit_field_execution_preserves_each_source_and_gradients():
    separate, shared = execute("separate"), execute("shared")
    assert (separate[0], shared[0]) == (2, 1)
    for a, b in zip(separate[1:3], shared[1:3]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(separate[3], shared[3]):
        torch.testing.assert_close(a, b, rtol=1e-14, atol=1e-14)
    with pytest.raises(ValueError, match="fm_field_execution"):
        ConditionalVelocityStage(Field(), fm_field_execution="automatic")


def test_routing_consumes_an_opaque_declared_key_not_a_model_family():
    encoders = {"x": nn.Identity(), "y": nn.Identity()}
    decoders = {"x": nn.Identity(), "y": nn.Identity()}
    encoder = ContentEncoderStage(encoders=encoders, selector_key="opaque", selector_aliases={"19": "x"})
    decoder = ContentDecoderStage(decoders=decoders, selector_key="opaque", selector_aliases={"19": "x"})
    batch = {"opaque": torch.tensor([19, 19]), "target": torch.ones(2, 3, 4)}
    assert encoder.encoder_for(batch) is encoders["x"]
    assert decoder.decoder_for(batch) is decoders["x"]
    with pytest.raises(ValueError):
        encoder.encoder_for({"opaque": torch.tensor([19, 20])})


def test_optional_moment_objective_survives_merge_without_default_loss_change():
    batch = {
        "target": torch.zeros(2, 2), "action_flow/reconstruction": torch.ones(2, 2),
        "action_flow/velocity_residual": torch.ones(2, 2),
        "action_flow/decoded_velocity_residual": torch.ones(2, 2),
        "action_flow/decoded_noise": torch.tensor([[1., 0.], [-1., 0.]]),
    }
    default = ActionFlowObjectiveStage()(dict(batch))
    enabled = ActionFlowObjectiveStage(moment_weight=2)(dict(batch))
    assert default["loss/action_flow"] == 3
    torch.testing.assert_close(
        enabled["loss/action_flow"], default["loss/action_flow"] + 2 * enabled["log/action_flow_decoded_noise_moments"],
    )


def test_lazy_proportional_adapter_retains_native_loader_contract():
    from egomimic.pl_utils.pl_data_utils import ProportionalMultiDataModuleWrapper
    from egomimic.pl_utils.data_context import ContextDataModule
    from egomimic.rldb.zarr.data_module import ProportionalZarrDataModule

    module = ProportionalZarrDataModule(
        train_datasets={}, valid_datasets={},
        train_dataloader_params={}, valid_dataloader_params={},
        proportional_train_batch_size=8, proportional_train_num_workers=0,
        proportional_train_seed=42,
    )
    assert isinstance(module, ContextDataModule)
    assert module.proportional_train_batch_size == 8
    assert type(module).train_dataloader is ProportionalMultiDataModuleWrapper.train_dataloader
    assert type(module).val_dataloader is ProportionalMultiDataModuleWrapper.val_dataloader


def test_partial_normalization_resume_is_not_used_for_training():
    from types import SimpleNamespace
    from egomimic.rldb.zarr.data_module import ZarrDataModule

    calls = []
    owner = SimpleNamespace(infer_norm_from_dataset=lambda *a, **kw: calls.append(kw))
    options = {"resume_partial_norm_path": "/exact/task/partial.json"}
    ZarrDataModule._fit_normalizer(None, owner, object(), "opaque", 19, options, "train", None)
    ZarrDataModule._fit_normalizer(None, owner, object(), "opaque", 19, options, "normalization", None)
    assert calls[0]["resume_partial_norm_path"] is None
    assert calls[1]["resume_partial_norm_path"] == options["resume_partial_norm_path"]
