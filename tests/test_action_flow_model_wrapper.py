import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pipeline.core import Stage
from egomimic.pipeline.stages_action_flow import (
    ConditionalVelocityStage,
    ContentDecoderStage,
    ContentEncoderStage,
)
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.pl_utils.pl_model_action_flow import ActionFlowModelWrapper


class _ToyPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))


class _ToyAlgo:
    def __init__(
        self, *, reconstruction_weight=1.0, flow_weight=1.0, total_metric_delta=0.0
    ):
        self.nets = nn.ModuleDict({"pipeline": _ToyPipeline()})
        self.device = torch.device("cpu")
        self.reconstruction_weight = float(reconstruction_weight)
        self.flow_weight = float(flow_weight)
        self.action_velocity_weight = 1.0
        self.total_metric_delta = float(total_metric_delta)
        self.process_count = 0

    def process_batch_for_training(self, batch):
        self.process_count += 1
        return OrderedDict((source, dict(values)) for source, values in batch.items())

    def forward_training(self, batch):
        anchor = self.nets["pipeline"].anchor
        results = OrderedDict()
        for source, values in batch.items():
            fm = anchor * anchor.new_tensor(values["fm"])
            reconstruction = anchor * anchor.new_tensor(values["reconstruction"])
            reconstruction_l1 = anchor * anchor.new_tensor(values["reconstruction_l1"])
            action_velocity = anchor * anchor.new_tensor(values["action_velocity"])
            total = (
                self.flow_weight * fm
                + self.reconstruction_weight * reconstruction
                + action_velocity
            )
            results[source] = {
                "target": values["target"],
                "loss/action_flow": total,
                "log/action_flow_total": total + self.total_metric_delta,
                "log/action_flow_fm": fm,
                "log/action_flow_reconstruction": reconstruction,
                "log/action_flow_reconstruction_l1": reconstruction_l1,
                "log/action_flow_action_velocity": action_velocity,
                "log/latent_rms": anchor.detach(),
            }
        return results

    @torch.inference_mode()
    def forward_eval(self, batch):
        anchor = self.nets["pipeline"].anchor
        return OrderedDict(
            (source, {"prediction": values["condition"] * anchor})
            for source, values in batch.items()
        )


class _ActionFlowHead(Stage):
    reads = ("target",)
    writes = (
        "loss/action_flow",
        "log/action_flow_total",
        "log/action_flow_fm",
        "log/action_flow_reconstruction",
        "log/action_flow_reconstruction_l1",
        "log/action_flow_action_velocity",
    )
    reads_by_mode = {"inference": ("condition",)}
    writes_by_mode = {"inference": ("prediction",)}

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))

    def execute(self, batch, *, mode):
        if mode == "inference":
            batch["prediction"] = self.anchor.expand(
                int(batch["condition"].shape[0]), 2, 3
            )
            return batch
        fm = self.anchor.square()
        reconstruction = (self.anchor - batch["target"]).square().mean()
        reconstruction_l1 = (self.anchor - batch["target"]).abs().mean()
        action_velocity = (2.0 * self.anchor).square()
        total = fm + reconstruction + action_velocity
        batch.update(
            {
                "loss/action_flow": total,
                "log/action_flow_total": total,
                "log/action_flow_fm": fm,
                "log/action_flow_reconstruction": reconstruction,
                "log/action_flow_reconstruction_l1": reconstruction_l1,
                "log/action_flow_action_velocity": action_velocity,
            }
        )
        return batch

    def forward(self, batch):
        return self.execute(batch, mode="train")


def _batch():
    return OrderedDict(
        source_a={
            "target": torch.zeros(1, 2, 3),
            "condition": torch.zeros(1, 3),
            "fm": 1.0,
            "reconstruction": 2.0,
            "reconstruction_l1": 4.0,
            "action_velocity": 3.0,
        },
        source_b={
            "target": torch.zeros(3, 2, 3),
            "condition": torch.zeros(3, 3),
            "fm": 5.0,
            "reconstruction": 6.0,
            "reconstruction_l1": 8.0,
            "action_velocity": 7.0,
        },
    )


def _capture_logs(monkeypatch, wrapper):
    logged = {}
    monkeypatch.setattr(
        wrapper,
        "log",
        lambda name, value, **kwargs: logged.__setitem__(name, (value, kwargs)),
    )
    return logged


def test_action_flow_wrapper_uses_only_explicit_optimizer_total(monkeypatch):
    wrapper = ActionFlowModelWrapper(
        pipeline=_ToyAlgo(reconstruction_weight=10.0),
        gradient_telemetry_cadence=0,
    )
    logged = _capture_logs(monkeypatch, wrapper)

    loss = wrapper.training_step(_batch(), batch_idx=0)

    # Sample-weighted components are fm=4, reconstruction=5, action=6, so the
    # configured optimizer total is 4 + 10*5 + 6 = 60. Summing every metric and
    # the total would incorrectly return 75.
    assert float(loss) == pytest.approx(60.0)
    assert float(loss) != pytest.approx(75.0)
    expected = {
        "Train/ActionFlow/TotalLoss": 60.0,
        "Train/ActionFlow/FlowMatchingLoss": 4.0,
        "Train/ActionFlow/ReconstructionLoss": 5.0,
        "Train/ActionFlow/ReconstructionL1": 7.0,
        "Train/ActionFlow/ActionVelocityLoss": 6.0,
    }
    for name, expected_value in expected.items():
        value, kwargs = logged[name]
        assert float(value) == pytest.approx(expected_value)
        assert kwargs == {
            "on_step": True,
            "on_epoch": True,
            "sync_dist": False,
            "batch_size": 4,
        }
    assert "Train/action_flow_total" not in logged
    assert "Train/latent_rms" in logged
    assert float(logged["Train/ActionFlow/TotalLoss/source_a"][0]) == pytest.approx(
        24.0
    )
    assert float(logged["Train/ActionFlow/TotalLoss/source_b"][0]) == pytest.approx(
        72.0
    )
    assert float(logged["Train/MSE"][0]) == pytest.approx(5.0)
    assert float(logged["Train/MSE/source_a"][0]) == pytest.approx(2.0)
    assert float(logged["Train/MSE/source_b"][0]) == pytest.approx(6.0)


def test_reconstruction_only_warmup_then_joint_objective(monkeypatch):
    wrapper = ActionFlowModelWrapper(
        pipeline=_ToyAlgo(reconstruction_weight=10.0),
        gradient_telemetry_cadence=0,
        reconstruction_only_warmup_steps=2,
    )
    logged = _capture_logs(monkeypatch, wrapper)

    warmup_loss = wrapper.training_step(_batch(), batch_idx=0)
    assert float(warmup_loss) == pytest.approx(50.0)
    assert float(logged["Train/ActionFlow/FlowMatchingLoss"][0]) == pytest.approx(4.0)
    assert float(logged["Train/ActionFlow/ActionVelocityLoss"][0]) == pytest.approx(
        6.0
    )
    assert float(logged["Train/ActionFlow/Schedule/ReconstructionOnly"][0]) == 1.0
    assert float(logged["Train/ActionFlow/Schedule/EffectiveFlowWeight"][0]) == 0.0

    predictions = wrapper.model.forward_training(
        wrapper.model.process_batch_for_training(_batch())
    )
    assert not wrapper._apply_reconstruction_only_warmup(
        predictions, optimizer_step=2
    )
    _, components, joint_loss, _ = wrapper._source_values(predictions)
    assert float(joint_loss) == pytest.approx(60.0)
    assert float(components["TotalLoss"]) == pytest.approx(60.0)


def test_reconstruction_only_warmup_is_loaded_from_training_config_tree(monkeypatch):
    monkeypatch.setattr(
        ActionFlowModelWrapper,
        "_instantiate_model",
        lambda self, config_tree: _ToyAlgo(reconstruction_weight=10.0),
    )
    wrapper = ActionFlowModelWrapper(
        config_tree={
            "model": {
                "pipeline": {},
                "reconstruction_only_warmup_steps": 2,
            }
        },
        gradient_telemetry_cadence=0,
    )

    assert wrapper.reconstruction_only_warmup_steps == 2
    checkpoint = {}
    wrapper.on_save_checkpoint(checkpoint)
    assert checkpoint["action_flow_loss_schedule"] == {
        "joint_objective_begins_at_global_step": 2,
        "reconstruction_only_optimizer_steps": 2,
        "joint_flow_weight": 1.0,
        "joint_reconstruction_weight": 10.0,
        "joint_action_velocity_weight": 1.0,
        "schema_version": 1,
    }


def test_joint_flow_weight_is_applied_and_logged(monkeypatch):
    wrapper = ActionFlowModelWrapper(
        pipeline=_ToyAlgo(reconstruction_weight=10.0, flow_weight=0.01),
        gradient_telemetry_cadence=0,
    )
    logged = _capture_logs(monkeypatch, wrapper)

    loss = wrapper.training_step(_batch(), batch_idx=0)

    assert float(loss) == pytest.approx(56.04)
    assert float(
        logged["Train/ActionFlow/Schedule/EffectiveFlowWeight"][0]
    ) == pytest.approx(0.01)


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_reconstruction_only_warmup_rejects_invalid_steps(value):
    with pytest.raises(ValueError, match="nonnegative integer"):
        ActionFlowModelWrapper(
            pipeline=_ToyAlgo(reconstruction_weight=10.0),
            gradient_telemetry_cadence=0,
            reconstruction_only_warmup_steps=value,
        )


def test_action_flow_wrapper_rejects_total_metric_drift(monkeypatch):
    wrapper = ActionFlowModelWrapper(
        pipeline=_ToyAlgo(total_metric_delta=1.0),
        gradient_telemetry_cadence=0,
    )
    _capture_logs(monkeypatch, wrapper)

    with pytest.raises(RuntimeError, match="optimizer loss and total metric disagree"):
        wrapper.training_step(_batch(), batch_idx=0)


def test_action_flow_wrapper_rejects_non_finite_components(monkeypatch):
    wrapper = ActionFlowModelWrapper(pipeline=_ToyAlgo(), gradient_telemetry_cadence=0)
    _capture_logs(monkeypatch, wrapper)
    batch = _batch()
    batch["source_a"]["fm"] = float("nan")

    with pytest.raises(RuntimeError, match="Non-finite Action Flow value"):
        wrapper.training_step(batch, batch_idx=0)


def test_distributed_gradient_makes_strided_autograd_values_contiguous(monkeypatch):
    gradient = torch.arange(12, dtype=torch.float32).reshape(3, 4).T
    original = gradient.clone()
    observed = {}

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def fake_all_reduce(value, op):
        observed["contiguous"] = value.is_contiguous()
        observed["op"] = op
        value.mul_(2)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    reduced = ActionFlowModelWrapper._distributed_gradient(gradient)

    assert observed == {
        "contiguous": True,
        "op": torch.distributed.ReduceOp.SUM,
    }
    assert reduced.is_contiguous()
    assert torch.equal(reduced, original)
    assert torch.equal(gradient, original)


def test_action_flow_wrapper_measures_component_gradient_intersections(monkeypatch):
    wrapper = ActionFlowModelWrapper(pipeline=_ToyAlgo(), gradient_telemetry_cadence=1)
    wrapper.flow_samples_per_content = 14
    logged = _capture_logs(monkeypatch, wrapper)
    batch = OrderedDict(
        source={
            "target": torch.zeros(2, 2, 3),
            "fm": 1.0,
            "reconstruction": 2.0,
            "reconstruction_l1": 2.0,
            "action_velocity": 3.0,
        }
    )

    loss = wrapper.training_step(batch, batch_idx=0)

    assert loss.requires_grad
    for name, expected in (
        ("Train/ActionFlow/GradientNorm/FM", 1.0),
        ("Train/ActionFlow/GradientNorm/Reconstruction", 2.0),
        ("Train/ActionFlow/GradientNorm/ActionVelocity", 3.0),
        (
            "Train/ActionFlow/GradientCosine/FM__Reconstruction",
            1.0,
        ),
        (
            "Train/ActionFlow/GradientCosine/FM__ActionVelocity",
            1.0,
        ),
        (
            "Train/ActionFlow/GradientCosine/Reconstruction__ActionVelocity",
            1.0,
        ),
    ):
        assert float(logged[name][0]) == pytest.approx(expected)
    assert float(
        logged[
            "Train/ActionFlow/GradientIntersectionParameterCount/FM__ActionVelocity"
        ][0]
    ) == pytest.approx(1.0)
    assert float(
        logged["Train/ActionFlow/Compute/FieldForwardCallsPerStep"][0]
    ) == pytest.approx(1.0)
    assert float(
        logged["Train/ActionFlow/Compute/FieldSampleEquivalentsPerStep"][0]
    ) == pytest.approx(14.0)
    assert float(
        logged["Train/ActionFlow/Compute/DecoderJVPCallsPerStep"][0]
    ) == pytest.approx(1.0)

    manifest = wrapper._gradient_route_manifest
    assert manifest is not None
    assert tuple(manifest["routes"]) == ("FM", "Reconstruction", "ActionVelocity")
    assert manifest["intersections"]["FM__ActionVelocity"] == [
        "nets.pipeline.anchor"
    ]
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    assert manifest["manifest_sha256"] == hashlib.sha256(
        json.dumps(
            core,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    checkpoint = {}
    wrapper.on_save_checkpoint(checkpoint)
    assert checkpoint["action_flow_gradient_route_manifest"] == manifest


def test_action_flow_wrapper_exposes_generic_inference():
    wrapper = ActionFlowModelWrapper(pipeline=_ToyAlgo(), gradient_telemetry_cadence=0)
    condition = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    result = wrapper.forward_eval({"opaque_source": {"condition": condition}})

    assert tuple(result) == ("opaque_source",)
    torch.testing.assert_close(result["opaque_source"]["prediction"], condition)


def test_action_flow_wrapper_executes_real_pipeline_algo(monkeypatch):
    wrapper = ActionFlowModelWrapper(
        pipeline=PipelineAlgo(stages=[_ActionFlowHead()], device="cpu"),
        gradient_telemetry_cadence=0,
    )
    _capture_logs(monkeypatch, wrapper)

    loss = wrapper.training_step(
        {"opaque_source": {"target": torch.zeros(2, 2, 3)}}, batch_idx=0
    )
    assert float(loss) == pytest.approx(6.0)
    loss.backward()
    anchor = wrapper.nets["pipeline"].stages[0].anchor
    assert float(anchor.grad) == pytest.approx(12.0)

    result = wrapper.forward_eval({"opaque_source": {"condition": torch.zeros(2, 4)}})
    assert result["opaque_source"]["prediction"].shape == (2, 2, 3)


def test_action_flow_wrapper_uses_standard_optimizer_config(monkeypatch):
    monkeypatch.setattr(
        ActionFlowModelWrapper,
        "_instantiate_model",
        lambda _self, _config_tree: _ToyAlgo(),
    )
    wrapper = ActionFlowModelWrapper(
        config_tree={
            "model": {
                "pipeline": {},
                "optimizer": {
                    "_target_": "torch.optim.AdamW",
                    "lr": 3.0e-5,
                    "betas": [0.9, 0.999],
                    "eps": 1.0e-8,
                    "weight_decay": 1.0e-4,
                },
                "gradient_telemetry_cadence": 0,
            }
        },
        enable_grad_norm=False,
    )
    wrapper._trainer = SimpleNamespace(model=wrapper)

    configured = wrapper.configure_optimizers()

    assert (
        ActionFlowModelWrapper.configure_optimizers is ModelWrapper.configure_optimizers
    )
    optimizer = configured["optimizer"]
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3.0e-5)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(1.0e-4)


def test_action_flow_wrapper_logs_composite_optimizer_family_rates(monkeypatch):
    wrapper = ActionFlowModelWrapper(
        pipeline=_ToyAlgo(),
        gradient_telemetry_cadence=0,
    )
    wrapper._trainer = SimpleNamespace(
        optimizers=[
            SimpleNamespace(
                adamw=SimpleNamespace(param_groups=[{"lr": 2.5e-5}]),
                muon=SimpleNamespace(param_groups=[{"lr": 7.5e-5}]),
            )
        ]
    )
    logged = _capture_logs(monkeypatch, wrapper)

    wrapper._log_composite_optimizer_learning_rates()

    assert float(logged["Optimizer/LR/AdamW"][0]) == pytest.approx(2.5e-5)
    assert float(logged["Optimizer/LR/Muon"][0]) == pytest.approx(7.5e-5)
    for _, kwargs in logged.values():
        assert kwargs == {"on_step": True, "on_epoch": False, "sync_dist": False}


def test_action_flow_wrapper_has_no_specialized_route_or_model_imports():
    root = Path(__file__).parents[1]
    sources = (
        root / "egomimic/pl_utils/pl_model_action_flow.py",
        root / "egomimic/pl_utils/action_flow_diagnostic_forward.py",
    )
    for path in sources:
        source = path.read_text().lower()
        assert "unite" not in source
        for forbidden in ("usocket", "robot", "embodiment", "domain"):
            assert forbidden not in source


class _LifecycleEvaluator:
    action_flow_diagnostics_enabled = True

    def __init__(self):
        self.model = None
        self.started = 0
        self.ended = 0
        self.prediction = None

    def on_validation_start(self):
        self.started += 1

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        del batch_idx, dataloader_idx
        self.prediction = self.model.forward_eval(batch)

    def on_validation_end(self):
        self.ended += 1


def test_action_flow_validation_logs_components_and_processes_once(monkeypatch):
    algo = _ToyAlgo(reconstruction_weight=10.0)
    evaluator = _LifecycleEvaluator()
    wrapper = ActionFlowModelWrapper(
        pipeline=algo,
        evaluator=evaluator,
        gradient_telemetry_cadence=0,
    )
    logged = {}
    monkeypatch.setattr(
        wrapper,
        "log_dict",
        lambda values, **kwargs: logged.update(values),
    )

    wrapper.on_validation_start()
    wrapper.validation_step(_batch(), batch_idx=0)
    wrapper.on_validation_epoch_end()
    wrapper.on_validation_end()

    assert evaluator.model is wrapper
    assert evaluator.started == 1
    assert evaluator.ended == 1
    # validation_step moves the batch once; the bound wrapper recognizes the
    # active processed mapping when the evaluator requests ordinary inference.
    assert algo.process_count == 1
    assert tuple(evaluator.prediction) == ("source_a", "source_b")
    assert float(logged["Valid/ActionFlow/TotalLoss"]) == pytest.approx(60.0)
    assert float(logged["Valid/ActionFlow/FlowMatchingLoss"]) == pytest.approx(4.0)
    assert float(logged["Valid/ActionFlow/ReconstructionLoss"]) == pytest.approx(5.0)
    assert float(logged["Valid/ActionFlow/ReconstructionL1"]) == pytest.approx(7.0)
    assert float(logged["Valid/ActionFlow/ActionVelocityLoss"]) == pytest.approx(6.0)


def test_action_flow_validation_preserves_ordinary_evaluator_model(monkeypatch):
    algo = _ToyAlgo()
    evaluator = _LifecycleEvaluator()
    evaluator.action_flow_diagnostics_enabled = False
    evaluator.model = algo
    wrapper = ActionFlowModelWrapper(
        pipeline=algo,
        evaluator=evaluator,
        gradient_telemetry_cadence=0,
    )
    monkeypatch.setattr(wrapper, "log_dict", lambda *_args, **_kwargs: None)

    wrapper.on_validation_start()
    wrapper.validation_step(_batch(), batch_idx=0)

    assert evaluator.model is algo
    assert algo.process_count == 1
    assert tuple(evaluator.prediction) == ("source_a", "source_b")


class _TinySequenceModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])

    def forward(self, value):
        for block in self.blocks:
            value = torch.tanh(block(value))
        return value


class _TinyField(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])

    def forward(
        self,
        value,
        time,
        condition,
        *,
        condition_drop_mask=None,
    ):
        if condition_drop_mask is None:
            condition_drop_mask = torch.zeros(
                value.shape[0], dtype=torch.bool, device=value.device
            )
        effective = torch.where(
            condition_drop_mask.unsqueeze(-1),
            torch.zeros_like(condition),
            condition,
        )
        hidden = value + time[:, None, None] + effective[:, None, :2]
        for block in self.blocks:
            hidden = torch.tanh(block(hidden))
        return hidden


class _BatchSensitiveDecoder(_TinySequenceModule):
    """Expose accidental coalescing of diagnostic outer axes into the batch."""

    def forward(self, value):
        decoded = super().forward(value)
        return decoded + decoded.new_tensor(float(value.shape[0]) * 1.0e-3)


def _diagnostic_wrapper():
    torch.manual_seed(9)
    encoder = _TinySequenceModule()
    field = _TinyField()
    decoder = _TinySequenceModule()
    wrapper = ActionFlowModelWrapper(
        pipeline=PipelineAlgo(
            stages=[
                ContentEncoderStage(encoder=encoder),
                ConditionalVelocityStage(field=field, num_inference_steps=2),
                ContentDecoderStage(decoder=decoder),
            ],
            device="cpu",
        ),
        gradient_telemetry_cadence=0,
    )
    wrapper.eval()
    return wrapper, encoder, field, decoder


@torch.inference_mode()
def _inference_mode_diagnostics(wrapper, batch):
    return wrapper.forward_action_flow_diagnostics(
        batch,
        raw_noise_levels=[0.0, 0.5, 0.75, 1.0],
        noise_seed=1234,
        max_samples=None,
        jacobian_samples=1,
        capture_activations=True,
    )


def test_action_flow_diagnostics_are_deterministic_complete_and_ad_safe():
    wrapper, encoder, field, decoder = _diagnostic_wrapper()
    batch = {
        "opaque_source": {
            "target": torch.randn(3, 2, 2),
            "condition": torch.randn(3, 2),
        }
    }

    first = _inference_mode_diagnostics(wrapper, batch)["opaque_source"]
    second = _inference_mode_diagnostics(wrapper, batch)["opaque_source"]

    assert wrapper.encoder_e is encoder
    assert wrapper.field_v is field
    assert wrapper.decoder_g is decoder
    assert first["schema"] == "action-flow-validation-diagnostics/v1"
    assert first["returned_samples"] == 3
    assert first["jacobian_samples"] == 1
    assert first["num_inference_steps"] == 2
    torch.testing.assert_close(
        first["noise_levels"], torch.tensor([0.0, 0.5, 0.75, 1.0])
    )
    torch.testing.assert_close(
        first["fixed_level_field_evaluations"], torch.tensor([0, 1, 2, 2])
    )
    torch.testing.assert_close(first["latent/noise"], second["latent/noise"])
    torch.testing.assert_close(
        first["latent/fixed_final"], second["latent/fixed_final"]
    )
    torch.testing.assert_close(first["latent/fixed_final"][0], first["latent/clean"])

    expected_shapes = {
        "target": (3, 2, 2),
        "condition": (3, 2),
        "latent/clean": (3, 2, 2),
        "latent/noise": (3, 2, 2),
        "latent/generated": (3, 2, 2),
        "latent/fixed_states": (4, 3, 2, 2),
        "latent/predicted_clean": (4, 3, 2, 2),
        "latent/fixed_final": (4, 3, 2, 2),
        "latent/trajectory": (3, 3, 2, 2),
        "field/predicted_velocity": (4, 3, 2, 2),
        "field/velocity_residual": (4, 3, 2, 2),
        "decoded/reconstruction": (3, 2, 2),
        "decoded/noise": (3, 2, 2),
        "decoded/generated": (3, 2, 2),
        "decoded/fixed_states": (4, 3, 2, 2),
        "decoded/predicted_clean": (4, 3, 2, 2),
        "decoded/fixed_final": (4, 3, 2, 2),
        "decoded/trajectory": (3, 3, 2, 2),
        "activation/encoder_blocks": (2, 3, 2, 2),
        "activation/field_blocks": (4, 2, 3, 2, 2),
        "decoder_jacobian/clean_singular_values": (1, 4),
        "decoder_jacobian/noise_singular_values": (1, 4),
        "decoder_jacobian/fixed_singular_values": (4, 1, 4),
    }
    for key, shape in expected_shapes.items():
        assert first[key].shape == shape, key
        assert bool(torch.isfinite(first[key]).all()), key
        assert not first[key].requires_grad, key
    torch.testing.assert_close(
        first["activation/encoder_block_indices"], torch.tensor([0, 1])
    )
    torch.testing.assert_close(
        first["activation/field_block_indices"], torch.tensor([0, 1])
    )
    torch.testing.assert_close(
        first["decoder_jacobian/fixed_singular_values"][0],
        first["decoder_jacobian/clean_singular_values"],
    )
    torch.testing.assert_close(
        first["decoder_jacobian/fixed_singular_values"][-1],
        first["decoder_jacobian/noise_singular_values"],
    )


def test_action_flow_diagnostic_decoding_preserves_logical_batch_geometry():
    wrapper, _, _, _ = _diagnostic_wrapper()
    decoder = _BatchSensitiveDecoder()
    wrapper.model.pipeline.stages[2].decoder = decoder
    result = wrapper.forward_action_flow_diagnostics(
        {
            "opaque_source": {
                "target": torch.randn(3, 2, 2),
                "condition": torch.randn(3, 2),
            }
        },
        raw_noise_levels=[0.0, 0.5, 1.0],
        noise_seed=7,
        jacobian_samples=1,
        capture_activations=False,
    )["opaque_source"]

    torch.testing.assert_close(
        result["decoded/fixed_states"][0], result["decoded/reconstruction"]
    )
    torch.testing.assert_close(
        result["decoded/trajectory"][0], result["decoded/noise"]
    )


def test_action_flow_diagnostic_sample_caps_are_independent():
    wrapper, _, _, _ = _diagnostic_wrapper()
    result = wrapper.forward_action_flow_diagnostics(
        {
            "opaque_source": {
                "target": torch.randn(5, 2, 2),
                "condition": torch.randn(5, 2),
            }
        },
        raw_noise_levels=[0.25, 1.0],
        noise_seed=1,
        max_samples=4,
        jacobian_samples=2,
    )["opaque_source"]

    assert result["returned_samples"] == 4
    assert result["target"].shape[0] == 4
    assert result["activation/encoder_blocks"].shape[1] == 4
    assert result["activation/field_blocks"].shape[2] == 4
    assert result["decoder_jacobian/clean_singular_values"].shape[0] == 2
