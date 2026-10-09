"""Actual native checkpoint-hook format and external normalization identities."""

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

R = Path(__file__).parents[1]
S = R / "scripts/train/verify_libero_native_action_flow_smoke.py"
spec = importlib.util.spec_from_file_location("native_verifier", S)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


def native_format():
    # Execute the real native save hook without constructing a model/checkpoint.
    tree = ast.parse(
        (R / "egomimic/pl_utils/training_behavior_action_flow.py").read_text()
    )
    owner = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ActionFlowTrainingBehavior"
    )
    hook = next(
        n
        for n in owner.body
        if isinstance(n, ast.FunctionDef) and n.name == "on_save_checkpoint"
    )
    namespace = {"Any": object}
    exec(
        compile(
            ast.Module(body=[hook], type_ignores=[]), "actual-native-save-hook", "exec"
        ),
        namespace,
    )
    context = SimpleNamespace(
        reconstruction_only_warmup_steps=0,
        _objective_weight=lambda key, default=None: 1.0,
        _gradient_route_manifest={"actual_hook_marker": True},
    )
    payload = {}
    namespace["on_save_checkpoint"](context, payload)
    model = json.loads(
        (R / "assets/libero/historical_af27m_scientific_contract_v1.json").read_text()
    )["historical_model"]
    binding = {
        "source_commit": "a" * 40,
        "file_sha256": "b" * 64,
        "split_receipt_sha256": "c" * 64,
    }
    provenance = {
        "source_commit": binding["source_commit"],
        "normalization_sha256": binding["file_sha256"],
        "split_manifest_sha256": binding["split_receipt_sha256"],
        "dataset_sha256": "d" * 64,
    }
    cfg = {"model": model, "run_provenance": provenance}
    train_tree = ast.parse((R / "egomimic/trainHydra.py").read_text())
    builder = next(
        n
        for n in train_tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_build_model_config_tree"
    )
    builder.returns = None
    for argument in builder.args.args:
        argument.annotation = None
    namespace = {"OmegaConf": OmegaConf}
    exec(
        compile(
            ast.Module(body=[builder], type_ignores=[]),
            "actual-native-hparams-builder",
            "exec",
        ),
        namespace,
    )
    actual_cfg = OmegaConf.create(
        {**cfg, "norm_stats": {"native_saved_state_binding": binding}}
    )
    recorded = OmegaConf.to_container(
        namespace["_build_model_config_tree"](actual_cfg), resolve=True
    )
    assert set(recorded) == {"model", "run_provenance"}
    payload["hyper_parameters"] = {"config_tree": recorded}
    return payload, cfg, binding


def test_real_native_save_hook_has_no_embedded_normalizer_requirement():
    payload, cfg, binding = native_format()
    assert "normalizer_state" not in payload
    assert (
        "action_flow_loss_schedule" in payload
        and "action_flow_gradient_route_manifest" in payload
    )
    assert (
        native.validate_native_checkpoint_context(payload, cfg, binding)
        == cfg["run_provenance"]
    )


@pytest.mark.parametrize(
    "key",
    [
        "source_commit",
        "normalization_sha256",
        "split_manifest_sha256",
        "dataset_sha256",
    ],
)
def test_recorded_native_checkpoint_identity_drift_rejected(key):
    payload, cfg, binding = native_format()
    payload["hyper_parameters"]["config_tree"]["run_provenance"][key] = "bad"
    with pytest.raises(ValueError):
        native.validate_native_checkpoint_context(payload, cfg, binding)


def test_native_checkpoint_model_drift_rejected():
    payload, cfg, binding = native_format()
    payload["hyper_parameters"]["config_tree"]["model"]["hidden_dim"] = 384
    with pytest.raises(ValueError):
        native.validate_native_checkpoint_context(payload, cfg, binding)


def test_native_verifier_uses_contextual_sidecar_and_actual_dataset_boundary():
    source = S.read_text()
    tree = ast.parse(source)
    method = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "verify_native_smoke"
    )
    calls = [ast.unparse(n.func) for n in ast.walk(method) if isinstance(n, ast.Call)]
    for name in [
        "instantiate_metadata_dataset",
        "bind_saved_native_state",
        "normalizer.to_state",
        "dataset.set_norm_stats_from",
        "restored.model.bind_data_context",
    ]:
        assert name in calls
    assert 'payload.get("normalizer_state")' not in source
    compares = [
        n
        for n in ast.walk(method)
        if isinstance(n, ast.Call)
        and ast.unparse(n.func) == "torch.testing.assert_close"
    ]
    assert len(compares) == 1
    assert [ast.unparse(x) for x in compares[0].args] == [
        "parameter.detach().cpu()",
        "ema[key].detach().cpu()",
    ]
    assert {k.arg: ast.literal_eval(k.value) for k in compares[0].keywords} == {
        "rtol": 0,
        "atol": 0,
    }
