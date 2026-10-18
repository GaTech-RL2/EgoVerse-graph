"""Offline readout expansion preserves phase A and loads through PipelineAlgo."""

import copy
import json
import pickle
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from egomimic.models.image_proprio import ImageProprioModel
from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pipeline.stages_image_proprio import ImageProprioStage
from scripts.world_model.initialize_pixel_readout import (
    initialize_readout,
    resolve_model_config,
    sha256_file,
    validate_transition,
    verify_phase_a,
)


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def configs(mode="continuous", pose_dim=9, action_dim=7):
    source = {
        "objective": "visual_only",
        "pose_dims": {"libero_gripper": pose_dim},
        "image_dim": 12,
        "width": 16,
        "heads": 2,
        "depth": 2,
        "mlp_ratio": 2,
        "max_cameras": 2,
        "max_tokens": 12,
        "time_mode": mode,
        "use_camera_embedding": True,
        "use_modality_embedding": False,
        "use_embodiment_embedding": False,
    }
    target = copy.deepcopy(source)
    target.update(
        objective="visual_state_action",
        action_dims={"libero_gripper": action_dim},
        max_tokens=14,
    )
    return source, target


def write_source(tmp_path, source_config):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        source = ImageProprioModel(**source_config)
    path = tmp_path / "stage_one.pt"
    torch.save(source.state_dict(), path)
    return path, source


def run_initialization(tmp_path, source_config=None, target_config=None, **overrides):
    if source_config is None:
        source_config, target_config = configs()
    path, source = write_source(tmp_path, source_config)
    arguments = {
        "source_checkpoint": path,
        "source_sha256": sha256_file(path),
        "source_config": source_config,
        "target_config": target_config,
        "output_path": tmp_path / "stage_two.pt",
        "receipt_path": tmp_path / "receipt.json",
        "seed": 42,
    }
    arguments.update(overrides)
    receipt = initialize_readout(**arguments)
    return source, receipt, arguments


@pytest.mark.parametrize(
    "mode,pose_dim,action_dim", [("continuous", 9, 7), ("discrete", 36, 28)]
)
def test_preserves_phase_a_and_strict_pipeline_load(
    tmp_path, mode, pose_dim, action_dim
):
    source_config, target_config = configs(mode, pose_dim, action_dim)
    rng_before = torch.random.get_rng_state().clone()
    source, receipt, arguments = run_initialization(
        tmp_path, source_config, target_config
    )
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert receipt["phase_a_preservation"]["bit_exact"]
    assert len(receipt["phase_a_preservation"]["checks"]) == 2
    assert not receipt["optimizer_state_transferred"]
    assert not receipt["training_step_transferred"]
    assert receipt == json.loads(arguments["receipt_path"].read_text())
    assert receipt["output"]["sha256"] == sha256_file(arguments["output_path"])
    state = torch.load(arguments["output_path"], weights_only=True)
    assert all(isinstance(value, torch.Tensor) for value in state.values())
    assert "camera.weight" in state
    assert "modality" not in state and "embodiment.weight" not in state
    target = ImageProprioModel(**target_config)
    pipeline = PipelineAlgo(
        [ImageProprioStage(target)],
        device="cpu",
        stage_ids={"image_proprio": 0},
        initialization=receipt["initialization"],
    )
    assert pipeline.initialization_receipts[0]["sha256"] == receipt["output"]["sha256"]
    for key, value in state.items():
        assert torch.equal(target.state_dict()[key], value)
    for name, value in source.state_dict().items():
        actual = target.state_dict()[name]
        if name in ("phase.weight", "proprio_role.weight"):
            actual = actual[:1]
        assert torch.equal(value, actual)
    assert verify_phase_a(source, target)["bit_exact"]


def test_new_parameters_and_rows_match_declared_seed_constructor(tmp_path):
    _, receipt, arguments = run_initialization(tmp_path)
    state = torch.load(arguments["output_path"], weights_only=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(arguments["seed"])
        fresh = ImageProprioModel(**receipt["target_model"]).state_dict()
    for name in receipt["transfer"]["new_tensors"]:
        assert torch.equal(state[name], fresh[name])
    for name in receipt["transfer"]["expanded_embeddings"]:
        assert torch.equal(state[name][1:], fresh[name][1:])
    # A second invocation creates identical weights, not an optimizer continuation.
    again = initialize_readout(
        **{
            **arguments,
            "output_path": tmp_path / "repeat.pt",
            "receipt_path": tmp_path / "repeat.json",
        }
    )
    repeated = torch.load(again["output"]["path"], weights_only=True)
    assert all(torch.equal(state[key], repeated[key]) for key in state)
    changed = initialize_readout(
        **{
            **arguments,
            "seed": 43,
            "output_path": tmp_path / "other_seed.pt",
            "receipt_path": tmp_path / "other_seed.json",
        }
    )
    changed_state = torch.load(changed["output"]["path"], weights_only=True)
    assert not torch.equal(state["action_query"], changed_state["action_query"])
    assert torch.equal(state["image_in.weight"], changed_state["image_in.weight"])


@pytest.mark.parametrize(
    "failure", ["hash", "missing", "unexpected", "shape", "dtype", "nan"]
)
def test_bad_source_never_creates_output(tmp_path, failure):
    source_config, target_config = configs()
    path, source = write_source(tmp_path, source_config)
    state = source.state_dict()
    if failure == "missing":
        state.pop("image_in.bias")
    elif failure == "unexpected":
        state["surprise"] = torch.zeros(1)
    elif failure == "shape":
        state["image_in.bias"] = torch.zeros(99)
    elif failure == "dtype":
        state["image_in.bias"] = state["image_in.bias"].double()
    elif failure == "nan":
        state["image_in.bias"][0] = float("nan")
    torch.save(state, path)
    output, receipt = tmp_path / "output.pt", tmp_path / "receipt.json"
    with pytest.raises(ValueError):
        initialize_readout(
            source_checkpoint=path,
            source_sha256="0" * 64 if failure == "hash" else sha256_file(path),
            source_config=source_config,
            target_config=target_config,
            output_path=output,
            receipt_path=receipt,
            seed=42,
        )
    assert not output.exists() and not receipt.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("time_mode", "discrete"),
        ("heads", 4),
        ("depth", 3),
        ("pose_dims", {"libero_gripper": 36}),
        ("use_camera_embedding", False),
        ("use_modality_embedding", True),
        ("use_embodiment_embedding", True),
        ("pose_conditioning", "clean"),
        ("max_tokens", 10),
    ],
)
def test_rejects_architecture_or_conditioning_drift(field, value):
    source, target = configs()
    target[field] = value
    with pytest.raises(ValueError):
        validate_transition(source, target)


def test_config_order_and_targets_cannot_be_inferred():
    source, target = configs()
    source["pose_dims"] = {"one": 9, "two": 9}
    target["pose_dims"] = {"two": 9, "one": 9}
    target["action_dims"] = {"two": 7, "one": 7}
    with pytest.raises(ValueError, match="order"):
        validate_transition(source, target)
    with pytest.raises(ValueError, match="Only ImageProprioModel"):
        resolve_model_config({**source, "_target_": "arbitrary.Code"})
    with pytest.raises(ValueError, match="resolved"):
        resolve_model_config({**source, "time_mode": "${data.mode}"})
    with pytest.raises(TypeError):
        resolve_model_config({**source, "unknown_parameter": True})


def test_exact_nested_namespace_and_checkpoint_config_binding(tmp_path):
    source_config, target_config = configs()
    path, source = write_source(tmp_path, source_config)
    prefix = "nets.pipeline.stages.0.model."
    payload = {
        "state_dict": {
            prefix + key: value for key, value in source.state_dict().items()
        },
        "model_config": source_config,
        "optimizer_states": [{"do_not_transfer": 123}],
        "global_step": 123,
    }
    payload["state_dict"]["another_stage.weight"] = torch.zeros(1)
    torch.save(payload, path)
    arguments = {
        "source_checkpoint": path,
        "source_sha256": sha256_file(path),
        "source_config": source_config,
        "target_config": target_config,
        "output_path": tmp_path / "output.pt",
        "receipt_path": tmp_path / "receipt.json",
        "seed": 42,
        "source_prefix": prefix,
        "state_dict_path": ["state_dict"],
    }
    with pytest.raises(ValueError, match="requires source_config_path"):
        initialize_readout(**arguments)
    arguments["source_config_path"] = ["model_config"]
    with pytest.raises(ValueError, match="namespace"):
        initialize_readout(**{**arguments, "source_prefix": "wrong."})
    receipt = initialize_readout(**arguments)
    assert receipt["source"]["config_binding"] == "checkpoint"
    state = torch.load(arguments["output_path"], weights_only=True)
    assert "optimizer_states" not in state and "global_step" not in state
    # Head count and physical/discrete time are not recoverable from weight shapes.
    payload["model_config"] = {**source_config, "heads": 4}
    torch.save(payload, path)
    with pytest.raises(ValueError, match="Checkpoint model configuration"):
        initialize_readout(
            **{
                **arguments,
                "source_sha256": sha256_file(path),
                "output_path": tmp_path / "bad.pt",
                "receipt_path": tmp_path / "bad.json",
            }
        )


def test_omegaconf_checkpoint_requires_explicit_trust_after_hash(tmp_path):
    source_config, target_config = configs()
    path, source = write_source(tmp_path, source_config)
    torch.save(
        {"state_dict": source.state_dict(), "config": OmegaConf.create(source_config)},
        path,
    )
    arguments = dict(
        source_checkpoint=path,
        source_sha256=sha256_file(path),
        source_config=source_config,
        target_config=target_config,
        output_path=tmp_path / "out.pt",
        receipt_path=tmp_path / "receipt.json",
        seed=42,
        state_dict_path=["state_dict"],
        source_config_path=["config"],
    )
    with pytest.raises(pickle.UnpicklingError):
        initialize_readout(**arguments)
    with pytest.raises(ValueError, match="SHA256"):
        initialize_readout(
            **{
                **arguments,
                "source_sha256": "0" * 64,
                "trusted_lightning_checkpoint": True,
            }
        )
    result = initialize_readout(**arguments, trusted_lightning_checkpoint=True)
    assert result["source"]["trusted_pickle"]


def test_refuses_overwrite_and_output_aliases(tmp_path):
    _, _, arguments = run_initialization(tmp_path)
    with pytest.raises(FileExistsError):
        initialize_readout(**arguments)
    with pytest.raises(ValueError, match="distinct"):
        initialize_readout(
            **{**arguments, "output_path": arguments["source_checkpoint"]}
        )


def test_phase_a_guard_catches_accidental_visual_weight_change(tmp_path):
    source_config, target_config = configs()
    source, receipt, arguments = run_initialization(
        tmp_path, source_config, target_config
    )
    target = ImageProprioModel(**target_config)
    target.load_state_dict(torch.load(arguments["output_path"], weights_only=True))
    with torch.no_grad():
        target.image_out.bias.add_(0.1)
    with pytest.raises(ValueError, match="Phase A preservation"):
        verify_phase_a(source, target)


def test_standalone_cli_from_outside_checkout(tmp_path):
    source_config, target_config = configs()
    path, _ = write_source(tmp_path, source_config)
    source_json, target_json = tmp_path / "source.json", tmp_path / "target.json"
    source_json.write_text(json.dumps(source_config))
    target_json.write_text(json.dumps(target_config))
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/world_model/initialize_pixel_readout.py"
    )
    output, receipt = tmp_path / "output.pt", tmp_path / "receipt.json"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--source-checkpoint",
            str(path),
            "--source-sha256",
            sha256_file(path),
            "--source-config",
            str(source_json),
            "--target-config",
            str(target_json),
            "--output",
            str(output),
            "--receipt",
            str(receipt),
            "--seed",
            "42",
        ],
        cwd=tmp_path,
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(result.stdout)["status"] == "PASS"
    assert json.loads(receipt.read_text())["output"]["sha256"] == sha256_file(output)
