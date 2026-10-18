"""Pilot AdaLN selection and strict preservation when adding frozen readouts."""

import copy

import pytest
import torch

from egomimic.models.image_proprio import ImageProprioModel
from egomimic.models.pixel_diffusion import LinearDDIMImageSampler
from egomimic.pipeline.core import Pipeline, sum_losses
from egomimic.pipeline.stages_pixel_readout import configure_readout_trainability
from scripts.world_model.initialize_pixel_readout import (
    initialize_readout,
    resolve_model_config,
    sha256_file,
    transfer_shared_weights,
    validate_transition,
    verify_phase_a,
)
from tests.test_pixel_readout_graph import IDS, fixture_batch, stages
from tests.test_pixel_token_execution_profiles import configuration, denoiser


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("stage", ["video", "readout"])
@pytest.mark.parametrize("arm", ["continuous", "discrete"])
@pytest.mark.parametrize("profile", [None, "packed_regional_compile"])
def test_current_pilots_select_adaln_without_other_identity_conditioning(
    stage, arm, profile
):
    model = denoiser(configuration(stage, arm, profile))
    assert model["noise_conditioning"] == "adaln_zero"
    assert model["use_phase_embedding"] is False
    assert model["use_modality_embedding"] is False
    assert model["use_embodiment_embedding"] is False
    assert model["use_camera_embedding"] is True
    assert model["time_mode"] == arm
    assert model["position_encoding"] == "spatiotemporal_rope"
    assert model["rope_axis_dims"] == [64, 32, 32]
    assert model["spatial_grid_size"] == [10, 10]


@pytest.mark.parametrize("arm", ["continuous", "discrete"])
def test_readout_requires_exact_noise_conditioning_and_legacy_default(arm):
    source = denoiser(configuration("video", arm, "packed_regional_compile"))
    target = denoiser(configuration("readout", arm, "packed_regional_compile"))
    left, right = validate_transition(source, target)
    assert left["noise_conditioning"] == right["noise_conditioning"] == "adaln_zero"
    del source["noise_conditioning"]
    assert resolve_model_config(source)["noise_conditioning"] == "additive"
    with pytest.raises(ValueError, match="configuration mismatch.*noise_conditioning"):
        validate_transition(source, target)
    del target["noise_conditioning"]
    left, right = validate_transition(source, target)
    assert left["noise_conditioning"] == right["noise_conditioning"] == "additive"


def model_arguments(mode="continuous"):
    return dict(
        objective="visual_only",
        phase_a_condition_on_proprio=False,
        pose_dims=None,
        action_dims=None,
        image_dim=12,
        width=16,
        depth=2,
        heads=2,
        mlp_ratio=2,
        max_cameras=2,
        max_tokens=16,
        time_mode=mode,
        use_phase_embedding=False,
        noise_conditioning="adaln_zero",
        loss_reduction="mean_per_example",
    )


def readout_arguments(source):
    return {
        **source,
        "objective": "visual_state_action",
        "pose_dims": {"robot": 3},
        "action_dims": {"robot": 2},
    }


def _trained_image_source(arguments):
    """Nonzero gates/modulations/head prevent a vacuous zero-output proof."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(349)
        model = ImageProprioModel(**arguments)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if any(part in name for part in ("noise_level", "adaln", "image_out")):
                    parameter.normal_(mean=0, std=0.05)
    return model


@pytest.mark.parametrize("mode", ["continuous", "discrete"])
@pytest.mark.parametrize("spatial_rope", [False, True])
def test_adaln_checkpoint_expansion_freezes_all_image_parameters(
    tmp_path, mode, spatial_rope
):
    source_arguments = model_arguments(mode)
    if spatial_rope:
        source_arguments.update(
            position_encoding="spatiotemporal_rope",
            rope_axis_dims=[4, 2, 2],
            spatial_grid_size=[2, 2],
        )
    target_arguments = readout_arguments(source_arguments)
    source = _trained_image_source(source_arguments)
    source_path = tmp_path / "source.pt"
    target_path = tmp_path / "readout.pt"
    torch.save(source.state_dict(), source_path)
    receipt = initialize_readout(
        source_checkpoint=source_path,
        source_sha256=sha256_file(source_path),
        source_config=source_arguments,
        target_config=target_arguments,
        output_path=target_path,
        receipt_path=tmp_path / "receipt.json",
        seed=42,
    )
    assert receipt["phase_a_preservation"]["bit_exact"]
    assert not receipt["optimizer_state_transferred"]
    assert not receipt["training_step_transferred"]
    assert receipt["output"]["sha256"] == sha256_file(target_path)
    assert receipt["source_model"]["noise_conditioning"] == "adaln_zero"
    assert receipt["target_model"]["noise_conditioning"] == "adaln_zero"
    conditioning = {
        name for name in source.state_dict() if "adaln" in name or "noise_level" in name
    }
    assert any(name.startswith("final_adaln.") for name in conditioning)
    assert any(".adaln_image." in name for name in conditioning)
    assert conditioning <= set(receipt["transfer"]["copied_tensors"])
    target = ImageProprioModel(**target_arguments)
    target.load_state_dict(torch.load(target_path, weights_only=True), strict=True)
    graph = Pipeline(stages(target), stage_ids=IDS)
    policy = graph.stage_by_id("pixel_denoiser").trainability
    assert policy["noise_conditioning"] == "adaln_zero"
    assert not policy["phase_embedding_present"]
    assert hasattr(target, "xy") is not spatial_rope
    assert conditioning <= set(policy["frozen_parameters"])
    assert set(source.state_dict()) <= set(policy["frozen_parameters"])
    before = {
        name: parameter.detach().clone()
        for name, parameter in target.named_parameters()
    }
    batch = fixture_batch()
    sampler = LinearDDIMImageSampler(3)
    torch.manual_seed(78)
    expected_images = sampler(source, {"image": batch["world"]["image"]}, "robot")
    # Check unknown pixels too: observed clean values would be nonzero even if
    # the image head were still at its constructor's zero initialization.
    unknown = batch["world"]["image"]["valid"] & ~batch["world"]["image"]["observed"]
    assert expected_images[unknown].abs().sum() > 0
    optimizer = torch.optim.AdamW(graph.parameters(), lr=5e-5, weight_decay=0.01)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        output = graph.execute(batch, mode="train")
        assert not output["loss/pixel_video"].requires_grad
        loss = sum_losses(output)
        assert loss.requires_grad and torch.isfinite(loss)
        loss.backward()
        for name, parameter in target.named_parameters():
            if parameter.requires_grad:
                assert (
                    parameter.grad is not None and torch.isfinite(parameter.grad).all()
                ), name
            else:
                assert parameter.grad is None, name
        optimizer.step()
    for name in policy["frozen_parameters"]:
        assert torch.equal(target.state_dict()[name], before[name]), name
    for name in ("pose_out.robot.weight", "action_out.robot.weight"):
        assert not torch.equal(target.state_dict()[name], before[name]), name
    assert verify_phase_a(source, target)["bit_exact"]
    torch.manual_seed(78)
    actual_images = sampler(target, batch["world"], "robot")
    torch.testing.assert_close(actual_images, expected_images, rtol=0, atol=0)
    restored = Pipeline(stages(ImageProprioModel(**target_arguments)), stage_ids=IDS)
    restored.load_state_dict(graph.state_dict(), strict=True)
    assert restored.stage_by_id("pixel_denoiser").trainability == policy


def test_direct_transfer_rejects_noise_mode_drift_and_joint_mode_unfreezes_adaln():
    arguments = model_arguments()
    source = ImageProprioModel(**arguments)
    target_arguments = readout_arguments(arguments)
    additive_target = ImageProprioModel(
        **{**target_arguments, "noise_conditioning": "additive"}
    )
    with pytest.raises(ValueError, match="noise_conditioning must match"):
        transfer_shared_weights(source, additive_target)
    target = ImageProprioModel(**target_arguments)
    transfer_shared_weights(source, target)
    policy = configure_readout_trainability(target, "frozen_image")
    assert policy["frozen_parameters"]
    policy = configure_readout_trainability(target, "joint")
    assert not policy["frozen_parameters"]
    assert all(parameter.requires_grad for parameter in target.parameters())
    # A fresh strict load retains its explicitly selected noise architecture.
    restored = ImageProprioModel(**copy.deepcopy(target_arguments))
    restored.load_state_dict(target.state_dict(), strict=True)


def test_padded_stage_two_sampler_caches_frame_groups_once(monkeypatch):
    model = _trained_image_source(readout_arguments(model_arguments()))
    original = model._prepare_image_execution
    calls = []

    def prepare(image, valid):
        calls.append(valid.shape)
        return original(image, valid)

    monkeypatch.setattr(model, "_prepare_image_execution", prepare)
    image = LinearDDIMImageSampler(3)(model, fixture_batch()["world"], "robot")
    assert torch.isfinite(image).all()
    assert calls == [torch.Size([2, 4])]


@pytest.mark.parametrize("arm", ["continuous", "discrete"])
def test_readout_initialization_binds_position_encoding_and_geometry(arm):
    source = denoiser(configuration("video", arm, "packed_regional_compile"))
    target = denoiser(configuration("readout", arm, "packed_regional_compile"))
    validate_transition(source, target)
    for field, value in (
        ("position_encoding", "linear_xy_temporal_rope"),
        ("rope_axis_dims", [32, 48, 48]),
        ("spatial_grid_size", [20, 20]),
    ):
        changed = {**target, field: value}
        with pytest.raises(ValueError, match=f"configuration mismatch.*{field}"):
            validate_transition(source, changed)
    for cfg in (source, target):
        for field in ("position_encoding", "rope_axis_dims", "spatial_grid_size"):
            cfg.pop(field)
    left, right = validate_transition(source, target)
    assert (
        left["position_encoding"]
        == right["position_encoding"]
        == "linear_xy_temporal_rope"
    )
    assert left["rope_axis_dims"] is right["rope_axis_dims"] is None
    assert left["spatial_grid_size"] is right["spatial_grid_size"] is None


@pytest.mark.parametrize(
    "field,value", [("rope_axis_dims", [2, 2, 4]), ("spatial_grid_size", [4, 4])]
)
def test_direct_tensor_transfer_rejects_same_shape_positional_drift(field, value):
    source_args = {
        **model_arguments(),
        "position_encoding": "spatiotemporal_rope",
        "rope_axis_dims": [4, 2, 2],
        "spatial_grid_size": [2, 2],
    }
    source = ImageProprioModel(**source_args)
    target = ImageProprioModel(**{**readout_arguments(source_args), field: value})
    with pytest.raises(ValueError, match=f"{field} must match"):
        transfer_shared_weights(source, target)
