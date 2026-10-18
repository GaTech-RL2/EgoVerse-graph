"""Readout providers and optional image losses share one stable model owner."""

import copy

import pytest
import torch

from egomimic.models.pixel_diffusion import (
    LinearDDIMImageSampler,
    LinearGaussianImageNoising,
    PlainPixelMSE,
)
from egomimic.pipeline.core import Pipeline, sum_losses
from egomimic.pipeline.stages_pixel_readout import (
    PixelReadoutImageDenoiserStage,
    PixelReadoutImageLossStage,
    PixelReadoutModelStage,
    PixelReadoutMSEStage,
    PixelReadoutNoisingStage,
    PixelReadoutSampledImageStage,
    PixelReadoutVideoDenoiserStage,
    PixelStateActionReadoutStage,
)
from egomimic.pipeline.stages_pixel_video import PixelVideoNoisingStage
from tests.test_pixel_noise_conditioning_profiles import _trained_image_source
from tests.test_pixel_readout_graph import arguments, fixture_batch, stages

MODEL_IDS = {"pixel_denoiser": 1, "image_proprio": 1}
IMAGE_PORT = "pixel_readout/image_prediction"


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model(mode="continuous", spatial=False):
    kwargs = {
        **arguments(mode),
        "noise_conditioning": "adaln_zero",
        "use_phase_embedding": False,
    }
    if spatial:
        kwargs.update(
            position_encoding="spatiotemporal_rope",
            rope_axis_dims=[4, 2, 2],
            spatial_grid_size=[2, 2],
        )
    return _trained_image_source(kwargs)


def modular_stages(network, training_mode="frozen_image"):
    return [
        PixelVideoNoisingStage(LinearGaussianImageNoising()),
        PixelReadoutModelStage(network, training_mode),
        PixelReadoutVideoDenoiserStage(),
        PixelReadoutImageLossStage(PlainPixelMSE(network.loss_reduction)),
        PixelReadoutNoisingStage(LinearGaussianImageNoising(0.2)),
        PixelReadoutImageDenoiserStage(prediction_key=IMAGE_PORT),
        PixelReadoutSampledImageStage(LinearDDIMImageSampler(3)),
        PixelStateActionReadoutStage(image_key=IMAGE_PORT),
        PixelReadoutMSEStage(PlainPixelMSE(network.loss_reduction), "proprio"),
        PixelReadoutMSEStage(PlainPixelMSE(network.loss_reduction), "action"),
    ]


@pytest.mark.parametrize("mode", ["continuous", "discrete"])
@pytest.mark.parametrize("policy", ["frozen_image", "joint"])
@pytest.mark.parametrize("spatial", [False, True])
def test_default_modular_graph_preserves_outputs_gradients_rng_and_checkpoints(
    mode, policy, spatial
):
    network = model(mode, spatial)
    legacy = Pipeline(stages(copy.deepcopy(network), policy), stage_ids=MODEL_IDS)
    graph = Pipeline(modular_stages(network, policy), stage_ids=MODEL_IDS)
    assert set(graph.state_dict()) == set(legacy.state_dict())
    assert all(key.startswith("stages.1.model.") for key in graph.state_dict())
    graph.load_state_dict(legacy.state_dict(), strict=True)
    assert sum(p.numel() for p in graph.parameters()) == sum(
        p.numel() for p in network.parameters()
    )
    batch = fixture_batch()
    torch.manual_seed(171)
    expected = legacy.execute(batch, mode="train")
    expected_rng = torch.get_rng_state().clone()
    sum_losses(expected).backward()
    torch.manual_seed(171)
    actual = graph.execute(batch, mode="train")
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for key in (
        "loss/pixel_video",
        "loss/measured_state",
        "loss/control_command",
        IMAGE_PORT,
    ):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    sum_losses(actual).backward()
    for name, parameter in graph.named_parameters():
        reference = dict(legacy.named_parameters())[name]
        if reference.grad is None:
            assert parameter.grad is None, name
        else:
            torch.testing.assert_close(parameter.grad, reference.grad, rtol=0, atol=0)
    torch.manual_seed(31)
    expected = legacy.execute(batch, mode="inference")["image_proprio_prediction"]
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(31)
    actual = graph.execute(batch, mode="inference")["image_proprio_prediction"]
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for key in ("image", "proprio", "action"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_high_noise_branch_can_be_removed_without_losing_owner_or_readout_gradients():
    network = model(spatial=True)
    full = modular_stages(network)
    # Keep the owner at index1 while dropping all three high-noise branch nodes.
    selected = [full[4], full[1], full[5], full[6], full[7], full[8], full[9]]
    graph = Pipeline(selected, stage_ids=MODEL_IDS)
    phases = []
    hook = network.register_forward_pre_hook(
        lambda module, args, kwargs: phases.append(kwargs["phase"]), with_kwargs=True
    )
    before = {key: value.clone() for key, value in network.state_dict().items()}
    optimizer = torch.optim.AdamW(graph.parameters(), lr=5e-5)
    result = graph.execute(fixture_batch(), mode="train")
    hook.remove()
    assert phases == ["A", "B"]
    assert "loss/pixel_video" not in result and "pixel_video/noisy_image" not in result
    assert graph.stage_by_id("image_proprio").model is network
    assert all(key.startswith("stages.1.model.") for key in graph.state_dict())
    sum_losses(result).backward()
    for name, parameter in network.named_parameters():
        if parameter.requires_grad:
            assert (
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
            ), name
        else:
            assert parameter.grad is None, name
    optimizer.step()
    for name in full[1].trainability["frozen_parameters"]:
        assert torch.equal(network.state_dict()[name], before[name]), name
    assert not torch.equal(
        network.pose_out["robot"].weight, before["pose_out.robot.weight"]
    )
    assert not torch.equal(
        network.action_out["robot"].weight, before["action_out.robot.weight"]
    )


def test_sampling_provider_swaps_training_images_with_same_weights_and_tensor_port():
    network = model(spatial=True)
    baseline = Pipeline(modular_stages(copy.deepcopy(network)), stage_ids=MODEL_IDS)
    selected = modular_stages(network)
    selected[5] = PixelReadoutSampledImageStage(
        LinearDDIMImageSampler(3),
        modes=["train"],
        prediction_key=IMAGE_PORT,
        gradient_policy="stop",
    )
    del selected[4]  # Sampling does not consume the low-noise corruption branch.
    graph = Pipeline(selected, stage_ids=MODEL_IDS)
    before = {key: value.clone() for key, value in graph.state_dict().items()}
    batch = fixture_batch()
    torch.manual_seed(41)
    expected = baseline.execute(batch, mode="train")
    torch.manual_seed(41)
    actual = graph.execute(batch, mode="train")
    unknown = batch["world"]["image"]["valid"] & ~batch["world"]["image"]["observed"]
    assert not torch.equal(actual[IMAGE_PORT][unknown], expected[IMAGE_PORT][unknown])
    assert not actual[IMAGE_PORT].requires_grad
    assert "pixel_readout/noisy_image" not in actual
    reference = network(
        batch["world"],
        "robot",
        actual[IMAGE_PORT],
        torch.zeros_like(batch["world"]["image"]["time"]),
        phase="B",
    )
    for stream in ("proprio", "action"):
        torch.testing.assert_close(
            actual["pixel_readout/prediction"][stream],
            reference[stream],
            rtol=0,
            atol=0,
        )
    assert all(
        torch.equal(value, before[key]) for key, value in graph.state_dict().items()
    )
    sum_losses(actual).backward()
    assert network.action_out["robot"].weight.grad is not None


def test_sampled_provider_declares_stopped_gradients_even_for_joint_model():
    network = model()
    PixelReadoutModelStage(network, training_mode="joint")
    provider = PixelReadoutSampledImageStage(
        LinearDDIMImageSampler(3), modes=["train", "inference"], gradient_policy="stop"
    )
    batch = {**fixture_batch(), "pixel_video/denoiser": network}
    result = provider.execute(batch, mode="train")
    assert not result[IMAGE_PORT].requires_grad
    assert all(parameter.grad is None for parameter in network.parameters())
    assert not provider.train_only and not provider.inference_only
    with pytest.raises(ValueError, match="gradient_policy=stop"):
        PixelReadoutSampledImageStage(
            LinearDDIMImageSampler(3), gradient_policy="through_sampler"
        )
    with pytest.raises(ValueError, match="modes"):
        PixelReadoutSampledImageStage(LinearDDIMImageSampler(3), modes="train")
    owner = PixelReadoutModelStage(network)
    assert owner.contract("train") == ((), ("pixel_video/denoiser",))
    assert owner.contract("inference") == owner.contract("train")
    readout = PixelStateActionReadoutStage(image_key=IMAGE_PORT)
    assert readout.contract("train")[0] == readout.contract("inference")[0]
