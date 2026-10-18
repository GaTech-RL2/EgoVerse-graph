"""Independent pre-extraction arithmetic/RNG references for graph components."""

import copy

import pytest
import torch

from egomimic.models.pixel_diffusion import (
    LinearDDIMImageSampler,
    LinearGaussianImageNoising,
    PlainPixelMSE,
)
from tests.test_image_proprio_image_only import joint_batch, joint_model
from tests.test_pixel_execution_model import batch, network


def previous_noising(image, clean, maximum=1.0):
    # Frozen arithmetic from the pre-extraction model; deliberately independent
    # of all new components so draw-order changes cannot validate themselves.
    levels = torch.zeros_like(image["time"], dtype=torch.float32)
    for row in range(levels.shape[0]):
        selected = image["valid"][row] & ~image["observed"][row]
        if selected.any():
            times, inverse = torch.unique(
                image["time"][row][selected], return_inverse=True
            )
            draws = torch.rand(times.numel(), device=levels.device) * maximum
            levels[row, selected] = draws[inverse]
    strength = levels[..., None].to(clean.dtype)
    noisy = (1 - strength) * clean + strength * torch.randn_like(clean)
    return noisy.masked_fill(~image["valid"][..., None], 0), levels


def previous_loss(prediction, target, mask, reduction):
    weight = mask.float()
    target = torch.where((weight > 0)[..., None], target, torch.zeros_like(target))
    error = (prediction.float() - target.float()).square().mean(-1)
    if reduction == "mean_per_example":
        numerator = (error * weight).flatten(1).sum(1)
        denominator = weight.flatten(1).sum(1)
        return (numerator / torch.where(denominator > 0, denominator, 1.0)).mean()
    denominator = weight.sum()
    return (error * weight).sum() / torch.where(denominator > 0, denominator, 1.0)


@torch.no_grad()
def previous_sample(model, data, steps, generator):
    model.validate(data, "gripper")
    image = data["image"]
    valid, observed = image["valid"], image["observed"]
    anchor = torch.where(observed[..., None], image["value"], 0)
    noise = torch.randn(
        image["value"].shape,
        device=anchor.device,
        dtype=anchor.dtype,
        generator=generator,
    )
    current = torch.where(observed[..., None], anchor, noise).masked_fill(
        ~valid[..., None], 0
    )
    execution = None
    if not model.has_proprio_branch:
        execution = model._prepare_execution([model._clock(image, valid)], [valid])
    for j in range(steps, 0, -1):
        level, next_level = j / steps, (j - 1) / steps
        levels = torch.full_like(image["time"], level, dtype=torch.float32).masked_fill(
            observed | ~valid, 0
        )
        x0 = model(
            data,
            "gripper",
            current,
            levels,
            phase="A",
            _validated=True,
            _execution=execution,
        )["image"].to(current.dtype)
        candidate = (1 - next_level) * x0 + (next_level / level) * (
            current - (1 - level) * x0
        )
        current = torch.where(observed[..., None], anchor, candidate).masked_fill(
            ~valid[..., None], 0
        )
    return current


@pytest.mark.parametrize("maximum", [0.0, 0.2, 1.0])
@pytest.mark.parametrize("all_observed", [False, True])
def test_noising_exact_legacy_draw_order_observed_padding_and_metadata(
    maximum, all_observed
):
    image = batch()["image"]
    image["value"][~image["valid"]] = float("nan")
    if all_observed:
        image["observed"] = image["valid"].clone()
    clean = torch.where(image["valid"][..., None], image["value"], 0)
    torch.manual_seed(91)
    expected, expected_levels = previous_noising(image, clean, maximum)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(91)
    actual = LinearGaussianImageNoising(maximum)(image)
    assert torch.equal(actual["image_values"], expected)
    assert torch.equal(actual["noise_levels"], expected_levels)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert torch.equal(actual["target"], clean)
    assert torch.equal(
        actual["image_values"][image["observed"]], image["value"][image["observed"]]
    )
    assert not actual["image_values"][~image["valid"]].any()
    assert actual["metadata"]["timestamps"] is image["time"]
    assert torch.equal(actual["mask"], image["valid"] & ~image["observed"])
    assert actual["noise_levels"].dtype == torch.float32


def test_noising_rejects_missing_or_nonfinite_recorded_query_targets():
    image = batch()["image"]
    image["target_valid"] = image["valid"].clone()
    image["target_valid"][0, 2] = False
    with pytest.raises(ValueError, match="recorded features"):
        LinearGaussianImageNoising()(image)
    image["target_valid"][0, 2] = True
    image["value"][0, 2] = float("nan")
    with pytest.raises(ValueError, match="Non-finite image target"):
        LinearGaussianImageNoising()(image)


@pytest.mark.parametrize("reduction", ["token_mean", "mean_per_example"])
def test_loss_preserves_fp32_arithmetic_gradients_and_empty_clips(reduction):
    target = torch.randn(2, 4, 3)
    mask = torch.tensor([[True, False, True, True], [False] * 4])
    target[~mask] = float("nan")
    predicted = torch.randn_like(target).bfloat16().requires_grad_()
    expected = previous_loss(predicted, target, mask, reduction)
    # Noise/time metadata must not accidentally turn plain MSE into a weighting.
    actual = PlainPixelMSE(reduction)(
        predicted,
        target,
        mask,
        noise_levels=torch.full((2, 4), 0.9),
        metadata={"timestamps": torch.arange(4)},
    )
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)
    expected_grad = torch.autograd.grad(expected, predicted, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(actual, predicted)[0]
    assert torch.equal(actual_grad, expected_grad)
    assert not actual_grad[1].any()


@pytest.mark.parametrize("reduction", ["token_mean", "mean_per_example"])
def test_separate_components_equal_legacy_training_loss_gradients_and_rng(reduction):
    torch.manual_seed(7)
    legacy = network()
    legacy.loss_reduction = reduction
    separate = copy.deepcopy(legacy)
    data = batch()
    image = data["image"]
    torch.manual_seed(111)
    clean = torch.where(image["valid"][..., None], image["value"], 0)
    noisy, levels = previous_noising(image, clean)
    expected_prediction = legacy(data, "gripper", noisy, levels)["image"]
    expected = previous_loss(
        expected_prediction, clean, image["valid"] & ~image["observed"], reduction
    )
    expected_rng = torch.get_rng_state().clone()
    expected.backward()
    torch.manual_seed(111)
    corruption = LinearGaussianImageNoising()(image)
    actual_prediction = separate(
        data, "gripper", corruption["image_values"], corruption["noise_levels"]
    )["image"]
    actual = PlainPixelMSE(reduction)(
        actual_prediction,
        corruption["target"],
        corruption["mask"],
        corruption["noise_levels"],
        corruption["metadata"],
    )
    assert torch.equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    actual.backward()
    for a, b in zip(legacy.parameters(), separate.parameters()):
        assert torch.equal(a.grad, b.grad)
    torch.manual_seed(111)
    wrapped = separate.training_losses(data, "gripper")["total"]
    assert torch.equal(wrapped, expected)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert legacy.state_dict().keys() == separate.state_dict().keys()


@pytest.mark.parametrize("steps", [1, 3, 20])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("explicit_generator", [False, True])
def test_sampler_component_is_bit_exact_to_previous_integration(
    steps, mixed, explicit_generator
):
    model = network(attention_dtype="bfloat16" if mixed else "native")
    data = batch()
    image = data["image"]
    image["value"][~image["observed"]] = float("nan")

    def generator():
        return torch.Generator().manual_seed(101) if explicit_generator else None

    torch.manual_seed(83)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=mixed):
        expected = previous_sample(model, data, steps, generator())
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(83)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=mixed):
        actual = LinearDDIMImageSampler(steps)(
            model, data, "gripper", generator=generator()
        )
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    torch.manual_seed(83)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=mixed):
        wrapped = model.sample(data, "gripper", steps=steps, generator=generator())[
            "image"
        ]
    assert torch.equal(wrapped, expected)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert torch.isfinite(actual).all()
    assert torch.equal(actual[image["observed"]], image["value"][image["observed"]])


def test_stage_two_wrapper_keeps_phase_b_readout_and_observed_states():
    model, data = joint_model(), joint_batch()
    current = previous_sample(model, data, 3, torch.Generator().manual_seed(23))
    with torch.no_grad():
        expected = model(
            data, "gripper", current, torch.zeros_like(data["image"]["time"]), phase="B"
        )
    actual = model.sample(
        data, "gripper", steps=3, generator=torch.Generator().manual_seed(23)
    )
    assert torch.equal(actual["image"], current)
    assert torch.equal(actual["action"], expected["action"])
    pose = data["proprio"]
    expected_pose = torch.where(
        pose["observed"][..., None], pose["value"], expected["proprio"]
    )
    assert torch.equal(
        actual["proprio"], expected_pose.masked_fill(~pose["valid"][..., None], 0)
    )


def test_sampler_prepares_once_and_components_register_no_parameters(monkeypatch):
    model, data = network(), batch()
    before = tuple(model.state_dict())
    calls = {"validate": 0, "prepare": 0}
    validate, prepare = model.validate, model._prepare_execution

    def checked(*args, **kwargs):
        calls["validate"] += 1
        return validate(*args, **kwargs)

    def prepared(*args, **kwargs):
        calls["prepare"] += 1
        return prepare(*args, **kwargs)

    monkeypatch.setattr(model, "validate", checked)
    monkeypatch.setattr(model, "_prepare_execution", prepared)
    LinearDDIMImageSampler(20)(model, data, "gripper")
    assert calls == {"validate": 1, "prepare": 1}
    assert tuple(model.state_dict()) == before
    for component in (
        LinearGaussianImageNoising(),
        PlainPixelMSE(),
        LinearDDIMImageSampler(),
    ):
        assert not isinstance(component, torch.nn.Module)
