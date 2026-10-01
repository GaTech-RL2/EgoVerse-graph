"""Image augmentation/normalization and checkpoint contract of the visual stem."""

import torch
from torchvision.transforms.functional import normalize

from egomimic.models.stems.resnet_mlp import ResNetMLPImageStem


def test_eval_normalizes_once_and_retains_spatial_tokens():
    stem = ResNetMLPImageStem(output_dim=16, weights=None).eval()
    images = torch.rand(2, 3, 64, 64)
    inputs = []
    handle = stem.encoder.net.register_forward_pre_hook(
        lambda module, args: inputs.append(args[0].detach().clone())
    )
    with torch.no_grad():
        first = stem(images)
        second = stem(images)
    handle.remove()
    expected = normalize(images, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    torch.testing.assert_close(inputs[0], expected)
    torch.testing.assert_close(inputs[1], expected)
    torch.testing.assert_close(first, second)
    assert first.shape == (2, 4, 16)


def test_training_augments_and_backpropagates_through_encoder_and_mlp():
    stem = ResNetMLPImageStem(output_dim=16, weights=None).train()
    images = torch.rand(2, 3, 64, 64)
    inputs = []
    handle = stem.encoder.net.register_forward_pre_hook(
        lambda module, args: inputs.append(args[0].detach().clone())
    )
    torch.manual_seed(11)
    first = stem(images)
    torch.manual_seed(29)
    stem(images)
    handle.remove()
    assert not torch.allclose(inputs[0], inputs[1])
    first.square().mean().backward()
    for parameter in [
        stem.encoder.net[0].weight,
        stem.encoder.proj.weight,
        stem.mlp.net[0].weight,
    ]:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_legacy_target_loads_identical_checkpoint_keys():
    from egomimic.models.stems.e1_image import E1ImageStem

    legacy = E1ImageStem(output_dim=16, weights=None).eval()
    current = ResNetMLPImageStem(output_dim=16, weights=None).eval()
    assert {
        "encoder.net.0.weight",
        "encoder.proj.weight",
        "mlp.net.0.weight",
    } <= legacy.state_dict().keys()
    current.load_state_dict(legacy.state_dict(), strict=True)
    images = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        torch.testing.assert_close(current(images), legacy(images))
