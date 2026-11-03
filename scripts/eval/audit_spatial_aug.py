"""Geometry and opt-in checks for the image stem's spatial augmentation."""

import torch

from egomimic.models.stems.resnet_mlp import (
    ResNetMLPImageStem,
    random_rotate_translate_crop,
)

H, W, B = 240, 320, 64


def disc(radius=40):
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    d = ((x + 0.5 - W / 2) ** 2 + (y + 0.5 - H / 2) ** 2).sqrt()
    return (d < radius).float().expand(B, 3, H, W).clone()


def extent(images):
    mask = images[:, 0] > 0.5
    return mask.any(1).sum(1).float(), mask.any(2).sum(1).float()


def center(images):
    mask = images[:, 0]
    rows, cols = torch.arange(H) + 0.5, torch.arange(W) + 0.5
    total = mask.sum((1, 2))
    return (mask.sum(1) * cols).sum(1) / total, (mask.sum(2) * rows).sum(1) / total


def main():
    torch.manual_seed(0)
    x = disc()

    # A rotation must be rigid on a 4:3 image: a centered disc stays that disc.
    rotated = random_rotate_translate_crop(x, max_rotation_deg=45.0)
    assert (rotated - x).abs().mean() < 0.01

    # A shift moves t of each side, and each image draws its own.
    shifted = random_rotate_translate_crop(x, max_translate=0.1)
    cx, cy = center(shifted)
    assert (cx - W / 2).abs().max() <= 0.1 * W + 1
    assert (cy - H / 2).abs().max() <= 0.1 * H + 1
    assert (cx - W / 2).abs().max() > 0.05 * W and cx.std() > 1

    # A crop only zooms in, by at most 1 / min_crop, without moving the center.
    cropped = random_rotate_translate_crop(x, min_crop=0.5)
    widths, heights = extent(cropped)
    assert widths.min() >= 79 and widths.max() <= 161 and widths.max() > 120
    assert (widths - heights).abs().max() <= 2

    # The nested [B, T, C, H, W] layout the encoder accepts passes through.
    assert random_rotate_translate_crop(x[:, None]).shape == (B, 1, 3, H, W)

    # No parameters are added, and evaluation never augments.
    kwargs = dict(output_dim=16, weights=None)
    off = ResNetMLPImageStem(**kwargs)
    on = ResNetMLPImageStem(
        **kwargs, max_rotation_deg=3.0, max_translate=0.03, min_crop=0.92
    )
    on.load_state_dict(off.state_dict(), strict=True)
    off.eval(), on.eval()
    with torch.no_grad():
        assert torch.equal(off(x[:2]), on(x[:2]))

    # Hue is opt-out: recipes that do not set it keep today's jitter.
    assert off.jitter.hue == (-0.05, 0.05)
    no_hue = ResNetMLPImageStem(**kwargs, jitter_hue=0.0)
    assert no_hue.jitter.hue is None and no_hue.jitter.brightness == (0.9, 1.1)

    # Off by default: training draws exactly what it drew before this option.
    off.train()
    torch.manual_seed(1)
    jitter_only = off.jitter(x[:2])
    torch.manual_seed(1)
    seen = {}
    hook = off.encoder.register_forward_pre_hook(lambda _, a: seen.update(x=a[0]))
    off(x[:2])
    hook.remove()
    assert torch.equal(seen["x"], off.normalize(jitter_only))
    print("spatial aug audit passed")


if __name__ == "__main__":
    main()
