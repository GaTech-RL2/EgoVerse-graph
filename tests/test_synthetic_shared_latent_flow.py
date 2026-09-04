import torch

from egomimic.synthetic.shared_latent_flow import SyntheticSharedLatentFlow


def test_both_objectives_are_finite_and_differentiable():
    for method in ("unite", "vfm"):
        model = SyntheticSharedLatentFlow()
        losses = model.losses(torch.randn(8, 2), torch.randn(8, 3), method=method)
        assert all(torch.isfinite(value) for value in losses.values())
        losses["loss"].backward()
        assert all(parameter.grad is not None for parameter in model.parameters())
