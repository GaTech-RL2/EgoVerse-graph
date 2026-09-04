import torch

from egomimic.synthetic.shared_latent_flow import (
    SyntheticDirectFlow,
    SyntheticSharedLatentFlow,
)
from scripts.train.train_synthetic_manifold import energy_distance


def test_direct_flow_has_no_encoder_or_decoder_and_updates_field():
    model = SyntheticDirectFlow(data_dim=3, field_width=16, field_depth=2)
    assert not hasattr(model, "encoder")
    assert not hasattr(model, "decoder")
    source = torch.randn(8, 3)
    target = torch.randn(8, 3)
    losses = model.losses(source, target, flow_samples=2)
    losses["loss"].backward()
    assert set(losses) == {"loss", "flow_loss"}
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert model.integrate(source, steps=2).shape == source.shape


def test_energy_distance_is_zero_for_identical_empirical_distributions():
    points = torch.randn(12, 3)
    torch.testing.assert_close(energy_distance(points, points), torch.tensor(0.0))


def test_both_objectives_are_finite_and_differentiable():
    for latent_dim in (2, 4):
        for method in ("unite", "vfm"):
            model = SyntheticSharedLatentFlow(latent_dim=latent_dim)
            losses = model.losses(
                torch.randn(8, latent_dim), torch.randn(8, 3), method=method
            )
            assert all(torch.isfinite(value) for value in losses.values())
            losses["loss"].backward()
            assert all(parameter.grad is not None for parameter in model.parameters())
