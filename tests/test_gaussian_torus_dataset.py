import math

import torch

from egomimic.synthetic import GaussianTorusDataset, generate_gaussian_torus


def test_gaussian_torus_is_deterministic_and_on_surface():
    first = generate_gaussian_torus(128, seed=7)
    second = generate_gaussian_torus(128, seed=7)
    assert torch.equal(first.source_2d, second.source_2d)
    assert torch.equal(first.target_3d, second.target_3d)
    radial = first.target_3d[:, :2].norm(dim=-1)
    residual = (radial - 2.0).square() + first.target_3d[:, 2].square()
    assert torch.allclose(residual, torch.full_like(residual, 0.65**2), atol=2e-6)
    assert torch.all((first.angles >= 0) & (first.angles <= 2 * math.pi))


def test_linear_cfm_bridge_endpoints_and_velocity():
    dataset = GaussianTorusDataset(16, seed=11)
    indices = torch.tensor([1, 4, 9])
    start, velocity = dataset.cfm_state_velocity(indices, torch.zeros(3))
    end, velocity_end = dataset.cfm_state_velocity(indices, torch.ones(3))
    assert torch.equal(start, dataset.data.source_3d[indices])
    assert torch.allclose(end, dataset.data.target_3d[indices])
    assert torch.equal(velocity, velocity_end)
    assert torch.allclose(end - start, velocity)


def test_four_dimensional_source_keeps_same_torus_coupling():
    batch = generate_gaussian_torus(64, seed=9, source_dim=4)
    assert batch.source_latent.shape == (64, 4)
    assert torch.equal(batch.source_latent[:, :2], batch.source_2d)
    reference = generate_gaussian_torus(64, seed=9, source_dim=2)
    assert torch.equal(batch.source_2d, reference.source_2d)
    assert torch.equal(batch.target_3d, reference.target_3d)
