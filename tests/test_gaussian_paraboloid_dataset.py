import torch

from egomimic.synthetic import (
    GaussianParaboloidDataset,
    generate_gaussian_paraboloid,
)


def test_gaussian_paraboloid_is_deterministic_and_on_surface():
    first = generate_gaussian_paraboloid(128, seed=7, curvature=0.25)
    second = generate_gaussian_paraboloid(128, seed=7, curvature=0.25)
    torch.testing.assert_close(first.source_latent, second.source_latent)
    torch.testing.assert_close(first.source_gaussian_3d, second.source_gaussian_3d)
    torch.testing.assert_close(first.target_3d, second.target_3d)
    torch.testing.assert_close(first.target_3d[:, :2], first.source_2d)
    torch.testing.assert_close(
        first.target_3d[:, 2], 0.25 * first.source_2d.square().sum(dim=-1)
    )


def test_latent_four_preserves_paired_problem_and_adds_nuisance_dimensions():
    wide = generate_gaussian_paraboloid(64, seed=9, source_dim=4)
    reference = generate_gaussian_paraboloid(64, seed=9, source_dim=2)
    assert wide.source_latent.shape == (64, 4)
    torch.testing.assert_close(wide.source_latent[:, :2], reference.source_latent)
    torch.testing.assert_close(wide.target_3d, reference.target_3d)
    torch.testing.assert_close(
        wide.source_gaussian_3d, reference.source_gaussian_3d
    )


def test_dataset_exposes_exact_linear_bridge():
    dataset = GaussianParaboloidDataset(32, seed=11)
    indices = torch.tensor([0, 3, 7])
    time = torch.tensor([0.0, 0.5, 1.0])
    state, velocity = dataset.cfm_state_velocity(indices, time)
    source = dataset.data.source_3d[indices]
    target = dataset.data.target_3d[indices]
    torch.testing.assert_close(velocity, target - source)
    torch.testing.assert_close(state[0], source[0])
    torch.testing.assert_close(state[1], 0.5 * (source[1] + target[1]))
    torch.testing.assert_close(state[2], target[2])
