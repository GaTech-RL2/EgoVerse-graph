import torch

from egomimic.synthetic import generate_gaussian_checkerboard


def test_checkerboard_is_deterministic_independent_and_dimensionally_distinct():
    left = generate_gaussian_checkerboard(4096, seed=42, source_dim=8)
    right = generate_gaussian_checkerboard(4096, seed=42, source_dim=8)
    torch.testing.assert_close(left.source_gaussian_latent, right.source_gaussian_latent)
    torch.testing.assert_close(left.target_2d, right.target_2d)
    assert left.source_gaussian_latent.shape == (4096, 8)
    assert left.target_2d.shape == (4096, 2)
    assert left.mode_centers.shape == (8, 2)
    assert left.mode_weights.shape == (8,)
    assert torch.isclose(left.mode_weights.sum(), torch.tensor(1.0))
    frequencies = torch.bincount(left.mode_indices, minlength=8).float() / 4096
    assert torch.max(torch.abs(frequencies - left.mode_weights)) < 0.03
    correlation = torch.corrcoef(
        torch.stack((left.source_gaussian_latent[:, 0], left.target_2d[:, 0]))
    )[0, 1]
    assert correlation.abs() < 0.08
