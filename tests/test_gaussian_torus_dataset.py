import math
import subprocess
import sys
from pathlib import Path

import torch

from egomimic.synthetic import GaussianTorusDataset, generate_gaussian_torus


def test_generator_script_resolves_the_checkout_outside_repo_cwd(tmp_path):
    source = Path(__file__).parents[1]
    output = tmp_path / "generated.npz"
    subprocess.run(
        [
            sys.executable,
            str(source / "scripts/data/generate_gaussian_torus.py"),
            "--output",
            str(output),
            "--count",
            "32",
            "--source-dim",
            "8",
        ],
        cwd=tmp_path,
        check=True,
    )
    assert output.is_file()


def test_gaussian_torus_is_deterministic_and_on_surface():
    first = generate_gaussian_torus(128, seed=7)
    second = generate_gaussian_torus(128, seed=7)
    assert torch.equal(first.source_2d, second.source_2d)
    assert torch.equal(first.source_gaussian_3d, second.source_gaussian_3d)
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
    assert torch.equal(batch.source_gaussian_3d, reference.source_gaussian_3d)
    assert batch.source_gaussian_latent.shape == (64, 4)
    assert torch.equal(batch.source_gaussian_3d, batch.source_gaussian_latent[:, :3])
