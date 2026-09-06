import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from egomimic.synthetic.multi_action_adapter_flow import (
    SyntheticMultiActionAdapterFlow,
)


def test_private_adapters_and_shared_field_receive_expected_gradients():
    model = SyntheticMultiActionAdapterFlow(
        embodiments=["shallow", "steep"], field_width=16, field_depth=2
    )
    losses = model.losses_for_embodiment(
        "shallow", torch.randn(8, 3), flow_samples=2
    )
    losses["loss"].backward()
    assert any(p.grad is not None and bool(p.grad.abs().sum()) for p in model.field.parameters())
    assert any(
        p.grad is not None and bool(p.grad.abs().sum())
        for p in model.encoders["shallow"].parameters()
    )
    assert any(
        p.grad is not None and bool(p.grad.abs().sum())
        for p in model.decoders["shallow"].parameters()
    )
    assert all(p.grad is None for p in model.encoders["steep"].parameters())
    assert all(p.grad is None for p in model.decoders["steep"].parameters())


def test_private_decoders_share_one_generated_latent_path():
    model = SyntheticMultiActionAdapterFlow(
        embodiments=["shallow", "steep"], field_width=16, field_depth=2
    )
    noise = torch.randn(5, 8)
    shallow = model.trajectory(noise, embodiment="shallow", steps=3)
    steep = model.trajectory(noise, embodiment="steep", steps=3)
    assert shallow.shape == steep.shape == (4, 5, 3)
    torch.testing.assert_close(shallow, steep)


def test_private_adapters_support_different_action_dimensions():
    model = SyntheticMultiActionAdapterFlow(
        embodiments=["torus", "checkerboard"],
        action_dims={"torus": 3, "checkerboard": 2},
        field_width=16,
        field_depth=2,
    )
    noise = torch.randn(5, 8)
    assert model.trajectory(noise, embodiment="torus", steps=3).shape == (4, 5, 3)
    assert model.trajectory(noise, embodiment="checkerboard", steps=3).shape == (
        4,
        5,
        2,
    )
    for name, dimension in model.action_dims.items():
        losses = model.losses_for_embodiment(
            name, torch.randn(6, dimension), flow_samples=2
        )
        assert all(bool(torch.isfinite(value)) for value in losses.values())


def test_paraboloid_surface_metric_is_zero_on_surface():
    from egomimic.eval.synthetic_trajectory_eval import SyntheticTrajectoryEval

    xy = torch.randn(20, 2)
    curvature = 0.5
    points = torch.cat(
        [xy, curvature * xy.square().sum(dim=-1, keepdim=True)], dim=-1
    )
    torch.testing.assert_close(
        SyntheticTrajectoryEval.paraboloid_surface_rmse(
            points, curvature=curvature
        ),
        torch.tensor(0.0),
    )


def test_multi_trainer_runs_optimizer_validation_and_immutable_checkpoint(tmp_path):
    repository = Path(__file__).parents[1]
    source = tmp_path / "source"
    source.mkdir()
    datasets = {}
    evaluations = {}
    target_keys = {}
    rng = np.random.default_rng(5)
    for name, curvature, action_dim in (
        ("shallow", 0.125, 3),
        ("planar", None, 2),
    ):
        latent = rng.normal(size=(24, 8)).astype(np.float32)
        xy = rng.normal(size=(24, 2)).astype(np.float32)
        if action_dim == 3:
            target = np.concatenate(
                [xy, curvature * np.square(xy).sum(axis=1, keepdims=True)], axis=1
            ).astype(np.float32)
            target_key = "target_3d"
        else:
            target = xy
            target_key = "target_2d"
        split = np.array([0] * 20 + [1] * 4, dtype=np.uint8)
        path = source / f"{name}.npz"
        np.savez_compressed(
            path,
            source_gaussian_latent=latent,
            split=split,
            **{target_key: target},
        )
        datasets[name] = str(path)
        evaluations[name] = str(path)
        target_keys[name] = target_key
    output = tmp_path / "run"
    config = {
        "seed": 42,
        "output_dir": str(output),
        "datasets": datasets,
        "evaluation_datasets": evaluations,
        "target_keys": target_keys,
        "surface_specs": {
            "shallow": {"kind": "paraboloid", "curvature": 0.125}
        },
        "source_key": "source_gaussian_latent",
        "batch_size_per_embodiment": 4,
        "flow_samples": 2,
        "learning_rate": 3e-4,
        "max_steps": 1,
        "log_every": 1,
        "checkpoint_every": 1,
        "inference_steps": 2,
        "evaluation_particles": 4,
        "model": {
            "embodiments": ["shallow", "planar"],
            "action_dims": {"shallow": 3, "planar": 2},
            "latent_dim": 8,
            "residual_width": 8,
            "residual_depth": 1,
            "field_width": 8,
            "field_depth": 1,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    subprocess.run(
        [
            sys.executable,
            str(repository / "scripts/train/train_synthetic_multi_action_flow.py"),
            "--config",
            str(config_path),
        ],
        check=True,
    )
    assert list((output / "checkpoints").glob("epoch-equivalent-*-global-step-000001.pt"))
    summary = json.loads((output / "summary.json").read_text())
    assert set(summary["embodiments"]) == {"shallow", "planar"}
    assert summary["shared_field_parameters"] > 0
    for name in ("shallow", "planar"):
        assert (output / f"validation_trajectory_{name}.npz").is_file()
        values = summary["embodiments"][name]
        assert values["validation_decoder_jacobian_singular_min"] > 0.0
        assert values["validation_decoder_jacobian_singular_median"] > 0.0
        assert values["validation_decoder_jacobian_singular_max"] > 0.0
