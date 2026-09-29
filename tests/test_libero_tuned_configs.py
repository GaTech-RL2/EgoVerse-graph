from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytest

from egomimic.benchmarks.libero.cluster import (
    configure_global_basis_candidate,
    global_basis_training_arguments,
)


def composed(basis, tmp_path):
    args = global_basis_training_arguments(
        basis, tmp_path / "data", tmp_path, "full", "test-run", gpus=4
    )
    with initialize_config_dir(
        config_dir=str(Path(__file__).parents[1] / "egomimic/hydra_configs"),
        version_base="1.3",
    ):
        return compose(
            config_name="train_zarr_cartesian",
            overrides=args[3:] + [f"paths.work_dir={tmp_path}"],
        )


def test_uniform_config_is_unchanged_and_rejects_tuning(tmp_path):
    cfg = composed("uniform", tmp_path)
    original = OmegaConf.to_container(cfg, resolve=True)
    assert configure_global_basis_candidate(cfg) is cfg
    assert OmegaConf.to_container(cfg, resolve=True) == original
    for override in ({"geometry": 64}, {"causal_attn": False}, {"token_affine": {}}):
        with pytest.raises(ValueError, match="frozen"):
            configure_global_basis_candidate(cfg, **override)


@pytest.mark.parametrize("basis", ["fourier", "chebyshev"])
def test_tuned_dims_attention_and_self_contained_normalization(tmp_path, basis):
    grid = "chebyshev_lobatto" if basis == "chebyshev" else "uniform"
    affine = {
        "kind": "bounded_train_std_v1",
        "basis": basis,
        "geometry": 128,
        "clock": 31,
        "geometry_fit_grid": grid,
        "fit_split": "train_only",
        "training_windows_sha256": "a" * 64,
        "center": [[0.0] * 12] * 104,
        "scale": [[0.01] * 12] * 104,
    }
    cfg = configure_global_basis_candidate(
        composed(basis, tmp_path),
        geometry=128,
        geometry_fit_grid=grid,
        causal_attn=False,
        token_affine=affine,
    )
    model = instantiate(cfg.model.pipeline, device="cpu")
    stages = model.pipeline.stages
    assert stages[1].representation_context() == stages[-1].representation_context()
    assert stages[1].codec.num_waypoints == 104
    assert stages[2].action_horizon == stages[3].action_horizon == 104
    network = stages[3].policy.model
    # GraphDiffusionTransformer wraps the maintained OAT Transformer.
    assert cfg.model.pipeline.stages[3].policy.model.causal_attn is False
    assert sum(p.numel() for p in model.nets.parameters()) == 27_203_476
    assert (
        sum(p.numel() for p in model.nets.parameters() if p.requires_grad) == 27_203_404
    )
    assert cfg.model.benchmark_protocol.horizon == 32
    assert cfg.model.benchmark_protocol.n_action_steps == 16
    path = tmp_path / f"{basis}.yaml"
    OmegaConf.save(cfg, path, resolve=True)
    loaded = OmegaConf.load(path)
    assert (
        loaded.model.pipeline.stages[1].token_affine
        == loaded.model.pipeline.stages[-1].token_affine
    )
    assert "${" not in path.read_text()
