import numpy as np
import pytest

from egomimic.rldb.zarr.arc_global_basis import evaluate_curve, fit_curve
from egomimic.rldb.zarr.libero_arc_global import LiberoArcGlobalCodec
from egomimic.rldb.zarr.libero_arc_timed import (
    codec_source_files,
    make_libero_arc_codec,
)


@pytest.mark.parametrize("basis", ["chebyshev", "fourier", "cosine"])
def test_shared_global_basis_is_anchored_and_roundtrips_coefficients(basis):
    coefficients = np.random.default_rng(12).normal(size=(16, 6))
    grid = np.linspace(0, 1, 257)
    values = evaluate_curve(coefficients, grid, basis)
    np.testing.assert_allclose(fit_curve(values, 16, basis), coefficients, atol=1e-11)
    np.testing.assert_allclose(values[0], coefficients[0], atol=1e-12)


@pytest.mark.parametrize("basis", ["uniform", "chebyshev", "fourier"])
@pytest.mark.parametrize("motion", ["translation", "rotation", "stationary"])
def test_constants_and_pi_crossing(basis, motion):
    actions = np.zeros((32, 7))
    actions[:, 6] = -1
    actions[11:23, 6] = 1
    if motion == "translation":
        actions[:, :3] = [0.1, -0.2, 0.05]
    if motion == "rotation":
        actions[:, 5] = 0.25  # Accumulated rotation crosses pi.
    codec = LiberoArcGlobalCodec(basis=basis)
    token = codec.encode(actions)
    assert token.shape == (32, 12)
    decoded = codec.decode(token)
    np.testing.assert_allclose(decoded[:, :3], actions[:, :3], atol=5e-6)
    np.testing.assert_allclose(decoded[:, 6], actions[:, 6], atol=2e-6)
    assert np.mean((decoded[:, 3:6] - actions[:, 3:6]) ** 2) < 2e-4
    assert codec.represented_seconds(token) == pytest.approx(1.6, abs=1e-6)


def test_open_fourier_curve_is_not_forced_closed():
    grid = np.linspace(0, 1, 257)
    values = np.column_stack((2 * grid, np.sin(2 * np.pi * grid)))
    coefficients = fit_curve(values, 16, "fourier")
    np.testing.assert_allclose(
        evaluate_curve(coefficients, grid, "fourier"), values, atol=1e-12
    )


def test_high_order_chebyshev_requires_stable_actual_sample_grid():
    coefficients = np.random.default_rng(5).normal(size=(128, 2)) / 128
    uniform = np.linspace(0, 1, 257)
    with pytest.raises(ValueError, match="Ill-conditioned"):
        fit_curve(evaluate_curve(coefficients, uniform, "chebyshev"), 128, "chebyshev")
    lobatto = (1 - np.cos(np.linspace(0, np.pi, 257))) / 2
    fitted = fit_curve(
        evaluate_curve(coefficients, lobatto, "chebyshev"),
        128,
        "chebyshev",
        sample_grid=lobatto,
    )
    np.testing.assert_allclose(fitted, coefficients, atol=1e-12)
    with pytest.raises(ValueError, match="sample grid"):
        fit_curve(np.zeros((257, 2)), 128, "chebyshev", sample_grid=lobatto[::-1])


def test_high_order_chebyshev_codec_lobatto_roundtrip_and_context():
    from egomimic.pipeline.stages_libero_arc import LiberoArcStage

    codec = LiberoArcGlobalCodec(
        basis="chebyshev", geometry=128, geometry_fit_grid="chebyshev_lobatto"
    )
    actions = np.zeros((32, 7))
    actions[:, :3] = [0.1, -0.05, 0.2]
    tokens = codec.encode(actions)
    assert tokens.shape == (104, 12) and abs(tokens).max() < 2
    np.testing.assert_allclose(codec.decode(tokens), actions, atol=1e-5)
    assert (
        LiberoArcStage(codec=codec).representation_context()["geometry_fit_grid"]
        == "chebyshev_lobatto"
    )
    with pytest.raises(ValueError, match="only supported"):
        LiberoArcGlobalCodec(basis="uniform", geometry_fit_grid="chebyshev_lobatto")


@pytest.mark.parametrize("basis", ["uniform", "fourier", "chebyshev"])
def test_stationary_rotation_tail_endpoint_roundoff_is_safe(monkeypatch, basis):
    codec = LiberoArcGlobalCodec(basis=basis)
    original = codec._stream

    def rounded_endpoint(*args):
        frames, progress, clock, duration = original(*args)
        progress[progress >= 1 - 1e-14] = np.nextafter(1.0, 0.0)
        return frames, progress, clock, duration

    monkeypatch.setattr(codec, "_stream", rounded_endpoint)
    actions = np.zeros((32, 7))
    actions[:16, 5] = 0.1
    assert np.isfinite(codec.decode(codec.encode(actions))).all()


def test_geometry_is_the_only_difference_between_arms():
    actions = np.random.default_rng(4).uniform(-0.2, 0.2, (32, 7))
    outputs = [
        LiberoArcGlobalCodec(basis=b).encode(actions).ravel()
        for b in ("uniform", "fourier", "chebyshev")
    ]
    for other in outputs[1:]:
        np.testing.assert_array_equal(other[9 * 32 :], outputs[0][9 * 32 :])


@pytest.mark.parametrize("basis", ["uniform", "chebyshev", "fourier"])
def test_independent_pauses_and_time_scaling(basis):
    actions = np.zeros((32, 7))
    actions[8:16, 0] = 0.2
    actions[24:28, 0] = 0.2
    actions[4:12, 5] = 0.2
    actions[20:24, 5] = -0.2
    actions[:, 6] = -1
    actions[16:, 6] = 1
    fast, slow = (LiberoArcGlobalCodec(basis=basis, dt=dt) for dt in (0.05, 0.1))
    a, b = fast.encode(actions), slow.encode(actions)
    np.testing.assert_array_equal(a.ravel()[:-2], b.ravel()[:-2])
    np.testing.assert_allclose(b.ravel()[-2:], a.ravel()[-2:] * 2)
    decoded = fast.decode(a)
    np.testing.assert_allclose(slow.decode(b), decoded, atol=1e-6)
    assert np.mean((decoded - actions) ** 2) < 5e-4
    assert abs(decoded[1, 0]) < 0.003
    assert abs(decoded[18, 0]) < 0.003
    np.testing.assert_array_equal(np.sign(decoded[:, 6]), actions[:, 6])


def test_fractional_independent_budgets_and_invalid_predictions():
    actions = np.zeros((32, 7))
    actions[:, 0] = 0.2
    actions[:, 5] = np.deg2rad(3) / 0.5
    codec = LiberoArcGlobalCodec(
        basis="uniform", max_translation=0.055, max_rotation_degrees=13.5
    )
    decoded = codec.decode(codec.encode(actions))
    np.testing.assert_allclose(decoded[:5, 0], 0.2, atol=4e-6)
    assert decoded[5, 0] == pytest.approx(0.1, abs=4e-6)
    np.testing.assert_allclose(decoded[6:, 0], 0, atol=4e-6)
    assert decoded[4, 5] == pytest.approx(actions[4, 5] / 2, abs=4e-6)
    np.testing.assert_allclose(codec.decode(np.zeros((32, 12))), 0, atol=1e-12)
    with pytest.raises(ValueError, match="nonfinite"):
        codec.decode(np.full((32, 12), np.nan))
    with pytest.raises(ValueError, match="rows differ"):
        LiberoArcGlobalCodec(num_waypoints=16)


def test_factory_source_gate_and_graph_normalization_once():
    import torch

    from egomimic.pipeline.stages_libero_arc import LiberoArcStage

    assert make_libero_arc_codec(mode="global_basis").mode == "global_basis"
    assert "egomimic/rldb/zarr/arc_global_basis.py" in codec_source_files(
        "global_basis"
    )
    stage = LiberoArcStage(
        arc_mode="global_basis", reconstruction=True, encode_cache_size=2
    )
    stage.action_scale.copy_(torch.arange(1, 8))
    stage.action_offset.copy_(torch.arange(7) / 5)
    native = torch.zeros(2, 32, 7)
    native[:, :, 0] = 0.1
    normalized = native * stage.action_scale + stage.action_offset
    result = stage.execute({"actions": normalized}, mode="inference")
    torch.testing.assert_close(result["pred_action"], normalized, atol=1e-5, rtol=1e-5)
    decoder = LiberoArcStage(arc_mode="global_basis", operation="decode")
    assert torch.isfinite(
        decoder.execute({"pred_arc": result["target"].bfloat16()}, mode="inference")[
            "pred_action"
        ]
    ).all()
    original = stage._encode(native.numpy()).copy()
    stage.codec.basis = "fourier"
    changed = stage._encode(native.numpy())
    assert not np.array_equal(original, changed)
    assert stage.representation_context()["basis"] == "fourier"


def test_matched_configs_and_real_network_dimensions():
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    resolved = []
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).parents[1] / "egomimic/hydra_configs"),
    ):
        for basis in ("uniform", "fourier", "chebyshev"):
            cfg = compose(
                config_name="train_zarr_cartesian",
                overrides=[
                    f"+experiment=oat/libero_arc_global_{basis}",
                    "benchmark.dataset=/unused-for-construction",
                    "hydra/launcher=basic",
                ],
            )
            graph = instantiate(cfg.model.pipeline, device="cpu")
            stages = graph.pipeline.stages
            assert (
                stages[1].representation_context()
                == stages[-1].representation_context()
            )
            assert sum(p.numel() for p in graph.nets.parameters()) == 27_185_044
            assert (
                sum(p.numel() for p in graph.nets.parameters() if p.requires_grad)
                == 27_184_972
            )
            assert stages[1].codec.scalars == 384
            assert not cfg.model.pipeline.stages[3].policy.noise_scheduler.clip_sample
            model = OmegaConf.to_container(cfg.model, resolve=True)
            for index in (1, -1):
                assert model["pipeline"]["stages"][index]["codec"].pop("basis") == basis
            resolved.append(model)
    assert resolved[0] == resolved[1] == resolved[2]


def test_fixed_replay_keeps_all_three_arms_and_native_reconstruction_path():
    from pathlib import Path

    import yaml

    from egomimic.benchmarks.libero.replay import (
        candidates_from_spec,
        reconstruct_episode,
        validate_spec,
    )

    spec = yaml.safe_load(
        (
            Path(__file__).parents[1]
            / "egomimic/hydra_configs/benchmark/libero_arc_global_replay.yaml"
        ).read_text()
    )
    validate_spec(spec)
    candidates = candidates_from_spec(spec)
    assert {c["basis"] for c in candidates.values()} == {
        "uniform",
        "fourier",
        "chebyshev",
    }
    actions = np.zeros((73, 7))
    actions[:, 0] = 0.1
    actions[:, 6] = -1
    for candidate in candidates.values():
        decoded, metrics = reconstruct_episode(actions, candidate, spec)
        np.testing.assert_allclose(decoded, actions, atol=5e-6)
        assert metrics["execution_coverage"] == pytest.approx(1)


def test_local_replay_never_initializes_object_store(tmp_path, monkeypatch):
    import importlib.metadata
    import json
    from pathlib import Path

    from egomimic.benchmarks.libero import replay

    spec = (
        Path(__file__).parents[1]
        / "egomimic/hydra_configs/benchmark/libero_arc_global_replay.yaml"
    )
    monkeypatch.setenv("SOURCE_COMMIT", "a" * 40)
    monkeypatch.setattr(replay.subprocess, "check_output", lambda *a, **kw: "a" * 40)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "fixture")

    def forbidden(*args, **kwargs):
        raise AssertionError("Local-only replay must not touch object storage")

    monkeypatch.setattr(replay, "ArtifactUploader", forbidden)
    monkeypatch.setattr(replay, "calibrate", lambda *a, **kw: {"confirmed": True})
    monkeypatch.setattr(
        "sys.argv",
        [
            "replay",
            "--root",
            str(tmp_path),
            "--suite",
            "libero_10",
            "--run-id",
            "test",
            "--spec",
            str(spec),
            "--local-only",
        ],
    )
    replay.main()
    runtime = json.loads((tmp_path / "evidence/runtime.json").read_text())
    assert runtime["local_only"] and runtime["artifact_prefix"] is None
