import numpy as np
import pytest

from scripts.benchmarks.compare_libero_arc_bases import fit_bounded_scaling, noise_audit
from egomimic.rldb.zarr.libero_arc_global import LiberoArcGlobalCodec


def test_scaling_is_train_only_bounded_and_reversible():
    rng = np.random.default_rng(8)
    values = rng.normal(size=(8, 32, 12))
    values[:, 0, 0] = 4
    mask = np.arange(8) < 4
    center, scale, std = fit_bounded_scaling(values, mask, 32)
    changed = values.copy()
    changed[~mask] += 1000
    for left, right in zip(
        (center, scale, std), fit_bounded_scaling(changed, mask, 32)
    ):
        np.testing.assert_array_equal(left, right)
    assert np.isfinite(scale).all() and scale.min() >= 1e-4
    assert std[0, 0] == 0
    np.testing.assert_allclose((values - center) / scale * scale + center, values)
    with pytest.raises(ValueError):
        fit_bounded_scaling(values, np.zeros(8, dtype=bool), 32)


@pytest.mark.parametrize("basis", ["fourier", "chebyshev"])
def test_noise_audit_is_deterministic_and_does_not_change_codec(basis):
    codec = LiberoArcGlobalCodec(basis=basis)
    raw = np.random.default_rng(3).uniform(-0.1, 0.1, (4, 32, 7)).astype(np.float32)
    tokens = np.stack([codec.encode(row) for row in raw])
    frozen = tokens.copy()
    identities = [{"split": "train" if i < 2 else "valid"} for i in range(4)]
    first = noise_audit(codec, tokens, raw, identities, sigmas=(0.0, 0.001))
    assert first == noise_audit(codec, tokens, raw, identities, sigmas=(0.0, 0.001))
    np.testing.assert_array_equal(tokens, frozen)
    for variant in first["variants"].values():
        assert variant["perturbations"]["0.0"]["decoded_delta_mse_mean"] < 1e-12
        assert variant["perturbations"]["0.001"]["decoded_delta_mse_mean"] > 0


def test_optional_affine_stage_roundtrip_strict_state_and_identity():
    import copy
    import torch
    from egomimic.pipeline.stages_libero_arc import LiberoArcStage

    codec = LiberoArcGlobalCodec(basis="fourier", geometry=64)
    raw = np.random.default_rng(1).uniform(-0.1, 0.1, (6, 32, 7)).astype(np.float32)
    tokens = np.stack([codec.encode(row) for row in raw])
    audit = noise_audit(
        codec,
        tokens,
        raw,
        [{"split": "train" if i < 4 else "valid"} for i in range(6)],
        sigmas=(0.0,),
    )
    affine = audit["affine"]
    stage = LiberoArcStage(codec=codec, token_affine=affine, reconstruction=True)
    stage.action_scale.copy_(torch.arange(1, 8))
    stage.action_offset.copy_(torch.arange(7) / 5)
    actions = torch.from_numpy(raw) * stage.action_scale + stage.action_offset
    result = stage.execute({"actions": actions}, mode="inference")
    expected = np.stack([codec.decode(row) for row in tokens])
    torch.testing.assert_close(
        result["pred_action"],
        torch.from_numpy(expected) * stage.action_scale + stage.action_offset,
        atol=2e-5,
        rtol=1e-5,
    )
    restored = LiberoArcStage(codec=codec, token_affine=affine, operation="decode")
    restored.load_state_dict(stage.state_dict(), strict=True)
    torch.testing.assert_close(
        restored.execute({"pred_arc": result["target"]}, mode="inference")[
            "pred_action"
        ],
        result["pred_action"],
    )
    assert torch.isfinite(
        restored.execute({"pred_arc": result["target"].bfloat16()}, mode="inference")[
            "pred_action"
        ]
    ).all()
    assert stage.representation_context() == restored.representation_context()
    wrong = copy.deepcopy(affine)
    wrong["geometry"] = 128
    with pytest.raises(ValueError, match="identity differ"):
        LiberoArcStage(codec=codec, token_affine=wrong)
    wrong = copy.deepcopy(affine)
    wrong["scale"][0][0] = 0
    with pytest.raises(ValueError, match="affine arrays"):
        LiberoArcStage(codec=codec, token_affine=wrong)


def test_default_uniform_has_no_affine_state_or_numeric_change():
    import torch
    from egomimic.pipeline.stages_libero_arc import LiberoArcStage

    stage = LiberoArcStage(codec=LiberoArcGlobalCodec(basis="uniform"))
    raw = torch.from_numpy(
        np.random.default_rng(2).uniform(-0.1, 0.1, (2, 32, 7)).astype(np.float32)
    )
    expected = torch.tensor(
        np.stack([stage.codec.encode(a) for a in raw.numpy()])
    ) / torch.tensor(stage.codec.token_scale())
    assert not any("token_" in key for key in stage.state_dict())
    assert "token_affine" not in stage.representation_context()
    assert "geometry_fit_grid" not in stage.representation_context()
    assert torch.equal(
        stage.execute({"actions": raw}, mode="train")["target"], expected
    )
