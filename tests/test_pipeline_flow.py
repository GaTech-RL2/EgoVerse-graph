"""Flow-matching stages: the head HPT trains with, as graph nodes."""

import pytest
import torch
import torch.nn as nn

from egomimic.pipeline.core import Pipeline
from egomimic.pipeline.stages_flow import (
    FlowDenoiserStage,
    FlowNoisingStage,
    FlowVelocityLossStage,
)

_H, _D, _C = 8, 5, 16


class _Field(nn.Module):
    """Tiny velocity field: sees the point, the time and the condition."""

    def __init__(self, action_dim: int = _D, cond_dim: int = _C):
        super().__init__()
        self.net = nn.Linear(action_dim + cond_dim + 1, action_dim)

    def forward(self, x, t, cond):
        batch, horizon, _ = x.shape
        cond = cond[:, None, :].expand(batch, horizon, cond.shape[-1])
        time = t[:, None, None].expand(batch, horizon, 1)
        return self.net(torch.cat([x, cond, time], dim=-1))


def _pipeline(steps: int = 4) -> Pipeline:
    return Pipeline(
        [
            FlowNoisingStage(action_horizon=_H, action_dim=_D),
            FlowDenoiserStage(
                model=_Field(),
                action_horizon=_H,
                action_dim=_D,
                condition_input_dim=_C,
                num_inference_steps=steps,
            ),
            FlowVelocityLossStage(),
        ]
    )


def _batch(size: int = 3) -> dict:
    return {
        "condition": torch.randn(size, _C),
        "target": torch.randn(size, _H, _D),
    }


# -- noising ----------------------------------------------------------------


def test_noising_declares_the_flow_namespace_it_writes():
    reads, writes = FlowNoisingStage(action_horizon=_H, action_dim=_D).contract("train")
    assert reads == ("target",)
    assert set(writes) == {"flow/noisy_action", "flow/velocity_target", "flow/time"}


def test_noising_is_train_only():
    stage = FlowNoisingStage(action_horizon=_H, action_dim=_D)
    _, excluded = Pipeline([stage]).plan(["target"], mode="inference")
    assert excluded[0][1] == ["<train-only>"]


def test_velocity_target_is_the_straight_line_field():
    """u_t = noise - action, and x_t = t*noise + (1-t)*action.

    Both are recoverable from the outputs, so this pins the identity rather
    than just the shapes: action = x_t - t * u_t.
    """
    stage = FlowNoisingStage(action_horizon=_H, action_dim=_D)
    batch = _batch()
    target = batch["target"].clone()
    out = stage.forward(batch)
    t = out["flow/time"][:, None, None]
    recovered = out["flow/noisy_action"] - t * out["flow/velocity_target"]
    torch.testing.assert_close(recovered, target, atol=1e-5, rtol=1e-5)


def test_sampled_time_stays_strictly_inside_the_unit_interval():
    stage = FlowNoisingStage(action_horizon=_H, action_dim=_D)
    times = torch.cat([stage.forward(_batch(64))["flow/time"] for _ in range(8)])
    assert times.min() > 0.0 and times.max() < 1.0


def test_beta_time_leans_toward_the_noisy_end():
    # beta(1.5, 1) is what HPT uses; it should put more mass above 0.5 than a
    # uniform draw does, which is the reason to prefer it.
    beta = FlowNoisingStage(action_horizon=_H, action_dim=_D, time_dist="beta")
    uniform = FlowNoisingStage(action_horizon=_H, action_dim=_D, time_dist="uniform")
    torch.manual_seed(0)
    beta_mean = torch.cat(
        [beta.forward(_batch(256))["flow/time"] for _ in range(4)]
    ).mean()
    uniform_mean = torch.cat(
        [uniform.forward(_batch(256))["flow/time"] for _ in range(4)]
    ).mean()
    assert beta_mean > uniform_mean


def test_noising_rejects_an_unknown_time_distribution():
    with pytest.raises(ValueError, match="time_dist"):
        FlowNoisingStage(action_horizon=_H, action_dim=_D, time_dist="cosine")


@pytest.mark.parametrize("kwargs", [{"beta_alpha": 0.0}, {"beta_beta": -1.0}])
def test_noising_rejects_degenerate_beta_parameters(kwargs):
    with pytest.raises(ValueError, match="beta_"):
        FlowNoisingStage(action_horizon=_H, action_dim=_D, **kwargs)


def test_noising_rejects_a_target_of_the_wrong_shape():
    stage = FlowNoisingStage(action_horizon=_H, action_dim=_D)
    with pytest.raises(ValueError, match="Flow target must be"):
        stage.forward({"target": torch.zeros(2, _H + 1, _D)})


def test_noising_keeps_the_working_dtype_it_was_given():
    # float32 in, float32 out. A float64 target is normalised instead, which
    # test_noising_normalises_a_float64_target covers.
    stage = FlowNoisingStage(action_horizon=_H, action_dim=_D)
    out = stage.forward({"target": torch.zeros(2, _H, _D, dtype=torch.float32)})
    assert out["flow/noisy_action"].dtype == torch.float32


# -- denoiser ---------------------------------------------------------------


def test_denoiser_contracts_differ_by_mode_like_the_diffusion_one():
    stage = FlowDenoiserStage(
        model=_Field(), action_horizon=_H, action_dim=_D, condition_input_dim=_C
    )
    assert stage.contract("train") == (
        ("condition", "flow/noisy_action", "flow/time"),
        ("flow/predicted_velocity",),
    )
    assert stage.contract("inference") == (("condition",), ("pred_action", "log/*"))


def test_denoiser_predicts_a_field_of_the_action_shape():
    out = _pipeline().execute(_batch(), mode="train")
    assert out["flow/predicted_velocity"].shape == (3, _H, _D)


def test_denoiser_integrates_to_an_action_at_inference():
    out = _pipeline(steps=6).execute(
        {"condition": torch.randn(4, _C)}, mode="inference"
    )
    assert out["pred_action"].shape == (4, _H, _D)
    assert torch.isfinite(out["pred_action"]).all()


def test_inference_needs_no_noising_stage_present():
    """The whole point of the mode split: sampling starts from its own noise."""
    pipeline = _pipeline()
    runnable, excluded = pipeline.plan(["condition"], mode="inference")
    assert [type(s).__name__ for s in runnable] == ["FlowDenoiserStage"]
    assert {e[1][0] for e in excluded} == {"<train-only>"}


def test_more_inference_steps_change_the_integration_result():
    torch.manual_seed(0)
    field = _Field()

    def sample(steps):
        torch.manual_seed(1)
        stage = FlowDenoiserStage(
            model=field,
            action_horizon=_H,
            action_dim=_D,
            condition_input_dim=_C,
            num_inference_steps=steps,
        )
        return stage.execute({"condition": torch.zeros(2, _C)}, mode="inference")[
            "pred_action"
        ]

    assert not torch.allclose(sample(2), sample(32))


def test_denoiser_rejects_a_condition_of_the_wrong_width():
    stage = FlowDenoiserStage(
        model=_Field(), action_horizon=_H, action_dim=_D, condition_input_dim=_C
    )
    with pytest.raises(ValueError, match="Flow condition must have shape"):
        stage.execute({"condition": torch.zeros(2, _C + 1)}, mode="inference")


def test_denoiser_rejects_a_three_dimensional_condition():
    # HPT pools to (B, D); a leftover token axis is a wiring bug, not input.
    stage = FlowDenoiserStage(
        model=_Field(), action_horizon=_H, action_dim=_D, condition_input_dim=_C
    )
    with pytest.raises(ValueError, match="Flow condition"):
        stage.execute({"condition": torch.zeros(2, 4, _C)}, mode="inference")


@pytest.mark.parametrize("steps", [0, -1])
def test_denoiser_rejects_a_nonpositive_step_count(steps):
    with pytest.raises(ValueError, match="num_inference_steps"):
        FlowDenoiserStage(
            model=_Field(),
            action_horizon=_H,
            action_dim=_D,
            condition_input_dim=_C,
            num_inference_steps=steps,
        )


# -- loss -------------------------------------------------------------------


def test_loss_writes_the_summed_namespace_and_diagnostics():
    out = _pipeline().execute(_batch(), mode="train")
    assert out["loss/flow_velocity"].ndim == 0
    assert "log/flow_target_rms" in out


def test_loss_is_zero_for_a_perfect_prediction():
    stage = FlowVelocityLossStage()
    field = torch.randn(2, _H, _D)
    out = stage.forward(
        {"flow/predicted_velocity": field, "flow/velocity_target": field.clone()}
    )
    assert float(out["loss/flow_velocity"]) == pytest.approx(0.0)


def test_loss_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="Flow loss shape mismatch"):
        FlowVelocityLossStage().forward(
            {
                "flow/predicted_velocity": torch.zeros(2, _H, _D),
                "flow/velocity_target": torch.zeros(2, _H + 1, _D),
            }
        )


def test_loss_is_train_only():
    _, excluded = Pipeline([FlowVelocityLossStage()]).plan(
        ["flow/predicted_velocity", "flow/velocity_target"], mode="inference"
    )
    assert excluded[0][1] == ["<train-only>"]


# -- the objective actually learns -----------------------------------------


def test_the_flow_objective_converges_on_a_fixed_target():
    """End to end sanity: the graph must be able to fit something.

    A constant action given a constant condition. If the interpolation, the
    velocity target or the loss were wrong, this would not come down.
    """
    torch.manual_seed(0)
    pipeline = _pipeline(steps=16)
    optimizer = torch.optim.Adam(pipeline.parameters(), lr=1e-2)
    condition = torch.ones(16, _C)
    target = torch.full((16, _H, _D), 0.5)

    first = None
    for step in range(150):
        out = pipeline.execute(
            {"condition": condition, "target": target.clone()}, mode="train"
        )
        loss = out["loss/flow_velocity"]
        if first is None:
            first = float(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert float(loss) < first * 0.7, f"{first} -> {float(loss)}"

    sampled = pipeline.execute({"condition": condition}, mode="inference")[
        "pred_action"
    ]
    # Sampling should land nearer the target than pure noise would.
    assert (sampled - target).abs().mean() < 1.0


# -- token-shaped conditions ------------------------------------------------


class _TokenCondField(nn.Module):
    """A denoiser that cross-attends, so it needs a (B, T, C) condition."""

    def __init__(self, action_dim: int = _D, cond_dim: int = _C):
        super().__init__()
        self.net = nn.Linear(action_dim + cond_dim, action_dim)

    def forward(self, x, t, cond):
        if cond.ndim != 3:
            raise AssertionError(f"expected a token condition, got {tuple(cond.shape)}")
        pooled = cond.mean(dim=1)[:, None, :].expand(-1, x.shape[1], -1)
        return self.net(torch.cat([x, pooled], dim=-1))


def test_condition_as_tokens_adds_a_token_axis_for_cross_attending_denoisers():
    stage = FlowDenoiserStage(
        model=_TokenCondField(),
        action_horizon=_H,
        action_dim=_D,
        condition_input_dim=_C,
        num_inference_steps=2,
        condition_as_tokens=True,
    )
    out = stage.execute({"condition": torch.randn(2, _C)}, mode="inference")
    assert out["pred_action"].shape == (2, _H, _D)


def test_condition_stays_pooled_by_default():
    # The default must keep feeding (B, C), which is what ConditionalUnet1D
    # and the existing diffusion path expect.
    stage = FlowDenoiserStage(
        model=_TokenCondField(),
        action_horizon=_H,
        action_dim=_D,
        condition_input_dim=_C,
        num_inference_steps=2,
    )
    with pytest.raises(AssertionError, match="expected a token condition"):
        stage.execute({"condition": torch.randn(2, _C)}, mode="inference")


def test_condition_validation_runs_before_any_reshape():
    stage = FlowDenoiserStage(
        model=_TokenCondField(),
        action_horizon=_H,
        action_dim=_D,
        condition_input_dim=_C,
        condition_as_tokens=True,
    )
    with pytest.raises(ValueError, match="Flow condition must have shape"):
        stage.execute({"condition": torch.zeros(2, _C + 3)}, mode="inference")


# -- working precision ------------------------------------------------------


def test_noising_normalises_a_float64_target():
    """Zarr actions arrive as Double; autocast never downcasts that."""
    stage = FlowNoisingStage(action_horizon=_H, action_dim=_D)
    batch = {"target": torch.zeros(2, _H, _D, dtype=torch.float64)}
    out = stage.forward(batch)
    for key in ("flow/noisy_action", "flow/velocity_target", "flow/time"):
        assert out[key].dtype == torch.float32, key
    # `target` is rewritten too, so the loss and any later stage agree.
    assert out["target"].dtype == torch.float32


def test_a_float64_target_runs_the_whole_graph():
    pipeline = _pipeline()
    batch = {
        "condition": torch.randn(3, _C),
        "target": torch.randn(3, _H, _D, dtype=torch.float64),
    }
    out = pipeline.execute(batch, mode="train")
    assert torch.isfinite(out["loss/flow_velocity"])


def test_an_explicit_working_dtype_is_honoured():
    stage = FlowNoisingStage(action_horizon=_H, action_dim=_D, dtype="float64")
    out = stage.forward({"target": torch.zeros(2, _H, _D, dtype=torch.float32)})
    assert out["flow/noisy_action"].dtype == torch.float64


def test_noising_rejects_a_non_floating_dtype():
    with pytest.raises(ValueError, match="floating torch dtype"):
        FlowNoisingStage(action_horizon=_H, action_dim=_D, dtype="int64")
