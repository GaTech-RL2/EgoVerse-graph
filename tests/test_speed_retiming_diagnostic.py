import numpy as np
import pytest
import torch

from egomimic.rldb.zarr.planar_retiming import PlanarCommandRetiming
from egomimic.pipeline.stages_speed import SharedSpeedCondition


@pytest.mark.parametrize("view,rate", list(enumerate((1, 1.25, 1.5, 1.75, 2))))
def test_constant_speed_native_retiming(view, rate):
    t = PlanarCommandRetiming((1, 1.25, 1.5, 1.75, 2))
    native = np.stack((np.arange(31), np.zeros(31), np.zeros(31),
                       np.linspace(0, 1, 31)), axis=-1)
    obs = np.array([5, 6, 7])
    b = t.transform({"actions": native, "_retiming_view": view, "obs": obs})
    np.testing.assert_allclose(b["actions"][:, 0], np.arange(16)*rate)
    np.testing.assert_allclose(b["actions"][:, 3], np.arange(16)*rate/30)
    np.testing.assert_allclose(b["retiming_rate"], [rate])
    assert b["obs"] is obs
    assert np.array_equal(native[:, 0], np.arange(31))


def test_wrap_stationary_and_tail_rejection():
    t = PlanarCommandRetiming((1.5,), horizon=3)
    a = np.zeros((4, 3))
    a[:, 2] = [3.0, 3.1, -3.1, -3.0]
    b = t.transform({"actions": a, "_retiming_view": 0})
    assert abs(abs(b["actions"][1, 2]) - np.pi) < 1e-6
    assert b["retiming_rate"].item() == 1.5
    assert "requested_speed" not in b
    with pytest.raises(ValueError, match="unpadded"):
        t.transform({"actions": a[:3], "_retiming_view": 0})


@pytest.mark.parametrize("encoding", ("scalar", "fourier"))
def test_condition_modes_rng_and_learning(encoding):
    torch.manual_seed(123)
    rng = torch.get_rng_state().clone()
    stage = SharedSpeedCondition(None, encoding)
    assert torch.equal(rng, torch.get_rng_state())
    assert stage.contract("train") == stage.contract("inference")
    cond = torch.zeros(2, 128)
    speed = torch.tensor([[1.], [2.]])
    out = stage({"condition": cond, "retiming_rate": speed})["speed_condition"]
    assert torch.equal(out, cond)
    opt = torch.optim.SGD(stage.parameters(), lr=.01)
    for _ in range(2):
        opt.zero_grad()
        out = stage({"condition": cond, "retiming_rate": speed})["speed_condition"]
        (out - 1).square().mean().backward()
        opt.step()
    assert stage.mlp[0].weight.grad.abs().sum() > 0
    assert not torch.equal(out[0], out[1])
    with pytest.raises(KeyError):
        stage({"condition": cond})
    with pytest.raises(ValueError):
        stage({"condition": cond, "retiming_rate": -torch.ones(2, 1)})


def test_unchanged_initialization_across_arms():
    weights = []
    for encoding in ("scalar", "fourier"):
        torch.manual_seed(42)
        before = torch.nn.Linear(10, 10)
        SharedSpeedCondition(None, encoding)
        after = torch.nn.Linear(10, 10)
        weights.append((before.weight.detach(), after.weight.detach()))
    assert all(torch.equal(a, b) for a, b in zip(*weights))


def test_condition_initializer_never_reseeds_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Condition initializer must not reseed CUDA")
    monkeypatch.setattr(torch, "manual_seed", forbidden)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", forbidden)
    rng = torch.get_rng_state().clone()
    for encoding in ("scalar", "fourier"):
        SharedSpeedCondition(None, encoding)
        assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("encoding", ("scalar", "fourier"))
def test_real_graph_consumes_speed_in_both_modes(encoding):
    from egomimic.pipeline.core import Pipeline
    from egomimic.pipeline.stages_action_flow import LatentBridgeStage, ConditionalVelocityStage

    class Field(torch.nn.Module):
        def forward(self, x, t, condition, **kwargs):
            return condition[:, :x.shape[-1]].unsqueeze(1).expand_as(x)

    speed = SharedSpeedCondition(None, encoding, condition_dim=4)
    torch.nn.init.constant_(speed.mlp[-1].weight, .1)
    bridge = LatentBridgeStage(samples_per_content=1, condition_key="speed_condition",
                              condition_dropout_probability=0)
    velocity = ConditionalVelocityStage(Field(), num_inference_steps=2,
                                         inference_condition_key="speed_condition")
    graph = Pipeline([speed, bridge, velocity])
    b = {"condition": torch.zeros(2, 4), "retiming_rate": torch.tensor([[1.], [2.]]),
         "action_flow/clean_latent": torch.zeros(2, 2, 4), "sampler/noise": torch.zeros(2, 2, 4)}
    for mode in ("train", "inference"):
        _, excluded = graph.plan(b.keys(), mode)
        assert all(missing == ["<train-only>"] for _, missing in excluded), excluded
        result = graph.execute(b, mode=mode)
        key = "action_flow/predicted_velocity" if mode == "train" else "action_flow/generated_latent"
        assert not torch.equal(result[key][0], result[key][1])
