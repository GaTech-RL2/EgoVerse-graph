"""Geometry/time factorization, causal native decoding and sampler contracts."""
import pytest
import torch
from omegaconf import OmegaConf

from egomimic.eval.control_replay import candidate_rank, passes_trace_gates
from egomimic.models.q_chunking import DecoupledQChunking
from egomimic.pipeline.stages_q_chunking import QChunkingStage
from egomimic.rldb.action_codec import ShapeTimeControlChunkCodec


def codec(dim=2, horizon=25, **kwargs):
    return ShapeTimeControlChunkCodec(dim, horizon, kind="arc", distance=100.,
                                     duration_reference=5., **kwargs)


def test_same_shape_different_speed_changes_only_clock_and_duration():
    c = codec(waypoints=16)
    actions = torch.zeros(2, 25, 2)
    actions[0, :2] = torch.tensor([[1., 0.], [0., 1.]])
    actions[1, :4] = torch.tensor([[.5, 0.], [.5, 0.], [0., .5], [0., .5]])
    z, lengths = c.encode(actions, lengths=torch.tensor([2, 4]))
    for factor in ["shape", "extent"]:
        torch.testing.assert_close(z[0, c.factor_slices[factor]], z[1, c.factor_slices[factor]])
    assert not torch.equal(z[0, c.factor_slices["clock"]], z[1, c.factor_slices["clock"]])
    decoded, recovered_lengths = c.decode(z)
    torch.testing.assert_close(recovered_lengths, lengths)
    for i, n in enumerate(lengths):
        torch.testing.assert_close(decoded[i, :n], actions[i, :n], atol=2e-6, rtol=2e-6)
    # Shape and timing can be recombined without modifying either shape factor.
    retimed = z[:1].clone()
    for factor in ["clock", "duration"]:
        retimed[:, c.factor_slices[factor]] = z[1:2, c.factor_slices[factor]]
    back, tau = c.decode(retimed)
    assert tau.item() == 4
    torch.testing.assert_close(back[0, :4], actions[1, :4], atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("mode", ["delta", "control"])
def test_holds_leading_and_trailing_pauses_and_first_native_control(mode):
    c = codec(dim=5, horizon=10, waypoints=16, path_mode=mode,
              action_scale=[.05, .05, .05, .3, 1.], geometry_units=[.05, .05, .05, .3, 1.])
    delta = torch.zeros(1, 10, 5)
    for t, channel in [(1, 0), (3, 1), (5, 3), (7, 4)]:
        delta[0, t, channel] = .5
    actions = delta if mode == "delta" else delta.cumsum(1)
    z, tau = c.encode(actions)
    back, decoded_tau = c.decode(z)
    torch.testing.assert_close(tau, decoded_tau)
    torch.testing.assert_close(back, actions, atol=3e-6, rtol=3e-6)
    # A rotation/grip change survives even while xyz is stationary.
    assert back[0, 5, 3] > .49 and back[0, 7, 4] > .49
    stationary = torch.zeros_like(actions) if mode == "delta" else torch.full_like(actions, .4)
    z, _ = c.encode(stationary)
    back, _ = c.decode(z)
    torch.testing.assert_close(back, stationary, atol=2e-6, rtol=2e-6)


def test_shape_is_independent_of_native_cap_and_unused_future_tail():
    short, long = codec(horizon=10, waypoints=16), codec(horizon=25, waypoints=16)
    actions = torch.zeros(1, 25, 2)
    actions[0, :4] = torch.tensor([[.25, 0.], [.25, 0.], [0., .25], [0., .25]])
    tau = torch.tensor([4])
    zs, _ = short.encode(actions[:, :10], lengths=tau)
    zl, _ = long.encode(actions, lengths=tau)
    for name in ["shape", "extent", "duration"]:
        torch.testing.assert_close(zs[:, short.factor_slices[name]], zl[:, long.factor_slices[name]])
    changed = actions.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]).clamp(-1, 1)
    zc, _ = long.encode(changed, lengths=tau)
    torch.testing.assert_close(zc, zl, rtol=0, atol=0)
    torch.testing.assert_close(short.decode(zs)[0][:, :4], long.decode(zl)[0][:, :4])


@pytest.mark.parametrize("m", [3, 16, 64])
def test_geometry_and_clock_noise_cannot_change_execution_duration(m):
    c = codec(waypoints=m)
    actions = torch.full((3, 25, 2), .1)
    z, tau = c.encode(actions, lengths=torch.tensor([1, 5, 25]))
    g = torch.Generator().manual_seed(14)
    noisy = z + torch.randn(z.shape, generator=g) * .1
    noisy[:, c.factor_slices["duration"]] = z[:, c.factor_slices["duration"]]
    decoded, recovered = c.decode(noisy)
    torch.testing.assert_close(recovered, tau)
    assert torch.isfinite(decoded).all() and decoded.abs().max() <= 1
    # Log duration is allowed outside [-1,1]; global latent clipping is wrong.
    assert z[0, c.factor_slices["duration"]].item() < -1
    assert z[2, c.factor_slices["duration"]].item() > 1
    projected = c.project_latent(noisy)
    torch.testing.assert_close(c.project_latent(projected), projected, atol=2e-6, rtol=2e-6)
    for i, n in enumerate(tau):
        assert torch.equal(projected[i, c.factor_slices["clock"]][n:], torch.full((25-int(n),), -1.))
    for name, span in c.factor_slices.items():
        assert c.factor_weights[span].sum().item() / c.factor_weights.sum().item() == pytest.approx(.25)


def test_codec_projection_happens_before_critic_ranking_without_global_clipping():
    agent = DecoupledQChunking(2, 1, 1, 3, 4, hidden_dims=(8,), flow_steps=2, best_of_n=3)
    seen = []
    handle = agent.action_critic.register_forward_pre_hook(lambda _, args: seen.append(args[-1].clone()))
    output = agent.sample(torch.zeros(2, 2), torch.zeros(2, 1),
                          action_projection=lambda z: torch.full_like(z, 1.5))
    handle.remove()
    assert torch.equal(output, torch.full((2, 4), 1.5))
    assert torch.equal(seen[0], torch.full((6, 4), 1.5))


def test_new_codec_graph_training_keeps_native_td_and_has_finite_gradients():
    c = codec(waypoints=16)
    stage = QChunkingStage(c, observation_dim=3, goal_dim=1, action_dim=2,
                          backup_horizon=25, hidden_dims=(16,), best_of_n=2, flow_steps=2)
    batch = {"observations": torch.randn(4, 3), "high_value_goals": torch.randn(4, 1),
             "high_value_action_chunks": torch.randn(4, 25, 2).clamp(-1, 1),
             "high_value_next_observations": torch.randn(4, 3),
             "high_value_rewards": torch.zeros(4), "high_value_masks": torch.ones(4),
             "high_value_backup_horizon": torch.full((4,), 25.)}
    before = {k: v.clone() for k,v in batch.items()}
    out = stage.execute(batch, mode="train")
    sum(v for k,v in out.items() if k.startswith("loss/")).backward()
    for k,v in before.items():
        torch.testing.assert_close(out[k], v, rtol=0, atol=0)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in stage.agent.actor_bc.parameters())
    prediction = stage.execute({"observations": before["observations"], "goals": before["high_value_goals"]}, mode="inference")
    assert prediction["pred_action"].shape == (4, 25, 2)
    assert ((prediction["action_lengths"] >= 1) & (prediction["action_lengths"] <= 25)).all()


def test_fidelity_ranking_never_prefers_compression_over_reconstruction():
    cfg = OmegaConf.create({"selection_objective": "fidelity", "gates": {
        "action_rmse_p90": .02, "mean_native_steps": 1.}})
    small = {"action_rmse_p90": .015, "action_rmse_mean": .01,
             "mean_native_steps": 5., "scalar_compression_ratio": 2.}
    large = {**small, "action_rmse_p90": .003, "action_rmse_mean": .002, "scalar_compression_ratio": .1}
    assert passes_trace_gates(large, cfg)
    assert candidate_rank(large, cfg) < candidate_rank(small, cfg)
