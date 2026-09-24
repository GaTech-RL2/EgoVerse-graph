"""Optional audit against an unmodified, pinned checkout of ColinQiyangLi/dqc.

Run with DQC_REFERENCE=/path/to/dqc pytest tests/test_q_chunking_upstream_parity.py.
The regular test suite does not download external code.
"""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

REFERENCE = os.environ.get("DQC_REFERENCE")
pytestmark = pytest.mark.skipif(not REFERENCE, reason="set DQC_REFERENCE to pinned upstream checkout")


def copy_mlp(torch_net, tree, ensemble_index=None, gradients=False):
    dense = norm = 0
    for layer in torch_net.layers:
        if isinstance(layer, torch.nn.Linear):
            values = tree[f"Dense_{dense}"]
            dense += 1
            pairs = [(layer.weight, "kernel", True), (layer.bias, "bias", False)]
        elif isinstance(layer, torch.nn.LayerNorm):
            values = tree[f"LayerNorm_{norm}"]
            norm += 1
            pairs = [(layer.weight, "scale", False), (layer.bias, "bias", False)]
        else:
            continue
        for parameter, key, transpose in pairs:
            value = np.asarray(values[key])
            if ensemble_index is not None:
                value = value[ensemble_index]
            value = value.T if transpose else value
            if gradients:
                np.testing.assert_allclose(parameter.grad.numpy(), value, atol=2e-5, rtol=3e-4)
            else:
                parameter.data.copy_(torch.from_numpy(value.copy()))


@pytest.mark.parametrize("aggregation", ["mean", "min"])
def test_native_losses_gradients_and_flow_equal_upstream(aggregation):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    assert subprocess.check_output(["git", "-C", REFERENCE, "rev-parse", "HEAD"], text=True).strip() == \
        "df898256a77f3594b54a7268bd5f89915981da35"
    sys.path.insert(0, str(Path(REFERENCE)))
    from agents.dqc import DQCAgent, get_config
    from egomimic.models.q_chunking import DecoupledQChunking
    rng = np.random.RandomState(291)
    batch = {"observations": rng.randn(7, 3).astype("f"),
        "actions": rng.uniform(-1, 1, (7, 2)).astype("f"),
        "high_value_action_chunks": rng.uniform(-1, 1, (7, 10)).astype("f"),
        "high_value_goals": rng.randn(7, 1).astype("f"),
        "high_value_next_observations": rng.randn(7, 3).astype("f"),
        "high_value_backup_horizon": np.array([0, 1, 2, 3, 4, 5, 5], dtype="f"),
        "high_value_rewards": np.array([1, .999, .998001, .997003, .996006, 0, 0], dtype="f"),
        "high_value_masks": np.array([0, 0, 0, 0, 0, 1, 1], dtype="f"),
        "valids": np.ones((7, 5), dtype="f")}
    config = get_config()
    config.policy_chunk_size, config.backup_horizon = 2, 5
    config.actor_hidden_dims = config.value_hidden_dims = (16, 16)
    config.q_agg, config.kappa_b, config.kappa_d = aggregation, .93, .8
    reference = DQCAgent.create(19, {k: jnp.asarray(v) for k, v in batch.items()}, config)
    model = DecoupledQChunking(3, 1, 2, 5, 4, hidden_dims=(16, 16),
                               q_agg=aggregation, kappa_b=.93, kappa_d=.8)
    for name in ["action_critic", "target_action_critic", "chunk_critic", "actor_bc", "value"]:
        net = getattr(model, name)
        tree = reference.network.params["modules_" + name]["mlp" if name == "actor_bc" else "value_net"]
        if hasattr(net, "members"):
            for index, member in enumerate(net.members):
                copy_mlp(member, tree, index)
        else:
            copy_mlp(net, tree)
    key = jax.random.PRNGKey(913)
    _, actor_rng, _, _ = jax.random.split(key, 4)
    _, noise_rng, time_rng, _ = jax.random.split(actor_rng, 4)
    noise = np.asarray(jax.random.normal(noise_rng, (7, 4)))
    times = np.asarray(jax.random.uniform(time_rng, (7, 1)))
    native = torch.from_numpy(batch["high_value_action_chunks"][:, :4].copy())
    losses = model.losses({k: torch.from_numpy(v.copy()) for k, v in batch.items()}, native,
                         noise=torch.from_numpy(noise.copy()), times=torch.from_numpy(times.copy()))
    (ref_loss, info), gradients = jax.value_and_grad(reference.total_loss, argnums=1, has_aux=True)(
        {k: jnp.asarray(v) for k, v in batch.items()}, reference.network.params, key)
    total = sum(losses.values())
    np.testing.assert_allclose(float(total), float(ref_loss), atol=2e-6, rtol=2e-6)
    for name, ref in [("chunk_critic", "chunk_critic/critic_loss"),
                      ("action_critic", "action_critic/critic_loss"),
                      ("value", "action_critic/value_loss"), ("actor_bc", "actor/bc_flow_loss")]:
        np.testing.assert_allclose(float(losses[name]), float(info[ref]), atol=2e-6, rtol=2e-6)
    total.backward()
    for name in ["action_critic", "chunk_critic", "actor_bc", "value"]:
        net = getattr(model, name)
        tree = gradients["modules_" + name]["mlp" if name == "actor_bc" else "value_net"]
        if hasattr(net, "members"):
            for index, member in enumerate(net.members):
                copy_mlp(member, tree, index, gradients=True)
        else:
            copy_mlp(net, tree, gradients=True)
    # Same latent noise, same Euler flow decoder, independent of PRNG backend.
    obs = torch.from_numpy(batch["observations"])
    flow = torch.from_numpy(noise.copy())
    with torch.no_grad():
        for step in range(model.flow_steps):
            flow += model.actor_bc(obs, flow, torch.full((7, 1), step / model.flow_steps)) / model.flow_steps
    expected = reference.compute_flow_actions(jnp.asarray(batch["observations"]), jnp.asarray(noise))
    np.testing.assert_allclose(flow.clamp(-1, 1).numpy(), np.asarray(expected), atol=2e-5, rtol=2e-4)
