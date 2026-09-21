"""Goal-conditioned Decoupled Q-chunking networks and objectives.

PyTorch port of ColinQiyangLi/dqc at df898256a77f3594b54a7268bd5f89915981da35.
See third_party/dqc/LICENSE. Task choices belong in Hydra recipes.
"""
from __future__ import annotations

import copy
import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, layer_norm=True):
        super().__init__()
        layers = []
        for width in hidden_dims:
            layers += [nn.Linear(input_dim, width), nn.GELU(approximate="tanh")]
            if layer_norm:
                layers.append(nn.LayerNorm(width, eps=1e-6))
            input_dim = width
        layers.append(nn.Linear(input_dim, output_dim))
        self.layers = nn.Sequential(*layers)
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, *values):
        return self.layers(torch.cat(values, dim=-1))


class Ensemble(nn.Module):
    def __init__(self, count, **kwargs):
        super().__init__()
        self.members = nn.ModuleList([MLP(**kwargs) for _ in range(count)])

    def forward(self, *values):
        return torch.stack([net(*values).squeeze(-1) for net in self.members])


class DecoupledQChunking(nn.Module):
    """Long native-action critic, distilled policy critic, and flow BC actor.

    Only the policy representation is supplied by the codec. With the long
    critic disabled, the action critic receives TD targets directly. All
    Bellman rewards and discount exponents remain in native environment time.
    """
    def __init__(self, observation_dim, goal_dim, action_dim, backup_horizon,
                 policy_dim, hidden_dims=(1024, 1024, 1024, 1024), num_qs=2,
                 layer_norm=True, actor_layer_norm=True, discount=0.999,
                 tau=0.005, kappa_b=0.9, kappa_d=0.5, q_agg="mean",
                 use_chunk_critic=True, flow_steps=10, best_of_n=32):
        super().__init__()
        if q_agg not in {"mean", "min"}:
            raise ValueError("q_agg must be mean or min")
        if not (0 < discount <= 1 and 0 <= tau <= 1
                and 0 < kappa_b < 1 and 0 < kappa_d < 1):
            raise ValueError("invalid discount, tau, or expectile/quantile")
        self.discount, self.tau = float(discount), float(tau)
        self.kappa_b, self.kappa_d, self.q_agg = kappa_b, kappa_d, q_agg
        self.flow_steps, self.best_of_n = int(flow_steps), int(best_of_n)
        self.policy_dim = int(policy_dim)
        self.use_chunk_critic = bool(use_chunk_critic)
        args = dict(output_dim=1, hidden_dims=hidden_dims, layer_norm=layer_norm)
        sg = int(observation_dim) + int(goal_dim)
        self.action_critic = Ensemble(num_qs, input_dim=sg + policy_dim, **args)
        self.target_action_critic = copy.deepcopy(self.action_critic)
        self.target_action_critic.requires_grad_(False)
        self.value = MLP(input_dim=sg, **args)
        self.chunk_critic = (Ensemble(num_qs, input_dim=sg + backup_horizon * action_dim,
                                     **args) if self.use_chunk_critic else None)
        self.actor_bc = MLP(observation_dim + policy_dim + 1, policy_dim,
                            hidden_dims, actor_layer_norm)

    def aggregate(self, q):
        return q.mean(0) if self.q_agg == "mean" else q.min(0).values

    @torch.no_grad()
    def update_target_before_optimizer(self):
        # Upstream uses pre-optimizer online parameters in its Polyak update.
        for target, online in zip(self.target_action_critic.parameters(),
                                  self.action_critic.parameters(), strict=True):
            target.lerp_(online, self.tau)

    @torch.no_grad()
    def prediction_metrics(self, *, action_q, chunk_q, target_q, value, next_value, backup):
        """Scalar diagnostics on replay actions, in probability space.

        Reuse the loss pass's predictions without extra forwards or RNG draws.
        Ensemble distributions use the same aggregation as the value target.
        """
        predictions = {"q_action": self.aggregate(action_q), "q_target": target_q,
                       "v": value, "v_next": next_value, "bellman_target": backup}
        if chunk_q is not None:
            predictions["q_chunk"] = self.aggregate(chunk_q)
        metrics = {}
        for name, prediction in predictions.items():
            prediction = prediction.detach()
            metrics.update({f"{name}/mean": prediction.mean(),
                            f"{name}/std": prediction.std(unbiased=False),
                            f"{name}/min": prediction.min(),
                            f"{name}/max": prediction.max()})
        # BCE includes target entropy, so even a perfectly fitted moving target
        # need not have a decreasing or zero BCE. xlogy handles exact 0 and 1.
        metrics["bellman_target/entropy"] = -(
            torch.special.xlogy(backup, backup)
            + torch.special.xlogy(1 - backup, 1 - backup)).mean()
        for name, q in [("q_action", action_q), ("q_chunk", chunk_q)]:
            if q is not None:
                metrics[f"{name}/ensemble_std"] = q.std(dim=0, unbiased=False).mean()
        return metrics

    def losses(self, batch, policy_actions, *, noise=None, times=None, metrics=None):
        """Return only optimization losses; optionally fill detached diagnostics."""
        obs, goals = batch["observations"], batch["high_value_goals"]
        native = batch["high_value_action_chunks"].flatten(1)
        with torch.no_grad():
            next_v = self.value(batch["high_value_next_observations"], goals).squeeze(-1).sigmoid()
            backup = (batch["high_value_rewards"] + self.discount **
                      batch["high_value_backup_horizon"] * batch["high_value_masks"] * next_v).clamp(0, 1)
        result = {}
        if self.chunk_critic is not None:
            chunk_logits = self.chunk_critic(obs, goals, native)
            result["chunk_critic"] = F.binary_cross_entropy_with_logits(
                chunk_logits, backup.expand_as(chunk_logits))
            policy_target = chunk_logits.detach().sigmoid()
        else:
            policy_target = backup[None].expand(len(self.action_critic.members), -1)
        logits = self.action_critic(obs, goals, policy_actions)
        action_q = logits.detach().sigmoid()
        weights = torch.where(policy_target >= action_q,
                              self.kappa_d, 1 - self.kappa_d)
        # Replay admits only complete native backup windows: no flattened-action
        # indexing into temporal masks (an upstream indexing error is inert there).
        valid = batch.get("policy_valid", torch.ones_like(backup))
        result["action_critic"] = (weights * F.binary_cross_entropy_with_logits(
            logits, policy_target, reduction="none") * valid).mean()
        with torch.no_grad():
            target_q = self.aggregate(self.target_action_critic(obs, goals, policy_actions).sigmoid())
            # Clamp only to prevent infinities when sigmoid saturates in float32.
            target_logit = torch.logit(target_q.clamp(1e-7, 1 - 1e-7))
        v_logits = self.value(obs, goals).squeeze(-1)
        value = v_logits.detach().sigmoid()
        v_weights = torch.where(target_q >= value,
                                self.kappa_b, 1 - self.kappa_b)
        result["value"] = (v_weights * (v_logits - target_logit).abs() * valid).mean()
        noise = torch.randn_like(policy_actions) if noise is None else noise
        times = torch.rand_like(policy_actions[:, :1]) if times is None else times
        mixed = (1 - times) * noise + times * policy_actions
        prediction = self.actor_bc(obs, mixed, times)
        result["actor_bc"] = ((prediction - (policy_actions - noise)).square().mean(-1) * valid).mean()
        if metrics is not None:
            metrics.update(self.prediction_metrics(action_q=action_q,
                chunk_q=policy_target if self.chunk_critic is not None else None,
                target_q=target_q, value=value, next_value=next_v, backup=backup))
        return result

    @torch.no_grad()
    def sample(self, observations, goals, *, generator=None):
        batch_size = observations.shape[0]
        obs = observations[:, None].expand(-1, self.best_of_n, -1).flatten(0, 1)
        goal = goals[:, None].expand(-1, self.best_of_n, -1).flatten(0, 1)
        actions = torch.randn((len(obs), self.policy_dim), device=obs.device,
                              dtype=obs.dtype, generator=generator)
        for step in range(self.flow_steps):
            t = torch.full_like(actions[:, :1], step / self.flow_steps)
            actions = actions + self.actor_bc(obs, actions, t) / self.flow_steps
        actions = actions.clamp(-1, 1)
        scores = self.aggregate(self.action_critic(obs, goal, actions)).view(batch_size, self.best_of_n)
        actions = actions.view(batch_size, self.best_of_n, self.policy_dim)
        return actions[torch.arange(batch_size, device=actions.device), scores.argmax(-1)]
