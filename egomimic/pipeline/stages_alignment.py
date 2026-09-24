"""Configured cross-source representation objectives and loss schedules."""

from __future__ import annotations

import math

import torch

from egomimic.pipeline.core import Stage


class RepresentationAlignment(Stage):
    """Sinkhorn divergence between two declared sets of feature vectors.

    Optional action supervision discounts the cost of nearest trajectory pairs.
    Action columns and feature keys are explicit: this stage has no knowledge
    of models, source registries, or robot layouts. The tensorized backend keeps
    the custom, sample-indexed cost well-defined; it cannot use KeOps clustering.
    """

    train_only = True

    def __init__(
        self,
        left_features,
        right_features,
        *,
        output_key="alignment/loss",
        supervision=None,
        left_actions=None,
        right_actions=None,
        left_action_indices=None,
        right_action_indices=None,
        match_weight=0.5,
        blur=0.05,
        truncate=18,
        dtw_gamma=0.1,
        cost_layout="pairwise",
    ):
        super().__init__()
        if supervision not in (None, "mse", "soft_dtw"):
            raise ValueError("supervision must be null, mse, or soft_dtw")
        for label, value in (
            ("match_weight", match_weight),
            ("blur", blur),
            ("dtw_gamma", dtw_gamma),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{label} must be finite and positive")
        self.left_features, self.right_features = left_features, right_features
        self.left_actions, self.right_actions = left_actions, right_actions
        self.left_action_indices = tuple(left_action_indices or ())
        self.right_action_indices = tuple(right_action_indices or ())
        self.output_key = str(output_key)
        self.supervision = supervision
        if cost_layout not in ("pairwise", "legacy_broadcast"):
            raise ValueError("cost_layout must be pairwise or legacy_broadcast")
        self.cost_layout = cost_layout
        self.match_weight, self.blur, self.truncate = match_weight, blur, truncate
        reads = [left_features, right_features]
        if supervision is not None:
            if (
                not left_actions
                or not right_actions
                or not self.left_action_indices
                or len(self.left_action_indices) != len(self.right_action_indices)
            ):
                raise ValueError(
                    "Supervision requires two action keys and matching explicit column selections"
                )
            if any(
                type(i) is not int or i < 0
                for i in (*self.left_action_indices, *self.right_action_indices)
            ):
                raise ValueError(
                    "Action column selections must be nonnegative integers"
                )
            reads += [left_actions, right_actions]
        self.reads = tuple(reads)
        self.writes = (self.output_key, "log/alignment_loss", "log/feature_distance")
        # Optional backends are imported only when this stage is configured.
        from geomloss import SamplesLoss

        self._samples_loss = SamplesLoss
        if supervision == "soft_dtw":
            from tslearn.metrics import SoftDTWLossPyTorch

            self.dtw = SoftDTWLossPyTorch(gamma=dtw_gamma)

    @staticmethod
    def _cost(x, y):
        return 0.5 * (x.unsqueeze(-2) - y.unsqueeze(-3)).square().sum(-1)

    @torch.no_grad()
    def _supervision_mask(self, batch, count):
        left = batch[self.left_actions][..., self.left_action_indices]
        right = batch[self.right_actions][..., self.right_action_indices]
        if left.ndim != 3 or right.shape != left.shape or left.shape[0] != count:
            raise ValueError(
                "Alignment supervision needs matched finite (B, T, C) trajectories"
            )
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise ValueError("Alignment supervision contains nonfinite actions")
        if self.supervision == "soft_dtw":
            a = right[:, None].expand(count, count, *right.shape[1:])
            b = left[None, :].expand(count, count, *left.shape[1:])
            distance = self.dtw(
                a.reshape(count * count, *right.shape[1:]),
                b.reshape(count * count, *left.shape[1:]),
            ).reshape(count, count)
        else:
            distance = (right[:, None] - left[None, :]).square().mean((2, 3))
        mask = torch.ones_like(distance)
        mask[torch.arange(count, device=mask.device), distance.argmin(1)] = (
            self.match_weight
        )
        return mask

    def forward(self, batch):
        left, right = batch[self.left_features], batch[self.right_features]
        if left.ndim < 2 or left.shape != right.shape or left.shape[0] < 1:
            raise ValueError(
                "Alignment features must have matching nonempty batch/feature shapes"
            )
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise ValueError("Alignment features contain nonfinite values")
        left, right = left.flatten(1).float(), right.flatten(1).float()
        kwargs = {}
        if self.supervision is not None:
            mask = self._supervision_mask(batch, len(left)).to(left)

            # Preserve the source experiment's sample-index mask for all four
            # Sinkhorn costs, including its debiasing self-cost terms.
            def cost(x, y):
                if self.cost_layout == "legacy_broadcast":
                    # The source's unsqueeze(1)/unsqueeze(0) receives tensors
                    # with a leading singleton batch from GeomLoss. Preserve
                    # that exact broadcasting only for declared reproductions.
                    return (
                        0.5 * (x.unsqueeze(1) - y.unsqueeze(0)).square().sum(-1) * mask
                    )
                return self._cost(x, y) * mask

            kwargs["cost"] = cost
        objective = self._samples_loss(
            "sinkhorn",
            p=2,
            blur=self.blur,
            truncate=self.truncate,
            backend="tensorized",
            **kwargs,
        )
        loss = objective(right, left)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError("Alignment produced a nonfinite or nonscalar loss")
        batch[self.output_key] = loss
        batch["log/alignment_loss"] = loss.detach()
        batch["log/feature_distance"] = (right - left).norm(dim=-1).mean().detach()
        return batch


class ScheduledLossReduction(Stage):
    """Sum configured components, with independently checkpointed warm starts.

    The counter measures training batches (first batch = 1), matching the
    retained source recipe. It is not optimizer/global-step time when gradient
    accumulation is enabled. Call this loss graph once per training batch.
    """

    train_only = True

    def __init__(self, components, denominator_key=None):
        super().__init__()
        self.components = {str(key): dict(spec) for key, spec in components.items()}
        if not self.components:
            raise ValueError("At least one loss component is required")
        for key, spec in self.components.items():
            if (
                type(spec.get("start_batch", 0)) is not int
                or spec.get("start_batch", 0) < 0
            ):
                raise ValueError(f"{key}: start_batch must be a nonnegative integer")
            if not math.isfinite(spec.get("weight", 1.0)):
                raise ValueError(f"{key}: weight must be finite")
        self.denominator_key = denominator_key
        self.reads = (
            *self.components,
            *((denominator_key,) if denominator_key else ()),
        )
        self.writes = ("loss/total", "log/training_batches")
        self.register_buffer("training_batches", torch.zeros((), dtype=torch.long))

    def forward(self, batch):
        terms = []
        next_batch = int(self.training_batches) + 1
        for key, spec in self.components.items():
            value = batch[key]
            if not torch.is_tensor(value) or not torch.isfinite(value).all():
                raise ValueError(f"Loss component {key} must be a finite tensor")
            weight = (
                spec.get("weight", 1.0)
                if next_batch >= spec.get("start_batch", 0)
                else 0.0
            )
            terms.append(value.sum() * weight)
        denominator = batch[self.denominator_key] if self.denominator_key else 1
        if not math.isfinite(float(denominator)) or float(denominator) <= 0:
            raise ValueError("Loss denominator must be finite and positive")
        batch["loss/total"] = sum(terms) / denominator
        self.training_batches.add_(1)
        batch["log/training_batches"] = self.training_batches.detach().clone()
        return batch
