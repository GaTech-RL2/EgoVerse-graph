"""Thin Lightning integration for the discrete Gaussian likelihood graph.

No FM, clean-reconstruction, Euler, or decoder-JVP diagnostics are borrowed.
The existing evaluator still measures actual generated normalized/native errors
and EnergyScore@32. Train/MSE is the noisy level-one boundary mean's MSE.
"""

from __future__ import annotations

import hashlib
import json

import torch

from egomimic.pl_utils.pl_model import ModelWrapper


def _json_hash(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()


class ActionFlowLikelihoodModelWrapper(ModelWrapper):
    def __init__(
        self, *, action_flow_method=None, gradient_telemetry_cadence=None, **kwargs
    ):
        super().__init__(**kwargs)
        tree = getattr(self.hparams, "config_tree", None)
        cfg = {} if tree is None else self._as_config(tree).model
        method = action_flow_method or cfg.get(
            "action_flow_method", "gaussian_bridge_likelihood"
        )
        if method != "gaussian_bridge_likelihood":
            raise ValueError("likelihood wrapper requires gaussian_bridge_likelihood")
        self.action_flow_method = method
        cadence = gradient_telemetry_cadence
        if cadence is None:
            cadence = cfg.get("gradient_telemetry_cadence", 100)
        if isinstance(cadence, bool) or not isinstance(cadence, int) or cadence < 0:
            raise ValueError("gradient_telemetry_cadence must be a nonnegative integer")
        self.gradient_telemetry_cadence = cadence
        self._gradient_route_manifest = None
        self.save_hyperparameters(
            {"action_flow_method": method, "gradient_telemetry_cadence": cadence}
        )

    def _log_telemetry(self, name, value):
        self.log(
            f"Train/ActionFlow/{name}",
            value,
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )

    def _capture_gradient_routes(self, components):
        named = sorted(
            (name, parameter)
            for name, parameter in self.nets.named_parameters(
                prefix="nets", remove_duplicate=True
            )
            if parameter.requires_grad
        )
        if not named:
            raise RuntimeError("likelihood graph has no trainable parameters")
        routes, vectors = {}, {}
        for label, loss in components.items():
            gradients = torch.autograd.grad(
                loss, tuple(p for _, p in named), retain_graph=True, allow_unused=True
            )
            active = {
                i: g.detach().float() for i, g in enumerate(gradients) if g is not None
            }
            if not active:
                raise RuntimeError(
                    f"likelihood {label} has no generative gradient route"
                )
            norm = torch.stack([g.square().sum() for g in active.values()]).sum().sqrt()
            if not bool(torch.isfinite(norm)):
                raise RuntimeError(f"non-finite likelihood {label} gradient")
            self._log_telemetry(f"GradientNorm/{label}", norm)
            self._log_telemetry(
                f"GradientParameterCount/{label}",
                sum(named[i][1].numel() for i in active),
            )
            routes[label] = [
                {
                    "name": named[i][0],
                    "dtype": str(named[i][1].dtype),
                    "numel": named[i][1].numel(),
                    "shape": list(named[i][1].shape),
                }
                for i in active
            ]
            vectors[label] = active
        left, right = "InteriorBridgeNLL", "BoundaryNLL"
        indices = sorted(set(vectors[left]) & set(vectors[right]))
        if not indices:
            raise RuntimeError(
                "interior and boundary likelihood must share a gradient pathway"
            )
        dot = torch.stack(
            [(vectors[left][i] * vectors[right][i]).sum() for i in indices]
        ).sum()
        norms = [
            torch.stack([vectors[label][i].square().sum() for i in indices])
            .sum()
            .sqrt()
            for label in (left, right)
        ]
        denominator = norms[0] * norms[1]
        defined = bool(denominator > 0)
        pair = left + "__" + right
        self._log_telemetry(
            f"GradientCosine/{pair}",
            (dot / denominator).clamp(-1, 1) if defined else dot.new_zeros(()),
        )
        self._log_telemetry(f"GradientCosineDefined/{pair}", float(defined))
        self._log_telemetry(
            f"GradientIntersectionParameterCount/{pair}",
            sum(named[i][1].numel() for i in indices),
        )
        core = {
            "schema_version": 1,
            "routes": routes,
            "route_sha256": {
                label: _json_hash(route) for label, route in routes.items()
            },
            "intersections": {pair: [named[i][0] for i in indices]},
        }
        self._gradient_route_manifest = {**core, "manifest_sha256": _json_hash(core)}

    def _log_prediction_metrics(self, predictions, reference):
        # Aggregate exactly as the generic pipeline objective: equal sources;
        # each source already averages its examples and sampled interior levels.
        for metric, source_values in self._prediction_log_metrics(
            predictions, reference
        ).items():
            for source, value in source_values:
                self.log(
                    f"Train/{metric}/{source}",
                    value,
                    sync_dist=True,
                    on_step=True,
                    on_epoch=True,
                )
            self.log(
                f"Train/{metric}",
                torch.stack([x for _, x in source_values]).mean(),
                sync_dist=True,
                on_step=True,
                on_epoch=True,
            )
        self._log_telemetry("OptimizerStep", float(self.global_step))
        stages = self.model.pipeline.stages
        samples = next(
            s.interior_samples_per_content
            for s in stages
            if hasattr(s, "interior_samples_per_content")
        )
        self._log_telemetry("Compute/FieldForwardCallsPerStep", 1)
        self._log_telemetry("Compute/FieldSampleEquivalentsPerStep", samples + 1)
        self._log_telemetry("Compute/DecoderJVPCallsPerStep", 0)
        if (
            self.gradient_telemetry_cadence
            and (int(self.global_step) + 1) % self.gradient_telemetry_cadence == 0
        ):
            self._capture_gradient_routes(
                {
                    label: torch.stack(
                        [
                            result[f"log/ActionFlow/{label}"]
                            for result in predictions.values()
                        ]
                    ).mean()
                    for label in ("InteriorBridgeNLL", "BoundaryNLL")
                }
            )

    def on_after_backward(self):
        self._log_telemetry(
            "Compute/PeakAllocatedBytes",
            (
                torch.cuda.max_memory_allocated(self.device)
                if self.device.type == "cuda"
                else 0
            ),
        )
        super().on_after_backward()

    def on_before_optimizer_step(self, optimizer):
        if (
            self.gradient_telemetry_cadence
            and (int(self.global_step) + 1) % self.gradient_telemetry_cadence == 0
        ):
            pieces = [
                p.grad.detach().float().square().sum()
                for p in self.parameters()
                if p.grad is not None
            ]
            if pieces:
                self._log_telemetry(
                    "GradientNorm/TotalPreclip", torch.stack(pieces).sum().sqrt()
                )
        super().on_before_optimizer_step(optimizer)

    def on_save_checkpoint(self, checkpoint):
        if self._gradient_route_manifest is not None:
            checkpoint["action_flow_gradient_route_manifest"] = (
                self._gradient_route_manifest
            )
        stages = self.model.pipeline.stages
        reference = next(
            s for s in stages if hasattr(s, "interior_samples_per_content")
        )
        noising = next(
            s
            for s in stages
            if hasattr(s, "schedule") and hasattr(s, "condition_dropout_probability")
        )
        objective = next(s for s in stages if getattr(s, "reduction", None))
        checkpoint["action_flow_likelihood_contract"] = {
            "schema_version": 1,
            "method": self.action_flow_method,
            "num_levels": reference.num_levels,
            "interior_samples_per_content": reference.interior_samples_per_content,
            "sigma_min": noising.sigma_min,
            "sigma_max": noising.sigma_max,
            "rho": noising.rho,
            "tau": objective.tau,
            "reduction": objective.reduction,
            "learned_reference_targets": "attached",
            "sampler": "stochastic_reverse_gaussian_chain_with_action_output_noise",
            "clean_reconstruction_objective": False,
            "latent_fm_objective": False,
        }
