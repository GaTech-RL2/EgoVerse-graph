"""Native LIBERO validation and fixed-bank Action Flow EnergyScore@32."""

import hashlib
import json
from pathlib import Path

import torch

from egomimic.eval.energy_score import energy_score
from egomimic.eval.libero_eval import LiberoActionEvaluator, action_metrics
from egomimic.rldb.zarr.libero_dataset import EMBODIMENT


def native_component_mse(left, right):
    delta = left - right
    return (delta[..., :3].square().mean(dim=(-2, -1)),
            delta[..., 3:6].square().mean(dim=(-2, -1)),
            delta[..., 6:].square().mean(dim=(-2, -1)))


def native_chunk_distance(left, right):
    translation, rotation, gripper = native_component_mse(left, right)
    return (translation.sqrt() + rotation.sqrt() + gripper.sqrt()) / 3


class LiberoActionFlowEvaluator(LiberoActionEvaluator):
    def __init__(
        self,
        energy_sample_count=32,
        energy_seed_bank_path=None,
        energy_seed_bank_sha256=None,
        diagnostic_raw_noise_levels=(0.0, 0.25, 0.5, 0.75, 1.0),
        artifact_root=None,
        artifact_identity=None,
        native_diagnostic_config=None,
    ):
        super().__init__()
        self.artifact_root = artifact_root
        self.artifact_identity = artifact_identity
        self.artifact_batches = []
        self._completed_validation_step = None
        if artifact_root is not None:
            from egomimic.benchmarks.libero.action_flow_artifacts import validate_identity
            validate_identity(dict(artifact_identity))
        # The inner PipelineAlgo has no diagnostic provider. Ask the Action
        # Flow behavior to bind its generic ModelWrapper at validation start.
        self.action_flow_diagnostics_enabled = True
        self.energy_sample_count = int(energy_sample_count)
        if self.energy_sample_count != 32:
            raise ValueError("LIBERO Action Flow requires EnergyScore@32")
        path = Path(str(energy_seed_bank_path))
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != energy_seed_bank_sha256:
            raise ValueError("LIBERO EnergyScore seed-bank SHA mismatch")
        seeds = json.loads(raw)["seeds"]
        if len(seeds) != 32 or len(set(seeds)) != 32:
            raise ValueError("LIBERO EnergyScore needs 32 distinct seeds")
        self.seeds = tuple(int(seed) for seed in seeds)
        self.diagnostic_raw_noise_levels = tuple(
            float(level) for level in diagnostic_raw_noise_levels
        )
        if native_diagnostic_config is None:
            raise ValueError("explicit native same-pass diagnostic profile required")
        from egomimic.eval.action_flow_diagnostics import ActionFlowDiagnostics
        self.shared_diagnostics = ActionFlowDiagnostics(native_diagnostic_config)
        if (self.shared_diagnostics.noise_levels != list(self.diagnostic_raw_noise_levels)
            or self.shared_diagnostics.noise_seeds[0] != self.seeds[0]
            or self.shared_diagnostics.max_samples != 8
            or self.shared_diagnostics.jacobian_samples != 2
            or self.shared_diagnostics.cknna_k != 2
            or self.shared_diagnostics.activation_layer_map != tuple((i,i) for i in range(12))):
            raise ValueError("explicit native8/k2/all12 same-pass diagnostic contract mismatch")

    def _predict(self, batch, seed):
        devices = sorted({
            int(value.device.index)
            for source_batch in batch.values()
            for value in source_batch.values()
            if torch.is_tensor(value)
            and value.device.type == "cuda"
            and value.device.index is not None
        })
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            return self.model.forward_eval(batch)

    @torch.inference_mode()
    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        del dataloader_idx
        if self.group == "valid" and batch_idx == 0:
            self._completed_validation_step = None
        first = self._predict(batch, self.seeds[0])
        prefix = "Valid" if self.group == "valid" else f"Valid_{self.group}"
        for source, values in batch.items():
            target_norm = values["actions"]
            prediction_norm = first[source]["pred_action"]
            target = self.normalizer.unnormalize(
                {"actions": target_norm}, EMBODIMENT
            )["actions"]
            prediction = self.normalizer.unnormalize(
                {"actions": prediction_norm}, EMBODIMENT
            )["actions"]
            metrics = action_metrics(prediction, target)
            metrics["normalized_reconst_mse"] = (
                prediction_norm - target_norm
            ).square().mean()
            if batch_idx == 0:
                sample_norm = torch.stack([prediction_norm.detach()] + [
                    self._predict(batch, seed)[source]["pred_action"].detach()
                    for seed in self.seeds[1:]
                ])
                sample_native = self.normalizer.unnormalize(
                    {"actions": sample_norm}, EMBODIMENT
                )["actions"]

                scores = energy_score(
                    sample_native, target, distance_fn=native_chunk_distance
                )
                metrics.update({
                    "energy_score32_native_equal_components": scores["score"],
                    "energy_accuracy32_native_equal_components": scores["accuracy"],
                    "energy_diversity32_native_equal_components": scores["diversity"],
                })
                diagnostics = self.model.run_diagnostic(
                    "action_flow", batch,
                    raw_noise_levels=self.diagnostic_raw_noise_levels,
                    noise_seed=self.seeds[0],
                    max_samples=min(8, len(target)),
                    jacobian_samples=2,
                    capture_activations=True,
                )[source]
                for diagnostic_key, metric_key in (
                    ("latent/clean", "clean_latent"),
                    ("decoded/reconstruction", "clean_decoded_action_normalized"),
                ):
                    tensor = diagnostics[diagnostic_key]
                    if not torch.is_tensor(tensor) or not bool(torch.isfinite(tensor).all()):
                        raise ValueError(
                            f"non-finite LIBERO Action Flow diagnostic: {diagnostic_key}"
                        )
                    metrics[f"diagnostic_{metric_key}_rms"] = tensor.square().mean().sqrt()
                from egomimic.benchmarks.libero.native_diagnostic_config import assert_native_diagnostic_owners
                assert_native_diagnostic_owners(self.model)
                self.shared_diagnostics.reset()
                def native_equal_blocks_mse(left, right):
                    left = self.normalizer.unnormalize({"actions": left}, EMBODIMENT)["actions"]
                    right = self.normalizer.unnormalize({"actions": right}, EMBODIMENT)["actions"]
                    translation, rotation, gripper = native_component_mse(left, right)
                    return (translation + rotation + gripper) / 3
                shared_analysis = self.shared_diagnostics.analyze_precomputed(
                    diagnostics={source: diagnostics}, batch={source: values},
                    batch_idx=int(batch_idx), rank=int(self.model.global_rank),
                    epoch=int(self.model.current_epoch), global_step=int(self.model.global_step),
                    precision=self.model.trainer.precision,
                    source_labels={source: source}, native_error_fns={source: native_equal_blocks_mse})
                for key, value in shared_analysis["metrics"].items():
                    if not bool(torch.isfinite(value)): raise ValueError(("nonfinite shared diagnostic", key))
                    self.model.log(key, value, batch_size=min(8,len(target)),
                        on_step=False, on_epoch=True, sync_dist=True)
            for key, metric in metrics.items():
                if not bool(torch.isfinite(metric)):
                    raise ValueError(f"non-finite LIBERO validation metric: {key}")
                self.model.log(
                    f"{prefix}/{key}", metric, batch_size=len(target),
                    on_step=False, on_epoch=True, sync_dist=True,
                )
            if self.artifact_root is not None:
                if source != "libero_panda":
                    raise ValueError("native LIBERO artifact source mismatch")
                if batch_idx == 0:
                    self.artifact_batches = []
                raw = target_norm.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
                from egomimic.benchmarks.libero.action_flow_artifacts import write_tensor_payload
                tensors = {"normalized_prediction": prediction_norm, "native_prediction": prediction,
                    "normalized_target": target_norm, "native_target": target}
                if batch_idx == 0:
                    tensors.update(normalized_samples=sample_norm, native_samples=sample_native)
                    # Preserve already computed diagnostic tensors only.
                    def capture(prefix, values):
                        for key, value in values.items():
                            name = prefix + str(key)
                            if torch.is_tensor(value): tensors[name] = value
                            elif isinstance(value, dict): capture(name + "/", value)
                    capture("diagnostic/", diagnostics)
                alignment = {key: values[key] for key in ("episode_hash", "frame_index", "task_uid") if key in values}
                if "episode_hash" not in alignment or "frame_index" not in alignment:
                    raise ValueError("native episode/frame alignment must reach evaluator")
                tensor_payload = write_tensor_payload(self.artifact_root, self.group,
                    int(self.model.global_step), int(batch_idx), tensors, alignment)
                tensor_payload["sample_seed_ids"] = list(self.seeds) if batch_idx == 0 else [self.seeds[0]]
                if batch_idx == 0:
                    tensor_payload["shared_analysis"] = shared_analysis["artifact"]
                if batch_idx == 0:
                    tensor_payload["diagnostic_contract"] = {
                        "raw_noise_levels": list(self.diagnostic_raw_noise_levels),
                        "noise_seed": self.seeds[0], "max_samples": min(8, len(target)),
                        "jacobian_samples": 2, "capture_activations": True,
                        "axes": "source_defined_shapes_preserved_no_axis_guess"}
                self.artifact_batches.append({
                    "batch_index": int(batch_idx), "batch_size": len(target),
                    "normalized_target_sha256": hashlib.sha256(raw).hexdigest(),
                    "tensor_payload": tensor_payload,
                    "metrics": {key: float(value.detach().cpu()) for key, value in metrics.items()},
                })

    def on_validation_start(self):
        self._completed_validation_step = None
        return super().on_validation_start()

    def has_completed_validation(self, global_step):
        """True only after a non-sanity valid pass published its artifacts."""
        return getattr(self, "_completed_validation_step", None) == int(global_step)

    def on_validation_end(self):
        if self.artifact_root is None:
            return super().on_validation_end()
        from egomimic.benchmarks.libero.action_flow_artifacts import canonical_sha, write_artifact
        identity = dict(self.artifact_identity)
        write_artifact(self.artifact_root, {
            "schema": "libero-native-action-flow-metrics/v2",
            "identity": identity, "identity_sha256": canonical_sha(identity),
            "global_step": int(self.model.global_step), "source": "libero_panda",
            "group": self.group, "batches": self.artifact_batches,
            "checkpoint_binding": "global_step_only_requires_scheduled_smoke_checkpoint_SHA_binding",
        })
        if (
            self.group == "valid"
            and self.artifact_batches
            and not getattr(self.trainer, "sanity_checking", False)
        ):
            self._completed_validation_step = int(self.model.global_step)
        self.artifact_batches = []
