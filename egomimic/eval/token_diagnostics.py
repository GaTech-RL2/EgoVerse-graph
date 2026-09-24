"""Generic token diagnostics consuming a configured provider capability."""

import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from egomimic.eval.eval import Eval, EvaluationDataRequirements
from egomimic.eval.latent_archive import (
    plot_reductions,
    reduce_features,
    safe_name,
    write_archive,
)


class TokenDiagnosticsEval(Eval):
    """Write bounded per-rank token archives, preserving real episode/frame IDs.

    A diagnostic provider owns capture and prediction. Reductions and storage
    consume tensors without discovering concrete stages or attention layouts.
    Distributed ranks emit independent, explicitly labeled archives; none is
    presented as a full-dataset reduction. CSV rebuilds need no model access.
    """

    def __init__(
        self,
        *,
        capability="token_activations",
        methods=("pca", "umap", "pca_umap", "tsne2d"),
        max_tokens_per_source_layer=4096,
        pca_components=50,
        pca_for_downstream=False,
        seed=0,
        save_plots=True,
        color_by="embodiment",
        limit_val_batches=16,
        prediction_evaluator=None,
        sample_id_key="episode_hash",
        frame_index_key="frame_index",
    ):
        if (
            type(max_tokens_per_source_layer) is not int
            or max_tokens_per_source_layer < 1
        ):
            raise ValueError("max_tokens_per_source_layer must be positive")
        if color_by not in ("embodiment", "hash"):
            raise ValueError("color_by must be embodiment or hash")
        self.capability, self.methods = capability, tuple(methods)
        self.max_tokens_per_source_layer = max_tokens_per_source_layer
        self.reduction_options = {
            "pca_components": pca_components,
            "pca_for_downstream": pca_for_downstream,
            "seed": seed,
        }
        self.save_plots, self.color_by = save_plots, color_by
        self.limit_val_batches = limit_val_batches
        self.prediction_evaluator = prediction_evaluator
        self.sample_id_key, self.frame_index_key = sample_id_key, frame_index_key
        self._validation_group = "valid"

    def data_requirements(self):
        requirements = (
            EvaluationDataRequirements()
            if self.prediction_evaluator is None
            else self.prediction_evaluator.data_requirements()
        )
        for name in ("sample_id_key", "frame_index_key"):
            value = getattr(requirements, name)
            if value is not None and value != getattr(self, name):
                raise ValueError(
                    f"Diagnostic and prediction evaluators disagree on {name}"
                )
        return replace(
            requirements,
            sample_id_key=self.sample_id_key,
            frame_index_key=self.frame_index_key,
        )

    def trainer_overrides(self):
        overrides = (
            {}
            if self.prediction_evaluator is None
            else dict(self.prediction_evaluator.trainer_overrides())
        )
        for name, value in {
            "limit_val_batches": self.limit_val_batches,
            "num_sanity_val_steps": 0,
        }.items():
            if name in overrides and overrides[name] != value:
                raise ValueError(
                    f"Diagnostic and prediction evaluators disagree on {name}"
                )
            overrides[name] = value
        return overrides

    def bind_data_context(self, *, normalizer):
        if self.prediction_evaluator is not None:
            self.prediction_evaluator.bind_data_context(normalizer=normalizer)

    def on_validation_start(self):
        self._chunks, self._counts, self._seen, self._observed = {}, {}, {}, {}
        root = Path(self.root_dir()) / "latents"
        root.mkdir(parents=True, exist_ok=True)
        prefix = f"epoch_{self.trainer.current_epoch}_step_{getattr(self.trainer, 'global_step', 0)}_rank_{getattr(self.trainer, 'global_rank', 0)}_"
        self.output_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=root))
        self._spool = tempfile.TemporaryDirectory(prefix="egoverse-latents-")
        if self.prediction_evaluator is not None:
            self.prediction_evaluator.model = self.model
            self.prediction_evaluator.trainer = self.trainer
            self.prediction_evaluator.on_validation_start()

    def _record(self, source, source_batch, layer, activation):
        layer, group = safe_name(layer), safe_name(self._validation_group or "valid")
        key = (group, layer)
        tokens = torch.as_tensor(activation["tokens"]).detach().float().cpu()
        mask = torch.as_tensor(activation["mask"], dtype=torch.bool).cpu()
        if (
            tokens.ndim != 3
            or mask.shape != tokens.shape[:2]
            or not torch.isfinite(tokens).all()
        ):
            raise ValueError(
                f"{layer}: expected finite B,T,D tokens and matching B,T mask"
            )
        hashes, frames = (
            source_batch[self.sample_id_key],
            source_batch[self.frame_index_key],
        )
        if len(hashes) != len(tokens) or len(frames) != len(tokens):
            raise ValueError("Diagnostic identities must match the captured batch")
        seen = self._seen.setdefault(key, set())
        rows, vectors = [], []
        self._observed[key] = self._observed.get(key, 0) + int(mask.sum())
        source_key = (group, str(source), layer)
        capacity = self.max_tokens_per_source_layer - self._counts.get(source_key, 0)
        for sample, token in mask.nonzero().tolist():
            identity = (str(source), str(hashes[sample]), int(frames[sample]), token)
            if identity in seen or len(rows) >= capacity:
                continue
            seen.add(identity)
            rows.append(
                dict(
                    zip(
                        ("embodiment", "video_hash", "frame_idx", "token_idx"), identity
                    )
                )
            )
            vectors.append(tokens[sample, token])
        if not rows:
            return
        chunks = self._chunks.setdefault(key, [])
        path = Path(self._spool.name) / f"{group}_{layer}_{len(chunks):06d}.npz"
        np.savez(path, keys=torch.stack(vectors).numpy())
        path.with_suffix(".json").write_text(json.dumps(rows))
        chunks.append(path)
        self._counts[source_key] = self._counts.get(source_key, 0) + len(rows)

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        del dataloader_idx
        outputs = self.model.run_diagnostic(self.capability, batch)
        if tuple(outputs) != tuple(batch):
            raise ValueError("Diagnostic result sources differ from input sources")
        for source, result in outputs.items():
            if not result["activations"]:
                raise ValueError(f"No activations captured for {source}")
            for layer, activation in result["activations"].items():
                self._record(source, batch[source], layer, activation)
        metrics = {}
        if self.prediction_evaluator is not None:
            self.prediction_evaluator.set_validation_group(self._validation_group)
            metrics = self.prediction_evaluator.evaluate_predictions(
                batch,
                {key: value["predictions"] for key, value in outputs.items()},
                batch_idx,
            )
        return metrics

    def on_validation_end(self):
        try:
            if self.prediction_evaluator is not None:
                self.prediction_evaluator.on_validation_end()
            for (group, layer), chunks in self._chunks.items():
                keys = np.concatenate([np.load(path)["keys"] for path in chunks])
                rows = [
                    row
                    for path in chunks
                    for row in json.loads(path.with_suffix(".json").read_text())
                ]
                coordinates, receipt = reduce_features(
                    keys, methods=self.methods, **self.reduction_options
                )
                receipt.update(
                    capability=self.capability,
                    rank=int(getattr(self.trainer, "global_rank", 0)),
                    scope="rank-local",
                    max_tokens_per_source_layer=self.max_tokens_per_source_layer,
                    observed_tokens=self._observed[(group, layer)],
                    source_count=len({r["embodiment"] for r in rows}),
                )
                directory = self.output_dir / group
                write_archive(
                    directory,
                    layer,
                    keys,
                    rows,
                    reductions=coordinates,
                    receipt=receipt,
                )
                if self.save_plots:
                    plot_reductions(
                        directory, layer, rows, coordinates, color_by=self.color_by
                    )
        finally:
            self._spool.cleanup()
