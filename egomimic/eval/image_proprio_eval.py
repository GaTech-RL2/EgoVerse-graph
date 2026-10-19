"""Held-out prediction metrics, kept separate for states and action commands."""

import torch

from egomimic.eval.eval import Eval, EvaluationDataRequirements


class ImageProprioEval(Eval):
    prediction_key = "image_proprio_prediction"

    def __init__(
        self,
        objective="visual_proprio",
        sample_seed=42,
        pixel_diagnostics=None,
        prediction_streams=None,
    ):
        if objective not in (
            "visual_only",
            "visual_proprio",
            "visual_state_action",
            "direct_action",
        ):
            raise ValueError("Unknown image/proprio evaluation objective")
        self.objective, self.sample_seed = objective, int(sample_seed)
        self.prediction_streams = (
            None if prediction_streams is None else tuple(prediction_streams)
        )
        if self.prediction_streams is not None and (
            not self.prediction_streams
            or len(set(self.prediction_streams)) != len(self.prediction_streams)
            or set(self.prediction_streams) - {"image", "proprio", "action"}
            or (objective != "direct_action" and self.prediction_streams[0] != "image")
        ):
            raise ValueError("Declare unique supported prediction_streams")
        self.totals = {}
        self._completed_validation_step = None
        self._validation_step = None
        self._validation_batches = 0
        self.global_validation_batches = 0
        self._global_totals = {}
        self._metrics_logged = False
        self._validation_failed = False
        self._validation_sanity = False
        self.pixel_diagnostics = None
        if pixel_diagnostics is not None:
            from egomimic.eval.pixel_diagnostics import PixelPredictionDiagnostics

            self.pixel_diagnostics = PixelPredictionDiagnostics(
                **dict(pixel_diagnostics)
            )

    def bind_data_context(self, *, normalizer):
        if normalizer is not None:
            raise ValueError("Image/proprio evaluator expects data-owned normalization")

    def data_requirements(self):
        keys = ("world", "embodiment", "task_kind")
        if self.pixel_diagnostics is not None:
            keys += ("pixel_metadata",)
        return EvaluationDataRequirements(required_keys=keys)

    def trainer_overrides(self):
        return {}

    def on_validation_start(self):
        self._completed_validation_step = None
        self._validation_step = int(self.model.global_step)
        self._validation_batches = 0
        self.global_validation_batches = 0
        self._global_totals = {}
        self._metrics_logged = False
        self._validation_failed = False
        self._validation_sanity = bool(
            getattr(getattr(self.model, "trainer", None), "sanity_checking", False)
        )
        self.totals = {}
        if self.pixel_diagnostics is not None:
            self.pixel_diagnostics.start()

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        self._completed_validation_step = None
        self._metrics_logged = False
        try:
            output = self._evaluate_batch(batch, batch_idx, dataloader_idx)
        except Exception:
            self._validation_failed = True
            raise
        self._validation_batches += 1
        return output

    @staticmethod
    def _sampling_index(batch, batch_idx):
        """Use the adapter's global batch identity across validation shards."""
        indexed = [item.get("validation_batch_index") for item in batch.values()]
        if all(value is None for value in indexed):
            return int(batch_idx)
        if any(value is None for value in indexed):
            raise ValueError(
                "Global validation batch identities must cover every source"
            )
        indices = []
        for item, values in zip(batch.values(), indexed):
            if not isinstance(values, (tuple, list)) or len(values) != len(
                item["task_kind"]
            ):
                raise ValueError(
                    "One global validation batch identity is required per sample"
                )
            if any(type(value) is not int or value < 0 for value in values):
                raise ValueError(
                    "Global validation batch identities must be nonnegative integers"
                )
            indices.extend(values)
        if len(set(indices)) != 1:
            raise ValueError("A validation batch must share one global batch identity")
        return indices[0]

    def _evaluate_batch(self, batch, batch_idx, dataloader_idx=0):
        devices = (
            [self.model.device.index or 0] if self.model.device.type == "cuda" else []
        )
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(
                self.sample_seed
                + self._sampling_index(batch, batch_idx)
                + 100000 * dataloader_idx
            )
            output = self.model.forward_eval(batch)
        names = (
            self.prediction_streams
            or {
                "visual_only": ("image",),
                "visual_proprio": ("image", "proprio"),
                "visual_state_action": ("image", "proprio", "action"),
                "direct_action": ("action",),
            }[self.objective]
        )
        for source, item in batch.items():
            if self.prediction_streams is not None and set(
                output[source][self.prediction_key]
            ) != set(names):
                raise ValueError(
                    "Evaluator prediction_streams differ from model output"
                )
            kinds = item["task_kind"]
            for name in names:
                stream = item["world"][name]
                mask = (
                    stream["valid"]
                    & stream.get("target_valid", stream["valid"])
                    & ~stream["observed"]
                )
                target = torch.where(mask[..., None], stream["value"], 0)
                prediction = output[source][self.prediction_key][name]
                if prediction.shape != target.shape:
                    raise ValueError(f"Prediction/target shape mismatch for {name}")
                error = (prediction.float() - target.float()).square().mean(-1)
                if not torch.isfinite(error[mask]).all():
                    raise ValueError(f"Non-finite held-out {name} prediction/target")
                metric = (
                    "action_command_mse"
                    if name == "action"
                    else f"{name}_normalized_mse"
                )
                if len(kinds) != error.shape[0]:
                    raise ValueError("One task kind is required per validation sample")
                for kind in sorted(set(kinds)):
                    if kind not in ("forecast", "completion", "direct_action"):
                        raise ValueError(f"Unknown validation task kind: {kind}")
                    rows = torch.tensor(
                        [value == kind for value in kinds], device=mask.device
                    )
                    selected = mask & rows[:, None]
                    numerator = error.masked_fill(~selected, 0).sum().double()
                    denominator = selected.sum().double()
                    key = f"val/{source}/{kind}/{metric}"
                    old = self.totals.get(
                        key, (numerator.new_zeros(()), denominator.new_zeros(()))
                    )
                    self.totals[key] = old[0] + numerator, old[1] + denominator
            if self.pixel_diagnostics is not None:
                self.pixel_diagnostics.record(
                    model=self.model,
                    source=source,
                    item=item,
                    prediction=output[source][self.prediction_key],
                    objective=self.objective,
                    batch_idx=batch_idx,
                    dataloader_idx=dataloader_idx,
                )
        return output

    def _reduce_validation_totals(self):
        """Sum target-weighted errors, including keys absent on some ranks."""
        self.global_validation_batches = self._validation_batches
        if not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            return self.totals
        keys_by_rank = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(keys_by_rank, sorted(self.totals))
        keys = sorted({key for rank_keys in keys_by_rank for key in rank_keys})
        totals = torch.zeros(
            len(keys), 2, dtype=torch.float64, device=self.model.device
        )
        for index, key in enumerate(keys):
            if key in self.totals:
                numerator, denominator = self.totals[key]
                totals[index, 0] = numerator
                totals[index, 1] = denominator
        valid = (torch.isfinite(totals).all() & (totals >= 0).all()).to(torch.int64)
        torch.distributed.all_reduce(valid, op=torch.distributed.ReduceOp.MIN)
        if not valid.item():
            raise ValueError(
                "Invalid accumulated validation metric on a distributed rank"
            )
        if keys:
            torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
        batches = torch.tensor(
            self._validation_batches, dtype=torch.int64, device=self.model.device
        )
        torch.distributed.all_reduce(batches, op=torch.distributed.ReduceOp.SUM)
        self.global_validation_batches = int(batches.item())
        return {
            key: (totals[index, 0], totals[index, 1]) for index, key in enumerate(keys)
        }

    def log_epoch_metrics(self):
        self._metrics_logged = False
        try:
            self._global_totals = self._reduce_validation_totals()
            for key, (numerator, denominator) in self._global_totals.items():
                if (
                    not torch.isfinite(numerator)
                    or not torch.isfinite(denominator)
                    or numerator < 0
                    or denominator < 0
                ):
                    raise ValueError(f"Invalid accumulated validation metric: {key}")
                if denominator > 0:
                    metric = (numerator / denominator).float()
                    if not torch.isfinite(metric):
                        raise ValueError(f"Non-finite validation metric: {key}")
                    self.model.log(key, metric, sync_dist=False)
        except Exception:
            self._validation_failed = True
            raise
        self._metrics_logged = True

    def has_completed_validation(self, global_step):
        """Prove a successful validation/publication at this exact optimizer step."""
        return bool(
            self._completed_validation_step is not None
            and self._completed_validation_step == int(global_step)
        )

    def on_validation_end(self):
        self._completed_validation_step = None
        publication_error = None
        if self.pixel_diagnostics is not None:
            try:
                self.pixel_diagnostics.log_to_wandb(self.model)
            except Exception as error:
                self._validation_failed = True
                publication_error = error
        # Publication must finish before advertising completion. A zero-target
        # or sanity pass cannot suppress the maintained runner's final check.
        complete = (
            self.global_validation_batches > 0
            and self._metrics_logged
            and not self._validation_failed
            and not self._validation_sanity
            and not getattr(
                getattr(self.model, "trainer", None), "sanity_checking", False
            )
            and self._validation_step == int(self.model.global_step)
            and any(denominator > 0 for _, denominator in self._global_totals.values())
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            success = torch.tensor(
                int(complete), dtype=torch.int64, device=self.model.device
            )
            torch.distributed.all_reduce(success, op=torch.distributed.ReduceOp.MIN)
            complete = bool(success.item())
        if publication_error is not None:
            raise publication_error
        if complete:
            self._completed_validation_step = self._validation_step
        return None
