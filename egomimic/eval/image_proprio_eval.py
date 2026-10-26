"""Held-out prediction metrics and explicit single-image generation validation."""

from collections.abc import Mapping

import torch

from egomimic.eval.eval import Eval, EvaluationDataRequirements


class ImageProprioEval(Eval):
    prediction_key = "image_proprio_prediction"

    def __init__(
        self, objective="visual_proprio", sample_seed=42, pixel_diagnostics=None, active_streams_from_data=False
    ):
        if objective not in (
            "visual_only",
            "visual_proprio",
            "visual_state_action",
            "direct_action",
            "image_generation",
        ):
            raise ValueError("Unknown image/proprio evaluation objective")
        self.objective, self.sample_seed = objective, int(sample_seed)
        self.active_streams_from_data = bool(active_streams_from_data)
        if objective == "image_generation" and self.active_streams_from_data:
            raise ValueError("Image generation requires exactly the image stream")
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
        if self.pixel_diagnostics is not None or self.objective == "image_generation":
            keys += ("pixel_metadata",)
        if self.objective == "image_generation":
            keys += ("language_features",)
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
            raise ValueError("Global validation batch identities must cover every source")
        indices = []
        for item, values in zip(batch.values(), indexed):
            if not isinstance(values, (tuple, list)) or len(values) != len(item["task_kind"]):
                raise ValueError("One global validation batch identity is required per sample")
            if any(type(value) is not int or value < 0 for value in values):
                raise ValueError("Global validation batch identities must be nonnegative integers")
            indices.extend(values)
        if len(set(indices)) != 1:
            raise ValueError("A validation batch must share one global batch identity")
        return indices[0]

    @staticmethod
    def _validate_image_generation(batch, output=None):
        """Validate the declared T1 RGB contract before sampling or display.

        Targets remain references for a paired MSE diagnostic, never observed
        context. Camera and time zero are explicit single-image coordinates.
        """
        from egomimic.eval.pixel_diagnostics import pixel_frame_records

        if not isinstance(batch, Mapping) or not batch:
            raise ValueError("Image generation requires a nonempty source batch")
        if output is not None and (
            not isinstance(output, Mapping) or set(output) != set(batch)
        ):
            raise ValueError("Image generation output sources must match the batch")
        for source, item in batch.items():
            world = item["world"]
            if not isinstance(world, Mapping) or set(world) != {"image"}:
                raise ValueError("Image generation requires exactly the image world")
            image = world["image"]
            value = image["value"]
            if (
                not torch.is_tensor(value)
                or not value.is_floating_point()
                or value.ndim != 3
                or value.shape[0] < 1
                or value.shape[1:] != (100, 768)
                or not torch.isfinite(value).all()
                or (value.abs() > 1.00001).any()
            ):
                raise ValueError("Generation targets require finite B x 100 x 768 RGB in [-1,1]")
            count = value.shape[0]
            for key, expected in (("valid", True), ("target_valid", True), ("observed", False)):
                mask = image[key]
                if (
                    not torch.is_tensor(mask)
                    or mask.shape != value.shape[:2]
                    or mask.dtype != torch.bool
                    or not (mask == expected).all()
                ):
                    raise ValueError("Generation requires fully valid targets and no observed pixels")
            for key in ("time", "index", "camera"):
                coordinate = image[key]
                if (
                    not torch.is_tensor(coordinate)
                    or coordinate.shape != value.shape[:2]
                    or coordinate.dtype == torch.bool
                    or coordinate.is_complex()
                    or not torch.isfinite(coordinate).all()
                    or (coordinate != 0).any()
                    or key == "camera" and coordinate.dtype != torch.long
                    or key == "time" and not coordinate.is_floating_point()
                ):
                    raise ValueError("Generation requires T1 time/index/camera zero")
            xy = image["xy"]
            if (
                not torch.is_tensor(xy)
                or xy.shape != (*value.shape[:2], 2)
                or not xy.is_floating_point()
                or not torch.isfinite(xy).all()
            ):
                raise ValueError("Generation requires a finite complete spatial patch grid")
            kinds = item["task_kind"]
            if not isinstance(kinds, (list, tuple)) or len(kinds) != count or any(
                kind != "image_generation" for kind in kinds
            ):
                raise ValueError("Generation requires only image_generation task kinds")
            language = item["language_features"]
            if (
                not torch.is_tensor(language)
                or language.shape != (count, 1024)
                or not language.is_floating_point()
                or not torch.isfinite(language).all()
            ):
                raise ValueError("Generation requires finite B x 1024 language features")
            metadata = item["pixel_metadata"]
            if not isinstance(metadata, (list, tuple)) or len(metadata) != count:
                raise ValueError("Every generated image requires its own pixel metadata")
            prediction = None
            if output is not None:
                predicted = output[source][ImageProprioEval.prediction_key]
                if not isinstance(predicted, Mapping) or set(predicted) != {"image"}:
                    raise ValueError("Generation predictions require exactly the image stream")
                prediction = predicted["image"]
                if (
                    not torch.is_tensor(prediction)
                    or not prediction.is_floating_point()
                    or prediction.shape != value.shape
                    or not torch.isfinite(prediction).all()
                ):
                    raise ValueError("Generation requires finite predictions for every RGB pixel")
            for row, details in enumerate(metadata):
                if (
                    not isinstance(details, Mapping)
                    or details.get("image_size") != [160, 160]
                    or details.get("patch_size") != 16
                    or details.get("timestamps_seconds") != [0.0]
                    or details.get("frame_indices") != [0]
                    or details.get("time_origin_seconds") != 0.0
                    or not isinstance(details.get("camera_names"), (list, tuple))
                    or len(details.get("camera_names", ())) != 1
                ):
                    raise ValueError("Generation diagnostics require one 160 x 160 P16 image at time/camera zero")
                # Shared diagnostics validate normalization, exact patch-grid
                # coverage and ordering, coordinate binding and every target
                # cell. Predictions are checked above; the shared metric loop
                # checks squared-error overflow before any artifact is written.
                pixel_frame_records(image, None, details, row)

    def _evaluate_batch(self, batch, batch_idx, dataloader_idx=0):
        if self.objective == "image_generation":
            self._validate_image_generation(batch)
        devices = (
            [self.model.device.index or 0] if self.model.device.type == "cuda" else []
        )
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(
                self.sample_seed + self._sampling_index(batch, batch_idx)
                + 100000 * dataloader_idx
            )
            output = self.model.forward_eval(batch)
        if self.objective == "image_generation":
            self._validate_image_generation(batch, output)
        names = {
            "visual_only": ("image",),
            "visual_proprio": ("image", "proprio"),
            "visual_state_action": ("image", "proprio", "action"),
            "direct_action": ("action",),
            "image_generation": ("image",),
        }[self.objective]
        for source, item in batch.items():
            kinds = item["task_kind"]
            selected_names = names
            if self.active_streams_from_data:
                selected_names = tuple(item['world'])
                if (not {'image','proprio','camera'} <= set(selected_names)
                        or not set(selected_names) <= {'image','proprio','camera','action'}
                        or set(output[source][self.prediction_key]) != set(selected_names)):
                    raise ValueError('Actual multimodal prediction keys must match native world streams')
            for name in selected_names:
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
                    allowed_kinds = (("image_generation",) if self.objective == "image_generation"
                                     else ("forecast", "completion", "direct_action"))
                    if kind not in allowed_kinds:
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
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return self.totals
        keys_by_rank = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(keys_by_rank, sorted(self.totals))
        keys = sorted({key for rank_keys in keys_by_rank for key in rank_keys})
        totals = torch.zeros(len(keys), 2, dtype=torch.float64, device=self.model.device)
        for index, key in enumerate(keys):
            if key in self.totals:
                numerator, denominator = self.totals[key]
                totals[index, 0] = numerator
                totals[index, 1] = denominator
        valid = (torch.isfinite(totals).all() & (totals >= 0).all()).to(torch.int64)
        torch.distributed.all_reduce(valid, op=torch.distributed.ReduceOp.MIN)
        if not valid.item():
            raise ValueError("Invalid accumulated validation metric on a distributed rank")
        if keys:
            torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
        batches = torch.tensor(self._validation_batches, dtype=torch.int64, device=self.model.device)
        torch.distributed.all_reduce(batches, op=torch.distributed.ReduceOp.SUM)
        self.global_validation_batches = int(batches.item())
        return {key: (totals[index, 0], totals[index, 1]) for index, key in enumerate(keys)}

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
            success = torch.tensor(int(complete), dtype=torch.int64, device=self.model.device)
            torch.distributed.all_reduce(success, op=torch.distributed.ReduceOp.MIN)
            complete = bool(success.item())
        if publication_error is not None:
            raise publication_error
        if complete:
            self._completed_validation_step = self._validation_step
        return None
