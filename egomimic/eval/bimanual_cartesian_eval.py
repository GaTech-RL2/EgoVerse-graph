"""Cartesian bimanual validation: per-embodiment MSE + on-image overlay videos.

Graph-native evaluator for time-indexed (baseline) cotrain runs on 14-D bimanual
cartesian action chunks. Mirrors ``PlanarActionEval`` on the metric side --
normalized and native MSE, aggregated across embodiments -- and adds the overlay
plumbing the arc-branch evaluator used: predicted vs. ground-truth trajectories
projected through per-embodiment revert transforms.

Episode videos use the configured identity keys and source frame rate. Frames
are spooled per rank, reassembled in frame order, and streamed to h264 on rank
zero. Chunked output is an explicit mode; missing episode metadata never
silently changes the output contract. Finished files are logged under
``Val_video/{embodiment}`` or ``Val_video_{group}/{embodiment}``.

The action space here is 14-D eef-frame [L pose(6), L grip(1), R pose(6), R
grip(1)]; ``viz_func`` per-embodiment partials receive it in native units and
apply the embodiment's ``get_revert_transform_list`` to reproject onto the front
camera before drawing. Both prediction and ground truth are unnormalized before
overlay so the color-graded polylines are directly comparable in metres.

The metric families are:

  * ``Valid/MSE``                 -- normalized action-chunk MSE, macro mean.
  * ``Valid/MSE/{embodiment}``    -- per-embodiment slices of the above.
  * ``Valid/Native_MSE``          -- unnormalized (metres/radians) MSE.
  * ``Valid/Native_MSE/{embod}``  -- per-embodiment slices.

Val group namespacing (``Valid_newtask/...``) follows the same rule the planar
evaluator uses: the default group keeps the bare ``Valid/...`` prefix so runs
overlay on existing charts, and every other group gets a ``Valid_{group}/...``
suffix instead.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from egomimic.eval.arc_metrics import arcmatch_metrics, chunk_metrics, dtw_metrics
from egomimic.eval.video import EvalVideo
from egomimic.pipeline.core import resolve_homogeneous_scalar
from egomimic.pl_utils.pl_data_utils import DEFAULT_VALID_GROUP
from egomimic.rldb.embodiment.embodiment import Embodiment, get_embodiment


def viz_annotation_key(viz_partial) -> str | None:
    """Return the batch key a viz partial wants to overlay, or None.

    ``Yam.viz_gt_preds`` / ``Human.viz_gt_preds`` draw language text when
    ``annotation_key`` is set. Hydra instantiates those as ``functools.partial``
    (``_partial_: true``), so the bound key lives on ``.keywords``.
    """
    keywords = getattr(viz_partial, "keywords", None) or {}
    key = keywords.get("annotation_key")
    if key is None:
        return None
    key = str(key).strip()
    if not key or key.lower() in {"null", "none"}:
        return None
    return key


def overlay_annotation_fields(viz_partial, source_batch: Mapping) -> dict:
    """Copy the configured annotation field onto the overlay batch.

    ``_maybe_log_overlay`` rebuilds a slim batch (image, actions, embodiment,
    intrinsics). Without this copy, ``annotation_key`` on the viz partial
    KeyErrors and the overlay is skipped.
    """
    key = viz_annotation_key(viz_partial)
    if key is None or key not in source_batch:
        return {}
    return {key: source_batch[key]}


class BimanualCartesianEval(EvalVideo):
    """Time-indexed cartesian bimanual val: MSE + optional overlay videos."""

    _validation_group = None

    def __init__(
        self,
        action_key: str = "actions_cartesian",
        obs_pose_key: str = "observations.state.ee_pose",
        image_key: str = "observations.images.front_img_1",
        viz_func: Mapping | None = None,
        revert_transforms: Mapping | None = None,
        # Arc metric families. Off by default here so a plain time-indexed run
        # is unchanged; the baseline arm of an arc ablation turns them ON so
        # both arms land on the same charts.
        arc_metrics: bool = False,
        # D and M for shared-D arcmatch: span = min(L_gt, L_pred, D), then
        # re-tokenize both sides to arcmatch_points. D must match the codec.
        arcmatch_distance: float = 0.40,
        arcmatch_points: int = 32,
        # Legacy knob kept for config/compat. Metrics no longer deinterp to
        # this length; arcmatch prep stays at tokenizer resolution (T=100).
        arc_chunk_rows: int = 45,
        # When True, ARC arcmatch always detokenizes first so codec
        # reconstruction is baked into the score. When False (default), score
        # ARC token waypoints unless L_gt < D. Baseline preds are already
        # time-indexed, so this flag is a no-op for them.
        include_reconstruction_loss: bool = False,
        rot_lever_m: float = 0.1,
        dtw_max_samples: int = 8,
        untokenized_action_key: str = "actions_cartesian_untokenized",
        metric_dt: float = 1.0 / 30.0,
        video_output_dir: str | None = None,
        video_chunk_frames: int = 1000,
        max_episode_frames: int = 6000,
        deterministic_seed: int = 420042,
        limit_val_batches: int | float | None = None,
        pose_metrics: bool = False,
        rkl_samples: int = 1,
        group_options: Mapping | None = None,
        viz_every_n_epochs: int = 1,
        viz_max_batches: int | None = None,
        source_fps: float = 30.0,
        sample_id_key: str = "episode_hash",
        frame_index_key: str = "frame_index",
        complete_video_episodes: bool = False,
        video_mode: str = "episode",
    ):
        self.configure_video(
            source_fps=source_fps,
            sample_id_key=sample_id_key,
            frame_index_key=frame_index_key,
            complete_episodes=complete_video_episodes,
            mode=video_mode,
        )
        self.pose_metrics = pose_metrics
        self.rkl_samples = int(rkl_samples)
        if self.rkl_samples < 1:
            raise ValueError("rkl_samples must be at least one")
        self.group_options = dict(group_options or {})
        self.viz_every_n_epochs = int(viz_every_n_epochs)
        self.viz_max_batches = viz_max_batches
        self.trainer = None
        self.model = None
        self.normalizer = None
        self.action_key = str(action_key)
        if not self.action_key:
            raise ValueError("action_key must be non-empty")
        self.obs_pose_key = str(obs_pose_key)
        self.image_key = str(image_key)
        self.viz_func = dict(viz_func or {})
        # Per-embodiment revert transform lists. Overlays are always drawn in
        # the front-camera frame, so eef-frame actions must be recomposed onto
        # the current EEF pose before projection. Each entry is the ALREADY-
        # instantiated list returned by ``_build_{embodiment}_revert_eef_frame_transform_list``.
        self.revert_transforms = dict(revert_transforms or {})
        self.arc_metrics = bool(arc_metrics)
        self.arcmatch_distance = float(arcmatch_distance)
        self.arcmatch_points = int(arcmatch_points)
        self.arc_chunk_rows = int(arc_chunk_rows)
        self.include_reconstruction_loss = bool(include_reconstruction_loss)
        self.rot_lever_m = float(rot_lever_m)
        self.dtw_max_samples = int(dtw_max_samples)
        self.untokenized_action_key = str(untokenized_action_key)
        self.metric_dt = float(metric_dt)
        if self.arcmatch_points < 2:
            raise ValueError("arcmatch_points must be at least two")
        if self.arcmatch_distance <= 0:
            raise ValueError("arcmatch_distance must be positive")
        if self.rot_lever_m < 0:
            raise ValueError("rot_lever_m must be non-negative")
        self.video_output_dir = (
            None if video_output_dir is None else Path(video_output_dir)
        )
        # Frames per file on the LEGACY chunked path (batches with no
        # ``episode_hash``); episode-aware batches cut on episode boundaries.
        self.video_chunk_frames = int(video_chunk_frames)
        # Safety cap on a single episode. Frames are spooled to private files
        # and streamed into the encoder rather than retaining the corpus in RAM.
        self.max_episode_frames = int(max_episode_frames)
        self.deterministic_seed = int(deterministic_seed)
        # Only present so ``ConfigStore`` overrides (used by fast probes) have a
        # place to land; ``BimanualCartesianEval`` does not override trainer
        # limits by default the way ``PlanarActionEval`` does.
        self._trainer_overrides = {}
        if limit_val_batches is not None:
            self._trainer_overrides["limit_val_batches"] = limit_val_batches
        # All keyed by (group, embodiment_name) so per-group runs don't spill
        # frames into each other.
        self.val_image_buffer: dict = {}
        self.val_counter: dict = {}
        self.val_open_episode: dict = {}
        self.val_written: dict = {}
        # (group, embodiment_name, path) accumulated across the epoch and
        # flushed to WandB in on_validation_end.
        self._written_paths: list = []

    def trainer_overrides(self):
        return dict(self._trainer_overrides)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def bind_data_context(self, *, normalizer):
        self.normalizer = normalizer

    def set_validation_group(self, group_name) -> None:
        """Namespace metrics by the val group Lightning is running."""
        self._validation_group = (
            None if group_name in (None, DEFAULT_VALID_GROUP) else str(group_name)
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _namespaced(self, metrics: dict) -> dict:
        """Prefix ``Valid/...`` keys with the current non-default val group."""
        if self._validation_group is None:
            return metrics
        suffix = self._validation_group
        return {
            (
                f"Valid_{suffix}/{key[len('Valid/') :]}"
                if key.startswith("Valid/")
                else key
            ): value
            for key, value in metrics.items()
        }

    @staticmethod
    def _embodiment(source_batch):
        if "embodiment" not in source_batch:
            raise KeyError("bimanual eval batch is missing embodiment metadata")
        embodiment_id = int(
            resolve_homogeneous_scalar(source_batch["embodiment"], label="embodiment")
        )
        name = get_embodiment(embodiment_id)
        if name is None:
            raise KeyError(f"Unknown bimanual embodiment id {embodiment_id}")
        return embodiment_id, name.lower()

    @staticmethod
    def _cuda_devices(batch):
        return sorted(
            {
                int(value.device.index)
                for source_batch in batch.values()
                for value in source_batch.values()
                if torch.is_tensor(value)
                and value.device.type == "cuda"
                and value.device.index is not None
            }
        )

    def _forward_deterministic(self, batch):
        """Run one prediction with a pinned seed, without leaking RNG state."""
        with torch.random.fork_rng(devices=self._cuda_devices(batch)):
            torch.manual_seed(self.deterministic_seed)
            return self.model.forward_eval(batch)

    def _native(self, normalized, embodiment_id):
        if self.normalizer is None:
            raise RuntimeError("bimanual evaluator data context was not bound")
        return self.normalizer.unnormalize(
            {self.action_key: normalized}, embodiment_id
        )[self.action_key]

    def _native_pose(self, normalized, embodiment_id):
        return self.normalizer.unnormalize(
            {self.obs_pose_key: normalized}, embodiment_id
        )[self.obs_pose_key]

    @staticmethod
    def _deinterpolate(chunk: np.ndarray, rows: int) -> np.ndarray:
        """Reduce a chunk to ``rows`` evenly spaced samples.

        Kept for probes/tests. Production arcmatch no longer uses this: shared-D
        clamps travel by ``arcmatch_distance`` at tokenizer resolution instead.

        Indices are selected rather than re-interpolated, so no new samples are
        invented and the rotation columns are never interpolated twice.
        """
        rows = int(rows)
        if rows <= 0 or chunk.shape[-2] <= rows:
            return chunk
        index = np.linspace(0, chunk.shape[-2] - 1, rows).round().astype(int)
        return chunk[..., index, :]

    def _arc_pred_time_indexed(self, prediction: torch.Tensor, embodiment_id: int):
        """Prediction as a time-indexed (B, T, 14) numpy chunk.

        Baseline: unnormalized pose chunk at tokenizer resolution (no
        deinterp). ArcBimanualCartesianEval detokenizes ARC tokens for DTW /
        chunk families and for the L_gt < D arcmatch branch.
        """
        native = self._native(prediction, embodiment_id).detach().cpu().numpy()
        return native.astype(np.float64, copy=False)

    def _arc_gt_time_indexed(self, source_batch, embodiment_id: int):
        """Ground truth as a time-indexed (B, T, 14) numpy chunk, or None.

        Prefers the chunk the tokenizer preserved when present; otherwise the
        action chunk itself, which for a baseline run IS time-indexed. Kept at
        loader resolution (typically T=100) -- shared-D arcmatch clamps by D
        rather than deinterping to ``arc_chunk_rows``.
        """
        raw = source_batch.get(self.untokenized_action_key)
        key = self.untokenized_action_key
        if raw is None:
            raw, key = source_batch.get(self.action_key), self.action_key
        if raw is None:
            return None
        native = self.normalizer.unnormalize({key: raw.detach()}, embodiment_id)[key]
        return native.detach().cpu().numpy().astype(np.float64, copy=False)

    def _arc_pred_for_arcmatch(
        self,
        prediction: torch.Tensor,
        ground_truth: np.ndarray,
        embodiment_id: int,
    ):
        """Prediction geometry for shared-D arcmatch.

        Baseline: same time-indexed poses as :meth:`_arc_pred_time_indexed`.
        Arc subclass returns token waypoints unless any arm has ``L_gt < D``.
        """
        del ground_truth
        return self._arc_pred_time_indexed(prediction, embodiment_id)

    def _extra_metrics(
        self,
        *,
        label: str,
        embodiment_id: int,
        source_batch,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> dict:
        """Arc-matched, DTW and time-domain families for this source.

        Shared by the baseline and arc evaluators so both land on the SAME
        charts. Arcmatch uses shared span ``min(L_gt, L_pred, D)`` at
        ``arcmatch_points``; ARC stays in waypoint space unless GT travel is
        shorter than D. DTW/chunk still use time-indexed chunks (ARC detok'd).
        """
        del target
        if not self.arc_metrics:
            return {}
        ground_truth = self._arc_gt_time_indexed(source_batch, embodiment_id)
        if ground_truth is None or len(ground_truth) == 0:
            return {}
        arcmatch_pred = self._arc_pred_for_arcmatch(
            prediction, ground_truth, embodiment_id
        )
        if isinstance(arcmatch_pred, list):
            predictions = arcmatch_pred
        else:
            predictions = [sample for sample in arcmatch_pred]
        truth = [sample for sample in ground_truth]

        values: dict[str, float] = {}
        values.update(
            arcmatch_metrics(
                predictions,
                truth,
                num_points=self.arcmatch_points,
                dt=self.metric_dt,
                lever_m=self.rot_lever_m,
                min_distance_unit=self.arcmatch_distance,
            )
        )
        decoded = self._arc_pred_time_indexed(prediction, embodiment_id)
        ti_pred = [sample for sample in decoded]
        values.update(dtw_metrics(ti_pred, truth, max_samples=self.dtw_max_samples))
        rows = min(decoded.shape[-2], ground_truth.shape[-2])
        if rows >= 2:
            values.update(
                chunk_metrics(
                    decoded[:, :rows],
                    ground_truth[:, :rows],
                    lever_m=self.rot_lever_m,
                )
            )
        # On the prediction's device: log_dict(sync_dist=True) all-reduces
        # these, and NCCL has no CPU support -- it raises rather than falling
        # back. The metrics are computed in numpy on the host, so this cast is
        # the one place device placement matters.
        return {
            f"Valid/{name}/{label}": torch.tensor(
                float(value), device=prediction.device
            )
            for name, value in values.items()
        }

    def _viz_source(self, actions: torch.Tensor, embodiment_id: int) -> torch.Tensor:
        """Rows to hand the revert transforms. Identity for a time-indexed run.

        A subclass whose action space is NOT a stack of poses must override
        this. The revert transforms rotate AND translate every row they are
        given, so any row that is not a pose comes out as a position -- see
        ArcBimanualCartesianEval, where row M is a velocity.
        """
        del embodiment_id
        return actions

    def _revert_to_camframe(
        self, *, actions: torch.Tensor, obs_pose: torch.Tensor, embodiment_name: str
    ) -> torch.Tensor | None:
        """Recompose eef-frame actions onto the obs pose and land in cam frame.

        Returns ``None`` when no revert transform is configured for this
        embodiment, which disables the overlay for that source without failing.
        """
        if embodiment_name not in self.revert_transforms:
            return None
        transform_list = self.revert_transforms[embodiment_name]
        if not transform_list:
            # An explicitly empty transform declares actions already in camera
            # coordinates; an absent entry means that conversion is unavailable.
            return actions.detach().cpu().numpy()
        batch = {
            self.action_key: actions.detach().cpu().numpy().astype(np.float32),
            self.obs_pose_key: obs_pose.detach().cpu().numpy().astype(np.float32),
        }
        reverted = Embodiment.apply_transform(batch, list(transform_list))
        return reverted[self.action_key]

    # ------------------------------------------------------------------
    # video buffering / writing
    # ------------------------------------------------------------------

    def _maybe_log_overlay(
        self,
        *,
        embodiment_name: str,
        source_batch: Mapping,
        predictions: Mapping,
        embodiment_id: int,
    ) -> None:
        """Render prediction vs GT overlays and buffer them into the epoch video.

        Every rank renders its own samples into private spool files. The shared
        video layer gathers by episode/frame and encodes only on rank zero.
        """
        viz_partial = self.viz_func.get(embodiment_name)
        if viz_partial is None:
            return
        if self.obs_pose_key not in source_batch:
            # Overlay needs the obs pose to recompose eef-frame actions into
            # cam frame; without it we can't draw a meaningful trajectory.
            return

        pred_normalized = predictions["pred_action"].detach()
        gt_normalized = source_batch[self.action_key].detach()
        if int(pred_normalized.shape[0]) <= 0:
            return

        pred_native = self._native(pred_normalized, embodiment_id).detach().cpu()
        gt_native = self._native(gt_normalized, embodiment_id).detach().cpu()
        # Everything downstream treats these rows as poses, so convert first.
        pred_native = self._viz_source(pred_native, embodiment_id)
        gt_native = self._viz_source(gt_native, embodiment_id)

        # Unnormalize the obs pose and drop an optional T_obs=1 axis. The revert
        # transforms expect per-sample obs of shape (D,), not (T_obs, D).
        obs_pose_normalized = source_batch[self.obs_pose_key].detach()
        obs_pose_native = self._native_pose(obs_pose_normalized, embodiment_id)
        if obs_pose_native.ndim == 3 and obs_pose_native.shape[1] == 1:
            obs_pose_native = obs_pose_native.squeeze(1)
        obs_pose_native = obs_pose_native.detach().cpu()

        pred_camframe = self._revert_to_camframe(
            actions=pred_native,
            obs_pose=obs_pose_native,
            embodiment_name=embodiment_name,
        )
        if pred_camframe is None:
            return
        gt_camframe = self._revert_to_camframe(
            actions=gt_native,
            obs_pose=obs_pose_native,
            embodiment_name=embodiment_name,
        )

        image_slab = source_batch[self.image_key]
        images = image_slab
        if images.ndim == 5:
            # (B, T=1, C, H, W) or (B, C, H, W) — collapse the optional obs-step axis.
            images = images[:, 0]
        images = images.detach().cpu()
        if images.ndim == 4 and images.shape[1] in (1, 3):
            images = images.permute(0, 2, 3, 1)

        # viz_gt_preds looks up predictions[f"{embodiment_name}_{action_key}"]
        # and reads batch[image_key], batch[action_key], batch['embodiment'],
        # batch.get('intrinsics'). Feed the cam-frame actions (numpy) so viz
        # can project them without doing its own revert.
        flat_predictions = {
            f"{embodiment_name}_{self.action_key}": pred_camframe,
        }
        flat_batch = {
            self.image_key: images,
            self.action_key: gt_camframe,
            "embodiment": source_batch["embodiment"].detach().cpu(),
        }
        if "intrinsics" in source_batch:
            flat_batch["intrinsics"] = source_batch["intrinsics"].detach().cpu()
        flat_batch.update(
            overlay_annotation_fields(viz_partial, {**source_batch, **predictions})
        )

        try:
            frames = viz_partial(
                predictions=flat_predictions,
                batch=flat_batch,
            )
        except Exception as exc:  # noqa: BLE001 -- overlay is best-effort, don't die val
            if self.complete_video_episodes:
                raise RuntimeError(
                    f"Complete-episode overlay failed for {embodiment_name}"
                ) from exc
            print(
                f"[BimanualCartesianEval] skipped {embodiment_name} overlay: {exc}",
                flush=True,
            )
            return

        if not isinstance(frames, np.ndarray):
            frames = np.asarray(frames)
        if frames.dtype != np.uint8:
            frames = np.clip(frames, 0, 255).astype(np.uint8)
        if frames.ndim == 3:
            frames = frames[None]

        # torchvision.io.write_video wants uint8 (T, H, W, 3) tensors; keep the
        # per-frame layout the numpy array already has.
        frame_tensor = torch.from_numpy(frames)

        group = self._validation_group or DEFAULT_VALID_GROUP
        buf_key = (group, embodiment_name)
        out_dir = self._group_video_dir(group, embodiment_name)
        os.makedirs(out_dir, exist_ok=True)

        self._record_video_frames(buf_key, frame_tensor, source_batch)

    # ------------------------------------------------------------------
    # main val entrypoint
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        del dataloader_idx
        group = self._validation_group or DEFAULT_VALID_GROUP
        options = self.group_options.get(group, {})
        limit = options.get("limit_batches")
        if limit is not None and batch_idx >= limit:
            return {}
        result = self._forward_deterministic(batch)
        return self.evaluate_predictions(batch, result, batch_idx)

    @torch.inference_mode()
    def evaluate_predictions(self, batch, result, batch_idx=0):
        """Score a declared prediction result, also usable by diagnostic providers."""
        group = self._validation_group or DEFAULT_VALID_GROUP
        options = self.group_options.get(group, {})
        limit = options.get("limit_batches")
        if limit is not None and batch_idx >= limit:
            return {}
        if tuple(result) != tuple(batch):
            raise ValueError("Prediction sources must align with the evaluation batch")

        metrics: dict[str, torch.Tensor] = {}
        normalized_values: list[torch.Tensor] = []
        native_values: list[torch.Tensor] = []
        seen_labels: set[str] = set()

        for source_id, source_batch in batch.items():
            embodiment_id, label = self._embodiment(source_batch)
            if label in seen_labels:
                raise ValueError(f"Duplicate bimanual eval embodiment {label!r}")
            seen_labels.add(label)

            prediction = result[source_id]["pred_action"]
            target = source_batch[self.action_key]
            if prediction.shape != target.shape:
                raise ValueError(
                    "bimanual prediction/target shape mismatch: "
                    f"{tuple(prediction.shape)} != {tuple(target.shape)}"
                )
            normalized_mse = (prediction - target).square().mean()
            native_mse = (
                (
                    self._native(prediction, embodiment_id)
                    - self._native(target, embodiment_id)
                )
                .square()
                .mean()
            )
            metrics[f"Valid/MSE/{label}"] = normalized_mse
            metrics[f"Valid/Native_MSE/{label}"] = native_mse
            normalized_values.append(normalized_mse)
            native_values.append(native_mse)

            metrics.update(
                self._extra_metrics(
                    label=label,
                    embodiment_id=embodiment_id,
                    source_batch=source_batch,
                    prediction=prediction,
                    target=target,
                )
            )

            if self.pose_metrics:
                from egomimic.eval.cartesian_metrics import (
                    cartesian_metrics,
                    sample_metrics,
                )

                native_pred = self._native(prediction, embodiment_id)
                native_target = self._native(target, embodiment_id)
                stem = f"Valid/{label}_{self.action_key}"
                pose_values = cartesian_metrics(native_pred, native_target)
                obs_pose = self._native_pose(
                    source_batch[self.obs_pose_key], embodiment_id
                )
                cam_pred = self._revert_to_camframe(
                    actions=native_pred, obs_pose=obs_pose, embodiment_name=label
                )
                cam_target = self._revert_to_camframe(
                    actions=native_target, obs_pose=obs_pose, embodiment_name=label
                )
                if cam_pred is not None:
                    pose_values.update(
                        {
                            "cam_" + k: v
                            for k, v in cartesian_metrics(
                                cam_pred, cam_target, distribution=False
                            ).items()
                        }
                    )
                count = int(options.get("rkl_samples", self.rkl_samples))
                if count > 1:
                    # Repeated graph inference is available to every stochastic model.
                    # Do not reseed each draw or reuse the deterministic metric sample.
                    samples = torch.stack(
                        [
                            self._native(
                                self.model.forward_eval({source_id: source_batch})[
                                    source_id
                                ]["pred_action"],
                                embodiment_id,
                            )
                            for _ in range(count)
                        ]
                    )
                    pose_values.update(sample_metrics(samples, native_target))
                metrics.update(
                    {
                        stem + "_" + k: torch.as_tensor(v, device=prediction.device)
                        for k, v in pose_values.items()
                    }
                )

            if self._should_viz(batch_idx):
                self._maybe_log_overlay(
                    embodiment_name=label,
                    source_batch=source_batch,
                    predictions=result[source_id],
                    embodiment_id=embodiment_id,
                )

        metrics["Valid/MSE"] = torch.stack(normalized_values).mean()
        metrics["Valid/Native_MSE"] = torch.stack(native_values).mean()

        self.trainer.lightning_module.log_dict(
            self._namespaced(metrics), sync_dist=True, add_dataloader_idx=False
        )
        return self._namespaced(metrics)
