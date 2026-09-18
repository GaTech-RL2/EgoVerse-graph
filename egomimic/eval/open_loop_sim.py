"""Episode-level open-loop segment simulation for bimanual cartesian policies.

This evaluator measures a policy over an entire recorded episode rather than
averaging independent action chunks.  Validation samples are observations at
known episode/frame indices.  We cache the prediction made from each
observation, then replay the episode at validation end:

* execute the first ``execute_fraction`` of a baseline control chunk, or the
  corresponding fraction of ARC waypoints;
* compare those control-frequency commands with the ground-truth commands;
* advance to the observation at the resulting frame;
* repeat until the episode ends.

The next observation is the recorded observation at the next boundary.  This
is an oracle-observation open-loop segment rollout: it measures compounding
segment error while keeping the evaluation deterministic and comparable
between baseline and ARC representations.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from egomimic.eval.bimanual_cartesian_eval import (
    BimanualCartesianEval,
    overlay_annotation_fields,
)
from egomimic.eval.video import EvalVideo
from egomimic.pl_utils.pl_data_utils import DEFAULT_VALID_GROUP
from egomimic.rldb.zarr.arc_length_tokenizer import (
    bimanual_arc_token_rows,
    validate_bimanual_velocity_mode,
)

XYZ_COLS = (0, 1, 2, 7, 8, 9)
YPR_COLS = (3, 4, 5, 10, 11, 12)
GRIP_COLS = (6, 13)
PAIRED_COLS = XYZ_COLS + GRIP_COLS


def executed_control_steps(control_horizon: int, execute_fraction: float) -> int:
    """Return the positive control-step prefix executed from each chunk."""

    horizon = int(control_horizon)
    fraction = float(execute_fraction)
    if horizon < 1:
        raise ValueError("control_horizon must be positive")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("execute_fraction must be in (0, 1]")
    return max(1, min(horizon, int(math.ceil(horizon * fraction))))


def truncate_arc_token(
    token: np.ndarray, execute_fraction: float, velocity_mode: str
) -> np.ndarray:
    """Keep the first waypoint fraction of an ARC token.

    ARC waypoints are already uniformly resampled in distance by the tokenizer.
    For ``M=100`` and ``execute_fraction=0.30``, this keeps exactly the first
    30 waypoint rows. ``mean`` has one timing row; the granular
    ``per_waypoint`` and ``duration`` modes have one timing row per waypoint.
    Timing is preserved so detokenization can recover the variable number of
    30 Hz control frames needed to traverse the retained waypoint prefix.
    """

    mode = validate_bimanual_velocity_mode(velocity_mode)
    value = np.asarray(token, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 14:
        raise ValueError(f"ARC token must have shape (rows, 14), got {value.shape}")
    rows = int(value.shape[0])
    granular = mode in ("per_waypoint", "duration")
    M = rows // 2 if granular else rows - 1
    if rows != bimanual_arc_token_rows(M, mode):
        raise ValueError(
            f"ARC token has {rows} rows, inconsistent with M={M} and mode={mode!r}"
        )
    fraction = float(execute_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("execute_fraction must be in (0, 1]")
    waypoint_count = max(2, min(M, int(math.ceil(M * fraction))))
    waypoints = value[:waypoint_count].copy()
    if granular:
        timing = value[M : M + waypoint_count].copy()
    else:
        timing = value[M : M + 1].copy()
    return np.concatenate((waypoints, timing), axis=0)


def arc_prefix_control_steps(
    token: np.ndarray,
    execute_fraction: float,
    velocity_mode: str,
    control_dt: float,
    *,
    max_steps: int | None = None,
) -> int:
    """Recover the control-frame stride for an ARC waypoint prefix."""

    dt = float(control_dt)
    if dt <= 0.0:
        raise ValueError("control_dt must be positive")
    partial = truncate_arc_token(token, execute_fraction, velocity_mode)
    mode = validate_bimanual_velocity_mode(velocity_mode)
    granular = mode in ("per_waypoint", "duration")
    M = len(partial) // 2 if granular else len(partial) - 1
    waypoints = partial[:M]
    timing = partial[M:]
    durations = []
    for xyz_off, xyz_slice in ((0, slice(0, 3)), (7, slice(7, 10))):
        xyz = waypoints[:, xyz_slice]
        interval_arc = np.linalg.norm(np.diff(xyz, axis=0), axis=-1)
        moving = interval_arc > 1e-12
        if not bool(np.any(moving)):
            durations.append(0.0)
            continue
        if mode == "duration":
            interval_duration = timing[:-1, xyz_off]
        elif mode == "per_waypoint":
            interval_rate = np.linalg.norm(timing[:-1, xyz_slice], axis=-1)
            interval_duration = np.full_like(interval_arc, np.inf)
            np.divide(
                interval_arc,
                interval_rate,
                out=interval_duration,
                where=interval_rate > 1e-8,
            )
        else:
            chord = float(np.linalg.norm(np.diff(xyz, axis=0), axis=-1).sum())
            speed = float(np.linalg.norm(timing[0, xyz_slice]))
            interval_duration = np.array(
                [chord / speed if chord > 1e-9 and speed > 1e-9 else np.inf]
            )
            moving = np.ones_like(interval_duration, dtype=bool)
        usable = moving & np.isfinite(interval_duration) & (interval_duration > 0.0)
        if not bool(np.all(usable[moving])):
            duration = math.inf
        else:
            duration = float(interval_duration[moving].sum())
        durations.append(duration)

    duration = max(durations, default=0.0)
    if math.isfinite(duration):
        steps = max(1, int(math.ceil(duration / dt - 1e-9)))
    elif max_steps is not None:
        steps = int(max_steps)
    else:
        raise ValueError("ARC prefix timing does not define a finite replan boundary")
    if max_steps is not None:
        steps = min(steps, int(max_steps))
    return max(1, steps)


class OpenLoopSimEval(BimanualCartesianEval):
    """Compare baseline and ARC policies over complete recorded episodes.

    ``execute_fraction`` is applied in representation space. A baseline uses
    the first prefix of its time-indexed action chunk. An ARC policy keeps the
    requested fraction of its distance-resampled ``M`` waypoints, carries the
    matching timing rows, and recovers the corresponding variable control-frame
    stride before requesting the next observation.

    The evaluator expects validation to contain every frame of each episode,
    with ``episode_hash`` and ``frame_index`` metadata. It accumulates model
    predictions during the normal validation loop and, when visualization is
    configured, buffers rendered frames until each complete episode closes.
    Video output therefore follows the metric unit: one MP4 per episode. All
    episode MP4s are kept on disk, while only the first MP4 for each
    validation-group/embodiment pair is uploaded to W&B per validation loop.
    """

    def __init__(
        self,
        *,
        action_key: str = "actions_cartesian",
        ground_truth_action_key: str = "actions_cartesian_untokenized",
        execute_fraction: float = 0.25,
        control_horizon: int = 100,
        control_dt: float = 1.0 / 30.0,
        action_mode: str = "auto",
        min_distance_unit: float = 0.40,
        resampled_vector_length: int = 100,
        velocity_mode: str = "mean",
        log_step: int | None = None,
        results_path: str | None = None,
        require_episode_start: bool = True,
        limit_val_episodes: int | None = None,
        requires_ordered_validation: bool = True,
        limit_val_batches: int | float | None = None,
        deterministic_seed: int = 420042,
        obs_pose_key: str = "observations.state.ee_pose",
        image_key: str = "observations.images.front_img_1",
        viz_func: Mapping | None = None,
        revert_transforms: Mapping | None = None,
        video_output_dir: str | None = None,
        video_chunk_frames: int = 1000,
        max_episode_frames: int = 6000,
        viz_every_n_epochs: int = 1,
        viz_max_batches: int | None = None,
        **kwargs,
    ):
        mode = str(action_mode)
        if mode not in ("auto", "baseline", "arc"):
            raise ValueError("action_mode must be auto, baseline, or arc")
        if float(control_dt) <= 0:
            raise ValueError("control_dt must be positive")
        validate_bimanual_velocity_mode(velocity_mode)
        self.execute_fraction = float(execute_fraction)
        self.control_horizon = int(control_horizon)
        self.control_dt = float(control_dt)
        self.action_mode = mode
        self.ground_truth_action_key = str(ground_truth_action_key)
        self.min_distance_unit = float(min_distance_unit)
        self.resampled_vector_length = int(resampled_vector_length)
        self.velocity_mode = str(velocity_mode)
        self.log_step = None if log_step is None else int(log_step)
        if self.log_step is not None and self.log_step < 0:
            raise ValueError("log_step must be nonnegative")
        self.results_path = Path(results_path) if results_path else None
        self.require_episode_start = bool(require_episode_start)
        self.limit_val_episodes = (
            None if limit_val_episodes is None else int(limit_val_episodes)
        )
        if self.limit_val_episodes is not None and self.limit_val_episodes < 1:
            raise ValueError("limit_val_episodes must be positive")
        self.requires_ordered_validation = bool(requires_ordered_validation)
        if limit_val_batches is not None:
            raise ValueError(
                "open_loop_sim is episode-based; use limit_val_episodes instead "
                "of limit_val_batches"
            )
        self._records: list[dict[str, Any]] = []
        self.last_results = None
        self._arc_tokenizer = None
        self._metric_device = torch.device("cpu")

        # Reuse the graph evaluator's normalizer binding, deterministic model,
        # embodiment resolution, metric-group namespacing, and episode-aware
        # video buffering. Chunk-level ARC metrics remain deliberately disabled
        # here because this evaluator scores the executed control prefix.
        # Existing ABC experiment blocks are merged into a selected evaluator
        # config by Hydra. Consume their legacy ARC-only knobs so selecting
        # this evaluator does not fail on an unrelated ``action_horizon`` (or
        # accidentally enable chunk/video metrics).
        for ignored_key in (
            "arc_metrics",
            "include_reconstruction_loss",
            "arcmatch_distance",
            "arcmatch_points",
            "arc_chunk_rows",
            "action_horizon",
            "rot_lever_m",
            "dtw_max_samples",
        ):
            kwargs.pop(ignored_key, None)
        super().__init__(
            action_key=action_key,
            obs_pose_key=obs_pose_key,
            image_key=image_key,
            viz_func=viz_func,
            revert_transforms=revert_transforms,
            arc_metrics=False,
            deterministic_seed=deterministic_seed,
            limit_val_batches=None,
            video_output_dir=video_output_dir,
            video_chunk_frames=video_chunk_frames,
            max_episode_frames=max_episode_frames,
            viz_every_n_epochs=viz_every_n_epochs,
            viz_max_batches=viz_max_batches,
            **kwargs,
        )
        self._video_enabled = bool(self.viz_func)
        self.execute_steps = executed_control_steps(
            self.control_horizon, self.execute_fraction
        )

    def on_validation_start(self):
        if self._video_enabled:
            EvalVideo.on_validation_start(self)
        self._video_replan_states = {}
        self._records = []
        self.last_results = None
        if self.model is not None:
            try:
                self._metric_device = next(self.model.parameters()).device
            except StopIteration:
                pass

    def _video_replan_batch(
        self,
        *,
        source_id: str,
        source_batch: Mapping,
        prediction: torch.Tensor,
        embodiment_id: int,
    ) -> tuple[Mapping, torch.Tensor]:
        """Reuse each executed chunk overlay until its replan boundary.

        The underlying validation image advances every frame. The predicted
        and ground-truth trajectory overlays are sampled only at evaluator
        replan frames and held fixed for the decoded execution length. This
        makes the MP4 show exactly the same schedule used by ``_score_episode``.
        """

        batch_size = int(prediction.shape[0])
        episodes = [
            str(value)
            for value in self._batch_values(
                source_batch["episode_hash"], batch_size, "episode_hash"
            )
        ]
        frames = [
            int(value)
            for value in self._batch_values(
                source_batch["frame_index"], batch_size, "frame_index"
            )
        ]
        target_key = (
            self.ground_truth_action_key
            if self.ground_truth_action_key in source_batch
            else self.action_key
        )
        prediction_native = (
            self._native(prediction, embodiment_id).detach().cpu().numpy()
        )
        target_native = (
            self._native_key(source_batch[target_key], target_key, embodiment_id)
            .detach()
            .cpu()
            .numpy()
        )
        if target_native.ndim != 3 or target_native.shape[0] != batch_size:
            raise ValueError(
                "open_loop_sim video ground truth must be batched as (B, T, 14), "
                f"got {target_native.shape}"
            )

        video_predictions = []
        video_targets = []
        video_observations = []
        group = self._validation_group or DEFAULT_VALID_GROUP
        for index, (episode, frame) in enumerate(zip(episodes, frames)):
            state_key = (group, source_id, episode)
            state = self._video_replan_states.get(state_key)
            if state is None or frame >= state["next_frame"]:
                if state is not None and frame > state["next_frame"]:
                    raise RuntimeError(
                        "open_loop_sim video skipped replan frame "
                        f"{state['next_frame']} for episode {episode!r}; got {frame}"
                    )
                _, steps = self._decode_prediction_with_steps(
                    prediction_native[index],
                    max_steps=int(target_native.shape[1]),
                )
                state = {
                    "next_frame": frame + steps,
                    "prediction": prediction[index].detach().clone(),
                    "target": source_batch[target_key][index].detach().clone(),
                    "observation": (
                        source_batch[self.obs_pose_key][index].detach().clone()
                    ),
                }
                self._video_replan_states[state_key] = state
            video_predictions.append(state["prediction"])
            video_targets.append(state["target"])
            video_observations.append(state["observation"])

        video_batch = dict(source_batch)
        video_batch[target_key] = torch.stack(video_targets)
        video_batch[self.obs_pose_key] = torch.stack(video_observations)
        return video_batch, torch.stack(video_predictions)

    def _decoded_video_predictions(
        self, prediction: torch.Tensor, embodiment_id: int, max_steps: int
    ) -> tuple[torch.Tensor, list[int]]:
        """Decode each predicted token/chunk to executed control-frequency rows."""

        native = self._native(prediction, embodiment_id).detach().cpu().numpy()
        decoded_with_steps = [
            self._decode_prediction_with_steps(sample, max_steps=max_steps)
            for sample in native
        ]
        lengths = [steps for _, steps in decoded_with_steps]
        width = max(lengths)
        decoded = []
        for value, steps in decoded_with_steps:
            if steps < width:
                value = np.concatenate(
                    (value, np.repeat(value[-1:], width - steps, axis=0)), axis=0
                )
            decoded.append(value)
        return (
            torch.from_numpy(np.stack(decoded).astype(np.float32, copy=False)),
            lengths,
        )

    def _maybe_log_open_loop_video(
        self,
        *,
        source_id: str,
        source_batch: Mapping,
        prediction: torch.Tensor,
        embodiment_id: int,
        embodiment_name: str,
    ) -> None:
        """Render one frame per validation sample and buffer by episode hash.

        The metric path stores only the executed prefix. The video path uses
        exactly that same decoded prefix for both baseline and ARC predictions,
        and the preserved control-frequency ground truth, so the visualization
        cannot silently show a different trajectory from the reported score.
        """

        if not getattr(self, "_video_enabled", False) or not getattr(
            self.trainer, "is_global_zero", True
        ):
            return
        viz_partial = self.viz_func.get(embodiment_name)
        if viz_partial is None or self.obs_pose_key not in source_batch:
            return
        source_batch, prediction = self._video_replan_batch(
            source_id=source_id,
            source_batch=source_batch,
            prediction=prediction,
            embodiment_id=embodiment_id,
        )
        target_key = (
            self.ground_truth_action_key
            if self.ground_truth_action_key in source_batch
            else self.action_key
        )
        gt_native = self._native_key(
            source_batch[target_key], target_key, embodiment_id
        ).detach().cpu()
        if gt_native.ndim != 3 or gt_native.shape[0] != prediction.shape[0]:
            raise ValueError(
                "open_loop_sim video ground truth must be batched as (B, T, 14), "
                f"got {tuple(gt_native.shape)}"
            )
        pred_native, prefix_lengths = self._decoded_video_predictions(
            prediction, embodiment_id, int(gt_native.shape[1])
        )
        width = int(pred_native.shape[1])
        gt_prefixes = []
        for index, steps in enumerate(prefix_lengths):
            value = gt_native[index, :steps]
            if steps < width:
                value = torch.cat(
                    (value, value[-1:].repeat(width - steps, 1)), dim=0
                )
            gt_prefixes.append(value)
        gt_native = torch.stack(gt_prefixes).to(dtype=pred_native.dtype)

        obs_pose_native = self._native_pose(
            source_batch[self.obs_pose_key], embodiment_id
        ).detach().cpu()
        if obs_pose_native.ndim == 3 and obs_pose_native.shape[1] == 1:
            obs_pose_native = obs_pose_native.squeeze(1)
        pred_camframe = self._revert_to_camframe(
            actions=pred_native,
            obs_pose=obs_pose_native,
            embodiment_name=embodiment_name,
        )
        gt_camframe = self._revert_to_camframe(
            actions=gt_native,
            obs_pose=obs_pose_native,
            embodiment_name=embodiment_name,
        )
        if pred_camframe is None or gt_camframe is None:
            return

        images = source_batch[self.image_key]
        if images.ndim == 5:
            images = images[:, 0]
        images = images.detach().cpu()
        if images.ndim == 4 and images.shape[1] in (1, 3):
            images = images.permute(0, 2, 3, 1)
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
            overlay_annotation_fields(viz_partial, {**source_batch, "source": source_id})
        )
        try:
            frames = viz_partial(predictions=flat_predictions, batch=flat_batch)
        except Exception as exc:  # noqa: BLE001 -- overlays are best effort
            print(
                f"[OpenLoopSimEval] skipped {embodiment_name} overlay: {exc}",
                flush=True,
            )
            return
        frames = np.asarray(frames)
        if frames.dtype != np.uint8:
            frames = np.clip(frames, 0, 255).astype(np.uint8)
        if frames.ndim == 3:
            frames = frames[None]
        frame_tensor = torch.from_numpy(frames)
        hashes = [
            str(value)
            for value in self._batch_values(
                source_batch["episode_hash"], int(frame_tensor.shape[0]), "episode_hash"
            )
        ]
        group = self._validation_group or DEFAULT_VALID_GROUP
        buf_key = (group, embodiment_name)
        out_dir = self._group_video_dir(group, embodiment_name)
        self._buffer_per_episode(buf_key, out_dir, list(frame_tensor), hashes)

    def _log_wandb_videos(self) -> None:
        """Upload only the first episode MP4 for each val loop/panel."""

        if not self._written_paths:
            return
        experiment = self._wandb_logger()
        if experiment is None:
            return
        import wandb  # type: ignore

        first: dict[tuple[str, str], str] = {}
        for group, embodiment_name, path in self._written_paths:
            first.setdefault((group, embodiment_name), path)
        payload = {}
        for (group, embodiment_name), path in first.items():
            prefix = (
                "Val_video" if group == DEFAULT_VALID_GROUP else f"Val_video_{group}"
            )
            payload[f"{prefix}/{embodiment_name}"] = wandb.Video(
                path, fps=self._video_fps(), format="mp4"
            )
        experiment.log(payload, step=self._resolved_log_step())

    def _resolved_log_step(self) -> int:
        log_step = getattr(self, "log_step", None)
        if log_step is not None:
            return int(log_step)
        return int(getattr(self.trainer, "global_step", 0))

    @staticmethod
    def _batch_values(value, batch_size: int, label: str) -> list:
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return [value.item()] * batch_size
            if int(value.shape[0]) != batch_size:
                raise ValueError(
                    f"{label} has batch dimension {tuple(value.shape)}, expected "
                    f"{batch_size}"
                )
            return [item.item() if item.ndim == 0 else item for item in value]
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return [value.item()] * batch_size
            if int(value.shape[0]) != batch_size:
                raise ValueError(
                    f"{label} has batch dimension {value.shape}, expected {batch_size}"
                )
            return [item.item() if np.ndim(item) == 0 else item for item in value]
        if isinstance(value, (list, tuple)):
            if len(value) != batch_size:
                raise ValueError(
                    f"{label} has {len(value)} entries, expected {batch_size}"
                )
            return list(value)
        return [value] * batch_size

    def _native_key(self, value: torch.Tensor, key: str, embodiment_id: int):
        if self.normalizer is None:
            raise RuntimeError("open_loop_sim evaluator data context was not bound")
        return self.normalizer.unnormalize({key: value}, embodiment_id).get(key, value)

    def _is_arc_prediction(self, prediction: np.ndarray) -> bool:
        expected = bimanual_arc_token_rows(
            self.resampled_vector_length, self.velocity_mode
        )
        return prediction.ndim == 2 and prediction.shape == (expected, 14)

    def _decode_prediction_with_steps(
        self, prediction: np.ndarray, *, max_steps: int | None = None
    ) -> tuple[np.ndarray, int]:
        is_arc = self._is_arc_prediction(prediction)
        if self.action_mode == "arc" and not is_arc:
            raise ValueError(
                "open_loop_sim action_mode='arc' received a non-ARC prediction "
                f"with shape {prediction.shape}"
            )
        if self.action_mode == "baseline" and is_arc:
            raise ValueError(
                "open_loop_sim action_mode='baseline' received an ARC prediction"
            )
        if not is_arc:
            if prediction.ndim != 2 or prediction.shape[1] != 14:
                raise ValueError(
                    "baseline open_loop_sim predictions must have shape (T, 14), "
                    f"got {prediction.shape}"
                )
            if prediction.shape[0] < self.execute_steps:
                raise ValueError(
                    f"baseline prediction has only {prediction.shape[0]} control "
                    f"steps, needs {self.execute_steps}"
                )
            steps = self.execute_steps
            if max_steps is not None:
                steps = min(steps, int(max_steps))
            return prediction[:steps].copy(), steps

        if self._arc_tokenizer is None:
            from egomimic.rldb.zarr.arc_length_tokenizer import (
                TokenizeBimanualArcLengthCartesian,
            )

            self._arc_tokenizer = TokenizeBimanualArcLengthCartesian(
                min_distance_unit=self.min_distance_unit,
                resampled_vector_length=self.resampled_vector_length,
                dt=self.control_dt,
                velocity_mode=self.velocity_mode,
            )
        partial = truncate_arc_token(
            prediction, self.execute_fraction, self.velocity_mode
        )
        steps = arc_prefix_control_steps(
            prediction,
            self.execute_fraction,
            self.velocity_mode,
            self.control_dt,
            max_steps=max_steps,
        )
        decoded = self._arc_tokenizer.detokenize(
            partial, action_horizon=steps
        ).astype(np.float64, copy=False)
        return decoded, steps

    def _decode_prediction(self, prediction: np.ndarray) -> np.ndarray:
        return self._decode_prediction_with_steps(prediction)[0]

    def _append_source_records(self, source_id: str, source_batch, prediction):
        if not isinstance(prediction, torch.Tensor) or prediction.ndim != 3:
            raise ValueError(
                f"open_loop_sim expects batched predictions, got {type(prediction)} "
                f"with shape {getattr(prediction, 'shape', None)}"
            )
        batch_size = int(prediction.shape[0])
        self._metric_device = prediction.device
        if "episode_hash" not in source_batch or "frame_index" not in source_batch:
            raise KeyError(
                "open_loop_sim requires episode_hash and frame_index in every "
                f"validation source ({source_id!r})"
            )
        embodiment_id, label = self._embodiment(source_batch)
        episode_hashes = self._batch_values(
            source_batch["episode_hash"], batch_size, "episode_hash"
        )
        frame_indices = self._batch_values(
            source_batch["frame_index"], batch_size, "frame_index"
        )
        pred_native = self._native(prediction, embodiment_id).detach().cpu().numpy()
        target_key = (
            self.ground_truth_action_key
            if self.ground_truth_action_key in source_batch
            else self.action_key
        )
        target_value = self._native_key(
            source_batch[target_key], target_key, embodiment_id
        )
        target_native = target_value.detach().cpu().numpy()
        if target_native.ndim != 3 or target_native.shape[0] != batch_size:
            raise ValueError(
                f"open_loop_sim ground truth {target_key!r} must be batched, got "
                f"{target_native.shape}"
            )
        for index in range(batch_size):
            episode = str(episode_hashes[index])
            frame = int(frame_indices[index])
            if not episode or frame < 0:
                raise ValueError(
                    f"invalid open_loop_sim episode/frame: {episode!r}/{frame}"
                )
            prediction_value = pred_native[index]
            self._records.append(
                {
                    "group": self._validation_group or DEFAULT_VALID_GROUP,
                    "source": str(source_id),
                    "label": label,
                    "episode": episode,
                    "frame": frame,
                    # ARC keeps its complete token and native ground-truth
                    # window because its timing payload determines a variable
                    # control-frame stride at episode replay time.
                    "prediction": np.asarray(prediction_value, dtype=np.float32),
                    "ground_truth": np.asarray(target_native[index], dtype=np.float32),
                }
            )

    @torch.inference_mode()
    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        del dataloader_idx
        result = self._forward_deterministic(batch)
        for source_id, source_batch in batch.items():
            prediction = result[source_id]["pred_action"]
            self._append_source_records(source_id, source_batch, prediction)
            if getattr(self, "_video_enabled", False) and self._should_viz(batch_idx):
                embodiment_id, embodiment_name = self._embodiment(source_batch)
                self._maybe_log_open_loop_video(
                    source_id=source_id,
                    source_batch=source_batch,
                    prediction=prediction,
                    embodiment_id=embodiment_id,
                    embodiment_name=embodiment_name,
                )
        return {}

    @staticmethod
    def _merge_records(states: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        unique: dict[tuple[str, str, str, int], dict[str, Any]] = {}
        for state in states:
            for record in state:
                key = (
                    record.get("group", DEFAULT_VALID_GROUP),
                    record["source"],
                    record["episode"],
                    int(record["frame"]),
                )
                unique.setdefault(key, record)
        return list(unique.values())

    def _all_records(self) -> list[dict[str, Any]]:
        local = self._records
        if not dist.is_available() or not dist.is_initialized():
            return self._merge_records([local])
        states = [None] * dist.get_world_size()
        dist.all_gather_object(states, local)
        return self._merge_records(states)

    def _score_episode(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        records = sorted(records, key=lambda record: int(record["frame"]))
        frames = [int(record["frame"]) for record in records]
        if len(set(frames)) != len(frames):
            raise RuntimeError("open_loop_sim received duplicate episode frames")
        if self.require_episode_start and frames[0] != 0:
            raise RuntimeError(
                f"open_loop_sim requires complete episodes from frame 0; "
                f"episode {records[0]['episode']!r} starts at frame {frames[0]}"
            )
        expected = set(range(frames[0], frames[-1] + 1))
        missing = sorted(expected.difference(frames))
        if missing:
            raise RuntimeError(
                f"open_loop_sim requires every validation frame in episode "
                f"{records[0]['episode']!r}; missing {len(missing)} frames, "
                f"first missing={missing[0]}"
            )
        by_frame = {frame: record for frame, record in zip(frames, records)}
        end_frame = frames[-1] + 1

        sq = {
            "mse": 0.0,
            "xyz_mse": 0.0,
            "ypr_mse": 0.0,
            "grip_mse": 0.0,
            "paired_mse": 0.0,
        }
        executed = 0
        segments = 0
        segment_control_steps = []
        cursor = frames[0]
        while cursor < end_frame:
            record = by_frame.get(cursor)
            if record is None:
                raise RuntimeError(
                    f"open_loop_sim has no observation at frame {cursor}"
                )
            remaining = end_frame - cursor
            ground_truth = record["ground_truth"]
            if ground_truth.ndim != 2 or ground_truth.shape[1] != 14:
                raise ValueError(
                    "open_loop_sim ground truth must be a control-frequency "
                    f"(T, 14) trajectory, got {ground_truth.shape}"
                )
            prediction, n = self._decode_prediction_with_steps(
                record["prediction"], max_steps=min(len(ground_truth), remaining)
            )
            error = prediction[:n] - ground_truth[:n]
            sq["mse"] += float(np.square(error).sum())
            sq["xyz_mse"] += float(np.square(error[:, XYZ_COLS]).sum())
            sq["ypr_mse"] += float(np.square(error[:, YPR_COLS]).sum())
            sq["grip_mse"] += float(np.square(error[:, GRIP_COLS]).sum())
            sq["paired_mse"] += float(np.square(error[:, PAIRED_COLS]).sum())
            executed += n
            segments += 1
            segment_control_steps.append(n)
            cursor += n

        episode_length = end_frame - frames[0]
        denominators = {
            "mse": executed * 14,
            "xyz_mse": executed * len(XYZ_COLS),
            "ypr_mse": executed * len(YPR_COLS),
            "grip_mse": executed * len(GRIP_COLS),
            "paired_mse": executed * len(PAIRED_COLS),
        }
        return {
            "group": records[0].get("group", DEFAULT_VALID_GROUP),
            "episode": records[0]["episode"],
            "source": records[0]["source"],
            "label": records[0]["label"],
            "executed_steps": executed,
            "segments": segments,
            "segment_control_steps": segment_control_steps,
            "coverage": executed / max(episode_length, 1),
            "metrics": {key: sq[key] / max(denominators[key], 1) for key in sq},
        }

    @staticmethod
    def _summarize_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
        if not episodes:
            raise RuntimeError("open_loop_sim received no validation episodes")

        total_steps = sum(item["executed_steps"] for item in episodes)
        total_segments = sum(item["segments"] for item in episodes)
        micro = {}
        dimensions = {
            "mse": 14,
            "xyz_mse": len(XYZ_COLS),
            "ypr_mse": len(YPR_COLS),
            "grip_mse": len(GRIP_COLS),
            "paired_mse": len(PAIRED_COLS),
        }
        for key, dims in dimensions.items():
            micro[key] = sum(
                item["metrics"][key] * item["executed_steps"] * dims
                for item in episodes
            ) / max(total_steps * dims, 1)
        labels = defaultdict(list)
        for item in episodes:
            labels[item["label"]].append(item)
        return {
            "episodes": len(episodes),
            "executed_control_steps": total_steps,
            "segments": total_segments,
            "coverage": float(np.mean([item["coverage"] for item in episodes])),
            "micro": micro,
            "macro": {
                key: float(np.mean([item["metrics"][key] for item in episodes]))
                for key in micro
            },
            "per_label": {
                label: {
                    "episodes": len(items),
                    "executed_control_steps": sum(
                        item["executed_steps"] for item in items
                    ),
                    "coverage": float(np.mean([item["coverage"] for item in items])),
                    "micro": {
                        key: float(
                            np.average(
                                [item["metrics"][key] for item in items],
                                weights=[item["executed_steps"] for item in items],
                            )
                        )
                        for key in micro
                    },
                }
                for label, items in labels.items()
            },
            "episode_results": episodes,
        }

    def _compute_results(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        grouped = defaultdict(list)
        for record in records:
            grouped[
                (
                    record.get("group", DEFAULT_VALID_GROUP),
                    record["source"],
                    record["episode"],
                )
            ].append(record)
        if self.limit_val_episodes is not None:
            grouped_by_source = defaultdict(list)
            for key, group in grouped.items():
                grouped_by_source[key[:2]].append((key[2], group))
            grouped = defaultdict(list)
            for (validation_group, source), episode_groups in grouped_by_source.items():
                for episode, group in sorted(episode_groups)[: self.limit_val_episodes]:
                    grouped[(validation_group, source, episode)] = group
        episodes = [self._score_episode(group) for group in grouped.values()]
        if not episodes:
            raise RuntimeError("open_loop_sim received no validation episodes")
        by_group = defaultdict(list)
        for item in episodes:
            by_group[item["group"]].append(item)
        results = self._summarize_episodes(episodes)
        results.update(
            {
                "execute_fraction": self.execute_fraction,
                "execute_control_steps": self.execute_steps,
                "execute_arc_waypoints": (
                    max(
                        2,
                        min(
                            self.resampled_vector_length,
                            int(
                                math.ceil(
                                    self.resampled_vector_length
                                    * self.execute_fraction
                                )
                            ),
                        ),
                    )
                    if self.action_mode == "arc"
                    else None
                ),
                "replan_stride_mode": (
                    "arc_waypoint_timing"
                    if self.action_mode == "arc"
                    else "fixed_control_frames"
                ),
                "control_horizon": self.control_horizon,
                "control_dt": self.control_dt,
                "limit_val_episodes": self.limit_val_episodes,
                "groups": sorted(by_group),
                "per_group": {
                    group: self._summarize_episodes(items)
                    for group, items in sorted(by_group.items())
                },
            }
        )
        return results

    def _metric_tensors(self, results: dict[str, Any]) -> dict[str, torch.Tensor]:
        metrics = {}
        for group, summary in results["per_group"].items():
            prefix = (
                "Valid/open_loop_sim"
                if group == DEFAULT_VALID_GROUP
                else f"Valid_{group}/open_loop_sim"
            )
            metrics.update(
                {
                    f"{prefix}/MSE": summary["micro"]["mse"],
                    f"{prefix}/Episode_MSE": summary["macro"]["mse"],
                    f"{prefix}/XYZ_MSE": summary["micro"]["xyz_mse"],
                    f"{prefix}/YPR_MSE": summary["micro"]["ypr_mse"],
                    f"{prefix}/Grip_MSE": summary["micro"]["grip_mse"],
                    f"{prefix}/Paired_MSE": summary["micro"]["paired_mse"],
                    f"{prefix}/Coverage": summary["coverage"],
                    f"{prefix}/Episodes": summary["episodes"],
                    f"{prefix}/Executed_Control_Steps": summary[
                        "executed_control_steps"
                    ],
                    f"{prefix}/Segments": summary["segments"],
                }
            )
        return {
            key: torch.tensor(float(value), dtype=torch.float32).to(self._metric_device)
            for key, value in metrics.items()
        }

    def on_validation_end(self):
        records = self._all_records()
        results = self._compute_results(records)
        self.last_results = results
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return results
        if self.trainer is not None and not getattr(
            self.trainer, "is_global_zero", True
        ):
            return results
        # Flush episode buffers and upload the first MP4 per panel after all
        # validation frames have arrived.  This must happen before returning
        # on the rank-zero path; otherwise the last episode never gets a file.
        if getattr(self, "_video_enabled", False):
            EvalVideo.on_validation_end(self)
        # This hook is called from LightningModule.on_validation_end(). Calling
        # LightningModule.log_dict() here recursively enters Lightning's
        # validation-hook guard and raises ``MisconfigurationException``.
        # ``_all_records`` has already merged every rank, so direct logger
        # logging on global zero is both sufficient and deterministic.  Keep
        # values scalar so this works for W&B, TensorBoard, and LoggerCollection
        # without asking Lightning to infer validation-step semantics.
        if self.trainer is not None:
            logger = getattr(self.trainer, "logger", None)
            if logger is not None:
                metrics = {
                    key: float(value.detach().cpu().item())
                    for key, value in self._metric_tensors(results).items()
                }
                logger.log_metrics(metrics, step=self._resolved_log_step())
        if self.results_path is not None:
            self.results_path.parent.mkdir(parents=True, exist_ok=True)
            self.results_path.write_text(json.dumps(results, indent=2) + "\n")
        return results
