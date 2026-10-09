"""Shared episode-aware validation video buffering for graph robot models."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torchvision.io as tvio

from egomimic.eval.eval import Eval, EvaluationDataRequirements
from egomimic.pl_utils.pl_data_utils import DEFAULT_VALID_GROUP


class EvalVideo(Eval):
    def configure_video(
        self,
        *,
        source_fps,
        sample_id_key,
        frame_index_key,
        complete_episodes=False,
        mode="episode",
    ):
        if mode not in {"episode", "chunked"}:
            raise ValueError("video_mode must be episode or chunked")
        if complete_episodes and mode != "episode":
            raise ValueError("Complete-episode videos require video_mode=episode")
        EvaluationDataRequirements(
            complete_episodes=complete_episodes,
            source_fps=source_fps,
            sample_id_key=sample_id_key,
            frame_index_key=frame_index_key,
        )
        if mode == "episode" and (sample_id_key is None or frame_index_key is None):
            raise ValueError(
                "Episode videos require explicit identity and frame-index keys"
            )
        self.source_fps = source_fps
        self.sample_id_key = sample_id_key
        self.frame_index_key = frame_index_key
        self.complete_video_episodes = complete_episodes
        self.video_mode = mode

    def data_requirements(self):
        if not self.viz_func or self.viz_every_n_epochs <= 0:
            return EvaluationDataRequirements()
        if self.complete_video_episodes and self.viz_max_batches is not None:
            raise ValueError("Complete-episode videos cannot set viz_max_batches")
        return EvaluationDataRequirements(
            ordered=True,
            complete_episodes=self.complete_video_episodes,
            sample_id_key=self.sample_id_key if self.video_mode == "episode" else None,
            frame_index_key=self.frame_index_key
            if self.video_mode == "episode"
            else None,
            source_fps=self.source_fps,
        )

    def _video_fps(self):
        return self.source_fps

    def _should_viz(self, batch_idx=0):
        cadence = self.viz_every_n_epochs
        return (
            cadence > 0
            and (getattr(self.trainer, "current_epoch", 0) + 1) % cadence == 0
            and (self.viz_max_batches is None or batch_idx < self.viz_max_batches)
        )

    def on_validation_start(self):
        # Per-epoch reset. ``val_written`` in particular MUST clear here: it is
        # the "already have a video for this episode" guard, and carrying it
        # across epochs would silently skip every episode after epoch 0.
        self.val_image_buffer = {}
        self.val_counter = {}
        self.val_open_episode = {}
        self.val_written = {}
        self._written_paths = []
        self._written_fps = {}
        self._frame_records = {}
        previous = getattr(self, "_video_spool", None)
        if previous is not None:
            previous.cleanup()
        self._video_spool = tempfile.TemporaryDirectory(prefix="egoverse-video-")
        if self.trainer is not None and getattr(self.trainer, "is_global_zero", True):
            os.makedirs(
                os.path.join(
                    self._video_dir_root(),
                    f"epoch_{int(getattr(self.trainer, 'current_epoch', 0))}",
                ),
                exist_ok=True,
            )

    def on_validation_end(self):
        self._finish_frame_records()
        # All ranks supplied episode frames; only rank zero writes/uploads files.
        # The legacy single-rank chunked path may still have a buffered tail.
        if self.trainer is not None and not getattr(
            self.trainer, "is_global_zero", True
        ):
            return None
        for buf_key in list(self.val_image_buffer):
            group, embodiment_name = buf_key
            out_dir = self._group_video_dir(group, embodiment_name)
            if self.val_open_episode.get(buf_key) is not None:
                self._flush_episode(buf_key, out_dir)
            elif len(self.val_image_buffer[buf_key]) != 0:
                # chunked fallback tail
                self._write_video(
                    out_dir,
                    f"validation_video_{self.val_counter.get(buf_key, 0)}",
                    self.val_image_buffer[buf_key],
                    group=group,
                    embodiment_name=embodiment_name,
                )
            self.val_counter[buf_key] = 0
            self.val_image_buffer[buf_key] = []
            self.val_open_episode[buf_key] = None

        self._log_wandb_videos()
        return None

    def _video_dir_root(self) -> str:
        """Base ``videos/`` directory. Prefer ``video_output_dir`` when set,
        else fall back to ``{trainer.default_root_dir}/videos``."""
        if self.video_output_dir is not None:
            return str(self.video_output_dir)
        if self.trainer is None:
            return os.path.join(os.getcwd(), "videos")
        return os.path.join(self.trainer.default_root_dir, "videos")

    def _group_video_dir(self, group: str, embodiment_name: str) -> str:
        """``videos/epoch_N/[group/]{embodiment}`` -- the group segment is
        omitted for the default group so existing runs keep their layout."""
        parts = [
            self._video_dir_root(),
            f"epoch_{int(getattr(self.trainer, 'current_epoch', 0))}",
        ]
        if group != DEFAULT_VALID_GROUP:
            parts.append(str(group))
        parts.append(str(embodiment_name))
        return os.path.join(*parts)

    def _write_video(self, out_dir, name, frames, *, group, embodiment_name, fps=None):
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{name}.mp4")
        fps = self._video_fps() if fps is None else fps
        tvio.write_video(
            path,
            torch.stack(list(frames)),
            fps=fps,
            video_codec="h264",
        )
        self._written_paths.append((group, embodiment_name, path))
        self._written_fps[path] = fps

    def _write_episode_stream(
        self, out_dir, name, frames, *, group, embodiment_name, fps
    ):
        """Encode one frame at a time; a whole episode never occupies RAM."""
        import av

        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{name}.mp4")
        with av.open(path, mode="w") as container:
            stream = None
            for array in frames:
                array = np.asarray(array, dtype=np.uint8)
                if stream is None:
                    stream = container.add_stream(
                        "libx264", rate=Fraction(float(fps)).limit_denominator(100000)
                    )
                    stream.height, stream.width = array.shape[:2]
                    stream.pix_fmt = "yuv420p"
                for packet in stream.encode(
                    av.VideoFrame.from_ndarray(array, format="rgb24")
                ):
                    container.mux(packet)
            if stream is not None:
                for packet in stream.encode():
                    container.mux(packet)
        self._written_paths.append((group, embodiment_name, path))
        self._written_fps[path] = fps

    def _flush_episode(self, buf_key, out_dir):
        """Write the open episode's buffer as ``{episode_hash}.mp4``."""
        episode = self.val_open_episode.get(buf_key)
        buffer = self.val_image_buffer.get(buf_key) or []
        if episode is not None and len(buffer) != 0:
            group, embodiment_name = buf_key
            self._write_video(
                out_dir,
                episode,
                buffer,
                group=group,
                embodiment_name=embodiment_name,
            )
            self.val_written.setdefault(buf_key, set()).add(episode)
        self.val_image_buffer[buf_key] = []
        self.val_open_episode[buf_key] = None

    def _buffer_per_episode(self, buf_key, out_dir, frames, hashes):
        """One mp4 per episode, cut where ``episode_hash`` changes.

        Validation runs with shuffle=False and MultiDataset lays its index map
        out episode by episode, so samples arrive grouped by episode and in
        frame order -- the boundary is simply where the hash changes.
        """
        written = self.val_written.setdefault(buf_key, set())
        for frame, episode in zip(frames, hashes):
            open_episode = self.val_open_episode.get(buf_key)
            if episode in written:
                # Replay from ``max_size_cycle`` (or frames past the safety
                # cap). Seeing a finished episode also means whatever is still
                # open has ended -- without this the LAST episode of a cycling
                # embodiment never hits a boundary and keeps accumulating a
                # copy of itself on every wrap.
                if open_episode is not None:
                    self._flush_episode(buf_key, out_dir)
                continue
            if open_episode is None:
                self.val_open_episode[buf_key] = episode
                self.val_image_buffer[buf_key] = []
            elif episode != open_episode:
                self._flush_episode(buf_key, out_dir)
                self.val_open_episode[buf_key] = episode
                self.val_image_buffer[buf_key] = []
            buffer = self.val_image_buffer[buf_key]
            buffer.append(frame)
            if len(buffer) >= self.max_episode_frames:
                # Episode longer than the safety cap: write what we have and
                # mark it done so the remainder is dropped rather than
                # spilling into the next episode's file.
                self._flush_episode(buf_key, out_dir)

    def _buffer_chunked(self, buf_key, out_dir, frames):
        """Legacy path for batches with no ``episode_hash``: fixed-size files."""
        if self.val_image_buffer.get(buf_key) is None:
            self.val_image_buffer[buf_key] = []
            self.val_counter[buf_key] = 0
        self.val_image_buffer[buf_key].extend(frames)
        if len(self.val_image_buffer[buf_key]) >= self.video_chunk_frames:
            group, embodiment_name = buf_key
            self._write_video(
                out_dir,
                f"validation_video_{self.val_counter[buf_key]}",
                self.val_image_buffer[buf_key],
                group=group,
                embodiment_name=embodiment_name,
            )
            self.val_image_buffer[buf_key].clear()
            self.val_counter[buf_key] += 1

    def _episode_hashes(self, source_batch: Mapping, n_images: int):
        """Per-sample ``episode_hash`` for this embodiment's images.

        ``episode_hash`` is stamped on every sample by ZarrDataset and falls
        through ``process_batch_for_training`` unchanged. Returns ``None`` when
        it is absent or shorter than the image count so callers fall back to
        fixed-size chunking rather than mislabelling frames.
        """
        hashes = (
            source_batch.get(self.sample_id_key)
            if isinstance(source_batch, Mapping)
            else None
        )
        if not isinstance(hashes, (list, tuple)) or len(hashes) < n_images:
            return None
        return [str(h) for h in hashes[:n_images]]

    def _record_video_frames(self, buf_key, frames, source_batch):
        """Buffer by declared identity/index, independent of loader cycling or rank.

        All ranks contribute at epoch end. Repeated samples (including DDP
        padding) are deduplicated by their data-owned frame identity.
        """
        if self.video_mode == "chunked":
            if (
                torch.distributed.is_initialized()
                and torch.distributed.get_world_size() > 1
            ):
                raise ValueError(
                    "Distributed videos require episode identity and frame indices"
                )
            return self._buffer_chunked(
                buf_key, self._group_video_dir(*buf_key), list(frames)
            )
        hashes = self._episode_hashes(source_batch, len(frames))
        indices = source_batch.get(self.frame_index_key)
        if hashes is None or indices is None or len(indices) != len(frames):
            raise ValueError(
                f"Video batch needs {self.sample_id_key!r} and {self.frame_index_key!r} for every frame"
            )
        for frame, episode, index in zip(frames, hashes, indices, strict=True):
            value = index.item() if torch.is_tensor(index) else index
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 0
            ):
                raise ValueError("Video frame indices must be nonnegative integers")
            key = (*buf_key, episode)
            records = self._frame_records.setdefault(key, {})
            if value in records:
                continue
            if len(records) >= self.max_episode_frames:
                if self.complete_video_episodes:
                    raise ValueError(
                        f"Complete video exceeds max_episode_frames: {episode!r}"
                    )
                continue
            name = hashlib.sha256(json.dumps([*key, int(value)]).encode()).hexdigest()
            path = Path(self._video_spool.name) / (name + ".npy")
            np.save(
                path, frame.detach().cpu().numpy().astype(np.uint8), allow_pickle=False
            )
            records[int(value)] = path

    def _finish_frame_records(self):
        local = getattr(self, "_frame_records", {})
        distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        metadata = {key: tuple(records) for key, records in local.items()}
        payloads = [metadata]
        rank = 0
        if distributed:
            payloads = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(payloads, metadata)
            rank = torch.distributed.get_rank()
        merged = {}
        for payload in payloads:
            for key, indices in payload.items():
                merged.setdefault(key, set()).update(indices)
        errors = []
        for key, indices in sorted(merged.items()):
            group, source, episode = key
            if self.complete_video_episodes and len(indices) > self.max_episode_frames:
                raise ValueError(
                    f"Complete video exceeds max_episode_frames: {episode!r}"
                )
            indices = sorted(indices)[: self.max_episode_frames]
            if not indices:
                continue
            stride = 1 if len(indices) < 2 else math.gcd(*np.diff(indices).tolist())
            if self.complete_video_episodes and indices != list(range(len(indices))):
                raise ValueError(
                    f"Video episode {episode!r} is incomplete after rank gathering"
                )
            records = {}
            # Only one bounded batch of pixel arrays is communicated at a time.
            # Other ranks never receive the entire corpus's video frames.
            for offset in range(0, len(indices), 32):
                chosen = indices[offset : offset + 32]
                values = {
                    i: np.load(local[key][i], allow_pickle=False)
                    for i in chosen
                    if i in local.get(key, {})
                }
                gathered = [values]
                if distributed:
                    gathered = [None] * len(payloads) if rank == 0 else None
                    torch.distributed.gather_object(values, gathered, dst=0)
                if rank == 0:
                    for contribution in gathered:
                        for index, frame in contribution.items():
                            if index not in records:
                                name = hashlib.sha256(
                                    json.dumps([*key, index, "gathered"]).encode()
                                ).hexdigest()
                                path = Path(self._video_spool.name) / (name + ".npy")
                                np.save(path, frame, allow_pickle=False)
                                records[index] = path
            if rank != 0:
                continue

            def frames():
                latest = records[indices[0]]
                for index in range(indices[0], indices[-1] + 1, stride):
                    latest = records.get(index, latest)
                    yield np.load(latest, allow_pickle=False)

            name = (
                episode
                if Path(episode).name == episode and episode not in {".", ".."}
                else hashlib.sha256(episode.encode()).hexdigest()
            )
            try:
                self._write_episode_stream(
                    self._group_video_dir(group, source),
                    name,
                    frames(),
                    group=group,
                    embodiment_name=source,
                    fps=self.source_fps / stride,
                )
            except Exception as error:
                # Finish collectives before reporting rank-zero encoding errors.
                errors.append(f"{episode}: {error}")
        if distributed:
            result = [errors]
            torch.distributed.broadcast_object_list(result, src=0)
            errors = result[0]
        self._frame_records = {}
        spool = getattr(self, "_video_spool", None)
        if spool is not None:
            spool.cleanup()
        if errors:
            raise RuntimeError("Episode video encoding failed: " + "; ".join(errors))

    def _wandb_logger(self):
        """Return the WandbLogger.experiment handle if wandb is configured."""
        if self.trainer is None:
            return None
        logger = getattr(self.trainer, "logger", None)
        loggers = getattr(self.trainer, "loggers", None) or ([logger] if logger else [])
        for entry in loggers:
            experiment = getattr(entry, "experiment", None)
            if experiment is None:
                continue
            if callable(getattr(experiment, "log", None)) and hasattr(experiment, "id"):
                return experiment
        return None

    def _log_wandb_videos(self) -> None:
        """Upload every mp4 written this epoch as a ``wandb.Video`` panel.

        One log call per (group, embodiment); the panel accepts a list so all
        episode files for that pair land on the same chart.
        """
        if not self._written_paths:
            return
        experiment = self._wandb_logger()
        if experiment is None:
            return
        import wandb  # type: ignore

        grouped: dict[tuple[str, str], list[str]] = {}
        for group, embodiment_name, path in self._written_paths:
            grouped.setdefault((group, embodiment_name), []).append(path)

        payload: dict = {}
        for (group, embodiment_name), paths in grouped.items():
            prefix = (
                "Val_video" if group == DEFAULT_VALID_GROUP else f"Val_video_{group}"
            )
            payload[f"{prefix}/{embodiment_name}"] = [
                wandb.Video(
                    p, fps=self._written_fps.get(p, self._video_fps()), format="mp4"
                )
                for p in paths
            ]
        experiment.log(
            payload,
            step=int(getattr(self.trainer, "global_step", 0)),
        )
