"""Shared episode-aware validation video buffering for graph robot models."""

from __future__ import annotations
import os
from pathlib import Path
from collections.abc import Mapping
import numpy as np
import torch
import torchvision.io as tvio
from egomimic.eval.eval import Eval
from egomimic.pl_utils.pl_data_utils import DEFAULT_VALID_GROUP


class EvalVideo(Eval):
    def _video_fps(self, source_fps=30):
        world = max(1, int(getattr(self.trainer, "world_size", 1) or 1))
        return max(1, round(source_fps / world))

    def _should_viz(self, batch_idx=0):
        cadence = self.viz_every_n_epochs
        return (
            getattr(self.trainer, "is_global_zero", True)
            and cadence > 0
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
        if self.trainer is not None and getattr(self.trainer, "is_global_zero", True):
            os.makedirs(
                os.path.join(
                    self._video_dir_root(),
                    f"epoch_{int(getattr(self.trainer, 'current_epoch', 0))}",
                ),
                exist_ok=True,
            )

    def on_validation_end(self):
        # Only rank 0 buffered / wrote frames, so only rank 0 has tails to
        # flush and paths to upload.
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

    def _write_video(self, out_dir, name, frames, *, group, embodiment_name):
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{name}.mp4")
        tvio.write_video(
            path,
            torch.stack(list(frames)),
            fps=self._video_fps(),
            video_codec="h264",
        )
        self._written_paths.append((group, embodiment_name, path))

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

    @staticmethod
    def _episode_hashes(source_batch: Mapping, n_images: int):
        """Per-sample ``episode_hash`` for this embodiment's images.

        ``episode_hash`` is stamped on every sample by ZarrDataset and falls
        through ``process_batch_for_training`` unchanged. Returns ``None`` when
        it is absent or shorter than the image count so callers fall back to
        fixed-size chunking rather than mislabelling frames.
        """
        hashes = (
            source_batch.get("episode_hash")
            if isinstance(source_batch, Mapping)
            else None
        )
        if not isinstance(hashes, (list, tuple)) or len(hashes) < n_images:
            return None
        return [str(h) for h in hashes[:n_images]]

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
                wandb.Video(p, fps=self._video_fps(), format="mp4") for p in paths
            ]
        experiment.log(
            payload,
            step=int(getattr(self.trainer, "global_step", 0)),
        )
