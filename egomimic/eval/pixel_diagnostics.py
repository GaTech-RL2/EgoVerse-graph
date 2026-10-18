"""Bounded, timestamp-faithful RGB prediction diagnostics for patch models."""

from __future__ import annotations

import json
import math
import re
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from egomimic.eval.artifact_paths import (
    artifact_destination,
    artifact_execution_identity,
)


def _component(value):
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in (".", ".."):
        raise ValueError("Artifact path component must be a simple identifier")
    return value


def unpatchify_rgb(patches, image_size, patch_size):
    """Invert row-major/HWC patchification without clipping predictions."""
    height, width = image_size
    gy, gx = height // patch_size, width // patch_size
    if patches.shape != (gy * gx, 3 * patch_size**2):
        raise ValueError("Pixel patch shape differs from declared image geometry")
    return (
        patches.reshape(gy, gx, patch_size, patch_size, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(height, width, 3)
    )


def pixel_frame_records(stream, prediction, metadata, row):
    """Validate exact physical metadata and restore each camera's RGB frames."""
    size, patch = metadata["image_size"], metadata["patch_size"]
    if (
        len(size) != 2
        or any(type(x) is not int or x < 1 for x in size)
        or type(patch) is not int
        or patch < 1
        or any(x % patch for x in size)
        or metadata.get("normalization") != "uint8_to_minus_one_one"
        or metadata.get("patch_order") != "row_major_hwc"
    ):
        raise ValueError(
            "Pixel diagnostics require explicit RGB patch geometry and normalization"
        )
    times = np.asarray(metadata["timestamps_seconds"], np.float64)
    frames = np.asarray(metadata["frame_indices"])
    cameras = metadata["camera_names"]
    origin = float(metadata["time_origin_seconds"])
    if (
        times.ndim != 1
        or len(times) == 0
        or not np.isfinite(times).all()
        or not (np.diff(times) > 0).all()
        or not math.isfinite(origin)
        or frames.shape != times.shape
        or frames.dtype.kind not in "iu"
        or (frames < 0).any()
        or not (np.diff(frames) > 0).all()
        or not cameras
        or len(set(cameras)) != len(cameras)
        or any(not isinstance(x, str) or not x for x in cameras)
    ):
        raise ValueError(
            "Pixel metadata requires ordered physical times, frame indices and camera names"
        )
    arrays = {key: value[row].detach().cpu().numpy() for key, value in stream.items()}
    predicted = (
        None if prediction is None else prediction[row].detach().float().cpu().numpy()
    )
    if predicted is not None and predicted.shape != arrays["value"].shape:
        raise ValueError("Pixel prediction and target shapes differ")
    valid = arrays["valid"]
    index, camera = arrays["index"], arrays["camera"]
    if (
        (index[valid] < 0).any()
        or (index[valid] >= len(times)).any()
        or not np.equal(index[valid], np.round(index[valid])).all()
        or (camera[valid] < 0).any()
        or (camera[valid] >= len(cameras)).any()
    ):
        raise ValueError("Pixel tokens do not match metadata frames/cameras")
    result = {}
    gy, gx = size[0] // patch, size[1] // patch
    for camera_id, camera_name in enumerate(cameras):
        sequence = []
        for ordinal, (timestamp, frame_index) in enumerate(zip(times, frames)):
            selected = np.flatnonzero(
                valid & (index == ordinal) & (camera == camera_id)
            )
            if len(selected) != gy * gx:
                raise ValueError(
                    "Pixel diagnostic requires one complete grid per frame/camera"
                )
            xy = arrays["xy"][selected]
            grid = np.rint((xy + 1) * np.array([gx, gy]) / 2 - 0.5).astype(int)
            expected_xy = (grid + 0.5) / np.array([gx, gy]) * 2 - 1
            order = grid[:, 1] * gx + grid[:, 0]
            if (
                not np.isfinite(xy).all()
                or not np.allclose(xy, expected_xy, atol=1e-5)
                or (grid < 0).any()
                or (grid >= np.array([gx, gy])).any()
                or not np.array_equal(np.sort(order), np.arange(gy * gx))
            ):
                raise ValueError(
                    "Pixel spatial coordinates do not form the declared grid"
                )
            selected = selected[np.argsort(order)]
            if not np.allclose(
                arrays["time"][selected], timestamp - origin, atol=1e-4, rtol=1e-5
            ):
                raise ValueError(
                    "Pixel token time does not match recorded physical timestamp"
                )
            if not arrays.get("target_valid", valid)[selected].all():
                raise ValueError("RGB diagnostics require recorded target pixels")
            target = arrays["value"][selected]
            estimate = None if predicted is None else predicted[selected]
            if not np.isfinite(target).all() or (
                estimate is not None and not np.isfinite(estimate).all()
            ):
                raise ValueError("Non-finite RGB diagnostic target/prediction")
            if (np.abs(target) > 1.00001).any():
                raise ValueError("Target pixels are not normalized to [-1,1]")
            observed = arrays["observed"][selected]
            status = (
                "observed"
                if observed.all()
                else "query"
                if not observed.any()
                else "mixed"
            )
            details = {
                "frame_index": int(frame_index),
                "timestamp_seconds": float(timestamp),
                "relative_time_seconds": float(timestamp - origin),
                "observation": status,
                "observed_patches": observed.tolist(),
                "query_pixel_mse_rgb01": None,
                "prediction_outside_rgb_range_fraction": None,
            }
            if estimate is not None:
                if (~observed).any():
                    details["query_pixel_mse_rgb01"] = float(
                        np.square((estimate[~observed] - target[~observed]) / 2).mean()
                    )
                details["prediction_outside_rgb_range_fraction"] = float(
                    (np.abs(estimate) > 1).mean()
                )
            sequence.append(
                {
                    "metadata": details,
                    "target": (unpatchify_rgb(target, size, patch) + 1) / 2,
                    "prediction": None
                    if estimate is None
                    else (unpatchify_rgb(estimate, size, patch) + 1) / 2,
                }
            )
        result[camera_id] = {"name": camera_name, "frames": sequence}
    return result


def _panel(frame, camera_name, *, direct=False):
    target, prediction, metadata = (
        frame["target"],
        frame["prediction"],
        frame["metadata"],
    )
    height, width = target.shape[:2]
    scale = 2
    columns = 1 if direct else 3
    panel = Image.new(
        "RGB", (max(480, columns * width * scale), height * scale + 58), (18, 21, 28)
    )
    draw = ImageDraw.Draw(panel)
    error = metadata["query_pixel_mse_rgb01"]
    label = (
        "context only; no RGB prediction"
        if direct
        else f"{metadata['observation']} | query MSE={error:.6f}"
        if error is not None
        else "observed; excluded from query metrics"
    )
    draw.text(
        (5, 3),
        f"{camera_name} | frame {metadata['frame_index']} | t={metadata['timestamp_seconds']:.6f}s",
        fill="white",
    )
    draw.text(
        (5, 19),
        label,
        fill=(120, 220, 170)
        if metadata["observation"] == "observed"
        else (255, 205, 110),
    )
    arrays, labels = [target], ["Recorded RGB"]
    if not direct:
        difference = np.abs(prediction - target).mean(-1)
        heat = np.stack(
            (difference, np.zeros_like(difference), np.zeros_like(difference)), -1
        )
        arrays += [prediction, heat]
        labels += ["Predicted RGB (display clipped)", "Absolute error (RGB [0,1])"]
    for i, (array, title) in enumerate(zip(arrays, labels)):
        image = Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8))
        image = image.resize((width * scale, height * scale), Image.Resampling.NEAREST)
        panel.paste(image, (i * width * scale, 58))
        draw.text((i * width * scale + 5, 39), title, fill="white")
    return panel


def write_timestamp_video(path, panels, timestamps, nominal_fps=10):
    """Store every sampled frame with physical-time PTS, without uniform retiming."""
    import av

    clock = np.asarray(timestamps, np.float64)
    ticks = np.rint((clock - clock[0]) * 1_000_000).astype(np.int64)
    if len(ticks) > 1 and not (np.diff(ticks) > 0).all():
        raise ValueError("Video timestamps must be distinct at microsecond precision")
    time_base = Fraction(1, 1_000_000)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream(
            "libx264", rate=Fraction(nominal_fps).limit_denominator()
        )
        stream.pix_fmt = "yuv420p"
        stream.time_base = stream.codec_context.time_base = time_base
        stream.options = {"crf": "18", "preset": "veryfast", "bf": "0"}
        for panel, tick in zip(panels, ticks):
            array = np.asarray(panel, np.uint8)
            stream.height, stream.width = array.shape[:2]
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            frame.pts, frame.time_base = int(tick), time_base
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class PixelPredictionDiagnostics:
    def __init__(
        self,
        artifact_dir,
        max_samples_per_source_kind=1,
        playback_fps=10,
        write_video=True,
        log_wandb=True,
        contact_sheet_frames=8,
    ):
        if not artifact_dir:
            raise ValueError("Pixel diagnostics require an explicit artifact_dir")
        if (
            type(max_samples_per_source_kind) is not int
            or max_samples_per_source_kind < 1
            or type(contact_sheet_frames) is not int
            or contact_sheet_frames < 1
        ):
            raise ValueError("Pixel diagnostic limits must be positive integers")
        if not math.isfinite(playback_fps) or playback_fps <= 0:
            raise ValueError("playback_fps must be positive")
        if type(write_video) is not bool or type(log_wandb) is not bool:
            raise ValueError("Pixel output switches must be boolean")
        self.root = Path(artifact_dir)
        self.limit, self.contact_sheet_frames = (
            max_samples_per_source_kind,
            contact_sheet_frames,
        )
        self.playback_fps, self.write_video, self.log_wandb = (
            playback_fps,
            write_video,
            log_wandb,
        )
        self.start()

    def start(self):
        self.counts, self.written = {}, []

    def record(
        self, *, model, source, item, prediction, objective, batch_idx, dataloader_idx
    ):
        if model.global_rank != 0:
            return
        if len(item["pixel_metadata"]) != len(item["task_kind"]):
            raise ValueError("Every pixel sample requires its own metadata")
        for row, (metadata, kind) in enumerate(
            zip(item["pixel_metadata"], item["task_kind"])
        ):
            key = (source, kind, dataloader_idx)
            if self.counts.get(key, 0) >= self.limit:
                continue
            direct = objective == "direct_action"
            frames = pixel_frame_records(
                item["world"]["image"],
                None if direct else prediction["image"],
                metadata,
                row,
            )
            root = (
                self.root
                / f"loader-{dataloader_idx}"
                / _component(source)
                / _component(kind)
            )
            execution = artifact_execution_identity()
            path = (
                artifact_destination(
                    root,
                    execution,
                    epoch=model.current_epoch,
                    global_step=model.global_step,
                    rank=model.global_rank,
                    batch_idx=batch_idx,
                ).with_suffix("")
                / f"sample-{row}"
            )
            path.mkdir(parents=True, exist_ok=False)
            report = {
                "schema": "pixel-prediction-diagnostics-v1",
                "objective": objective,
                "source": source,
                "task_kind": kind,
                "pixel_metadata": metadata,
                "execution": execution,
                "global_step": int(model.global_step),
                "video_timing": "variable PTS from recorded seconds; no uniform-frame retiming",
                "last_frame_hold_seconds": 1 / self.playback_fps,
                "pixel_error_space": "RGB [0,1], unclipped predictions; observed patches excluded",
                "display": "2x nearest-neighbor; predictions clipped only for display; error is absolute RGB error, red channel",
                "cameras": {},
            }
            for name in ("proprio", "action"):
                if name not in prediction:
                    continue
                stream = item["world"][name]
                valid = (
                    stream["valid"][row]
                    & stream.get("target_valid", stream["valid"])[row]
                    & ~stream["observed"][row]
                )
                report[name] = {
                    "prediction": prediction[name][row][valid].detach().cpu().tolist(),
                    "target": stream["value"][row][valid].detach().cpu().tolist(),
                    "relative_time_seconds": stream["time"][row][valid]
                    .detach()
                    .cpu()
                    .tolist(),
                    "squared_error": (
                        prediction[name][row][valid] - stream["value"][row][valid]
                    )
                    .square()
                    .detach()
                    .cpu()
                    .tolist(),
                    "value_space": "normalized_measured_state"
                    if name == "proprio"
                    else "native_action_commands",
                    "note": "RGB is observed context only; this model does not predict images"
                    if direct
                    else "Readout after generated RGB; observed state entries excluded",
                }
            for camera_id, camera in frames.items():
                sequence = camera["frames"]
                selected = np.unique(
                    np.linspace(
                        0,
                        len(sequence) - 1,
                        min(self.contact_sheet_frames, len(sequence)),
                        dtype=int,
                    )
                )
                thumbnails = [
                    _panel(sequence[i], camera["name"], direct=direct) for i in selected
                ]
                sheet = Image.new(
                    "RGB", (thumbnails[0].width, sum(im.height for im in thumbnails))
                )
                for offset, thumbnail in enumerate(thumbnails):
                    sheet.paste(thumbnail, (0, offset * thumbnail.height))
                sheet_path = path / f"camera-{camera_id}.png"
                sheet.save(sheet_path)
                report["cameras"][str(camera_id)] = {
                    "name": camera["name"],
                    "frames": [f["metadata"] for f in sequence],
                    "contact_sheet_frame_indices": [
                        int(metadata["frame_indices"][i]) for i in selected
                    ],
                    "contact_sheet": sheet_path.name,
                    "video": None,
                }
                self.written.append((source, kind, "image", sheet_path))
                if self.write_video and not direct:
                    video_path = path / f"camera-{camera_id}.mp4"
                    write_timestamp_video(
                        video_path,
                        (_panel(f, camera["name"]) for f in sequence),
                        metadata["timestamps_seconds"],
                        self.playback_fps,
                    )
                    report["cameras"][str(camera_id)]["video"] = video_path.name
                    self.written.append((source, kind, "video", video_path))
            (path / "diagnostic.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
            self.counts[key] = self.counts.get(key, 0) + 1

    def log_to_wandb(self, model):
        if not self.log_wandb or model.global_rank != 0 or not self.written:
            return
        import wandb
        from lightning.pytorch.loggers import WandbLogger

        for logger in model.trainer.loggers:
            if not isinstance(logger, WandbLogger):
                continue
            payload = {}
            for source, kind, media, path in self.written:
                key = f"val_pixel/{source}/{kind}/{media}"
                value = (
                    wandb.Image(str(path))
                    if media == "image"
                    else wandb.Video(str(path), format="mp4")
                )
                payload.setdefault(key, []).append(value)
            # Lightning records trainer/global_step as the metric axis. W&B's
            # internal history counter also includes validation/media rows.
            logger.log_metrics(payload, step=int(model.global_step))
