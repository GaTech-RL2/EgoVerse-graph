"""Record a YAM rollout as a demonstration-format HDF5 episode.

The file's top level is exactly what ``collect_demo.EpisodeWriter`` writes for a
GELLO demonstration: measured joints and EEF poses, the joint command sent to
each arm, the forward kinematics of that command, and one RGB frame per camera
per row. A rollout episode therefore replays, converts and loads like a demo.
Rollout-only data lives under ``rollout/``: per-row timestamps and plan index,
the policy's Cartesian target, every plan with its inference settings, and under
``rollout/model/`` the checkpoint bundle's own text sidecars (resolved training
config, inference config, provenance) so the episode names the exact model.

Operator labels (outcome, partial score, project, model, notes) live in a
``<episode>.labels.json`` sidecar beside the finalized file. They can be edited
after the take; each revision keeps the previous ones, and the HDF5 file is never
reopened for them. ``python -m egomimic.robot.rollout_episode <directory>``
prints the joined index (episode, model/W&B identity, labels) without opening
any device.

A background thread owns the file, so the multi-megabyte image rows are never
written inside the control tick; the rollout loop only enqueues copies. Like the
MP4 recorder, this observes the rollout and has no robot command path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import re
import shutil
import sys
import threading
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np

from egomimic.robot.collect_demo import EpisodeWriter

EPISODE_RECORDING_DEFAULTS = {
    "enabled": False,
    # Start an episode (and the MP4, when video recording is enabled) at every
    # rollout start. The dashboard can switch this off for the session.
    "auto_start": False,
    # Ending a recording take (End success / failure / End episode, or Discard)
    # also restarts the rollout exactly like r: back to the ready gate, both
    # arms home when reset_home_on_restart is set, and the next Start begins a
    # new take.
    "reset_on_end": False,
    "directory": "/home/rohan/rollouts/yam_hptflow/episodes",
    "min_free_gb": 20.0,
    "max_queued_rows": 60,
}
OUTCOMES = ("success", "failure", "unlabeled")
LABELS_SCHEMA = "rl2_yam.rollout_labels.v1"
LABEL_FIELDS = ("outcome", "partial_score", "project", "model", "notes")
LABEL_TEXT_LIMITS = {"project": 120, "model": 200, "notes": 2000}
END_REASONS = (
    "saved",
    "restart",
    "model_change",
    "camera_reconnect",
    "stop",
    "error",
    "low_disk",
    "writer_error",
)
_EPISODE_ID = re.compile(r"^rollout_\d{8}-\d{6}-\d{6}$")
_DISK_CHECK_SECONDS = 1.0
_END_POLL_SECONDS = 0.05


def validate_episode_recording(config: Mapping[str, object] | None) -> dict:
    """Validate the local episode-output policy without touching cameras or disk."""
    config = {} if config is None else dict(config)
    unknown = config.keys() - EPISODE_RECORDING_DEFAULTS.keys()
    if unknown:
        raise ValueError(
            "Unknown episode_recording option(s): " + ", ".join(sorted(unknown))
        )
    result = {**deepcopy(EPISODE_RECORDING_DEFAULTS), **config}
    for name in ("enabled", "auto_start", "reset_on_end"):
        if type(result[name]) is not bool:
            raise ValueError(f"episode_recording.{name} must be a boolean")
    directory = result["directory"]
    if not isinstance(directory, str) or not directory:
        raise ValueError("episode_recording.directory must be a nonempty path")
    if not Path(directory).is_absolute():
        raise ValueError("episode_recording.directory must be an absolute path")
    free = result["min_free_gb"]
    if (
        isinstance(free, bool)
        or not isinstance(free, (int, float))
        or not math.isfinite(free)
        or free < 0
    ):
        raise ValueError("episode_recording.min_free_gb must be a finite number >= 0")
    result["min_free_gb"] = float(free)
    rows = result["max_queued_rows"]
    if type(rows) is not int or not 1 <= rows <= 600:
        raise ValueError(
            "episode_recording.max_queued_rows must be an integer in [1, 600]"
        )
    return result


def _episode_id() -> str:
    return "rollout_" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")


def is_episode_id(value: object) -> bool:
    return isinstance(value, str) and _EPISODE_ID.fullmatch(value) is not None


def list_rollout_episodes(directory: str | Path) -> list[dict]:
    """Return finalized rollout episode manifests with their labels, newest first.

    ``outcome`` is the operator's latest label when one exists; the outcome the
    recorder wrote at the end of the take stays in ``recorded_outcome``.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    episodes = []
    for manifest_path in directory.glob("rollout_*.json"):
        if manifest_path.name.endswith((".labels.json", ".upload.json")):
            continue
        try:
            payload = json.loads(manifest_path.read_text())
            episode_id, filename = payload["id"], payload["filename"]
            if (
                not isinstance(episode_id, str)
                or _EPISODE_ID.fullmatch(episode_id) is None
                or filename != f"{episode_id}.hdf5"
                or type(payload["frames"]) is not int
                or payload["frames"] <= 0
            ):
                continue
            path = directory / filename
            if not path.is_file():
                continue
            episode = {
                **payload,
                "size_bytes": path.stat().st_size,
                "recorded_outcome": payload.get("outcome", "unlabeled"),
                "labels": None,
                "labels_revision": 0,
            }
            try:
                record = read_episode_labels(directory, episode_id)
            except ValueError:
                record = None  # an unreadable sidecar is left alone, not hidden data
            if record is not None:
                episode.update(
                    labels=record["labels"],
                    labels_revision=record["revision"],
                    labeled_at=record["labeled_at"],
                    outcome=record["labels"]["outcome"],
                )
            episodes.append(episode)
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return sorted(episodes, key=lambda episode: episode["id"], reverse=True)


def parse_partial_score(value: object) -> tuple[float | None, str]:
    """Parse ``3/5``, ``0.6`` or ``60%`` into a fraction in [0, 1].

    Returns ``(fraction, text as entered)``; an empty entry is ``(None, "")``.
    """
    if value is None:
        return None, ""
    if isinstance(value, bool):
        raise ValueError("partial_score must be text such as 3/5, 0.6 or 60%")
    text = str(value).strip()
    if not text:
        return None, ""
    if len(text) > 32:
        raise ValueError("partial_score is too long")
    number = r"(\d+(?:\.\d*)?|\.\d+)"
    try:
        if match := re.fullmatch(number + r"\s*/\s*" + number, text):
            done, total = float(match.group(1)), float(match.group(2))
            if total <= 0 or done > total:
                raise ValueError
            score = done / total
        elif match := re.fullmatch(number + r"\s*%", text):
            score = float(match.group(1)) / 100
        elif re.fullmatch(number, text):
            score = float(text)
        else:
            raise ValueError
    except ValueError:
        raise ValueError(
            "partial_score must be a fraction such as 3/5, 0.6 or 60%"
        ) from None
    if not 0 <= score <= 1:
        raise ValueError("partial_score must be between 0 and 1 (0% to 100%)")
    return round(score, 6), text


def validate_episode_labels(labels: object) -> dict:
    """Validate one operator label entry and return its stored form."""
    if not isinstance(labels, Mapping):
        raise ValueError("Episode labels must be a mapping")
    unknown = set(labels) - set(LABEL_FIELDS)
    if unknown:
        raise ValueError("Unknown episode label(s): " + ", ".join(sorted(unknown)))
    outcome = labels.get("outcome", "unlabeled")
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}")
    score, score_text = parse_partial_score(labels.get("partial_score"))
    result = {
        "outcome": outcome,
        "partial_score": score,
        "partial_score_text": score_text,
    }
    for name, limit in LABEL_TEXT_LIMITS.items():
        value = labels.get(name)
        value = "" if value is None else value
        if not isinstance(value, str):
            raise ValueError(f"{name} must be text")
        value = value.strip()
        if len(value) > limit:
            raise ValueError(f"{name} must be at most {limit} characters")
        allowed = "\n\t" if name == "notes" else ""
        if any(ord(char) < 32 and char not in allowed for char in value):
            raise ValueError(f"{name} contains control characters")
        result[name] = value
    return result


def _labels_path(directory: Path, episode_id: str) -> Path:
    return directory / f"{episode_id}.labels.json"


def read_episode_labels(directory: str | Path, episode_id: str) -> dict | None:
    """Return one episode's label record, or None when it has never been labeled."""
    if not is_episode_id(episode_id):
        raise ValueError("Unknown rollout episode")
    path = _labels_path(Path(directory), episode_id)
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text())
        if (
            record.get("schema") != LABELS_SCHEMA
            or record.get("id") != episode_id
            or type(record.get("revision")) is not int
            or not isinstance(record.get("labels"), dict)
        ):
            raise ValueError
        record["labels"] = validate_episode_labels(
            {
                key: (
                    record["labels"].get("partial_score_text")
                    if key == "partial_score"
                    else record["labels"].get(key)
                )
                for key in LABEL_FIELDS
            }
        )
    except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
        raise ValueError(f"{path.name} is unreadable; fix or move it by hand") from None
    return record


_LABELS_LOCK = threading.Lock()


def write_episode_labels(
    directory: str | Path, episode_id: str, labels: Mapping[str, object]
) -> dict:
    """Store an operator label revision for one finalized episode.

    The sidecar is replaced atomically and keeps every earlier revision under
    ``history``; the HDF5 file is not opened. An unreadable existing sidecar is
    never overwritten.
    """
    directory = Path(directory)
    if not is_episode_id(episode_id) or not all(
        (directory / f"{episode_id}{suffix}").is_file() for suffix in (".hdf5", ".json")
    ):
        raise FileNotFoundError("Unknown or still-recording rollout episode")
    clean = validate_episode_labels(labels)
    with _LABELS_LOCK:
        current = read_episode_labels(directory, episode_id)
        if current is not None and current["labels"] == clean:
            return current
        history = []
        if current is not None:
            history = list(current.get("history") or [])
            history.append(
                {
                    "revision": current["revision"],
                    "labeled_at": current.get("labeled_at"),
                    "labels": current["labels"],
                }
            )
        record = {
            "schema": LABELS_SCHEMA,
            "id": episode_id,
            "revision": 1 if current is None else current["revision"] + 1,
            "labeled_at": datetime.now(timezone.utc).isoformat(),
            "labels": clean,
            "history": history,
        }
        temporary = directory / f".{episode_id}.labels.partial.json"
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(_labels_path(directory, episode_id))
    return record


def episode_label_form(directory: str | Path, episode_id: str) -> dict:
    """One episode with its labels, prefilled suggestions and earlier choices.

    Suggestions for an unlabeled take: the recorded outcome, the most recent
    project label, the model label last used for the same checkpoint, else the
    bundle's label hint (W&B/provenance run name and step).
    """
    episodes = list_rollout_episodes(directory)
    episode = next((item for item in episodes if item["id"] == episode_id), None)
    if episode is None:
        raise FileNotFoundError("Unknown or still-recording rollout episode")
    projects, models = [], []
    project = model = None
    for other in episodes:
        labels = other.get("labels")
        if not labels:
            continue
        if labels["project"] and labels["project"] not in projects:
            projects.append(labels["project"])
        if labels["model"] and labels["model"] not in models:
            models.append(labels["model"])
        if other["id"] == episode_id:
            continue
        project = project or labels["project"] or None
        if (
            model is None
            and labels["model"]
            and episode.get("checkpoint")
            and other.get("checkpoint") == episode.get("checkpoint")
        ):
            model = labels["model"]
    if episode["labels"] is not None:
        suggested = dict(episode["labels"])
    else:
        hint = (episode.get("model") or {}).get("label_hint")
        suggested = {
            "outcome": episode.get("recorded_outcome", "unlabeled"),
            "partial_score": None,
            "partial_score_text": "",
            "project": project or "",
            "model": model or (hint if isinstance(hint, str) else ""),
            "notes": "",
        }
    return {
        "episode": episode,
        "suggested": suggested,
        "choices": {"projects": projects[:50], "models": models[:50]},
    }


INDEX_COLUMNS = (
    "id",
    "created_at",
    "frames",
    "duration_seconds",
    "complete",
    "end_reason",
    "outcome",
    "partial_score",
    "partial_score_text",
    "project",
    "model",
    "notes",
    "labels_revision",
    "checkpoint",
    "global_step",
    "checkpoint_sha256",
    "wandb_entity",
    "wandb_project",
    "wandb_run_id",
    "wandb_url",
    "video_id",
    "auto_started",
    "station_commit",
    "size_bytes",
)


def episode_index(directory: str | Path) -> list[dict]:
    """Flat rows joining each episode's recording, model identity and labels."""
    rows = []
    for episode in list_rollout_episodes(directory):
        labels = episode.get("labels") or {}
        model = episode.get("model") or {}
        checkpoint = model.get("checkpoint") or {}
        wandb = model.get("wandb") or {}
        runtime = model.get("runtime") or {}
        rows.append(
            {
                "id": episode["id"],
                "created_at": episode.get("created_at"),
                "frames": episode.get("frames"),
                "duration_seconds": episode.get("duration_seconds"),
                "complete": episode.get("complete"),
                "end_reason": episode.get("end_reason"),
                "outcome": episode.get("outcome"),
                "partial_score": labels.get("partial_score"),
                "partial_score_text": labels.get("partial_score_text", ""),
                "project": labels.get("project", ""),
                "model": labels.get("model", ""),
                "notes": labels.get("notes", ""),
                "labels_revision": episode.get("labels_revision", 0),
                "checkpoint": episode.get("checkpoint"),
                "global_step": checkpoint.get("global_step"),
                "checkpoint_sha256": checkpoint.get("sha256"),
                "wandb_entity": wandb.get("entity"),
                "wandb_project": wandb.get("project"),
                "wandb_run_id": wandb.get("run_id"),
                "wandb_url": wandb.get("url"),
                "video_id": episode.get("video_id"),
                "auto_started": episode.get("auto_started"),
                "station_commit": runtime.get("code_commit"),
                "size_bytes": episode.get("size_bytes"),
            }
        )
    return rows


def _attr(value):
    """HDF5-storable attribute: scalars as is, anything else as JSON text."""
    if value is None:
        return ""
    if isinstance(value, (bool, int, float, str, np.integer, np.floating)):
        return value
    return json.dumps(value, default=str, sort_keys=True)


class RolloutEpisodeRecorder:
    """One rollout episode at a time, written to disk by a background thread."""

    def __init__(self, camera_res, config: Mapping[str, object]) -> None:
        self.camera_res = {
            str(name): (int(height), int(width))
            for name, (height, width) in dict(camera_res).items()
        }
        if not self.camera_res:
            raise ValueError("A rollout episode needs at least one configured camera")
        self.config = validate_episode_recording(config)
        self.directory = Path(self.config["directory"])
        self._lock = threading.Lock()
        self._finished: list[dict] = []
        self._thread: threading.Thread | None = None
        self._queue: queue.Queue | None = None
        self._request: tuple | None = None
        self._episode_id: str | None = None
        self._failure: str | None = None
        self._failure_detail: str | None = None
        self._rows = 0
        self._plan_index = -1
        self._t0 = 0.0
        self._next_disk_check = 0.0

    @property
    def recording(self) -> bool:
        """True from ``start`` until ``close``/``discard`` (finalizing excluded)."""
        return self._episode_id is not None

    @property
    def busy(self) -> bool:
        """True while a closed episode is still being written out."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def frames(self) -> int:
        return self._rows

    @property
    def failure(self) -> str | None:
        """``low_disk`` or ``writer_error`` once recording can no longer continue."""
        return self._failure

    def free_gb(self) -> float:
        return shutil.disk_usage(self.directory).free / 1e9

    def start(
        self,
        metadata: Mapping[str, object] | None = None,
        documents: Mapping[str, str] | None = None,
    ) -> str:
        """Create the episode file; rows are accepted until ``close``.

        ``documents`` are small text files (the model bundle's resolved config,
        inference config, provenance) stored verbatim under ``rollout/model``.
        """
        if self.recording:
            raise RuntimeError("A rollout episode is already recording")
        if self.busy:
            raise RuntimeError("The previous rollout episode is still being saved")
        self.directory.mkdir(parents=True, exist_ok=True)
        free = self.free_gb()
        if free < self.config["min_free_gb"]:
            raise RuntimeError(
                f"Only {free:.1f} GB free under {self.directory}; episode recording "
                f"needs at least {self.config['min_free_gb']:g} GB"
            )
        episode_id = _episode_id()
        while (self.directory / f"{episode_id}.hdf5").exists() or (
            self.directory / f"{episode_id}.json"
        ).exists():
            time.sleep(0.001)
            episode_id = _episode_id()
        path = self.directory / f"{episode_id}.hdf5"
        writer = EpisodeWriter(path, self.camera_res)  # exclusive creation
        try:
            rollout = writer.file.create_group("rollout")
            rollout.create_dataset(
                "timestamps", (0,), maxshape=(None,), dtype="float64", chunks=(1024,)
            )
            rollout.create_dataset(
                "plan_index", (0,), maxshape=(None,), dtype="int32", chunks=(1024,)
            )
            rollout.create_dataset(
                "target_eepose",
                (0, 14),
                maxshape=(None, 14),
                dtype="float32",
                chunks=(256, 14),
            )
            plans = rollout.create_group("plans")
            for name, dtype in (
                ("start_row", "int64"),
                ("executed_rows", "int32"),
                ("horizon", "int32"),
                ("inference_seconds", "float64"),
            ):
                plans.create_dataset(
                    name, (0,), maxshape=(None,), dtype=dtype, chunks=(256,)
                )
            plans.create_dataset(
                "info", (0,), maxshape=(None,), dtype=h5py.string_dtype(), chunks=(256,)
            )
            model = rollout.create_group("model")
            for name, text in dict(documents or {}).items():
                if not isinstance(name, str) or not name.isidentifier():
                    raise ValueError(f"Invalid episode document name {name!r}")
                model.create_dataset(name, data=str(text), dtype=h5py.string_dtype())
            created_at = datetime.now(timezone.utc).isoformat()
            attrs = {
                "kind": "rollout",
                "rollout_id": episode_id,
                "created_at": created_at,
                "outcome": "unlabeled",
                "end_reason": "recording",
                "cameras": list(self.camera_res),
            }
            attrs.update(dict(metadata or {}))
            for key, value in attrs.items():
                writer.file.attrs[key] = _attr(value)
            writer.file.flush()
        except BaseException:
            writer.close(complete=False)
            path.unlink(missing_ok=True)
            raise
        self._queue = queue.Queue(maxsize=self.config["max_queued_rows"])
        self._request = None
        self._episode_id = episode_id
        self._failure = self._failure_detail = None
        self._rows, self._plan_index = 0, -1
        self._t0 = time.monotonic()
        self._next_disk_check = self._t0 + _DISK_CHECK_SECONDS
        metadata = dict(metadata or {})
        manifest = {
            "id": episode_id,
            "filename": path.name,
            "created_at": created_at,
            "checkpoint": metadata.get("checkpoint"),
            "cameras": list(self.camera_res),
            "auto_started": bool(metadata.get("auto_started", False)),
            "video_id": metadata.get("video_id"),
            "model": metadata.get("model"),
            "session": metadata.get("session"),
            "upload": metadata.get("upload"),
        }
        self._thread = threading.Thread(
            target=self._run,
            args=(writer, self._queue, manifest),
            name=f"rollout_episode_{episode_id}",
            daemon=False,
        )
        self._thread.start()
        return episode_id

    def _enqueue(self, item) -> bool:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            self._failure = "writer_error"
            self._failure_detail = (
                f"the writer fell {self.config['max_queued_rows']} rows behind the "
                "control loop (disk too slow)"
            )
            return False
        return True

    def append(
        self, obs, joints, eepose, target=None, now: float | None = None
    ) -> bool:
        """Queue one executed control tick. Never blocks the control loop."""
        if not self.recording or self._failure is not None:
            return False
        now = time.monotonic() if now is None else float(now)
        if now >= self._next_disk_check:
            self._next_disk_check = now + _DISK_CHECK_SECONDS
            try:
                free = self.free_gb()
            except OSError as error:
                free, self._failure_detail = 0.0, str(error)
            if free < self.config["min_free_gb"]:
                self._failure = "low_disk"
                self._failure_detail = self._failure_detail or (
                    f"free space fell to {free:.1f} GB"
                )
                return False
        row = {
            "joint_positions": np.array(obs["joint_positions"], dtype=np.float64),
            "ee_poses": np.array(obs["ee_poses"], dtype=np.float64),
        }
        for name in self.camera_res:
            frame = obs.get(name)
            row[name] = None if frame is None else np.array(frame, copy=True)
        target = (
            np.full(14, np.nan)
            if target is None
            else np.array(target, dtype=np.float64)
        )
        item = (
            "row",
            row,
            np.array(joints, dtype=np.float64),
            np.array(eepose, dtype=np.float64),
            target,
            now - self._t0,
            self._plan_index,
        )
        if not self._enqueue(item):
            return False
        self._rows += 1
        return True

    def mark_plan(
        self, actions, executed_rows: int, inference_seconds=None, info=None
    ) -> bool:
        """Record one policy plan; rows that follow carry its index."""
        if not self.recording or self._failure is not None:
            return False
        actions = np.array(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise ValueError("A rollout plan must be shaped (H, 14)")
        seconds = math.nan if inference_seconds is None else float(inference_seconds)
        item = (
            "plan",
            actions,
            int(executed_rows),
            seconds,
            json.dumps(info or {}, default=str, sort_keys=True),
            self._rows,
        )
        if not self._enqueue(item):
            return False
        self._plan_index += 1
        return True

    def close(
        self,
        outcome: str = "unlabeled",
        complete: bool = True,
        end_reason: str = "saved",
    ):
        """Stop accepting rows and finalize in the background; return the episode id."""
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}")
        if end_reason not in END_REASONS:
            raise ValueError(f"end_reason must be one of {END_REASONS}")
        if not self.recording:
            return None
        if self._failure is not None:
            complete, end_reason = False, self._failure
        episode_id, self._episode_id = self._episode_id, None
        self._end(("close", outcome, bool(complete), end_reason, self._failure_detail))
        return episode_id

    def discard(self):
        """Stop recording and delete the episode file; return the episode id."""
        if not self.recording:
            return None
        episode_id, self._episode_id = self._episode_id, None
        self._end(("discard",))
        return episode_id

    def _end(self, request: tuple) -> None:
        """Hand the end request to the writer without ever blocking the control loop.

        The writer reads ``_request`` once its queue runs dry, so the request is
        never lost; the sentinel only shortens that wait when the queue has room.
        A blocking put here would stall control for as long as one HDF5 write
        takes on a slow disk, which is exactly when the queue is full.
        """
        with self._lock:
            self._request = request
        try:
            self._queue.put_nowait(("end",))
        except queue.Full:
            pass

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for the writer to finish; return whether it did."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
        return not self.busy

    def take_finished(self) -> list[dict]:
        """Return and clear the results of episodes finalized since the last call."""
        with self._lock:
            finished, self._finished = self._finished, []
        return finished

    def _run(self, writer: EpisodeWriter, items: queue.Queue, manifest: dict) -> None:
        error = None
        first_timestamp = last_timestamp = None
        plans = writer.file["rollout/plans"]
        while True:
            try:
                item = items.get(timeout=_END_POLL_SECONDS)
            except queue.Empty:
                with self._lock:
                    ended = self._request is not None
                if ended:
                    break  # the sentinel was dropped on a full queue; it is drained now
                continue
            kind = item[0]
            if kind == "end":
                break
            if error is not None:
                continue  # keep draining so the control loop never blocks on put
            try:
                if kind == "row":
                    _, row, joints, eepose, target, timestamp, plan_index = item
                    writer.append(row, joints, eepose)
                    n = writer.frames
                    rollout = writer.file["rollout"]
                    for name, value in (
                        ("timestamps", timestamp),
                        ("plan_index", plan_index),
                        ("target_eepose", target),
                    ):
                        dataset = rollout[name]
                        dataset.resize(n, axis=0)
                        dataset[n - 1] = value
                    first_timestamp = (
                        timestamp if first_timestamp is None else first_timestamp
                    )
                    last_timestamp = timestamp
                else:
                    _, actions, executed, seconds, info, start_row = item
                    if "actions" not in plans:
                        plans.create_dataset(
                            "actions",
                            (0, *actions.shape),
                            maxshape=(None, *actions.shape),
                            dtype="float32",
                            chunks=(1, *actions.shape),
                        )
                    stored = plans["actions"]
                    horizon = stored.shape[1]
                    padded = np.full((horizon, 14), np.nan, dtype=np.float32)
                    padded[: min(horizon, len(actions))] = actions[:horizon]
                    p = stored.shape[0] + 1
                    for name, value in (
                        ("actions", padded),
                        ("start_row", start_row),
                        ("executed_rows", executed),
                        ("horizon", len(actions)),
                        ("inference_seconds", seconds),
                        ("info", info),
                    ):
                        plans[name].resize(p, axis=0)
                        plans[name][p - 1] = value
            except Exception as caught:  # surfaced to the loop through ``failure``
                error = f"{type(caught).__name__}: {caught}"
                self._failure, self._failure_detail = "writer_error", error
        with self._lock:
            request = self._request
        result = {
            "id": manifest["id"],
            "saved": None,
            "discarded": False,
            "error": None,
        }
        path = writer.path
        try:
            if request[0] == "discard":
                writer.close(complete=False)
                path.unlink(missing_ok=True)
                result["discarded"] = True
            else:
                _, outcome, complete, end_reason, detail = request
                if error is not None:
                    complete, end_reason, detail = False, "writer_error", error
                frames = writer.frames
                ended_at = datetime.now(timezone.utc).isoformat()
                duration = (
                    0.0 if first_timestamp is None else last_timestamp - first_timestamp
                )
                attrs = {
                    "outcome": outcome,
                    "end_reason": end_reason,
                    "ended_at": ended_at,
                    "frames": frames,
                    "duration_seconds": duration,
                    "end_detail": detail or "",
                }
                for key, value in attrs.items():
                    writer.file.attrs[key] = _attr(value)
                plan_count = plans["start_row"].shape[0]
                writer.close(complete=complete)
                if frames == 0:
                    path.unlink(missing_ok=True)
                else:
                    payload = {
                        **manifest,
                        "frames": frames,
                        "plans": plan_count,
                        "duration_seconds": duration,
                        "outcome": outcome,
                        "complete": complete,
                        "end_reason": end_reason,
                        "end_detail": detail or "",
                        "ended_at": ended_at,
                    }
                    final = self.directory / f"{manifest['id']}.json"
                    temporary = self.directory / f".{manifest['id']}.partial.json"
                    temporary.write_text(json.dumps(payload, indent=2) + "\n")
                    temporary.replace(final)
                    result["saved"] = payload
        except Exception as caught:
            result["error"] = f"{type(caught).__name__}: {caught}"
            if writer.file is not None:
                try:
                    writer.close(complete=False)
                except Exception:
                    pass
        with self._lock:
            self._finished.append(result)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Print the rollout episode index (recording, model/W&B identity "
        "and operator labels). Reads JSON sidecars only; opens no device."
    )
    parser.add_argument("directory", help="episode_recording.directory")
    parser.add_argument("--format", choices=("table", "csv", "jsonl"), default="table")
    args = parser.parse_args(argv)
    rows = episode_index(args.directory)
    if args.format == "jsonl":
        for row in rows:
            print(json.dumps(row, sort_keys=True))
    elif args.format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    else:
        for row in rows:
            score = row["partial_score_text"] or "-"
            print(
                f"{row['id']}  {row['outcome']:<9}  score {score:<6}  "
                f"{row['project'] or '-'} | {row['model'] or '-'} | "
                f"W&B {row['wandb_run_id'] or '-'} | {row['frames']} frames"
            )
        if not rows:
            print("No finalized rollout episodes.")


if __name__ == "__main__":
    main()
