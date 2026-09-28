"""Simulator-owned immutable episode artifacts; no learner/model imports."""

import json
import os
import uuid
from pathlib import Path

import h5py
import numpy as np

from egomimic.experiments.astra_push.artifacts import file_hash, publish_json

OBSERVATION_KEYS = {"external_rgb", "wrist_rgb", "proprioception", "timestamps"}


def write_episode(path, *, observations, actions, instruction, provenance, audit):
    """Row t pairs the observation before the executed command with action t."""
    if set(observations) != OBSERVATION_KEYS:
        raise ValueError("Episode observation keys differ from the student allowlist")
    if (
        not isinstance(instruction, str)
        or not instruction
        or len(instruction.encode("utf-8")) > 254
    ):
        raise ValueError("Instruction must fit the UTF-8 byte-language contract")
    actions = np.asarray(actions)
    count = len(actions)
    if (
        actions.dtype != np.float32
        or actions.shape != (count, 7)
        or not 1 <= count <= 150
        or not np.isfinite(actions).all()
        or np.abs(actions).max() > 1
    ):
        raise ValueError(
            "Executed actions must be finite bounded float32 [T,7], T<=150"
        )
    for key in ("external_rgb", "wrist_rgb"):
        image = np.asarray(observations[key])
        if image.dtype != np.uint8 or image.shape != (count, 224, 224, 3):
            raise ValueError("Raw cameras must be uint8 [T,224,224,3]")
    state = np.asarray(observations["proprioception"])
    timestamps = np.asarray(observations["timestamps"])
    if (
        state.shape != (count, 9)
        or not np.isfinite(state).all()
        or timestamps.shape != (count,)
        or not np.isfinite(timestamps).all()
    ):
        raise ValueError("Invalid proprioception/timestamp arrays")
    if count > 1 and not np.allclose(np.diff(timestamps), 0.1, atol=1e-6, rtol=0):
        raise ValueError(
            "Observation/action rows must be consecutive 10 Hz control steps"
        )
    required = {
        "episode_id",
        "scene_hash",
        "task_hash",
        "teacher_hash",
        "initial_state_hash",
        "phase",
        "arm",
        "round",
        "stage",
        "partition",
        "accepted",
    }
    if set(provenance) != required or type(provenance["accepted"]) is not bool:
        raise ValueError("Episode provenance must contain the complete declared fields")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    with h5py.File(temporary, "x") as output:
        output.attrs["schema_version"] = "astrapush-episode-1"
        output.attrs["instruction"] = instruction
        output.attrs["provenance"] = json.dumps(
            provenance, sort_keys=True, allow_nan=False
        )
        output.attrs["alignment"] = "observation_t_before_executed_action_t"
        for key, values in observations.items():
            output.create_dataset(
                f"observations/{key}",
                data=values,
                compression="gzip",
                compression_opts=1,
            )
        output.create_dataset("actions", data=actions)
        output.create_dataset(
            "audit/json",
            data=json.dumps(audit, sort_keys=True, allow_nan=False),
            dtype=h5py.string_dtype("utf-8"),
        )
        output.flush()
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()
    receipt = {
        "path": str(path.resolve()),
        "sha256": file_hash(path),
        "steps": count,
        "provenance": provenance,
    }
    publish_json(path.with_suffix(".json"), receipt)
    return receipt


def verified_episode(record):
    if file_hash(record["path"]) != record["sha256"]:
        raise ValueError("Episode hash differs from its frozen manifest")
    return h5py.File(record["path"], "r")
