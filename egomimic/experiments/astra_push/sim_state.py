"""Versioned full simulator/controller snapshots without pickle.

Only the fixed single-Panda, non-interpolating OSC runtime is supported.
Observables, controller targets, gripper action, integration state, buffers and
RNG are restored together so counterfactual instructions share the same start.
"""

import hashlib
import json
import random
from pathlib import Path

import numpy as np

from egomimic.experiments.astra_push.artifacts import file_hash, publish_json

DATA_FIELDS = (
    "qpos",
    "qvel",
    "act",
    "qacc",
    "qacc_warmstart",
    "ctrl",
    "qfrc_applied",
    "xfrc_applied",
    "mocap_pos",
    "mocap_quat",
    "userdata",
    "eq_active",
)
OBS_FIELDS = (
    "_time_since_last_sample",
    "_current_delay",
    "_current_observed_value",
    "_sampled",
    "_is_number",
    "_data_shape",
)


def _encode(value, arrays):
    if isinstance(value, np.ndarray):
        key = f"a{len(arrays):05d}"
        arrays[key] = value.copy()
        if value.dtype.hasobject:
            raise TypeError("Object arrays are forbidden in simulator snapshots")
        return {"array": key}
    if isinstance(value, np.generic):
        return _encode(value.item(), arrays)
    if isinstance(value, tuple):
        return {"tuple": [_encode(v, arrays) for v in value]}
    if isinstance(value, list):
        return {"list": [_encode(v, arrays) for v in value]}
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {"dict": {k: _encode(v, arrays) for k, v in value.items()}}
    if value is None or isinstance(value, (str, bool, int, float)):
        return {"value": value}
    raise TypeError(f"Unsupported snapshot value: {type(value).__name__}")


def _decode(value, arrays):
    if set(value) == {"array"}:
        return arrays[value["array"]].copy()
    if set(value) == {"tuple"}:
        return tuple(_decode(v, arrays) for v in value["tuple"])
    if set(value) == {"list"}:
        return [_decode(v, arrays) for v in value["list"]]
    if set(value) == {"dict"}:
        return {k: _decode(v, arrays) for k, v in value["dict"].items()}
    if set(value) == {"value"}:
        return value["value"]
    raise ValueError("Invalid snapshot encoding")


def _model_hash(env):
    return hashlib.sha256(env.sim.model.get_xml().encode()).hexdigest()


def snapshot(environment, path):
    env, robot = environment.env, environment.robots[0]
    if (
        len(environment.robots) != 1
        or robot.controller.interpolator_pos is not None
        or robot.controller.interpolator_ori is not None
    ):
        raise ValueError(
            "Snapshot contract requires one non-interpolating OSC controller"
        )
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    state = {
        "data": {k: np.asarray(getattr(env.sim.data, k)).copy() for k in DATA_FIELDS},
        "time": float(env.sim.data.time),
        "controller": {k: v for k, v in vars(robot.controller).items() if k != "sim"},
        "robot_buffers": {
            k: vars(v) for k, v in vars(robot).items() if k.startswith("recent_")
        },
        "robot_torques": robot.torques,
        "gripper_action": robot.gripper.current_action,
        "environment": {
            k: getattr(env, k) for k in ("cur_time", "timestep", "done", "_obs_cache")
        },
        "observables": {
            k: {f: getattr(v, f) for f in OBS_FIELDS}
            for k, v in env._observables.items()
        },
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    arrays = {}
    encoded = _encode(state, arrays)
    with (path / "arrays.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    publish_json(
        path / "snapshot.json",
        {
            "schema_version": "astrapush-full-state-1",
            "model_xml_sha256": _model_hash(env),
            "arrays_sha256": file_hash(path / "arrays.npz"),
            "state": encoded,
        },
    )
    return file_hash(path / "snapshot.json")


def restore(environment, path):
    env, robot = environment.env, environment.robots[0]
    path = Path(path)
    metadata = json.loads((path / "snapshot.json").read_text())
    if (
        metadata["schema_version"] != "astrapush-full-state-1"
        or metadata["arrays_sha256"] != file_hash(path / "arrays.npz")
        or metadata["model_xml_sha256"] != _model_hash(env)
    ):
        raise ValueError("Snapshot content or simulator model differs")
    with np.load(path / "arrays.npz", allow_pickle=False) as arrays:
        state = _decode(metadata["state"], arrays)
    env.sim.data.time = state["time"]
    for k, value in state["data"].items():
        getattr(env.sim.data, k)[:] = value
    env.sim.forward()
    # forward recomputes derived fields; preserve integration warm-start state.
    for k in ("qacc", "qacc_warmstart", "ctrl", "qfrc_applied", "xfrc_applied"):
        getattr(env.sim.data, k)[:] = state["data"][k]
    for k, value in state["controller"].items():
        setattr(robot.controller, k, value)
    for k, attributes in state["robot_buffers"].items():
        for name, value in attributes.items():
            setattr(getattr(robot, k), name, value)
    robot.torques = state["robot_torques"]
    robot.gripper.current_action = state["gripper_action"]
    for k, value in state["environment"].items():
        setattr(env, k, value)
    for k, fields in state["observables"].items():
        for name, value in fields.items():
            setattr(env._observables[k], name, value)
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    return env._get_observations()
