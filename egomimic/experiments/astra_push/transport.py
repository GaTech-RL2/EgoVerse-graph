"""Bounded JSON plus non-pickled NumPy arrays over a local policy socket."""

import io
import json
import socket
import struct
import threading
from pathlib import Path

import numpy as np

MAX_FRAME_BYTES = 4 * 1024 * 1024


def read_exact(connection, count):
    data = bytearray()
    while len(data) < count:
        part = connection.recv(count - len(data))
        if not part:
            raise EOFError("Policy transport closed before a complete frame")
        data.extend(part)
    return bytes(data)


def send_frame(connection, metadata, arrays):
    header = json.dumps(metadata, allow_nan=False).encode()
    buffer = io.BytesIO()
    if any(np.asarray(a).dtype.hasobject for a in arrays.values()):
        raise ValueError("Object arrays are forbidden")
    np.savez(buffer, **arrays)
    payload = buffer.getvalue()
    if len(header) > 4096 or len(payload) > MAX_FRAME_BYTES:
        raise ValueError("Policy frame exceeds the transport bound")
    connection.sendall(struct.pack("!II", len(header), len(payload)) + header + payload)


def receive_frame(connection):
    hlen, plen = struct.unpack("!II", read_exact(connection, 8))
    if hlen > 4096 or plen > MAX_FRAME_BYTES:
        raise ValueError("Policy frame exceeds the transport bound")
    metadata = json.loads(read_exact(connection, hlen))
    with np.load(io.BytesIO(read_exact(connection, plen)), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    return metadata, arrays


def validate_observation(metadata, arrays):
    expected = {
        "version",
        "kind",
        "request_id",
        "episode_id",
        "step",
        "timestamp",
        "policy_seed",
        "instruction",
    }
    if (
        set(metadata) != expected
        or metadata["version"] != 1
        or metadata["kind"] != "observation"
    ):
        raise ValueError("Policy request metadata differs from the protocol")
    if (
        type(metadata["step"]) is not int
        or not 0 <= metadata["step"] < 150
        or not np.isclose(metadata["timestamp"], metadata["step"] / 10, atol=1e-8)
    ):
        raise ValueError("Policy timestamp or control step is invalid")
    if (
        type(metadata["policy_seed"]) is not int
        or not isinstance(metadata["request_id"], str)
        or not isinstance(metadata["episode_id"], str)
    ):
        raise ValueError("Policy request identity/seed is invalid")
    if (
        not isinstance(metadata["instruction"], str)
        or not 1 <= len(metadata["instruction"].encode()) <= 254
    ):
        raise ValueError("Invalid policy language")
    if set(arrays) != {"external_rgb", "wrist_rgb", "proprioception"}:
        raise ValueError("Missing or privileged student observation fields")
    for key in ("external_rgb", "wrist_rgb"):
        if arrays[key].dtype != np.uint8 or arrays[key].shape != (224, 224, 3):
            raise ValueError("Policy camera shape/type differs")
    state = arrays["proprioception"]
    if state.dtype != np.float32 or state.shape != (9,) or not np.isfinite(state).all():
        raise ValueError("Policy proprioception shape/type differs")


def validate_actions(metadata, arrays, request_id):
    if metadata != {"version": 1, "kind": "actions", "request_id": request_id} or set(
        arrays
    ) != {"actions"}:
        raise ValueError("Policy response identity differs")
    actions = arrays["actions"]
    if (
        actions.dtype != np.float32
        or actions.shape != (10, 7)
        or not np.isfinite(actions).all()
    ):
        raise ValueError(
            "Policy must return ten finite float32 seven-dimensional commands"
        )
    return actions


class PolicyServer:
    """One inference thread; no training occurs while this context is active."""

    def __init__(self, path, predict, *, timeout=120):
        self.path, self.predict, self.timeout = Path(path), predict, timeout
        self.error = None
        self.closed = threading.Event()

    def __enter__(self):
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(str(self.path))
        self.listener.listen(1)
        self.listener.settimeout(1)
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def run(self):
        try:
            while not self.closed.is_set():
                try:
                    connection, _ = self.listener.accept()
                except TimeoutError:
                    continue
                with connection:
                    connection.settimeout(self.timeout)
                    previous = {}
                    while not self.closed.is_set():
                        try:
                            metadata, arrays = receive_frame(connection)
                        except EOFError:
                            break
                        validate_observation(metadata, arrays)
                        episode = metadata["episode_id"]
                        if metadata["step"] != previous.get(episode, -1) + 1:
                            raise ValueError(
                                "Policy requests were repeated or reordered"
                            )
                        previous[episode] = metadata["step"]
                        action = self.predict(metadata, arrays)
                        reply = {
                            "version": 1,
                            "kind": "actions",
                            "request_id": metadata["request_id"],
                        }
                        validate_actions(
                            reply, {"actions": action}, metadata["request_id"]
                        )
                        send_frame(connection, reply, {"actions": action})
        except BaseException as exc:
            self.error = exc

    def __exit__(self, exc_type, exc, tb):
        self.closed.set()
        self.thread.join(timeout=self.timeout + 2)
        self.listener.close()
        self.path.unlink(missing_ok=True)
        if self.thread.is_alive():
            raise RuntimeError("Policy worker failed to stop")
        if self.error and exc is None:
            raise RuntimeError("Policy transport failed") from self.error
