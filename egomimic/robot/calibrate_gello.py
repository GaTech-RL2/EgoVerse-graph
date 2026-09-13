"""Interactively capture one passive USB/Dynamixel GELLO calibration.

This utility never creates or commands a Yam follower. It only opens the
selected GELLO leader through the read-only adapter, which disables leader
torque before reading encoder positions.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import yaml
from omegaconf import OmegaConf

from egomimic.robot.collect_gello import (
    KeyEdge,
    TerminalKeyView,
    validate_gello_config,
)
from egomimic.robot.yam.gello import (
    DynamixelPositionBus,
    list_serial_ports,
    validate_serial_ports,
)


class GelloCalibration:
    """Captured raw offsets/endpoints and manually selected joint signs."""

    def __init__(self, arm: str, leader: Mapping) -> None:
        self.arm = str(arm)
        self.port = str(leader["port"])
        self.joint_ids = tuple(int(value) for value in leader["joint_ids"])
        self.gripper_id = int(leader["gripper_id"])
        self.joint_signs = np.asarray(leader["joint_signs"], dtype=int).copy()
        if self.arm not in ("left", "right"):
            raise ValueError("GELLO calibration arm must be left or right")
        if len(self.joint_ids) != 6 or self.joint_signs.shape != (6,):
            raise ValueError("GELLO calibration needs six joint IDs and signs")
        if not np.all(np.isin(self.joint_signs, (-1, 1))):
            raise ValueError("GELLO joint signs must be -1 or 1")
        self.raw = None
        self.joint_offsets = None
        self.gripper_open = None
        self.gripper_closed = None

    def update(self, raw) -> np.ndarray:
        raw = np.asarray(raw, dtype=float)
        if raw.shape != (7,) or not np.isfinite(raw).all():
            raise ValueError("GELLO calibration requires seven finite raw radians")
        self.raw = raw.copy()
        return self.raw

    def _sample(self) -> np.ndarray:
        if self.raw is None:
            raise RuntimeError("Wait for a GELLO sample before capturing calibration")
        return self.raw

    def capture_joint_offsets(self) -> None:
        self.joint_offsets = self._sample()[:6].copy()

    def capture_gripper_open(self) -> None:
        self.gripper_open = float(self._sample()[6])

    def capture_gripper_closed(self) -> None:
        self.gripper_closed = float(self._sample()[6])

    def toggle_joint_sign(self, index: int) -> None:
        if not 0 <= index < 6:
            raise ValueError("GELLO joint index must be in [0, 5]")
        self.joint_signs[index] *= -1

    @property
    def complete(self) -> bool:
        return (
            self.joint_offsets is not None
            and self.gripper_open is not None
            and self.gripper_closed is not None
            and not np.isclose(self.gripper_open, self.gripper_closed)
        )

    def calibration_mapping(self) -> dict:
        if not self.complete:
            raise RuntimeError(
                "Capture joint zero, gripper open, and gripper closed before export"
            )
        return {
            "gello": {
                "leaders": {
                    self.arm: {
                        "port": self.port,
                        "joint_ids": list(self.joint_ids),
                        "gripper_id": self.gripper_id,
                        "joint_signs": [int(value) for value in self.joint_signs],
                        "joint_offsets_rad": [
                            float(value) for value in self.joint_offsets
                        ],
                        "gripper_open_rad": float(self.gripper_open),
                        "gripper_closed_rad": float(self.gripper_closed),
                    }
                }
            }
        }

    def yaml_text(self) -> str:
        return yaml.safe_dump(self.calibration_mapping(), sort_keys=False)

    def write_yaml(self, path: str | Path) -> Path:
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing calibration: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.yaml_text())
        return path


def _leader(config: Mapping, arm: str) -> dict:
    leaders = config["gello"]["leaders"]
    if arm not in leaders:
        raise ValueError(f"GELLO config does not define {arm} leader")
    return dict(leaders[arm])


def _bus(config: Mapping, leader: Mapping) -> DynamixelPositionBus:
    return DynamixelPositionBus(
        port=leader["port"],
        ids=(*leader["joint_ids"], leader["gripper_id"]),
        **dict(config["gello"]["bus"]),
    )


def _status(calibration: GelloCalibration) -> str:
    raw = (
        "waiting"
        if calibration.raw is None
        else " ".join(f"{value:+.3f}" for value in calibration.raw)
    )
    captured = ", ".join(
        (
            f"zero={'yes' if calibration.joint_offsets is not None else 'no'}",
            f"open={'yes' if calibration.gripper_open is not None else 'no'}",
            f"closed={'yes' if calibration.gripper_closed is not None else 'no'}",
        )
    )
    signs = " ".join(f"{value:+d}" for value in calibration.joint_signs)
    return f"\r{calibration.arm} raw [ {raw} ] | signs [ {signs} ] | {captured}"


def _handle_key(
    key: str,
    calibration: GelloCalibration,
    output: str | Path | None,
    emit: Callable[[str], None],
) -> bool:
    """Handle one key; return true only when the session should exit."""
    key = key.lower()
    if key in ("q", "\x1b"):
        return True
    try:
        if key == "z":
            calibration.capture_joint_offsets()
            emit("Captured six joint zero offsets.")
        elif key == "o":
            calibration.capture_gripper_open()
            emit("Captured gripper open endpoint.")
        elif key == "c":
            calibration.capture_gripper_closed()
            emit("Captured gripper closed endpoint.")
        elif key in "123456":
            index = int(key) - 1
            calibration.toggle_joint_sign(index)
            emit(f"Joint {index + 1} sign is now {calibration.joint_signs[index]:+d}.")
        elif key == "p":
            emit("\n" + calibration.yaml_text())
        elif key == "w":
            if output is None:
                emit(
                    "Pass --output PATH to write a calibration file; use p to print it."
                )
            else:
                emit(f"Wrote {calibration.write_yaml(output)}")
        else:
            emit("Keys: z=zero o=open c=closed 1-6=toggle sign p=print w=write q=quit")
    except (FileExistsError, RuntimeError, ValueError) as error:
        emit(f"Calibration not exported: {error}")
    return False


def run_calibration(
    bus,
    calibration: GelloCalibration,
    view,
    rate_hz: float,
    output: str | Path | None = None,
    max_steps: int | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    time_fn: Callable[[], float] = time.monotonic,
    emit: Callable[..., None] = print,
) -> int:
    """Read one passive leader and capture calibration with terminal keys."""
    rate_hz = float(rate_hz)
    if not np.isfinite(rate_hz) or rate_hz <= 0:
        raise ValueError("Calibration rate_hz must be positive")
    edge, steps = KeyEdge(), 0
    emit("Keys: z=zero o=open c=closed 1-6=toggle sign p=print w=write q=quit")
    while max_steps is None or steps < max_steps:
        tick = time_fn()
        calibration.update(bus.read_radians())
        emit(_status(calibration), end="", flush=True)
        key = edge.update(view.update({}, recording=False))
        if key is not None and _handle_key(key, calibration, output, emit):
            break
        steps += 1
        sleep_fn(max(0.0, 1 / rate_hz - (time_fn() - tick)))
    emit("")
    return steps


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAM GELLO collection YAML")
    parser.add_argument("--arm", choices=("left", "right"))
    parser.add_argument("--output", help="new YAML snippet path written only by w")
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="validate YAML only; open no serial, CAN, camera, or robot device",
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="list stable serial paths without opening a device",
    )
    args = parser.parse_args(argv)
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    config = validate_gello_config(config)
    if args.list_ports:
        ports = list_serial_ports()
        print("\n".join(ports) if ports else "No stable serial devices found.")
        return 0
    if args.check_config:
        print("GELLO calibration configuration is valid; no devices were opened.")
        return 0
    if args.arm is None:
        parser.error("--arm is required for interactive calibration")

    leader = _leader(config, args.arm)
    validate_serial_ports((leader["port"],))
    calibration = GelloCalibration(args.arm, leader)
    bus = view = None
    try:
        view = TerminalKeyView()
        bus = _bus(config, leader)
        run_calibration(bus, calibration, view, args.rate_hz, output=args.output)
    finally:
        errors = []
        if view is not None:
            try:
                view.close()
            except Exception as error:
                errors.append(error)
        if bus is not None:
            try:
                bus.close()
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("GELLO calibration cleanup failed", errors)
    return 0


if __name__ == "__main__":
    main()
