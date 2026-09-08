"""Real bash/child regression for Slurm batch-shell-only signal delivery."""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
HELPER = ROOT / "scripts/ice/relay_batch_signals.sh"
CHILD = r"""
import json, os, pathlib, signal, sys, time
root = pathlib.Path(sys.argv[1])
root.joinpath("started").write_text(str(os.getpid()))
if sys.argv[2] == "early":
    raise SystemExit(int(sys.argv[3]))
time.sleep(float(sys.argv[4]))
def handle(received, frame):
    root.joinpath("received").write_text(signal.Signals(received).name)
    if received == signal.SIGUSR1 and sys.argv[2] != "boundary":
        root.joinpath("COMPLETE.json").write_text("complete")
    time.sleep(.05)
    raise SystemExit(0 if received == signal.SIGUSR1 else 128 + received)
for received in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
    signal.signal(received, handle)
pathlib.Path(os.environ["ICE_RUNNER_SIGNAL_READY_FILE"]).write_text(str(os.getpid()))
if sys.argv[2] == "normal":
    raise SystemExit(int(sys.argv[3]))
time.sleep(5)
"""


def wait_file(path):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if path.exists() and path.stat().st_size:
            return
        time.sleep(0.01)
    raise AssertionError(f"child readiness timeout: {path}")


def launch(root, mode="signal", exit_code=0, delay=0):
    argv = [sys.executable, "-c", CHILD, str(root), mode, str(exit_code), str(delay)]
    command = (
        f"set -euo pipefail; source {shlex.quote(str(HELPER))}; "
        f"ice_run_with_signal_relay {shlex.quote(str(root / 'ready'))} "
        f"{shlex.join(argv)}; "
        f'if test "$ICE_BATCH_BOUNDARY_FORWARDED" = 1 && test ! -s {shlex.quote(str(root / "COMPLETE.json"))}; '
        "then printf 'REQUEUE_ACCEPTED\\n'; exit 0; fi; "
        "printf 'POST_RUNNER_VERIFIER\\n'"
    )
    return subprocess.Popen(
        ["bash", "-c", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def assert_reaped(root):
    pid = int((root / "started").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.parametrize("received", [signal.SIGUSR1, signal.SIGTERM, signal.SIGINT])
@pytest.mark.parametrize("during_startup", [False, True])
def test_exact_signal_reaches_child_and_wait_preserves_status(
    tmp_path, received, during_startup
):
    child = launch(tmp_path, delay=0.2 if during_startup else 0)
    try:
        wait_file(tmp_path / ("started" if during_startup else "ready"))
        child.send_signal(received)
        output, errors = child.communicate(timeout=4)
        expected = 0 if received == signal.SIGUSR1 else 128 + received
        assert child.returncode == expected, (output, errors)
        assert (tmp_path / "received").read_text() == received.name
        assert ("POST_RUNNER_VERIFIER" in output) == (expected == 0)
        assert_reaped(tmp_path)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=3)


@pytest.mark.parametrize("mode", ["normal", "early"])
@pytest.mark.parametrize("exit_code", [0, 7])
def test_early_and_normal_exit_status_is_not_replaced_by_wait(
    tmp_path, mode, exit_code
):
    child = launch(tmp_path, mode=mode, exit_code=exit_code)
    output, errors = child.communicate(timeout=4)
    assert child.returncode == exit_code, (output, errors)
    assert ("POST_RUNNER_VERIFIER" in output) == (exit_code == 0)
    assert_reaped(tmp_path)


def test_existing_readiness_marker_refuses_before_launch(tmp_path):
    (tmp_path / "ready").write_text("stale")
    child = launch(tmp_path)
    output, errors = child.communicate(timeout=3)
    assert child.returncode == 73, (output, errors)
    assert not (tmp_path / "started").exists()
    assert "POST_RUNNER_VERIFIER" not in output


def test_successful_requeue_without_completion_does_not_run_final_verifier(tmp_path):
    child = launch(tmp_path, mode="boundary")
    wait_file(tmp_path / "ready")
    child.send_signal(signal.SIGUSR1)
    output, errors = child.communicate(timeout=4)
    assert child.returncode == 0, (output, errors)
    assert "REQUEUE_ACCEPTED" in output
    assert "POST_RUNNER_VERIFIER" not in output
    assert_reaped(tmp_path)


def test_actual_runner_publishes_readiness_before_command_validation(tmp_path):
    ready = tmp_path / "ready.json"
    environment = {**os.environ, "ICE_RUNNER_SIGNAL_READY_FILE": str(ready)}
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/ice/ice_requeue_runner.py"),
            "--state-dir",
            str(tmp_path / "state"),
            "--checkpoint-glob",
            str(tmp_path / "*.ckpt"),
            "--checkpoint-signal",
            "USR2",
            "--checkpoint-forwarding",
            "slurm-steps",
            "--requeue-owner",
            "runner",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert result.returncode != 0
    payload = json.loads(ready.read_text())
    assert payload["signal_handlers_ready"] is True
    assert payload["pid"] > 0
    launcher = (ROOT / "scripts/train/launch_action_flow_usocket.sbatch").read_text()
    assert 'source "$AF_REPO/scripts/ice/relay_batch_signals.sh"' in launcher
    assert 'ice_run_with_signal_relay "$ATTEMPT/runner-signal-ready.json"' in launcher
