"""Real Linux child delivery and fail-closed trainer identity regressions."""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from unittest import mock

import pytest
from egomimic.utils import slurm_process_identity as identity

pytestmark = pytest.mark.skipif(not hasattr(os, "pidfd_open"), reason="Linux pidfd required")


def test_real_child_pidfd_delivery(tmp_path):
    ready = tmp_path / "ready.json"
    hit = tmp_path / "received"
    child_code = """
import json, os, signal, sys, time
from pathlib import Path
from egomimic.utils.slurm_process_identity import ready_record
signal.signal(signal.SIGUSR2, lambda *_: Path(sys.argv[2]).write_text('USR2'))
Path(sys.argv[1]).write_text(json.dumps(ready_record('nonce', '123', 0)))
time.sleep(30)
"""
    source = identity.source_identity()
    child = subprocess.Popen([sys.executable, "-c", child_code, str(ready), str(hit)], cwd=source["root"])
    try:
        deadline = time.monotonic() + 15
        while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(.05)
        assert ready.exists()
        original = json.loads(ready.read_text())
        arguments = dict(pid=child.pid, nonce="nonce", job_id="123", restart=0,
                         parent_identity=identity.process_identity(os.getpid()),
                         expected_source=source, expected_process=identity.process_identity(child.pid))
        for field, bad in (("nonce", "forged"), ("job_id", "124"), ("restart_count", 1),
                           ("source", {}), ("handler_installed", False)):
            changed = copy.deepcopy(original); changed[field] = bad
            ready.write_text(json.dumps(changed))
            with pytest.raises(ValueError): identity.signal_ready_trainer(ready, **arguments)
        for field, bad in (("uid", [999]*4), ("pid", child.pid+1), ("starttime", 0),
                           ("pid_namespace", "forged"), ("cgroup", "forged"),
                           ("cwd", "/tmp"), ("exe", "/bin/false"), ("boot_id", "forged")):
            changed = copy.deepcopy(original); changed["process"][field] = bad
            ready.write_text(json.dumps(changed))
            with pytest.raises(ValueError): identity.signal_ready_trainer(ready, **arguments)
        ready.write_text(json.dumps(original))
        with mock.patch.object(identity, "source_identity", return_value={}):
            with pytest.raises(ValueError, match="source changed"):
                identity.signal_ready_trainer(ready, **arguments)
        assert not hit.exists(), "forged records must never deliver the signal"
        receipt = identity.signal_ready_trainer(ready, **arguments)
        deadline = time.monotonic() + 5
        while not hit.exists() and time.monotonic() < deadline: time.sleep(.05)
        assert hit.read_text() == "USR2"
        assert receipt["pid"] == child.pid
    finally:
        child.terminate(); child.wait(timeout=10)
