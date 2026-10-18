"""Mirror failures never terminate training; final persistence remains strict."""

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

COMMON = Path(__file__).resolve().parents[1] / "scripts/train/log_mirror.sh"


def command(script):
    return ["/bin/bash", "-c", f"source {shlex.quote(str(COMMON))}; {script}"]


def fixture(tmp_path, plan):
    tools = tmp_path / "bin"
    tools.mkdir()
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    tool = (
        f"#!{sys.executable}\n"
        + r"""
import fcntl,json,os,pathlib,subprocess,sys,time
root=pathlib.Path(os.environ['MIRROR_TEST_ROOT'])
kind=pathlib.Path(sys.argv[0]).name
if kind=='flock':
    assert sys.argv[1:3]==['--wait','330']
    fcntl.flock(int(sys.argv[3]),fcntl.LOCK_EX)
elif kind=='timeout':
    assert sys.argv[1:3]==['--kill-after=15s','300s']
    counter=root/'attempts'
    count=int(counter.read_text()) if counter.exists() else 0
    counter.write_text(str(count+1))
    plan=json.loads((root/'plan.json').read_text())
    code=plan[min(count,len(plan)-1)]
    if code: sys.exit(code)
    os.execvp(sys.argv[3],sys.argv[3:])
elif kind=='rsync':
    with (root/'rsync-arguments.jsonl').open('a') as stream:
        stream.write(json.dumps(sys.argv[1:])+'\n')
    with (root/'events').open('a') as stream: stream.write('start\n')
    time.sleep(float(os.environ.get('MIRROR_TEST_COPY_SECONDS','0')))
    actual=os.environ.get('MIRROR_TEST_REAL_RSYNC')
    if actual:
        result=subprocess.run([actual,*sys.argv[1:]])
        if result.returncode: sys.exit(result.returncode)
    with (root/'events').open('a') as stream: stream.write('end\n')
elif kind=='sleep':
    time.sleep(min(float(sys.argv[1]),.02))
"""
    )
    for name in ("flock", "timeout", "rsync", "sleep"):
        path = tools / name
        path.write_text(tool)
        path.chmod(0o755)
    source, destination = tmp_path / "local", tmp_path / "shared"
    source.mkdir()
    env = dict(
        os.environ,
        PATH=str(tools) + os.pathsep + os.environ["PATH"],
        MIRROR_TEST_ROOT=str(tmp_path),
    )
    return source, destination, env


def run_mirror(source, destination, env, attempts=3):
    script = f"mirror_once {shlex.quote(str(source))} {shlex.quote(str(destination))} {attempts}"
    return subprocess.run(
        command(script), env=env, capture_output=True, text=True, timeout=10
    )


@pytest.mark.parametrize("code", [20, 24, 124])
def test_final_flush_recovers_from_interrupted_vanished_or_timeout_copy(tmp_path, code):
    source, destination, env = fixture(tmp_path, [code, 0])
    result = run_mirror(source, destination, env)
    assert result.returncode == 0, result.stderr
    health = json.loads((source / ".mirror-health.json").read_text())
    assert health["schema"] == "training-log-mirror-health-v1"
    assert health["last_exit_code"] == 0
    assert health["last_success_unix"] > 0
    assert health["consecutive_failures"] == 0
    assert (tmp_path / "attempts").read_text() == "2"
    assert not list(source.glob(".mirror-health.tmp.*"))


def test_final_failure_is_nonzero_and_health_records_all_failures(tmp_path):
    source, destination, env = fixture(tmp_path, [20, 24, 124])
    result = run_mirror(source, destination, env)
    assert result.returncode == 124
    health = json.loads((source / ".mirror-health.json").read_text())
    assert health["last_exit_code"] == 124
    assert health["last_success_unix"] == 0
    assert health["consecutive_failures"] == 3
    assert (tmp_path / "attempts").read_text() == "3"


def test_periodic_failures_and_recovery_never_kill_healthy_trainer(tmp_path):
    source, destination, env = fixture(tmp_path, [20, 24, 124, 0])
    admission = tmp_path / "admitted"
    admission.touch()
    failure = tmp_path / "MIRROR_FAILED"
    child = subprocess.Popen(["/bin/sleep", "30"])
    args = [str(child.pid), str(source), str(destination), str(admission), ".01"]
    watcher = subprocess.Popen(
        command("watch_log_mirror " + " ".join(map(shlex.quote, args))),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            receipt = source / ".mirror-health.json"
            if receipt.exists():
                health = json.loads(receipt.read_text())
                if health["last_success_unix"] > 0:
                    break
            assert child.poll() is None
            time.sleep(0.01)
        else:
            pytest.fail("Periodic watcher never recovered")
        assert child.poll() is None
        assert not failure.exists()
        assert health["consecutive_failures"] == 0
        child.terminate()
        child.wait(timeout=3)
        assert watcher.wait(timeout=3) == 0
        _, errors = watcher.communicate()
        assert "training continues" in errors
    finally:
        for process in (watcher, child):
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=3)


def test_transfer_excludes_only_disposable_cache_lock_and_temp_receipts(tmp_path):
    actual = shutil.which("rsync")
    if actual is None:
        pytest.skip("Local rsync unavailable")
    source, destination, env = fixture(tmp_path, [0])
    env["MIRROR_TEST_REAL_RSYNC"] = actual
    (source / "wandb-cache").mkdir()
    (source / "wandb-cache" / "disposable").write_text("cache")
    (source / "wandb").mkdir()
    (source / "wandb" / "run.wandb").write_text("important spool")
    (source / "metrics.csv").write_text("step,loss\n12001,0.2\n")
    (source / ".mirror-health.tmp.orphan").write_text("partial")
    destination.mkdir()
    (destination / "historical.txt").write_text("retained")
    result = run_mirror(source, destination, env)
    assert result.returncode == 0, result.stderr
    assert (destination / "metrics.csv").read_bytes() == (
        source / "metrics.csv"
    ).read_bytes()
    assert (destination / "wandb" / "run.wandb").read_text() == "important spool"
    assert (destination / "historical.txt").read_text() == "retained"
    assert not (destination / "wandb-cache").exists()
    assert not (destination / ".mirror.lock").exists()
    assert not (destination / ".mirror-health.tmp.orphan").exists()
    # A second final flush updates the same CSV rather than ignoring it.
    (source / "metrics.csv").write_text("step,loss\n12001,0.2\n12002,0.19\n")
    assert run_mirror(source, destination, env).returncode == 0
    assert (destination / "metrics.csv").read_bytes() == (
        source / "metrics.csv"
    ).read_bytes()


def test_concurrent_calls_are_serialized(tmp_path):
    source, destination, env = fixture(tmp_path, [0])
    env["MIRROR_TEST_COPY_SECONDS"] = ".12"
    args = command(
        f"mirror_once {shlex.quote(str(source))} {shlex.quote(str(destination))} 1"
    )
    first = subprocess.Popen(args, env=env)
    second = subprocess.Popen(args, env=env)
    assert first.wait(timeout=5) == 0
    assert second.wait(timeout=5) == 0
    assert (tmp_path / "events").read_text().splitlines() == [
        "start",
        "end",
        "start",
        "end",
    ]
    assert (
        json.loads((source / ".mirror-health.json").read_text())["consecutive_failures"]
        == 0
    )


def test_shell_syntax():
    subprocess.run(["/bin/bash", "-n", str(COMMON)], check=True)
