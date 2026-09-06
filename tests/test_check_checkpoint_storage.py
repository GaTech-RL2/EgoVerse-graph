import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "train" / "check_checkpoint_storage.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_checkpoint_storage", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _filesystem(*, available_blocks=1000, free_blocks=1100, read_only=False):
    return SimpleNamespace(
        f_frsize=4096,
        f_bavail=available_blocks,
        f_bfree=free_blocks,
        f_blocks=2000,
        f_flag=int(getattr(os, "ST_RDONLY", 1)) if read_only else 0,
    )


def test_storage_budget_uses_larger_measured_or_expected_size(tmp_path, monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module.os, "statvfs", lambda _: _filesystem())
    report, failures = module.storage_evidence(
        tmp_path,
        planned_checkpoint_count=3,
        measured_checkpoint_bytes=100,
        expected_checkpoint_bytes=120,
        safety_reserve_bytes=500,
    )
    assert failures == []
    assert report["bytes_per_checkpoint_budget"] == 120
    assert report["checkpoint_budget_bytes"] == 360
    assert report["required_available_bytes"] == 860
    assert report["filesystem_available_bytes"] == 1000 * 4096


def test_storage_budget_reports_shortfall_and_read_only(tmp_path, monkeypatch):
    module = _load_module()
    monkeypatch.setattr(
        module.os,
        "statvfs",
        lambda _: _filesystem(available_blocks=1, free_blocks=2, read_only=True),
    )
    report, failures = module.storage_evidence(
        tmp_path,
        planned_checkpoint_count=3,
        measured_checkpoint_bytes=2000,
        expected_checkpoint_bytes=None,
        safety_reserve_bytes=1000,
    )
    assert report["required_available_bytes"] == 7000
    assert {failure["field"] for failure in failures} == {
        "filesystem_read_only",
        "filesystem_available_bytes",
    }


def test_storage_cli_writes_pass_evidence(tmp_path):
    output = tmp_path / "storage.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--target",
            str(tmp_path),
            "--planned-checkpoint-count",
            "1",
            "--measured-checkpoint-bytes",
            "1",
            "--expected-checkpoint-bytes",
            "2",
            "--safety-reserve-bytes",
            "1",
            "--output",
            str(output),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    payload = json.loads(output.read_text())
    assert payload["status"] == "CHECKPOINT_STORAGE_VALIDATED"
    assert payload["failures"] == []
    assert payload["storage"]["target"] == str(tmp_path.resolve())


def test_storage_cli_writes_failure_evidence_for_insufficient_space(tmp_path):
    available = os.statvfs(tmp_path).f_bavail * os.statvfs(tmp_path).f_frsize
    output = tmp_path / "storage.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--target",
            str(tmp_path),
            "--planned-checkpoint-count",
            "2",
            "--expected-checkpoint-bytes",
            str(available + 1),
            "--safety-reserve-bytes",
            "1",
            "--output",
            str(output),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 1
    payload = json.loads(output.read_text())
    assert payload["status"] == "CHECKPOINT_STORAGE_FAILED"
    failure = payload["failures"][0]
    assert failure["field"] == "filesystem_available_bytes"
    assert failure["shortfall_bytes"] > 0


def test_storage_cli_requires_checkpoint_size(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--target",
            str(tmp_path),
            "--planned-checkpoint-count",
            "1",
            "--safety-reserve-bytes",
            "1",
            "--output",
            str(tmp_path / "storage.json"),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode != 0
    assert "one or both" in completed.stderr
