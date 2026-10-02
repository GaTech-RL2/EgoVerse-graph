"""Companion tests must exercise checkout code and never overwrite local work."""

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/restore_validation.py"
SPEC = importlib.util.spec_from_file_location("restore_validation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def checkout(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Validation fixture")
    git(tmp_path, "config", "user.email", "fixture@example.invalid")
    (tmp_path / "policy.py").write_text("source = 'companion snapshot'\n")
    (tmp_path / "tests").mkdir()
    payload = b"assert True\n"
    (tmp_path / "tests/test_policy.py").write_bytes(payload)
    (tmp_path / ".github").mkdir()
    manifest = {"tests/test_policy.py": hashlib.sha256(payload).hexdigest()}
    (tmp_path / ".github/validation-paths.json").write_text(json.dumps(manifest))
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "Companion snapshot")
    revision = git(tmp_path, "rev-parse", "HEAD")
    git(tmp_path, "rm", "-qr", "tests", ".github/validation-paths.json")
    (tmp_path / ".github").mkdir(exist_ok=True)
    (tmp_path / ".github/validation-ref").write_text(revision + "\n")
    (tmp_path / "policy.py").write_text("source = 'current checkout'\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "Runtime checkout with pinned validation")
    return tmp_path


def test_restore_is_pinned_idempotent_and_preserves_checkout_code(checkout):
    # A moving companion tip must not change this checkout's selected tests.
    git(checkout, "commit", "--allow-empty", "-qm", "A later commit")
    head = git(checkout, "rev-parse", "HEAD")
    assert MODULE.restore(checkout) == ["tests/test_policy.py"]
    assert MODULE.restore(checkout) == ["tests/test_policy.py"]
    assert (checkout / "tests/test_policy.py").read_text() == "assert True\n"
    assert (checkout / "policy.py").read_text() == "source = 'current checkout'\n"
    assert git(checkout, "rev-parse", "HEAD") == head
    assert git(checkout, "diff", "--exit-code") == ""


def test_restore_rejects_a_branch_name(checkout):
    (checkout / ".github/validation-ref").write_text("main\n")
    with pytest.raises(ValueError, match="full commit SHA"):
        MODULE.restore(checkout)
    assert not (checkout / "tests/test_policy.py").exists()


@pytest.mark.parametrize("tracked", [False, True])
def test_restore_never_overwrites_local_or_tracked_files(checkout, tracked):
    path = checkout / "tests/test_policy.py"
    path.parent.mkdir()
    path.write_text("local work\n")
    if tracked:
        git(checkout, "add", "tests")
    with pytest.raises(ValueError, match="would replace"):
        MODULE.restore(checkout)
    assert path.read_text() == "local work\n"


def test_restore_rejects_symlink_destinations(checkout, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")
    (checkout / "tests").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        MODULE.restore(checkout)
    assert not list(outside.iterdir())


@pytest.mark.parametrize("bad_path", ["../outside", ".git/config", "/absolute"])
def test_restore_rejects_paths_outside_the_checkout(checkout, monkeypatch, bad_path):
    original = MODULE.git

    def malicious_manifest(root, *args):
        if args[0] == "show":
            return json.dumps({bad_path: "0" * 64}).encode()
        return original(root, *args)

    monkeypatch.setattr(MODULE, "git", malicious_manifest)
    with pytest.raises(ValueError, match="Invalid validation manifest"):
        MODULE.restore(checkout)
    assert not (checkout / "tests/test_policy.py").exists()
