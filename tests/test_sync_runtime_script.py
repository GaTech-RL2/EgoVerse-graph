import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "clusters"
    / "common"
    / "sync_runtime.sh"
)


def test_sync_runtime_is_valid_bash_and_rejects_relative_paths():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    result = subprocess.run(
        [str(SCRIPT), "relative/source", "/tmp/runtime"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "must be absolute paths" in result.stderr


def test_sync_runtime_uses_frozen_lock_and_verifies_torchdiffeq():
    text = SCRIPT.read_text()

    assert 'uv sync --frozen --project "$repo"' in text
    assert 'version("torchdiffeq")' in text
    assert 'item.startswith("torchdiffeq==")' in text
    assert "/coc/" not in text
    assert "/storage/" not in text
