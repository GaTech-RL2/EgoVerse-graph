import importlib.util
from pathlib import Path

import h5py
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "egomimic/scripts/data_upload/yam_presigned_uploader.py"
)
SPEC = importlib.util.spec_from_file_location("yam_presigned_uploader", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
uploader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(uploader)


def _item(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _write_demo(path: Path, *, complete: bool) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["complete"] = complete


def test_source_state_accepts_a_stable_completed_demo(tmp_path: Path) -> None:
    path = tmp_path / "demo.hdf5"
    _write_demo(path, complete=True)

    resolved, state = uploader.source_state(tmp_path, _item(path))

    assert resolved == path
    assert state["size"] == path.stat().st_size


def test_source_state_rejects_an_incomplete_demo(tmp_path: Path) -> None:
    path = tmp_path / "demo.hdf5"
    _write_demo(path, complete=False)

    with pytest.raises(RuntimeError, match="did not mark the demo complete"):
        uploader.source_state(tmp_path, _item(path))
