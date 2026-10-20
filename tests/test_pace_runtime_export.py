import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "scripts/clusters/pace"
spec = importlib.util.spec_from_file_location(
    "runtime_export", ROOT / "export_runtime.py"
)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def test_bounded_runtime_inventory_preserves_links_and_omits_logs(tmp_path):
    root = tmp_path / "runtime"
    (root / "lib").mkdir(parents=True)
    (root / "lib/code.py").write_text("public code")
    (root / "lib64").symlink_to("lib")
    (root / "build.log").write_text("omitted")
    entries, links, omitted = runtime.inventory([(root, "/workspace/runtime")])
    assert set(entries) == {
        "/workspace/runtime/lib/code.py",
        "/workspace/runtime/lib64",
    }
    assert links == [dict(path="workspace/runtime/lib64", target="lib")]
    assert omitted == ["/workspace/runtime/build.log"]


@pytest.mark.parametrize("target", ["/etc/passwd", "../unselected", "/workspace/other"])
def test_link_cannot_escape_exact_closure(tmp_path, target):
    (tmp_path / "leaf").symlink_to(target)
    with pytest.raises(ValueError, match="closure"):
        runtime.inventory([(tmp_path, "/workspace/runtime")])


def test_checkpoint_and_credential_like_runtime_inputs_rejected(tmp_path):
    path = tmp_path / "model.ckpt"
    path.write_text("checkpoint")
    with pytest.raises(ValueError, match="checkpoint"):
        runtime.inventory([(tmp_path, "/workspace/runtime")])
    path.unlink()
    (tmp_path / "id_rsa_unexpected").write_text("not packaged")
    with pytest.raises(ValueError, match="Credential"):
        runtime.inventory([(tmp_path, "/workspace/runtime")])
