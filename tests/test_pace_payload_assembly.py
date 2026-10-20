import hashlib
import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "scripts/clusters/pace"
spec = importlib.util.spec_from_file_location(
    "assemble_payload", ROOT / "assemble_payload.py"
)
assembler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assembler)


def binding(path):
    return dict(
        path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )


def component(tmp_path, name, prefix="workspace/runtime", link=None, hardlink=False):
    archive, manifest = tmp_path / (name + ".tar.gz"), tmp_path / (name + ".sha256")
    data = b"exact public bytes"
    digest = hashlib.sha256(data).hexdigest()
    with tarfile.open(archive, "w:gz") as stream:
        item = tarfile.TarInfo("lib/code.py")
        item.size = len(data)
        stream.addfile(item, io.BytesIO(data))
        if link is not None or hardlink:
            item = tarfile.TarInfo("lib/alias")
            item.type = tarfile.LNKTYPE if hardlink else tarfile.SYMTYPE
            item.linkname = "lib/code.py" if hardlink else link
            stream.addfile(item)
    manifest.write_text(
        digest + "  lib/code.py\n" + (digest + "  lib/alias\n" if hardlink else "")
    )
    return dict(
        archive=binding(archive),
        manifest=binding(manifest),
        prefix=prefix,
        regular_bytes=len(data),
    )


def request(*components):
    return dict(schema="workspace-payload-components-v1", components=list(components))


def test_component_prefixes_and_full_file_hashes(tmp_path):
    a = component(tmp_path, "runtime")
    b = component(tmp_path, "data", "workspace/data")
    result = assembler.assemble(request(a, b), tmp_path / "payload")
    assert result["status"] == "PASS" and result["regular_files"] == 2
    with tarfile.open(result["archive"]["path"], "r:gz") as stream:
        assert set(stream.getnames()) == {
            "workspace/runtime/lib/code.py",
            "workspace/data/lib/code.py",
        }


def test_duplicate_component_member_rejected(tmp_path):
    a, b = component(tmp_path, "first"), component(tmp_path, "second")
    with pytest.raises(ValueError, match="Duplicate"):
        assembler.assemble(request(a, b), tmp_path / "payload")


@pytest.mark.parametrize(
    "damage", ["archive_digest", "member_digest", "missing_member", "budget", "prefix"]
)
def test_changed_or_unbounded_component_rejected(tmp_path, damage):
    a = component(tmp_path, "source")
    if damage == "archive_digest":
        a["archive"]["sha256"] = "f" * 64
    elif damage in ["member_digest", "missing_member"]:
        manifest = Path(a["manifest"]["path"])
        manifest.write_text(
            ("f" * 64 + "  lib/code.py\n")
            if damage == "member_digest"
            else manifest.read_text() + "f" * 64 + "  absent.py\n"
        )
        a["manifest"] = binding(manifest)
    elif damage == "budget":
        a["regular_bytes"] = 1
    else:
        a["prefix"] = "../escape"
    with pytest.raises(ValueError):
        assembler.assemble(request(a), tmp_path / "payload")
    assert not (tmp_path / "payload/receipt.json").exists()


def test_literal_external_python_link_preserved(tmp_path):
    a = component(tmp_path, "runtime", link="/usr/bin/python")
    result = assembler.assemble(request(a), tmp_path / "payload")
    with tarfile.open(result["archive"]["path"], "r:gz") as stream:
        assert (
            stream.getmember("workspace/runtime/lib/alias").linkname
            == "/usr/bin/python"
        )


def test_link_outside_selected_closure_rejected(tmp_path):
    a = component(tmp_path, "runtime", link="/etc/passwd")
    with pytest.raises(ValueError, match="closure"):
        assembler.assemble(request(a), tmp_path / "payload")


def test_hardlink_manifest_and_target_are_preserved(tmp_path):
    a = component(tmp_path, "runtime", hardlink=True)
    result = assembler.assemble(request(a), tmp_path / "payload")
    assert result["regular_files"] == 2
    with tarfile.open(result["archive"]["path"], "r:gz") as stream:
        assert (
            stream.getmember("workspace/runtime/lib/alias").linkname
            == "workspace/runtime/lib/code.py"
        )


def test_existing_output_is_never_replaced(tmp_path):
    a = component(tmp_path, "runtime")
    output = tmp_path / "payload"
    assembler.assemble(request(a), output)
    with pytest.raises(FileExistsError):
        assembler.assemble(request(a), output)
