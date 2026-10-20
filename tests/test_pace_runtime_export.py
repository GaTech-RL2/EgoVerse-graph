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


def test_selected_public_namespaces_are_library_code_not_run_logs(tmp_path):
    for name in [
        "wandb/__init__.py",
        "logs/message_pb2.py",
        "credentials/__init__.pyi",
    ]:
        path = tmp_path / name
        path.parent.mkdir(parents=True)
        path.write_text("public library asset")
    entries, _, omissions = runtime.inventory([(tmp_path, "/workspace/library")], True)
    assert len(entries) == 3 and not omissions


def test_public_package_exception_preserves_secret_guard(tmp_path):
    (tmp_path / "id_rsa_unexpected").write_text("never export")
    with pytest.raises(ValueError, match="Credential"):
        runtime.inventory([(tmp_path, "/workspace/library")], True)


def test_public_ca_bundle_is_validated_and_private_keys_rejected(tmp_path):
    root = tmp_path / "botocore"
    root.mkdir()
    path = root / "cacert.pem"
    path.write_text(
        "-----BEGIN CERTIFICATE-----\npublic ca\n-----END CERTIFICATE-----\n"
    )
    assert not runtime.omitted(path, True)
    path.write_text("-----BEGIN CERTIFICATE-----\n-----BEGIN PRIVATE KEY-----\n")
    with pytest.raises(ValueError, match="public certificates"):
        runtime.omitted(path, True)


def test_selected_file_and_directory_remain_inside_bound_mounts(tmp_path):
    root = tmp_path / "runtime"
    (root / "bin").mkdir(parents=True)
    (root / "bin/wandb").write_text("public entrypoint")
    selected = runtime.select_mappings(
        [(str(root), "/workspace/runtime")], ["/workspace/runtime/bin/wandb"]
    )
    entries, _, _ = runtime.inventory(selected, True)
    assert list(entries) == ["/workspace/runtime/bin/wandb"]
    with pytest.raises(ValueError, match="outside"):
        runtime.select_mappings(
            [(str(root), "/workspace/runtime")], ["/workspace/other"]
        )


def test_selected_single_file_keeps_checkpoint_guard(tmp_path):
    path = tmp_path / "model.ckpt"
    path.write_text("never exported")
    with pytest.raises(ValueError, match="checkpoint"):
        runtime.inventory([(path, "/workspace/runtime/model.ckpt")], True)


def test_selected_input_cannot_traverse_a_link(tmp_path):
    root, other = tmp_path / "runtime", tmp_path / "other"
    root.mkdir()
    other.mkdir()
    (other / "code.py").write_text("outside closure")
    (root / "alias").symlink_to(other)
    with pytest.raises(ValueError, match="symlink"):
        runtime.select_mappings(
            [(str(root), "/workspace/runtime")], ["/workspace/runtime/alias/code.py"]
        )
