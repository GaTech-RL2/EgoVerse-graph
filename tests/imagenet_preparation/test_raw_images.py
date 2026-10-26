"""Synthetic fixtures only: no actual ImageNet, Qwen execution or native proof."""

import hashlib
import errno
import json
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "imagenet_native"))
from imagenet_fixture import make_fixture
from egomimic.rldb import imagenet_preparation as prep


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def fixture(tmp_path):
    return make_fixture(tmp_path.resolve(), train_per_class=2, val_per_class=1)


def verify(fixture, output, **changes):
    values = dict(expected_classes=2, expected_train_images=4, expected_val_images=2,
                  train_sidecar_exclusions=[], workers=2, max_inflight=3)
    values.update(changes)
    return prep.verify_images(fixture["data_root"] / "train", fixture["data_root"] / "val", output, **values)


def check(fixture, receipt):
    return prep.validate_raw_inventory(receipt["path"], receipt["sha256"],
                    expected_classes=2, expected_train_images=4, expected_val_images=2)


def test_complete_portable_inventory_and_no_qwen(fixture, monkeypatch):
    monkeypatch.setattr(prep.adapter, "load_class_language", lambda *a, **k: pytest.fail("raw producer read Qwen"))
    receipt = verify(fixture, fixture["root"] / "raw.json")
    report = check(fixture, receipt)
    assert receipt["status"] == prep.RAW_STATUS and report["data_admitted"] is False
    assert report["decoder"]["module_sha256"] == prep.ADAPTER_SHA256
    assert report["physical_sources"] == {s: str(fixture["data_root"] / s) for s in ("train", "val")}
    assert "physical_sources" not in report["portable_content"]
    for split, part in report["portable_content"]["splits"].items():
        assert part["count"] == (4 if split == "train" else 2)
        assert all(set(r) == {"path", "wnid", "bytes", "sha256", "rgb160_sha256"} for r in part["images"])
        assert [r["path"] for r in part["images"]] == sorted(r["path"] for r in part["images"])
    journal = [json.loads(x) for x in Path(receipt["path"] + ".progress.jsonl").read_text().splitlines()]
    assert [x["event"] for x in journal] == ["STARTED", "ALL_IMAGES_DECODED", "PUBLISHED"]
    assert Path(receipt["path"] + ".payload").stat().st_ino == Path(receipt["path"]).stat().st_ino


def test_explicit_known_sidecars_retained(fixture):
    rows = []
    for name, size in prep.SIDECARS.items():
        path = fixture["data_root"] / "train" / name
        path.write_bytes(b"x" * size)
        rows.append(dict(name=name, bytes=size, sha256=sha(path)))
    receipt = verify(fixture, fixture["root"] / "raw.json", train_sidecar_exclusions=rows)
    assert check(fixture, receipt)["excluded_sidecars"] == [dict(split="train", **r) for r in sorted(rows, key=lambda r: r["name"])]


@pytest.mark.parametrize("split,name", [("train", "unknown.txt"), ("val", "tar_extract_script.sh"),
                                       ("train", "tar_extract_script.sh")])
def test_unknown_or_undeclared_sidecar_rejected(fixture, split, name):
    (fixture["data_root"] / split / name).write_bytes(b"x")
    with pytest.raises(ValueError):
        verify(fixture, fixture["root"] / "raw.json")
    assert not (fixture["root"] / "raw.json").exists()
    assert json.loads((fixture["root"] / "raw.json.failure.json").read_text())["status"] == "FAILED_NO_DATA_ADMISSION"


def test_wrong_sidecar_hash_rejected(fixture):
    (fixture["data_root"] / "train" / "tar_extract_script.sh").write_bytes(b"x" * 108)
    with pytest.raises(ValueError, match="differs"):
        verify(fixture, fixture["root"] / "raw.json", train_sidecar_exclusions=[
            dict(name="tar_extract_script.sh", bytes=108, sha256="0" * 64)])


def test_missing_declared_sidecar_rejected(fixture):
    with pytest.raises(ValueError, match="absent"):
        verify(fixture, fixture["root"] / "raw.json", train_sidecar_exclusions=[
            dict(name="tar_extract_script.sh~", bytes=0, sha256=hashlib.sha256(b"").hexdigest())])


@pytest.mark.parametrize("what", ["root_alias", "image_symlink", "class_symlink"])
def test_aliases_and_source_symlinks_rejected(fixture, what):
    if what == "root_alias":
        alias = fixture["root"] / "alias"
        alias.symlink_to(fixture["data_root"], target_is_directory=True)
        with pytest.raises(ValueError, match="physical"):
            prep.verify_images(alias / "train", alias / "val", fixture["root"] / "raw.json",
                        expected_classes=2, expected_train_images=4, expected_val_images=2,
                        train_sidecar_exclusions=[])
    else:
        folder = fixture["data_root"] / "train" / "n00000001"
        if what == "image_symlink":
            (folder / "linked.png").symlink_to(next(folder.iterdir()))
        else:
            (fixture["data_root"] / "train" / "n00000003").symlink_to(folder, target_is_directory=True)
        with pytest.raises(ValueError):
            verify(fixture, fixture["root"] / "raw.json")


def test_corrupt_image_has_preserved_failure_no_pass(fixture):
    next((fixture["data_root"] / "train" / "n00000001").iterdir()).write_bytes(b"not an image")
    out = fixture["root"] / "raw.json"
    with pytest.raises(Exception):
        verify(fixture, out)
    assert not out.exists() and not Path(str(out) + ".payload").exists()
    assert Path(str(out) + ".intent.json").exists()
    assert Path(str(out) + ".failure.json").exists()
    before = Path(str(out) + ".failure.json").read_bytes()
    with pytest.raises(ValueError, match="Fresh immutable"):
        verify(fixture, out)
    assert Path(str(out) + ".failure.json").read_bytes() == before


@pytest.mark.parametrize("changes", [dict(expected_train_images=3), dict(expected_val_images=3),
                                    dict(expected_classes=3), dict(workers=5), dict(max_inflight=17),
                                    dict(workers=3, max_inflight=2)])
def test_exact_counts_and_concurrency_fail_closed(fixture, changes):
    with pytest.raises(ValueError):
        verify(fixture, fixture["root"] / "raw.json", **changes)


def test_changed_source_directory_rejected(fixture, monkeypatch):
    original = prep._decode_task
    changed = False
    def decode(task):
        nonlocal changed
        row = original(task)
        if not changed:
            changed = True
            directory = fixture["data_root"] / "train" / "n00000001"
            (directory / "new.txt").write_text("changed while decoding")
        return row
    monkeypatch.setattr(prep, "_decode_task", decode)
    with pytest.raises(ValueError, match="directory changed"):
        verify(fixture, fixture["root"] / "raw.json")


def test_decoder_module_pin_fails_before_image_decode(fixture, monkeypatch):
    monkeypatch.setattr(prep, "ADAPTER_SHA256", "0" * 64)
    monkeypatch.setattr(prep.adapter, "_decode_rgb160", lambda x: pytest.fail("decode after pin mismatch"))
    with pytest.raises(ValueError, match="sealed"):
        verify(fixture, fixture["root"] / "raw.json")


def test_bounded_submission_not_executor_map(monkeypatch):
    counts = dict(outstanding=0, peak=0)
    class Future:
        def __init__(self, task): self.task = task
        def result(self):
            counts["outstanding"] -= 1
            return self.task
        def cancel(self): pass
    class Pool:
        def __init__(self, **kwargs): assert kwargs["max_workers"] == 4
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def submit(self, fn, task):
            counts["outstanding"] += 1
            counts["peak"] = max(counts["peak"], counts["outstanding"])
            return Future(task)
    monkeypatch.setattr(prep, "ThreadPoolExecutor", Pool)
    assert list(prep._ordered_bounded_map(iter(range(1000)), workers=4, max_inflight=16)) == list(range(1000))
    assert counts == dict(outstanding=0, peak=16)


def test_join_preserves_exact_sealed_schema_without_image_io(fixture, monkeypatch):
    receipt = verify(fixture, fixture["root"] / "raw.json")
    monkeypatch.setattr(prep, "_read_regular", lambda x: pytest.fail("join read a source image"))
    monkeypatch.setattr(prep.adapter, "_decode_rgb160", lambda x: pytest.fail("join decoded a source image"))
    joined = prep.join_raw_inventory(receipt["path"], receipt["sha256"],
                fixture["class_language_path"], fixture["class_language_sha256"],
                fixture["language_features_path"], fixture["language_features_sha256"],
                fixture["root"] / "joined.json", expected_classes=2, expected_train_images=4, expected_val_images=2)
    assert json.loads(Path(joined["manifest"]["path"]).read_text()) == json.loads(fixture["manifest_path"].read_text())
    assert joined["raw_report"]["sha256"] == receipt["sha256"]
    assert joined["images_rehashed_or_decoded"] == 0 and joined["data_admitted"] is False


def test_raw_report_hash_and_failed_namespace_rejected(fixture):
    receipt = verify(fixture, fixture["root"] / "raw.json")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        prep.validate_raw_inventory(receipt["path"], "0" * 64)
    Path(receipt["path"] + ".failure.json").write_text("failure")
    with pytest.raises(ValueError, match="failure receipt"):
        check(fixture, receipt)


def test_empty_image_and_unknown_class_member_rejected(fixture):
    next((fixture["data_root"] / "val" / "n00000001").iterdir()).write_bytes(b"")
    with pytest.raises(ValueError, match="Empty source"):
        verify(fixture, fixture["root"] / "raw.json")


def test_train_val_class_mismatch_rejected(fixture):
    (fixture["data_root"] / "val" / "n00000002").rename(fixture["data_root"] / "val" / "n00000003")
    with pytest.raises(ValueError, match="WNID sets differ"):
        verify(fixture, fixture["root"] / "raw.json")


@pytest.mark.parametrize("field,value", [
    ("physical_sources", {"train": "relative/train", "val": "relative/val"}),
    ("physical_sources", {"train": "/a/train", "val": "/b/val"}),
    ("physical_sources", {"train": "/a/../b/train", "val": "/a/../b/val"}),
    ("excluded_sidecars", [dict(split="val", name="tar_extract_script.sh~", bytes=0, sha256="0" * 64)]),
    ("excluded_sidecars", [dict(split="train", name="unknown", bytes=0, sha256="0" * 64)]),
    ("concurrency", dict(workers=4, max_inflight=17)),
    ("concurrency", dict(workers=True, max_inflight=16)),
    ("producer", dict(job_id="412999", pid=12, native_completion="COMPLETED", host_completion="EXTERNAL_REQUIRED")),
    ("producer", dict(job_id="UNBOUND", pid=12, native_completion="EXTERNAL_REQUIRED", host_completion="EXTERNAL_REQUIRED")),
    ("started_at_utc", "2030-01-01T00:00:00+00:00"),
    ("completed_at_utc", "2026-01-01T00:00:00"),
    ("status", "PASS"),
])
def test_retained_provenance_strictly_revalidated(fixture, field, value):
    receipt = verify(fixture, fixture["root"] / "raw.json")
    report = json.loads(Path(receipt["path"]).read_text())
    report[field] = value
    other = fixture["root"] / "tampered.json"
    other.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        prep.validate_raw_inventory(other, sha(other), expected_classes=2, expected_train_images=4, expected_val_images=2)


def unreadable_sidecar(fixture):
    path = fixture["data_root"] / "train" / "tar_extract_script.sh"
    path.write_bytes(b"x" * 108)
    item = path.stat()
    return path, dict(name=path.name, bytes=108, status="UNREADABLE_EXCLUDED", uid=item.st_uid,
                     mode=format(stat.S_IMODE(item.st_mode), "04o"), mtime_seconds=item.st_mtime_ns // 1000000000)


def test_known108b_actual_permission_error_retained_without_claiming_hash(fixture, monkeypatch):
    path, recipe = unreadable_sidecar(fixture)
    original = prep._read_regular
    def read(value):
        if value == path: raise PermissionError(errno.EACCES, "synthetic permission denied", str(path))
        return original(value)
    monkeypatch.setattr(prep, "_read_regular", read)
    receipt = verify(fixture, fixture["root"] / "raw.json", train_sidecar_exclusions=[recipe])
    report = check(fixture, receipt)
    row = report["excluded_sidecars"][0]
    assert "sha256" not in row and row["status"] == "UNREADABLE_EXCLUDED"
    assert row["read_attempt"] == dict(operation="open_read_O_NOFOLLOW", outcome="PermissionError", errno=13)


def test_unreadable_recipe_rejects_readable_file(fixture):
    _, recipe = unreadable_sidecar(fixture)
    with pytest.raises(ValueError, match="readable file"):
        verify(fixture, fixture["root"] / "raw.json", train_sidecar_exclusions=[recipe])


@pytest.mark.parametrize("change", [dict(uid=999999), dict(mode="0000"), dict(mtime_seconds=0),
                                   dict(name="tar_extract_script.sh~", bytes=0)])
def test_unreadable_recipe_rejects_wrong_metadata_or_other_sidecar(fixture, monkeypatch, change):
    path, recipe = unreadable_sidecar(fixture)
    recipe.update(change)
    original = prep._read_regular
    def read(value):
        if value == path: raise PermissionError(errno.EACCES, "synthetic permission denied", str(path))
        return original(value)
    monkeypatch.setattr(prep, "_read_regular", read)
    with pytest.raises(ValueError):
        verify(fixture, fixture["root"] / "raw.json", train_sidecar_exclusions=[recipe])


def test_unreadable_recipe_does_not_hide_different_io_failure(fixture, monkeypatch):
    path, recipe = unreadable_sidecar(fixture)
    original = prep._read_regular
    def read(value):
        if value == path: raise PermissionError(errno.EPERM, "synthetic different denial", str(path))
        return original(value)
    monkeypatch.setattr(prep, "_read_regular", read)
    with pytest.raises(PermissionError) as error:
        verify(fixture, fixture["root"] / "raw.json", train_sidecar_exclusions=[recipe])
    assert error.value.errno == errno.EPERM
