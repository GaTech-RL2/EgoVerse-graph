import hashlib
import json

import pytest

from scripts.world_model.verify_joint_data_cache import sha, verify_cache


@pytest.fixture
def cache_request(tmp_path):
    parent = tmp_path / "cache"
    root = parent / "123"
    root.mkdir(parents=True)
    records, episodes = [], []

    def artifact(name, rgb=False):
        path = root / name
        path.write_bytes(b"immutable")
        digest = sha(path)
        records.append(
            dict(path=name, bytes=path.stat().st_size, sha256=digest, rgb_hardlink=rgb)
        )
        return digest

    norm = artifact("normalization.json")
    for i in range(500):
        name = f"episode-{i:05d}"
        episodes.append(
            dict(
                split="train" if i < 450 else "val",
                path=name + ".h5",
                sha256=artifact(name + ".h5"),
                rgb_path=name + ".rgb.h5",
                rgb_sha256=artifact(name + ".rgb.h5", True),
                action_provenance=dict(
                    path=name + ".alignment.json",
                    sha256=artifact(name + ".alignment.json"),
                ),
            )
        )
    manifest = dict(
        normalization=dict(path="normalization.json", sha256=norm), episodes=episodes
    )
    (root / "manifest.json").write_text(json.dumps(manifest))
    digest = sha(root / "manifest.json")
    records.append(
        dict(
            path="manifest.json",
            bytes=(root / "manifest.json").stat().st_size,
            sha256=digest,
            rgb_hardlink=False,
        )
    )
    receipt = dict(
        schema="joint-complete-data-cache-v1",
        status="PASS",
        job_id="123",
        node="gpu21",
        manifest_sha256=digest,
        logical_data_root="/logical/data",
        cache_root=str(root),
        files=1502,
        rgb_files=500,
        splits=dict(train=450, val=50),
        artifacts=records,
        total_bytes=sum(r["bytes"] for r in records),
    )
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    native = tmp_path / "native.txt"
    native.write_text(
        "JobId=123 JobName=joint-complete-data-v1 UserId=paphiwetsa3(42) "
        "JobState=COMPLETED ExitCode=0:0 NodeList=gpu21"
    )
    request = dict(
        data_manifest="/logical/data/manifest.json",
        data_manifest_sha256=digest,
        data_cache=dict(
            job_id=123,
            node="gpu21",
            root=str(root),
            receipt=str(receipt_path),
            receipt_sha256=sha(receipt_path),
            native_receipt=str(native),
            native_receipt_sha256=sha(native),
        ),
    )
    return request, parent, root, receipt


def test_complete_joint_cache_admission(cache_request):
    request, parent, _, _ = cache_request
    result = verify_cache(request, "gpu21", parent)
    assert result["status"] == "DATA_CACHE_ADMISSION_PASS"
    assert result["files"] == 1502 and result["rgb_files"] == 500


@pytest.mark.parametrize(
    "damage", ["missing", "partial", "symlink", "manifest", "receipt", "native", "node"]
)
def test_joint_cache_rejects_invalid_or_partial_evidence(cache_request, damage):
    request, parent, root, _ = cache_request
    binding = request["data_cache"]
    if damage == "missing":
        (root / "episode-00000.h5").unlink()
    elif damage == "partial":
        (root / "episode-00000.h5").write_bytes(b"x")
    elif damage == "symlink":
        (root / "episode-00000.h5").unlink()
        (root / "episode-00000.h5").symlink_to(root / "episode-00001.h5")
    elif damage == "manifest":
        (root / "manifest.json").write_text("{}")
    elif damage == "receipt":
        binding["receipt_sha256"] = hashlib.sha256(b"wrong").hexdigest()
    elif damage == "native":
        from pathlib import Path

        Path(binding["native_receipt"]).write_text("JobId=123 JobState=RUNNING")
        binding["native_receipt_sha256"] = sha(binding["native_receipt"])
    elif damage == "node":
        binding["node"] = "other-node"
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_cache(request, "gpu21", parent)


def test_joint_cache_rejects_forged_complete_inventory(cache_request):
    from pathlib import Path

    request, parent, _, receipt = cache_request
    receipt["artifacts"].pop()
    Path(request["data_cache"]["receipt"]).write_text(json.dumps(receipt))
    request["data_cache"]["receipt_sha256"] = sha(request["data_cache"]["receipt"])
    with pytest.raises(ValueError, match="inventory"):
        verify_cache(request, "gpu21", parent)
