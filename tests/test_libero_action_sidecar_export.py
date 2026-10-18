"""Exporter integrity, source-row alignment, and loader interoperability."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from egomimic.rldb.dino_proprio import PreparedDinoProprioDataset
from scripts.world_model.export_libero_action_sidecars import export_sidecars


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value))
    return path


def bundle(tmp_path, *, bad_row=False, time_offset=0):
    raw = tmp_path / "raw"
    raw.mkdir()
    source_name, source_sha = "/fixture/original.hdf5", "a" * 64
    episodes, exports, payload = [], [], bytearray()
    for i, split in enumerate(("train", "val")):
        times = np.arange(4, dtype=np.float64) * 0.05
        native = tmp_path / f"native-{i}"
        native.mkdir()
        rows = np.arange(4)
        if bad_row:
            rows += 1
        np.savez(native / "aligned.npz", source_row=rows, time=times)
        native_source = dict(
            source_demo=source_name, source_demo_sha256=source_sha, episode=f"demo_{i}"
        )
        native_manifest = write_json(
            native / "manifest.json",
            dict(
                schema="libero-same-recorded-state-v1",
                **native_source,
                artifacts={"aligned.npz": digest(native / "aligned.npz")},
            ),
        )
        feature = tmp_path / f"feature-{i}.h5"
        with h5py.File(feature, "w") as data:
            data["time"] = times
            data["image"] = np.zeros((4, 2, 1, 8), np.float32)
            data["image_xy"] = np.zeros((1, 2), np.float32)
            data["pose"] = np.zeros((4, 9), np.float32)
            data["pose_valid"] = np.ones(4, bool)
        receipt = write_json(
            feature.with_suffix(".h5.json"),
            dict(
                schema="mot-world-feature-cache-v1",
                status="PASS",
                output_sha256=digest(feature),
                native_source=native_source,
                episode_id=f"episode_{i}",
                embodiment="libero_gripper",
                native_cache=str(native),
                request={
                    "inputs": {
                        "native_manifest_sha256": digest(native_manifest),
                        "native_aligned_sha256": digest(native / "aligned.npz"),
                    }
                },
            ),
        )
        ep = dict(
            embodiment="libero_gripper",
            episode_id=f"episode_{i}",
            split=split,
            frames=4,
            path=feature.name,
            sha256=digest(feature),
            receipt_sha256=digest(receipt),
            native_source=native_source,
            frame="fixed_reference",
            pose_kind="measured_pose",
        )
        episodes.append(ep)
        actions = np.arange(28, dtype="<f8").reshape(4, 7) / 30
        action_offset = len(payload)
        payload.extend(actions.tobytes())
        time_start = len(payload)
        payload.extend(np.asarray(times + time_offset, dtype="<f8").tobytes())
        exports.append(
            dict(
                embodiment=ep["embodiment"],
                episode_id=ep["episode_id"],
                split=split,
                frames=4,
                feature_sha256=ep["sha256"],
                native_source=native_source,
                action_dtype="<f8",
                action_shape=[4, 7],
                action_offset=action_offset,
                action_nbytes=actions.nbytes,
                state_time_dtype="<f8",
                state_time_shape=[4],
                state_time_offset=time_start,
                state_time_nbytes=times.nbytes,
            )
        )
    manifest = write_json(
        tmp_path / "prepared.json",
        dict(
            schema="dino-proprio-prepared-v1",
            episodes=episodes,
            dino_identity={
                "model": "fixture",
                "revision": "fixture",
                "image_size": [16, 16],
                "patch_grid": [1, 1],
                "feature_layer": "last",
            },
            pose_schemas={
                "libero_gripper": {
                    "dimension": 9,
                    "fields": [f"p{i}" for i in range(9)],
                }
            },
            normalization={
                "scope": "train_only",
                "sha256": "0" * 64,
                "arrays_normalized": True,
            },
        ),
    )
    (raw / "payload.f64le").write_bytes(payload)
    (raw / "export-commands.sh").write_text("# synthetic fixture export\n")
    (raw / "create_dataset.py").write_text(
        "# Synthetic source fixture: apply actions[t] to states[t].\n"
    )
    (raw / "raw-hashes.txt").write_text(f"{source_sha}  {source_name}\n")
    (raw / "headers.txt").write_text("synthetic float64 actions[T,7], states[T,92]\n")
    write_json(raw / "validation.json", {"status": "PASS"})
    output_min, output_max = [-0.05] * 3 + [-0.5] * 3, [0.05] * 3 + [0.5] * 3
    write_json(
        raw / "controllers.json",
        [
            {
                "source_demo": source_name,
                "control_freq": 20,
                "controller_configs": {
                    "type": "OSC_POSE",
                    "control_delta": True,
                    "input_min": -1,
                    "input_max": 1,
                    "output_min": output_min,
                    "output_max": output_max,
                },
            }
        ],
    )
    schema = dict(
        dimension=7,
        fields=[
            "delta_x",
            "delta_y",
            "delta_z",
            "delta_rx",
            "delta_ry",
            "delta_rz",
            "gripper_command",
        ],
        semantics="action_command",
        value_space="native_commands",
        controller_type="OSC_POSE",
        control_delta=True,
        control_frequency_hz=20,
        input_min=[-1.0] * 7,
        input_max=[1.0] * 7,
        pose_output_min=output_min,
        pose_output_max=output_max,
    )
    metadata = dict(
        schema="mot-libero-raw-command-export-v1",
        status="PASS",
        source_manifest_sha256=digest(manifest),
        episodes=exports,
        payload_path="payload.f64le",
        payload_sha256=digest(raw / "payload.f64le"),
        payload_nbytes=len(payload),
        expected_payload_nbytes=len(payload),
        export_script_sha256=digest(raw / "export-commands.sh"),
        action_schema=schema,
        producer_convention={
            "repository": "Lifelong-Robot-Learning/LIBERO",
            "commit": "b" * 40,
            "path": "scripts/create_dataset.py",
            "sha256": digest(raw / "create_dataset.py"),
            "local_path": "create_dataset.py",
            "relation": "action_t_applied_to_state_t",
            "evidence_lines": [1],
            "url": f"https://github.com/Lifelong-Robot-Learning/LIBERO/blob/{'b' * 40}/scripts/create_dataset.py",
        },
    )
    for key, filename in (
        ("raw_source_hashes", "raw-hashes.txt"),
        ("headers", "headers.txt"),
        ("controller_metadata", "controllers.json"),
        ("validation", "validation.json"),
    ):
        metadata[key] = {"path": filename, "sha256": digest(raw / filename)}
    write_json(raw / "metadata.json", metadata)
    return manifest, raw


def test_exported_sidecars_load_as_native_commands_and_are_immutable(tmp_path):
    manifest, raw = bundle(tmp_path)
    output = tmp_path / "output"
    result = export_sidecars(manifest, raw, output)
    assert (
        result["status"] == "PASS" and result["episodes"] == 2 and result["frames"] == 8
    )
    data = PreparedDinoProprioDataset(
        manifest,
        task="direct_action",
        split="val",
        action_manifest_path=output / "manifest.json",
    )
    sample = data[2]
    np.testing.assert_array_equal(
        sample["world"]["action"]["value"].numpy(),
        (np.arange(28).reshape(4, 7) / 30)[2:3].astype(np.float32),
    )
    assert not sample["world"]["action"]["observed"].any()
    exported = json.loads((output / "manifest.json").read_text())
    assert exported["raw_export_metadata_sha256"] == digest(raw / "metadata.json")
    receipt = json.loads(
        (
            output / exported["episodes"][0]["alignment"]["source_alignment_path"]
        ).read_text()
    )
    assert (
        receipt["source_state_index"]
        == receipt["source_action_index"]
        == list(range(4))
    )
    assert receipt["feature_source_binding"]["native_aligned_sha256"]
    before = digest(output / "manifest.json")
    with pytest.raises(FileExistsError, match="immutable"):
        export_sidecars(manifest, raw, output)
    assert digest(output / "manifest.json") == before


@pytest.mark.parametrize(
    "damage", ["payload_hash", "overlap", "shape", "source_sha", "producer_hash"]
)
def test_export_rejects_corruption_and_invalid_binary_contract_before_publication(
    tmp_path, damage
):
    manifest, raw = bundle(tmp_path)
    meta = json.loads((raw / "metadata.json").read_text())
    if damage == "payload_hash":
        path = raw / "payload.f64le"
        content = bytearray(path.read_bytes())
        content[0] ^= 1
        path.write_bytes(content)
    elif damage == "overlap":
        meta["episodes"][1]["action_offset"] = 0
    elif damage == "shape":
        meta["episodes"][0]["action_shape"] = [4, 6]
    elif damage == "source_sha":
        meta["episodes"][0]["native_source"]["source_demo_sha256"] = "c" * 64
    else:
        (raw / "create_dataset.py").write_text("changed producer source\n")
    write_json(raw / "metadata.json", meta)
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        export_sidecars(manifest, raw, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "kwargs,message",
    [({"bad_row": True}, "source_row"), ({"time_offset": 0.05}, "timestamps differ")],
)
def test_export_rejects_semantically_shifted_source_rows_or_times_even_with_valid_hashes(
    tmp_path, kwargs, message
):
    manifest, raw = bundle(tmp_path, **kwargs)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match=message):
        export_sidecars(manifest, raw, output)
    assert not output.exists()
