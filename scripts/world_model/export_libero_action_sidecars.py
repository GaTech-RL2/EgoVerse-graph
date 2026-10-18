#!/usr/bin/env python3
"""Publish immutable LIBERO action sidecars after exact source-row/time checks.

CPU-only. Run where the prepared caches and their native alignment artifacts
exist. The raw-export bundle records the separate source/controller audit; this
tool verifies that evidence and binds its commands to actual cached frames.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from egomimic.rldb.world_model_preparation import ArtifactVerificationMemo


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _inside(root, name):
    candidate = (root / name).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError("Raw-export artifact paths must stay inside their bundle")
    return candidate


def _source_hashes(path):
    result = {}
    for line in path.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip(" *")
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or name in result:
            raise ValueError("Invalid or duplicate raw source hash evidence")
        result[name] = digest
    return result


def _verified_bundle(raw_dir, manifest_path, verifier):
    metadata_path = raw_dir / "metadata.json"
    metadata = _json(metadata_path)
    if (
        metadata.get("schema") != "mot-libero-raw-command-export-v1"
        or metadata.get("status") != "PASS"
    ):
        raise ValueError("Require completed raw command export evidence")
    if metadata.get("source_manifest_sha256") != sha256(manifest_path):
        raise ValueError("Raw export is bound to a different prepared manifest")
    evidence = {"metadata.json": metadata_path}
    for key in ("raw_source_hashes", "headers", "controller_metadata", "validation"):
        artifact = metadata[key]
        path = _inside(raw_dir, artifact["path"])
        verifier.verify(path, artifact["sha256"])
        evidence[artifact["path"]] = path
    script = raw_dir / "export-commands.sh"
    verifier.verify(script, metadata["export_script_sha256"])
    evidence[script.name] = script
    producer = metadata["producer_convention"]
    expected_url = (
        f"https://github.com/{producer.get('repository')}/blob/"
        f"{producer.get('commit')}/{producer.get('path')}"
    )
    if (
        producer.get("repository") != "Lifelong-Robot-Learning/LIBERO"
        or not re.fullmatch(r"[0-9a-f]{40}", producer.get("commit", ""))
        or producer.get("path") != "scripts/create_dataset.py"
        or producer.get("relation") != "action_t_applied_to_state_t"
        or producer.get("url") != expected_url
    ):
        raise ValueError(
            "Missing pinned official LIBERO state/action convention evidence"
        )
    source = _inside(raw_dir, producer["local_path"])
    verifier.verify(source, producer["sha256"])
    evidence[producer["local_path"]] = source
    lines = producer.get("evidence_lines", [])
    if not lines or any(
        type(i) is not int or not 1 <= i <= len(source.read_text().splitlines())
        for i in lines
    ):
        raise ValueError("Producer convention evidence lines are invalid")
    schema = metadata["action_schema"]
    if (
        schema.get("dimension") != 7
        or schema.get("fields")
        != [
            "delta_x",
            "delta_y",
            "delta_z",
            "delta_rx",
            "delta_ry",
            "delta_rz",
            "gripper_command",
        ]
        or schema.get("semantics") != "action_command"
        or schema.get("value_space") != "native_commands"
        or schema.get("controller_type") != "OSC_POSE"
        or schema.get("control_delta") is not True
    ):
        raise ValueError(
            "Require explicit seven-coordinate native delta OSC_POSE command schema"
        )
    hashes = _source_hashes(evidence[metadata["raw_source_hashes"]["path"]])
    controllers = _json(evidence[metadata["controller_metadata"]["path"]])
    by_source = {item["source_demo"]: item for item in controllers}
    if len(by_source) != len(controllers) or set(by_source) != set(hashes):
        raise ValueError(
            "Controller evidence must cover each hashed raw demonstration once"
        )
    for item in controllers:
        config = item["controller_configs"]
        if (
            config.get("type") != schema["controller_type"]
            or config.get("control_delta") is not True
            or item.get("control_freq") != schema["control_frequency_hz"]
            or not np.array_equal(
                np.broadcast_to(config["input_min"], (7,)), schema["input_min"]
            )
            or not np.array_equal(
                np.broadcast_to(config["input_max"], (7,)), schema["input_max"]
            )
            or config["output_min"] != schema["pose_output_min"]
            or config["output_max"] != schema["pose_output_max"]
        ):
            raise ValueError("Raw controller evidence disagrees with command schema")
    payload = _inside(raw_dir, metadata["payload_path"])
    verifier.verify(payload, metadata["payload_sha256"])
    if (
        payload.stat().st_size != metadata["payload_nbytes"]
        or metadata["payload_nbytes"] != metadata["expected_payload_nbytes"]
    ):
        raise ValueError("Raw binary payload size mismatch")
    evidence[metadata["payload_path"]] = payload
    return metadata, evidence, hashes, payload


def _frame_binding(episode, root, verifier):
    feature = (root / episode["path"]).resolve()
    verifier.verify(
        feature,
        episode["sha256"],
        trusted_fingerprint=episode.get("verified_fingerprint"),
    )
    receipt_path = feature.with_suffix(feature.suffix + ".json")
    verifier.verify(receipt_path, episode["receipt_sha256"])
    receipt = _json(receipt_path)
    if (
        receipt.get("schema") != "mot-world-feature-cache-v1"
        or receipt.get("status") != "PASS"
        or receipt.get("output_sha256") != episode["sha256"]
        or receipt.get("native_source") != episode["native_source"]
        or receipt.get("episode_id") != episode["episode_id"]
        or receipt.get("embodiment") != episode["embodiment"]
    ):
        raise ValueError("Feature receipt disagrees with prepared episode")
    inputs = receipt["request"]["inputs"]
    native = Path(receipt["native_cache"])
    verifier.verify(native / "manifest.json", inputs["native_manifest_sha256"])
    native_manifest = _json(native / "manifest.json")
    if (
        native_manifest.get("schema") != "libero-same-recorded-state-v1"
        or any(native_manifest.get(k) != v for k, v in episode["native_source"].items())
        or native_manifest.get("artifacts", {}).get("aligned.npz")
        != inputs["native_aligned_sha256"]
    ):
        raise ValueError("Native alignment manifest identity mismatch")
    aligned = native / "aligned.npz"
    verifier.verify(aligned, inputs["native_aligned_sha256"])
    with np.load(aligned, allow_pickle=False) as data:
        rows, native_time = data["source_row"], data["time"]
    with h5py.File(feature, "r") as data:
        feature_time = data["time"][:]
    n = episode["frames"]
    if (
        rows.shape != (n,)
        or rows.dtype.kind not in "iu"
        or not np.array_equal(rows, np.arange(n))
        or native_time.shape != (n,)
        or feature_time.shape != (n,)
        or not np.array_equal(native_time, feature_time)
        or not np.isfinite(feature_time).all()
        or not (np.diff(feature_time) > 0).all()
    ):
        raise ValueError(
            "Native source_row or feature/native timestamp alignment mismatch"
        )
    return (
        rows.astype(np.int64),
        feature_time.astype(np.float64),
        {
            "feature_receipt_sha256": episode["receipt_sha256"],
            "native_manifest_sha256": inputs["native_manifest_sha256"],
            "native_aligned_sha256": inputs["native_aligned_sha256"],
            "native_cache": str(native),
        },
    )


def export_sidecars(manifest_path, raw_export_dir, output_dir):
    manifest_path, raw_dir, output = map(
        Path, (manifest_path, raw_export_dir, output_dir)
    )
    if output.exists():
        raise FileExistsError(f"Refusing to replace immutable sidecar output: {output}")
    verifier = ArtifactVerificationMemo()
    manifest = _json(manifest_path)
    metadata, evidence, raw_hashes, payload = _verified_bundle(
        raw_dir, manifest_path, verifier
    )
    raw_episodes = {(e["embodiment"], e["episode_id"]): e for e in metadata["episodes"]}
    expected = {(e["embodiment"], e["episode_id"]) for e in manifest["episodes"]}
    if len(raw_episodes) != len(metadata["episodes"]) or set(raw_episodes) != expected:
        raise ValueError("Raw export must match every prepared episode exactly once")
    # Validate the complete small binary and all cache bindings before publishing.
    values, ranges = [], []
    schema = metadata["action_schema"]
    for episode in manifest["episodes"]:
        raw = raw_episodes[(episode["embodiment"], episode["episode_id"])]
        n = episode["frames"]
        if (
            episode["embodiment"] != "libero_gripper"
            or raw["frames"] != n
            or raw["split"] != episode["split"]
            or raw["feature_sha256"] != episode["sha256"]
            or raw["native_source"] != episode["native_source"]
            or raw_hashes.get(raw["native_source"]["source_demo"])
            != raw["native_source"]["source_demo_sha256"]
        ):
            raise ValueError(
                "Raw export episode or original demonstration SHA mismatch"
            )
        arrays = []
        for name, shape in (("action", [n, 7]), ("state_time", [n])):
            offset, size = raw[f"{name}_offset"], raw[f"{name}_nbytes"]
            if (
                raw[f"{name}_dtype"] != "<f8"
                or raw[f"{name}_shape"] != shape
                or type(offset) is not int
                or offset < 0
                or offset % 8
                or type(size) is not int
                or size != int(np.prod(shape)) * 8
                or offset + size > metadata["payload_nbytes"]
            ):
                raise ValueError("Invalid raw command binary dtype/shape/range")
            ranges.append((offset, offset + size))
            arrays.append(
                np.fromfile(
                    payload, dtype="<f8", count=int(np.prod(shape)), offset=offset
                ).reshape(shape)
            )
        actions, times = arrays
        if (
            not np.isfinite(actions).all()
            or not np.isfinite(times).all()
            or (actions < np.asarray(schema["input_min"])).any()
            or (actions > np.asarray(schema["input_max"])).any()
        ):
            raise ValueError("Non-finite or out-of-controller-bound commands")
        rows, cached_time, binding = _frame_binding(
            episode, manifest_path.parent, verifier
        )
        if not np.array_equal(times, cached_time):
            raise ValueError(
                "Raw states[:,0] timestamps differ from exact cached feature frames"
            )
        values.append((episode, actions.astype(np.float32), cached_time, rows, binding))
    ordered = sorted(ranges)
    if (
        not ordered
        or ordered[0][0] != 0
        or ordered[-1][1] != metadata["payload_nbytes"]
        or any(left[1] != right[0] for left, right in zip(ordered, ordered[1:]))
    ):
        raise ValueError(
            "Raw binary ranges must cover the payload without gaps or overlaps"
        )
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    for name, source in evidence.items():
        target = provenance / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst)
    producer_sha = sha256(__file__)
    metadata_sha = sha256(raw_dir / "metadata.json")
    output_episodes = []
    for number, (episode, actions, times, rows, binding) in enumerate(values):
        name = f"episode-{number:05d}"
        sidecar = output / f"{name}.h5"
        with h5py.File(sidecar, "x") as data:
            for key, value in dict(
                action=actions,
                time=times,
                valid=np.ones(len(rows), bool),
                cache_frame_index=np.arange(len(rows)),
                source_state_index=rows,
                source_action_index=rows,
            ).items():
                data.create_dataset(key, data=value)
        native = episode["native_source"]
        identity = dict(
            feature_sha256=episode["sha256"],
            source_demo_sha256=native["source_demo_sha256"],
            source_episode=native["episode"],
        )
        receipt_path = output / f"{name}.alignment.json"
        receipt = dict(
            schema="dino-proprio-source-alignment-v1",
            **identity,
            relation="action_t_applied_to_state_t",
            producer_sha256=producer_sha,
            basis={
                "cache_index_source": f"SHA-verified native aligned.npz source_row; {binding}",
                "action_index_source": f"Raw {native['source_demo']} data/{native['episode']}/actions, SHA-verified export metadata {metadata_sha}",
                "state_action_relation_source": json.dumps(
                    metadata["producer_convention"], sort_keys=True
                ),
            },
            cache_frame_index=list(range(len(rows))),
            source_state_index=rows.tolist(),
            source_action_index=rows.tolist(),
            raw_export_metadata_sha256=metadata_sha,
            raw_payload_sha256=metadata["payload_sha256"],
            feature_source_binding=binding,
            producer_convention=metadata["producer_convention"],
        )
        _write_json(receipt_path, receipt)
        output_episodes.append(
            dict(
                embodiment=episode["embodiment"],
                episode_id=episode["episode_id"],
                path=sidecar.name,
                sha256=sha256(sidecar),
                **identity,
                alignment={
                    "relation": "action_t_applied_to_state_t",
                    "cache_frame_mapping": "explicit_source_indices",
                    "producer_sha256": producer_sha,
                    "source_alignment_path": receipt_path.name,
                    "source_alignment_sha256": sha256(receipt_path),
                },
            )
        )
    result = dict(
        schema="dino-proprio-actions-v1",
        action_schemas={"libero_gripper": schema},
        episodes=output_episodes,
        source_manifest_sha256=sha256(manifest_path),
        raw_export_metadata_sha256=metadata_sha,
        raw_payload_sha256=metadata["payload_sha256"],
        exporter_sha256=producer_sha,
        producer_convention=metadata["producer_convention"],
        normalization="none; native controller commands preserved as float32",
    )
    _write_json(output / "manifest.json", result)
    return {
        "status": "PASS",
        "manifest": str(output / "manifest.json"),
        "manifest_sha256": sha256(output / "manifest.json"),
        "episodes": len(values),
        "frames": sum(len(v[2]) for v in values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--raw-export-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            export_sidecars(args.manifest, args.raw_export_dir, args.output_dir)
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
