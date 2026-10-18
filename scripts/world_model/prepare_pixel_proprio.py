#!/usr/bin/env python3
"""Prepare immutable RGB pixels and same-state measured proprio, without an encoder.

CPU preparation preserves every recorded timestamp, camera and source identity.
Splits are preserved unless an explicit balanced derivation is requested.
Native RGB is reduced by exact integer area averaging with ties-to-even
uint8 rounding; no crop, frame resampling, encoder or track arrays are used.
Only training episodes fit pose statistics. Action commands are optional and
require a separately audited export. Run on an allocated cluster CPU node.
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

from egomimic.rldb.world_model_preparation import (
    ArtifactVerificationMemo,
    StreamingMoments,
    training_identities,
)
from scripts.world_model.export_libero_action_sidecars import (
    _json,
    _verified_bundle,
    _write_json,
    sha256,
)


def _key(episode):
    return episode["embodiment"], episode["episode_id"]


def _index(episodes):
    result = {_key(episode): episode for episode in episodes}
    if len(result) != len(episodes):
        raise ValueError("Duplicate embodiment/episode identity")
    return result


def _json_sha256(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def derive_split(episodes, validation_episodes_per_task=None, split_seed=42):
    """Derive balanced gripper splits without mutating source assignments.

    A task is the verified native demonstration filename without ``_demo``;
    its episode_id must independently equal that task plus native demo index.
    Existing held-out episodes always remain held out. Remaining choices use
    SHA256 ranks, independent of source ordering and NumPy/RNG versions.
    """
    if type(split_seed) is not int or split_seed < 0:
        raise ValueError("split_seed must be a nonnegative integer")
    original = _index(episodes)
    if not original or any(ep["split"] not in ("train", "val") for ep in episodes):
        raise ValueError("Require original explicit source splits")
    result = {key: dict(ep, original_split=ep["split"]) for key, ep in original.items()}
    policy = {"kind": "preserved_source_episode_assignments"}
    if validation_episodes_per_task is not None:
        if (
            type(validation_episodes_per_task) is not int
            or validation_episodes_per_task < 1
        ):
            raise ValueError(
                "validation_episodes_per_task must be a positive integer or None"
            )
        groups, task_sources = {}, {}
        for key, ep in original.items():
            native = ep["native_source"]
            source = Path(native["source_demo"])
            task = source.stem.removesuffix("_demo")
            source_episode = native["episode"]
            source_sha = native["source_demo_sha256"]
            if (
                key[0] != "libero_gripper"
                or source.suffix not in (".hdf5", ".h5")
                or not source.stem.endswith("_demo")
                or not task
                or re.fullmatch(r"demo_\d+", source_episode) is None
                or ep["episode_id"] != f"{task}-{source_episode}"
                or re.fullmatch(r"[0-9a-f]{64}", source_sha) is None
            ):
                raise ValueError(
                    "Balanced split requires matching native LIBERO task filename and episode identity"
                )
            if task in task_sources and task_sources[task] != source_sha:
                raise ValueError(
                    "One task filename refers to conflicting native source hashes"
                )
            task_sources[task] = source_sha
            groups.setdefault(task, []).append(key)
            result[key]["task_id"] = task
        task_counts = {}
        for task, keys in sorted(groups.items()):
            retained = {key for key in keys if original[key]["split"] == "val"}
            if len(
                retained
            ) > validation_episodes_per_task or validation_episodes_per_task >= len(
                keys
            ):
                raise ValueError(
                    "Requested balance must retain existing validation and nonempty training per task"
                )
            candidates = sorted(
                (key for key in keys if key not in retained),
                key=lambda key: (_json_sha256([split_seed, task, key[0], key[1]]), key),
            )
            validation = retained | set(
                candidates[: validation_episodes_per_task - len(retained)]
            )
            for key in keys:
                result[key]["split"] = "val" if key in validation else "train"
            task_counts[task] = dict(
                train=len(keys) - len(validation),
                val=len(validation),
                retained_original_val=len(retained),
                source_demo_sha256=task_sources[task],
            )
        policy = dict(
            kind="balanced_native_libero_tasks_v1",
            seed=split_seed,
            validation_episodes_per_task=validation_episodes_per_task,
            task_identity="native source filename without _demo, cross-checked against episode_id and native demo index",
            selection="retain_original_validation_then_lowest_sha256_of_compact_json_[seed,task,embodiment,episode_id]",
            task_counts=task_counts,
        )

    def identities(values):
        return sorted(
            [
                ep["embodiment"],
                ep["episode_id"],
                ep["native_source"]["source_demo_sha256"],
                ep["native_source"]["episode"],
                ep["frames"],
                ep["split"],
            ]
            for ep in values
        )

    original_ids, derived_ids = (
        identities(original.values()),
        identities(result.values()),
    )
    policy.update(
        original_identities=original_ids,
        derived_identities=derived_ids,
        original_identities_sha256=_json_sha256(original_ids),
        derived_identities_sha256=_json_sha256(derived_ids),
        counts={
            emb: {
                split: sum(
                    ep["embodiment"] == emb and ep["split"] == split
                    for ep in result.values()
                )
                for split in ("train", "val")
            }
            for emb in sorted({ep["embodiment"] for ep in episodes})
        },
    )
    return result, policy


def _raw_actions(raw_dir, source_manifest, verifier):
    """Validate the entire small command export, independent of feature caches."""
    metadata, evidence, hashes, payload = _verified_bundle(
        raw_dir, source_manifest, verifier
    )
    reference = _index(_json(source_manifest)["episodes"])
    raw = _index(metadata["episodes"])
    if set(raw) != set(reference):
        raise ValueError("Raw command export must exactly cover its source manifest")
    ranges, values = [], {}
    schema = metadata["action_schema"]
    for key, item in raw.items():
        episode, n = reference[key], item["frames"]
        if (
            key[0] != "libero_gripper"
            or type(n) is not int
            or n < 1
            or n != episode["frames"]
            or item["split"] != episode["split"]
            or item["native_source"] != episode["native_source"]
            or hashes.get(item["native_source"]["source_demo"])
            != item["native_source"]["source_demo_sha256"]
        ):
            raise ValueError("Raw source identity does not match source split manifest")
        arrays = []
        for name, shape in (("action", [n, 7]), ("state_time", [n])):
            offset, size = item[f"{name}_offset"], item[f"{name}_nbytes"]
            if (
                item[f"{name}_dtype"] != "<f8"
                or item[f"{name}_shape"] != shape
                or type(offset) is not int
                or offset < 0
                or offset % 8
                or type(size) is not int
                or size != int(np.prod(shape)) * 8
                or offset + size > metadata["payload_nbytes"]
            ):
                raise ValueError("Invalid raw command binary dtype, shape or range")
            ranges.append((offset, offset + size))
            arrays.append(
                np.fromfile(payload, "<f8", int(np.prod(shape)), offset=offset).reshape(
                    shape
                )
            )
        actions, times = arrays
        if (
            not np.isfinite(actions).all()
            or not np.isfinite(times).all()
            or not (np.diff(times) > 0).all()
            or (actions < np.asarray(schema["input_min"])).any()
            or (actions > np.asarray(schema["input_max"])).any()
        ):
            raise ValueError("Invalid raw commands or state times")
        values[key] = (actions.astype(np.float32), times, item)
    ranges.sort()
    if (
        not ranges
        or ranges[0][0] != 0
        or ranges[-1][1] != metadata["payload_nbytes"]
        or any(a[1] != b[0] for a, b in zip(ranges, ranges[1:]))
    ):
        raise ValueError("Raw payload ranges must be contiguous without overlap")
    return metadata, evidence, values


def resize_rgb_area(rgb, image_size):
    """Exact integer area mean in RGB, rounded to nearest uint8 with ties to even."""
    if rgb.ndim < 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Area resize requires uint8 RGB[...,H,W,3]")
    height, width = image_size
    source_h, source_w = rgb.shape[-3:-1]
    if (
        height < 1
        or width < 1
        or source_h % height
        or source_w % width
        or height > source_h
        or width > source_w
    ):
        raise ValueError(
            "Area resize requires integer downsampling factors with no upscaling"
        )
    fy, fx = source_h // height, source_w // width
    if fy == fx == 1:
        return rgb.copy()
    cells = rgb.reshape(*rgb.shape[:-3], height, fy, width, fx, 3)
    summed = cells.sum(axis=(-4, -2), dtype=np.uint64)
    return np.rint(summed / (fy * fx)).astype(np.uint8)


def _native_binding(episode, locations, pose_schema, image_size, verifier):
    native = Path(locations["native_cache"]).resolve(strict=True)
    rgb_path = Path(locations["rgb_cache"]).resolve(strict=True)
    receipt_path = Path(locations["rgb_receipt"]).resolve(strict=True)
    receipt_sha = sha256(receipt_path)
    receipt = _json(receipt_path)
    height, width = receipt.get("height"), receipt.get("width")
    if (
        receipt.get("schema") != "libero-native-rgb-v1"
        or receipt.get("status") != "PASS"
        or receipt.get("native_render") is not True
        or receipt.get("native_geometry_replay_exact") is not True
        or type(height) is not int
        or type(width) is not int
        or height < image_size[0]
        or width < image_size[1]
        or height % image_size[0]
        or width % image_size[1]
        or receipt.get("frames") != episode["frames"]
    ):
        raise ValueError(
            "Require successful native RGB receipt compatible with integer area downsampling"
        )
    native_manifest_path = native / "manifest.json"
    verifier.verify(native_manifest_path, receipt["source_manifest_sha256"])
    native_manifest = _json(native_manifest_path)
    source = episode["native_source"]
    cameras = receipt["cameras"]
    if (
        native_manifest.get("schema") != "libero-same-recorded-state-v1"
        or any(native_manifest.get(k) != v for k, v in source.items())
        or native_manifest.get("pose_fields") != pose_schema["fields"]
        or native_manifest.get("cameras") != cameras
        or not isinstance(cameras, list)
        or not cameras
        or len(set(cameras)) != len(cameras)
        or native_manifest.get("artifacts", {}).get("aligned.npz")
        != receipt["source_aligned_sha256"]
    ):
        raise ValueError("Native measured-pose/source/camera schema mismatch")
    aligned_path = native / "aligned.npz"
    aligned_fingerprint = verifier.verify(
        aligned_path,
        receipt["source_aligned_sha256"],
        trusted_fingerprint=locations.get("aligned_verified_fingerprint"),
    )
    # Load no tracks, original low-resolution RGB, or feature arrays.
    with np.load(aligned_path, allow_pickle=False) as aligned:
        pose = aligned["pose"].astype(np.float32)
        time = aligned["time"]
        rows = aligned["source_row"]
        state_time = aligned["state"][:, 0]
    n, d = episode["frames"], pose_schema["dimension"]
    if (
        pose.shape != (n, d)
        or not np.isfinite(pose).all()
        or time.shape != (n,)
        or not np.isfinite(time).all()
        or not (np.diff(time) > 0).all()
        or not np.array_equal(time, state_time)
        or rows.shape != (n,)
        or rows.dtype.kind not in "iu"
        or not np.array_equal(rows, np.arange(n))
    ):
        raise ValueError("Native pose/state-time/source-row alignment mismatch")
    rgb_fingerprint = verifier.verify(
        rgb_path,
        receipt["rgb_cache_sha256"],
        trusted_fingerprint=locations.get("rgb_verified_fingerprint"),
    )
    with h5py.File(rgb_path, "r") as rgb:
        if (
            rgb.attrs.get("schema") != "libero-native-rgb-v1"
            or not bool(rgb.attrs.get("native_render", False))
            or rgb["rgb"].dtype != np.dtype("uint8")
            or rgb["rgb"].shape != (n, len(cameras), height, width, 3)
            or not np.array_equal(rgb["time"][:], time)
        ):
            raise ValueError("Native RGB shape/type/time differs from measured proprio")
    # Receipts are small; also rule out mutation during the long data hash.
    verifier.verify(receipt_path, receipt_sha)
    binding = dict(
        source_rgb_path=str(rgb_path),
        source_rgb_sha256=receipt["rgb_cache_sha256"],
        source_image_size=[height, width],
        rgb_receipt_path=str(receipt_path),
        rgb_receipt_sha256=receipt_sha,
        native_cache=str(native),
        native_manifest_sha256=receipt["source_manifest_sha256"],
        native_aligned_sha256=receipt["source_aligned_sha256"],
    )
    if rgb_fingerprint:
        binding["source_rgb_verified_fingerprint"] = rgb_fingerprint
    if aligned_fingerprint:
        binding["native_aligned_verified_fingerprint"] = aligned_fingerprint
    return pose, time.astype(np.float64), rows.astype(np.int64), cameras, binding


def _write_resized_rgb(path, time, cameras, binding, image_size, batch_size, verifier):
    source_path = Path(binding["source_rgb_path"])
    verifier.verify(
        source_path,
        binding["source_rgb_sha256"],
        trusted_fingerprint=binding.get("source_rgb_verified_fingerprint"),
    )
    before = verifier.identity(source_path)
    with h5py.File(source_path, "r") as source, h5py.File(path, "x") as target:
        target.attrs["schema"] = "pixel-proprio-rgb-v1"
        target.attrs["resize"] = "integer_area_mean_round_ties_to_even"
        target.attrs["source_sha256"] = binding["source_rgb_sha256"]
        target.create_dataset("time", data=time)
        target.create_dataset(
            "cameras", data=np.asarray(cameras, dtype=h5py.string_dtype("utf-8"))
        )
        output = target.create_dataset(
            "rgb",
            shape=(len(time), len(cameras), *image_size, 3),
            dtype=np.uint8,
            chunks=(1, 1, *image_size, 3),
            compression="lzf",
        )
        for start in range(0, len(time), batch_size):
            stop = min(start + batch_size, len(time))
            output[start:stop] = resize_rgb_area(source["rgb"][start:stop], image_size)
    if verifier.identity(source_path) != before:
        raise ValueError("Native RGB source changed during pixel preparation")
    return sha256(path)


def prepare_pixels(
    source_manifest,
    native_spec,
    output_dir,
    *,
    image_size,
    patch_size,
    raw_export_dir=None,
    raw_source_manifest=None,
    std_floor=1e-5,
    resize_batch_size=4,
    embodiments=None,
    validation_episodes_per_task=None,
    split_seed=42,
):
    source_manifest, native_spec, output = map(
        lambda p: Path(p).resolve(), (source_manifest, native_spec, output_dir)
    )
    if output.exists():
        raise FileExistsError(f"Preserve immutable pixel preparation output: {output}")
    if (
        len(image_size) != 2
        or any(type(v) is not int or v <= 0 for v in image_size)
        or type(patch_size) is not int
        or patch_size <= 0
        or any(v % patch_size for v in image_size)
        or not np.isfinite(std_floor)
        or std_floor <= 0
        or type(resize_batch_size) is not int
        or resize_batch_size < 1
    ):
        raise ValueError(
            "Require patch-divisible image dimensions, positive batch size and std floor"
        )
    source_sha, locations_sha = sha256(source_manifest), sha256(native_spec)
    source, spec = _json(source_manifest), _json(native_spec)
    all_episodes, locations = _index(source["episodes"]), _index(spec["episodes"])
    if not all_episodes:
        raise ValueError("Require source episodes")
    available_embodiments = {key[0] for key in all_episodes}
    schemas = {
        emb: source["pose_schemas"][emb] for emb in sorted(available_embodiments)
    }
    source_keys = set()
    for ep in all_episodes.values():
        if (
            ep["split"] not in {"train", "val"}
            or type(ep["frames"]) is not int
            or ep["frames"] < 1
        ):
            raise ValueError("Invalid source episode frames/split")
        source_key = (
            ep["embodiment"],
            ep["native_source"]["source_demo_sha256"],
            ep["native_source"]["episode"],
        )
        if source_key in source_keys:
            raise ValueError("Duplicate native demonstration episode in source split")
        source_keys.add(source_key)
    for emb in available_embodiments:
        schema = schemas[emb]
        if (
            type(schema["dimension"]) is not int
            or schema["dimension"] < 1
            or len(schema["fields"]) != schema["dimension"]
            or len(set(schema["fields"])) != schema["dimension"]
            or {ep["split"] for ep in all_episodes.values() if ep["embodiment"] == emb}
            != {"train", "val"}
        ):
            raise ValueError(
                "Require complete pose schema and nonempty train/val split per embodiment"
            )
    selected = set(embodiments) if embodiments is not None else available_embodiments
    if (
        not selected
        or not selected.issubset(available_embodiments)
        or (embodiments is not None and len(selected) != len(embodiments))
    ):
        raise ValueError(
            "Select nonempty unique embodiments present in the frozen source manifest"
        )
    embodiments = selected
    episodes = {key: ep for key, ep in all_episodes.items() if key[0] in selected}
    schemas = {emb: schemas[emb] for emb in sorted(selected)}
    if not set(episodes).issubset(locations):
        raise ValueError("Native locations must cover each selected source episode")
    verifier = ArtifactVerificationMemo()
    raw_metadata, raw_evidence, raw_values = None, {}, {}
    raw_reference = (
        Path(raw_source_manifest).resolve() if raw_source_manifest else source_manifest
    )
    if raw_export_dir:
        raw_metadata, raw_evidence, raw_values = _raw_actions(
            Path(raw_export_dir).resolve(), raw_reference, verifier
        )
        if not set(raw_values).issubset(episodes):
            raise ValueError(
                "Action export contains episodes absent from selected pixel corpus"
            )
    elif raw_source_manifest:
        raise ValueError("raw_source_manifest requires raw_export_dir")
    # Validate policy metadata before long file reads; apply only after every
    # native/raw alignment below has been checked against its original split.
    derived_episodes, split_provenance = derive_split(
        list(episodes.values()), validation_episodes_per_task, split_seed
    )
    values, moments = [], {emb: StreamingMoments() for emb in embodiments}
    for key, ep in episodes.items():
        pose, time, rows, cameras, binding = _native_binding(
            ep, locations[key], schemas[key[0]], image_size, verifier
        )
        actions = None
        if key in raw_values:
            actions, raw_times, raw = raw_values[key]
            if (
                raw["native_source"] != ep["native_source"]
                or raw["split"] != ep["split"]
                or not np.array_equal(raw_times, time)
            ):
                raise ValueError(
                    "Raw action source/split/state time differs from native RGB/proprio"
                )
        values.append((ep, pose, time, rows, cameras, binding, actions))
    verifier.verify(source_manifest, source_sha)
    verifier.verify(native_spec, locations_sha)
    values = [(derived_episodes[_key(ep)], *rest) for ep, *rest in values]
    for ep, pose, *_ in values:
        if ep["split"] == "train":
            moments[ep["embodiment"]].update(pose)
    # Publish only after every input has passed the CPU validation above.
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    split_path = provenance / "split.json"
    _write_json(split_path, split_provenance)
    for name, path in (
        ("source-manifest.json", source_manifest),
        ("native-locations.json", native_spec),
    ):
        with path.open("rb") as src, (provenance / name).open("xb") as dst:
            shutil.copyfileobj(src, dst)
    for name, path in raw_evidence.items():
        target = provenance / "raw-commands" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with path.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst)
    producer_sha = sha256(__file__)
    raw_metadata_sha = (
        sha256(Path(raw_export_dir) / "metadata.json") if raw_export_dir else None
    )
    result_episodes = []
    for index, (ep, pose, time, rows, cameras, binding, actions) in enumerate(values):
        name = f"episode-{index:05d}"
        path = output / f"{name}.h5"
        rgb_path = output / f"{name}.rgb.h5"
        rgb_sha = _write_resized_rgb(
            rgb_path, time, cameras, binding, image_size, resize_batch_size, verifier
        )
        with h5py.File(path, "x") as data:
            data.attrs["schema"] = "pixel-proprio-metadata-v1"
            arrays = dict(pose=pose, pose_valid=np.ones(len(time), bool), time=time)
            if actions is not None:
                arrays.update(action=actions, action_valid=np.ones(len(time), bool))
            for field, value in arrays.items():
                data.create_dataset(field, data=value)
        item = dict(
            embodiment=ep["embodiment"],
            episode_id=ep["episode_id"],
            split=ep["split"],
            original_split=ep["original_split"],
            frames=len(time),
            path=path.name,
            sha256=sha256(path),
            frame="fixed_reference",
            pose_kind="measured_pose",
            native_source=ep["native_source"],
            cameras=cameras,
            rgb_path=rgb_path.name,
            rgb_sha256=rgb_sha,
            **binding,
        )
        if "task_id" in ep:
            item["task_id"] = ep["task_id"]
        if actions is not None:
            native = ep["native_source"]
            receipt = dict(
                schema="pixel-proprio-action-alignment-v1",
                source_demo_sha256=native["source_demo_sha256"],
                source_episode=native["episode"],
                relation="action_t_applied_to_state_t",
                producer_sha256=producer_sha,
                basis={
                    "cache_index_source": "SHA-verified native source_row equals frame index; RGB time, pose time and state[:,0] are exactly equal",
                    "action_index_source": f"Raw {native['source_demo']} data/{native['episode']}/actions; verified export {raw_metadata_sha}",
                    "state_action_relation_source": json.dumps(
                        raw_metadata["producer_convention"], sort_keys=True
                    ),
                },
                cache_frame_index=list(range(len(time))),
                source_state_index=rows.tolist(),
                source_action_index=rows.tolist(),
                raw_export_metadata_sha256=raw_metadata_sha,
                raw_payload_sha256=raw_metadata["payload_sha256"],
                producer_convention=raw_metadata["producer_convention"],
                pixel_source_binding=binding,
            )
            receipt_path = output / f"{name}.alignment.json"
            _write_json(receipt_path, receipt)
            item["action_provenance"] = dict(
                path=receipt_path.name, sha256=sha256(receipt_path)
            )
        result_episodes.append(item)
    # Cache exact content proofs for the newly finalized files when their ctime
    # is old enough. The final manifest binds these same-device fast paths.
    for item in result_episodes:
        for field, digest_field, fingerprint_field in (
            ("path", "sha256", "verified_fingerprint"),
            ("rgb_path", "rgb_sha256", "rgb_verified_fingerprint"),
        ):
            proof = verifier.verify(output / item[field], item[digest_field])
            if proof is not None:
                item[fingerprint_field] = proof
    stats = dict(
        schema="pixel-proprio-normalization-v1",
        scope="train_only",
        train_episodes=training_identities(result_episodes),
        statistics={
            f"pose/{emb}": moments[emb].result(std_floor) for emb in sorted(embodiments)
        },
        std_floor=std_floor,
    )
    stats_path = output / "normalization.json"
    _write_json(stats_path, stats)
    manifest = dict(
        schema="pixel-proprio-prepared-v1",
        episodes=result_episodes,
        pose_schemas=schemas,
        pixel_identity=dict(
            image_size=list(image_size),
            patch_size=patch_size,
            channels=3,
            patch_order="row_major_hwc",
            normalization="uint8_to_minus_one_one",
            resize="integer_area_mean_round_ties_to_even",
            resize_numpy_version=np.__version__,
            crop="none",
            temporal_resampling="none",
        ),
        normalization=dict(
            scope="train_only",
            arrays_normalized=False,
            path=stats_path.name,
            sha256=sha256(stats_path),
        ),
        source_manifest_sha256=source_sha,
        native_locations_sha256=locations_sha,
        selected_embodiments=sorted(embodiments),
        split_provenance={
            **split_provenance,
            "path": str(split_path.relative_to(output)),
            "sha256": sha256(split_path),
        },
        preparer_sha256=producer_sha,
    )
    if raw_metadata:
        manifest.update(
            action_schemas={"libero_gripper": raw_metadata["action_schema"]},
            raw_export_metadata_sha256=raw_metadata_sha,
            raw_payload_sha256=raw_metadata["payload_sha256"],
        )
    verifier.verify(source_manifest, source_sha)
    verifier.verify(native_spec, locations_sha)
    _write_json(output / "manifest.json", manifest)
    return dict(
        status="PASS",
        manifest=str(output / "manifest.json"),
        manifest_sha256=sha256(output / "manifest.json"),
        episodes=len(values),
        frames=sum(len(item[2]) for item in values),
        image_size=list(image_size),
        patch_size=patch_size,
        rgb_copied=True,
        resize="integer_area_mean_round_ties_to_even",
        action_episodes=sum(item[-1] is not None for item in values),
        split_counts=split_provenance["counts"],
        derived_split_identities_sha256=split_provenance["derived_identities_sha256"],
        split_provenance_sha256=sha256(split_path),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--native-spec", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--image-size", required=True, type=int, nargs=2, metavar=("HEIGHT", "WIDTH")
    )
    parser.add_argument("--patch-size", required=True, type=int)
    parser.add_argument(
        "--embodiments",
        required=True,
        nargs="+",
        choices=("libero_gripper", "libero_sharpa"),
    )
    parser.add_argument("--raw-export-dir", type=Path)
    parser.add_argument("--raw-source-manifest", type=Path)
    parser.add_argument("--std-floor", type=float, default=1e-5)
    parser.add_argument("--resize-batch-size", type=int, default=4)
    parser.add_argument("--validation-episodes-per-task", type=int)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare_pixels(**vars(args))), flush=True)


if __name__ == "__main__":
    main()
