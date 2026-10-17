"""Stream verified native RGB through the maintained DINO forward into raw HDF5.

These caches deliberately remain unnormalized. Dataset splitting, train-only
statistics and the training manifest belong to world_model_preparation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from egomimic.rldb.world_model import open_prepared_episode
from egomimic.rldb.world_model_preparation import StreamingMoments


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _require_hash(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"Artifact hash mismatch: {path}")


def _safe_name(name):
    if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("Unsafe episode/embodiment identity")
    return name


def _source_identity(native_cache, rgb_cache, rgb_receipt):
    native_cache, rgb_cache, rgb_receipt = map(Path, (native_cache, rgb_cache, rgb_receipt))
    manifest_path = native_cache / "manifest.json"
    manifest, receipt = _json(manifest_path), _json(rgb_receipt)
    if manifest.get("schema") != "libero-same-recorded-state-v1":
        raise ValueError("Unsupported native cache contract")
    if (receipt.get("schema") != "libero-native-rgb-v1" or receipt.get("status") != "PASS"
            or receipt.get("native_render") is not True
            or receipt.get("native_geometry_replay_exact") is not True):
        raise ValueError("Require a successful native RGB and exact geometry receipt")
    _require_hash(manifest_path, receipt["source_manifest_sha256"])
    _require_hash(native_cache / "aligned.npz", receipt["source_aligned_sha256"])
    _require_hash(native_cache / "aligned.npz", manifest["artifacts"]["aligned.npz"])
    _require_hash(rgb_cache, receipt["rgb_cache_sha256"])
    if receipt["cameras"] != manifest["cameras"]:
        raise ValueError("Native and RGB camera ordering differs")
    return manifest, receipt, {
        "native_manifest_sha256": sha256(manifest_path),
        "native_aligned_sha256": receipt["source_aligned_sha256"],
        "rgb_cache_sha256": receipt["rgb_cache_sha256"],
        "rgb_receipt_sha256": sha256(rgb_receipt),
    }


def _geometry(native, manifest, rgb, receipt, side):
    times = native["time"]
    shape = rgb["rgb"].shape
    if (len(shape) != 5 or shape[-1] != 3 or rgb["rgb"].dtype != np.uint8
            or shape[:2] != (len(times), len(manifest["cameras"]))
            or shape[2:4] != (receipt["height"], receipt["width"])
            or len(times) != receipt["frames"]):
        raise ValueError("RGB dimensions disagree with native episode/receipt")
    # No low-resolution source may be silently upscaled to the chosen DINO side.
    if min(shape[2:4]) < side:
        raise ValueError("Native RGB resolution is below requested DINO resolution")
    if shape[2] != shape[3] or manifest["height"] != manifest["width"]:
        raise ValueError("Native simulator cache requires square camera views; do not infer letterbox track coordinates")
    if not np.isfinite(times).all() or not (np.diff(times) > 0).all():
        raise ValueError("Require finite increasing physical timestamps")
    np.testing.assert_array_equal(times, rgb["time"][:], err_msg="Native/RGB time alignment")
    np.testing.assert_array_equal(native["source_row"], np.arange(len(times)))
    pose, track = native["pose"], native["xyz"]
    if (pose.shape != (len(times), len(manifest["pose_fields"]))
            or track.ndim != 3 or track.shape[0] != len(times) or track.shape[2] != 3
            or not np.isfinite(pose).all() or not np.isfinite(track).all()):
        raise ValueError("Invalid measured pose/track arrays")
    available = native["coordinate_available"]
    if available.dtype != np.bool_ or available.shape != track.shape[:2] or not available.all():
        raise ValueError("Native simulator coordinates must remain available under occlusion")
    xy = native["reference_grid_xy"][native["valid_grid_indices"]].astype(np.float32)
    if xy.shape != (track.shape[1], 2):
        raise ValueError("Track reference grid mismatch")
    xy = xy / np.array([manifest["width"] - 1, manifest["height"] - 1], np.float32) * 2 - 1
    return {
        "time": times.astype(np.float64), "track": track.astype(np.float32),
        "track_xy": xy, "track_camera": np.zeros(track.shape[1], np.int64),
        "track_confidence": available.astype(np.float32), "pose": pose.astype(np.float32),
        "pose_valid": np.ones(len(times), bool), "visibility": native["visibility"],
    }


def extract_episode(*, native_cache, rgb_cache, rgb_receipt, output, embodiment,
                    episode_id, embedder, model_identity, batch_size=4):
    """Write one immutable, raw FP32 cache; no corpus-sized in-memory arrays."""
    if batch_size < 1 or embedder.dtype != torch.float32 or not embedder.image_side:
        raise ValueError("Require positive batch size and explicit FP32 DINO resolution")
    if any(p.dtype != torch.float32 for p in embedder.model.parameters()):
        raise ValueError("DINO weights must be FP32")
    if embedder.model.training or any(p.requires_grad for p in embedder.model.parameters()):
        raise ValueError("DINO must be frozen in evaluation mode")
    if embedder.device.type == "cuda" and (torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
        raise ValueError("FP32 feature extraction requires TF32 disabled")
    for key in ("model", "revision", "weights_sha256", "config_sha256", "processor_sha256"):
        if not model_identity.get(key):
            raise ValueError(f"Missing pinned DINO {key}")
    _safe_name(embodiment); _safe_name(episode_id)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest, receipt, inputs = _source_identity(native_cache, rgb_cache, rgb_receipt)
    request = {"inputs": inputs, "model": model_identity, "image_side": embedder.image_side,
               "batch_size": batch_size, "embodiment": embodiment, "episode_id": episode_id,
               "producer_sha256": sha256(__file__)}
    saved_receipt = output.with_suffix(output.suffix + ".json")
    if output.exists():
        prior = _json(saved_receipt)
        if prior.get("status") != "PASS" or prior.get("request") != request:
            raise ValueError("Existing feature cache request differs or is incomplete")
        _require_hash(output, prior["output_sha256"])
        return {**prior, "reused": True}
    lock = output.with_suffix(output.suffix + ".lock")
    temporary = output.with_suffix(output.suffix + ".partial")
    with lock.open("x"):
        pass
    try:
        if temporary.exists():
            raise FileExistsError(f"Prior partial cache requires inspection: {temporary}")
        metadata = None
        moments = {name: StreamingMoments() for name in ("image", "track", "pose")}
        with open_prepared_episode(Path(native_cache) / "aligned.npz") as native, h5py.File(rgb_cache, "r") as rgb:
            arrays = _geometry(native, manifest, rgb, receipt, embedder.image_side)
            moments["track"].update(arrays["track"][arrays["track_confidence"] > 0])
            moments["pose"].update(arrays["pose"][arrays["pose_valid"]])
            n, cameras = rgb["rgb"].shape[:2]
            with h5py.File(temporary, "x") as target:
                target.attrs["arrays_normalized"] = False
                target.attrs["reference_frame"] = manifest["reference_frame"]
                for name, value in arrays.items():
                    target.create_dataset(name, data=value, compression="lzf")
                image_dataset = None
                for camera in range(cameras):
                    for start in range(0, n, batch_size):
                        end = min(n, start + batch_size)
                        images = list(rgb["rgb"][start:end, camera])
                        with torch.autocast(device_type=embedder.device.type, enabled=False):
                            features = embedder._embed_batch(images)
                        current = copy.deepcopy(embedder.last_preprocessing)
                        frames = current.pop("frames")
                        if not frames or any(item != frames[0] for item in frames):
                            raise ValueError("Inconsistent source image geometry")
                        current["geometry"] = frames[0]
                        if metadata is not None and current != metadata:
                            raise ValueError("DINO preprocessing changed within episode")
                        metadata = current
                        grid = metadata["patch_grid"]
                        if (features.dtype != np.float32 or not np.isfinite(features).all()
                                or features.shape != (end - start, int(np.prod(grid)), 768)
                                or metadata["processed_hw"] != [embedder.image_side] * 2
                                or metadata["precision"] != "torch.float32"):
                            raise ValueError("Invalid DINO FP32 patch features/metadata")
                        if image_dataset is None:
                            image_dataset = target.create_dataset(
                                "image", shape=(n, cameras, features.shape[1], features.shape[2]),
                                dtype="f4", chunks=(1, 1, features.shape[1], features.shape[2]))
                            yy, xx = np.meshgrid((np.arange(grid[0]) + .5) / grid[0] * 2 - 1,
                                                 (np.arange(grid[1]) + .5) / grid[1] * 2 - 1,
                                                 indexing="ij")
                            target.create_dataset("image_xy", data=np.stack([xx, yy], -1).reshape(-1, 2).astype(np.float32))
                        image_dataset[start:end, camera] = features
                        moments["image"].update(features)
                target.attrs["dino_preprocessing"] = json.dumps(metadata)
        result = {
            "schema": "mot-world-feature-cache-v1", "status": "PASS", "request": request,
            "arrays_normalized": False, "training_ready": False, "embodiment": embodiment,
            "episode_id": episode_id, "frames": n, "pose_fields": manifest["pose_fields"],
            "cameras": manifest["cameras"], "reference_frame": manifest["reference_frame"],
            "native_cache": str(Path(native_cache).resolve()),
            "native_source": {key: manifest.get(key) for key in ("source_demo", "source_demo_sha256", "episode")},
            "xy_convention": "minus_one_to_one; image patch centers; camera0 reference track pixels",
            "track_confidence_policy": "exact simulator coordinate availability, independent of visibility",
            "dino_identity": {**model_identity, "image_size": metadata["processed_hw"],
                              "patch_grid": metadata["patch_grid"], "feature_layer": metadata["feature_layer"],
                              "precision": "float32"},
            "dino_preprocessing": metadata, "native_rgb_receipt": receipt,
            "moments": {name: {"count": value.count, "mean": value.mean.tolist(),
                               "m2": value.m2.tolist()} for name, value in moments.items()},
            "output": str(output), "output_sha256": sha256(temporary),
        }
        # Hard-link publication refuses to replace an existing immutable artifact.
        os.link(temporary, output)
        temporary.unlink()
        with saved_receipt.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        return result
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="Explicit episodes and pinned local DINO model identity")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard")
    spec = _json(args.spec)
    versions = {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers", "torch", "h5py", "numpy")}
    for name, expected in spec.get("runtime_versions", {}).items():
        if versions.get(name) != expected:
            raise ValueError(f"Runtime {name} differs from requested version {expected}")
    if args.device.startswith("cuda") and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise ValueError("DINO cache extraction requires exactly one visible CUDA GPU")
    print(json.dumps({"status": "DINO_RUNTIME_READY", "versions": versions, "device": args.device}), flush=True)
    identities, physical = set(), set()
    if not spec.get("episodes"):
        raise ValueError("Explicit nonempty episode inventory required")
    for episode in spec["episodes"]:
        key = (_safe_name(episode["embodiment"]), _safe_name(episode["episode_id"]))
        native = str(Path(episode["native_cache"]).resolve())
        if key in identities or native in physical:
            raise ValueError("Duplicate feature-cache identity or native source")
        identities.add(key); physical.add(native)
    local_model = Path(spec["model"]["path"])
    _require_hash(local_model / "model.safetensors", spec["model"]["weights_sha256"])
    _require_hash(local_model / "config.json", spec["model"]["config_sha256"])
    _require_hash(local_model / "preprocessor_config.json", spec["model"]["processor_sha256"])
    from egomimic.scripts.embedding_process.dinov3_embedding import DINOv3ImageEmbedding
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    embedder = DINOv3ImageEmbedding(input_keys=["rgb"], model_name=str(local_model),
                                  revision=spec["model"]["revision"], dtype=torch.float32,
                                  image_side=spec["image_side"], device=args.device)
    embedder.setup()
    identity = {key: spec["model"][key] for key in ("model", "revision", "weights_sha256", "config_sha256", "processor_sha256")}
    for index, episode in enumerate(spec["episodes"]):
        key = (_safe_name(episode["embodiment"]), _safe_name(episode["episode_id"]))
        if index % args.shard_count != args.shard_index:
            continue
        result = extract_episode(**episode, output=Path(args.output) / key[0] / (key[1] + ".h5"),
                                 embedder=embedder, model_identity=identity,
                                 batch_size=spec.get("batch_size", 4))
        print(json.dumps({key: result[key] for key in ("status", "episode_id", "frames", "output", "output_sha256")}), flush=True)


if __name__ == "__main__":
    main()
