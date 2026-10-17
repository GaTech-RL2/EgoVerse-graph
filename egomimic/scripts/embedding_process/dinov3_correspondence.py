"""Screen frozen DINO resolutions using independent visible-point annotations.

This is descriptor matching, not a learned tracker. Manifest schema:
{model, revision, frames: [{id, path, sha256, episode,
points: {left_index: {xy: [x,y], visible: true, annotation_source: "..."}}}],
pairs: [[source_id, target_id], ...]}. Coordinates are original-image pixels.
Run with --manifest FILE --output DIR; no dataset writes or model substitution.

Optional reference_mode="projected_landmark" requires projection_provenance
and per-point reference_valid instead of visible. This reports agreement with
estimated landmarks, not visible-point accuracy; unknown occlusion is retained.
Use --feature-results for SHA-verified extraction reuse after changing labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from egomimic.scripts.embedding_process.dinov3_embedding import DINOv3ImageEmbedding


def patch_geometry(metadata):
    """Native-image patch centers and a mask excluding all padded/mixed patches."""
    gh, gw = metadata["patch_grid"]
    ph, pw = metadata["patch_hw"]
    geometry = metadata["frames"][0]
    h, w = geometry["original_hw"]
    rh, rw = geometry["resized_hw"]
    top, left = geometry["padding_top_left"]
    yy, xx = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    valid = (
        (xx * pw >= left)
        & ((xx + 1) * pw <= left + rw)
        & (yy * ph >= top)
        & ((yy + 1) * ph <= top + rh)
    )
    xy = np.stack(
        [
            ((xx + 0.5) * pw - left) * w / rw - 0.5,
            ((yy + 0.5) * ph - top) * h / rh - 0.5,
        ],
        -1,
    )
    if not valid.any():
        raise ValueError("No fully unpadded patches at this resolution")
    return xy.reshape(-1, 2), valid.reshape(-1)


def match_point(source, target, source_meta, target_meta, xy):
    """Bilinear source descriptor; cosine nearest neighbor among valid target patches."""
    g = source_meta["frames"][0]
    h, w = g["original_hw"]
    rh, rw = g["resized_hw"]
    top, left = g["padding_top_left"]
    processed_h, processed_w = source_meta["processed_hw"]
    qx = 2 * (((xy[0] + 0.5) * rw / w + left) / processed_w) - 1
    qy = 2 * (((xy[1] + 0.5) * rh / h + top) / processed_h) - 1
    gh, gw = source_meta["patch_grid"]
    _, source_valid = patch_geometry(source_meta)
    gx, gy = (qx + 1) * gw / 2 - 0.5, (qy + 1) * gh / 2 - 0.5
    x0, y0 = int(np.floor(gx)), int(np.floor(gy))
    for iy in (y0, y0 + 1):
        for ix in (x0, x0 + 1):
            weight = max(0, 1 - abs(gx - ix)) * max(0, 1 - abs(gy - iy))
            index = np.clip(iy, 0, gh - 1) * gw + np.clip(ix, 0, gw - 1)
            if weight > 1e-6 and not source_valid[index]:
                raise ValueError(
                    "Source descriptor touches a padded or mixed patch; exclude this label at every resolution"
                )
    feature_grid = torch.as_tensor(source).reshape(gh, gw, -1).permute(2, 0, 1)[None]
    query = F.grid_sample(
        feature_grid,
        torch.tensor([[[[qx, qy]]]], dtype=torch.float32),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).flatten()
    descriptors = F.normalize(torch.as_tensor(target), dim=-1)
    scores = descriptors @ F.normalize(query, dim=0)
    centers, valid = patch_geometry(target_meta)
    scores[~torch.from_numpy(valid)] = -torch.inf
    index = int(scores.argmax())
    return centers[index], float(scores[index])


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    mode = manifest.get("reference_mode", "visible_manual")
    if mode not in {"visible_manual", "projected_landmark"}:
        raise ValueError("Unknown reference_mode")
    if mode == "projected_landmark" and not manifest.get("projection_provenance"):
        raise ValueError("Projected references require projection_provenance")
    if not manifest.get("model") or not manifest.get("revision"):
        raise ValueError("An explicit model and immutable revision are required")
    if len(manifest["revision"]) != 40 or any(
        c not in "0123456789abcdef" for c in manifest["revision"]
    ):
        raise ValueError("revision must be a full lowercase Git commit")
    frames = {}
    for frame in manifest["frames"]:
        if frame["id"] in frames:
            raise ValueError("Duplicate frame id")
        file = Path(frame["path"])
        if not file.is_absolute():
            file = Path(path).parent / file
        if hashlib.sha256(file.read_bytes()).hexdigest() != frame["sha256"]:
            raise ValueError("Frame SHA mismatch")
        image = np.asarray(Image.open(file).convert("RGB"))
        for point in frame["points"].values():
            if not point.get("annotation_source"):
                raise ValueError("Independent annotation_source is required")
            if (
                mode == "visible_manual"
                and point.get("visible")
                and point.get("visibility_status", "").startswith("UNKNOWN")
            ):
                raise ValueError("Unknown visibility is not a visible manual label")
            if eligible(point, mode):
                x, y = point["xy"]
                if not (
                    np.isfinite([x, y]).all()
                    and 0 <= x < image.shape[1]
                    and 0 <= y < image.shape[0]
                ):
                    raise ValueError(
                        "Visible annotation must be inside the native image"
                    )
        frames[frame["id"]] = (frame, image)
    for a, b in manifest["pairs"]:
        if a == b or frames[a][0]["episode"] != frames[b][0]["episode"]:
            raise ValueError("Pairs must be distinct frames from the same episode")
        if not any(
            eligible(p, mode) and eligible(frames[b][0]["points"].get(k, {}), mode)
            for k, p in frames[a][0]["points"].items()
        ):
            raise ValueError(
                "Each pair needs at least one eligible correspondence for its reference mode"
            )
    if not manifest["pairs"]:
        raise ValueError("At least one pair is required")
    return manifest, frames


def eligible(point, mode):
    """Projection validity is not image visibility; never conflate the two."""
    key = "reference_valid" if mode == "projected_landmark" else "visible"
    return point.get(key) is True


def cached_features(result_paths, model, revision):
    cache = {}
    for result_path in result_paths:
        result_path = Path(result_path)
        result = json.loads(result_path.read_text())
        if (result["model"], result["revision"], result["precision"]) != (
            model,
            revision,
            "float32",
        ):
            raise ValueError("Feature cache model/revision/precision mismatch")
        for resolution in result["resolutions"]:
            for entry in resolution["feature_caches"].values():
                path = (result_path.parent / entry["path"]).resolve()
                if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                    raise ValueError("Feature cache SHA mismatch")
                with np.load(path, allow_pickle=False) as saved:
                    metadata = json.loads(str(saved["provenance_json"]))
                    features = saved["features"]
                prep = metadata["preprocessing"]
                if (
                    prep["model"] != model
                    or prep["resolved_revision"] != revision
                    or prep["policy"] != "full_fov_letterbox_bicubic"
                    or prep["precision"] != "torch.float32"
                    or prep["processed_hw"] != [resolution["side"]] * 2
                    or features.dtype != np.float32
                    or not np.isfinite(features).all()
                    or features.shape[0] != np.prod(prep["patch_grid"])
                ):
                    raise ValueError("Feature cache preprocessing/array mismatch")
                cache[(resolution["side"], metadata["frame_sha256"])] = (
                    features,
                    prep,
                    {"path": str(path), "sha256": entry["sha256"]},
                )
    return cache


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def run(manifest_path, output, sides, device, feature_results=(), features_only=False):
    manifest, frames = load_manifest(manifest_path)
    mode = manifest.get("reference_mode", "visible_manual")
    reuse = cached_features(feature_results, manifest["model"], manifest["revision"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        raise FileExistsError(
            "Preserve existing results; choose a fresh output directory"
        )
    extractor = DINOv3ImageEmbedding(
        ["image"],
        model_name=manifest["model"],
        revision=manifest["revision"],
        image_side=sides[0],
        device=device,
        dtype=torch.float32,
        batch_size=1,
    )
    extractor.setup()  # Access failure occurs before any model substitution or matching.
    results = {
        "model": manifest["model"],
        "revision": manifest["revision"],
        "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
        "precision": "float32",
        "device": str(device),
        "resolutions": [],
        "method": "feature_extraction_only"
        if features_only
        else "cosine_dense_descriptor_matching_not_learned_tracking",
        "reference_mode": mode,
        "metric_interpretation": "Agreement with projected estimated 3D landmarks; unknown occlusion, NOT verified visible-point accuracy"
        if mode == "projected_landmark"
        else "Error against visible manual annotations",
        "manifest_limitations": manifest.get("limitations", []),
    }
    for side in sides:
        extractor.image_side = side
        first = next(iter(frames.values()))[1]
        for _ in range(2):
            extractor._embed_batch([first])
        synchronize(extractor.device)
        if extractor.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(extractor.device)
        cache, latencies, cache_files = {}, [], {}
        cache_dir = output / f"features-{side}"
        cache_dir.mkdir(exist_ok=True)
        for key, (_, image) in frames.items():
            previous = reuse.get((side, frames[key][0]["sha256"]))
            if previous is not None:
                features, metadata, entry = previous
                cache[key] = (features, metadata)
                cache_files[key] = entry
                continue
            synchronize(extractor.device)
            start = time.perf_counter()
            features = extractor._embed_batch([image])[0]
            synchronize(extractor.device)
            latencies.append(time.perf_counter() - start)
            cache[key] = (features, extractor.last_preprocessing)
            cache_path = cache_dir / (
                hashlib.sha256(key.encode()).hexdigest()[:16] + ".npz"
            )
            if cache_path.exists():
                raise FileExistsError(f"Preserve existing feature cache: {cache_path}")
            provenance = {
                "frame_id": key,
                "frame_sha256": frames[key][0]["sha256"],
                "preprocessing": extractor.last_preprocessing,
            }
            np.savez_compressed(
                cache_path, features=features, provenance_json=json.dumps(provenance)
            )
            cache_files[key] = {
                "path": str(cache_path.relative_to(output)),
                "sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
            }
        matches, excluded = [], 0
        for a, b in ([] if features_only else manifest["pairs"]):
            source, target = frames[a][0], frames[b][0]
            sf, sm = cache[a]
            tf, tm = cache[b]
            for name, p in source["points"].items():
                q = target["points"].get(name)
                if not eligible(p, mode) or q is None or not eligible(q, mode):
                    excluded += 1
                    continue
                pred, cosine = match_point(sf, tf, sm, tm, p["xy"])
                back, _ = match_point(tf, sf, tm, sm, pred)
                candidates = {
                    k: np.linalg.norm(pred - v["xy"])
                    for k, v in target["points"].items()
                    if eligible(v, mode)
                }
                closest = min(candidates, key=candidates.get)
                matches.append(
                    {
                        "source": a,
                        "target": b,
                        "point": name,
                        "hand": p.get("hand"),
                        "region": p.get("region"),
                        "visibility_status": p.get("visibility_status"),
                        "prediction_xy": pred.tolist(),
                        "target_xy": q["xy"],
                        "error_native_px": float(np.linalg.norm(pred - q["xy"])),
                        "cycle_error_native_px": float(np.linalg.norm(back - p["xy"])),
                        "cosine": cosine,
                        "nearest_labeled_point": closest,
                        "wrong_labeled_point": closest != name
                        if len(candidates) > 1
                        else None,
                    }
                )
        errors = np.asarray([m["error_native_px"] for m in matches])
        metadata = next(iter(cache.values()))[1]
        result = {
            "side": side,
            "patch_tokens": int(np.prod(metadata["patch_grid"])),
            "image_only_attention_pairs_vs_first_resolution": (side / sides[0]) ** 4,
            "extract_median_seconds_including_transfer": float(np.median(latencies))
            if latencies
            else None,
            "reused_frames": len(frames) - len(latencies),
            "latencies_seconds": latencies,
            "matched_points": len(matches),
            "excluded_occluded_or_unlabeled": excluded,
            "mean_error_native_px": float(errors.mean()) if len(errors) else None,
            "pck": {str(t): float((errors <= t).mean()) for t in [8, 16, 32]}
            if len(errors)
            else None,
            "wrong_labeled_point_fraction": (
                float(
                    np.mean(
                        [
                            m["wrong_labeled_point"]
                            for m in matches
                            if m["wrong_labeled_point"] is not None
                        ]
                    )
                )
                if any(m["wrong_labeled_point"] is not None for m in matches)
                else None
            ),
            "wrong_finger_metric_limit": "Nearest annotated point only; null when fewer than two visible identities are labeled, not proof of correct finger identity",
            "preprocessing": {k: v[1] for k, v in cache.items()},
            "feature_caches": cache_files,
            "matches": matches,
        }
        if extractor.device.type == "cuda":
            result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(
                extractor.device
            )
        results["resolutions"].append(result)
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sides", type=int, nargs="+", default=[224, 336, 448, 640])
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--features-only", action="store_true",
        help="Extract verified image features without running point correspondence; preserves labels for downstream probes",
    )
    parser.add_argument(
        "--feature-results",
        nargs="*",
        default=[],
        help="SHA-verified existing result JSONs to reuse features",
    )
    args = parser.parse_args()
    if not args.sides or any(s < 16 or s % 16 for s in args.sides):
        parser.error("sides must be positive multiples of 16")
    run(args.manifest, args.output, args.sides, args.device, args.feature_results, args.features_only)


if __name__ == "__main__":
    main()
