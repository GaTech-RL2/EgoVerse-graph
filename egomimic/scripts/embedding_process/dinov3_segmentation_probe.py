"""FP32 frozen-feature binary segmentation probe with episode-held-out evaluation.

Annotation manifests bind original RGB hashes to visible-hand polygons, holes,
and uncertain regions. Labels must be visually reviewed before training. This
tool never makes masks from landmarks or treats model-generated masks as truth.
Uses the existing DINO cache verifier and landmark probe's split/count helpers.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from torch import nn

from egomimic.scripts.embedding_process.dinov3_correspondence import cached_features
from egomimic.scripts.embedding_process.dinov3_pose_probe import episode_split, parameter_count


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def rasterize(frame, hw, band):
    h, w = hw
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for polygon in frame["hand_polygons"]:
        draw.polygon([tuple(p) for p in polygon], fill=1)
    for polygon in frame.get("background_holes", []):
        draw.polygon([tuple(p) for p in polygon], fill=0)
    mask = np.asarray(mask).astype(bool)
    uncertain = Image.new("L", (w, h), 0)
    for polygon in frame.get("ignore_polygons", []):
        ImageDraw.Draw(uncertain).polygon([tuple(p) for p in polygon], fill=1)
    certain = ~np.asarray(uncertain).astype(bool)
    edge_band = binary_dilation(mask, iterations=band) ^ binary_erosion(mask, iterations=band)
    return mask, certain, certain & ~edge_band


class SegmentationProbe(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.head = nn.Linear(dim, 2)

    def forward(self, features):
        return self.head(F.layer_norm(features, (features.shape[-1],)))


def patch_targets(mask, valid, prep):
    """Pixel-area targets; ignore padding and uncertain labels before reduction."""
    geometry = prep["frames"][0]
    rh, rw = geometry["resized_hw"]
    top, left = geometry["padding_top_left"]
    ph, pw = prep["processed_hw"]
    gh, gw = prep["patch_grid"]
    planes = torch.tensor(np.stack([(~mask) & valid, mask & valid]), dtype=torch.float32)[None]
    # Area weights retain thin structures and ignore coverage, unlike nearest labels.
    resized = F.interpolate(planes, size=(rh, rw), mode="area")
    padded = F.pad(resized, (left, pw - rw - left, top, ph - rh - top))
    counts = F.adaptive_avg_pool2d(padded, (gh, gw))[0].flatten(1).T
    if counts.shape != (gh * gw, 2) or not torch.isfinite(counts).all():
        raise ValueError("Invalid label/grid geometry")
    return counts


def objective(logits, counts, class_weights):
    weights = counts * class_weights
    denom = weights.sum()
    if denom <= 0:
        raise ValueError("No labeled pixels")
    return -(weights * logits.log_softmax(-1)).sum() / denom


def native_probability(logits, prep):
    gh, gw = prep["patch_grid"]
    geometry = prep["frames"][0]
    top, left = geometry["padding_top_left"]
    rh, rw = geometry["resized_hw"]
    # Interpolate logits, crop letterbox, then restore original image coordinates.
    grid = logits.T.reshape(1, 2, gh, gw)
    grid = F.interpolate(grid, size=prep["processed_hw"], mode="bilinear", align_corners=False)
    grid = grid[:, :, top:top + rh, left:left + rw]
    native = F.interpolate(grid, size=geometry["original_hw"], mode="bilinear", align_corners=False)
    return native.softmax(1)[0, 1].numpy()


def overlap(pred, target, valid):
    tp = int((pred & target & valid).sum())
    fp = int((pred & ~target & valid).sum())
    fn = int((~pred & target & valid).sum())
    return {"iou": tp / max(1, tp + fp + fn), "dice": 2 * tp / max(1, 2 * tp + fp + fn),
            "tp": tp, "fp": fp, "fn": fn}


def boundary_f1(pred, target, certain, tolerance):
    # Ignore clipped image edges and artificial cut edges around ambiguous wrists.
    valid = binary_erosion(certain, iterations=tolerance + 1, border_value=0)
    p = (pred ^ binary_erosion(pred)) & valid
    t = (target ^ binary_erosion(target)) & valid
    if not p.any() or not t.any():
        return 0.0
    precision = float((distance_transform_edt(~t)[p] <= tolerance).mean())
    recall = float((distance_transform_edt(~p)[t] <= tolerance).mean())
    return 2 * precision * recall / max(1e-12, precision + recall)


def load_labels(path):
    manifest = json.loads(Path(path).read_text())
    if not manifest.get("annotation_source") or manifest["boundary_ignore_px"] < 1:
        raise ValueError("Explicit label origin and positive uncertainty band required")
    data, meta = [], []
    for frame in manifest["frames"]:
        if sha(frame["path"]) != frame["sha256"]:
            raise ValueError("Image SHA mismatch")
        image = np.asarray(Image.open(frame["path"]).convert("RGB"))
        mask, certain, valid = rasterize(frame, image.shape[:2], manifest["boundary_ignore_px"])
        if not (mask & valid).any() or not (~mask & valid).any():
            raise ValueError("Each annotation needs certain foreground and background")
        data.append((image, mask, certain, valid))
        meta.append({"episode": frame["episode"], "image_sha256": frame["sha256"]})
    if len({m["episode"] for m in meta}) < 3:
        raise ValueError("At least three distinct episodes required")
    for episode in sorted({m["episode"] for m in meta}):
        episode_split(meta, episode)
    return manifest, data, meta


def review(args):
    import matplotlib.pyplot as plt
    manifest, data, _ = load_labels(args.annotations)
    fig, axes = plt.subplots(len(data), 2, figsize=(12, 3.6 * len(data)), squeeze=False)
    for i, (image, mask, certain, valid) in enumerate(data):
        axes[i, 0].imshow(image)
        axes[i, 1].imshow(image)
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask & valid] = [0, 1, 0.5, .45]
        overlay[~valid] = [1, .65, 0, .65]
        axes[i, 1].imshow(overlay)
        axes[i, 1].contour(mask, levels=[.5], colors="white", linewidths=.6)
        axes[i, 0].set_title(manifest["frames"][i]["id"])
        axes[i, 1].set_title("Green: hand | orange: ignored | white: polygon")
        for ax in axes[i]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.output, dpi=100)
    plt.close(fig)


def prepare(args):
    manifest, data, meta = load_labels(args.annotations)
    if manifest.get("review_status") != "visually_reviewed_before_training":
        raise ValueError("Mask visual review must precede training")
    cache = cached_features(args.feature_results, manifest["model"], manifest["revision"])
    packed = {}
    for side in args.sides:
        xs, targets, preps, receipts = [], [], [], []
        for frame, (_, mask, _, valid) in zip(manifest["frames"], data):
            features, prep, receipt = cache[(side, frame["sha256"])]
            if tuple(prep["frames"][0]["original_hw"]) != mask.shape:
                raise ValueError("Annotation and feature geometry disagree")
            xs.append(torch.from_numpy(features))
            targets.append(patch_targets(mask, valid, prep))
            preps.append(prep)
            receipts.append(receipt)
        packed[side] = (torch.stack(xs), torch.stack(targets), preps, receipts)
    return manifest, data, meta, packed


def preflight(args, manifest, meta, packed):
    if min(args.steps, args.save_every, args.threads) < 1 or args.lr <= 0:
        raise ValueError("Positive training settings required")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if list(out.glob("side-*")) or (out / "COMPLETE.json").exists():
        raise FileExistsError("Preserve previous probe outputs; no overwrite or pruning")
    dim = next(iter(packed.values()))[0].shape[-1]
    head = SegmentationProbe(dim)
    trainable = sum(p.numel() for p in head.parameters())
    frozen = parameter_count(manifest)
    episodes = sorted({r["episode"] for r in meta})
    count = len(args.sides) * len(episodes) * ((args.steps + args.save_every - 1) // args.save_every)
    estimated_bytes = trainable * 12 + 65536
    free = shutil.disk_usage(out).free
    if free < count * estimated_bytes + 100_000_000:
        raise RuntimeError("Insufficient storage for all immutable checkpoints")
    config = {"status": "PASS", "total_parameters": frozen + trainable, "frozen_parameters": frozen,
              "trainable_parameters": trainable, "feature_dim": dim, "output_classes": 2,
              "action_dim": None, "denoising_dim": None, "latent_bottleneck": None,
              "sides": args.sides, "tokens": {s: int(packed[s][0].shape[1]) for s in args.sides},
              "steps_per_fold": args.steps, "save_every": args.save_every, "seed": args.seed,
              "optimizer": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay,
              "scheduler": None, "precision": "float32", "device": "cpu", "cpu_threads": args.threads,
              "batch": "all training images per update", "folds": len(episodes),
              "splits": {ep: dict(zip(("train", "test"), episode_split(meta, ep))) for ep in episodes},
              "checkpoint_count": count, "estimated_bytes_each": estimated_bytes, "free_bytes": free,
              "naming": "side-S/fold-F/epoch_EEEE-step_SSSSSSSSS.pt",
              "triggers": "fixed step cadence plus terminal only if off cadence; retain all; no signal saves",
              "output": str(out.resolve()), "annotation_sha256": sha(args.annotations),
              "source_sha256": sha(__file__), "features": {s: packed[s][3] for s in args.sides},
              "fairness": "FAIR WITH CAVEATS: paired seed, full-batch order, steps and labels; intended resolution/token difference; equal updates not equal compute; nine model-annotated images only",
              "selection": "terminal checkpoint fixed before training; no selection on test scores",
              "threshold": .5, "postprocessing": None,
              "annotation_source": manifest["annotation_source"]}
    save_json(out / "PREFLIGHT.json", config)
    return config


def run(args):
    torch.set_num_threads(args.threads)
    manifest, data, meta, packed = prepare(args)
    config = preflight(args, manifest, meta, packed)
    print(json.dumps(config, indent=2), flush=True)
    if args.preflight_only:
        return
    out = Path(args.output)
    episodes = sorted({r["episode"] for r in meta})
    rows, checkpoints, init_hashes = [], [], []
    for side, (x, targets, preps, _) in packed.items():
        for fold, episode in enumerate(episodes):
            train, test = episode_split(meta, episode)
            folder = out / f"side-{side}" / f"fold-{fold}"
            folder.mkdir(parents=True, exist_ok=False)
            torch.manual_seed(args.seed)
            model = SegmentationProbe(x.shape[-1])
            init_hashes.append(sha_tensor_state(model.state_dict()))
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            counts = targets[train]
            # Native labeled pixel counts keep class balancing exactly paired
            # across resolutions, independent of grid/resize rounding.
            frequencies = torch.tensor([
                sum(int(((~data[i][1]) & data[i][3]).sum()) for i in train),
                sum(int((data[i][1] & data[i][3]).sum()) for i in train),
            ], dtype=torch.float32)
            if (frequencies <= 0).any():
                raise ValueError("Missing training class")
            weights = frequencies.sum() / (2 * frequencies)
            losses = []
            for step in range(1, args.steps + 1):
                optimizer.zero_grad(set_to_none=True)
                loss = objective(model(x[train]), counts, weights)
                loss.backward()
                gradients = torch.cat([p.grad.flatten() for p in model.parameters()])
                if not torch.isfinite(loss) or not torch.isfinite(gradients).all() or gradients.norm() == 0:
                    raise RuntimeError("Nonfinite loss/gradients or zero gradient")
                optimizer.step()
                losses.append(float(loss.detach()))
                if step % args.save_every == 0 or step == args.steps:
                    path = folder / f"epoch_{step:04d}-step_{step:09d}.pt"
                    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                "epoch": step, "step": step, "config": config,
                                "side": side, "heldout_episode": episode, "class_weights": weights}, path)
                    saved = torch.load(path, weights_only=False, map_location="cpu")
                    restored = SegmentationProbe(x.shape[-1])
                    restored.load_state_dict(saved["model"], strict=True)
                    restored_opt = torch.optim.AdamW(restored.parameters(), lr=args.lr)
                    restored_opt.load_state_dict(saved["optimizer"])
                    with torch.no_grad():
                        if not torch.equal(restored(x[:1]), model(x[:1])):
                            raise RuntimeError("Strict model reload changed predictions")
                    for key in optimizer.state_dict()["state"]:
                        for name, value in optimizer.state_dict()["state"][key].items():
                            if not torch.equal(value, restored_opt.state_dict()["state"][key][name]):
                                raise RuntimeError("Optimizer reload mismatch")
                    checkpoints.append({"path": str(path.resolve()), "size_bytes": path.stat().st_size,
                                        "sha256": sha(path), "global_step": step,
                                        "strict_model_optimizer_reload": True})
            save_json(folder / "losses.json", losses)
            with torch.no_grad():
                logits = model(x)
            prior_numerator = np.stack([data[i][1] & data[i][3] for i in train]).sum(0)
            prior_denominator = np.stack([data[i][3] for i in train]).sum(0)
            prior = prior_numerator / np.maximum(1, prior_denominator) >= .5
            for index in train + test:
                probability = native_probability(logits[index], preps[index])
                prediction = probability >= .5
                _, mask, certain, valid = data[index]
                row = {"side": side, "fold": fold, "split": "test" if index in test else "train",
                       "id": manifest["frames"][index]["id"], "episode": meta[index]["episode"],
                       "full_polygon": overlap(prediction, mask, certain),
                       "uncertainty_excluded": overlap(prediction, mask, valid),
                       "training_spatial_prior": overlap(prior, mask, valid),
                       "boundary_f1_5px": boundary_f1(prediction, mask, certain, manifest["boundary_tolerance_px"])}
                if index in test:
                    dest = folder / (row["id"] + ".npz")
                    np.savez_compressed(dest, probability=probability, prediction=prediction,
                                        polygon_mask=mask, certain=certain, scored=valid)
                    row["prediction"] = str(dest.resolve())
                    row["prediction_sha256"] = sha(dest)
                rows.append(row)
            print(f"side={side} fold={fold} loss={losses[-1]:.6f} complete", flush=True)
    if len(set(init_hashes)) != 1:
        raise RuntimeError("Paired head initialization mismatch")
    summary = {}
    for side in args.sides:
        tests = [r for r in rows if r["side"] == side and r["split"] == "test"]
        summary[side] = {"test_images": len(tests), "episodes": len(episodes)}
        for mode in ("full_polygon", "uncertainty_excluded", "training_spatial_prior"):
            summary[side][mode] = {name: float(np.mean([r[mode][name] for r in tests])) for name in ("iou", "dice")}
        summary[side]["boundary_f1_5px"] = float(np.mean([r["boundary_f1_5px"] for r in tests]))
    save_json(out / "predictions.json", rows)
    save_json(out / "SUMMARY.json", summary)
    save_json(out / "COMPLETE.json", {"status": "COMPLETE_EXPLORATORY_AGENT_ANNOTATED_MASKS",
              "config": config, "summary": summary, "checkpoints": checkpoints,
              "paired_initialization_sha256": init_hashes[0], "predictions_sha256": sha(out / "predictions.json")})


def sha_tensor_state(state):
    return hashlib.sha256(b"".join(t.detach().numpy().tobytes() for t in state.values())).hexdigest()


def plot_results(args):
    import matplotlib.pyplot as plt
    manifest, data, _ = load_labels(args.annotations)
    out = Path(args.output)
    rows = json.loads((out / "predictions.json").read_text())
    rows = {(r["side"], r["id"]): r for r in rows if r["split"] == "test"}
    # Predeclared middle frame per episode, not selected by error.
    for label, indices in (("heldout-examples", [1, 4, 7]), ("all-heldout", list(range(len(data))))):
        fig, axes = plt.subplots(len(indices), 4, figsize=(16, 3.7 * len(indices) + 1), squeeze=False)
        fig.suptitle("Frozen DINOv3 hand segmentation | FP32 | held-out episodes\n"
                     "Exploratory Codex polygon labels; not independently human-annotated ground truth", fontsize=15)
        for y, i in enumerate(indices):
            image, mask, _, valid = data[i]
            frame = manifest["frames"][i]
            for ax in axes[y]:
                ax.imshow(image)
                ax.axis("off")
            axes[y, 0].set_title(f"Original • frame {frame['frame']}")
            overlay = np.zeros((*mask.shape, 4))
            overlay[mask & valid] = [0, 1, .5, .5]
            overlay[~valid] = [1, .65, 0, .55]
            axes[y, 1].imshow(overlay)
            axes[y, 1].set_title("Reference mask • orange ignored")
            for col, side in enumerate(args.sides, 2):
                row = rows[(side, frame["id"])]
                if sha(row["prediction"]) != row["prediction_sha256"]:
                    raise ValueError("Prediction SHA mismatch")
                pred = np.load(row["prediction"], allow_pickle=False)["prediction"]
                overlay[:] = 0
                overlay[pred] = [.05, .65, 1., .55]
                axes[y, col].imshow(overlay)
                axes[y, col].contour(mask, levels=[.5], colors="lime", linewidths=.5)
                axes[y, col].set_title(f"{side} prediction • IoU {row['uncertainty_excluded']['iou']:.1%}")
        fig.subplots_adjust(left=.01, right=.99, bottom=.02,
                            top=.90 if len(indices) == 3 else .96,
                            hspace=.24, wspace=.04)
        fig.savefig(out / f"{label}.png", dpi=140)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--review-only", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--feature-results", nargs="+", default=[])
    parser.add_argument("--sides", nargs="+", type=int, default=[448, 640])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--weight-decay", type=float, default=.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.review_only:
        review(args)
    elif args.plot_only:
        plot_results(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
