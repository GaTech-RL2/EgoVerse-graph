"""Frozen-DINO-only linear landmark heatmap probe on cached image features.

Leave one whole episode out; fixed-budget training without test-based selection.
No track, pose, timestamp, camera-pose or target coordinates enter the readout.
Projected references remain estimated targets, not visible-joint ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from egomimic.scripts.embedding_process.dinov3_correspondence import (
    cached_features,
    eligible,
    load_manifest,
    patch_geometry,
)


class PoseHeatmapProbe(nn.Module):
    def __init__(self, feature_dim, landmarks):
        super().__init__()
        self.head = nn.Linear(feature_dim, landmarks, bias=False)

    def forward(self, features, valid_patches):
        features = F.layer_norm(features, (features.shape[-1],))
        logits = self.head(features).transpose(1, 2)
        return logits.masked_fill(~valid_patches[None, None], -1e9)


def heatmap_targets(xy, centers, valid_patches, sigma):
    distances = (xy[:, :, None] - centers[None, None]).square().sum(-1)
    logits = -distances / (2 * sigma**2)
    return logits.masked_fill(~valid_patches[None, None], -1e9).softmax(-1)


def objective(logits, targets, valid_labels):
    per_point = -(targets * logits.log_softmax(-1)).sum(-1)
    return (per_point * valid_labels).sum() / valid_labels.sum().clamp_min(1)


def mean_pose_baseline(xy, valid):
    count = valid.sum(0)
    if (count == 0).any():
        raise ValueError("Every landmark needs at least one training label")
    return (xy * valid[..., None]).sum(0) / count[:, None]


def episode_split(metadata, heldout):
    train = [i for i, row in enumerate(metadata) if row["episode"] != heldout]
    test = [i for i, row in enumerate(metadata) if row["episode"] == heldout]
    if not train or not test:
        raise ValueError("Nonempty train and held-out episodes required")
    if {metadata[i]["image_sha256"] for i in train} & {
        metadata[i]["image_sha256"] for i in test
    }:
        raise ValueError("Identical image bytes occur in both train and test")
    return train, test


def prepare(args):
    manifest, frames = load_manifest(args.manifest)
    cached = cached_features(
        args.feature_results, manifest["model"], manifest["revision"]
    )
    names = sorted(next(iter(frames.values()))[0]["points"])
    features, targets, valid, metadata = [], [], [], []
    centers = patch_valid = None
    for key, (frame, _) in frames.items():
        if sorted(frame["points"]) != names:
            raise ValueError("All frames require identical landmark schema")
        feature, prep, entry = cached[(args.side, frame["sha256"])]
        coordinates, mask = patch_geometry(prep)
        if centers is not None and (
            not np.array_equal(centers, coordinates)
            or not np.array_equal(patch_valid, mask)
        ):
            raise ValueError("Probe currently requires identical camera/image geometry")
        centers, patch_valid = coordinates, mask
        features.append(feature)
        label_valid = [
            eligible(
                frame["points"][n], manifest.get("reference_mode", "visible_manual")
            )
            for n in names
        ]
        target = np.asarray([frame["points"][n]["xy"] for n in names], np.float32)
        target[~np.asarray(label_valid)] = 0
        targets.append(target)
        valid.append(label_valid)
        metadata.append(
            {
                "id": key,
                "episode": frame["episode"],
                "image_sha256": frame["sha256"],
                "feature": entry,
            }
        )
    episodes = sorted({r["episode"] for r in metadata})
    if len(episodes) < 3:
        raise ValueError("At least three episodes required for this pilot")
    for episode in episodes:
        episode_split(metadata, episode)
    if not np.isfinite(np.stack(targets)).all():
        raise ValueError("Valid references must have finite coordinates")
    return (
        manifest,
        names,
        metadata,
        episodes,
        torch.from_numpy(np.stack(features)).float(),
        torch.from_numpy(np.stack(targets)).float(),
        torch.tensor(valid),
        torch.tensor(centers, dtype=torch.float32),
        torch.tensor(patch_valid),
    )


def parameter_count(manifest):
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(
        manifest["model"], revision=manifest["revision"], local_files_only=True
    )
    with torch.device("meta"):
        backbone = AutoModel.from_config(config)
    return sum(p.numel() for p in backbone.parameters())


def run(args):
    if args.steps < 1 or args.save_every < 1 or args.lr <= 0 or args.sigma <= 0:
        raise ValueError("Positive training settings required")
    manifest, names, meta, episodes, x, xy, label_valid, centers, patch_valid = prepare(
        args
    )
    head = PoseHeatmapProbe(x.shape[-1], len(names))
    trainable = sum(p.numel() for p in head.parameters())
    frozen = parameter_count(manifest)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if list(out.glob("fold-*")) or (out / "COMPLETE.json").exists():
        raise FileExistsError(
            "Preserve previous probe checkpoints; use a new experiment output"
        )
    checkpoint_count = len(episodes) * math.ceil(args.steps / args.save_every)
    reserve = checkpoint_count * (trainable * 12 + 65536)
    free = shutil.disk_usage(out).free
    if free < reserve + 100_000_000:
        raise RuntimeError("Insufficient immutable checkpoint reserve")
    config = {
        "model": manifest["model"],
        "revision": manifest["revision"],
        "precision": "float32",
        "device": args.device,
        "side": args.side,
        "frozen_parameters": frozen,
        "trainable_parameters": trainable,
        "total_parameters": frozen + trainable,
        "feature_dim": x.shape[-1],
        "patch_tokens": x.shape[1],
        "landmarks": len(names),
        "output_coordinates": len(names) * 2,
        "denoising_dim": None,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler": None,
        "steps_per_fold": args.steps,
        "seed": args.seed,
        "sigma_native_pixels": args.sigma,
        "save_every": args.save_every,
        "checkpoint_count": checkpoint_count,
        "checkpoint_reserve_bytes": reserve,
        "free_bytes": free,
        "initialization": "fresh linear head per fold; frozen cached official DINOv3",
        "input_contract": "image features only; per-token nonlearned LayerNorm; no poses, tracks or timestamps",
        "baseline": "training-only mean pose per landmark",
        "reference_mode": manifest.get("reference_mode", "visible_manual"),
        "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        "folds": [
            {
                "heldout_episode": e,
                "train_frames": [r["id"] for r in meta if r["episode"] != e],
                "test_frames": [r["id"] for r in meta if r["episode"] == e],
            }
            for e in episodes
        ],
    }
    config.update(
        {
            "optimizer": "AdamW",
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "batch_size_per_fold": [len(episode_split(meta, e)[0]) for e in episodes],
            "gradient_accumulation": 1,
            "runtime": {
                "torch": torch.__version__,
                "python": platform.python_version(),
            },
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "features": meta,
            "evaluation_cadence": "terminal only; no test-based model selection",
        }
    )
    (out / "PREFLIGHT.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2), flush=True)
    if args.preflight_only:
        return
    torch.set_num_threads(args.threads)
    x = x.to(args.device)
    xy = xy.to(args.device)
    label_valid = label_valid.to(args.device)
    centers = centers.to(args.device)
    patch_valid = patch_valid.to(args.device)
    target = heatmap_targets(xy, centers, patch_valid, args.sigma)
    predictions = []
    receipts = []
    for fold, e in enumerate(episodes):
        torch.manual_seed(args.seed)
        model = PoseHeatmapProbe(x.shape[-1], len(names)).float().to(args.device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        train_ids, test_ids = episode_split(meta, e)
        train = torch.tensor(train_ids, device=args.device)
        test = torch.tensor(test_ids, device=args.device)
        baseline = mean_pose_baseline(xy[train], label_valid[train])
        fold_dir = out / f"fold-{fold}"
        fold_dir.mkdir()
        (fold_dir / "split.json").write_text(
            json.dumps(config["folds"][fold], indent=2) + "\n"
        )
        with (fold_dir / "train.jsonl").open("w") as log:
            for step in range(1, args.steps + 1):
                optimizer.zero_grad(set_to_none=True)
                logits = model(x[train], patch_valid)
                loss = objective(logits, target[train], label_valid[train])
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite probe loss")
                loss.backward()
                if any(
                    p.grad is None or not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                ):
                    raise RuntimeError("Invalid probe gradient")
                optimizer.step()
                if step == 1 or step % 100 == 0:
                    row = {"step": step, "loss": float(loss.detach().cpu())}
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    print(f'fold={fold} step={step} loss={row["loss"]:.4f}', flush=True)
                if step % args.save_every == 0 or step == args.steps:
                    checkpoint = fold_dir / f"epoch_{step:04d}-step_{step:09d}.pt"
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "global_step": step,
                            "epoch": step,
                            "config": config,
                            "heldout_episode": e,
                            "torch_rng": torch.get_rng_state(),
                        },
                        checkpoint,
                    )
                    receipts.append(
                        {
                            "path": str(checkpoint.resolve()),
                            "global_step": step,
                            "epoch": step,
                            "bytes": checkpoint.stat().st_size,
                            "sha256": hashlib.sha256(
                                checkpoint.read_bytes()
                            ).hexdigest(),
                        }
                    )
        with torch.no_grad():
            logits = model(x, patch_valid)
            pred = centers[logits.argmax(-1)]
            shuffled = pred[test.roll(1)]
            for split, indices in [("train", train), ("test", test)]:
                for k, i in enumerate(indices.tolist()):
                    predictions.append(
                        {
                            "fold": fold,
                            "split": split,
                            "frame_id": meta[i]["id"],
                            "episode": meta[i]["episode"],
                            "landmarks": names,
                            "predicted_xy": pred[i].cpu().tolist(),
                            "target_xy": xy[i].cpu().tolist(),
                            "valid": label_valid[i].cpu().tolist(),
                            "baseline_xy": baseline.cpu().tolist(),
                            "shuffled_xy": shuffled[k].cpu().tolist()
                            if split == "test"
                            else None,
                        }
                    )
        # Verify strict reload of actual terminal head and optimizer state.
        saved = torch.load(checkpoint, map_location=args.device, weights_only=False)
        restored = PoseHeatmapProbe(x.shape[-1], len(names)).float().to(args.device)
        restored.load_state_dict(saved["model"], strict=True)
        restored_opt = torch.optim.AdamW(
            restored.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        restored_opt.load_state_dict(saved["optimizer"])
        with torch.no_grad():
            if not torch.equal(
                restored(x[test], patch_valid), model(x[test], patch_valid)
            ):
                raise RuntimeError("Strict reload output changed")
        (out / "predictions.json").write_text(json.dumps(predictions) + "\n")
    summary = {}
    for split in ["train", "test"]:
        rows = [v for v in predictions if v["split"] == split]
        metrics = {}
        for key in ["predicted_xy", "baseline_xy"] + (
            ["shuffled_xy"] if split == "test" else []
        ):
            errors = np.concatenate(
                [
                    np.linalg.norm(
                        np.array(v[key]) - np.array(v["target_xy"]), axis=-1
                    )[v["valid"]]
                    for v in rows
                ]
            )
            metrics[key] = {
                "n": len(errors),
                "mean_px": float(errors.mean()),
                "median_px": float(np.median(errors)),
                "pck16": float((errors <= 16).mean()),
            }
        summary[split] = metrics
    result = {
        "status": "COMPLETE",
        "scope": "Small leave-one-episode-out pilot; estimated projected references, unknown occlusion; not verified visible-joint or future-pose accuracy",
        "summary": summary,
        "config": config,
        "checkpoints": receipts,
        "strict_reload": "PASS all folds",
        "predictions_sha256": hashlib.sha256(
            (out / "predictions.json").read_bytes()
        ).hexdigest(),
    }
    (out / "COMPLETE.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--feature-results", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--side", type=int, default=768)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--sigma", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="mps")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--preflight-only", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
