"""Native MoT clean-input pose-readout component diagnostic, not full training.

Real image features and measured hand coordinates; no invented scene tracks.
Oracle future images isolate readout quality. Whole-episode holdout is mandatory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from egomimic.models.world_model import MODALITIES, MoTWorldModel
from egomimic.scripts.embedding_process.dinov3_correspondence import (
    cached_features,
    patch_geometry,
)
from egomimic.utils.pose_utils import ee_pose_to_cam_frame, wxyz_to_xyzw


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_inputs(reference_path, feature_results, model, revision, side, output):
    """Shared generator/readout input contract; no training and no DINO rerun."""
    references = json.loads(Path(reference_path).read_text())
    records = sorted(
        references["records"], key=lambda x: (x["episode"], x["obs_rgb_timestamps_ns"])
    )
    cache = cached_features(feature_results, model, revision)
    features, world, transforms, metadata = [], [], [], []
    xy = valid = None
    for r in records:
        if sha(r["path"]) != r["sha256"] or not 0 <= r["frame"] < r["total_frames"]:
            raise ValueError("Original frame hash or logical frame bound mismatch")
        feature, prep, entry = cache[side, r["sha256"]]
        centers, mask = patch_geometry(prep)
        h, w = prep["frames"][0]["original_hw"]
        normalized = (centers + 0.5) / [w, h] * 2 - 1
        if xy is not None and (
            not np.allclose(xy, normalized) or not np.array_equal(valid, mask)
        ):
            raise ValueError("Component probe requires matching camera geometry")
        xy, valid = normalized.astype(np.float32), mask
        pose = np.concatenate(
            [r["left.obs_keypoints"], r["right.obs_keypoints"]]
        ).reshape(42, 3)
        if not np.isfinite(pose).all() or np.abs(pose).max() > 100:
            raise ValueError("Invalid measured world XYZ")
        head = np.asarray(r["obs_head_pose"])
        if not np.isfinite(head).all() or abs(np.linalg.norm(head[3:]) - 1) > 0.01:
            raise ValueError("Invalid reference camera pose")
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_quat(wxyz_to_xyzw(head[3:])).as_matrix()
        transform[:3, 3] = head[:3]
        features.append(feature)
        world.append(pose)
        transforms.append(transform)
        metadata.append(
            {
                "id": Path(r["path"]).stem,
                "episode": r["episode"],
                "frame": r["frame"],
                "timestamp_ns": r["obs_rgb_timestamps_ns"],
                "image_sha256": r["sha256"],
                "feature": entry,
            }
        )
    pairs, pair_pose, dt = [], [], []
    for a, b in zip(range(len(records) - 1), range(1, len(records))):
        if records[a]["episode"] != records[b]["episode"]:
            continue
        delta = (
            records[b]["obs_rgb_timestamps_ns"] - records[a]["obs_rgb_timestamps_ns"]
        ) * 1e-9
        if not np.isfinite(delta) or delta <= 0:
            raise ValueError("Physical timestamps must strictly increase")
        pairs.append([a, b])
        dt.append(delta)
        pair_pose.append(
            np.stack(
                [
                    ee_pose_to_cam_frame(world[i], transforms[a]).reshape(-1)
                    for i in [a, b]
                ]
            )
        )
    episodes = sorted({r["episode"] for r in metadata})
    if len(episodes) < 3:
        raise ValueError("At least three independent episodes required")
    for e in episodes:
        train_sha = {m["image_sha256"] for m in metadata if m["episode"] != e}
        test_sha = {m["image_sha256"] for m in metadata if m["episode"] == e}
        if train_sha & test_sha:
            raise ValueError("Image bytes overlap across heldout episodes")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "INPUTS.npz").exists():
        raise FileExistsError("Preserve prepared component inputs")
    np.savez_compressed(
        out / "INPUTS.npz",
        features=np.stack(features),
        patch_xy=xy,
        patch_valid=valid,
        pairs=np.asarray(pairs, np.int64),
        poses=np.asarray(pair_pose, np.float32),
        dt=np.asarray(dt, np.float32),
        reference_world_from_camera=np.asarray(transforms),
        timestamps_ns=np.asarray([m["timestamp_ns"] for m in metadata], np.int64),
    )
    info = {
        "frames": metadata,
        "episodes": episodes,
        "pose_schema": "42 recorded raw-Aria-supported world XYZ points; not canonical MANO;126 coordinates",
        "pose_reference": "Both poses in each pair transformed with inverse first-frame world_from_camera; future camera pose is not model input",
        "scene_tracks": "unavailable; empty track stream, never hand-joints-as-tracks",
        "model": model,
        "revision": revision,
        "side": side,
        "precision": "float32",
        "source_reference_sha256": sha(reference_path),
        "input_sha256": sha(out / "INPUTS.npz"),
    }
    (out / "INPUTS.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def load_inputs(directory):
    directory = Path(directory)
    meta = json.loads((directory / "INPUTS.json").read_text())
    if sha(directory / "INPUTS.npz") != meta["input_sha256"]:
        raise ValueError("Prepared component input SHA mismatch")
    with np.load(directory / "INPUTS.npz", allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    return data, meta


def fold_data(data, meta, heldout):
    train_frames = np.asarray(
        [i for i, r in enumerate(meta["frames"]) if r["episode"] != heldout]
    )
    train = np.asarray(
        [
            i
            for i, (a, b) in enumerate(data["pairs"])
            if meta["frames"][a]["episode"] != heldout
        ]
    )
    test = np.asarray(
        [
            i
            for i, (a, b) in enumerate(data["pairs"])
            if meta["frames"][a]["episode"] == heldout
        ]
    )
    if len(train) == 0 or len(test) == 0:
        raise ValueError("Nonempty train and heldout pairs required")
    f = data["features"][train_frames][:, data["patch_valid"]].reshape(
        -1, data["features"].shape[-1]
    )
    p = data["poses"][train].reshape(-1, data["poses"].shape[-1])
    norm = {
        "image_mean": f.mean(0),
        "image_std": f.std(0).clip(1e-5),
        "pose_mean": p.mean(0),
        "pose_std": p.std(0).clip(0.05),
    }
    return train, test, norm


def model_for_probe(width=128, depth=2):
    model = MoTWorldModel(
        {"human": 126}, width=width, depth=depth, heads=4, image_adapter_dim=16
    ).float()
    # These outputs/track-private parameters are not used by a pose-only loss.
    track_index = MODALITIES.index("track")
    unused = [
        model.image_out,
        model.track_in,
        model.track_out,
        model.point,
        model.norm[track_index],
    ]
    for block in model.blocks:
        unused.extend(
            [
                block.norm1[track_index],
                block.qkv[track_index],
                block.out[track_index],
                block.norm2[track_index],
                block.ffn[track_index],
            ]
        )
    for module in unused:
        module.requires_grad_(False)
    return model


def make_batch(data, norm, indices, device="cpu", visual=True):
    indices = np.asarray(indices)
    pairs = data["pairs"][indices]
    b = len(indices)
    n = data["features"].shape[1]
    features = (data["features"][pairs] - norm["image_mean"]) / norm["image_std"]
    pose = (data["poses"][indices] - norm["pose_mean"]) / norm["pose_std"]
    target = torch.tensor(pose[:, 1], device=device)
    pose[:, 1] = 0  # Query targets never enter the forward batch.
    times = np.stack([np.zeros(b), data["dt"][indices]], axis=1).astype(np.float32)

    def tensor(x):
        return torch.as_tensor(x, device=device)

    image_valid = np.tile(data["patch_valid"], (b, 2)) & visual
    image = {
        "value": tensor(features.reshape(b, 2 * n, -1)),
        "time": tensor(np.repeat(times, n, axis=1)),
        "valid": tensor(image_valid),
        "observed": tensor(image_valid.copy()),
        "xy": tensor(np.tile(data["patch_xy"], (b, 2, 1))),
        "camera": torch.zeros(b, 2 * n, dtype=torch.long, device=device),
    }
    empty = {
        "value": torch.zeros(b, 0, 3, device=device),
        "time": torch.zeros(b, 0, device=device),
        "valid": torch.zeros(b, 0, dtype=torch.bool, device=device),
        "observed": torch.zeros(b, 0, dtype=torch.bool, device=device),
        "xy": torch.zeros(b, 0, 2, device=device),
        "camera": torch.zeros(b, 0, dtype=torch.long, device=device),
        "point_id": torch.zeros(b, 0, dtype=torch.long, device=device),
    }
    proprio = {
        "value": tensor(pose),
        "time": tensor(times),
        "valid": torch.ones(b, 2, dtype=torch.bool, device=device),
        "observed": tensor(np.tile([True, False], (b, 1))),
    }
    batch = {"image": image, "track": empty, "proprio": proprio}
    return batch, target


def predict(model, batch):
    return model(
        batch,
        "human",
        {k: batch[k]["value"] for k in ("image", "track")},
        {k: torch.zeros_like(batch[k]["time"]) for k in ("image", "track")},
    )["proprio"][:, 1]


def evaluate(model, data, meta, norm, indices, device, visual):
    rows = []
    model.eval()
    with torch.no_grad():
        for i in indices:
            batch, _ = make_batch(data, norm, [i], device, visual)
            estimate = (
                predict(model, batch).cpu().numpy()[0] * norm["pose_std"]
                + norm["pose_mean"]
            )
            truth = data["poses"][i, 1]
            hold = data["poses"][i, 0]
            a, b = data["pairs"][i]
            rows.append(
                {
                    "source": meta["frames"][a]["id"],
                    "target": meta["frames"][b]["id"],
                    "dt_seconds": float(data["dt"][i]),
                    "prediction_xyz": estimate.reshape(42, 3).tolist(),
                    "target_xyz": truth.reshape(42, 3).tolist(),
                    "mpjpe_m": float(
                        np.linalg.norm(
                            (estimate - truth).reshape(42, 3), axis=-1
                        ).mean()
                    ),
                    "hold_pose_mpjpe_m": float(
                        np.linalg.norm((hold - truth).reshape(42, 3), axis=-1).mean()
                    ),
                }
            )
    return rows


def run(args):
    torch.set_num_threads(args.threads)
    data, meta = load_inputs(args.inputs)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if list(out.glob("fold-*")):
        raise FileExistsError("Preserve existing diagnostic checkpoints")
    model = model_for_probe(args.width, args.depth)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    count = len(meta["episodes"]) * 2 * math.ceil(args.steps / args.save_every)
    reserve = count * (total * 4 + trainable * 8 + 65536)
    free = shutil.disk_usage(out).free
    if free < reserve + 100_000_000:
        raise RuntimeError("Insufficient immutable checkpoint reserve")
    config = {
        "input_sha256": meta["input_sha256"],
        "width": args.width,
        "depth": args.depth,
        "heads": 4,
        "image_adapter_dim": 16,
        "parameters_total": total,
        "parameters_requires_grad": trainable,
        "denoising_dim": None,
        "hidden_dim": args.width,
        "output_pose_dim": 126,
        "action_dim": None,
        "input_image_dim": data["features"].shape[-1],
        "image_tokens_per_sequence": 2 * data["features"].shape[1],
        "pose_tokens": 2,
        "track_tokens": 0,
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": 0.01,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "scheduler": None,
        "batch": 2,
        "steps_per_condition_per_fold": args.steps,
        "conditions": ["observed_pose_only", "oracle_images_and_observed_pose"],
        "seed": 42,
        "precision": "FP32",
        "device": args.device,
        "cpu_threads": args.threads,
        "save_every": args.save_every,
        "checkpoint_count": count,
        "checkpoint_reserve_bytes": reserve,
        "free_bytes": free,
        "folds": meta["episodes"],
        "source_sha256": sha(__file__),
        "evaluation": "terminal only, all heldout pairs; no test-based selection",
        "full_model_training": False,
    }
    (out / "PREFLIGHT.json").write_text(json.dumps(config, indent=2) + "\n")
    # Empty modality streams can pass MPS forward but fail its backward kernel.
    # Probe the exact device, batch shape and both conditions before any update.
    guard = {}
    train, _, norm = fold_data(data, meta, meta["episodes"][0])
    for visual in (False, True):
        torch.manual_seed(42)
        probe = model_for_probe(args.width, args.depth).to(args.device)
        batch, target = make_batch(data, norm, train[:2], args.device, visual)
        loss = (predict(probe, batch) - target).square().mean()
        loss.backward()
        if not torch.isfinite(loss) or not all(
            p.grad is None or torch.isfinite(p.grad).all() for p in probe.parameters()
        ):
            raise RuntimeError("Exact-runtime backward preflight failed")
        guard[str(visual)] = {"loss": float(loss.detach().cpu()), "finite": True}
    (out / "RUNTIME_BACKWARD_GUARD.json").write_text(
        json.dumps(
            {"device": args.device, "conditions": guard, "optimizer_updates": 0},
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(config, indent=2), flush=True)
    if args.preflight_only:
        return
    receipts = []
    all_rows = []
    for fold, heldout in enumerate(meta["episodes"]):
        train, test, norm = fold_data(data, meta, heldout)
        for visual in (False, True):
            condition = (
                "oracle_images_and_observed_pose" if visual else "observed_pose_only"
            )
            directory = out / f"fold-{fold}" / condition
            directory.mkdir(parents=True)
            torch.manual_seed(42)
            model = model_for_probe(args.width, args.depth).to(args.device)
            initial_sha = hashlib.sha256(
                b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())
            ).hexdigest()
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.lr,
                weight_decay=0.01,
            )
            rng = np.random.default_rng(42)
            order = np.asarray([], dtype=np.int64)
            epoch = 0
            orders = []
            split = {
                "heldout_episode": heldout,
                "train_pair_indices": train.tolist(),
                "test_pair_indices": test.tolist(),
                "normalization": {k: v.tolist() for k, v in norm.items()},
            }
            (directory / "split.json").write_text(json.dumps(split, indent=2) + "\n")
            with (directory / "train.jsonl").open("w") as log:
                for step in range(1, args.steps + 1):
                    if len(order) == 0:
                        order = rng.permutation(train)
                        epoch += 1
                    ids, order = order[:2], order[2:]
                    orders.append(ids.tolist())
                    batch, target = make_batch(data, norm, ids, args.device, visual)
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    loss = (predict(model, batch) - target).square().mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite pose loss")
                    loss.backward()
                    active = (
                        sum(
                            p.numel()
                            for p in model.parameters()
                            if p.grad is not None and torch.count_nonzero(p.grad) > 0
                        )
                        if step == 1
                        else None
                    )
                    if not torch.stack(
                        [
                            torch.isfinite(p.grad).all()
                            for p in model.parameters()
                            if p.grad is not None
                        ]
                    ).all():
                        raise RuntimeError("Nonfinite pose gradient")
                    optimizer.step()
                    if step == 1 or step % 25 == 0:
                        row = {
                            "step": step,
                            "epoch": epoch,
                            "loss": float(loss.detach().cpu()),
                            "nonzero_gradient_parameters": active,
                        }
                        log.write(json.dumps(row) + "\n")
                        log.flush()
                        print(f"fold{fold} {condition} {row}", flush=True)
                    if step % args.save_every == 0 or step == args.steps:
                        path = directory / f"epoch_{epoch:04d}-step_{step:09d}.pt"
                        torch.save(
                            {
                                "model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "step": step,
                                "epoch": epoch,
                                "normalization": norm,
                                "config": config,
                                "condition": condition,
                                "heldout_episode": heldout,
                            },
                            path,
                        )
                        receipts.append(
                            {
                                "path": str(path.resolve()),
                                "sha256": sha(path),
                                "bytes": path.stat().st_size,
                            }
                        )
            before = evaluate(model, data, meta, norm, test, args.device, visual)
            (directory / "RUN_IDENTITY.json").write_text(
                json.dumps(
                    {
                        "initial_state_sha256": initial_sha,
                        "sample_order_sha256": hashlib.sha256(
                            json.dumps(orders).encode()
                        ).hexdigest(),
                        "sample_order": orders,
                    },
                    indent=2,
                )
                + "\n"
            )
            clone = model_for_probe(args.width, args.depth).to(args.device)
            loaded = torch.load(path, map_location=args.device, weights_only=False)
            clone.load_state_dict(loaded["model"], strict=True)
            restored_optimizer = torch.optim.AdamW(
                [p for p in clone.parameters() if p.requires_grad],
                lr=args.lr,
                weight_decay=0.01,
            )
            restored_optimizer.load_state_dict(loaded["optimizer"])
            if len(restored_optimizer.state) != len(optimizer.state):
                raise RuntimeError("Optimizer reload state count differs")
            for original, restored in zip(
                optimizer.state.values(), restored_optimizer.state.values()
            ):
                for key, value in original.items():
                    if torch.is_tensor(value) and not torch.equal(
                        value.to("cpu"), restored[key].to("cpu")
                    ):
                        raise RuntimeError("Optimizer reload tensor differs")
            after = evaluate(clone, data, meta, norm, test, args.device, visual)
            if before != after:
                raise RuntimeError("Strict reload predictions differ")
            for row in before:
                row.update(fold=fold, condition=condition, split="test")
            all_rows.extend(before)
            train_rows = evaluate(model, data, meta, norm, train, args.device, visual)
            for row in train_rows:
                row.update(fold=fold, condition=condition, split="train")
            all_rows.extend(train_rows)
            (directory / "predictions.json").write_text(
                json.dumps(before + train_rows, indent=2) + "\n"
            )
    summary = {}
    for condition in config["conditions"]:
        summary[condition] = {
            split: {
                key: float(
                    np.mean(
                        [
                            r[key]
                            for r in all_rows
                            if r["condition"] == condition and r["split"] == split
                        ]
                    )
                )
                for key in ["mpjpe_m", "hold_pose_mpjpe_m"]
            }
            for split in ["train", "test"]
        }
    complete = {
        "status": "COMPLETE",
        "config": config,
        "summary": summary,
        "checkpoints": receipts,
        "strict_reload": "PASS",
        "limits": "Small3episode real recorded-coordinate readout diagnostic, not full model, not scene-track test, not visiblehandtruth; clean future images are oracle context.",
    }
    (out / "COMPLETE.json").write_text(json.dumps(complete, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="mps")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args()
    if args.steps < 1 or args.save_every < 1 or args.lr <= 0 or args.threads < 1:
        p.error("Positive training settings required")
    run(args)


if __name__ == "__main__":
    main()
