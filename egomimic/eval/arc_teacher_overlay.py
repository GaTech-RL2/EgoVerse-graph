"""Immutable teacher-forced action overlay for Pipeline planar arc checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from egomimic.eval.core.ckpt_loading import _load_rollout_graph
from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_uint8(value: torch.Tensor) -> np.ndarray:
    image = value.detach().cpu().numpy()
    while image.ndim > 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3:
        raise RuntimeError(f"front image must have 3 dimensions, got {image.shape}")
    if image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise RuntimeError(f"front image must have three channels, got {image.shape}")
    if image.max(initial=0) <= 1.5:
        image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def _model_batch(sample: dict, device: torch.device) -> dict:
    image = sample["front_img_1"].float()
    if image.ndim == 3 and image.shape[-1] == 3:
        image = image.permute(2, 0, 1)
    if image.max().item() > 1.5:
        image = image / 255.0
    state = sample["state_agent_obj"].float()
    if state.ndim == 2 and state.shape[0] == 1:
        state = state[0]
    return {
        "front_img_1": image.unsqueeze(0).to(device),
        "state_agent_obj": state.unsqueeze(0).to(device),
    }


def _draw_path(frame: np.ndarray, xy: np.ndarray, color, alpha: float) -> np.ndarray:
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    # PushShapes cursor coordinates live in the native 512 x 512 workspace.
    points = np.rint(xy * np.array([width / 512.0, height / 512.0])).astype(int)
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    if len(points) > 1:
        cv2.polylines(canvas, [points.reshape(-1, 1, 2)], False, color, 1, cv2.LINE_AA)
    # At 96x96, the former 4 px start radius obscured the pusher and target.
    # Mark only the first waypoint with a one-pixel-radius dot; the polyline
    # communicates the remaining ordered waypoints without a chain of blobs.
    if len(points):
        cv2.circle(canvas, tuple(points[0]), 1, color, -1, cv2.LINE_AA)
    return cv2.addWeighted(canvas, float(alpha), frame, 1.0 - float(alpha), 0)


def render_prediction_artifact(
    artifact_path: Path,
    video_path: Path,
    first_frame_path: Path,
    *,
    fps: float = 10.0,
    gt_alpha: float = 0.6,
    pred_alpha: float = 1.0,
) -> int:
    """Render paired GT/prediction paths without running stochastic inference."""
    artifact_path = Path(artifact_path).resolve(strict=True)
    video_path = Path(video_path).resolve()
    first_frame_path = Path(first_frame_path).resolve()
    for path in (video_path, first_frame_path):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite immutable render {path}")
    data = np.load(artifact_path)
    images = data["images"]
    targets = data["target_tokens"]
    predictions = data["predicted_tokens"]
    indices = data["frame_indices"]
    if images.shape[0] == 0 or targets.shape != predictions.shape or targets.shape[1:] != (17, 5):
        raise RuntimeError(
            f"invalid arc prediction artifact: images={images.shape}, "
            f"target={targets.shape}, prediction={predictions.shape}"
        )
    if len(images) != len(targets) or len(indices) != len(targets):
        raise RuntimeError("artifact image, token, and frame counts differ")
    rendered = []
    for image, target, pred, index in zip(images, targets, predictions, indices):
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        frame = _draw_path(frame, target[:16, :2], (0, 220, 0), gt_alpha)
        frame = _draw_path(frame, pred[:16, :2], (0, 0, 255), pred_alpha)
        cv2.putText(frame, f"GT green | Pred red | frame {int(index)}", (4, 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
        rendered.append(frame)
    height, width = rendered[0].shape[:2]
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open MP4 writer")
    for frame in rendered:
        writer.write(frame)
    writer.release()
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise RuntimeError("overlay video was not written")
    cv2.imwrite(str(first_frame_path), rendered[0])
    return len(rendered)


def _validate_split(config, split_path: Path, episode_id: str):
    payload = json.loads(split_path.read_text())
    domain = "pushshapes_sim_u_socket"
    split = payload["domains"][domain]
    valid_ids = list(split["valid_ids"])
    if payload.get("status") != "PASS" or int(payload.get("split_seed")) != 42:
        raise RuntimeError("validation split manifest is not the approved seed-42 PASS split")
    if len(valid_ids) != int(split["valid_count"]):
        raise RuntimeError("validation split count mismatch")
    names_hash = hashlib.sha256("\n".join(sorted(valid_ids)).encode()).hexdigest()
    # The training split hash uses newline-terminated canonical names.
    names_hash_nl = hashlib.sha256(("\n".join(sorted(valid_ids)) + "\n").encode()).hexdigest()
    if split["valid_names_sha256"] not in {names_hash, names_hash_nl}:
        raise RuntimeError("validation split names SHA mismatch")
    if episode_id not in valid_ids:
        raise RuntimeError(f"episode {episode_id!r} is not in the frozen validation split")
    cfg_valid = config.data.valid_datasets[domain]
    if int(cfg_valid.split_seed) != 42 or int(cfg_valid.expected_valid_episode_count) != len(valid_ids):
        raise RuntimeError("resolved config does not bind the frozen validation split")
    if str(cfg_valid.expected_valid_episode_names_sha256) != str(split["valid_names_sha256"]):
        raise RuntimeError("resolved config validation hash differs from split manifest")
    return split, cfg_valid


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=320)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=420042)
    parser.add_argument("--embodiment-name", default="pushshapes_sim_u_socket")
    parser.add_argument("--embodiment-id", type=int, required=True)
    parser.add_argument("--gt-alpha", type=float, default=0.6)
    parser.add_argument("--pred-alpha", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.frame_stride <= 0 or args.max_frames <= 0 or args.fps <= 0:
        raise RuntimeError("frame stride, max frames, and FPS must be positive")

    ckpt = Path(args.ckpt).resolve(strict=True)
    config_path = Path(args.config_path).resolve(strict=True)
    split_path = Path(args.split_manifest).resolve(strict=True)
    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifact_path = output / "predictions_arc17x5.npz"
    video_path = output / "teacher_forced_arc_overlay.mp4"
    first_frame_path = output / "first_frame.png"
    manifest_path = output / "overlay_manifest.json"
    for path in (artifact_path, video_path, first_frame_path, manifest_path):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite immutable output {path}")

    config = OmegaConf.load(config_path)
    split, valid_cfg = _validate_split(config, split_path, args.episode_id)
    dataset_root = Path(str(valid_cfg.resolver.folder_path)).resolve(strict=True)
    episode_path = (dataset_root / f"{args.episode_id}.zarr").resolve(strict=True)
    key_map = instantiate(valid_cfg.resolver.key_map)
    transforms = instantiate(valid_cfg.resolver.transform_list)
    dataset = ZarrDataset(episode_path, key_map, transforms)
    configured_override = str(valid_cfg.resolver.embodiment_override)
    if configured_override != args.embodiment_name:
        raise RuntimeError(
            f"resolver embodiment override {configured_override!r} != {args.embodiment_name!r}"
        )

    graph, checkpoint = _load_rollout_graph(
        str(ckpt), str(config_path), args.embodiment_name, args.embodiment_id, False
    )
    graph.algo.nets.eval()
    indices = list(range(0, len(dataset), args.frame_stride))[: args.max_frames]
    if not indices:
        raise RuntimeError("selected validation episode has no frames")

    images, targets, predictions = [], [], []
    for ordinal, index in enumerate(indices):
        sample = dataset[index]
        target = sample["actions"].detach().float().cpu().numpy()
        if target.shape != (17, 5):
            raise RuntimeError(f"arc target must be 17x5, got {target.shape}")
        devices = [graph.device.index or 0] if graph.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            frame_seed = int(args.seed) + ordinal
            torch.manual_seed(frame_seed)
            if graph.device.type == "cuda":
                torch.cuda.manual_seed_all(frame_seed)
            pred = graph.predict_tokens(_model_batch(sample, graph.device))[0]
        pred = pred.detach().float().cpu().numpy()
        if pred.shape != (17, 5) or not np.isfinite(pred).all():
            raise RuntimeError(f"prediction is not finite 17x5: {pred.shape}")
        images.append(_image_uint8(sample["front_img_1"]))
        targets.append(target)
        predictions.append(pred)

    images = np.stack(images)
    targets = np.stack(targets)
    predictions = np.stack(predictions)
    np.savez_compressed(
        artifact_path,
        images=images,
        target_tokens=targets,
        predicted_tokens=predictions,
        frame_indices=np.asarray(indices, dtype=np.int64),
        episode_id=np.asarray(args.episode_id),
        rng_seed=np.asarray(args.seed, dtype=np.int64),
    )

    render_prediction_artifact(
        artifact_path, video_path, first_frame_path,
        fps=args.fps, gt_alpha=args.gt_alpha, pred_alpha=args.pred_alpha,
    )

    token_mse = float(np.mean(np.square(predictions - targets)))
    xy_rmse = float(np.sqrt(np.mean(np.square(predictions[:, :16, :2] - targets[:, :16, :2]))))
    first_xy_rmse = float(np.sqrt(np.mean(np.square(predictions[:, 0, :2] - targets[:, 0, :2]))))
    metrics = {"token_mse_physical": token_mse, "waypoint_xy_rmse_px": xy_rmse,
               "first_waypoint_xy_rmse_px": first_xy_rmse}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError(f"non-finite overlay metrics: {metrics}")
    manifest = {
        "status": "PASS",
        "kind": "teacher_forced_gt_vs_prediction_arc_token_overlay",
        "not_a_policy_score": True,
        "checkpoint": str(ckpt), "checkpoint_sha256": _sha256(ckpt),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_global_step": int(checkpoint["global_step"]),
        "config": str(config_path), "config_sha256": _sha256(config_path),
        "split_manifest": str(split_path), "split_manifest_sha256": _sha256(split_path),
        "split_seed": 42, "valid_count": int(split["valid_count"]),
        "episode_id": args.episode_id, "episode_path": str(episode_path),
        "episode_stored_embodiment": str(dataset.embodiment),
        "configured_embodiment_override": configured_override,
        "episode_total_frames": len(dataset), "frame_indices": indices,
        "frame_stride": args.frame_stride, "rng_seed_base": args.seed,
        "weights": "raw", "precision": "float32",
        "inference_graph": graph.stage_names,
        "token_shape": [17, 5], "drawn_waypoints": 16,
        "timing_row_drawn": False, "native_workspace_size": [512, 512],
        "normalization_path": str(graph.norm_path), "normalization_mode": graph.norm_mode,
        "prediction_artifact": str(artifact_path),
        "prediction_artifact_sha256": _sha256(artifact_path),
        "video": str(video_path), "video_sha256": _sha256(video_path),
        "first_frame": str(first_frame_path), "metrics": metrics,
        "simulator": "not_used_teacher_forced", "rollout_count": 0,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
