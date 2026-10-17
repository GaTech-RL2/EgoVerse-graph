"""Synthetic FP32 forward/backward memory probe; never a real-data launch gate.

Run in a scheduled GPU allocation. No optimizer updates, dataset access, or
checkpoints. Pose widths are explicit diagnostic inputs, not schema approval.
"""
import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path

import torch

from egomimic.models.world_model import MoTWorldModel


def synthetic_batch(frames, cameras, patches, tracks, pose_dim, device):
    result = {}
    for name, count, dim in (
        ("image", cameras * patches, 768), ("track", tracks, 3),
        ("proprio", 1, pose_dim),
    ):
        clock = torch.linspace(-3, 3, frames, device=device).repeat_interleave(count)[None]
        n = frames * count
        result[name] = dict(
            value=torch.randn(1, n, dim, device=device), time=clock,
            valid=torch.ones(1, n, dtype=torch.bool, device=device),
            target_valid=torch.ones(1, n, dtype=torch.bool, device=device),
            observed=clock < 0,
        )
        if name != "proprio":
            result[name].update(
                xy=torch.rand(1, n, 2, device=device),
                camera=(torch.arange(cameras, device=device).repeat_interleave(patches)
                        .repeat(frames)[None] if name == "image"
                        else torch.zeros(1, n, dtype=torch.long, device=device)),
            )
        if name == "track":
            result[name].update(
                point_id=torch.arange(tracks, device=device).repeat(frames)[None],
                confidence=torch.ones(1, n, device=device),
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--cameras", type=int, required=True)
    parser.add_argument("--side", type=int, required=True)
    parser.add_argument("--tracks", type=int, required=True)
    parser.add_argument("--pose-dims", type=json.loads, required=True)
    parser.add_argument("--embodiment", required=True)
    parser.add_argument("--query-chunk", type=int, default=512)
    parser.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attention-backend", choices=["sdpa", "efficient", "flash4"], default="sdpa")
    parser.add_argument("--attention-dtype", choices=["native", "bfloat16"], default="native")
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.frames) < 2 or min(args.cameras, args.tracks, args.side) < 1 or args.side % 16:
        parser.error("Need >=2 frames, positive counts and side divisible by16")
    if args.embodiment not in args.pose_dims:
        parser.error("Active embodiment missing from diagnostic pose widths")
    if args.device == "cuda" and torch.cuda.device_count() != 1:
        parser.error("Require exactly one visible allocated GPU")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cuda = args.device == "cuda"
    report = {
        "scope": "Synthetic memory/backward diagnostic; not actual data, optimizer or reload proof",
        "torch": torch.__version__, "device": torch.cuda.get_device_name(0) if cuda else "cpu",
        "precision": f"FP32 parameters/model/loss; attention={args.attention_dtype}; autocast/TF32 off", "results": [],
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model_sha256": hashlib.sha256(Path(inspect.getfile(MoTWorldModel)).read_bytes()).hexdigest(),
    }
    for frames in args.frames:
        torch.manual_seed(42)
        patches = (args.side // 16) ** 2
        tokens = frames * (args.cameras * patches + args.tracks + 1)
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        row = {"frames": frames, "tokens": tokens, "status": "RUNNING"}
        report["results"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        try:
            model = MoTWorldModel(
                args.pose_dims, width=args.width, depth=args.depth, heads=args.heads,
                max_cameras=args.cameras, max_tracks=args.tracks, max_tokens=tokens,
                attention_query_chunk_size=args.query_chunk, activation_checkpointing=args.activation_checkpointing,
                attention_backend=args.attention_backend, attention_dtype=args.attention_dtype,
            ).float().to(args.device)
            batch = synthetic_batch(frames, args.cameras, patches, args.tracks,
                                    args.pose_dims[args.embodiment], args.device)
            loss = model.training_losses(batch, args.embodiment)
            loss["total"].backward()
            if cuda:
                torch.cuda.synchronize()
            if not all(torch.isfinite(v).all() for v in loss.values()):
                raise ValueError("Nonfinite loss")
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):
                raise ValueError("Nonfinite or absent gradients")
            row.update(status="PASS", parameters=sum(p.numel() for p in model.parameters()),
                       loss={k: float(v.detach()) for k, v in loss.items()},
                       elapsed_seconds=time.perf_counter()-started,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated() if cuda else None,
                       peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda else None)
            del model, batch, loss, grads
        except Exception as error:
            row.update(status="FAILED", error=str(error), elapsed_seconds=time.perf_counter()-started)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            raise
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
