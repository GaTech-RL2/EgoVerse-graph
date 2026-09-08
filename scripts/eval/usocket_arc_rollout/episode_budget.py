#!/usr/bin/env python3
"""Compute the data-derived rollout budget for a PushShapes corpus.

Implements the sim_v2 eval protocol's horizon revision 3:

    budget(dataset, level) = ceil(1.1 * p99(total_frames | that level's episodes))

p99, never max: one 2255-frame outlier in small_circle_3000_v2sub (second
longest 442) once inflated a whole embodiment's budget ~5x. `total_frames` is
read from the episode attrs, not the array length, because episode arrays are
chunk-padded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import zarr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--multiplier", type=float, default=1.1)
    # Eval resets with seed 0..N-1. If collection used the same seeds, those
    # rollouts replay TRAINING initial states and the score is a train score.
    ap.add_argument("--eval-seed-count", type=int, default=40)
    args = ap.parse_args()

    episodes = sorted(args.dataset.glob("episode_*.zarr"))
    if not episodes:
        raise FileNotFoundError(f"no episode_*.zarr under {args.dataset}")

    by_level: dict[int, list[int]] = {}
    names = []
    reset_seeds: list[int] = []
    action_space = None
    for path in episodes:
        attrs = dict(zarr.open_group(str(path), mode="r").attrs)
        frames = int(attrs.get("total_frames", 0))
        if frames <= 0:
            raise ValueError(f"{path.name}: missing/invalid total_frames")
        env_args = json.loads(attrs["task_description"])["env_args"]
        level = int(env_args.get("obstacle_level", 0))
        by_level.setdefault(level, []).append(frames)
        names.append(path.name)
        try:
            seed = json.loads(attrs["episode_init"]).get("reset_seed")
            if seed is not None:
                reset_seeds.append(int(seed))
        except (KeyError, ValueError, TypeError):
            pass
        action_space = attrs.get("action_space", action_space)

    rows = {}
    for level, frames in sorted(by_level.items()):
        arr = np.asarray(frames, dtype=np.float64)
        p99 = float(np.percentile(arr, 99))
        p95 = float(np.percentile(arr, 95))
        rows[str(level)] = {
            "n": int(arr.size),
            "max": int(arr.max()),
            "p95": round(p95, 3),
            "p99": round(p99, 3),
            "budget": int(math.ceil(args.multiplier * p99)),
            # >~1.5 means one episode drives the level; diagnostic under p99.
            "max_over_p95": round(float(arr.max() / p95), 4) if p95 > 0 else None,
        }

    eval_seeds = set(range(args.eval_seed_count))
    collected = set(reset_seeds)
    overlap = sorted(eval_seeds & collected)
    seed_report = {
        "episodes_with_reset_seed": len(reset_seeds),
        "distinct_reset_seeds": len(collected),
        "reset_seed_min": min(collected) if collected else None,
        "reset_seed_max": max(collected) if collected else None,
        "eval_seed_range": [0, args.eval_seed_count - 1],
        "eval_seed_overlap_count": len(overlap),
        "eval_seed_overlap": overlap[:40],
        "verdict": (
            "CONTAMINATED: eval seeds reproduce training initial states"
            if overlap
            else "clean: no eval seed appears as a collected reset_seed"
        ),
    }

    digest = hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()
    payload = {
        "dataset": args.dataset.name,
        "statistic": "p99",
        "multiplier": args.multiplier,
        "formula": "ceil(multiplier * p99(total_frames | level))",
        "action_space": action_space or "cursor",
        "by_level": rows,
        "episode_count": len(names),
        "eval_seed_check": seed_report,
        "content_sha256": digest,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
