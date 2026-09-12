"""Compute the per-level rollout budget for a dataset, and cache it as JSON.

    budget(dataset, level) = ceil(multiplier * stat(total_frames | level))

with stat = max by default: the budget is set by the SLOWEST demonstration at
that level, plus headroom. Evaluating with this budget makes "success" mean
"solved it in roughly the time the demonstrations took", instead of "solved it
within an arbitrary fixed horizon".

The budget travels with the DATASET, so a slow-pusher variant (0.25x, ~4x longer
episodes) automatically gets ~4x the budget and is not truncated -- which a flat
horizon cannot do.

Usage:
  python episode_budget.py <dataset_dir> [--multiplier 1.1] [--statistic max]
                           [--out scripts/eval/budgets/<name>.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import zarr

def _attrs(p) -> dict:
    """Read a store's attributes straight from zarr.json.

    These are Zarr V3 stores; the zarr 2.x available on the cluster raises
    GroupNotFoundError on them. Only scalar attributes are needed here, and
    zarr.json is plain JSON, so read it directly and stay version-agnostic.
    """
    import json as _json
    try:
        return _json.load(open(p / "zarr.json")).get("attributes", {}) or {}
    except Exception:
        return {}


DEFAULT_BUDGET_DIR = Path("/coc/flash7/paphiwetsa3/scripts/eval/budgets")


def detect_action_space(dataset: Path) -> str:
    """What convention does this dataset's `actions` follow?

    Datasets built by respeed_dataset.py stamp `action_space`; anything collected
    before that field existed is cursor-target data. A policy trained on one
    convention is not comparable to a policy trained on the other, so the eval
    has to record which it is.
    """
    for p in sorted(dataset.iterdir()):
        if not p.name.endswith(".zarr"):
            continue
        a = _attrs(p)
        return str(a.get("action_space", "cursor"))
    return "cursor"


def collect(dataset: Path) -> dict[int, list[int]]:
    by: dict[int, list[int]] = {}
    for p in sorted(dataset.iterdir()):
        if not p.name.endswith(".zarr"):
            continue
        m = re.search(r"_obs(\d+)_", p.name)
        lvl = int(m.group(1)) if m else -1
        attrs = _attrs(p)
        T = attrs.get("total_frames")
        if T:
            by.setdefault(lvl, []).append(int(T))
    return by


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--multiplier", type=float, default=1.1)
    ap.add_argument("--statistic", choices=("max", "p95", "p99", "median"), default="max")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    ds = a.dataset.resolve()
    by = collect(ds)
    if not by:
        raise SystemExit("no episodes with total_frames in %s" % ds)

    def stat(v: np.ndarray) -> float:
        if a.statistic == "median":
            return float(np.percentile(v, 50))
        if a.statistic == "max":
            return float(v.max())
        return float(np.percentile(v, 95 if a.statistic == "p95" else 99))

    out: dict = {}
    print("%-6s %6s %6s %7s %7s %6s %8s" %
          ("level", "n", "min", "mean", "p95", "max", "budget"))
    print("-" * 52)
    for lvl in sorted(by):
        v = np.array(by[lvl])
        budget = int(np.ceil(a.multiplier * stat(v)))
        out[str(lvl)] = {
            "n": int(v.size), "min": int(v.min()), "mean": round(float(v.mean()), 1),
            "p95": int(np.percentile(v, 95)), "max": int(v.max()), "budget": budget,
            "median": int(np.percentile(v, 50)),
            # one long demo can set the whole budget -- record how outlier-driven
            # this level is so a reviewer can see it without recomputing
            "max_over_p95": round(float(v.max()) / max(float(np.percentile(v, 95)), 1.0), 3),
        }
        print("%-6d %6d %6d %7.0f %7.0f %6d %8d"
              % (lvl, v.size, v.min(), v.mean(), np.percentile(v, 95), v.max(), budget))

    doc = {
        "dataset": str(ds),
        "dataset_name": ds.name,
        "n_episodes": int(sum(len(x) for x in by.values())),
        "statistic": a.statistic,
        "multiplier": a.multiplier,
        "action_space": detect_action_space(ds),
        "formula": "ceil(multiplier * %s(total_frames | level))" % a.statistic,
        "by_level": out,
    }
    doc["content_sha256"] = hashlib.sha256(
        json.dumps(doc, sort_keys=True).encode()).hexdigest()[:16]

    dest = a.out or (DEFAULT_BUDGET_DIR / ("%s.json" % ds.name))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=2))
    print("\nwrote %s   (sha %s)" % (dest, doc["content_sha256"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
