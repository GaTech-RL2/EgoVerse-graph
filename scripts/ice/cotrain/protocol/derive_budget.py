"""Budgets for an embodiment that has no data at some levels.

    ratio        = stat(A | obs0) / stat(B | obs0)
    budget(A, L) = ceil( multiplier * stat(B | L) * ratio )

A = target embodiment (missing level-L data), B = source embodiment that has it.
The ratio is a per-embodiment speed factor measured where BOTH have data
(level 0) and assumed to hold at every level. Derived rows are marked so they are
never mistaken for measured ones.

Usage:
  python derive_budget.py --target <dsA> --source <dsB> [--multiplier 1.1]
                          [--statistic max] [--out <json>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


BUDGET_DIR = Path("/coc/flash7/paphiwetsa3/scripts/eval/budgets")


def per_level(dataset: Path) -> dict[int, np.ndarray]:
    by: dict[int, list[int]] = {}
    for p in sorted(dataset.iterdir()):
        if not p.name.endswith(".zarr"):
            continue
        m = re.search(r"_obs(\d+)_", p.name)
        lvl = int(m.group(1)) if m else -1
        T = _attrs(p).get("total_frames")
        if T:
            by.setdefault(lvl, []).append(int(T))
    return {k: np.array(v) for k, v in by.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, type=Path)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--multiplier", type=float, default=1.1)
    ap.add_argument("--statistic", choices=("max", "p95", "p99"), default="max")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    A, B = a.target.resolve(), a.source.resolve()
    ta, tb = per_level(A), per_level(B)

    def stat(v):
        if a.statistic == "max":
            return float(v.max())
        return float(np.percentile(v, 95 if a.statistic == "p95" else 99))

    if 0 not in ta or 0 not in tb:
        raise SystemExit("both datasets need level-0 data to form the ratio")
    a0, b0 = stat(ta[0]), stat(tb[0])
    ratio = a0 / b0
    # the p95 alternative, recorded so the choice is visible
    ratio_p95 = float(np.percentile(ta[0], 95)) / float(np.percentile(tb[0], 95))

    print("target : %s" % A.name)
    print("source : %s" % B.name)
    print("ratio  = %s(A|obs0) / %s(B|obs0) = %.0f / %.0f = %.4f"
          % (a.statistic, a.statistic, a0, b0, ratio))
    print("         (p95-based alternative would be %.4f)" % ratio_p95)
    print("multiplier = %s   statistic = %s\n" % (a.multiplier, a.statistic))

    print("%-6s %-9s %8s %8s %10s %9s   %s"
          % ("level", "kind", "srcstat", "xratio", "xmult", "BUDGET", "calculation"))
    print("-" * 104)

    out: dict[str, dict] = {}
    for lvl in sorted(set(tb) | set(ta)):
        if lvl in ta and ta[lvl].size:
            s = stat(ta[lvl])
            budget = int(math.ceil(a.multiplier * s))
            out[str(lvl)] = {"budget": budget, "derived": False, "n": int(ta[lvl].size),
                             "stat": int(s), "max": int(ta[lvl].max()),
                             "p95": int(np.percentile(ta[lvl], 95))}
            print("%-6d %-9s %8d %8s %10s %9d   ceil(%.1f * %d)  [own data, n=%d]"
                  % (lvl, "measured", s, "-", "-", budget, a.multiplier, s, ta[lvl].size))
        elif lvl in tb and tb[lvl].size:
            s = stat(tb[lvl])
            scaled = s * ratio
            budget = int(math.ceil(a.multiplier * scaled))
            out[str(lvl)] = {"budget": budget, "derived": True,
                             "source_embodiment": B.name, "source_stat": int(s),
                             "ratio": round(ratio, 4), "scaled": round(scaled, 1),
                             "ratio_basis": {"A_obs0": int(a0), "B_obs0": int(b0)}}
            print("%-6d %-9s %8d %8.1f %10.1f %9d   ceil(%.1f * %d * %.4f)  [from %s]"
                  % (lvl, "DERIVED", s, scaled, a.multiplier * scaled, budget,
                     a.multiplier, s, ratio, B.name))

    def _aspace(d: Path) -> str:
        for q in sorted(d.iterdir()):
            if q.name.endswith(".zarr"):
                return str(_attrs(q).get("action_space", "cursor"))
        return "cursor"

    doc = {"dataset": str(A), "dataset_name": A.name,
           "statistic": a.statistic, "multiplier": a.multiplier,
           "action_space": _aspace(A),
           "derivation": {"source_dataset": str(B), "source_name": B.name,
                          "ratio": round(ratio, 4),
                          "ratio_formula": "%s(A|obs0)/%s(B|obs0)" % (a.statistic, a.statistic),
                          "A_obs0": int(a0), "B_obs0": int(b0),
                          "ratio_p95_alternative": round(ratio_p95, 4)},
           "formula_measured": "ceil(multiplier * %s(total_frames|level))" % a.statistic,
           "formula_derived": "ceil(multiplier * %s(source|level) * ratio)" % a.statistic,
           "by_level": out}
    doc["content_sha256"] = hashlib.sha256(
        json.dumps(doc, sort_keys=True).encode()).hexdigest()[:16]

    dest = a.out or (BUDGET_DIR / ("%s.json" % A.name))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=2))
    n_d = sum(1 for v in out.values() if v["derived"])
    print("\nmeasured %d levels, derived %d levels" % (len(out) - n_d, n_d))
    print("wrote %s   (sha %s)" % (dest, doc["content_sha256"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
