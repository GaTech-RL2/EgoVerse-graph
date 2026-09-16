#!/usr/bin/env python
"""Verify a norm_stats.json covers exactly the domains its experiment trains on.

cache_stats writes after each embodiment, so an interrupted pass leaves a file
covering some domains and not others, and the missing ones would train
unnormalized -- normalize() returns the tensor UNCHANGED when it finds no entry
for an embodiment id, which is silent and produces plausible numbers.

The count is taken from the experiment's own data.train_datasets, NOT hardcoded:
the co-train arms have seven domains but a single-embodiment BC baseline has
one, and a literal `== 7` rejects the baseline as if its stats were truncated.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, help="e.g. pusht/artic_cotrain7_dp_paper")
    ap.add_argument("--json", required=True, help="path to norm_stats.json")
    ap.add_argument("--config-name", default="train_zarr_cartesian")
    a = ap.parse_args()

    from hydra import compose, initialize_config_dir
    from egomimic.rldb.embodiment.embodiment import get_embodiment_id

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_dir = os.path.join(repo, "egomimic", "hydra_configs")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name=a.config_name,
                      overrides=[f"+experiment={a.experiment}"])
    domains = list(cfg.data.train_datasets.keys())
    want = {str(get_embodiment_id(d)) for d in domains}

    if not os.path.exists(a.json) or os.path.getsize(a.json) == 0:
        print(f"  FATAL no norm stats at {a.json}")
        return 1
    stats = json.load(open(a.json))["stats"]
    have = {k for k, v in stats.items() if len(v) > 0}

    missing = sorted(want - have, key=int)
    empty = sorted({k for k in want & set(stats) if not stats[k]}, key=int)
    extra = sorted(have - want, key=int)

    print(f"  norm stats {a.experiment}: {len(domains)} train domain(s) "
          f"{sorted(want, key=int)} -> present {sorted(have & want, key=int)}"
          + (f" extra {extra}" if extra else ""))
    if missing or empty:
        print(f"  FATAL missing={missing} empty={empty} — those domains would "
              f"train unnormalized")
        return 1
    print("  COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
