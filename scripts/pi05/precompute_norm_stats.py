"""Compatibility entry point for the shared normalization tool."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "data/precompute_norm_stats.py"),
        run_name="__main__",
    )
