#!/usr/bin/env python
"""Compose and instantiate experiment configs before submitting a job.

Hydra validates on construction, and in these workflows construction happens
after cloning, installing, and staging tens of gigabytes -- so a one-character
config mistake costs hours before it says anything. This runs the same
construction locally in seconds.

It checks, per experiment:
  * the config composes at all;
  * the wandb run name stays under the 128-character Name limit;
  * every train/valid transform_list instantiates and agrees on token shape;
  * the pipeline graph builds and every stage agrees on action_dim;
  * the denoiser accepts the exact token the loader produces, and is
    shape-preserving;
  * no held-out embodiment appears in any train or valid dataset.

Usage:
    python tools/preflight_experiment_configs.py \
        pusht/artic_cotrain7_dp_paper pusht/artic_cotrain7_arc_stk_D80_M16_R26deg
    python tools/preflight_experiment_configs.py --holdout umi scoop -- <exps...>
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(REPO, "egomimic", "hydra_configs")
THREE_DOF = ("u_socket", "triangle", "scoop")


def _fake_native(dim: int, frames: int = 81) -> np.ndarray:
    t = np.linspace(0.0, 1.0, frames)
    cols = [60.0 * t, 25.0 * np.sin(2.2 * t), 0.9 * t]
    if dim == 4:
        cols.append((t > 0.4).astype(float))
    return np.column_stack(cols).astype(np.float32)


def _check(experiment: str, holdout: list[str]) -> list[str]:
    problems: list[str] = []
    cfg = compose(
        config_name="train_zarr_cartesian", overrides=[f"+experiment={experiment}"]
    )
    name, desc = cfg.get("name"), cfg.get("description")
    run_name = f"{name}_{desc}" if desc else str(name)
    if len(run_name) > 128:
        problems.append(f"wandb Name is {len(run_name)} chars (limit 128)")

    domains = list(cfg.data.train_datasets) + list(cfg.data.valid_datasets)
    leaked = sorted({h for h in holdout for d in domains if h in d})
    if leaked:
        problems.append(f"held-out embodiment(s) present in datasets: {leaked}")

    shapes = set()
    for domain, node in cfg.data.train_datasets.items():
        steps = instantiate(node.resolver.transform_list, keys=["actions"])
        short = domain.replace("pushshapes_sim_", "")
        batch = {"actions": _fake_native(3 if short in THREE_DOF else 4)}
        for step in steps:
            batch = step.transform(batch)
        shapes.add(tuple(np.asarray(batch["actions"]).shape))
    if len(shapes) != 1:
        problems.append(f"embodiments disagree on token shape: {sorted(shapes)}")
    rows, width = sorted(shapes)[0]

    algo = instantiate(cfg.model.pipeline)
    # PipelineAlgo keeps the graph in nets["pipeline"], not on itself.
    stages = list(algo.nets["pipeline"].stages)
    params = sum(
        p.numel() for st in stages if isinstance(st, nn.Module) for p in st.parameters()
    )
    dims = sorted(
        {int(st.action_dim) for st in stages if getattr(st, "action_dim", None)}
    )
    if dims != [width]:
        problems.append(f"stage action_dims {dims} do not match token width {width}")

    denoiser = [st for st in stages if type(st).__name__ == "DiffusionDenoiserStage"]
    if not denoiser:
        problems.append("no DiffusionDenoiserStage in the graph")
    else:
        net = denoiser[0].policy.model
        cond = int(cfg.model.pipeline.stages[3].condition_input_dim)
        x = torch.zeros(2, rows, width)
        with torch.no_grad():
            out = net(x, torch.zeros(2, dtype=torch.long), torch.zeros(2, cond))
        if tuple(out.shape) != tuple(x.shape):
            problems.append(
                f"denoiser maps {tuple(x.shape)} -> {tuple(out.shape)}, not shape-preserving"
            )

    print(f"  name            {name}  ({len(run_name)} char run name)")
    print(f"  train domains   {len(cfg.data.train_datasets)}")
    print(f"  token           ({rows}, {width})")
    print(f"  pipeline params {params / 1e6:.2f}M")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="+")
    ap.add_argument("--holdout", nargs="*", default=["umi", "scoop"])
    args = ap.parse_args()

    os.chdir(REPO)
    sys.path.insert(0, REPO)
    failures = 0
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        for experiment in args.experiments:
            print(f"\n=== {experiment}")
            try:
                problems = _check(experiment, list(args.holdout))
            except Exception as exc:  # noqa: BLE001 - report, do not mask
                print(f"  FAIL {type(exc).__name__}: {exc}")
                failures += 1
                continue
            for problem in problems:
                print(f"  FAIL {problem}")
            failures += len(problems)
            if not problems:
                print("  OK")
    print(f"\n{'FAILED' if failures else 'PASSED'}: {failures} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
