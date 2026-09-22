"""Replay a frozen shared-suite ARC setting, then train its native policy."""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[3]
PROFILES = ROOT / "docs/results/libero_arc_common_configs_20260921.json"


def profile_settings(profile_id, mode):
    """Bind a frozen selection to the actual Hydra recipe and requested mode."""
    report = json.loads(PROFILES.read_text())
    matches = [p for p in report["profiles"] if p["id"] == profile_id]
    if len(matches) != 1 or mode not in matches[0]["modes"]:
        raise ValueError("Unknown ARC profile or incompatible mode")
    profile = matches[0]
    expected = {
        "arc_mode": mode,
        "arc_action_dim": 12,
        "arc_waypoints": profile["num_waypoints"],
        "arc_max_translation": profile["max_translation"],
        "arc_max_rotation_degrees": profile["max_rotation_degrees"],
        "arc_velocity_norm_bound": 3**0.5,
    }
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "egomimic/hydra_configs")
    ):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=[
                f"+experiment={profile['recipe']}",
                f"benchmark.arc_mode={mode}",
            ],
        )
    if any(cfg.benchmark.get(k) != v for k, v in expected.items()):
        raise ValueError("ARC recipe differs from the frozen calibration selection")
    return expected


def fixed_replay_spec(profile_id, mode):
    settings = profile_settings(profile_id, mode)
    spec = yaml.safe_load(
        (
            ROOT / f"egomimic/hydra_configs/benchmark/libero_arc_replay_{mode}.yaml"
        ).read_text()
    )
    spec.update(
        rotation_degrees=[settings["arc_max_rotation_degrees"]],
        translation_metres=[settings["arc_max_translation"]],
        waypoints=[settings["arc_waypoints"]],
        # Fresh demonstrations, distinct from the earlier 30–34 pilot and
        # 35–49 per-suite confirmation. The selected parameters stay fixed.
        confirmation_demos=list(range(10, 25)),
        frozen_profile=profile_id,
        selection_source="libero_arc_common_configs_20260921",
    )
    from egomimic.benchmarks.libero.cluster import digest
    from egomimic.benchmarks.libero.replay import validate_spec

    spec["selection_artifact_sha256"] = digest(PROFILES)
    validate_spec(spec)
    return spec


def main():
    from egomimic.benchmarks.libero.catalog import TASKS
    from egomimic.benchmarks.libero.cluster import execute

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--suite", choices=TASKS, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--arc-mode", choices=("stk", "dur"), required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--epochs", type=int, default=5001)
    args = parser.parse_args()
    resuming = bool(os.environ.get("RESUME_FROM_RUN"))
    replay_runs = json.loads(os.environ.get("ARC_REPLAY_RUNS_JSON") or "{}")
    if resuming and args.mode == "full" and set(replay_runs) != {args.arc_mode}:
        raise ValueError("Resuming a sweep requires its completed mode-specific replay")
    replay_run = replay_runs.get(args.arc_mode) if resuming else args.run_id + "-replay"
    replay_run = replay_run or args.run_id + "-replay"
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", replay_run):
        raise ValueError("Sweep run ID is too long or invalid")
    args.root.mkdir(parents=True, exist_ok=True)
    spec = fixed_replay_spec(args.profile, args.arc_mode)
    # Match replay.main's YAML reader. JSON's 1e-12 spelling is parsed as a
    # string by PyYAML; safe_dump writes an unambiguous numeric scalar.
    path = args.root / "fixed-replay-spec.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    if args.mode == "full" and not resuming:
        if not os.environ.get("REPLAY_CALIBRATION_PARENT"):
            raise ValueError("A full sweep requires its audited calibration parent")
        execute(
            [
                sys.executable,
                "-m",
                "egomimic.benchmarks.libero.replay",
                "--root",
                str(args.root / "replay-check"),
                "--suite",
                args.suite,
                "--run-id",
                replay_run,
                "--spec",
                str(path),
            ],
            args.root / "replay-preflight.log",
        )
    os.environ["ARC_MODES_JSON"] = json.dumps([args.arc_mode])
    os.environ["ARC_REPLAY_RUNS_JSON"] = json.dumps({args.arc_mode: replay_run})
    execute(
        [
            sys.executable,
            "-m",
            "egomimic.benchmarks.libero.cluster",
            "--root",
            str(args.root),
            "--suite",
            args.suite,
            "--run-id",
            args.run_id,
            "--mode",
            args.mode,
            "--epochs",
            str(args.epochs),
            "--arc-only",
            "--arc-profile",
            args.profile,
        ],
        args.root / "arc-policy.log",
    )


if __name__ == "__main__":
    main()
