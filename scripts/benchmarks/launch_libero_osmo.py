"""Render a pinned, single-L40S ARC/OAT workflow; submit with the OSMO CLI."""

import argparse
import json
import re
from pathlib import Path

import yaml

from egomimic.benchmarks.libero.catalog import TASKS
from egomimic.benchmarks.libero.cluster import arc_method_modes, campaign_sources


def workflow(
    commit,
    run_id,
    suite,
    mode="smoke",
    epochs=5001,
    evaluate_from_run=None,
    campaign_id=None,
    replay=False,
    resume_from_run=None,
    arc_replay_run=None,
    replay_spec="libero_arc_replay",
    calibration_parent=None,
    raw_cache=None,
    campaign_runs=None,
    arc_modes=None,
    arc_replay_runs=None,
    arc_profile=None,
    oat_reference_run=None,
):
    arc_modes = list(arc_modes or (["joint_dur"] if arc_replay_run else ["dur", "stk"]))
    arc_method_modes(arc_modes)
    arc_replay_runs = dict(arc_replay_runs or {})
    if arc_replay_run and arc_modes != ["joint_dur"]:
        raise ValueError("Independent modes require mode-specific replay runs")
    if set(arc_replay_runs) - set(arc_modes):
        raise ValueError("Unexpected ARC replay mode")
    if arc_profile:
        from egomimic.benchmarks.libero.arc_sweep import profile_settings

        if (
            len(arc_modes) != 1
            or replay
            or campaign_id
            or evaluate_from_run
            or arc_replay_runs
            or arc_replay_run
        ):
            raise ValueError("ARC profile requires one standalone policy mode")
        profile_settings(arc_profile, arc_modes[0])
        if len(run_id) > 56:
            raise ValueError("ARC sweep run ID must leave room for -replay")
        if mode == "full" and (not calibration_parent or not oat_reference_run):
            raise ValueError(
                "Full ARC sweep requires calibration and OAT reference runs"
            )
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Use an immutable 40-character Git commit")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", run_id):
        raise ValueError("run_id must be a DNS label of at most 63 characters")
    if suite not in TASKS or mode not in {"smoke", "full"} or epochs < 1:
        raise ValueError("Invalid suite, mode or epoch budget")
    if evaluate_from_run is not None and not re.fullmatch(
        r"[a-z0-9][a-z0-9-]{0,62}", evaluate_from_run
    ):
        raise ValueError("Invalid evaluation source run ID")
    if campaign_runs is not None and campaign_id is None:
        raise ValueError("Campaign source manifest requires a campaign ID")
    expected_run = (
        campaign_sources(campaign_id, commit, campaign_runs)[suite]
        if campaign_id
        else None
    )
    if campaign_id is not None and (
        mode != "full"
        or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,44}", campaign_id)
        or run_id != expected_run["run_id"]
        or commit != expected_run["source_commit"]
    ):
        raise ValueError("Campaign requires full mode and matching suite run IDs")
    if replay and (evaluate_from_run or campaign_id):
        raise ValueError("Replay calibration is separate from a policy campaign")
    for source in (
        resume_from_run,
        arc_replay_run,
        calibration_parent,
        oat_reference_run,
        *arc_replay_runs.values(),
    ):
        if source is not None and not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", source):
            raise ValueError("Invalid training/replay source run ID")
    if evaluate_from_run and resume_from_run:
        raise ValueError("Choose evaluation recovery or training resume")
    if replay_spec not in {
        "libero_arc_replay",
        "libero_arc_replay_expanded",
        "libero_arc_replay_refine",
        "libero_arc_replay_stk",
        "libero_arc_replay_dur",
    }:
        raise ValueError("Unknown checked-in replay specification")
    if raw_cache is not None and not Path(raw_cache).is_absolute():
        raise ValueError("Raw cache must be an absolute path")
    replay_config = yaml.safe_load(
        (
            Path(__file__).parents[2]
            / "egomimic/hydra_configs/benchmark"
            / f"{replay_spec}.yaml"
        ).read_text()
    )
    replay_workers = replay_config["workers"]
    preflight = replay or (arc_profile is not None and mode == "full")
    if arc_profile:
        replay_workers = 16
    entry = Path(__file__).with_name("libero_osmo_entry.sh").read_text()
    return {
        "workflow": {
            "name": run_id,
            "resources": {
                "default": {
                    "cpu": max(12, replay_workers + 4) if preflight else 12,
                    "gpu": 1,
                    "memory": "128Gi"
                    if arc_profile and preflight
                    else f"{replay_config.get('memory_gib', 64)}Gi"
                    if preflight
                    else "64Gi",
                    "storage": "128Gi" if replay else "240Gi",
                    "platform": "ovx-l40s",
                }
            },
            "timeout": {
                "queue_timeout": "2d" if replay or mode == "full" else "4h",
                "exec_timeout": "2d"
                if replay
                else ("4h" if mode == "smoke" else "60d"),
            },
            "tasks": [
                {
                    "name": "libero",
                    "image": "nvcr.io/nvidia/pytorch:25.06-py3",
                    "credentials": {
                        "egoverse-github": {"GITHUB_TOKEN": "github_token"},
                        "grabber-arc-r2-20260916": {
                            "R2_ACCESS_KEY_ID": "r2_access_key_id",
                            "R2_SECRET_ACCESS_KEY": "r2_secret_access_key",
                            "R2_ENDPOINT_URL": "r2_endpoint_url",
                        },
                    },
                    "environment": {
                        "SOURCE_COMMIT": commit,
                        "RUN_ID": run_id,
                        "SUITE": suite,
                        "RUN_MODE": mode,
                        "EPOCHS": str(epochs),
                        "EVALUATE_FROM_RUN": evaluate_from_run or "",
                        "CAMPAIGN_ID": campaign_id or "",
                        "CAMPAIGN_RUNS_JSON": json.dumps(campaign_runs),
                        "RUN_KIND": "arc_sweep"
                        if arc_profile
                        else "replay"
                        if replay
                        else "benchmark",
                        "ARC_PROFILE": arc_profile or "",
                        "OAT_REFERENCE_RUN": oat_reference_run or "",
                        "RESUME_FROM_RUN": resume_from_run or "",
                        "ARC_REPLAY_RUN": arc_replay_run or "",
                        "ARC_MODES_JSON": json.dumps(arc_modes),
                        "ARC_REPLAY_RUNS_JSON": json.dumps(arc_replay_runs),
                        "REPLAY_SPEC": replay_spec,
                        "REPLAY_CALIBRATION_PARENT": calibration_parent or "",
                        "LIBERO_RAW_CACHE": str(raw_cache) if raw_cache else "",
                    },
                    "command": ["bash"],
                    "args": ["/tmp/entry.sh"],
                    "files": [{"path": "/tmp/entry.sh", "contents": entry}],
                }
            ],
        }
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--suite", choices=TASKS, default="libero_10")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--epochs", type=int, default=5001)
    parser.add_argument("--evaluate-from-run")
    parser.add_argument("--campaign-id")
    parser.add_argument("--campaign-runs-file", type=Path)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--resume-from-run")
    parser.add_argument("--arc-replay-run")
    parser.add_argument("--arc-modes", nargs="+", choices=("joint_dur", "stk", "dur"))
    parser.add_argument("--arc-replay-runs-file", type=Path)
    parser.add_argument("--arc-profile")
    parser.add_argument("--oat-reference-run")
    parser.add_argument("--replay-spec", default="libero_arc_replay")
    parser.add_argument("--calibration-parent")
    parser.add_argument("--raw-cache")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x") as handle:
        yaml.safe_dump(
            workflow(
                args.commit,
                args.run_id,
                args.suite,
                args.mode,
                args.epochs,
                args.evaluate_from_run,
                args.campaign_id,
                args.replay,
                args.resume_from_run,
                args.arc_replay_run,
                args.replay_spec,
                args.calibration_parent,
                args.raw_cache,
                json.loads(args.campaign_runs_file.read_text())
                if args.campaign_runs_file
                else None,
                args.arc_modes,
                json.loads(args.arc_replay_runs_file.read_text())
                if args.arc_replay_runs_file
                else None,
                args.arc_profile,
                args.oat_reference_run,
            ),
            handle,
            sort_keys=False,
        )


if __name__ == "__main__":
    main()
