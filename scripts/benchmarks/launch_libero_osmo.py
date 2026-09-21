"""Render a pinned, single-L40S ARC/OAT workflow; submit with the OSMO CLI."""

import argparse
import re
from pathlib import Path

import yaml

from egomimic.benchmarks.libero.catalog import TASKS


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
):
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
    if campaign_id is not None and (
        mode != "full"
        or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,44}", campaign_id)
        or run_id != f"{campaign_id}-{suite.replace('_', '-')}"
    ):
        raise ValueError("Campaign requires full mode and matching suite run IDs")
    if replay and (evaluate_from_run or campaign_id):
        raise ValueError("Replay calibration is separate from a policy campaign")
    for source in (resume_from_run, arc_replay_run):
        if source is not None and not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", source):
            raise ValueError("Invalid training/replay source run ID")
    if evaluate_from_run and resume_from_run:
        raise ValueError("Choose evaluation recovery or training resume")
    if replay_spec not in {"libero_arc_replay", "libero_arc_replay_expanded"}:
        raise ValueError("Unknown checked-in replay specification")
    entry = Path(__file__).with_name("libero_osmo_entry.sh").read_text()
    return {
        "workflow": {
            "name": run_id,
            "resources": {
                "default": {
                    "cpu": 12,
                    "gpu": 1,
                    "memory": "64Gi",
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
                        "RUN_KIND": "replay" if replay else "benchmark",
                        "RESUME_FROM_RUN": resume_from_run or "",
                        "ARC_REPLAY_RUN": arc_replay_run or "",
                        "REPLAY_SPEC": replay_spec,
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
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--resume-from-run")
    parser.add_argument("--arc-replay-run")
    parser.add_argument("--replay-spec", default="libero_arc_replay")
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
            ),
            handle,
            sort_keys=False,
        )


if __name__ == "__main__":
    main()
