"""Run Yam's maintained hardware runtime in its own uv environment."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess

from egomimic.robot.yam import UPSTREAM_REVISION

COMMANDS = {"teleop": "quest-teleop", "rollout": "yam-policy", "cameras": "camera-only"}


def runtime_command(repo, mode, args):
    repo = Path(repo).expanduser().resolve()
    if mode not in COMMANDS:
        raise ValueError(f"Unknown Yam runtime mode: {mode}")
    if not (repo / "rl2_yam/runtime").is_dir():
        raise ValueError(f"Not a yam-pipeline checkout: {repo}")
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(
            f"Yam runtime revision {revision} differs from the integrated revision "
            f"{UPSTREAM_REVISION}. Use a separate checkout at that revision."
        )
    # --no-sync prevents launching hardware from changing its installed packages.
    return repo, ["uv", "run", "--no-sync", COMMANDS[mode], *args]


def main(default_mode=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if default_mode is None:
        parser.add_argument("mode", choices=COMMANDS)
    parser.add_argument("--yam-repo", default=os.environ.get("YAM_PIPELINE_ROOT"))
    parser.add_argument("--print-command", action="store_true")
    args, forwarded = parser.parse_known_args()
    if not args.yam_repo:
        parser.error("set --yam-repo or YAM_PIPELINE_ROOT to the Yam checkout")
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    # Resolve station/output paths before changing to the hardware workspace.
    for flag in ("--config", "--record-dir", "--trajectory-dir"):
        for i, value in enumerate(forwarded):
            if value == flag and i + 1 < len(forwarded):
                forwarded[i + 1] = str(Path(forwarded[i + 1]).expanduser().resolve())
            elif value.startswith(flag + "="):
                forwarded[i] = (
                    flag
                    + "="
                    + str(Path(value.split("=", 1)[1]).expanduser().resolve())
                )
    repo, command = runtime_command(args.yam_repo, default_mode or args.mode, forwarded)
    if args.print_command:
        print(f"cd {shlex.quote(str(repo))} && {shlex.join(command)}")
        return
    # Replace this process so Ctrl+C and shutdown reach the upstream runtime.
    os.chdir(repo)
    os.execvp(command[0], command)
