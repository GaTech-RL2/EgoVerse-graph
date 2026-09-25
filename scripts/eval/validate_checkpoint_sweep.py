#!/usr/bin/env python3
"""Run validation-only Hydra jobs over every checkpoint in one training run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from egomimic.eval.distance_budget_dtw import (  # noqa: E402
    ARC_CHUNKING_MODES,
    METRIC_VERSION,
)
from egomimic.utils.hydra_override import encode_hydra_string_override  # noqa: E402


def completion_signature(command: list[str], checkpoint: Checkpoint) -> str:
    """Old MSE outputs and differently configured sweeps are not cache hits."""
    identity = {
        "metric_version": METRIC_VERSION,
        "command": command,
        "checkpoint": asdict(checkpoint),
        "source_commit": subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "checkpoint_mtime_ns": Path(checkpoint.path).stat().st_mtime_ns,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def completed_evaluation(marker: Path, result: Path, signature: str) -> bool:
    if not marker.is_file() or not result.is_file():
        return False
    try:
        payload = json.loads(result.read_text())
        version_matches = payload.get("distance_dtw_enabled") is False or (
            payload.get("distance_dtw_enabled") is True
            and payload.get("distance_dtw_metric_version") == METRIC_VERSION
        )
        return (
            version_matches
            and json.loads(marker.read_text()).get("signature") == signature
        )
    except (OSError, ValueError):
        return False


_STEP_PATTERNS = (
    re.compile(r"(?:^|[-_])step[=_-]?(\d+)(?:[-_.]|$)", re.IGNORECASE),
    re.compile(r"global[-_]?step[=_-]?(\d+)(?:[-_.]|$)", re.IGNORECASE),
)


@dataclass(frozen=True)
class Checkpoint:
    path: str
    step: int
    size_bytes: int


def checkpoint_step(path: Path) -> int:
    for pattern in _STEP_PATTERNS:
        match = pattern.search(path.name)
        if match:
            return int(match.group(1))
    # Some long-running HPT recipes name checkpoints by epoch only. mmap keeps
    # the 4+ GB tensor storages lazy while reading the small Lightning metadata.
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    try:
        step = checkpoint.get("global_step")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError(
                f"checkpoint has no nonnegative integer global_step: {path}"
            )
        return step
    finally:
        del checkpoint


def discover_checkpoints(root: Path, pattern: str) -> list[Checkpoint]:
    root = root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"checkpoint root is not a directory: {root}")
    selected: dict[int, Checkpoint] = {}
    seen_paths: set[Path] = set()
    for candidate in sorted(root.glob(pattern)):
        if not candidate.is_file() or candidate.name == "last.ckpt":
            continue
        resolved = candidate.resolve(strict=True)
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        step = checkpoint_step(candidate)
        checkpoint = Checkpoint(
            path=str(resolved), step=step, size_bytes=resolved.stat().st_size
        )
        previous = selected.get(step)
        if previous is not None and previous.path != checkpoint.path:
            raise ValueError(
                f"multiple checkpoints resolve to global step {step}: "
                f"{previous.path} and {checkpoint.path}"
            )
        selected[step] = checkpoint
    if not selected:
        raise ValueError(f"no numbered checkpoints matched {root / pattern}")
    return [selected[step] for step in sorted(selected)]


def validation_command(
    *,
    python: str,
    experiment: str,
    checkpoint: Checkpoint,
    output_dir: Path,
    action_mode: str,
    execute_fraction: float,
    limit_val_episodes: int | None,
    wandb_run_id: str,
    wandb_name: str,
    wandb_group: str,
    extra_overrides: list[str],
    video_only: bool = False,
    arc_execution_cap_mode: str = "waypoints",
    arc_chunking_mode: str | None = None,
) -> list[str]:
    results_path = output_dir / "open_loop_sim.json"
    video_dir = output_dir / "val_videos"
    command = [
        python,
        "egomimic/trainHydra.py",
        f"+experiment={experiment}",
        "mode=eval",
        "eval_logger_enabled=true",
        "trainer.limit_val_batches=1.0",
        "evaluator=eval_open_loop_sim",
        "evaluator.distance_dtw_enabled=true",
        f"evaluator.action_mode={action_mode}",
        f"evaluator.execute_fraction={execute_fraction}",
        f"evaluator.arc_execution_cap_mode={arc_execution_cap_mode}",
        f"evaluator.log_step={checkpoint.step}",
        encode_hydra_string_override("ckpt_path", checkpoint.path),
        (
            "evaluator.results_path=null"
            if video_only
            else encode_hydra_string_override(
                "evaluator.results_path", str(results_path)
            )
        ),
        encode_hydra_string_override("evaluator.video_output_dir", str(video_dir)),
        encode_hydra_string_override("logger.wandb.id", wandb_run_id),
        encode_hydra_string_override("+logger.wandb.name", wandb_name),
        encode_hydra_string_override("logger.wandb.group", wandb_group),
        encode_hydra_string_override(
            "logger.wandb.job_type",
            (
                "offline_checkpoint_video"
                if video_only
                else "offline_checkpoint_validation"
            ),
        ),
        encode_hydra_string_override("+logger.wandb.resume", "allow"),
    ]
    if arc_chunking_mode is not None:
        if arc_chunking_mode not in ARC_CHUNKING_MODES:
            raise ValueError(f"Invalid arc_chunking_mode: {arc_chunking_mode!r}")
        command.append(f"evaluator.arc_chunking_mode={arc_chunking_mode}")
    if video_only:
        command.append("evaluator.video_only=true")
    if limit_val_episodes is not None:
        command.append(f"evaluator.limit_val_episodes={limit_val_episodes}")
    command.extend(extra_overrides)
    return command


def _run_and_tee(command: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover every numbered Lightning checkpoint in a run and execute "
            "the open-loop ARC evaluator sequentially. The default is a dry run."
        )
    )
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--checkpoint-pattern", default="**/*.ckpt")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--wandb-run-id", required=True)
    parser.add_argument("--wandb-name", required=True)
    parser.add_argument("--wandb-group", default="offline-checkpoint-validation")
    parser.add_argument(
        "--action-mode",
        choices=("arc", "baseline"),
        required=True,
        help="Interpret predictions as ARC tokens or control-frame chunks.",
    )
    parser.add_argument(
        "--execute-fraction",
        type=float,
        required=True,
        help=(
            "Fraction before replanning: ARC waypoints or distance according "
            "to --arc-execution-cap-mode; control frames for baseline mode."
        ),
    )
    parser.add_argument(
        "--arc-execution-cap-mode",
        choices=("waypoints", "distance"),
        default="waypoints",
        help=(
            "ARC prefix semantics before detokenization. Waypoint mode requires "
            "M * execute_fraction to be integral and per-waypoint velocity."
        ),
    )
    parser.add_argument("--limit-val-episodes", type=int)
    parser.add_argument(
        "--arc-chunking-mode",
        choices=ARC_CHUNKING_MODES,
        help="ARC translation mode; omitted uses the experiment/codec default. Match the checkpoint's training mode.",
    )
    parser.add_argument(
        "--video-only",
        action="store_true",
        help="Render and upload validation videos without scoring or metric JSON.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override; repeat for multiple values.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.0 < args.execute_fraction <= 1.0:
        raise ValueError("--execute-fraction must be in (0, 1]")
    if args.limit_val_episodes is not None and args.limit_val_episodes < 1:
        raise ValueError("--limit-val-episodes must be positive")

    checkpoints = discover_checkpoints(args.checkpoint_root, args.checkpoint_pattern)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint_root": str(args.checkpoint_root.expanduser().resolve()),
        "checkpoint_pattern": args.checkpoint_pattern,
        "experiment": args.experiment,
        "action_mode": args.action_mode,
        "execute_fraction": args.execute_fraction,
        "arc_execution_cap_mode": args.arc_execution_cap_mode,
        "arc_chunking_mode": args.arc_chunking_mode,
        "distance_dtw_metric_version": METRIC_VERSION,
        "limit_val_episodes": args.limit_val_episodes,
        "video_only": args.video_only,
        "wandb_run_id": args.wandb_run_id,
        "wandb_name": args.wandb_name,
        "wandb_group": args.wandb_group,
        "checkpoints": [asdict(checkpoint) for checkpoint in checkpoints],
        "commands": [],
    }

    for checkpoint in checkpoints:
        checkpoint_output = output_root / f"step-{checkpoint.step:09d}"
        checkpoint_output.mkdir(parents=True, exist_ok=True)
        result_path = checkpoint_output / "open_loop_sim.json"
        command = validation_command(
            python=args.python,
            experiment=args.experiment,
            checkpoint=checkpoint,
            output_dir=checkpoint_output,
            action_mode=args.action_mode,
            execute_fraction=args.execute_fraction,
            limit_val_episodes=args.limit_val_episodes,
            wandb_run_id=args.wandb_run_id,
            wandb_name=args.wandb_name,
            wandb_group=args.wandb_group,
            video_only=args.video_only,
            arc_execution_cap_mode=args.arc_execution_cap_mode,
            arc_chunking_mode=args.arc_chunking_mode,
            extra_overrides=list(args.override),
        )
        manifest["commands"].append(command)
        completion_path = (
            checkpoint_output / "video_only_complete.json"
            if args.video_only
            else result_path
        )
        signature = completion_signature(command, checkpoint)
        metric_marker = checkpoint_output / "evaluation_complete.json"
        if args.resume and (
            (args.video_only and completion_path.is_file())
            or (
                not args.video_only
                and completed_evaluation(metric_marker, result_path, signature)
            )
        ):
            print(
                f"skip completed checkpoint step={checkpoint.step}: {completion_path}"
            )
            continue
        print(json.dumps(command))
        if args.execute:
            if not args.video_only:
                metric_marker.unlink(missing_ok=True)
            _run_and_tee(command, checkpoint_output / "validation.log")
            if not args.video_only:
                # A successful subprocess without its result is not a completed sweep item.
                result = json.loads(result_path.read_text(encoding="utf-8"))
                dtw_requested = next(
                    (
                        item.split("=", 1)[1].lower() != "false"
                        for item in reversed(command)
                        if item.startswith("evaluator.distance_dtw_enabled=")
                    ),
                    True,
                )
                if dtw_requested and (
                    not result.get("distance_dtw_enabled")
                    or result.get("distance_dtw_metric_version") != METRIC_VERSION
                ):
                    raise RuntimeError(
                        "Validation output has a stale distance-DTW metric version"
                    )
                metric_marker.write_text(
                    json.dumps({"signature": signature}) + "\n", encoding="utf-8"
                )
            if args.video_only:
                completion_path.write_text(
                    json.dumps({"checkpoint_step": checkpoint.step}) + "\n",
                    encoding="utf-8",
                )

    (output_root / "checkpoint_sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"prepared {len(checkpoints)} checkpoint validation command(s) under "
        f"{output_root}; execute={args.execute}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
