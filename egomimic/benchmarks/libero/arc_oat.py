"""Train a native OAT tokenizer on ARC supports, then its observation policy."""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from egomimic.benchmarks.libero.arc_sweep import profile_settings
from egomimic.benchmarks.libero.catalog import TASKS
from egomimic.benchmarks.libero.cluster import (
    ArtifactUploader,
    configure_simulator,
    digest,
    execute,
    load_arc_calibration,
    restore_checkpoints,
    stage_dataset,
    training_arguments,
    training_layout,
    validate_gpu_allocation,
    write_json,
)


def input_representation(settings):
    from egomimic.pipeline.stages_libero_arc import LiberoArcStage

    return LiberoArcStage(
        arc_mode=settings["arc_mode"],
        num_waypoints=settings["arc_waypoints"],
        horizon=32,
        max_translation=settings["arc_max_translation"],
        max_rotation_degrees=settings["arc_max_rotation_degrees"],
        velocity_norm_bound=settings["arc_velocity_norm_bound"],
    ).representation_context()


def verify_checkpoint(path, *, settings, suite, epochs, mode, method, complete=True):
    """Never continue from a raw-action tokenizer or incompatible ARC units."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if payload.get("oat_input_representation") != input_representation(settings):
        raise ValueError("Checkpoint uses a different ARC+OAT representation")
    protocol = payload["hyper_parameters"]["config_tree"]["model"]["benchmark_protocol"]
    if (
        payload["benchmark_data_context"]["suite"] != suite
        or protocol.get("suite") != suite
        or protocol.get("action_representation") != "arc_oat"
        or protocol.get("horizon") != 32
        or protocol.get("n_action_steps") != 16
        or protocol.get("n_obs_steps") != (0 if method == "tokenizer" else 2)
    ):
        raise ValueError("ARC+OAT checkpoint uses a different data/control protocol")
    completed = payload["loops"]["fit_loop"]["epoch_progress"]["current"]["completed"]
    if complete and completed != epochs:
        raise ValueError("ARC+OAT training has not completed its epoch budget")
    if payload["ema_num_updates"] != payload["global_step"] or not payload.get(
        "ema_state_dict"
    ):
        raise ValueError("ARC+OAT checkpoint lacks matching EMA updates")
    if mode == "full":
        budget = payload["training_budget"]
        if budget["epochs"] != epochs or budget["global_batch_size"] != 1024:
            raise ValueError("ARC+OAT training budget differs")
        if complete and payload["global_step"] != budget["total_optimizer_steps"]:
            raise ValueError("ARC+OAT optimizer budget is incomplete")
    if method == "oat" and not payload.get("oat_tokenizer_config"):
        raise ValueError("ARC+OAT policy lacks its frozen tokenizer architecture")
    return {
        "epochs_completed": completed,
        "global_step": payload["global_step"],
        "ema_num_updates": payload["ema_num_updates"],
        "input_representation": payload["oat_input_representation"],
    }


def compare_reference(client, root, evidence, reference, suite, arc_mode):
    """Compare only complete records with matching initial states; retain rejection."""
    from egomimic.benchmarks.libero.report import compare_runs

    directory = root / "reference-oat"
    directory.mkdir(exist_ok=False)
    receipts = {}
    for name in ("protocol.json", "episodes.jsonl"):
        key = f"experiments/arc-oat-20260919/{reference}/oat/{suite}/{name}"
        obj = client.get_object(Bucket="rldb", Key=key)
        path = directory / name
        path.write_bytes(obj["Body"].read())
        if obj["Metadata"]["sha256"] != digest(path):
            raise ValueError("OAT comparison artifact hash differs")
        receipts[name] = {"uri": "s3://rldb/" + key, "sha256": digest(path)}
    write_json(evidence / "oat-reference.json", receipts)
    return compare_runs(
        evidence / "arc_oat" / suite,
        directory,
        arc_method="arc_oat",
        arc_mode=arc_mode,
    )


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--suite", choices=TASKS, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--epochs", type=int, default=5001)
    args = parser.parse_args()
    if args.mode == "full" and args.epochs != 5001:
        raise ValueError(
            "Full ARC+OAT uses the released 5001 epochs per training stage"
        )
    modes = json.loads(os.environ["ARC_MODES_JSON"])
    if len(modes) != 1 or modes[0] not in ("stk", "dur"):
        raise ValueError("An ARC+OAT run requires one STK or DUR profile")
    profile = os.environ["ARC_PROFILE"]
    settings = profile_settings(profile, modes[0])
    replay_runs = json.loads(os.environ.get("ARC_REPLAY_RUNS_JSON", "{}"))
    if args.mode == "full" and set(replay_runs) != set(modes):
        raise ValueError("Full ARC+OAT requires the completed ARC replay selection")
    for run in (args.run_id, *replay_runs.values()):
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", run):
            raise ValueError("Invalid ARC+OAT run ID")
    gpus = int(os.environ["TRAINING_GPUS"])
    gpu_type = os.environ["BENCHMARK_GPU_TYPE"]
    validate_gpu_allocation(gpus, gpu_type)
    layout = training_layout(gpus, args.mode)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if commit != os.environ["SOURCE_COMMIT"]:
        raise ValueError("Unexpected ARC+OAT source revision")
    epochs = 1 if args.mode == "smoke" else args.epochs
    resume = os.environ.get("RESUME_FROM_RUN") or None
    reference = os.environ.get("OAT_REFERENCE_RUN") or None
    evidence = args.root / "evidence"
    evidence.mkdir(parents=True, exist_ok=False)
    uploader = ArtifactUploader(evidence, args.run_id)
    write_json(
        evidence / "runtime.json",
        {
            "run_kind": "arc_oat",
            "source_commit": commit,
            "suite": args.suite,
            "mode": args.mode,
            "epochs": epochs,
            "global_batch_size": layout["global_batch_size"],
            "training_layout": layout,
            "gpu_type": gpu_type,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "arc_profile": profile,
            "arc_modes": modes,
            "arc_replay_runs": replay_runs,
            "resume_from_run": resume,
            "oat_reference_run": reference,
            "evaluation_workers": 5 if args.mode == "full" else 1,
            "input_representation": input_representation(settings),
        },
    )
    uploader.thread.start()
    try:
        write_json(evidence / "status.json", {"state": "STAGING_DATA"})
        dataset = stage_dataset(args.root, args.suite, evidence)
        from egomimic.rldb.zarr.decoded_replay import prepare_decoded_replay
        from egomimic.rldb.zarr.libero_dataset import keymap

        cache = prepare_decoded_replay(
            dataset, [v["zarr_key"] for v in keymap().values()]
        )
        write_json(evidence / "decoded-replay-cache.json", cache.manifest)
        configure_simulator(args.root)
        if args.mode == "full":
            calibrated = load_arc_calibration(
                uploader.client,
                replay_runs[modes[0]],
                evidence,
                suite=args.suite,
                arc_mode=modes[0],
            )
            if any(settings.get(key) != value for key, value in calibrated.items()):
                raise ValueError("ARC+OAT profile differs from the audited replay")
        restored = (
            restore_checkpoints(
                uploader.client,
                resume,
                evidence,
                suite=args.suite,
                mode=args.mode,
                epochs=epochs,
                allow_partial=True,
                methods=("tokenizer", "oat"),
            )
            if resume
            else {}
        )
        for method in ("tokenizer", "oat"):
            checkpoint = evidence / f"training/{method}/checkpoints/last.ckpt"
            validation = dict(
                settings=settings,
                suite=args.suite,
                epochs=epochs,
                mode=args.mode,
                method=method,
            )
            if method in restored:
                verify_checkpoint(checkpoint, complete=False, **validation)
            if not restored.get(method, {}).get("complete"):
                write_json(
                    evidence / "status.json", {"state": "TRAINING", "method": method}
                )
                argv = training_arguments(
                    method,
                    args.suite,
                    dataset,
                    evidence,
                    args.mode,
                    epochs,
                    gpus=gpus,
                    oat_on_arc=True,
                )
                argv += [f"benchmark.{key}={value}" for key, value in settings.items()]
                if method in restored:
                    argv.append(f"ckpt_path={checkpoint}")
                write_json(evidence / f"{method}-arguments.json", argv)
                execute(argv, evidence / f"{method}-training.log")
            write_json(
                evidence / f"{method}-completion.json",
                verify_checkpoint(checkpoint, **validation),
            )
            if method == "tokenizer":
                write_json(
                    evidence / "status.json", {"state": "TOKENIZER_RECONSTRUCTION"}
                )
                argv = [
                    sys.executable,
                    "-m",
                    "egomimic.benchmarks.libero.cli",
                    "reconstruct",
                    "--checkpoint",
                    str(checkpoint),
                    "--dataset",
                    str(dataset),
                    "--suite",
                    args.suite,
                    "--output",
                    str(evidence / "tokenizer-reconstruction.json"),
                ]
                if args.mode == "smoke":
                    argv += ["--limit", "8", "--batch-size", "4"]
                execute(argv, evidence / "tokenizer-reconstruction.log")
        write_json(evidence / "status.json", {"state": "ROLLOUTS", "method": "arc_oat"})
        if args.mode == "full":
            from egomimic.benchmarks.libero.evaluate import parallel_rollouts

            parallel_rollouts(checkpoint, evidence, "arc_oat", args.suite)
        else:
            execute(
                [
                    sys.executable,
                    "-m",
                    "egomimic.benchmarks.libero.cli",
                    "rollout",
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(evidence / "arc_oat" / args.suite),
                    "--trials-per-task",
                    "1",
                    "--repetitions",
                    "1",
                    "--max-episode-steps",
                    "4",
                    "--video-trials",
                    "0",
                ],
                evidence / "arc-oat-rollout.log",
            )
        from egomimic.benchmarks.libero.report import (
            read_run,
            summarize,
            validate_full_protocol,
        )

        protocol, episodes = read_run(evidence / "arc_oat" / args.suite)
        if args.mode == "full":
            validate_full_protocol(protocol)
        if (
            protocol["method"] != "arc_oat"
            or protocol["representation"]["arc"]["mode"] != modes[0]
        ):
            raise ValueError("Evaluation used a different ARC+OAT representation")
        write_json(evidence / "scores.json", summarize(episodes))
        if reference and args.mode == "full":
            try:
                comparison = compare_reference(
                    uploader.client,
                    args.root,
                    evidence,
                    reference,
                    args.suite,
                    modes[0],
                )
            except ValueError as error:
                comparison = {"paired_validation_passed": False, "error": str(error)}
            else:
                comparison["paired_validation_passed"] = True
            write_json(evidence / "oat-comparison.json", comparison)
        write_json(
            evidence / "status.json",
            {
                "state": "ARC_OAT_COMPLETE"
                if args.mode == "full"
                else "ARC_OAT_SMOKE_PASSED",
                "episodes": len(episodes),
                "benchmark_performance": args.mode == "full",
            },
        )
    except BaseException as error:
        write_json(
            evidence / "status.json",
            {"state": "FAILED", "type": type(error).__name__, "error": str(error)},
        )
        raise
    finally:
        uploader.stop.set()
        uploader.thread.join()
        uploader.upload(final=True)
        uploader.upload(final=True)


if __name__ == "__main__":
    main()
