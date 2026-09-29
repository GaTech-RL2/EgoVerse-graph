"""Train matched ARC decoder ablations and hand final EMA checkpoints to eval."""

import argparse
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

from egomimic.benchmarks.libero.arc_sweep import profile_settings
from egomimic.benchmarks.libero.catalog import TASKS
from egomimic.benchmarks.libero.cluster import (
    ArtifactUploader,
    arc_checkpoint_settings,
    configure_simulator,
    execute,
    load_arc_calibration,
    restore_checkpoints,
    stage_dataset,
    training_arguments,
    training_layout,
    validate_gpu_allocation,
    write_json,
)
from egomimic.models.arc_diffusion import (
    DECODER_VARIANTS,
    SHAPE_COLUMNS,
    TIMING_COLUMNS,
)


def compare_decoder_runs(candidate_directory, reference_directory):
    """Compare complete physical ARC policies only after exact episode pairing."""
    from egomimic.benchmarks.libero.report import (
        read_run,
        summarize,
        validate_full_protocol,
    )

    candidate, rows = read_run(candidate_directory)
    reference, old_rows = read_run(reference_directory)
    for protocol in (candidate, reference):
        validate_full_protocol(protocol)
        if protocol["method"] != "arc":
            raise ValueError("Decoder comparison requires physical ARC policies")
    for key in (
        "suite",
        "max_episode_steps",
        "n_obs_steps",
        "n_action_steps",
        "horizon",
        "data_context",
        "observations_sha256",
        "use_ema",
        "oat_commit",
        "libero_commit",
        "plan",
        "representation",
    ):
        if key not in candidate or candidate[key] != reference.get(key):
            raise ValueError(f"ARC decoder comparison protocol differs: {key}")
    if rows.keys() != old_rows.keys():
        raise ValueError("ARC decoder comparison episode identities differ")
    if any(
        rows[key]["initial_state_sha256"] != old_rows[key]["initial_state_sha256"]
        for key in rows
    ):
        raise ValueError("ARC decoder comparison initial states differ")
    actual_score, reference_score = summarize(rows), summarize(old_rows)
    return {
        "suite": candidate["suite"],
        "paired_episodes": len(rows),
        "initial_state_mismatches": 0,
        "complete_protocol": True,
        "candidate_variant": candidate.get("arc_decoder_variant", "joint_features"),
        "reference_variant": reference.get("arc_decoder_variant", "joint_features"),
        "candidate_checkpoint": candidate["checkpoint_sha256"],
        "reference_checkpoint": reference["checkpoint_sha256"],
        "candidate": actual_score,
        "reference": reference_score,
        "success_delta_percentage_points": 100
        * (actual_score["mean_success_rate"] - reference_score["mean_success_rate"]),
    }


def audit_reference(client, request, candidate_directory, evidence):
    """Download checksummed completed reference rollouts; never rerun/select trials."""
    reference = request["paired_reference"]
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", reference["run_id"]):
        raise ValueError("Invalid paired reference run")
    if reference["method"] not in ("arc_stk", "arc_dur"):
        raise ValueError("Paired reference must be a timed ARC policy")
    destination = Path(evidence) / "paired-reference"
    destination.mkdir(exist_ok=False)
    receipts = {}
    for name in ("protocol.json", "episodes.jsonl"):
        key = (
            f"experiments/arc-oat-20260919/{reference['run_id']}/"
            f"{reference['method']}/{request['suite']}/{name}"
        )
        obj = client.get_object(Bucket="rldb", Key=key)
        body = obj["Body"].read()
        sha = hashlib.sha256(body).hexdigest()
        if obj.get("Metadata", {}).get("sha256") != sha:
            raise ValueError(f"Paired reference checksum differs: {name}")
        (destination / name).write_bytes(body)
        receipts[name] = {"uri": "s3://rldb/" + key, "sha256": sha, "bytes": len(body)}
    return {
        **compare_decoder_runs(candidate_directory, destination),
        "reference_artifacts": receipts,
    }


def decoder_training_arguments(
    suite, dataset, evidence, mode, epochs, *, variant, profile, arc_mode, gpus=8
):
    if variant not in DECODER_VARIANTS:
        raise ValueError("Unknown ARC decoder variant")
    if mode == "full" and epochs != 5001:
        raise ValueError("Matched ARC decoder training requires 5001 epochs")
    settings = profile_settings(profile, arc_mode)
    args = training_arguments(
        f"arc_{arc_mode}",
        suite,
        dataset,
        evidence,
        mode,
        epochs,
        gpus=gpus,
        arc_backbone="oat_dp",
    )
    args[args.index("+experiment=oat/libero_arc_oat_dp_policy")] = (
        "+experiment=oat/libero_arc_stream_policy"
    )
    args.extend(f"benchmark.{key}={value}" for key, value in settings.items())
    args.extend(
        [
            f"benchmark.arc_decoder_variant={variant}",
            f"benchmark.arc_profile={profile}",
        ]
    )
    return args


def validate_decoder_config(
    config, *, suite, variant=None, profile=None, arc_mode=None
):
    """Require the declared split, topology, codec and released DP hyperparameters."""
    model = config["model"]
    protocol = model["benchmark_protocol"]
    declared_variant = protocol.get("arc_decoder_variant")
    declared_profile = protocol.get("arc_profile")
    settings = arc_checkpoint_settings(config, suite=suite)
    declared_mode = settings["arc_mode"]
    if (
        declared_variant not in DECODER_VARIANTS
        or (variant is not None and declared_variant != variant)
        or (profile is not None and declared_profile != profile)
        or (arc_mode is not None and declared_mode != arc_mode)
        or settings.pop("arc_backbone", None) != "oat_dp"
        or settings != profile_settings(declared_profile, declared_mode)
    ):
        raise ValueError("ARC decoder topology or frozen tokenizer profile differs")
    expected_protocol = {
        "arc_decoder_budget": "total_layers",
        "arc_decoder_total_layers": 4,
        "arc_decoder_width": 256,
        "arc_shape_columns": list(SHAPE_COLUMNS),
        "arc_timing_columns": list(TIMING_COLUMNS),
    }
    if any(protocol.get(key) != value for key, value in expected_protocol.items()):
        raise ValueError("ARC decoder parameter budget or channel partition differs")
    stages = model["pipeline"]["stages"]
    expected_stages = [
        "egomimic.pipeline.stages_oat.OATObservationStage",
        "egomimic.pipeline.stages_libero_arc.LiberoArcStage",
        "egomimic.pipeline.stages_diffusion.DiffusionNoisingStage",
        "egomimic.pipeline.stages_diffusion.DiffusionDenoiserStage",
        "egomimic.pipeline.stages_diffusion.DiffusionEpsilonLossStage",
        "egomimic.pipeline.stages_libero_arc.ArcPredictionName",
        "egomimic.pipeline.stages_libero_arc.LiberoArcStage",
    ]
    if [stage["_target_"] for stage in stages] != expected_stages:
        raise ValueError("ARC decoder experiment changed the shared policy graph")
    policy = stages[3]["policy"]
    network = policy["model"]
    expected_network = {
        "_target_": "egomimic.models.arc_diffusion.ArcStreamDiffusionTransformer",
        "decoder_variant": declared_variant,
        "input_dim": 12,
        "output_dim": 12,
        "horizon": settings["arc_waypoints"],
        "n_obs_steps": 2,
        "cond_dim": 138,
        "n_layer": 4,
        "n_head": 4,
        "n_emb": 256,
        "p_drop_emb": 0.1,
        "p_drop_attn": 0.1,
        "causal_attn": True,
        "time_as_cond": True,
        "obs_as_cond": True,
        "n_cond_layers": 0,
    }
    if network != expected_network or policy["num_inference_steps"] != 10:
        raise ValueError("ARC decoder backbone or inference budget differs")
    return {"variant": declared_variant, "profile": declared_profile, **settings}


def verify_checkpoint(path, *, suite, variant, profile, arc_mode, mode, complete=True):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    validate_decoder_config(
        payload["hyper_parameters"]["config_tree"],
        suite=suite,
        variant=variant,
        profile=profile,
        arc_mode=arc_mode,
    )
    if (
        payload["benchmark_data_context"]["suite"] != suite
        or not payload.get("optimizer_states")
        or "normalizer_state" not in payload
        or not payload.get("ema_state_dict")
        or payload["ema_num_updates"] != payload["global_step"]
    ):
        raise ValueError(
            "ARC decoder checkpoint lacks matching data/optimizer/EMA state"
        )
    epochs = payload["loops"]["fit_loop"]["epoch_progress"]["current"]["completed"]
    target_epochs = 5001 if mode == "full" else 1
    if epochs > target_epochs or (complete and epochs != target_epochs):
        raise ValueError("ARC decoder checkpoint epoch budget differs")
    if mode == "full":
        budget = payload["training_budget"]
        if budget["epochs"] != 5001 or budget["global_batch_size"] != 1024:
            raise ValueError("ARC decoders require 5001 epochs and global batch 1024")
        if complete and payload["global_step"] != budget["total_optimizer_steps"]:
            raise ValueError("ARC decoder optimizer budget is incomplete")
    return {
        "epochs_completed": epochs,
        "global_step": payload["global_step"],
        "ema_num_updates": payload["ema_num_updates"],
        "training_budget": payload.get("training_budget"),
        "arc_decoder_variant": variant,
    }


def smoke_policy(checkpoint, evidence, method, suite):
    from egomimic.benchmarks.libero.report import read_run

    # The rollout protocol calls every physical ARC policy "arc", while its
    # artifact directory includes the concrete STK/DUR codec mode.
    execute(
        [
            sys.executable,
            "-m",
            "egomimic.benchmarks.libero.cli",
            "rollout",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(evidence / method / suite),
            "--trials-per-task",
            "1",
            "--repetitions",
            "1",
            "--max-episode-steps",
            "4",
            "--video-trials",
            "0",
        ],
        evidence / "smoke-rollout.log",
    )
    protocol, records = read_run(evidence / method / suite)
    if protocol["method"] != "arc" or len(records) != len(TASKS[suite]):
        raise ValueError("ARC decoder simulator smoke coverage differs")


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--suite", choices=TASKS, required=True)
    parser.add_argument("--variant", choices=DECODER_VARIANTS, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--arc-mode", choices=("stk", "dur"), required=True)
    parser.add_argument("--replay-run")
    parser.add_argument("--reference-run")
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--epochs", type=int, default=5001)
    args = parser.parse_args()
    if args.mode == "full" and (args.epochs != 5001 or not args.replay_run):
        raise ValueError("Full ARC decoder runs require 5001 epochs and audited replay")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,57}", args.run_id):
        raise ValueError("Run ID must leave room for its evaluation suffix")
    expected = profile_settings(args.profile, args.arc_mode)
    gpus = int(os.environ["TRAINING_GPUS"])
    gpu_type = os.environ["BENCHMARK_GPU_TYPE"]
    validate_gpu_allocation(gpus, gpu_type)
    layout = training_layout(gpus, args.mode)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if commit != os.environ["SOURCE_COMMIT"]:
        raise ValueError("Unexpected ARC decoder source revision")
    evidence = args.root / "evidence"
    evidence.mkdir(parents=True, exist_ok=False)
    uploader = ArtifactUploader(evidence, args.run_id)
    resume = os.environ.get("RESUME_FROM_RUN") or None
    epochs = 5001 if args.mode == "full" else 1
    method = f"arc_{args.arc_mode}"
    write_json(
        evidence / "runtime.json",
        {
            "run_kind": "arc_decoders",
            "source_commit": commit,
            "suite": args.suite,
            "method": method,
            "mode": args.mode,
            "epochs": epochs,
            "global_batch_size": layout["global_batch_size"],
            "training_layout": layout,
            "gpu_type": gpu_type,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "resume_from_run": resume,
            "arc_decoder_variant": args.variant,
            "arc_profile": args.profile,
            "arc_settings": expected,
            "arc_backbone": "oat_dp",
            "arc_replay_run": args.replay_run,
            "paired_reference_run": args.reference_run,
            "decoder_parameters": 4_783_372 + expected["arc_waypoints"] * 256,
            "decoder_total_layers": 4,
            "decoder_layers_per_stream": 2 if args.variant == "separate" else 4,
            "evaluation_run": args.run_id + "-eval",
            "evaluation_workers": 5,
        },
    )
    validation = dict(
        suite=args.suite,
        variant=args.variant,
        profile=args.profile,
        arc_mode=args.arc_mode,
    )
    training = dict(
        variant=args.variant, profile=args.profile, arc_mode=args.arc_mode, gpus=gpus
    )
    uploader.thread.start()
    try:
        if args.mode == "full":
            replay_settings = load_arc_calibration(
                uploader.client,
                args.replay_run,
                evidence,
                suite=args.suite,
                arc_mode=args.arc_mode,
            )
            if any(
                expected.get(key) != value for key, value in replay_settings.items()
            ):
                raise ValueError(
                    "Audited replay differs from the selected ARC tokenizer"
                )
        write_json(evidence / "status.json", {"state": "STAGING_DATA"})
        dataset = stage_dataset(args.root, args.suite, evidence)
        from egomimic.rldb.zarr.decoded_replay import prepare_decoded_replay
        from egomimic.rldb.zarr.libero_dataset import keymap

        cache = prepare_decoded_replay(
            dataset, [v["zarr_key"] for v in keymap().values()]
        )
        write_json(evidence / "decoded-replay-cache.json", cache.manifest)
        configure_simulator(args.root)
        checkpoint = evidence / f"training/{method}/checkpoints/last.ckpt"
        restored = (
            restore_checkpoints(
                uploader.client,
                resume,
                evidence,
                suite=args.suite,
                mode=args.mode,
                epochs=epochs,
                allow_partial=True,
                methods=(method,),
            )
            if resume
            else {}
        )
        if restored:
            verify_checkpoint(checkpoint, mode=args.mode, complete=False, **validation)
        elif args.mode == "full":
            preflight = evidence / "preflight"
            preflight.mkdir()
            write_json(evidence / "status.json", {"state": "GPU_PREFLIGHT"})
            execute(
                decoder_training_arguments(
                    args.suite, dataset, preflight, "smoke", 1, **training
                ),
                preflight / "training.log",
            )
            smoke = preflight / f"training/{method}/checkpoints/last.ckpt"
            proof = verify_checkpoint(smoke, mode="smoke", **validation)
            smoke_policy(smoke, preflight, method, args.suite)
            write_json(evidence / "gpu-preflight.json", {"passed": True, **proof})
        if not restored.get(method, {}).get("complete"):
            write_json(
                evidence / "status.json", {"state": "TRAINING", "method": method}
            )
            argv = decoder_training_arguments(
                args.suite, dataset, evidence, args.mode, epochs, **training
            )
            if restored:
                argv.append(f"ckpt_path={checkpoint}")
            write_json(evidence / "training-command.json", argv)
            execute(argv, evidence / "training.log")
        proof = verify_checkpoint(checkpoint, mode=args.mode, **validation)
        write_json(evidence / "training-completion.json", proof)
        if args.mode == "smoke":
            smoke_policy(checkpoint, evidence, method, args.suite)
        write_json(
            evidence / "status.json",
            {"state": "TRAINING_COMPLETE", "method": method, **proof},
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
    if args.mode == "full":
        from egomimic.benchmarks.libero.evaluate import checkpoint_completion

        request = {
            "source_run": args.run_id,
            "source_commit": commit,
            "suite": args.suite,
            "method": method,
            "epochs": 5001,
            "total_optimizer_steps": proof["training_budget"]["total_optimizer_steps"],
            "checkpoint": uploader.receipts[f"training/{method}/checkpoints/last.ckpt"],
            "arc_decoder_variant": args.variant,
            "arc_profile": args.profile,
        }
        if args.reference_run:
            request["paired_reference"] = {
                "run_id": args.reference_run,
                "method": method,
            }
        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False, mmap=True
        )
        if checkpoint_completion(payload, request) is None:
            raise ValueError("Final ARC decoder checkpoint is incomplete")
        write_json(args.output / "evaluation-request.json", request)


if __name__ == "__main__":
    main()
