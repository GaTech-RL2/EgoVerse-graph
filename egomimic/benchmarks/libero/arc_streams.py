"""Paired reconstruction and fixed-budget LIBERO ARC stream experiments."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

import yaml

from egomimic.benchmarks.libero.catalog import TASKS
from egomimic.benchmarks.libero.cluster import (
    ArtifactUploader,
    configure_simulator,
    digest,
    execute,
    stage_dataset,
    training_arguments,
    training_layout,
    validate_gpu_allocation,
    write_json,
)
from egomimic.rldb.zarr.libero_arc_grouped import LiberoArcGroupedCodec

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "egomimic/hydra_configs/benchmark/libero_arc_streams.yaml"


def campaign():
    return yaml.safe_load(MANIFEST.read_text())


def variant_settings(variant, mode):
    spec = campaign()
    if variant not in spec["variants"] or mode not in spec["modes"]:
        raise ValueError("Unknown stream experiment or timing mode")
    path = ROOT / "egomimic/hydra_configs/arc_stream" / f"{variant}.yaml"
    row = yaml.safe_load(path.read_text())
    result = {**spec["geometry"], "mode": mode, "stream_spec": row["spec"]}
    codec = LiberoArcGroupedCodec(**result)
    if codec.action_dim != row["action_dim"] or row["name"] != variant:
        raise ValueError("Declared stream layout differs from its codec")
    return result


def candidates():
    spec = campaign()
    return {
        f"{variant}-{mode}": variant_settings(variant, mode)
        for variant in spec["variants"]
        for mode in spec["modes"]
    }


def source_receipt():
    paths = [
        "egomimic/rldb/zarr/libero_arc.py",
        "egomimic/rldb/zarr/libero_arc_timed.py",
        "egomimic/rldb/zarr/libero_arc_grouped.py",
        "egomimic/rldb/zarr/planar_arc.py",
        "egomimic/benchmarks/libero/replay.py",
        "egomimic/benchmarks/libero/arc_streams.py",
        "egomimic/hydra_configs/benchmark/libero_arc_streams.yaml",
    ]
    paths += [
        f"egomimic/hydra_configs/arc_stream/{v}.yaml" for v in campaign()["variants"]
    ]
    return {path: digest(ROOT / path) for path in paths}


def validate_replay_proof(proof, suite, commit):
    if (
        proof.get("source_commit") != commit
        or proof.get("suite") != suite
        or proof.get("sources") != source_receipt()
        or proof.get("candidates") != candidates()
        or proof.get("controls_passed") is not True
        or proof.get("episodes")
        != len(TASKS[suite]) * len(campaign()["replay"]["demos"])
    ):
        raise ValueError(
            "Stream replay proof has different source, geometry or coverage"
        )


def run_replay(args, evidence, commit):
    from egomimic.benchmarks.libero.replay import (
        replay_job,
        run_jobs,
        stage_raw_dataset,
        summarize,
        validate_controls,
    )

    spec = campaign()
    (args.root / "data").mkdir(parents=True, exist_ok=True)
    configure_simulator(args.root)
    raw = stage_raw_dataset(args.root, args.suite, evidence)
    # reconstruct_episode owns horizon/dt; avoid passing them twice there.
    selected = {
        name: {k: v for k, v in row.items() if k not in {"horizon", "dt"}}
        for name, row in candidates().items()
    }
    tested = {
        "raw": None,
        "raw_repeat": None,
        "source_precision_raw": None,
        "dense": {
            "num_waypoints": 33,
            "max_translation": None,
            "max_rotation_degrees": None,
        },
        **selected,
    }
    replay_spec = {
        **spec["geometry"],
        **spec["replay"],
        "execute_steps": spec["execute_steps"],
        "minimum_raw_success": 0.0,
    }
    jobs = [
        (str(raw / f"{task}_demo.hdf5"), demo, tested, replay_spec, args.suite)
        for task in TASKS[args.suite]
        for demo in spec["replay"]["demos"]
    ]
    with multiprocessing.get_context("spawn").Pool(
        spec["replay"]["workers"], maxtasksperchild=2
    ) as pool:
        rows = run_jobs(pool, replay_job, jobs, evidence, "paired-replay")
    summary = summarize(rows, tested)
    for key in selected:
        for field in (
            "translation_command_mse",
            "rotation_command_mse",
            "gripper_mismatch",
            "execution_coverage",
        ):
            summary[key][field] = sum(r["candidates"][key][field] for r in rows) / len(
                rows
            )
    write_json(evidence / "replay-results.json", summary)
    validate_controls(summary, replay_spec)
    # This is a fixed categorical comparison. Poor replay SR/coverage is a
    # result; it must not silently prune an unfavorable representation.
    proof = {
        "source_commit": commit,
        "suite": args.suite,
        "sources": source_receipt(),
        "candidates": candidates(),
        "controls_passed": True,
        "episodes": len(rows),
        "demonstrations": spec["replay"]["demos"],
        "use": "paired_reconstruction_not_parameter_selection",
        "results_sha256": digest(evidence / "replay-results.json"),
        "replay_run": args.run_id,
    }
    write_json(evidence / "replay-completion.json", proof)
    write_json(args.output / "replay-completion.json", proof)
    return proof


def stream_training_arguments(
    suite, dataset, evidence, mode, *, variant, arc_mode, gpus
):
    spec = campaign()
    variant_settings(variant, arc_mode)
    if spec["backbone"] != "unet":
        raise ValueError(
            "This campaign freezes the validated four-suite U-Net backbone"
        )
    args = training_arguments(
        f"arc_{arc_mode}",
        suite,
        dataset,
        evidence,
        mode,
        spec["epochs"] if mode == "full" else 1,
        gpus=gpus,
    )
    index = next(i for i, arg in enumerate(args) if arg.startswith("+experiment="))
    args[index] = "+experiment=oat/libero_arc_streams"
    args += [
        f"arc_stream={variant}",
        f"benchmark.arc_mode={arc_mode}",
        f"seed={spec['seed']}",
    ]
    if mode == "smoke":
        # Exercise the actual per-rank memory footprint before committing to
        # a long run. Smoke has two steps and a separate fresh initialization.
        args = [arg for arg in args if not arg.startswith("benchmark.batch_size=")]
        args.append(
            f"benchmark.batch_size={training_layout(gpus, 'full')['microbatch_size']}"
        )
    return args


def validate_stream_config(config, *, suite, variant, arc_mode):
    """Validate the resolved model tree actually serialized by ModelWrapper."""
    expected = variant_settings(variant, arc_mode)
    codec = LiberoArcGroupedCodec(**expected)
    protocol = config["model"]["benchmark_protocol"]
    geometry = {
        "arc_mode": arc_mode,
        "arc_action_dim": codec.action_dim,
        "arc_stream_spec": expected["stream_spec"],
        "arc_stream_variant": variant,
        "arc_waypoints": expected["num_waypoints"],
        "arc_max_translation": expected["max_translation"],
        "arc_max_rotation_degrees": expected["max_rotation_degrees"],
        "suite": suite,
        "horizon": 32,
        "n_obs_steps": 2,
        "n_action_steps": 16,
        "max_episode_steps": 550,
        "decoded_replay_cache": True,
    }
    if any(protocol.get(k) != v for k, v in geometry.items()):
        raise ValueError("Stream checkpoint geometry, timing or data recipe differs")
    if (
        protocol.get("arc_stream_variant") != variant
        or protocol.get("arc_stream_spec") != expected["stream_spec"]
        or protocol.get("arc_backbone") != campaign()["backbone"]
        or protocol.get("seed") != campaign()["seed"]
    ):
        raise ValueError("Stream checkpoint protocol or training seed differs")
    stages = config["model"]["pipeline"]["stages"]
    expected_stages = [
        "egomimic.pipeline.stages_oat.OATObservationStage",
        "egomimic.pipeline.stages_libero_arc.LiberoArcStage",
        "egomimic.pipeline.stages_diffusion.DiffusionNoisingStage",
        "egomimic.pipeline.stages_diffusion.DiffusionDenoiserStage",
        "egomimic.pipeline.stages_diffusion.DiffusionEpsilonLossStage",
        "egomimic.pipeline.stages_libero_arc.ArcPredictionName",
        "egomimic.pipeline.stages_libero_arc.LiberoArcStage",
    ]
    if [s["_target_"] for s in stages] != expected_stages:
        raise ValueError("Stream policy graph differs")
    for stage, operation in ((stages[1], "encode"), (stages[-1], "decode")):
        for key, value in {
            "operation": operation,
            "stream_spec": expected["stream_spec"],
            "arc_mode": arc_mode,
            "num_waypoints": expected["num_waypoints"],
            "horizon": 32,
            "max_translation": expected["max_translation"],
            "max_rotation_degrees": expected["max_rotation_degrees"],
        }.items():
            if stage.get(key) != value:
                raise ValueError(f"Stream encoder/decoder differs: {key}")
    for stage in stages[2:4]:
        if (
            stage.get("action_dim") != codec.action_dim
            or stage.get("action_horizon") != codec.num_waypoints
        ):
            raise ValueError("Stream model dimensions differ")
    network = stages[3]["policy"]["model"]
    if (
        network
        != {
            "_target_": "egomimic.models.denoising_nets.ConditionalUnet1D",
            "input_dim": codec.action_dim,
            "cond_dim": 276,
            "ac_latent_seq": 1,
            "diffusion_step_embed_dim": 256,
            "down_dims": [256, 512, 1024],
            "kernel_size": 3,
        }
        or stages[3]["policy"]["num_inference_steps"] != 100
    ):
        raise ValueError(
            "Stream experiment changed the fixed U-Net or denoising budget"
        )
    return codec.representation_context()


def verify_checkpoint(path, *, suite, variant, arc_mode, mode):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    representation = validate_stream_config(
        payload["hyper_parameters"]["config_tree"],
        suite=suite,
        variant=variant,
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
            "Stream checkpoint lacks matched normalization, optimizer or EMA state"
        )
    completed = payload["loops"]["fit_loop"]["epoch_progress"]["current"]["completed"]
    if completed != (campaign()["epochs"] if mode == "full" else 1):
        raise ValueError("Stream checkpoint epoch budget is incomplete")
    if mode == "full":
        budget = payload["training_budget"]
        for key, value in {
            "epochs": 5001,
            "global_batch_size": 1024,
            "total_optimizer_steps": campaign()["optimizer_steps"][suite],
        }.items():
            if budget.get(key) != value:
                raise ValueError(f"Stream checkpoint budget differs: {key}")
        if payload["global_step"] != budget["total_optimizer_steps"]:
            raise ValueError("Stream checkpoint optimizer budget is incomplete")
    return {
        "global_step": payload["global_step"],
        "ema_num_updates": payload["ema_num_updates"],
        "epochs_completed": completed,
        "training_budget": payload.get("training_budget"),
        "representation": representation,
    }


def smoke_policy(checkpoint, evidence, suite):
    from egomimic.benchmarks.libero.report import read_run

    output = evidence / "smoke-rollout"
    execute(
        [
            sys.executable,
            "-m",
            "egomimic.benchmarks.libero.cli",
            "rollout",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
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
    protocol, records = read_run(output)
    if protocol["method"] != "arc" or len(records) != len(TASKS[suite]):
        raise ValueError("Stream policy simulator smoke coverage differs")


def run_training(args, evidence, commit, uploader, gpus):
    import torch
    from hydra.utils import instantiate

    from egomimic.rldb.zarr.decoded_replay import prepare_decoded_replay
    from egomimic.rldb.zarr.libero_dataset import keymap

    if args.mode == "full":
        if args.replay_proof is None:
            raise ValueError("Full stream training requires paired replay")
        proof = json.loads(args.replay_proof.read_text())
        validate_replay_proof(proof, args.suite, commit)
        write_json(evidence / "replay-completion.json", proof)
    dataset = stage_dataset(args.root, args.suite, evidence)
    cache = prepare_decoded_replay(dataset, [v["zarr_key"] for v in keymap().values()])
    write_json(evidence / "decoded-replay-cache.json", cache.manifest)
    configure_simulator(args.root)
    validation = dict(suite=args.suite, variant=args.variant, arc_mode=args.arc_mode)
    method = f"arc_{args.arc_mode}"
    if args.mode == "full":
        preflight = evidence / "preflight"
        preflight.mkdir()
        write_json(
            evidence / "status.json",
            {"state": "GPU_PREFLIGHT", "variant": args.variant},
        )
        execute(
            stream_training_arguments(
                args.suite,
                dataset,
                preflight,
                "smoke",
                gpus=gpus,
                variant=args.variant,
                arc_mode=args.arc_mode,
            ),
            preflight / "training.log",
        )
        checkpoint = preflight / f"training/{method}/checkpoints/last.ckpt"
        proof = verify_checkpoint(checkpoint, mode="smoke", **validation)
        smoke_policy(checkpoint, preflight, args.suite)
        write_json(evidence / "gpu-preflight.json", {"passed": True, **proof})
    argv = stream_training_arguments(
        args.suite,
        dataset,
        evidence,
        args.mode,
        gpus=gpus,
        variant=args.variant,
        arc_mode=args.arc_mode,
    )
    write_json(evidence / "training-command.json", argv)
    write_json(evidence / "status.json", {"state": "TRAINING", "variant": args.variant})
    execute(argv, evidence / "training.log")
    checkpoint = evidence / f"training/{method}/checkpoints/last.ckpt"
    proof = verify_checkpoint(checkpoint, mode=args.mode, **validation)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    network = instantiate(
        payload["hyper_parameters"]["config_tree"]["model"]["pipeline"]["stages"][3][
            "policy"
        ]["model"]
    )
    proof["action_parameters"] = sum(p.numel() for p in network.parameters())
    del network
    write_json(evidence / "training-completion.json", proof)
    if args.mode == "smoke":
        smoke_policy(checkpoint, evidence, args.suite)
    return proof, checkpoint


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--suite", choices=campaign()["suites"], required=True)
    parser.add_argument("--operation", choices=("replay", "train"), required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--variant", choices=campaign()["variants"])
    parser.add_argument("--arc-mode", choices=("stk", "dur"))
    parser.add_argument("--replay-proof", type=Path)
    args = parser.parse_args()
    gpus = int(os.environ["TRAINING_GPUS"])
    if os.environ["BENCHMARK_GPU_TYPE"] != "L40S":
        raise ValueError("This campaign is authorized on L40S")
    validate_gpu_allocation(gpus, "L40S")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if commit != os.environ["SOURCE_COMMIT"]:
        raise ValueError("Stream workflow source commit differs")
    if args.operation == "train":
        variant_settings(args.variant, args.arc_mode)
    evidence = args.root / "evidence"
    evidence.mkdir(parents=True, exist_ok=False)
    args.output.mkdir(parents=True, exist_ok=True)
    uploader = ArtifactUploader(evidence, args.run_id)
    runtime = {
        "run_kind": "arc_streams",
        "source_commit": commit,
        "suite": args.suite,
        "mode": args.mode,
        "operation": args.operation,
        "gpu_type": "L40S",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "arc_stream_variant": args.variant,
        "arc_mode": args.arc_mode,
        "training_layout": training_layout(gpus, args.mode),
        "epochs": 5001 if args.mode == "full" else 1,
        "global_batch_size": training_layout(gpus, args.mode)["global_batch_size"],
        "campaign_manifest_sha256": digest(MANIFEST),
        "sources": source_receipt(),
    }
    write_json(evidence / "runtime.json", runtime)
    uploader.thread.start()
    result = None
    try:
        write_json(evidence / "status.json", {"state": "STAGING_DATA"})
        if args.operation == "replay":
            result = run_replay(args, evidence, commit)
        else:
            result, checkpoint = run_training(args, evidence, commit, uploader, gpus)
        write_json(
            evidence / "status.json",
            {
                "state": "REPLAY_COMPLETE"
                if args.operation == "replay"
                else "TRAINING_COMPLETE",
                **result,
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
    if args.operation == "train" and args.mode == "full":
        from egomimic.benchmarks.libero.evaluate import checkpoint_completion

        method = f"arc_{args.arc_mode}"
        request = {
            "source_run": args.run_id,
            "source_commit": commit,
            "suite": args.suite,
            "method": method,
            "epochs": 5001,
            "total_optimizer_steps": result["training_budget"]["total_optimizer_steps"],
            "arc_stream_variant": args.variant,
            "checkpoint": uploader.receipts[f"training/{method}/checkpoints/last.ckpt"],
        }
        if args.variant != "reference":
            request["stream_reference_run"] = os.environ["ARC_STREAM_REFERENCE_RUN"]
        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False, mmap=True
        )
        if checkpoint_completion(payload, request) is None:
            raise ValueError("Final stream checkpoint is incomplete")
        write_json(args.output / "evaluation-request.json", request)


if __name__ == "__main__":
    main()
