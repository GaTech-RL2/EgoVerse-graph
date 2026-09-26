"""Train raw-action DP controls, then hand final checkpoints to GPU evaluation."""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from egomimic.benchmarks.libero.catalog import TASKS
from egomimic.benchmarks.libero.cluster import (
    ArtifactUploader,
    configure_simulator,
    execute,
    restore_checkpoints,
    stage_dataset,
    training_arguments,
    training_layout,
    validate_gpu_allocation,
    write_json,
)

METHODS = {"unet": "dp_unet", "oat_dp": "dp_oat"}


def validate_raw_dp_config(config, method):
    """Reject tokenizer/codec graphs or changes to the raw-action control."""
    backbone = {value: key for key, value in METHODS.items()}[method]
    model = config["model"]
    protocol = model["benchmark_protocol"]
    stages = model["pipeline"]["stages"]
    expected = (
        "egomimic.pipeline.stages_oat.OATObservationStage",
        "egomimic.pipeline.stages_io.ActionTargetBuilder",
        "egomimic.pipeline.stages_diffusion.DiffusionNoisingStage",
        "egomimic.pipeline.stages_diffusion.DiffusionDenoiserStage",
        "egomimic.pipeline.stages_diffusion.DiffusionEpsilonLossStage",
    )
    if (
        tuple(s["_target_"] for s in stages) != expected
        or protocol.get("action_representation") != "raw_actions"
        or protocol.get("dp_backbone") != backbone
        or tuple(protocol.get(k) for k in ("horizon", "n_obs_steps", "n_action_steps"))
        != (32, 2, 16)
        or stages[1].get("action_key") != "actions"
        or any(
            s.get("action_dim") != 7 or s.get("action_horizon") != 32
            for s in stages[2:4]
        )
    ):
        raise ValueError("Checkpoint is not the requested raw-action DP control")
    denoiser = stages[3]["policy"]
    wanted = (
        "egomimic.models.denoising_nets.ConditionalUnet1D"
        if backbone == "unet"
        else "egomimic.models.oat.diffusion.GraphDiffusionTransformer"
    )
    if (
        denoiser["model"]["_target_"] != wanted
        or denoiser["num_inference_steps"] != (100 if backbone == "unet" else 10)
        or denoiser["action_horizon"] != 32
    ):
        raise ValueError("Raw DP backbone or sampling configuration differs")


def verify_checkpoint(path, *, suite, method, mode, complete=True):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    config = payload["hyper_parameters"]["config_tree"]
    validate_raw_dp_config(config, method)
    if (
        payload["benchmark_data_context"]["suite"] != suite
        or config["model"]["benchmark_protocol"]["suite"] != suite
        or payload.get("oat_tokenizer_config")
        or payload.get("oat_input_representation")
        or not payload.get("optimizer_states")
        or "normalizer_state" not in payload
        or not payload.get("ema_state_dict")
        or payload["ema_num_updates"] != payload["global_step"]
    ):
        raise ValueError("Raw DP checkpoint lacks matching data/optimizer/EMA state")
    epochs = payload["loops"]["fit_loop"]["epoch_progress"]["current"]["completed"]
    target_epochs = 5001 if mode == "full" else 1
    if epochs > target_epochs or (complete and epochs != target_epochs):
        raise ValueError("Raw DP checkpoint epoch budget differs")
    if mode == "full":
        budget = payload["training_budget"]
        if budget["epochs"] != 5001 or budget["global_batch_size"] != 1024:
            raise ValueError("Raw DP requires 5001 epochs and global batch 1024")
        if complete and payload["global_step"] != budget["total_optimizer_steps"]:
            raise ValueError("Raw DP optimizer budget is incomplete")
    return {
        "epochs_completed": epochs,
        "global_step": payload["global_step"],
        "ema_num_updates": payload["ema_num_updates"],
        "training_budget": payload.get("training_budget"),
    }


def smoke_policy(checkpoint, evidence, method, suite):
    """Exercise real simulator observations and raw-action inference before training."""
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
    from egomimic.benchmarks.libero.report import read_run

    protocol, records = read_run(evidence / method / suite)
    if protocol["method"] != method or len(records) != len(TASKS[suite]):
        raise ValueError("Raw DP simulator smoke coverage differs")


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--suite", choices=TASKS, required=True)
    parser.add_argument("--backbone", choices=METHODS, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--epochs", type=int, default=5001)
    args = parser.parse_args()
    if args.mode == "full" and args.epochs != 5001:
        raise ValueError("Full DP controls require the released 5001 epochs")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,57}", args.run_id):
        raise ValueError("Run ID must leave room for its evaluation suffix")
    method = METHODS[args.backbone]
    gpus = int(os.environ["TRAINING_GPUS"])
    gpu_type = os.environ["BENCHMARK_GPU_TYPE"]
    validate_gpu_allocation(gpus, gpu_type)
    layout = training_layout(gpus, args.mode)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if commit != os.environ["SOURCE_COMMIT"]:
        raise ValueError("Unexpected raw DP source revision")
    evidence = args.root / "evidence"
    evidence.mkdir(parents=True, exist_ok=False)
    uploader = ArtifactUploader(evidence, args.run_id)
    resume = os.environ.get("RESUME_FROM_RUN") or None
    epochs = 5001 if args.mode == "full" else 1
    write_json(
        evidence / "runtime.json",
        {
            "run_kind": "dp_baseline",
            "source_commit": commit,
            "suite": args.suite,
            "method": method,
            "dp_backbone": args.backbone,
            "mode": args.mode,
            "epochs": epochs,
            "global_batch_size": layout["global_batch_size"],
            "training_layout": layout,
            "gpu_type": gpu_type,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "resume_from_run": resume,
            "action_representation": "raw_actions",
            "evaluation_run": args.run_id + "-eval",
            "evaluation_workers": 5,
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
        validation = dict(suite=args.suite, method=method, mode=args.mode)
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
            verify_checkpoint(checkpoint, complete=False, **validation)
        elif args.mode == "full":
            # Separate output/process and fresh initialization: these two updates
            # validate DDP, bf16, checkpoint reload and simulation, not the budget.
            preflight = evidence / "preflight"
            preflight.mkdir()
            write_json(evidence / "status.json", {"state": "GPU_PREFLIGHT"})
            argv = training_arguments(
                method, args.suite, dataset, preflight, "smoke", 1, gpus=gpus
            )
            execute(argv, preflight / "training.log")
            smoke = preflight / f"training/{method}/checkpoints/last.ckpt"
            proof = verify_checkpoint(
                smoke, suite=args.suite, method=method, mode="smoke"
            )
            smoke_policy(smoke, preflight, method, args.suite)
            write_json(evidence / "gpu-preflight.json", {"passed": True, **proof})
        if not restored.get(method, {}).get("complete"):
            write_json(
                evidence / "status.json", {"state": "TRAINING", "method": method}
            )
            argv = training_arguments(
                method, args.suite, dataset, evidence, args.mode, epochs, gpus=gpus
            )
            if restored:
                argv.append(f"ckpt_path={checkpoint}")
            write_json(evidence / "training-command.json", argv)
            execute(argv, evidence / "training.log")
        proof = verify_checkpoint(checkpoint, **validation)
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
        }
        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False, mmap=True
        )
        if checkpoint_completion(payload, request) is None:
            raise ValueError("Final checkpoint is incomplete")
        write_json(args.output / "evaluation-request.json", request)


if __name__ == "__main__":
    main()
