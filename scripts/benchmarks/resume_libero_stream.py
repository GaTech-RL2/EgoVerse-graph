"""Resume a checksummed stream checkpoint against its unchanged training source.

This orchestration helper is embedded in the workflow. The checked-out model,
codec and trainer remain at the original SOURCE_COMMIT; the helper's own hash
is recorded separately. A missing or incompatible checkpoint never starts a
fresh training run.
"""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from egomimic.benchmarks.libero.arc_streams import (
    MANIFEST,
    campaign,
    configure_simulator,
    digest,
    execute,
    smoke_policy,
    source_receipt,
    stage_dataset,
    stream_training_arguments,
    training_layout,
    validate_replay_proof,
    validate_stream_config,
    verify_checkpoint,
    write_json,
)
from egomimic.benchmarks.libero.cluster import ArtifactUploader, validate_gpu_allocation
from egomimic.benchmarks.libero.evaluate import checkpoint_completion, validate_request


def validate_resume_payload(payload, request):
    """Require an epoch-boundary checkpoint with the complete training state."""
    validate_request(request)
    spec = campaign()
    suite, variant = request["suite"], request["arc_stream_variant"]
    mode = request["method"].removeprefix("arc_")
    representation = validate_stream_config(
        payload["hyper_parameters"]["config_tree"],
        suite=suite,
        variant=variant,
        arc_mode=mode,
    )
    budget = payload["training_budget"]
    total = spec["optimizer_steps"][suite]
    per_epoch = total // spec["epochs"]
    expected = {
        "epochs": spec["epochs"],
        "total_optimizer_steps": total,
        "global_batch_size": 1024,
        "world_size": 4,
        "microbatch_size": 256,
        "gradient_accumulation": 1,
        "optimizer_steps_per_epoch": per_epoch,
    }
    if (
        any(budget.get(k) != v for k, v in expected.items())
        or request["total_optimizer_steps"] != total
    ):
        raise ValueError("Resume training budget differs")
    loop = payload["loops"]["fit_loop"]
    progress = loop["epoch_progress"]["current"]
    serialized_epochs = progress["completed"]
    steps = payload["global_step"]
    epochs, remainder = divmod(steps, per_epoch)
    if not (
        0 < epochs < spec["epochs"]
        and remainder == 0
        and serialized_epochs in (epochs - 1, epochs)
    ):
        raise ValueError("Resume requires an incomplete epoch-boundary checkpoint")
    # Lightning's periodic callback saves after the final optimizer step but
    # before incrementing epoch_progress.completed. Preserve its loop state;
    # Trainer handles finishing that epoch when restoring the checkpoint.
    if serialized_epochs == epochs - 1:
        batches = loop.get("epoch_loop.batch_progress", {})
        if (
            not batches.get("is_last_batch")
            or batches.get("current", {}).get("completed") != per_epoch
            or progress.get("started") != epochs
            or progress.get("ready") != epochs
        ):
            raise ValueError("Periodic checkpoint is not at the last batch")
    if (
        payload["benchmark_data_context"]["suite"] != suite
        or not payload.get("optimizer_states")
        or "normalizer_state" not in payload
        or not payload.get("ema_state_dict")
        or payload.get("ema_num_updates") != steps
    ):
        raise ValueError("Resume lacks matched optimizer, normalization or EMA state")
    return {
        "global_step": steps,
        "epochs_completed": epochs,
        "serialized_epoch_completed": serialized_epochs,
        "ema_num_updates": steps,
        "representation": representation,
        "training_budget": budget,
    }


def restore_resume(client, request, root, commit):
    import torch

    validate_request(request)
    if request["source_commit"] != commit:
        raise ValueError("Resume must use the original training source")
    prefix = f"experiments/arc-oat-20260919/{request['source_run']}/"

    def read(name):
        obj = client.get_object(Bucket="rldb", Key=prefix + name)
        raw = obj["Body"].read()
        if hashlib.sha256(raw).hexdigest() != obj.get("Metadata", {}).get("sha256"):
            raise ValueError(f"Resume artifact checksum differs: {name}")
        return json.loads(raw)

    runtime = read("runtime.json")
    expected = {
        "source_commit": commit,
        "run_kind": "arc_streams",
        "operation": "train",
        "suite": request["suite"],
        "mode": "full",
        "epochs": 5001,
        "global_batch_size": 1024,
        "gpu_type": "L40S",
        "arc_stream_variant": request["arc_stream_variant"],
        "arc_mode": request["method"].removeprefix("arc_"),
        "sources": source_receipt(),
        "campaign_manifest_sha256": digest(MANIFEST),
        "training_layout": training_layout(4, "full"),
    }
    if any(runtime.get(k) != v for k, v in expected.items()):
        raise ValueError("Resume runtime source, representation or recipe differs")
    receipt = request["checkpoint"]
    receipts = read("checkpoint-receipts.json")
    if receipt not in receipts.values():
        raise ValueError("Resume checkpoint is absent from source receipts")
    preflight = read("gpu-preflight.json")
    if preflight.get("passed") is not True:
        raise ValueError("Original GPU preflight did not pass")
    checkpoint = Path(root) / "recovered" / "last.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=False)
    client.download_file(
        "rldb", receipt["uri"].removeprefix("s3://rldb/"), str(checkpoint)
    )
    if (
        checkpoint.stat().st_size != receipt["bytes"]
        or digest(checkpoint) != receipt["sha256"]
    ):
        raise ValueError("Resume checkpoint bytes or hash differ")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    proof = validate_resume_payload(payload, request)
    if preflight.get("representation") != proof["representation"]:
        raise ValueError("Original preflight representation differs")
    return checkpoint, runtime, proof


def main():
    import torch
    from hydra.utils import instantiate

    from egomimic.rldb.zarr.decoded_replay import prepare_decoded_replay
    from egomimic.rldb.zarr.libero_dataset import keymap

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--suite", choices=campaign()["suites"], required=True)
    parser.add_argument("--operation", choices=["train"], required=True)
    parser.add_argument("--mode", choices=["full"], required=True)
    parser.add_argument("--variant", choices=campaign()["variants"], required=True)
    parser.add_argument("--arc-mode", choices=["stk", "dur"], required=True)
    parser.add_argument("--replay-proof", type=Path, required=True)
    parser.add_argument("--resume-request", type=Path, required=True)
    args = parser.parse_args()
    if os.environ["BENCHMARK_GPU_TYPE"] != "L40S" or os.environ["TRAINING_GPUS"] != "4":
        raise ValueError("Recovery preserves the original four-L40S layout")
    validate_gpu_allocation(4, "L40S")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if commit != os.environ["SOURCE_COMMIT"]:
        raise ValueError("Unexpected training source")
    request = json.loads(args.resume_request.read_text())
    method = f"arc_{args.arc_mode}"
    if (request["suite"], request["arc_stream_variant"], request["method"]) != (
        args.suite,
        args.variant,
        method,
    ):
        raise ValueError("Resume request and task configuration differ")
    replay = json.loads(args.replay_proof.read_text())
    validate_replay_proof(replay, args.suite, commit)
    evidence = args.root / "evidence"
    evidence.mkdir(parents=True, exist_ok=False)
    args.output.mkdir(parents=True, exist_ok=True)
    uploader = ArtifactUploader(evidence, args.run_id)
    uploader.thread.start()
    try:
        write_json(evidence / "status.json", {"state": "RESTORING_TRAINING"})
        checkpoint, runtime, restored = restore_resume(
            uploader.client, request, args.root, commit
        )
        runtime.update(
            resume_from_run=request["source_run"],
            resume_helper_sha256=digest(__file__),
            gpu=torch.cuda.get_device_name(0),
            torch=torch.__version__,
        )
        write_json(evidence / "runtime.json", runtime)
        write_json(
            evidence / "recovered-training.json",
            {"request": request, "restored": restored},
        )
        write_json(evidence / "replay-completion.json", replay)
        write_json(evidence / "status.json", {"state": "STAGING_DATA"})
        dataset = stage_dataset(args.root, args.suite, evidence)
        cache = prepare_decoded_replay(
            dataset, [v["zarr_key"] for v in keymap().values()]
        )
        write_json(evidence / "decoded-replay-cache.json", cache.manifest)
        configure_simulator(args.root)
        preflight = evidence / "resume-preflight"
        preflight.mkdir()
        smoke_policy(checkpoint, preflight, args.suite)
        write_json(
            evidence / "gpu-preflight.json",
            {"passed": True, "kind": "resumed_checkpoint_inference", **restored},
        )
        argv = stream_training_arguments(
            args.suite,
            dataset,
            evidence,
            "full",
            variant=args.variant,
            arc_mode=args.arc_mode,
            gpus=4,
        )
        argv.append(f"ckpt_path={checkpoint}")
        write_json(evidence / "training-command.json", argv)
        write_json(
            evidence / "status.json",
            {"state": "TRAINING", "resumed_from_step": restored["global_step"]},
        )
        execute(argv, evidence / "training.log")
        final = evidence / f"training/{method}/checkpoints/last.ckpt"
        result = verify_checkpoint(
            final,
            suite=args.suite,
            variant=args.variant,
            arc_mode=args.arc_mode,
            mode="full",
        )
        payload = torch.load(final, map_location="cpu", weights_only=False, mmap=True)
        network = instantiate(
            payload["hyper_parameters"]["config_tree"]["model"]["pipeline"]["stages"][
                3
            ]["policy"]["model"]
        )
        result["action_parameters"] = sum(p.numel() for p in network.parameters())
        del network, payload
        write_json(evidence / "training-completion.json", result)
        write_json(evidence / "status.json", {"state": "TRAINING_COMPLETE", **result})
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
    evaluation = {
        "source_run": args.run_id,
        "source_commit": commit,
        "suite": args.suite,
        "method": method,
        "epochs": 5001,
        "arc_stream_variant": args.variant,
        "total_optimizer_steps": campaign()["optimizer_steps"][args.suite],
        "checkpoint": uploader.receipts[f"training/{method}/checkpoints/last.ckpt"],
    }
    if args.variant != "reference":
        evaluation["stream_reference_run"] = os.environ["ARC_STREAM_REFERENCE_RUN"]
    payload = torch.load(final, map_location="cpu", weights_only=False, mmap=True)
    if checkpoint_completion(payload, evaluation) is None:
        raise ValueError("Resumed final checkpoint remains incomplete")
    write_json(args.output / "evaluation-request.json", evaluation)


if __name__ == "__main__":
    main()
