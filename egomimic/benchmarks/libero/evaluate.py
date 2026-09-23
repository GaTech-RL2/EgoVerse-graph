"""Evaluate one completed, content-addressed policy without waiting for training peers."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from egomimic.benchmarks.libero.catalog import TASKS
from egomimic.benchmarks.libero.cluster import (
    ArtifactUploader,
    configure_simulator,
    digest,
    execute,
    write_json,
)


def validate_request(request):
    """Pin the training source, immutable checkpoint and complete optimizer budget."""
    if (
        not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", request["source_run"])
        or not re.fullmatch(r"[0-9a-f]{40}", request["source_commit"])
        or request["suite"] not in TASKS
        or request["method"] not in ("oat", "arc", "arc_stk", "arc_dur")
        or request["epochs"] != 5001
        or type(request["total_optimizer_steps"]) is not int
        or request["total_optimizer_steps"] < 1
    ):
        raise ValueError("Invalid full policy evaluation request")
    receipt = request["checkpoint"]
    if (
        not re.fullmatch(r"[0-9a-f]{64}", receipt["sha256"])
        or type(receipt["bytes"]) is not int
        or receipt["bytes"] < 1
        or receipt["uri"]
        != f"s3://rldb/experiments/arc-oat-20260919/{request['source_run']}/"
        f"checkpoints/{receipt['sha256']}/last.ckpt"
    ):
        raise ValueError("Evaluation checkpoint must be pinned within its source run")


def checkpoint_completion(payload, request):
    """Return proof only after every epoch, optimizer update and EMA update finishes."""
    validate_request(request)
    budget = payload["training_budget"]
    expected = {
        "epochs": request["epochs"],
        "global_batch_size": 1024,
        "total_optimizer_steps": request["total_optimizer_steps"],
    }
    if any(budget.get(key) != value for key, value in expected.items()):
        raise ValueError("Evaluation checkpoint training budget differs")
    if (
        payload["benchmark_data_context"]["suite"] != request["suite"]
        or payload["hyper_parameters"]["config_tree"]["model"]["benchmark_protocol"][
            "suite"
        ]
        != request["suite"]
    ):
        raise ValueError("Evaluation checkpoint suite differs")
    epochs = payload["loops"]["fit_loop"]["epoch_progress"]["current"]["completed"]
    steps = payload["global_step"]
    ema = payload["ema_num_updates"]
    if epochs > request["epochs"] or steps > request["total_optimizer_steps"]:
        raise ValueError("Evaluation checkpoint exceeds the requested budget")
    if epochs < request["epochs"] or steps < request["total_optimizer_steps"]:
        return None
    if ema != steps or not payload.get("ema_state_dict"):
        raise ValueError("Completed checkpoint lacks matching EMA weights")
    if "normalizer_state" not in payload:
        raise ValueError("Evaluation checkpoint lacks its normalization state")
    if request["method"] == "oat" and not payload.get("oat_tokenizer_config"):
        raise ValueError("OAT evaluation requires a self-contained tokenizer")
    return {"epochs_completed": epochs, "global_step": steps, "ema_num_updates": ema}


def restore_policy(client, request, root, evidence):
    """Download the requested immutable object, independent of mutable last receipts."""
    import torch

    validate_request(request)
    prefix = f"experiments/arc-oat-20260919/{request['source_run']}/"
    runtime = json.loads(
        client.get_object(Bucket="rldb", Key=prefix + "runtime.json")["Body"].read()
    )
    expected = {
        "source_commit": request["source_commit"],
        "suite": request["suite"],
        "epochs": request["epochs"],
        "global_batch_size": 1024,
        "mode": "full",
    }
    if any(runtime.get(key) != value for key, value in expected.items()):
        raise ValueError("Evaluation source runtime differs from the request")
    receipt = request["checkpoint"]
    # Keep the recovered weights outside evidence: the immutable source is
    # already durable, so the uploader only needs the receipt and rollout data.
    path = Path(root) / "checkpoint" / "last.ckpt"
    path.parent.mkdir(parents=True, exist_ok=False)
    client.download_file("rldb", receipt["uri"].removeprefix("s3://rldb/"), str(path))
    if path.stat().st_size != receipt["bytes"] or digest(path) != receipt["sha256"]:
        raise ValueError("Evaluation checkpoint hash or size differs")
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    proof = checkpoint_completion(payload, request)
    if proof is None:
        raise ValueError("Cannot evaluate an incomplete training checkpoint")
    write_json(
        evidence / "recovered-training.json", {**request, **proof, "runtime": runtime}
    )
    return path


def main():
    from egomimic.benchmarks.libero.cluster import validate_gpu_allocation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    validate_request(request)
    validate_gpu_allocation(1, os.environ.get("BENCHMARK_GPU_TYPE", "L40S"))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if commit != os.environ["SOURCE_COMMIT"]:
        raise RuntimeError("Unexpected evaluation source revision")
    evidence = args.root / "evidence"
    evidence.mkdir(parents=True, exist_ok=False)
    uploader = ArtifactUploader(evidence, args.run_id)
    write_json(evidence / "evaluation-request.json", request)
    write_json(
        evidence / "runtime.json",
        {
            "source_commit": commit,
            "suite": request["suite"],
            "method": request["method"],
            "mode": "full",
            "run_kind": "policy_evaluation",
            "evaluate_from_run": request["source_run"],
            "gpu_type": os.environ.get("BENCHMARK_GPU_TYPE", "L40S"),
        },
    )
    uploader.thread.start()
    try:
        write_json(evidence / "status.json", {"state": "RESTORING_POLICY"})
        checkpoint = restore_policy(uploader.client, request, args.root, evidence)
        (args.root / "data").mkdir(exist_ok=True)
        configure_simulator(args.root)
        method, suite = request["method"], request["suite"]
        write_json(evidence / "status.json", {"state": "ROLLOUTS", "method": method})
        output = evidence / method / suite
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
            ],
            evidence / f"{method}-rollout.log",
        )
        from egomimic.benchmarks.libero.report import (
            read_run,
            summarize,
            validate_full_protocol,
        )

        protocol, records = read_run(output)
        validate_full_protocol(protocol)
        if (
            protocol["checkpoint_sha256"] != request["checkpoint"]["sha256"]
            or protocol["suite"] != suite
            or protocol["method"] != ("oat" if method == "oat" else "arc")
        ):
            raise ValueError("Rollout policy differs from the evaluation request")
        write_json(evidence / "scores.json", summarize(records))
        write_json(
            evidence / "status.json",
            {
                "state": "POLICY_EVALUATION_COMPLETE",
                "method": method,
                "episodes": len(records),
                "complete_protocol": True,
            },
        )
    except BaseException as error:
        write_json(
            evidence / "status.json",
            {
                "state": "FAILED",
                "type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    finally:
        uploader.stop.set()
        uploader.thread.join()
        uploader.upload(final=True)
        uploader.upload(final=True)


if __name__ == "__main__":
    main()
