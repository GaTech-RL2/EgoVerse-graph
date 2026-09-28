"""Fit OAT's FAST BPE, measure held-out replay, train, then release GPUs for eval."""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml

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


def fit_tokenizer(dataset_path, suite, output, *, decoded_cache=True):
    from egomimic.eval.libero_eval import action_metrics
    from egomimic.models.oat.fast import load_tokenizer
    from egomimic.models.oat.tokenizer.fast.processing_action_tokenizer import (
        UniversalActionProcessor,
    )
    from egomimic.models.oat.tokenizer.fast.tokenizer_wrapper import FASTTok
    from egomimic.rldb.zarr.libero_dataset import (
        EMBODIMENT,
        LiberoDataset,
        LiberoNormalizer,
        LiberoReplayResolver,
        keymap,
    )

    cfg = yaml.safe_load(
        (
            Path(__file__).parents[2] / "hydra_configs/benchmark/libero_fast.yaml"
        ).read_text()
    )
    resolver = LiberoReplayResolver(
        dataset_path,
        keymap(n_obs_steps=0, horizon=cfg["horizon"]),
        suite,
        decoded_cache=decoded_cache,
    )
    splits = {
        split: LiberoDataset._from_resolver(
            resolver, split, cfg["valid_ratio"], cfg["split_seed"]
        )
        for split in ("train", "valid")
    }
    normalizer = LiberoNormalizer()
    normalizer.populate_from_datasets({"libero_panda": splits["train"]})
    normalizer.infer_norm_from_dataset(splits["train"], "libero_panda")
    # The released TrainFASTTokWorkspace fits BPE on *raw* training chunks;
    # FASTTok normalizes before tokenization. Preserve and report that choice.
    if cfg["fit_units"] != "raw_actions":
        raise ValueError("Expected OAT's released FAST fitting convention")
    actions = [
        splits["train"][i]["actions"].numpy() for i in range(len(splits["train"]))
    ]
    if not actions or not all(np.isfinite(a).all() for a in actions):
        raise ValueError("FAST fitting requires finite training actions")
    processor = UniversalActionProcessor.fit(
        actions,
        scale=cfg["scale"],
        vocab_size=cfg["vocab_size"],
        time_horizon=cfg["horizon"],
        action_dim=cfg["action_dim"],
    )
    tokenizer = FASTTok.from_processor(processor, normalizer.tokenizer_context())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    artifact = {
        "format": "egoverse_fast_v1",
        "fast_tokenizer_config": tokenizer._native_config,
        "benchmark_data_context": normalizer.tokenizer_context(),
        "normalizer_state": normalizer.to_state(),
        "fit_configuration": cfg,
        "train_chunks": len(actions),
    }
    temporary = output / "fast-tokenizer.pt.tmp"
    torch.save(artifact, temporary)
    temporary.replace(output / "fast-tokenizer.pt")
    restored = load_tokenizer(output / "fast-tokenizer.pt")
    probe = torch.from_numpy(np.stack(actions[:8]))
    if restored.tokenize(probe) != tokenizer.tokenize(probe):
        raise ValueError("FAST artifact reload changes tokenization")
    del actions
    validation = splits["valid"]
    validation.set_norm_stats_from(normalizer)
    metrics, lengths, truncated, invalid, count = {}, [], 0, 0, 0
    for start in range(0, len(validation), 256):
        batch = torch.stack(
            [
                validation[i]["actions"]
                for i in range(start, min(start + 256, len(validation)))
            ]
        )
        tokens = tokenizer.tokenize(batch)
        lengths.extend(map(len, tokens))
        truncated += sum(len(seq) >= cfg["max_seq_len"] for seq in tokens)
        for seq in tokens:
            text = processor.bpe_tokenizer.decode(seq)
            invalid += len(text) != cfg["horizon"] * cfg["action_dim"]
        prediction = tokenizer.detokenize(tokens)
        target = normalizer.unnormalize({"actions": batch}, EMBODIMENT)["actions"]
        prediction = normalizer.unnormalize({"actions": prediction}, EMBODIMENT)[
            "actions"
        ]
        for key, value in action_metrics(prediction, target).items():
            metrics[key] = metrics.get(key, 0.0) + float(value) * len(batch)
        count += len(batch)
    report = {
        "suite": suite,
        "fit_configuration": cfg,
        "train_chunks": artifact["train_chunks"],
        "validation_chunks": count,
        "fitted_vocabulary": processor.bpe_tokenizer.vocab_size,
        "configured_vocabulary": processor.vocab_size,
        "token_count_percentiles": dict(
            zip(
                ("min", "p50", "p95", "p99", "max"),
                np.percentile(lengths, [0, 50, 95, 99, 100]).tolist(),
            )
        ),
        "policy_truncation_fraction": truncated / count,
        "invalid_reconstruction_fraction": invalid / count,
        "metrics": {key: value / count for key, value in metrics.items()},
        "data_context": normalizer.tokenizer_context(),
        "artifact_reload_tokens_equal": True,
        "normalizer_scope": "All replay frame limits, matching the OAT release; BPE uses training episodes only.",
    }
    write_json(output / "fast-reconstruction.json", report)
    return report


def validate_fast_checkpoint(payload, *, suite):
    """Reject mislabeled, non-release, or externally dependent FAST checkpoints."""
    from egomimic.models.oat.tokenizer.fast.tokenizer_wrapper import FASTTok

    model = payload["hyper_parameters"]["config_tree"]["model"]
    stages = model["pipeline"]["stages"]
    protocol = model["benchmark_protocol"]
    if (
        len(stages) != 1
        or stages[0]["_target_"] != "egomimic.pipeline.stages_fast.FASTPolicyStage"
        or protocol.get("action_representation") != "fast_dct_bpe"
        or protocol.get("suite") != suite
        or tuple(protocol.get(k) for k in ("horizon", "n_obs_steps", "n_action_steps"))
        != (32, 2, 16)
        or not payload.get("fast_tokenizer_config")
        or payload.get("oat_tokenizer_config")
        or payload.get("oat_input_representation")
    ):
        raise ValueError("Checkpoint is not the native released FAST policy")
    expected = dict(
        embed_dim=256,
        n_layers=4,
        n_heads=4,
        dropout=0.1,
        max_seq_len=128,
        temperature=1.0,
        topk=10,
    )
    if any(stages[0]["policy"].get(k) != v for k, v in expected.items()):
        raise ValueError("FAST backbone or sampling configuration differs")
    tokenizer = FASTTok(**payload["fast_tokenizer_config"])
    if (
        tokenizer._training_data_context != payload["benchmark_data_context"]
        or payload["benchmark_data_context"]["suite"] != suite
        or (
            tokenizer.vocab_size,
            tokenizer.fast_tok.scale,
            tokenizer.fast_tok.time_horizon,
            tokenizer.fast_tok.action_dim,
        )
        != (1024, 10, 32, 7)
    ):
        raise ValueError("FAST tokenizer configuration or training data differs")


def verify_checkpoint(path, *, suite, mode, complete=True):
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    validate_fast_checkpoint(payload, suite=suite)
    epochs = payload["loops"]["fit_loop"]["epoch_progress"]["current"]["completed"]
    target = 5001 if mode == "full" else 1
    if (
        epochs > target
        or (complete and epochs != target)
        or not payload.get("optimizer_states")
        or not payload.get("normalizer_state")
        or not payload.get("ema_state_dict")
        or payload["ema_num_updates"] != payload["global_step"]
    ):
        raise ValueError(
            "FAST checkpoint lacks the expected training/optimizer/EMA state"
        )
    if mode == "full":
        budget = payload["training_budget"]
        if budget["epochs"] != 5001 or budget["global_batch_size"] != 1024:
            raise ValueError("FAST requires 5001 epochs and global batch 1024")
        if complete and payload["global_step"] != budget["total_optimizer_steps"]:
            raise ValueError("FAST optimizer budget is incomplete")
    return {
        "epochs_completed": epochs,
        "global_step": payload["global_step"],
        "ema_num_updates": payload["ema_num_updates"],
        "training_budget": payload.get("training_budget"),
    }


def main():
    from egomimic.benchmarks.libero.baseline import smoke_policy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--suite", choices=TASKS, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--epochs", type=int, default=5001)
    args = parser.parse_args()
    if args.mode == "full" and args.epochs != 5001:
        raise ValueError("Full FAST requires 5001 epochs")
    gpus = int(os.environ["TRAINING_GPUS"])
    gpu_type = os.environ["BENCHMARK_GPU_TYPE"]
    validate_gpu_allocation(gpus, gpu_type)
    layout = training_layout(gpus, args.mode)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if commit != os.environ["SOURCE_COMMIT"]:
        raise ValueError("Unexpected FAST source revision")
    evidence = args.root / "evidence"
    evidence.mkdir(parents=True, exist_ok=False)
    uploader = ArtifactUploader(evidence, args.run_id)
    resume = os.environ.get("RESUME_FROM_RUN") or None
    epochs = 5001 if args.mode == "full" else 1
    write_json(
        evidence / "runtime.json",
        {
            "run_kind": "fast",
            "source_commit": commit,
            "suite": args.suite,
            "method": "fast",
            "mode": args.mode,
            "epochs": epochs,
            "global_batch_size": layout["global_batch_size"],
            "training_layout": layout,
            "gpu_type": gpu_type,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "resume_from_run": resume,
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
        checkpoint = evidence / "training/fast/checkpoints/last.ckpt"
        restored = (
            restore_checkpoints(
                uploader.client,
                resume,
                evidence,
                suite=args.suite,
                mode=args.mode,
                epochs=epochs,
                allow_partial=True,
                methods=("fast",),
            )
            if resume
            else {}
        )
        if restored:
            verify_checkpoint(
                checkpoint, suite=args.suite, mode=args.mode, complete=False
            )
            payload = torch.load(
                checkpoint, map_location="cpu", weights_only=False, mmap=True
            )
            torch.save(
                {
                    "format": "egoverse_fast_v1",
                    "fast_tokenizer_config": payload["fast_tokenizer_config"],
                    "benchmark_data_context": payload["benchmark_data_context"],
                },
                evidence / "fast-tokenizer.pt",
            )
            del payload
        else:
            write_json(evidence / "status.json", {"state": "FAST_FIT_AND_REPLAY"})
            fit_tokenizer(dataset, args.suite, evidence)
        if not restored and args.mode == "full":
            preflight = evidence / "preflight"
            preflight.mkdir()
            shutil.copyfile(
                evidence / "fast-tokenizer.pt", preflight / "fast-tokenizer.pt"
            )
            write_json(evidence / "status.json", {"state": "GPU_PREFLIGHT"})
            execute(
                training_arguments(
                    "fast", args.suite, dataset, preflight, "smoke", 1, gpus=gpus
                ),
                preflight / "training.log",
            )
            smoke = preflight / "training/fast/checkpoints/last.ckpt"
            proof = verify_checkpoint(smoke, suite=args.suite, mode="smoke")
            smoke_policy(smoke, preflight, "fast", args.suite)
            write_json(evidence / "gpu-preflight.json", {"passed": True, **proof})
        if not restored.get("fast", {}).get("complete"):
            write_json(
                evidence / "status.json", {"state": "TRAINING", "method": "fast"}
            )
            argv = training_arguments(
                "fast", args.suite, dataset, evidence, args.mode, epochs, gpus=gpus
            )
            if restored:
                argv.append(f"ckpt_path={checkpoint}")
            write_json(evidence / "training-command.json", argv)
            execute(argv, evidence / "training.log")
        proof = verify_checkpoint(checkpoint, suite=args.suite, mode=args.mode)
        write_json(evidence / "training-completion.json", proof)
        if args.mode == "smoke":
            smoke_policy(checkpoint, evidence, "fast", args.suite)
        write_json(
            evidence / "status.json",
            {"state": "TRAINING_COMPLETE", "method": "fast", **proof},
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
            "method": "fast",
            "epochs": 5001,
            "total_optimizer_steps": proof["training_budget"]["total_optimizer_steps"],
            "checkpoint": uploader.receipts["training/fast/checkpoints/last.ckpt"],
        }
        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False, mmap=True
        )
        if checkpoint_completion(payload, request) is None:
            raise ValueError("Final FAST checkpoint is incomplete")
        write_json(args.output / "evaluation-request.json", request)


if __name__ == "__main__":
    main()
