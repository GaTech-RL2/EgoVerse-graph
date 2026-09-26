"""Plan, reconstruct, roll out and compare the complete ARC/OAT suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

from egomimic.benchmarks.libero.catalog import LIBERO_COMMIT, OAT_COMMIT, TASKS


def policy_method(stages, protocol):
    """Identify the actual graph; never label raw diffusion actions as ARC."""
    from egomimic.pipeline.stages_diffusion import DiffusionDenoiserStage
    from egomimic.pipeline.stages_libero_arc import LiberoArcStage
    from egomimic.pipeline.stages_oat import OATPolicyStage

    is_oat = any(isinstance(stage, OATPolicyStage) for stage in stages)
    is_arc = any(
        isinstance(stage, LiberoArcStage) and not stage.reconstruction
        for stage in stages
    )
    if is_oat or is_arc:
        return "arc_oat" if is_oat and is_arc else "oat" if is_oat else "arc"
    denoisers = [s for s in stages if isinstance(s, DiffusionDenoiserStage)]
    if (
        len(denoisers) == 1
        and denoisers[0].action_dim == 7
        and denoisers[0].action_horizon == protocol["horizon"]
        and protocol.get("action_representation") == "raw_actions"
        and protocol.get("dp_backbone") in ("unet", "oat_dp")
    ):
        return "dp_unet" if protocol["dp_backbone"] == "unet" else "dp_oat"
    raise ValueError("Expected a native ARC, OAT, ARC+OAT or raw DP policy")


def reconstruction(args):
    import numpy as np
    from torch.utils.data import DataLoader

    from egomimic.eval.libero_eval import action_metrics
    from egomimic.models.oat.factory import load_tokenizer
    from egomimic.rldb.zarr.libero_arc_timed import make_libero_arc_codec
    from egomimic.rldb.zarr.libero_dataset import (
        EMBODIMENT,
        LiberoDataset,
        LiberoNormalizer,
        LiberoReplayResolver,
        keymap,
    )

    tokenizer = (
        load_tokenizer(args.checkpoint, use_ema=not args.online).to(args.device).eval()
    )
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    normalizer = LiberoNormalizer(state=payload["normalizer_state"])
    arc_stages = None
    if payload.get("oat_input_representation") is not None:
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        from egomimic.eval.checkpoint_loading import strict_load_pipeline_checkpoint
        from egomimic.models.oat.checkpoint import validate_input_representation
        from egomimic.pipeline.stages_libero_arc import LiberoArcStage

        config = OmegaConf.create(payload["hyper_parameters"]["config_tree"])
        graph = instantiate(config.model.pipeline, device=args.device)
        graph.bind_data_context(normalizer=normalizer)
        strict_load_pipeline_checkpoint(graph, payload, use_ema=not args.online)
        validate_input_representation(graph.pipeline.stages, payload)
        arc_stages = [
            stage
            for stage in graph.pipeline.stages
            if isinstance(stage, LiberoArcStage)
        ]
        graph.nets.eval()
    context = normalizer.tokenizer_context()
    if args.suite != context["suite"]:
        raise ValueError(
            "Reconstruction suite differs from the tokenizer training suite"
        )
    horizon = int(
        payload["hyper_parameters"]["config_tree"]["model"]["benchmark_protocol"][
            "horizon"
        ]
    )
    resolver = LiberoReplayResolver(args.dataset, keymap(0, horizon), args.suite)
    dataset = LiberoDataset._from_resolver(
        resolver,
        mode="valid",
        valid_ratio=context["val_ratio"],
        split_seed=context["split_seed"],
    )
    check = LiberoNormalizer()
    check.populate_from_datasets({"libero_panda": dataset})
    check.infer_norm_from_dataset(dataset, "libero_panda")
    check.assert_tokenizer_context(context)
    dataset.set_norm_stats_from(normalizer)
    budgets = args.tokens
    if any(not 1 <= k <= tokenizer.latent_horizon for k in budgets):
        raise ValueError("OAT prefix budgets exceed the tokenizer length")
    codecs = {
        f"arc_{mode}_m{m}": make_libero_arc_codec(
            mode, num_waypoints=m, horizon=horizon
        )
        for mode in args.arc_modes
        for m in args.waypoints
    }
    if arc_stages:
        # Isolate the error introduced by learned quantization from ARC itself.
        codecs = {"arc_only": arc_stages[0].codec}
    learned_prefix = "arc_oat" if arc_stages else "oat"
    totals, count = {}, 0
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    with torch.inference_mode():
        for batch in loader:
            normalized = batch["actions"].to(args.device)
            if args.limit is not None:
                normalized = normalized[: max(0, args.limit - count)]
            if len(normalized) == 0:
                break
            target = normalizer.unnormalize({"actions": normalized}, EMBODIMENT)[
                "actions"
            ]
            token_input = (
                arc_stages[0].execute({"actions": normalized}, mode="inference")[
                    "target"
                ]
                if arc_stages
                else normalized
            )
            token_ids = tokenizer.tokenize(token_input)
            predictions = {}
            for k in budgets:
                decoded = tokenizer.detokenize(token_ids[:, :k])
                if arc_stages:
                    decoded = arc_stages[1].execute(
                        {"pred_arc": decoded}, mode="inference"
                    )["pred_action"]
                predictions[f"{learned_prefix}_k{k}"] = normalizer.unnormalize(
                    {"actions": decoded}, EMBODIMENT
                )["actions"]
            for name, codec in codecs.items():
                decoded = np.stack(
                    [codec.decode(codec.encode(row)) for row in target.cpu().numpy()]
                )
                predictions[name] = torch.from_numpy(decoded).to(target)
            for name, prediction in predictions.items():
                accumulator = totals.setdefault(name, {})
                for metric, value in action_metrics(prediction, target).items():
                    accumulator[metric] = accumulator.get(
                        metric, 0.0
                    ) + value.item() * len(target)
            count += len(target)
    if count == 0:
        raise ValueError("No held-out samples evaluated")
    sizes = {
        f"{learned_prefix}_k{k}": {
            "discrete_tokens": k,
            "ideal_bits": k * math.log2(tokenizer.quantizer.codebook_size),
        }
        for k in budgets
    }
    sizes.update(
        {
            name: {
                "float32_scalars": codec.num_waypoints
                * (12 if hasattr(codec, "mode") else 11),
                "bits": codec.num_waypoints
                * (12 if hasattr(codec, "mode") else 11)
                * 32,
            }
            for name, codec in codecs.items()
        }
    )
    return {
        "suite": args.suite,
        "data_context": context,
        "samples": count,
        "complete_validation": count == len(dataset),
        "oat_commit": OAT_COMMIT,
        "checkpoint_sha256": file_hash(args.checkpoint),
        "metrics": {
            name: {metric: value / count for metric, value in result.items()}
            for name, result in totals.items()
        },
        "representation_sizes": sizes,
        "oat_input_representation": payload.get("oat_input_representation"),
    }


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--output")
    rollout = commands.add_parser("rollout")
    rollout.add_argument("--checkpoint", required=True)
    rollout.add_argument("--output", required=True)
    rollout.add_argument("--device", default="cuda")
    rollout.add_argument("--online", action="store_true")
    rollout.add_argument("--tokens", type=int)
    rollout.add_argument("--trials-per-task", type=int, default=50)
    rollout.add_argument("--repetitions", type=int, default=5)
    rollout.add_argument("--repetition-index", type=int)
    rollout.add_argument("--start-seed", type=int, default=1000)
    rollout.add_argument("--max-episode-steps", type=int, default=550)
    rollout.add_argument("--video-trials", type=int, default=20)
    rollout.add_argument("--resume", action="store_true")
    recon = commands.add_parser("reconstruct")
    for option in ("checkpoint", "dataset", "suite", "output"):
        recon.add_argument("--" + option, required=True)
    recon.add_argument("--device", default="cuda")
    recon.add_argument("--online", action="store_true")
    recon.add_argument("--batch-size", type=int, default=256)
    recon.add_argument("--limit", type=int)
    recon.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 4, 8])
    recon.add_argument("--waypoints", type=int, nargs="+", default=[2, 4, 8, 16, 33])
    recon.add_argument(
        "--arc-modes",
        nargs="+",
        choices=("joint_dur", "stk", "dur"),
        default=["dur", "stk"],
    )
    compare = commands.add_parser("compare")
    compare.add_argument("--arc-root", required=True)
    compare.add_argument("--oat-root", required=True)
    compare.add_argument("--output", required=True)
    args = parser.parse_args()
    if getattr(args, "output", None) and Path(args.output).exists():
        raise FileExistsError(args.output)
    if args.command == "plan":
        from egomimic.benchmarks.libero.rollout import rollout_plan

        result = {
            "oat_commit": OAT_COMMIT,
            "libero_commit": LIBERO_COMMIT,
            "suites": {
                suite: {
                    "tasks": tasks,
                    "episodes_per_method": len(rollout_plan(suite)),
                    "plan": [asdict(spec) for spec in rollout_plan(suite)],
                }
                for suite, tasks in TASKS.items()
            },
        }
    elif args.command == "reconstruct":
        result = reconstruction(args)
    elif args.command == "rollout":
        from egomimic.benchmarks.libero.rollout import (
            load_policy,
            rollout_plan,
            run_rollouts,
        )
        from egomimic.pipeline.stages_libero_arc import LiberoArcStage
        from egomimic.pipeline.stages_oat import OATPolicyStage

        policy, protocol = load_policy(
            args.checkpoint,
            device=args.device,
            use_ema=not args.online,
            use_k_tokens=args.tokens,
        )
        stages = list(policy.algo.pipeline.stages)
        is_oat = any(isinstance(stage, OATPolicyStage) for stage in stages)
        is_arc = any(
            isinstance(stage, LiberoArcStage) and not stage.reconstruction
            for stage in stages
        )
        method = policy_method(stages, protocol)
        metadata = {
            **protocol,
            "method": method,
            "use_k_tokens": args.tokens,
            "checkpoint_sha256": file_hash(args.checkpoint),
            "data_context": policy.normalizer.tokenizer_context(),
            "observations_sha256": policy.normalizer.context["observations_sha256"],
            "use_ema": not args.online,
        }
        metadata["parameters"] = {
            "total": sum(p.numel() for p in policy.algo.nets.parameters()),
            "trainable": sum(
                p.numel() for p in policy.algo.nets.parameters() if p.requires_grad
            ),
        }
        if method.startswith("dp_"):
            if args.tokens is not None:
                raise ValueError("Raw diffusion policies do not use tokenizer prefixes")
            metadata["representation"] = {
                "kind": "raw actions",
                "horizon": protocol["horizon"],
                "float32_channels": 7,
            }
        if is_oat:
            model = next(
                stage.policy for stage in stages if isinstance(stage, OATPolicyStage)
            )
            metadata["representation"] = {
                "kind": "OAT FSQ",
                "codebook_size": model.action_tokenizer.quantizer.codebook_size,
                "prefix_tokens": args.tokens or model.max_seq_len,
            }
        if is_arc:
            codec = next(
                stage.codec for stage in stages if isinstance(stage, LiberoArcStage)
            )
            arc_representation = {
                "kind": "SE3 ARC",
                "mode": getattr(codec, "mode", "joint_dur"),
                "waypoints": codec.num_waypoints,
                "float32_channels": 12 if hasattr(codec, "mode") else 11,
                "dt": codec.dt,
                "max_translation": codec.max_translation,
                "max_rotation_degrees": codec.max_rotation_degrees,
                "independent_clocks": hasattr(codec, "mode"),
            }
            if not hasattr(codec, "mode"):
                arc_representation.update(
                    rotation_radius=codec.rotation_radius,
                    gripper_radius=codec.gripper_radius,
                )
            if is_oat:
                metadata["representation"].update(
                    kind="ARC + OAT FSQ", arc=arc_representation
                )
            else:
                metadata["representation"] = arc_representation
        if args.repetition_index is not None:
            metadata["evaluation_repetition"] = args.repetition_index
        run_rollouts(
            policy,
            rollout_plan(
                protocol["suite"],
                args.trials_per_task,
                args.repetitions,
                args.start_seed,
                repetition_index=args.repetition_index,
            ),
            args.output,
            max_episode_steps=args.max_episode_steps,
            video_trials=args.video_trials if args.repetition_index in (None, 0) else 0,
            metadata=metadata,
            resume=args.resume,
        )
        return
    else:
        from egomimic.benchmarks.libero.report import compare_suite

        result = compare_suite(args.arc_root, args.oat_root)
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        with Path(args.output).open("x") as handle:
            handle.write(encoded)
    else:
        print(encoded)


if __name__ == "__main__":
    main()
