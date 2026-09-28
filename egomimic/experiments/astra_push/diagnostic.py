"""Post-evaluation historical-gradient score and discarded controlled branches."""

import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils._pytree import tree_map

from egomimic.experiments.astra_push.artifacts import named_seed, publish_json
from egomimic.experiments.astra_push.data import EpisodeWindows, collate_windows
from egomimic.experiments.astra_push.learner import read_checkpoint, tensor_hash
from egomimic.experiments.astra_push.partitions import config
from egomimic.pipeline.stages_masked_flow import MaskedFlowNoisingStage


def preconditioned_direction(parameters, optimizer, lookback):
    direction, steps = {}, {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            name = next(n for n, p in parameters.items() if p is parameter)
            moment = optimizer.state[parameter]
            if "exp_avg_sq" not in moment or "step" not in moment:
                raise ValueError(f"Missing Adam second moment: {name}")
            step = int(moment["step"])
            if step <= 0:
                raise ValueError("Adam step counter must include warm-start history")
            vhat = moment["exp_avg_sq"].float() / (1 - group["betas"][1] ** step)
            direction[name] = (
                group["lr"]
                / (vhat.sqrt() + group["eps"])
                * (lookback[name].to(parameter).float() - parameter.detach().float())
            ).detach()
            if not bool(torch.isfinite(direction[name]).all()):
                raise FloatingPointError("Nonfinite historical direction")
            steps[name] = step
    return direction, steps


def finite_difference_fixture():
    theta = torch.tensor([0.2, -0.4, 0.7], dtype=torch.float64, requires_grad=True)
    direction = torch.tensor([0.03, 0.1, -0.06], dtype=torch.float64)
    target = torch.tensor([0.3, 0.8, -0.2], dtype=torch.float64)

    def function(x):
        return ((x - target) ** 2).mean()

    derivative = (torch.autograd.grad(function(theta), theta)[0] * direction).sum()
    epsilon = 1e-6
    numerical = (
        function(theta + epsilon * direction) - function(theta - epsilon * direction)
    ) / (2 * epsilon)
    error = float(abs(derivative - numerical))
    if error > 1e-9:
        raise ValueError("Gradient dot finite-difference fixture failed")
    return {
        "kind": "deterministic_float64_autograd_fixture",
        "signed_dot": float(derivative),
        "central_difference": float(numerical.detach()),
        "absolute_error": error,
        "passed": True,
    }


@contextmanager
def fixed_draws(graph):
    stage = next(
        s for s in graph.pipeline.stages if isinstance(s, MaskedFlowNoisingStage)
    )
    original = (stage.noise_key, stage.time_key, stage.reads)
    stage.noise_key, stage.time_key = "diagnostic_noise", "diagnostic_time"
    stage.reads = ("target", stage.mask_key, stage.noise_key, stage.time_key)
    try:
        yield
    finally:
        stage.noise_key, stage.time_key, stage.reads = original


def batch_at(dataset, indices, *, draw_seed=None):
    batch = collate_windows([dataset[i] for i in indices])["libero_push"]
    batch = tree_map(
        lambda x: x.to("cuda") if isinstance(x, torch.Tensor) else x, batch
    )
    if draw_seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(draw_seed)
        batch["diagnostic_noise"] = torch.randn(
            batch["actions"].shape, device="cuda", generator=generator
        )
        values = 0.001 + 0.999 * np.random.default_rng(draw_seed).beta(
            1.5, 1.0, len(indices)
        )
        batch["diagnostic_time"] = torch.tensor(
            values, dtype=torch.float32, device="cuda"
        )
    return batch


def measurement_loss(learner, batch):
    learner.graph.nets.eval()
    # FP32 scoring/reductions, dropout disabled and BatchNorm buffers frozen.
    output = learner.graph.forward_training({"libero_push": batch})
    return learner.graph.compute_losses(output, {"libero_push": batch})["loss"]


def run_diagnostic(root, *, learner, final_checkpoint):
    root = Path(root)
    output = root / "diagnostic"
    output.mkdir(exist_ok=True)
    if (output / "results.json").exists():
        return json.loads((output / "results.json").read_text())
    cfg = config()["diagnostic"]
    final = read_checkpoint(final_checkpoint)
    back = read_checkpoint(json.loads((root / "checkpoints/A2/final.json").read_text()))
    if (back["total_updates"], final["total_updates"]) != (2000, 3000):
        raise ValueError(
            "Lookback must be A branch updates 1000 and 2000, including 1000 shared updates"
        )
    uniform = read_checkpoint(
        json.loads((root / "checkpoints/U4/final.json").read_text())
    )
    records = {r["sha256"]: r for r in final["records"] + uniform["records"]}
    dataset = EpisodeWindows(list(records.values()), learner.stats)
    rng = np.random.default_rng(named_seed("gradient_candidate_selection"))
    candidates = []
    for stage in ("S1", "S2", "S3"):
        pool = [
            i
            for i, r in enumerate(dataset.records)
            if r["provenance"]["stage"] == stage
            and r["provenance"]["arm"] in {"A", "common"}
        ]
        if len(pool) < cfg["candidates_per_stage"]:
            pool = [
                i
                for i, r in enumerate(dataset.records)
                if r["provenance"]["stage"] == stage
            ]
        ordered = sorted(
            pool,
            key=lambda i: (
                dataset.records[i]["provenance"]["task_hash"],
                dataset.records[i]["sha256"],
            ),
        )
        rng.shuffle(ordered)
        unique, remaining = [], []
        seen = set()
        for i in ordered:
            task = dataset.records[i]["provenance"]["task_hash"]
            (remaining if task in seen else unique).append(i)
            seen.add(task)
        for i in (unique + remaining)[: cfg["candidates_per_stage"]]:
            windows = sorted(
                int(dataset.offsets[i] + v)
                for v in rng.choice(
                    dataset.lengths[i], size=cfg["windows_per_candidate"], replace=False
                )
            )
            candidates.append(
                {
                    "episode_index": i,
                    "episode_hash": dataset.records[i]["sha256"],
                    "task_hash": dataset.records[i]["provenance"]["task_hash"],
                    "stage": stage,
                    "source_arm": dataset.records[i]["provenance"]["arm"],
                    "windows": windows,
                }
            )
    if len(candidates) != 12:
        raise ValueError("Not enough verified candidate episodes")
    excluded = {c["episode_index"] for c in candidates}
    transfer = []
    for stage in ("S1", "S2", "S3"):
        pool = [
            i
            for i, r in enumerate(dataset.records)
            if i not in excluded and r["provenance"]["stage"] == stage
        ]
        if not pool:
            raise ValueError("No disjoint training-only transfer pool")
        chosen = set()
        while len(chosen) < cfg["transfer_windows_per_stage"]:
            i = int(rng.choice(pool))
            chosen.add(int(dataset.offsets[i] + rng.integers(dataset.lengths[i])))
        transfer.extend(sorted(chosen))
    bank = {
        "candidates": candidates,
        "transfer_windows": transfer,
        "episode_hashes": [r["sha256"] for r in dataset.records],
        "selection_seed": named_seed("gradient_candidate_selection"),
        "candidate_unit": "verified task/reset episode; task hashes retained to expose repeated templates",
        "source_fallback": "use uniform pool if adaptive stage has fewer than four distinct accepted episodes",
        "draws_per_measurement": cfg["draws"],
    }
    if (output / "bank.json").exists():
        if json.loads((output / "bank.json").read_text()) != bank:
            raise ValueError("Diagnostic candidate bank changed")
    else:
        publish_json(output / "bank.json", bank)
    fixture = finite_difference_fixture()
    results = []
    for index, candidate in enumerate(candidates):
        path = output / f"candidate-{index:02d}.json"
        if path.exists():
            results.append(json.loads(path.read_text()))
            continue
        attempt_index = len(list(output.glob(f"candidate-{index:02d}-attempt-*.json")))
        publish_json(
            output / f"candidate-{index:02d}-attempt-{attempt_index:02d}.json",
            {
                "candidate": index,
                "attempt": attempt_index,
                "time": time.time(),
                "maximum_discarded_updates": cfg["branch_updates"],
                "earlier_uncommitted_attempts": attempt_index,
            },
        )
        started = time.monotonic()
        learner.restore(final_checkpoint)
        learner.configure_data(final["records"], final["phase"], arm="A", round_index=4)
        parameters = dict(learner.graph.nets.named_parameters())
        direction, steps = preconditioned_direction(
            parameters, learner.optimizer, back["model"]
        )
        buffer_before = tensor_hash(dict(learner.graph.nets.named_buffers()))
        signed, losses, norms = [], [], []

        def transfer_loss():
            values = []
            with torch.no_grad(), fixed_draws(learner.graph):
                for d in range(cfg["draws"]):
                    for offset in range(0, len(transfer), 4):
                        batch = batch_at(
                            dataset,
                            transfer[offset : offset + 4],
                            draw_seed=named_seed(f"transfer:{d}:{offset}"),
                        )
                        values.append(float(measurement_loss(learner, batch)))
            return float(np.mean(values))

        with fixed_draws(learner.graph):
            for draw in range(cfg["draws"]):
                learner.optimizer.zero_grad(set_to_none=True)
                loss = measurement_loss(
                    learner,
                    batch_at(
                        dataset,
                        candidate["windows"],
                        draw_seed=named_seed(f"candidate:{index}:{draw}"),
                    ),
                )
                gradients = torch.autograd.grad(
                    loss, tuple(parameters.values()), allow_unused=True
                )
                dot = torch.zeros((), dtype=torch.float64, device="cuda")
                norm = torch.zeros((), dtype=torch.float64, device="cuda")
                for (name, _), gradient in zip(
                    parameters.items(), gradients, strict=True
                ):
                    if gradient is not None:
                        dot += (gradient.double() * direction[name].double()).sum()
                        norm += gradient.double().square().sum()
                signed.append(float(dot))
                losses.append(float(loss))
                norms.append(float(norm.sqrt()))
        before = transfer_loss()
        if tensor_hash(dict(learner.graph.nets.named_buffers())) != buffer_before:
            raise ValueError("Score/transfer measurement changed running buffers")
        del direction, gradients
        # Restore *all* state again, so score measurements cannot influence the
        # candidate-only optimizer branch or its dropout/random streams.
        learner.restore(final_checkpoint)
        learner.configure_data(final["records"], final["phase"], arm="A", round_index=4)
        update_losses = []
        for _ in range(cfg["branch_updates"]):
            update_losses.append(
                learner.update(batch_at(dataset, candidate["windows"]))["loss"]
            )
        after = transfer_loss()
        result = {
            **candidate,
            "candidate_index": index,
            "compute_attempts": attempt_index + 1,
            "signed_dots": signed,
            "score_mean_absolute_dot": float(np.mean(np.abs(signed))),
            "loss": float(np.mean(losses)),
            "gradient_norm": float(np.mean(norms)),
            "transfer_loss_before": before,
            "transfer_loss_after": after,
            "transfer_improvement": before - after,
            "diagnostic_updates": cfg["branch_updates"],
            "update_losses": update_losses,
            "optimizer_parameter_step_range": [
                min(steps.values()),
                max(steps.values()),
            ],
            "seconds": time.monotonic() - started,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        }
        if not all(
            np.isfinite(result[k])
            for k in (
                "score_mean_absolute_dot",
                "loss",
                "gradient_norm",
                "transfer_loss_before",
                "transfer_loss_after",
            )
        ):
            raise FloatingPointError("Diagnostic produced invalid numbers")
        publish_json(path, result)
        results.append(result)
        print(
            json.dumps(
                {
                    "event": "gradient_candidate",
                    "index": index,
                    "stage": candidate["stage"],
                    "score": result["score_mean_absolute_dot"],
                    "transfer_improvement": before - after,
                }
            ),
            flush=True,
        )
    learner.restore(final_checkpoint)
    associations = {}
    for metric in ("score_mean_absolute_dot", "loss", "gradient_norm"):
        value = spearmanr(
            [r[metric] for r in results], [r["transfer_improvement"] for r in results]
        )
        associations[metric] = {
            "spearman_rho": float(value.statistic)
            if np.isfinite(value.statistic)
            else None,
            "p_value": float(value.pvalue) if np.isfinite(value.pvalue) else None,
        }
    summary = {
        "candidates": results,
        "associations": associations,
        "numerical_fixture": fixture,
        "total_discarded_updates": sum(r["diagnostic_updates"] for r in results),
        "online_generator_feedback": False,
        "policy_checkpoints_unchanged": True,
        "interpretation": "exploratory supervised-transfer association over twelve candidates, not policy improvement or an online reward",
    }
    publish_json(output / "results.json", summary)
    return summary
