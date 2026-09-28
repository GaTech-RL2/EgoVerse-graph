"""Paired counts, template-cluster uncertainty and a self-contained video report."""

import base64
import html
import json
import os
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from egomimic.experiments.astra_push.artifacts import file_hash, publish_json
from egomimic.experiments.astra_push.partitions import config


def paired_outcomes(result):
    groups = defaultdict(list)
    for episode in result["episodes"]:
        groups[episode["pair_id"]].append(episode)
    pairs = []
    for pair_id, episodes in sorted(groups.items()):
        if (
            len(episodes) != 2
            or {e["instruction_index"] for e in episodes} != {0, 1}
            or len({e["initial_state_sha256"] for e in episodes}) != 1
            or len({e["policy_seed"] for e in episodes}) != 1
        ):
            raise ValueError(
                "Counterfactual pair does not share the exact initial state/noise seed"
            )
        pairs.append(
            {
                "pair_id": pair_id,
                "template_id": episodes[0]["template_id"],
                "stage": episodes[0]["stage"],
                "joint_success": all(e["metrics"]["success"] for e in episodes),
                "instruction_successes": sum(e["metrics"]["success"] for e in episodes),
                "initial_state_sha256": episodes[0]["initial_state_sha256"],
                "policy_seed": episodes[0]["policy_seed"],
            }
        )
    return pairs


def paired_cluster_summary(results, *, resamples=10000, seed=17):
    outcomes = {name: paired_outcomes(result) for name, result in results.items()}
    labels = list(outcomes)
    reference = outcomes[labels[0]]

    def identity(rows):
        return [
            (
                r["pair_id"],
                r["template_id"],
                r["initial_state_sha256"],
                r["policy_seed"],
            )
            for r in rows
        ]

    if any(identity(rows) != identity(reference) for rows in outcomes.values()):
        raise ValueError(
            "Compared checkpoints were evaluated on different states/noise seeds"
        )
    templates = sorted({p["template_id"] for p in reference})
    matrix = np.array(
        [
            [
                sum(
                    p["joint_success"] for p in outcomes[label] if p["template_id"] == t
                )
                for t in templates
            ]
            for label in labels
        ],
        dtype=float,
    )
    counts = np.array(
        [sum(p["template_id"] == t for p in reference) for t in templates]
    )
    draws = np.random.default_rng(seed).integers(
        len(templates), size=(resamples, len(templates))
    )
    boot = matrix[:, draws].sum(axis=2) / counts[draws].sum(axis=1)[None, :]
    model_metrics = {}
    for i, label in enumerate(labels):
        pairs = outcomes[label]
        model_metrics[label] = {
            "joint_successes": sum(p["joint_success"] for p in pairs),
            "pairs": len(pairs),
            "instruction_successes": sum(p["instruction_successes"] for p in pairs),
            "instructions": 2 * len(pairs),
            "joint_success_fraction": float(matrix[i].sum() / counts.sum()),
            "cluster_percentile_95": np.quantile(boot[i], [0.025, 0.975]).tolist(),
            "stages": {
                s: {
                    "joint_successes": sum(
                        p["joint_success"] for p in pairs if p["stage"] == s
                    ),
                    "pairs": sum(p["stage"] == s for p in pairs),
                }
                for s in ("S2", "S3")
            },
        }
    differences = {}
    for a, b in (
        ("A_final", "U_final"),
        ("A_final", "warm_start"),
        ("U_final", "warm_start"),
    ):
        if a in labels and b in labels:
            delta = boot[labels.index(a)] - boot[labels.index(b)]
            differences[f"{a}-minus-{b}"] = {
                "pair_count_difference": model_metrics[a]["joint_successes"]
                - model_metrics[b]["joint_successes"],
                "fraction_difference": model_metrics[a]["joint_success_fraction"]
                - model_metrics[b]["joint_success_fraction"],
                "cluster_percentile_95": np.quantile(delta, [0.025, 0.975]).tolist(),
            }
    return {
        "models": model_metrics,
        "differences": differences,
        "bootstrap": {
            "unit": "scene_instruction_template_cluster",
            "clusters": len(templates),
            "resamples": resamples,
            "seed": seed,
            "paired_across_checkpoints": True,
            "training_seed_uncertainty": False,
        },
        "raw_pairs": outcomes,
    }


def report_campaign(root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    output = root / "report"
    output.mkdir(exist_ok=True)
    if (output / "complete.json").exists():
        receipt = json.loads((output / "complete.json").read_text())
        if file_hash(output / "dashboard.zip") != receipt["zip_sha256"]:
            raise ValueError("Completed report archive changed")
        return json.loads((output / "results.json").read_text())
    labels = ["warm_start", "U_final", "A_final"]
    sealed = {
        label: json.loads(
            (root / "evaluations" / label / "sealed/results.json").read_text()
        )
        for label in labels
    }
    development = {
        label: json.loads(
            (root / "evaluations" / label / "development/results.json").read_text()
        )
        for label in labels
    }
    summary = paired_cluster_summary(
        sealed, resamples=config()["evaluation"]["bootstrap_resamples"]
    )
    summary["development"] = paired_cluster_summary(
        development, resamples=config()["evaluation"]["bootstrap_resamples"]
    )
    controls = {
        label: json.loads((root / "reports" / f"{label}.json").read_text())
        for label in ["common"] + [f"{a}{r}" for r in range(1, 5) for a in ("U", "A")]
    }
    losses = {}
    for phase in ["seed"] + [f"{a}{r}" for a in ("U", "A") for r in range(1, 5)]:
        losses[phase] = [
            v
            for p in sorted((root / "checkpoints" / phase).glob("block-*.json"))
            for v in json.loads(p.read_text())["updates"]
        ]
    decisions = {}
    for a in ("U", "A"):
        for r in range(1, 5):
            phase = f"{a}{r}"
            directory = root / "generation" / phase
            ready = json.loads((directory / "READY.json").read_text())
            exchange = directory / ready["exchange_directory"]
            if directory.resolve() not in exchange.resolve().parents:
                raise ValueError("Selected generation archive escapes its phase")
            if file_hash(exchange / "receipt.json") != ready["receipt_sha256"]:
                raise ValueError("Selected generation receipt changed")
            decisions[phase] = json.loads((exchange / "answer.json").read_text())
    episodes = sum(
        len(json.loads(p.read_text())["episodes"])
        for p in (root / "evaluations").glob("*/*/results.json")
    )
    if episodes != 708 or sum(len(v) for v in losses.values()) != 5000:
        raise ValueError(
            "Full report requires all 708 policy rollouts and 5000 main updates"
        )
    acquisitions = {
        phase: json.loads((root / "data" / phase / "manifest.json").read_text())
        for phase in losses
    }
    summary.update(
        evaluation_episodes=episodes,
        evaluation_attempts=sum(
            e["attempts"]
            for p in (root / "evaluations").glob("*/*/results.json")
            for e in json.loads(p.read_text())["episodes"]
        ),
        main_updates=5000,
        distinct_imitation_episodes=sum(
            len(v["accepted"]) for v in acquisitions.values()
        ),
        prior_commissioning_attempts=config()["prior_commissioning_attempts"],
        imitation_attempts=sum(v["attempts"] for v in acquisitions.values()),
        control_reports=controls,
        allocations={p: d["allocations"] for p, d in decisions.items()},
        source_manifest_hash=file_hash(root / "run-manifest.json"),
    )
    summary["signals"] = {
        "learning": {
            label: summary["models"][label]["joint_successes"]
            - summary["models"]["warm_start"]["joint_successes"]
            >= 6
            and all(
                summary["models"][label]["stages"][s]["joint_successes"] > 0
                for s in ("S2", "S3")
            )
            for label in ("U_final", "A_final")
        },
        "adaptive_pair_gain_at_least_five": summary["differences"][
            "A_final-minus-U_final"
        ]["pair_count_difference"]
        >= 5,
        "interpretation": "one training seed; teacher prior and frozen pretrained Astra generator; all HPT weights started random; custom AstraPush, not official LIBERO benchmark",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, stage in zip(axes, (None, "per_stage"), strict=True):
        if stage is None:
            values = [summary["models"][label]["joint_successes"] for label in labels]
            ax.bar(
                ["Warm start", "Uniform", "Adaptive"],
                values,
                color=["#7d8790", "#3478b8", "#db7d29"],
            )
            for i, v in enumerate(values):
                ax.text(i, v + 0.5, f"{v}/48", ha="center")
            ax.set(
                ylim=(0, 52),
                ylabel="Both instructions succeeded",
                title="Sealed counterfactual pairs",
            )
        else:
            for i, s in enumerate(("S2", "S3")):
                vals = [
                    summary["models"][label]["stages"][s]["joint_successes"]
                    for label in labels
                ]
                ax.bar(np.arange(3) + (i - 0.5) * 0.35, vals, width=0.35, label=s)
            ax.set(
                xticks=range(3),
                xticklabels=["Warm start", "Uniform", "Adaptive"],
                ylim=(0, 26),
                ylabel="Successful pairs / 24",
                title="Object choice and spatial reference",
            )
            ax.legend()
    fig.savefig(output / "paired-success.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    for ax, s in zip(axes, ("S1", "S2", "S3"), strict=True):
        for a in ("U", "A"):
            vals = [controls["common"]["stages"][s]["successes"]] + [
                controls[f"{a}{r}"]["stages"][s]["successes"] for r in range(1, 5)
            ]
            ax.plot(
                range(5), vals, marker="o", label={"U": "Uniform", "A": "Adaptive"}[a]
            )
        ax.set(
            title=s,
            xticks=range(5),
            xlabel="Completed branch round",
            ylabel="Control successes / 12",
            ylim=(-0.5, 12.5),
        )
        ax.legend()
    fig.savefig(output / "control-progress.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for a in ("U", "A"):
        rows = losses["seed"] + [v for r in range(1, 5) for v in losses[f"{a}{r}"]]
        values = np.array([v["loss"] for v in rows])
        smooth = np.convolve(values, np.ones(25) / 25, mode="valid")
        axes[0].plot(
            np.arange(len(smooth)) + 25,
            smooth,
            label={"U": "Uniform", "A": "Adaptive"}[a],
        )
    axes[0].set(
        title="Training loss (25-update mean)",
        xlabel="Updates including shared warm start",
        ylabel="Masked flow MSE",
    )
    axes[0].legend()
    bottom = np.zeros(4)
    for s in ("S1", "S2", "S3"):
        values = np.array([decisions[f"A{r}"]["allocations"][s] for r in range(1, 5)])
        axes[1].bar(range(1, 5), values, bottom=bottom, label=s)
        bottom += values
    axes[1].set(
        title="Astra's adaptive allocations",
        xlabel="Round",
        ylabel="Accepted demonstrations",
        xticks=range(1, 5),
        ylim=(0, 80),
    )
    axes[1].legend()
    fig.savefig(output / "loss-and-allocations.png", dpi=160)
    plt.close(fig)
    if (output / "results.json").exists():
        if json.loads((output / "results.json").read_text()) != summary:
            raise ValueError("Published results changed during report recovery")
    else:
        publish_json(output / "results.json", summary)
    cards = []
    for label in labels:
        for episode in sealed[label]["episodes"]:
            if episode["video_path"]:
                video = base64.b64encode(
                    Path(episode["video_path"]).read_bytes()
                ).decode()
                cards.append(
                    f'<article><h3>{html.escape(label)} · {html.escape(episode["pair_id"])} · instruction {episode["instruction_index"]}</h3><p>Success: {episode["metrics"]["success"]}; {episode["steps"]} commands</p><video controls preload="none" src="data:video/mp4;base64,{video}"></video></article>'
                )
    images = "".join(
        '<img alt="'
        + name
        + '" src="data:image/png;base64,'
        + base64.b64encode((output / name).read_bytes()).decode()
        + '">'
        for name in (
            "paired-success.png",
            "control-progress.png",
            "loss-and-allocations.png",
        )
    )
    rows = "".join(
        f'<tr><td>{html.escape(label)}</td><td>{summary["models"][label]["joint_successes"]}/48</td><td>{summary["models"][label]["instruction_successes"]}/96</td><td>{summary["models"][label]["cluster_percentile_95"]}</td></tr>'
        for label in labels
    )
    page = (
        f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AstraPush learner comparison</title><style>body{{font:16px/1.5 system-ui;max-width:1200px;margin:auto;padding:24px;background:#f2f5f8;color:#162938}}img,video{{width:100%}}article{{background:white;padding:16px;border-radius:8px}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}td,th{{padding:10px;text-align:left}}@media(max-width:700px){{.grid{{grid-template-columns:1fr}}}}</style><h1>AstraPush: adaptive versus uniform generation</h1><p>Fully random HPT; one training seed; 720 distinct imitation demonstrations; 5,000 main optimizer updates; 708 policy rollouts. Each final learner saw 420 demonstrations and 3,000 updates including the shared warm start. Executes one of ten predicted commands before replanning.</p><p>Primary metric: both complementary instructions succeed from the identical initial state. Intervals resample the 16 scene templates with all three resets together, paired across checkpoints; they do not measure training-seed variation. This is a custom pushing task family, not official LIBERO benchmark performance.</p><table><tr><th>Checkpoint</th><th>Joint pairs</th><th>Instructions</th><th>95% cluster interval (fraction)</th></tr>{rows}</table>{images}<h2>Paired rollout videos</h2><p>Two prospectively selected scene templates per stage, first reset, both instructions and all three checkpoints. External camera left, wrist camera right. All displayed videos are embedded.</p><div class="grid">'
        + "".join(cards)
        + "</div></html>"
    )
    (output / "dashboard.html").write_text(page)
    temporary_zip = output / f".dashboard-{time.time_ns()}.zip"
    with zipfile.ZipFile(
        temporary_zip, "x", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for name in (
            "dashboard.html",
            "results.json",
            "paired-success.png",
            "control-progress.png",
            "loss-and-allocations.png",
        ):
            archive.write(output / name, arcname=name)
    if (output / "dashboard.zip").exists():
        with zipfile.ZipFile(output / "dashboard.zip") as previous:
            if previous.testzip() is not None:
                raise ValueError("Existing report archive is corrupt")
            for name in previous.namelist():
                if previous.read(name) != (output / name).read_bytes():
                    raise ValueError("Existing report archive contents changed")
    else:
        os.link(temporary_zip, output / "dashboard.zip")
    temporary_zip.unlink()
    publish_json(
        output / "complete.json",
        {"zip_sha256": file_hash(output / "dashboard.zip")},
    )
    return summary
