"""Validate the complete frozen pusher matrix and rebuild SR@80 CSV exports."""
import csv
import hashlib
import itertools
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REGIMES = ("clean", "cotrain")
MODELS = ("dp16", "xy_absdur_m56")
SEEDS = (42, 43, 44)
METRICS = ("sr80_pct", "peak_coverage_pct", "final_coverage_pct")


def write_csv(name, rows):
    with (ROOT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: round(v, 6) if isinstance(v, float) else v
                             for k, v in row.items()})


def aggregate(rows):
    count = sum(r["episodes"] for r in rows)
    successes = sum(r["successes_at_80"] for r in rows)
    result = {"episodes": count, "successes_at_80": successes}
    for metric in METRICS:
        result[metric] = sum(r[metric] * r["episodes"] for r in rows) / count
    assert math.isclose(result["sr80_pct"], 100 * successes / count, abs_tol=1e-9)
    return result


def main():
    source = json.loads((ROOT / "source-level-summaries.json").read_text())
    records = {}
    for item in source:
        match = re.fullmatch(r"pa1113-(clean|cotrain)-(42|43|44)-(dp16|xy_absdur_m56)", item["run"])
        assert match, item["run"]
        regime, seed, model = match.groups()
        seed, level = int(seed), item["level"]
        scores = item["scores"]
        assert scores["episodes"] == 50
        for metric in ("SR80", "mean_peak_coverage", "mean_final_coverage"):
            assert math.isfinite(scores[metric]) and 0 <= scores[metric] <= 1
        successes = round(scores["SR80"] * scores["episodes"])
        assert math.isclose(successes, scores["SR80"] * scores["episodes"], abs_tol=1e-8)
        key = (regime, model, seed, level)
        assert key not in records
        records[key] = {
            "training_regime": regime, "model": model, "training_seed": seed,
            "level": level, "episodes": 50, "successes_at_80": successes,
            "sr80_pct": 100 * scores["SR80"],
            "peak_coverage_pct": 100 * scores["mean_peak_coverage"],
            "final_coverage_pct": 100 * scores["mean_final_coverage"],
            "failure_count": scores["failure_count"], "run_id": item["run"],
            "source_uri": "s3://rldb/experiments/pusher-main-arrival-20261105/evaluation/"
                + item["run"] + f"-triangle-s{level}/level-{level:02d}-summary.json",
        }
    expected = set(itertools.product(REGIMES, MODELS, SEEDS, range(31)))
    assert set(records) == expected, (expected - set(records), set(records) - expected)
    assert sum(r["episodes"] for r in records.values()) == 18600
    by_seed = [records[key] for key in sorted(records)]
    write_csv("per_level_by_seed.csv", by_seed)

    comparison = []
    for regime, level in itertools.product(REGIMES, range(31)):
        groups = {model: [records[regime, model, seed, level] for seed in SEEDS]
                  for model in MODELS}
        stats = {model: aggregate(rows) for model, rows in groups.items()}
        row = {"training_regime": regime, "level": level,
               "training_seeds": "42;43;44", "episodes_per_model": 150}
        for model, label in (("dp16", "dp16"), ("xy_absdur_m56", "arc_abs_duration")):
            for key in ("successes_at_80", *METRICS):
                row[f"{label}_{key}"] = stats[model][key]
            row[f"{label}_sr80_seed_sd_pp"] = statistics.stdev(r["sr80_pct"] for r in groups[model])
        row["arc_minus_dp_sr80_pp"] = stats["xy_absdur_m56"]["sr80_pct"] - stats["dp16"]["sr80_pct"]
        row["arc_minus_dp_peak_coverage_pp"] = stats["xy_absdur_m56"]["peak_coverage_pct"] - stats["dp16"]["peak_coverage_pct"]
        comparison.append(row)
    assert len(comparison) == 62
    write_csv("per_level_comparison.csv", comparison)

    summary = []
    for regime, model in itertools.product(REGIMES, MODELS):
        row = {"training_regime": regime, "model": model, "training_seeds": "42;43;44"}
        for scope, levels in (("L0", [0]), ("L1_30", range(1, 31))):
            chosen = [records[regime, model, seed, level] for seed in SEEDS for level in levels]
            stats = aggregate(chosen)
            assert stats["episodes"] == (150 if scope == "L0" else 4500)
            for key, value in stats.items():
                row[f"{scope}_{key}"] = value
        summary.append(row)
    write_csv("summary.csv", summary)
    manifest = {
        "status": "COMPLETE", "primary_metric": "SR@80", "level_summaries": 372,
        "episodes": 18600, "comparison_rows": 62, "models": list(MODELS),
        "training_seeds": list(SEEDS), "levels": list(range(31)),
        "checks": ["exact model/regime/seed/level matrix", "50 episodes per seed-level",
                   "integer success counts", "finite bounded metrics", "weighted aggregate parity"],
        "files": {},
    }
    for name in ("source-level-summaries.json", "provenance.json", "build_results.py",
                 "per_level_by_seed.csv", "per_level_comparison.csv", "summary.csv", "README.md"):
        path = ROOT / name
        if path.exists():
            raw = path.read_bytes()
            manifest["files"][name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Validated: 372 complete level summaries, 18,600 episodes, 62 paired level rows.")
    for row in summary:
        print(row["training_regime"], row["model"],
              "L0 SR@80", round(row["L0_sr80_pct"], 2),
              "L1-30 SR@80", round(row["L1_30_sr80_pct"], 2))


if __name__ == "__main__":
    main()
