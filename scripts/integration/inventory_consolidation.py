"""Compose source-pinned trees and inventory the model/data/evaluator relationships.

Run from the integrated checkout with --source A=/checkout ... --output path.
Source directories are read only. Audit contexts are synthetic for fragments;
experiment rows retain their actual defaults. Hash equality is not GPU parity.
"""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import yaml
from omegaconf import OmegaConf

from scripts import audit_hydra_configs as audit


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def source_contexts(configs):
    contexts = {}
    for path in sorted((configs / "experiment").rglob("*.yaml")):
        experiment = path.relative_to(configs / "experiment").with_suffix("").as_posix()
        for default in yaml.safe_load(path.read_text()).get("defaults", []):
            if not isinstance(default, dict):
                continue
            for group, name in default.items():
                group = group.removeprefix("override ").strip("/")
                if group in {"model", "data", "evaluator"}:
                    contexts.setdefault(f"{group}/{name}", experiment)
    # Prefer the curated contexts when present in this source tree.
    contexts.update(
        {
            key: value
            for key, value in audit.EXPERIMENT_CONTEXTS.items()
            if (configs / f"experiment/{value}.yaml").exists()
        }
    )
    return contexts


def inventory(root, label, blobs):
    configs = root / "egomimic/hydra_configs"
    contexts = source_contexts(configs)
    curated = {
        key: value
        for key, value in audit.EXPERIMENT_CONTEXTS.items()
        if (configs / f"experiment/{value}.yaml").exists()
    }
    original_context = audit.audit_context

    def context(path):
        config, overrides = original_context(path)
        name = path.relative_to(configs).with_suffix("").as_posix()
        valid = []
        for override in overrides:
            if override.startswith("+experiment="):
                recipe = override.split("=", 1)[1]
                if not (configs / f"experiment/{recipe}.yaml").exists():
                    replacement = contexts.get(name)
                    if replacement:
                        valid.append(f"+experiment={replacement}")
                    continue
            valid.append(override)
        return config, valid

    records = []
    with (
        patch.object(audit, "CONFIGS", configs),
        patch.object(audit, "audit_context", context),
        patch.object(audit, "EXPERIMENT_CONTEXTS", curated),
    ):
        for path in sorted(configs.rglob("*.yaml")):
            relative = path.relative_to(configs).as_posix()
            if relative.split("/")[0] not in {
                "model",
                "data",
                "evaluator",
                "experiment",
            }:
                continue
            row = {
                "path": relative,
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "composition": context(path),
            }
            row["defaults"] = yaml.safe_load(path.read_text()).get("defaults", [])
            if relative.startswith("model/"):
                declared = yaml.safe_load(path.read_text())
                row["declared_model"] = digest(declared)
                blobs.setdefault(row["declared_model"], declared)
            try:
                with audit.compose_for_audit(path) as cfg:
                    row["selected"] = OmegaConf.to_container(
                        cfg.hydra.runtime.choices, resolve=True
                    )
                    for group in (
                        "model",
                        "data",
                        "evaluator",
                        "norm_stats",
                        "normalizer",
                        "hpt",
                        "abc",
                        "e1",
                        "arc_tokenizer",
                        "trainer",
                    ):
                        if group not in cfg:
                            continue
                        value = (
                            OmegaConf.to_container(
                                cfg[group], resolve=True, throw_on_missing=True
                            )
                            if OmegaConf.is_config(cfg[group])
                            else cfg[group]
                        )
                        key = digest(value)
                        blobs.setdefault(key, value)
                        row[group] = key
                    row["initialization"] = {
                        key: cfg.get(key)
                        for key in ("ckpt_path", "init_weights_from", "mode", "seed")
                    }
                    row["status"] = "resolved"
            except Exception as error:
                row.update(
                    status="source_error",
                    error=f"{type(error).__name__}: {error}".splitlines()[0],
                )
            records.append(row)
    return {
        "source": label,
        "sha": git(root, "rev-parse", "HEAD"),
        "counts": dict(Counter(row["path"].split("/")[0] for row in records)),
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", action="append", required=True, help="label=/absolute/checkout"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    OmegaConf.register_new_resolver(
        "now", lambda fmt: datetime(2026, 10, 1).strftime(fmt), replace=True
    )
    blobs, sources = {}, []
    for spec in args.source:
        label, path = spec.split("=", 1)
        result = inventory(Path(path).resolve(), label, blobs)
        sources.append(result)
        failures = [row for row in result["records"] if row["status"] != "resolved"]
        print(label, result["counts"], "source errors:", len(failures), flush=True)
        for row in failures:
            print(row["path"], row["error"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema": 1,
                "clock": "2026-10-01T00:00:00",
                "scope": "resolved offline specifications; no pretrained weights/data/GPU execution",
                "sources": sources,
                "content": blobs,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
