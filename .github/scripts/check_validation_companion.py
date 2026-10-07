"""Validate companion artifacts without executing the historical runtime tree."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
METADATA_PATHS = {
    ".github/workflows/ci.yml",
    ".github/scripts/check_validation_companion.py",
    ".github/validation-paths.json",
}
EVIDENCE_SOURCE_PIN_FIELDS = {
    "aidan-final-validation.json": (
        "existing_stack",
        "bc_donor",
        "rollout_donor",
        "bc_source_verified_before_validation_pin",
        "historical_m28",
        "pr193",
    ),
    "aidan-rollout-decoder-review.json": (
        "stack",
        "BC",
        "rollout_donor",
        "BC_child_commit",
        "PR193",
        "M28",
    ),
}


def git(*arguments):
    return subprocess.check_output(["git", "-C", str(ROOT), *arguments])


def full_commit(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError(f"{label} must be an immutable full commit SHA")
    return value


def regular_file(name):
    relative = PurePosixPath(name)
    if (
        not name
        or str(relative) != name
        or relative.is_absolute()
        or any(part in {"..", ".git"} for part in relative.parts)
        or "\\" in name
        or name.startswith("-")
    ):
        raise ValueError(f"Invalid companion path: {name}")
    path = ROOT / name
    if (
        not path.is_file()
        or path.is_symlink()
        or any(parent.is_symlink() for parent in path.parents if parent != ROOT)
    ):
        raise ValueError(f"Companion entry must be a regular file: {name}")
    return path


def classifier():
    spec = importlib.util.spec_from_file_location(
        "validation_manifest",
        ROOT / "scripts/integration/update_validation_manifest.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.is_companion_file


def check_source_pins():
    pins = []
    parity = json.loads((ROOT / "docs/integration/parity-manifest.json").read_text())
    for field in ("legacy_source", "graph_source"):
        pins.append((f"parity-manifest.{field}", parity[field]))
    consolidation = json.loads((ROOT / "docs/arc_consolidation.json").read_text())
    pins.append(("arc_consolidation.base", consolidation["base"]))
    pins.extend(consolidation["source_tips"].items())
    ledger = json.loads(
        (
            ROOT / "docs/integration/evidence/consolidation/source-pr-ledger.json"
        ).read_text()
    )
    for row in ledger:
        for field in ("head", "destination_commit"):
            pins.append((f"source-pr-ledger.PR{row['pr']}.{field}", row[field]))
    preservation = (
        ROOT / "docs/integration/evidence/aidan-bc-contract-preservation.json"
    )
    if preservation.exists():
        pins.append(
            (
                "aidan-bc.old_revision",
                json.loads(preservation.read_text())["old_revision"],
            )
        )
    reference = ROOT / ".github/validation-ref"
    if reference.exists():
        pins.append(("validation-ref", reference.read_text().strip()))
    for name, required in EVIDENCE_SOURCE_PIN_FIELDS.items():
        path = ROOT / "docs/integration/evidence" / name
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        values = record.get("source_pins") if isinstance(record, dict) else None
        if not isinstance(values, dict):
            raise ValueError(f"{name} requires a source_pins mapping")
        missing = sorted(set(required) - set(values))
        if missing:
            raise ValueError(f"{name} is missing source pins: {', '.join(missing)}")
        pins.extend(
            (f"{name}.source_pins.{field}", value) for field, value in values.items()
        )
    disposition = ROOT / "docs/integration/evidence/aidan-rollout-dispositions.json"
    if disposition.exists():
        record = json.loads(disposition.read_text())
        for field in ("source", "common_base", "runtime_parent"):
            if not isinstance(record, dict) or field not in record:
                raise ValueError(f"aidan-rollout-dispositions.json is missing {field}")
            pins.append((f"aidan-rollout-dispositions.{field}", record[field]))
    for label, value in pins:
        full_commit(value, label)
    return len(pins)


def check(base, head):
    full_commit(base, "base")
    full_commit(head, "head")
    if git("rev-parse", "HEAD").decode().strip() != head:
        raise ValueError("Companion validation must inspect the declared head checkout")
    is_companion = classifier()
    # Include tracked worktree edits and untracked artifacts for local preflight.
    changed = set(
        git("diff", "--name-only", "--no-renames", "-z", base, "--")
        .decode()
        .split("\0")
    )
    changed.update(
        git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")
    )
    invalid = sorted(
        name
        for name in changed
        if name and name not in METADATA_PATHS and not is_companion(name)
    )
    if invalid:
        raise ValueError("Companion PR changes runtime files: " + ", ".join(invalid))

    manifest = json.loads((ROOT / ".github/validation-paths.json").read_text())
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("Companion inventory must be a nonempty path-to-hash mapping")
    paths = (
        git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
        .decode()
        .split("\0")
    )
    expected = {name for name in paths if name and is_companion(name)}
    if expected != set(manifest):
        raise ValueError(
            "Companion inventory differs: missing="
            + repr(sorted(expected - set(manifest)))
            + ", extra="
            + repr(sorted(set(manifest) - expected))
        )
    syntax_files = 0
    for name, digest in manifest.items():
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid companion SHA-256: {name}")
        path = regular_file(name)
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"Companion hash mismatch: {name}")
        if path.suffix == ".py":
            ast.parse(content, filename=name)
            syntax_files += 1
    pins = check_source_pins()
    print(
        f"Validated {len(manifest)} companion artifacts, {syntax_files} Python syntax files, and {pins} immutable source pins. Runtime tests run on the runtime PR."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    arguments = parser.parse_args()
    check(arguments.base, arguments.head)


if __name__ == "__main__":
    main()
