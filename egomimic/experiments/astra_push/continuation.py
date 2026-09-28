"""Versioned ledger continuation with frozen scientific code and provenance."""

import argparse
import json
import re
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

from egomimic.experiments.astra_push.artifacts import file_hash, publish_json
from egomimic.experiments.astra_push.schemas import canonical_hash

ORCHESTRATION_FILES = {
    "egomimic/experiments/astra_push/campaign.py",
    "egomimic/experiments/astra_push/revisions.py",
    "egomimic/experiments/astra_push/continuation.py",
    "egomimic/experiments/astra_push/reporting.py",
}


def read(path):
    return json.loads(Path(path).read_text())


def active_ledger(root):
    root = Path(root)
    activations = sorted(root.glob("continuations/continuation-*/activation.json"))
    if not activations:
        return root / "ledger.sqlite"
    previous = None
    for path in activations:
        value = read(path)
        if value["previous_activation_sha256"] != previous:
            raise ValueError("Active continuation lineage changed")
        previous = file_hash(path)
        if value["receipt_sha256"] != file_hash(path.parent / "receipt.json"):
            raise ValueError("Active continuation receipt changed")
    receipt = path.parent / "receipt.json"
    if value["receipt_sha256"] != file_hash(receipt):
        raise ValueError("Active continuation receipt changed")
    ledger = (root / read(receipt)["ledger_path"]).resolve()
    if ledger.parent != path.parent.resolve():
        raise ValueError("Active ledger escapes its versioned continuation")
    return ledger


def source_audit(original, execution):
    repository = Path(__file__).resolve().parents[3]

    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(repository), *arguments])

    if git("rev-parse", "HEAD").decode().strip() != execution:
        raise ValueError("Continuation must execute its exact committed source")
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise ValueError("Continuation source has uncommitted tracked changes")
    changed = git("diff", "--name-only", original, execution).decode().splitlines()
    forbidden = [
        path
        for path in changed
        if path not in ORCHESTRATION_FILES and not path.startswith(("tests/", "docs/"))
    ]
    if forbidden:
        raise ValueError(f"Continuation changes frozen scientific code: {forbidden}")
    tracked = git("ls-tree", "-r", "--name-only", original).decode().splitlines()
    frozen = {}
    for path in tracked:
        if path in ORCHESTRATION_FILES or not path.startswith(
            ("egomimic/", "configs/", "scripts/astra_push/")
        ):
            continue
        before = git("rev-parse", f"{original}:{path}").decode().strip()
        after = git("rev-parse", f"{execution}:{path}").decode().strip()
        if before != after:
            raise ValueError(
                f"Frozen learner/simulator/controller/config changed: {path}"
            )
        frozen[path] = before
    return {
        "changed_files": changed,
        "frozen_git_blob_hashes": frozen,
        "frozen_code_hash": canonical_hash(frozen),
    }


def verify_paused_process(pause):
    process = Path("/proc") / str(pause["pid"])
    if (process / "stat").read_text().split()[21] != pause["process_starttime_ticks"]:
        raise ValueError("Paused coordinator PID was reused")
    command = [
        part.decode()
        for part in (process / "cmdline").read_bytes().split(b"\0")
        if part
    ]
    if command != pause["cmdline"] or str((process / "cwd").resolve()) != pause["cwd"]:
        raise ValueError("Paused coordinator identity changed")
    if "State:\tT" not in (process / "status").read_text():
        raise ValueError("Original coordinator must remain stopped during continuation")


def verify_ledger_prefix(parent, current):
    with (
        sqlite3.connect(f"file:{parent}?mode=ro", uri=True) as before,
        sqlite3.connect(f"file:{current}?mode=ro", uri=True) as after,
    ):
        before.row_factory = after.row_factory = sqlite3.Row
        for table in ("metadata", "phases", "transitions"):
            original = [tuple(row) for row in before.execute(f"SELECT * FROM {table}")]
            resumed = {tuple(row) for row in after.execute(f"SELECT * FROM {table}")}
            if not set(original) <= resumed:
                raise ValueError(f"Continuation changed original {table}")
        for row in before.execute("SELECT * FROM operations"):
            resumed = after.execute(
                "SELECT * FROM operations WHERE id=?", (row["id"],)
            ).fetchone()
            if resumed is None:
                raise ValueError("Continuation removed an incurred operation")
            protected = set(row.keys())
            if row["status"] == "reserved":
                protected -= {"status", "result", "completed_at"}
            if any(row[key] != resumed[key] for key in protected):
                raise ValueError(
                    "Continuation changed incurred operation identity or result"
                )


def prepare_continuation(
    root,
    *,
    name,
    parent_ledger,
    pause_receipt,
    execution_source_commit,
    validation_receipt,
):
    root = Path(root).resolve()
    if not re.fullmatch(r"continuation-\d{3}", name):
        raise ValueError("Continuation needs a versioned safe name")
    manifest = read(root / "run-manifest.json")
    original = manifest["source_commit"]
    audit = source_audit(original, execution_source_commit)
    validation = read(validation_receipt)
    if (
        validation.get("source_commit") != execution_source_commit
        or validation.get("conclusion") != "success"
    ):
        raise ValueError(
            "Continuation execution source requires its own successful validation"
        )
    pause = read(pause_receipt)
    verify_paused_process(pause)
    with sqlite3.connect(f"file:{parent_ledger}?mode=ro", uri=True) as database:
        database.row_factory = sqlite3.Row
        counts = [
            dict(row)
            for row in database.execute(
                "SELECT phase,kind,template,status,COUNT(*) AS count FROM operations "
                "GROUP BY phase,kind,template,status"
            )
        ]
        phases = [dict(row) for row in database.execute("SELECT * FROM phases")]
        if counts != pause["counts"] or phases != pause["phases"]:
            raise ValueError(
                "Continuation ledger snapshot differs from the paused costs/quotas"
            )
    pending_receipts = []
    for operation in pause["pending"]:
        receipt = root / "data" / operation["phase"] / operation["id"] / "receipt.json"
        if not receipt.is_file():
            raise ValueError(
                "The previously reserved simulator attempt has not finished"
            )
        pending_receipts.append(receipt)
    directory = root / "continuations" / name
    directory.mkdir(parents=True, exist_ok=False)
    initial = directory / "initial-ledger.sqlite"
    shutil.copyfile(parent_ledger, initial)
    ledger = directory / "ledger.sqlite"
    shutil.copyfile(initial, ledger)
    artifacts = [
        root / "run-manifest.json",
        root / "partition-definitions.json",
        root / "frozen-bank/manifest.json",
        root / "checkpoints/initial.json",
    ]
    artifacts.extend(root.glob("checkpoints/*/final.json"))
    artifacts.extend(root.glob("data/*/manifest.json"))
    artifacts.extend(pending_receipts)
    value = {
        "schema_version": "astrapush-continuation-1",
        "name": name,
        "created_at": time.time(),
        "source_commit": original,
        "execution_source_commit": execution_source_commit,
        "parent_ledger_path": str(Path(parent_ledger).resolve()),
        "parent_ledger_sha256": file_hash(parent_ledger),
        "initial_ledger_path": str(initial.relative_to(root)),
        "initial_ledger_sha256": file_hash(initial),
        "ledger_path": str(ledger.relative_to(root)),
        "pause_receipt_path": str(Path(pause_receipt).resolve()),
        "pause_receipt_sha256": file_hash(pause_receipt),
        "validation_receipt_path": str(Path(validation_receipt).resolve()),
        "validation_receipt_sha256": file_hash(validation_receipt),
        "frozen_artifacts": {
            str(path.relative_to(root)): file_hash(path) for path in sorted(artifacts)
        },
        "execution_source_audit": audit,
    }
    publish_json(directory / "receipt.json", value)
    return directory / "receipt.json"


def open_continuation(root, receipt_path, *, source_commit):
    root, receipt_path = Path(root).resolve(), Path(receipt_path).resolve()
    if root not in receipt_path.parents:
        raise ValueError("Continuation receipt escapes the experiment tree")
    value = read(receipt_path)
    if value["source_commit"] != source_commit:
        raise ValueError("Continuation original source differs from frozen manifest")
    execution = value["execution_source_commit"]
    if source_audit(source_commit, execution) != value["execution_source_audit"]:
        raise ValueError("Continuation scientific source audit changed")
    for key in ("parent_ledger", "pause_receipt", "validation_receipt"):
        if file_hash(value[f"{key}_path"]) != value[f"{key}_sha256"]:
            raise ValueError(f"Continuation {key} receipt changed")
    validation = read(value["validation_receipt_path"])
    if (
        validation.get("source_commit") != execution
        or validation.get("conclusion") != "success"
    ):
        raise ValueError("Execution source validation does not match continuation")
    verify_paused_process(read(value["pause_receipt_path"]))
    for path, digest in value["frozen_artifacts"].items():
        if file_hash(root / path) != digest:
            raise ValueError(f"Frozen continuation artifact changed: {path}")
    initial = root / value["initial_ledger_path"]
    ledger = root / value["ledger_path"]
    if initial.parent != receipt_path.parent or ledger.parent != receipt_path.parent:
        raise ValueError("Continuation ledger escapes its version directory")
    if file_hash(initial) != value["initial_ledger_sha256"]:
        raise ValueError("Original continuation ledger snapshot changed")
    verify_ledger_prefix(initial, ledger)
    activations = sorted(root.glob("continuations/continuation-*/activation.json"))
    activation = receipt_path.parent / "activation.json"
    if not activation.exists():
        if activations and receipt_path.parent.name <= activations[-1].parent.name:
            raise ValueError("Continuation versions must advance monotonically")
        publish_json(
            activation,
            {
                "receipt_sha256": file_hash(receipt_path),
                "previous_activation_sha256": file_hash(activations[-1])
                if activations
                else None,
                "activated_at": time.time(),
            },
        )
    if active_ledger(root) != ledger:
        raise ValueError("Cannot resume an inactive historical continuation")
    return ledger, execution


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--parent-ledger", required=True, type=Path)
    parser.add_argument("--pause-receipt", required=True, type=Path)
    parser.add_argument("--execution-source-commit", required=True)
    parser.add_argument("--validation-receipt", required=True, type=Path)
    print(prepare_continuation(**vars(parser.parse_args())))
