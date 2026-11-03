"""Checkpoint, W&B and station identity recorded with every rollout episode.

The rollout loop reads this once per loaded model, never per control tick.
Everything comes from files already stored beside the checkpoint (the bundle's
``resolved-config.yaml``, ``inference-config.yaml``, ``data-context.json``,
``SHA256SUMS`` and ``<checkpoint stem>.provenance.json``) and from the runtime
checkout, so a recorded episode names the exact model and W&B run that produced
it without anyone typing it in. The checkpoint itself is never re-hashed: its
SHA-256 comes from ``SHA256SUMS``, the provenance file or the file name.

A missing or unreadable sidecar is reported under ``errors``; it never prevents
a rollout or a recording.
"""

from __future__ import annotations

import hashlib
import json
import re
import socket
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import yaml

from egomimic.pipeline.inference_config import find_inference_config

MODEL_METADATA_SCHEMA = "rl2_yam.rollout_model.v1"
# Text sidecars stored verbatim in each episode (``rollout/model/<name>``).
MAX_DOCUMENT_BYTES = 4 * 2**20
_SHA_IN_NAME = re.compile(r"__sha256-([0-9a-f]{8,64})(?:$|__)")
# Lightning's ``epoch-E-step-S`` (alone or as the relayed ``__epoch-E-step-S__``
# part). Run names such as ``..._step120k`` are deliberately not parsed.
_EPOCH_STEP = re.compile(r"(?:^|__)epoch[-=](\d+)-step[-=](\d+)(?:$|__)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_text(path: Path) -> str:
    size = path.stat().st_size
    if size > MAX_DOCUMENT_BYTES:
        raise ValueError(f"{path.name} is {size} bytes; at most {MAX_DOCUMENT_BYTES}")
    return path.read_text()


def _sums_entry(checkpoint: Path) -> str | None:
    """The checkpoint's SHA-256 from a ``SHA256SUMS`` file beside it, if listed."""
    sums = checkpoint.parent / "SHA256SUMS"
    if not sums.is_file():
        return None
    for line in sums.read_text().splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2 and parts[1].lstrip("*") == checkpoint.name:
            if re.fullmatch(r"[0-9a-f]{64}", parts[0]):
                return parts[0]
    return None


def _provenance_path(checkpoint: Path) -> Path | None:
    for name in (f"{checkpoint.stem}.provenance.json", "provenance.json"):
        candidate = checkpoint.parent / name
        if candidate.is_file():
            return candidate
    return None


def _short_step(step: int | None) -> str | None:
    if step is None:
        return None
    return f"{step // 1000}k" if step and step % 1000 == 0 else str(step)


def _int_or_none(value) -> int | None:
    return value if type(value) is int else None


def wandb_identity(training: Mapping | None, provenance: Mapping | None = None) -> dict:
    """W&B entity/project/run of the training run, with a run URL when complete."""
    training = training if isinstance(training, Mapping) else {}
    logger = training.get("logger")
    wandb = logger.get("wandb") if isinstance(logger, Mapping) else None
    wandb = wandb if isinstance(wandb, Mapping) else {}
    artifact = (provenance or {}).get("artifact")
    artifact = artifact if isinstance(artifact, Mapping) else {}

    def pick(*values):
        for value in values:
            if isinstance(value, str) and value:
                return value
        return None

    entity = pick(wandb.get("entity"), artifact.get("wandb_entity"))
    project = pick(wandb.get("project"), artifact.get("wandb_project"))
    run_id = pick(wandb.get("id"), artifact.get("wandb_run_id"))
    tags = wandb.get("tags")
    return {
        "entity": entity,
        "project": project,
        "run_id": run_id,
        "name": pick(wandb.get("name"), training.get("name")),
        "group": pick(wandb.get("group")),
        "tags": [str(tag) for tag in tags] if isinstance(tags, list) else [],
        "url": (
            f"https://wandb.ai/{entity}/{project}/runs/{run_id}"
            if entity and project and run_id
            else None
        ),
    }


@lru_cache(maxsize=1)
def runtime_identity() -> dict:
    """The station host and the checkout this rollout process runs from."""
    root = Path(__file__).resolve().parents[2]
    result = {
        "host": socket.gethostname(),
        "code_dir": str(root),
        "code_commit": None,
        "code_dirty": None,
        "commit_source": None,
    }
    commit_file = root / "COMMIT"
    if commit_file.is_file():
        # Station runtimes made with ``git archive`` record their commit here.
        lines = commit_file.read_text().split()
        result.update(code_commit=lines[0] if lines else None, commit_source="COMMIT")
        return result
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return result
    result.update(code_commit=commit, code_dirty=bool(status), commit_source="git")
    return result


def model_metadata(policy: Mapping | None) -> tuple[dict, dict[str, str]]:
    """Return (metadata, documents) for the model a rollout policy config loads.

    ``metadata`` is a small JSON-safe mapping (checkpoint, artifact hashes, W&B
    identity, provenance, station checkout and a model-label hint). ``documents``
    holds the bundle's text sidecars, stored verbatim with each episode.
    """
    policy = policy if isinstance(policy, Mapping) else {}
    errors: list[str] = []
    documents: dict[str, str] = {}
    metadata: dict = {
        "schema": MODEL_METADATA_SCHEMA,
        "policy_kind": policy.get("kind"),
        "runtime": dict(runtime_identity()),
        "errors": errors,
    }
    checkpoint_value = policy.get("checkpoint")
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        metadata["label_hint"] = str(policy.get("kind") or "policy")
        return metadata, documents
    checkpoint = Path(checkpoint_value)
    name_sha = _SHA_IN_NAME.search(checkpoint.stem)
    epoch_step = _EPOCH_STEP.search(checkpoint.stem)
    info = {
        "path": str(checkpoint),
        "name": checkpoint.name,
        "sha256": None,
        "sha256_source": None,
        "sha256_prefix": name_sha.group(1) if name_sha else None,
        "epoch": int(epoch_step.group(1)) if epoch_step else None,
        "global_step": int(epoch_step.group(2)) if epoch_step else None,
        "size_bytes": None,
    }
    metadata["checkpoint"] = info
    try:
        info["size_bytes"] = checkpoint.stat().st_size
        sha = _sums_entry(checkpoint)
        if sha is not None:
            info.update(sha256=sha, sha256_source="SHA256SUMS")
    except OSError as error:
        errors.append(f"checkpoint: {error}")

    provenance = None
    path = _provenance_path(checkpoint)
    if path is not None:
        try:
            text = _read_text(path)
            provenance = json.loads(text)
            if not isinstance(provenance, dict):
                raise ValueError("provenance is not a JSON object")
            documents["provenance_json"] = text
            metadata["provenance"] = {"path": str(path), **provenance}
        except (OSError, ValueError) as error:
            errors.append(f"provenance: {error}")
            provenance = None
    artifact = (provenance or {}).get("artifact")
    artifact = artifact if isinstance(artifact, Mapping) else {}
    if info["sha256"] is None and isinstance(artifact.get("checkpoint_sha256"), str):
        info.update(sha256=artifact["checkpoint_sha256"], sha256_source="provenance")
    if info["global_step"] is None:
        info["global_step"] = _int_or_none(artifact.get("global_step"))
    if info["epoch"] is None:
        info["epoch"] = _int_or_none(artifact.get("epoch"))

    sources = dict(policy)
    if policy.get("kind") == "graph" and not policy.get("inference_config"):
        # Without an explicit path the graph loader reads the model-owned
        # artifact beside the checkpoint; record that same file.
        try:
            found = find_inference_config(checkpoint)
        except OSError as error:
            errors.append(f"inference_config: {error}")
            found = None
        if found is not None:
            sources["inference_config"] = str(found)

    training = None
    for key, document in (
        ("training_config", "resolved_config_yaml"),
        ("inference_config", "inference_config_yaml"),
        ("normalizer_path", None),
    ):
        value = sources.get(key)
        if not isinstance(value, str) or not value:
            metadata[key] = None
            continue
        path = Path(value)
        try:
            metadata[key] = {"path": str(path), "sha256": _sha256(path)}
            if document is not None:
                text = _read_text(path)
                documents[document] = text
                if key == "training_config":
                    training = yaml.safe_load(text)
        except (OSError, ValueError, yaml.YAMLError) as error:
            metadata[key] = {"path": str(path), "sha256": None}
            errors.append(f"{key}: {error}")

    training = training if isinstance(training, Mapping) else {}
    wandb = wandb_identity(training, provenance)
    metadata["wandb"] = wandb
    metadata["run"] = {
        "name": training.get("name") if isinstance(training.get("name"), str) else None,
        "seed": _int_or_none(training.get("seed")),
        "experiment": artifact.get("experiment"),
        "code_commit": ((provenance or {}).get("code") or {}).get("commit"),
    }
    base = (
        artifact.get("run")
        if isinstance(artifact.get("run"), str) and artifact.get("run")
        else wandb["name"] or checkpoint.stem.split("__", 1)[0]
    )
    short = _short_step(info["global_step"])
    metadata["label_hint"] = f"{base} @ {short}" if short else base
    return metadata, documents
