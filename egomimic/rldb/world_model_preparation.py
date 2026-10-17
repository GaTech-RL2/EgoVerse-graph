"""Finalize immutable feature receipts into train-only prepared world-model data.

Raw feature caches stay unchanged. Statistics are fitted by streaming training
frames, then applied to selected frame slices by PreparedWorldDataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from egomimic.rldb.zarr.episode_split import split_dataset_names


class ArtifactVerificationMemo:
    """Reuse exact verified files, including fingerprints in a trusted manifest.

    Persistent fingerprints are created only after a successful content hash.
    They are a fast path for immutable files on the same filesystem, not an
    independent authenticity proof: the prepared manifest must remain trusted.
    Copies, relocations, metadata changes, and old manifests fall back to SHA.
    """

    _STAT_FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    _FINGERPRINT_SCHEMA = "mot-verified-artifact-stat-v1"
    _MINIMUM_CTIME_AGE_NS = 1_000_000_000

    def __init__(self):
        self.verified = {}
        self.fingerprints = {}

    @staticmethod
    def _now_ns():
        return time.time_ns()

    @staticmethod
    def identity(path):
        stat = path.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    @classmethod
    def _fingerprint(cls, path, expected, identity, verification_started_ns):
        return {"schema": cls._FINGERPRINT_SCHEMA, "path": str(path),
                "sha256": expected, "verification_started_ns": verification_started_ns,
                **dict(zip(cls._STAT_FIELDS, identity))}

    @classmethod
    def _matches_fingerprint(cls, fingerprint, path, expected, identity):
        if not isinstance(fingerprint, dict):
            return False
        # Reject incomplete/extra fields and bool-as-int metadata. A malformed
        # cache hint must never silently turn off content verification.
        if any(type(fingerprint.get(key)) is not int for key in cls._STAT_FIELDS):
            return False
        started = fingerprint.get("verification_started_ns")
        if (type(started) is not int or started > cls._now_ns()
                or started - identity[-1] < cls._MINIMUM_CTIME_AGE_NS):
            return False
        return fingerprint == cls._fingerprint(path, expected, identity, started)

    def verify(self, path, expected, *, trusted_fingerprint=None):
        path = Path(path).resolve(strict=True)
        before = self.identity(path)
        started = self._now_ns()
        if self.verified.get((path, expected)) == before:
            return self.fingerprints[(path, expected)].copy()
        if self._matches_fingerprint(trusted_fingerprint, path, expected, before):
            self.verified[(path, expected)] = before
            self.fingerprints[(path, expected)] = trusted_fingerprint.copy()
            return trusted_fingerprint.copy()
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if before != self.identity(path):
            raise ValueError(f"Artifact changed while hashing: {path}")
        if actual != expected:
            raise ValueError(f"Artifact hash mismatch: {path}")
        # A same-clock-tick rewrite can preserve ctime as well as restored mtime.
        # Only author reusable proof when ctime was already old BEFORE hashing.
        # Merely waiting after a recent hash must never turn it into a proof.
        if started - before[-1] < self._MINIMUM_CTIME_AGE_NS:
            self.verified.pop((path, expected), None)
            self.fingerprints.pop((path, expected), None)
            return None
        fingerprint = self._fingerprint(path, expected, before, started)
        self.verified[(path, expected)] = before
        self.fingerprints[(path, expected)] = fingerprint
        return fingerprint.copy()


def training_identities(episodes):
    return sorted(
        [ep["embodiment"], ep["episode_id"], ep["sha256"]]
        for ep in episodes if ep["split"] == "train"
    )


def load_normalization(manifest, root, verifier):
    contract = manifest["normalization"]
    if contract.get("arrays_normalized") is True and not contract.get("path"):
        return None  # Historical pre-normalized cache contract.
    if not contract.get("path"):
        raise ValueError("Raw caches require a bound normalization statistics file")
    path = root / contract["path"]
    verifier.verify(path, contract["sha256"])
    stats = json.loads(path.read_text())
    if (stats.get("schema") != "mot-world-normalization-v1"
            or stats.get("scope") != "train_only"
            or stats.get("train_episodes") != training_identities(manifest["episodes"])):
        raise ValueError("Normalization statistics do not match training episode identities")
    keys = {"image", "track"} | {f"pose/{name}" for name in manifest["pose_schemas"]}
    if set(stats.get("statistics", {})) != keys:
        raise ValueError("Normalization statistics have incomplete modality/pose schemas")
    arrays = {}
    for key, item in stats["statistics"].items():
        mean, scale = np.asarray(item["mean"], np.float32), np.asarray(item["std"], np.float32)
        if (mean.ndim != 1 or not len(mean) or mean.shape != scale.shape
                or not np.isfinite(mean).all() or not np.isfinite(scale).all()
                or np.any(scale <= 0) or item.get("count", 0) < 1):
            raise ValueError(f"Invalid normalization statistics: {key}")
        expected = (3 if key == "track" else
                    manifest["pose_schemas"][key[5:]]["dimension"] if key.startswith("pose/") else None)
        if expected is not None and len(mean) != expected:
            raise ValueError(f"Normalization schema width mismatch: {key}")
        arrays[key] = (mean, scale)
    return arrays if contract["arrays_normalized"] is False else None


class StreamingMoments:
    def __init__(self):
        self.count, self.mean, self.m2 = 0, None, None

    def update(self, values):
        values = np.asarray(values, np.float64)
        values = values.reshape(-1, values.shape[-1])
        if not len(values):
            return
        if not np.isfinite(values).all():
            raise ValueError("Non-finite supervised values while fitting normalization")
        mean = values.mean(0)
        m2 = ((values - mean) ** 2).sum(0)
        self.merge({"count": len(values), "mean": mean, "m2": m2})

    def merge(self, summary):
        count = summary["count"]
        mean, m2 = np.asarray(summary["mean"], np.float64), np.asarray(summary["m2"], np.float64)
        if (type(count) is not int or count < 1 or mean.ndim != 1 or not len(mean)
                or mean.shape != m2.shape or not np.isfinite(mean).all()
                or not np.isfinite(m2).all() or np.any(m2 < 0)):
            raise ValueError("Invalid per-episode normalization moments")
        if self.count == 0:
            self.count, self.mean, self.m2 = count, mean.copy(), m2.copy()
            return
        if self.mean.shape != mean.shape:
            raise ValueError("Inconsistent feature dimensions in normalization corpus")
        total = self.count + count
        delta = mean - self.mean
        self.m2 += m2 + delta ** 2 * (self.count * count / total)
        self.mean += delta * (count / total)
        self.count = total

    def result(self, floor):
        if not self.count:
            raise ValueError("No supervised training values for normalization")
        return {"count": self.count, "mean": self.mean.tolist(),
                "std": np.maximum(np.sqrt(self.m2 / self.count), floor).tolist()}


def _write_immutable(path, value):
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Refusing to replace a different finalized artifact: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
    return hashlib.sha256(payload).hexdigest()


def finalize(inventory_path, output, *, valid_ratio=0.01, seed=42, std_floor=1e-5,
             verification_memo=None):
    """Finalize an explicit inventory, optionally sharing verified immutable files.

    Reusing a memo across overlapping corpora avoids a second content scan;
    each corpus still fits its own train-only statistics and episode split.
    The verifier invalidates changed file identities or expected digests.
    """
    from egomimic.rldb.world_model import open_prepared_episode

    inventory_path, output = Path(inventory_path).resolve(), Path(output).resolve()
    inventory = json.loads(inventory_path.read_text())
    if not 0 < valid_ratio < 1 or not np.isfinite(std_floor) or std_floor <= 0:
        raise ValueError("Require a nonempty train/val split and positive finite std floor")
    episodes, identities, paths, dino, schemas = [], set(), set(), None, inventory["pose_schemas"]
    source_episodes = set()
    episode_moments = {}
    verifier = verification_memo if verification_memo is not None else ArtifactVerificationMemo()
    for receipt_path in inventory["receipts"]:
        receipt_path = (inventory_path.parent / receipt_path).resolve()
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("schema") != "mot-world-feature-cache-v1"
                or receipt.get("status") != "PASS" or receipt.get("arrays_normalized") is not False):
            raise ValueError(f"Feature cache is not a verified raw episode: {receipt_path}")
        # The producer places its receipt beside <output>, named <output>.json.
        artifact = Path(str(receipt_path)[:-5]).resolve()
        identity = (receipt["embodiment"], receipt["episode_id"])
        if identity in identities or artifact in paths:
            raise ValueError("Duplicate episode identity or physical path in corpus")
        identities.add(identity)
        paths.add(artifact)
        native = receipt["native_source"]
        native_identity = (native["source_demo_sha256"], native["episode"])
        if not all(native_identity) or native_identity in source_episodes:
            raise ValueError("Duplicate or missing physical source episode identity")
        source_episodes.add(native_identity)
        if "moments" in receipt:
            if set(receipt["moments"]) != {"image", "track", "pose"}:
                raise ValueError("Incomplete per-episode normalization moments")
            episode_moments[identity] = receipt["moments"]
        schema = schemas[identity[0]]
        if schema["fields"] != receipt["pose_fields"] or schema["dimension"] != len(receipt["pose_fields"]):
            raise ValueError("Measured pose receipt disagrees with configured schema")
        if dino is not None and dino != receipt["dino_identity"]:
            raise ValueError("Mixed DINO extraction identities in corpus")
        dino = receipt["dino_identity"]
        fingerprint = verifier.verify(artifact, receipt["output_sha256"])
        episodes.append(dict(embodiment=identity[0], episode_id=identity[1],
                             path=os.path.relpath(artifact, output.parent), sha256=receipt["output_sha256"],
                             frames=receipt["frames"], frame="fixed_reference", pose_kind="measured_pose",
                             verified_fingerprint=fingerprint,
                             native_source=native,
                             receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest()))
    if not episodes or set(schemas) != {e["embodiment"] for e in episodes}:
        raise ValueError("Corpus and pose schema embodiment sets must match and be nonempty")
    split_evidence = {}
    for name in sorted(schemas):
        ids = {e["episode_id"] for e in episodes if e["embodiment"] == name}
        train, val = split_dataset_names(ids, valid_ratio=valid_ratio, seed=seed)
        if not train or not val or train & val or train | val != ids:
            raise ValueError(f"Incomplete episode split: {name}")
        split_evidence[name] = {"train": sorted(train), "val": sorted(val)}
        for ep in episodes:
            if ep["embodiment"] == name:
                ep["split"] = "train" if ep["episode_id"] in train else "val"
    episodes.sort(key=lambda e: (e["embodiment"], e["episode_id"]))
    moments = {key: StreamingMoments() for key in ["image", "track", *(f"pose/{n}" for n in schemas)]}
    for ep in episodes:
        if ep["split"] != "train":
            continue
        summaries = episode_moments.get((ep["embodiment"], ep["episode_id"]))
        if summaries is not None:
            for key in ("image", "track", "pose"):
                destination = f"pose/{ep['embodiment']}" if key == "pose" else key
                moments[destination].merge(summaries[key])
            continue
        with open_prepared_episode(output.parent / ep["path"]) as cache:
            for frame in range(ep["frames"]):
                moments["image"].update(cache["image"][frame])
                track = cache["track"][frame]
                confidence = cache["track_confidence"][frame]
                if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
                    raise ValueError("Track confidence must lie in [0,1]")
                moments["track"].update(track[confidence > 0])
                if cache["pose_valid"][frame]:
                    moments[f"pose/{ep['embodiment']}"].update(cache["pose"][frame])
    stats = {"schema": "mot-world-normalization-v1", "scope": "train_only",
             "method": "coordinate_population_mean_std", "std_floor": std_floor,
             "train_episodes": training_identities(episodes),
             "statistics": {key: value.result(std_floor) for key, value in moments.items()}}
    stats_path = output.with_name(output.stem + "-normalization.json")
    stats_sha = _write_immutable(stats_path, stats)
    manifest = {"schema": "mot-world-prepared-v1", "dino_identity": dino,
                "pose_schemas": schemas, "track_provenance": inventory["track_provenance"],
                "episodes": episodes,
                "normalization": {"scope": "train_only", "sha256": stats_sha,
                                  "path": stats_path.name, "arrays_normalized": False},
                "split_provenance": {"seed": seed, "validation_ratio": valid_ratio,
                                     "episode_ids": split_evidence, "id_overlap": 0, "physical_path_overlap": 0,
                                     "complete_corpus_coverage": True},
                "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest()}
    _write_immutable(output, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--valid-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--std-floor", type=float, default=1e-5)
    args = parser.parse_args()
    result = finalize(args.inventory, args.output, valid_ratio=args.valid_ratio, seed=args.seed, std_floor=args.std_floor)
    print(json.dumps({"status": "PASS", "manifest": args.output, "episodes": len(result["episodes"])}))


if __name__ == "__main__":
    main()
