"""Manifest-bound prepared pixel corpus admission; legacy requests remain exact."""
from collections import Counter
import json
from pathlib import Path
from scripts.train.joint.contract import need, sha


def verify_dataset(request):
    from scripts.train.joint import imagenet, native_subtasks
    if imagenet.selected(request):
        return imagenet.verify_dataset(request)
    if native_subtasks.selected(request):
        return native_subtasks.verify_dataset(request)
    policy = request.get("dataset_contract")
    if policy is None:
        return {"train": 450, "val": 50}, 100
    need(policy.get("schema") == "prepared-pixel-dataset-v1", "Unknown dataset contract")
    manifest_path = Path(request["data_manifest"])
    need(sha(manifest_path) == request["data_manifest_sha256"] == policy["manifest_sha256"],
         "Prepared dataset manifest changed")
    manifest = json.loads(manifest_path.read_text())
    need(manifest["schema"] == "pixel-proprio-prepared-v1", "Wrong prepared pixel schema")
    episodes = manifest["episodes"]
    counts = dict(Counter(ep["split"] for ep in episodes))
    need(set(counts) == {"train", "val"} and counts == policy["splits"]
         and sum(counts.values()) == policy["episodes"] and min(counts.values()) > 0,
         "Incomplete prepared split")
    physical = [(ep["native_source"]["source_demo_sha256"],
                 ep["native_source"]["episode"]) for ep in episodes]
    need(len(set(physical)) == len(physical)
         and len({(ep["embodiment"], ep["episode_id"]) for ep in episodes}) == len(episodes),
         "Duplicate physical episode or train/validation overlap")
    norm = manifest["normalization"]
    need(norm["scope"] == "train_only" and norm["sha256"] == policy["normalization_sha256"]
         and sha(manifest_path.parent / norm["path"]) == norm["sha256"],
         "Train-only normalization changed")
    anchors = policy["validation_anchors_per_episode"]
    need(type(anchors) is int and anchors > 0 and policy["validation_batches"] == counts["val"] * anchors,
         "Unbound validation anchor count")
    return counts, policy["validation_batches"]
