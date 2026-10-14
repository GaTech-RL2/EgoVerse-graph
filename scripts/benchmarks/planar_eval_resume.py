"""Reuse checksummed rollout artifacts after repartitioning an evaluation."""
import hashlib
import json

import yaml


SEMANTIC_FIELDS = (
    "contract", "domain", "arm", "simulator", "seed_bank", "budgets",
    "data_manifest_sha256", "source_selection_sha256", "seed_bank_contract",
    "training_seed", "regime",
)


def read_checked(client, prefix, name):
    key = prefix + name
    try:
        response = client.get_object(Bucket="rldb", Key=key)
    except client.exceptions.NoSuchKey:
        return None
    raw = response["Body"].read()
    sha = hashlib.sha256(raw).hexdigest()
    if response.get("Metadata", {}).get("sha256") != sha:
        raise ValueError("Resume artifact checksum mismatch: " + key)
    return raw, {"uri": "s3://rldb/" + key, "sha256": sha, "bytes": len(raw)}


class EvaluationResume:
    def __init__(self, client, target):
        self.client = client
        self.target = target
        self.sources = []
        for source in target.get("resume_sources", []):
            prefix = source["prefix"]
            if not prefix.endswith("/") or prefix == target["output_prefix"]:
                raise ValueError("Resume source must be a distinct complete prefix")
            raw, config_ref = self._required(prefix, "resolved-evaluation.yaml")
            config = yaml.safe_load(raw)
            started = json.loads(self._required(prefix, "STARTED.json")[0])
            runtime = json.loads(self._required(prefix, "RESTORE_VERIFIED.json")[0])
            if started["config_sha256"] != config_ref["sha256"]:
                raise ValueError("Resume source configuration was replaced")
            if started["harness_sha256"] != source["harness_sha256"]:
                raise ValueError("Unexpected resume harness")
            for field in SEMANTIC_FIELDS:
                if config.get(field) != target.get(field):
                    raise ValueError("Resume protocol mismatch: " + field)
            if runtime["restored"] != target["arm"]["restore"]:
                raise ValueError("Resume checkpoint/config/normalizer mismatch")
            for field in ("checkpoint_step", "sampler_steps", "use_ema", "source_commit"):
                if runtime[field] != target["contract"][field]:
                    raise ValueError("Resume runtime mismatch: " + field)
            for level in config["task_levels"]:
                expected = target["artifact_policy"]["video_seeds_by_level"].get(str(level))
                if expected is not None and config["artifact_policy"]["video_seeds_by_level"][str(level)] != expected:
                    raise ValueError("Resume preview seeds changed")
            self.sources.append({"prefix": prefix, "levels": set(config["task_levels"]),
                                 "config": config_ref})
        owned = [level for source in self.sources for level in source["levels"]]
        if len(owned) != len(set(owned)):
            raise ValueError("Overlapping resume sources")

    def _required(self, prefix, name):
        value = read_checked(self.client, prefix, name)
        if value is None:
            raise ValueError("Missing resume evidence: " + prefix + name)
        return value

    def lookup(self, level, seed, budget):
        name = f"episodes/L{level:02d}_seed{seed:07d}.json"
        for source in self.sources:
            if level not in source["levels"]:
                continue
            value = read_checked(self.client, source["prefix"], name)
            if value is None:
                return None
            raw, reference = value
            row = json.loads(raw)
            expected = {"arm": self.target["arm"]["arm"],
                        "embodiment": self.target["domain"]["label"],
                        "level": level, "seed": seed, "budget": budget,
                        "checkpoint_step": self.target["contract"]["checkpoint_step"]}
            for key, value in expected.items():
                if row[key] != value:
                    raise ValueError("Resume episode mismatch: " + key)
            return row, reference
        return None

    def receipt(self):
        return [{"prefix": s["prefix"], "levels": sorted(s["levels"]),
                 "config": s["config"]} for s in self.sources]


def balanced_level_partitions(budgets, completed, *, partitions):
    """Balance remaining inference calls, keeping complete levels disjoint."""
    if not 1 <= partitions <= len(budgets):
        raise ValueError("Invalid evaluation partition count")
    bins = [[] for _ in range(partitions)]
    loads = [0] * partitions
    costs = {int(level): ((int(value["budget"]) + 7) // 8)
             * (50 - int(completed.get(int(level), 0))) for level, value in budgets.items()}
    if any(cost < 0 for cost in costs.values()):
        raise ValueError("Too many completed rollouts")
    for level in sorted(costs, key=lambda level: (-costs[level], level)):
        slot = min(range(partitions), key=lambda slot: (loads[slot], len(bins[slot]), slot))
        bins[slot].append(level)
        loads[slot] += costs[level]
    return [sorted(levels) for levels in bins], loads
