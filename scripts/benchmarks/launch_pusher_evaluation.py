"""Re-evaluate completed pusher checkpoints with a separately pinned harness.

The policy checkout remains at the exact training commit. Only the evaluation
harness and its invocation are supplied by this revision; model code, weights,
normalizers, frozen resets, budgets and replanning remain pinned to training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "scripts/benchmarks"

REBIND = '''import json, os
from pathlib import Path
import yaml
p = Path("/tmp/evaluation.yaml")
c = yaml.safe_load(p.read_text())
training = json.loads(Path("/tmp/training-complete.json").read_text())
assert c["contract"]["source_commit"] == training["source_commit"] == os.environ["SOURCE_COMMIT"]
old = c["output_prefix"]
assert "/evaluation/" in old
c["output_prefix"] = old.replace("/evaluation/", "/evaluation-" + os.environ["EVAL_REVISION"] + "/", 1)
c["run_id"] += "-" + os.environ["EVAL_REVISION"]
c["evaluation_revision"] = {"name": os.environ["EVAL_REVISION"],
    "harness_commit": os.environ["EVAL_HARNESS_COMMIT"],
    "harness_sha256": os.environ["EVAL_HARNESS_SHA256"],
    "training_source_commit": training["source_commit"], "previous_output_prefix": old}
p.write_text(yaml.safe_dump(c, sort_keys=False))
'''


def workflow(receipts, *, name, regime, seed, shards, revision, harness_commit):
    if not re.fullmatch(r"[a-z0-9-]{1,23}", name):
        raise ValueError("Expected a short workflow name")
    if not re.fullmatch(r"[a-z0-9-]+", revision) or not re.fullmatch(r"[0-9a-f]{40}", harness_commit):
        raise ValueError("Expected an evaluation revision and full harness commit")
    if not shards or len(shards) != len(set(shards)) or not set(shards) <= {0, 1, 2}:
        raise ValueError("Select unique shards from 0, 1, 2")
    campaign = yaml.safe_load((ROOT / "egomimic/hydra_configs/benchmark/pusher_triangle_holdout.yaml").read_text())
    selected = [r for r in receipts if r["regime"] == regime and r["seed"] == seed]
    assert len(selected) == len(campaign["arms"]) == 6
    assert {r["arm"] for r in selected} == set(campaign["arms"])
    credentials = {"grabber-arc-r2-20260916": {"R2_ACCESS_KEY_ID": "r2_access_key_id",
        "R2_SECRET_ACCESS_KEY": "r2_secret_access_key", "R2_ENDPOINT_URL": "r2_endpoint_url"},
        "egoverse-github": {"GITHUB_TOKEN": "github_token"},
        "egoverse-wandb": {"WANDB_API_KEY": "wandb_api_key"}}
    entry = (BENCHMARKS / "pusher_holdout_entry.sh").read_text()
    original = "    python scripts/benchmarks/evaluate_planar_streams.py\n"
    assert entry.count(original) == 1
    entry = entry.replace(original, "    python /tmp/rebind_evaluation.py\n    python /tmp/evaluate_planar_streams.py\n")
    harness = (BENCHMARKS / "evaluate_planar_streams.py").read_text()
    harness_sha = hashlib.sha256(harness.encode()).hexdigest()
    tasks = []
    for complete in selected:
        assert complete["global_step"] == campaign["optimizer_updates"]
        assert complete["checkpoint"]["global_step"] == complete["checkpoint"]["ema_num_updates"] == complete["global_step"]
        assert complete["checkpoint"]["source_commit"] == complete["source_commit"]
        for shard in sorted(shards):
            files = [{"path": "/tmp/entry.sh", "contents": entry},
                {"path": "/tmp/training-complete.json", "contents": json.dumps(complete)},
                {"path": "/tmp/rebind_evaluation.py", "contents": REBIND},
                {"path": "/tmp/evaluate_planar_streams.py", "contents": harness}]
            files += [{"path": "/tmp/" + f, "contents": (BENCHMARKS / f).read_text()}
                for f in ("planar_rollout_core.py", "planar_seed_rules.py", "planar_imports.py")]
            tasks.append({"name": "eval-" + complete["arm"].replace("_", "-") + f"-s{shard}",
                "resource": "evaluation", "image": "nvcr.io/nvidia/pytorch:25.06-py3", "credentials": credentials,
                "environment": {"SOURCE_COMMIT": complete["source_commit"], "RUN_KIND": "evaluation",
                    "EVAL_SHARD": str(shard), "EVAL_REVISION": revision,
                    "EVAL_HARNESS_COMMIT": harness_commit, "EVAL_HARNESS_SHA256": harness_sha},
                "command": ["bash"], "args": ["/tmp/entry.sh"], "files": files})
    return {"workflow": {"name": name, "tasks": tasks,
        "resources": {"evaluation": {"cpu": 8, "gpu": 1, "memory": "64Gi", "storage": "120Gi", "platform": "ovx-l40s"}},
        "timeout": {"queue_timeout": "48h", "exec_timeout": "168h"}}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-receipts", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--regime", choices=("cotrain", "clean"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shards", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--revision", required=True)
    parser.add_argument("--harness-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = workflow(json.loads(args.training_receipts.read_text()), name=args.name, regime=args.regime,
        seed=args.seed, shards=args.shards, revision=args.revision, harness_commit=args.harness_commit)
    args.output.write_text(yaml.safe_dump(result, sort_keys=False))
    print(args.output)
