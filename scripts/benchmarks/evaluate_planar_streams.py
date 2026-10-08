"""Run additional frozen evaluation levels with fixed preview-video sampling."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tarfile
import time
import traceback

import boto3
from botocore.config import Config
import numpy as np
import torch
import yaml

from planar_rollout_core import EpisodeWatchdog, run_episode, summarize

ROOT = Path("/workspace/cotrain")
OUT = ROOT / "evidence"
CONFIG = yaml.safe_load(Path("/tmp/evaluation.yaml").read_text())
CONTRACT = CONFIG["contract"]
TASK_LEVELS = CONFIG["task_levels"]
assert TASK_LEVELS == sorted(set(TASK_LEVELS)) and set(TASK_LEVELS) <= set(CONTRACT["levels"])
EXPECTED_ROLLOUTS = len(TASK_LEVELS) * CONTRACT["rollouts_per_model_embodiment_level"]
CLIENT = boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT_URL"],
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name="auto", config=Config(max_pool_connections=12, retries={"max_attempts": 6}))
CAN_WRITE = False


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""): value.update(part)
    return value.hexdigest()


def upload(path):
    if not CAN_WRITE: raise RuntimeError("Output prefix has not been checked")
    key = CONFIG["output_prefix"] + path.relative_to(OUT).as_posix()
    sha = digest(path)
    CLIENT.upload_file(str(path), "rldb", key, ExtraArgs={"Metadata": {"sha256": sha}})
    return {"uri": "s3://rldb/" + key, "sha256": sha, "bytes": path.stat().st_size}


def write(name, value):
    path = OUT / name; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return upload(path)


def download(spec, path):
    bucket, key = spec["uri"].removeprefix("s3://").split("/", 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    CLIENT.download_file(bucket, key, str(path))
    if digest(path) != spec["sha256"]: raise ValueError("Artifact SHA-256 mismatch: " + str(path))
    return path


def read_existing(name):
    try:
        response = CLIENT.get_object(Bucket="rldb", Key=CONFIG["output_prefix"] + name)
    except CLIENT.exceptions.NoSuchKey:
        return None
    raw = response["Body"].read()
    assert hashlib.sha256(raw).hexdigest() == response["Metadata"]["sha256"]
    return json.loads(raw)


def arm_watchdog(seconds):
    def expired(_signal, _frame): raise EpisodeWatchdog()
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))


def video_episode(policy, env, entry, budget, contract, *, label, record_video):
    from egomimic.eval.planar_rollout import PlanarActionQueue, PlanarTimedArcExecutionSelector
    from planar_seed_rules import assert_reset_matches
    seed = int(entry["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    obs, info = env.reset(seed=seed)
    assert_reset_matches(env, entry["initial_states"][CONFIG["domain"]["label"]], contract)
    queue = (PlanarActionQueue(policy, execution_horizon=contract["dp_execution_steps"])
             if CONFIG["arm"]["arm"] == "dp" else PlanarActionQueue(policy,
             execution_selector=PlanarTimedArcExecutionSelector(fraction=contract["arc_support_fraction"])))
    def step(count, coverage, predictions):
        if count % 100 == 0:
            write("LIVE.json", {"label": label, "level": env.obstacle_level, "seed": seed,
                "control_steps": count, "budget": budget, "coverage": coverage,
                "predictions": predictions, "record_video": record_video,
                "checked_at_utc": datetime.now(timezone.utc).isoformat()})
    if not record_video:
        try:
            arm_watchdog(contract["episode_watchdog_seconds"])
            row = run_episode(queue, env, obs, info, budget, on_step=step)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        row["video"] = None
        row["video_recording"] = "skipped_by_fixed_seed_policy"
        return row
    first = np.ascontiguousarray(env.render(), dtype=np.uint8)
    path = OUT / "videos" / (label + ".mp4"); path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path = path.with_suffix(".ffmpeg.log")
    with stderr_path.open("w") as log:
        process = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{first.shape[1]}x{first.shape[0]}",
            "-r", str(contract["video_fps"]), "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-threads", "1", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", str(path)],
            stdin=subprocess.PIPE, stderr=log)
        def frame():
            process.stdin.write(np.ascontiguousarray(env.render(), dtype=np.uint8).tobytes())
        try:
            arm_watchdog(contract["episode_watchdog_seconds"])
            row = run_episode(queue, env, obs, info, budget, on_frame=frame, on_step=step)
            signal.setitimer(signal.ITIMER_REAL, 0)
            process.stdin.close()
            if process.wait(timeout=45) != 0: raise RuntimeError("Video encoder failed")
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if process.poll() is None:
                process.terminate(); process.wait(timeout=10)
    row["video"] = upload(path)
    row["video_recording"] = "fixed_seed_preview"
    return row


def main():
    global CAN_WRITE
    assert CONTRACT["canonical_oec56"] is False
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert "L40S" in torch.cuda.get_device_name(0)
    previous = read_existing("STARTED.json")
    if previous:
        assert previous["source_commit"] == CONTRACT["source_commit"]
        assert previous["harness_sha256"] == digest(__file__)
        assert previous["config_sha256"] == digest("/tmp/evaluation.yaml")
    else:
        assert not CLIENT.list_objects_v2(Bucket="rldb", Prefix=CONFIG["output_prefix"], MaxKeys=1).get("KeyCount")
    CAN_WRITE = True; OUT.mkdir(parents=True, exist_ok=True)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT / "source", text=True).strip()
    assert source == CONTRACT["source_commit"]
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT / "source", text=True).strip()
    write("STARTED.json", {"label": CONTRACT["label"], "canonical_oec56": False,
        "arm": CONFIG["arm"]["arm"], "embodiment": CONFIG["domain"]["label"],
        "gpu": torch.cuda.get_device_name(0), "source_commit": source,
        "harness_sha256": digest(__file__), "core_sha256": digest(Path(__file__).with_name("planar_rollout_core.py")),
        "seed_rules_sha256": digest(Path(__file__).with_name("planar_seed_rules.py")), "config_sha256": digest("/tmp/evaluation.yaml")})
    (OUT / "resolved-evaluation.yaml").write_bytes(Path("/tmp/evaluation.yaml").read_bytes())
    upload(OUT / "resolved-evaluation.yaml")
    archive = download(CONFIG["simulator"], ROOT / "simulator.tar")
    with tarfile.open(archive) as stream: stream.extractall(ROOT / "simulator", filter="data")
    sys.path.insert(0, str(ROOT / "simulator"))
    from Tsimulation.sim_v2.pushshapes.env import PushShapesEnv, SIM_VERSION
    from egomimic.eval.planar_rollout import load_planar_graph_policy
    assert SIM_VERSION == 2 and all(getattr(PushShapesEnv, k) is True for k in CONTRACT["fixed_physics"])
    assert digest(ROOT / "simulator/Tsimulation/sim_v2/pushshapes/obstacles.py") == CONTRACT["obstacles_sha256"]
    versions = {name: importlib.metadata.version(name) for name in CONTRACT["simulator_dependencies"]}
    assert versions == CONTRACT["simulator_dependencies"], versions
    bank_path = download(CONFIG["seed_bank"], ROOT / "seed-bank.json")
    bank = json.loads(bank_path.read_text())
    assert bank["contract"] == CONFIG["seed_bank_contract"]
    assert sorted(map(int, bank["levels"])) == CONTRACT["levels"]
    budgets_path = download(CONFIG["budgets"], ROOT / "budgets.json")
    budgets = json.loads(budgets_path.read_text())
    assert budgets["statistic"] == CONTRACT["budget_statistic"]
    assert budgets["multiplier"] == CONTRACT["budget_multiplier"]
    assert budgets["source_content_manifest_sha256"] == CONFIG["data_content_manifest_sha256"]
    paths = {name: download(spec, ROOT / "restored" / name) for name, spec in CONFIG["arm"]["restore"].items()}
    checkpoint = torch.load(paths["checkpoint.ckpt"], map_location="cpu", weights_only=False, mmap=True)
    assert int(checkpoint["global_step"]) == CONTRACT["checkpoint_step"]
    assert int(checkpoint["ema_num_updates"]) == CONTRACT["checkpoint_step"]
    epoch = int(checkpoint["epoch"]); del checkpoint
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision(CONTRACT["matmul_precision"])
    torch.backends.cudnn.benchmark = False
    policy = load_planar_graph_policy(checkpoint_path=paths["checkpoint.ckpt"], config_path=paths["config.yaml"],
        normalizer_path=paths["normalizer.json"], checkpoint_sha256=CONFIG["arm"]["restore"]["checkpoint.ckpt"]["sha256"],
        config_sha256=CONFIG["arm"]["restore"]["config.yaml"]["sha256"],
        normalizer_sha256=CONFIG["arm"]["restore"]["normalizer.json"]["sha256"],
        embodiment_name=CONFIG["domain"]["embodiment"], device="cuda", use_ema=CONTRACT["use_ema"])
    samplers = [module for module in policy.graph.nets.modules() if hasattr(module, "num_inference_steps")]
    assert len(samplers) == 1 and samplers[0].num_inference_steps == CONTRACT["sampler_steps"]
    runtime = {"checkpoint_step": CONTRACT["checkpoint_step"], "true_epoch": epoch,
        "data_manifest_sha256": CONFIG["data_manifest_sha256"],
        "restored": CONFIG["arm"]["restore"], "use_ema": CONTRACT["use_ema"],
        "sampler_steps": samplers[0].num_inference_steps, "source_commit": source,
        "simulator": CONFIG["simulator"], "dependencies": versions,
        "torch": torch.__version__, "python": sys.version, "gpu": torch.cuda.get_device_name(0),
        "token_shape": policy.token_shape, "native_shape": policy.native_shape,
        "seed_bank": CONFIG["seed_bank"], "budgets": CONFIG["budgets"], "canonical_oec56": False}
    write("RESTORE_VERIFIED.json", runtime)
    video_seeds = {int(level): set(seeds) for level, seeds in CONFIG["artifact_policy"]["video_seeds_by_level"].items()}
    write("EVAL_PLAN_VERIFIED.json", {"expected_rollouts": EXPECTED_ROLLOUTS,
        "artifact_policy": CONFIG["artifact_policy"], "levels": TASK_LEVELS,
        "parent_sweep": CONFIG["parent_sweep"]})
    import wandb
    wb = wandb.init(entity="rl2-group", project="pushshapes-planar-v2", id=CONFIG["run_id"],
        name=CONFIG["run_id"], job_type="evaluation", group=CONTRACT["name"], resume="allow",
        config={"comparison": CONTRACT, "arm": CONFIG["arm"]["arm"], "embodiment": CONFIG["domain"]["label"],
                "checkpoint": CONFIG["arm"]["restore"]["checkpoint.ckpt"], "seed_bank": CONFIG["seed_bank"],
                "artifact_policy": CONFIG["artifact_policy"], "task_levels": TASK_LEVELS,
                "parent_sweep": CONFIG["parent_sweep"]})
    results = []; summaries = {}
    for level in TASK_LEVELS:
        entries = bank["levels"][str(level)]["selected"]
        assert len(entries) == CONTRACT["rollouts_per_model_embodiment_level"]
        assert len({int(r["seed"]) for r in entries}) == len(entries)
        budget_info = budgets["by_level"][str(level)]; budget = int(budget_info["budget"])
        environment = dict(CONFIG["domain"]["environment"], obstacle_level=level)
        env = PushShapesEnv(**environment); env.SUCCESS_THRESHOLD = CONTRACT["coverage_threshold"]
        level_results = []
        try:
            for entry in entries:
                label = f"L{level:02d}_seed{entry['seed']:07d}"
                pair = (level, int(entry["seed"]))
                row = read_existing("episodes/" + label + ".json")
                if row is None:
                    row = video_episode(policy, env, entry, budget, CONTRACT, label=label,
                                        record_video=pair[1] in video_seeds[level])
                else:
                    assert row["arm"] == CONFIG["arm"]["arm"] and row["embodiment"] == CONFIG["domain"]["label"]
                    assert row["level"] == level and row["seed"] == pair[1] and row["budget"] == budget
                    assert row["checkpoint_step"] == CONTRACT["checkpoint_step"]
                row.update(arm=CONFIG["arm"]["arm"], embodiment=CONFIG["domain"]["label"], level=level,
                    seed=int(entry["seed"]), checkpoint_step=CONTRACT["checkpoint_step"], true_epoch=epoch,
                    budget_derived=bool(budget_info["derived"]), label=CONTRACT["label"], canonical_oec56=False,
                    initial_state=entry["initial_states"][CONFIG["domain"]["label"]])
                details = write("episodes/" + label + ".json", row)
                compact = {k: v for k, v in row.items() if k not in ["actions", "coverages", "replans", "initial_state"]}
                compact["details"] = details
                results.append(compact); level_results.append(compact)
                wb.log({"episode/index": len(results), "episode/level": level, "episode/seed": row["seed"],
                    "episode/peak_coverage": row["peak_coverage"], "episode/SR80": row["SR80"],
                    "episode/SR95": row["SR95"], "episode/failure": int(row["failure"] is not None),
                    "episode/wall_seconds": row["wall_seconds"],
                    "episode/video_saved": int(row.get("video") is not None)})
                write("progress.json", {"completed": len(results), "expected": EXPECTED_ROLLOUTS,
                    "active_level": level, "level_completed": len(level_results), "results": results,
                    "checked_at_utc": datetime.now(timezone.utc).isoformat()})
                print("EPISODE_COMPLETE", label, row["peak_coverage"],
                      row["failure"], round(row["wall_seconds"], 2), flush=True)
        finally: env.close()
        summaries[str(level)] = {**summarize(level_results), "budget": budget,
                                "budget_derived": bool(budget_info["derived"])}
        write(f"level-{level:02d}-summary.json", summaries[str(level)])
        for key, value in summaries[str(level)].items(): wb.summary[f"level_{level:02d}/{key}"] = value
    assert len(results) == EXPECTED_ROLLOUTS
    write("EVAL_COMPLETE.json", {"status": "COMPLETE", "arm": CONFIG["arm"]["arm"],
        "embodiment": CONFIG["domain"]["label"], "runtime": runtime, "contract": CONTRACT,
        "results": results, "by_level": summaries, "canonical_oec56": False, "wandb_url": wb.url,
        "parent_sweep": CONFIG["parent_sweep"], "task_levels": TASK_LEVELS,
        "artifact_policy": CONFIG["artifact_policy"]})
    wb.finish()


if __name__ == "__main__":
    try: main()
    except BaseException:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if CAN_WRITE: write("FAILED.json", {"traceback": traceback.format_exc()})
        raise
