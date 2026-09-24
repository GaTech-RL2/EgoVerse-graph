"""Run native LIBERO training/evaluation in an isolated OSMO GPU container."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

from egomimic.benchmarks.libero.catalog import TASKS

DATA_REPO = "yifengzhu-hf/LIBERO-datasets"
DATA_REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
REPLAY_REPO = "chaoqi-liu/libero10_N500.zarr"
REPLAY_REVISION = "685b2b764e525ad33ab36d7315adbcab07494251"
REPLAY_NAME = "libero10_N500.zarr.zip"
REPLAY_SHA256 = "176de6aed271a76a5d6afc43af1ef6562614e86e9b08aaae5eb4c337538a0550"
GPU_PLATFORMS = {"L40S": "ovx-l40s", "H100": "dgx-h100"}


def validate_gpu_allocation(gpus, gpu_type):
    """Reject a different count or accelerator family before staging any data."""
    import torch

    if gpu_type not in GPU_PLATFORMS:
        raise ValueError(f"Unsupported GPU type: {gpu_type}")
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != gpus
        or any(gpu_type not in torch.cuda.get_device_name(i) for i in range(gpus))
    ):
        raise RuntimeError(f"This workflow requests exactly {gpus} {gpu_type} GPUs")


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def download(url, path, expected_sha256):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and digest(path) == expected_sha256:
        return
    temporary = path.with_suffix(path.suffix + ".partial")
    with (
        urllib.request.urlopen(url, timeout=120) as source,
        temporary.open("wb") as target,
    ):
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    if digest(temporary) != expected_sha256:
        raise ValueError(f"Download hash mismatch: {path.name}")
    temporary.replace(path)


def extract_replay(archive, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            path = Path(member.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or (member.external_attr >> 16) & 0o170000 == 0o120000
            ):
                raise ValueError("Unsafe replay archive member")
        handle.extractall(output)
    roots = [
        path.parent
        for path in output.rglob(".zgroup")
        if (path.parent / "meta/episode_ends").is_dir()
    ]
    if len(roots) != 1:
        raise ValueError("Expected exactly one OAT replay in the archive")
    return roots[0]


def stage_dataset(root, suite, evidence):
    data = Path(root) / "data"
    data.mkdir(parents=True, exist_ok=True)
    if suite == "libero_10":
        archive = data / REPLAY_NAME
        url = f"https://huggingface.co/datasets/{REPLAY_REPO}/resolve/{REPLAY_REVISION}/{REPLAY_NAME}"
        download(url, archive, REPLAY_SHA256)
        replay = extract_replay(archive, data / "released")
        write_json(
            evidence / "data-source.json",
            {
                "repo": REPLAY_REPO,
                "revision": REPLAY_REVISION,
                "sha256": REPLAY_SHA256,
                "url": url,
            },
        )
        return replay
    # The release supplies one ready-made replay. Other catalog suites use
    # official demonstrations and the same native conversion for both methods.
    from egomimic.benchmarks.libero.convert import convert_suite

    url = f"https://huggingface.co/api/datasets/{DATA_REPO}/revision/{DATA_REVISION}?blobs=true"
    with urllib.request.urlopen(url, timeout=120) as response:
        metadata = json.load(response)
    if metadata["sha"] != DATA_REVISION:
        raise ValueError("Dataset revision changed")
    sources = {
        Path(row["rfilename"]).name: row
        for row in metadata["siblings"]
        if row["rfilename"].startswith(suite + "/")
    }
    receipts = []
    cache = os.environ.get("LIBERO_RAW_CACHE") or None
    raw = Path(cache) if cache else data / "raw"
    for task in TASKS[suite]:
        source = sources[task + "_demo.hdf5"]
        url = f"https://huggingface.co/datasets/{DATA_REPO}/resolve/{DATA_REVISION}/{source['rfilename']}"
        path = raw / source["rfilename"]
        if cache:
            if not path.is_file() or digest(path) != source["lfs"]["sha256"]:
                raise ValueError(f"Cached demonstration checksum differs: {path}")
        else:
            download(url, path, source["lfs"]["sha256"])
        receipts.append(source)
    replay = data / f"{suite}.zarr"
    convert_suite(raw / suite, replay, suite)
    write_json(
        evidence / "data-source.json",
        {
            "repo": DATA_REPO,
            "revision": DATA_REVISION,
            "files": receipts,
            "verified_read_only_cache": cache,
        },
    )
    return replay


class ArtifactUploader:
    """Snapshot completed checkpoints and stream evidence to a unique R2 prefix."""

    def __init__(self, evidence, run_id):
        import boto3

        self.client = boto3.client(
            "s3",
            endpoint_url=os.environ["R2_ENDPOINT_URL"],
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        )
        self.evidence = Path(evidence)
        self.prefix = f"experiments/arc-oat-20260919/{run_id}/"
        if self.client.list_objects_v2(
            Bucket="rldb", Prefix=self.prefix, MaxKeys=1
        ).get("KeyCount"):
            raise FileExistsError("Refusing to overwrite an existing run prefix")
        self.sent, self.receipts = {}, {}
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def upload(self, final=False):
        for path in sorted(self.evidence.rglob("*")):
            if not path.is_file() or path.suffix in {".tmp", ".partial"}:
                continue
            try:
                before = (path.stat().st_size, path.stat().st_mtime_ns)
                if self.sent.get(str(path)) == before:
                    continue
                checkpoint = path.suffix == ".ckpt"
                if checkpoint and not final and time.time() - path.stat().st_mtime < 5:
                    continue
                # Copy first: last.ckpt can be replaced by Lightning mid-upload.
                snapshot = self.evidence.parent / "upload-snapshot"
                shutil.copyfile(path, snapshot)
                if before != (path.stat().st_size, path.stat().st_mtime_ns):
                    continue
                sha = digest(snapshot)
                relative = path.relative_to(self.evidence).as_posix()
                key = self.prefix + (
                    f"checkpoints/{sha}/{path.name}" if checkpoint else relative
                )
                self.client.upload_file(
                    str(snapshot), "rldb", key, ExtraArgs={"Metadata": {"sha256": sha}}
                )
                snapshot.unlink()
                self.sent[str(path)] = before
                if checkpoint:
                    self.receipts[relative] = {
                        "sha256": sha,
                        "bytes": before[0],
                        "uri": "s3://rldb/" + key,
                    }
                    write_json(
                        self.evidence / "checkpoint-receipts.json", self.receipts
                    )
            except FileNotFoundError:
                # save_top_k can retire the previous file during a scan.
                continue

    def loop(self):
        while not self.stop.wait(30):
            try:
                self.upload()
            except Exception as error:
                print("ARTIFACT_UPLOAD_RETRY", type(error).__name__, flush=True)


def execute(argv, log):
    print("EXECUTE", json.dumps(argv), flush=True)
    with Path(log).open("w") as handle:
        process = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            print(line, end="", flush=True)
        if process.wait():
            raise RuntimeError(f"Command failed; see {log}")


def training_layout(gpus, mode):
    """Preserve the optimizer batch while distributing microbatches across GPUs."""
    if gpus not in (1, 2, 4, 8) or mode not in ("full", "smoke"):
        raise ValueError("Training requires 1, 2, 4 or 8 GPUs and full/smoke mode")
    microbatch = 4 if mode == "smoke" else min(256, 1024 // gpus)
    accumulation = 1 if mode == "smoke" else 1024 // (gpus * microbatch)
    return {
        "world_size": gpus,
        "microbatch_size": microbatch,
        "gradient_accumulation": accumulation,
        "global_batch_size": gpus * microbatch * accumulation,
    }


def training_arguments(
    method, suite, dataset, evidence, mode, epochs, *, gpus=1, arc_backbone="unet"
):
    if arc_backbone not in {"unet", "oat_dp"}:
        raise ValueError("Unknown ARC backbone")
    layout = training_layout(gpus, mode)
    experiment = {
        "tokenizer": "libero_oattok",
        "oat": "libero_oatpolicy",
        "arc": "libero_arc_policy",
        "arc_stk": "libero_arc_stk_policy",
        "arc_dur": "libero_arc_dur_policy",
    }[method]
    if method.startswith("arc") and arc_backbone == "oat_dp":
        experiment = "libero_arc_oat_dp_policy"
    run = Path(evidence) / "training" / method
    args = [
        sys.executable,
        "-m",
        "egomimic.trainHydra",
        f"+experiment=oat/{experiment}",
        "hydra/launcher=basic",
        f"benchmark.suite={suite}",
        f"benchmark.dataset={dataset}",
        f"hydra.run.dir={run}",
        f"paths.output_dir={run}",
        "runtime.slurm_requeue_owner=none",
        f"trainer.devices={gpus}",
        f"benchmark.batch_size={layout['microbatch_size']}",
        f"trainer.accumulate_grad_batches={layout['gradient_accumulation']}",
        "trainer.precision=bf16-mixed",
        f"trainer.max_epochs={epochs}",
        "callbacks.model_checkpoint.save_top_k=1",
        "++trainer.enable_progress_bar=false",
    ]
    if gpus > 1:
        args.append("++trainer.strategy=ddp_find_unused_parameters_true")
    if method == "oat":
        args.append(
            f"benchmark.tokenizer_checkpoint={Path(evidence) / 'training/tokenizer/checkpoints/last.ckpt'}"
        )
    if mode == "full":
        args.append("logger=csv")
    if mode == "smoke":
        args.extend(
            [
                "benchmark.batch_size=4",
                "trainer.max_epochs=1",
                "trainer.limit_train_batches=2",
                "trainer.limit_val_batches=1",
                "trainer.accumulate_grad_batches=1",
                "callbacks.batch_budget=null",
                "trainer.check_val_every_n_epoch=1",
                "callbacks.model_checkpoint.every_n_epochs=1",
                "data.train_dataloader_params.libero_panda.num_workers=0",
                "data.train_dataloader_params.libero_panda.persistent_workers=false",
                "data.valid_dataloader_params.libero_panda.num_workers=0",
                "data.valid_dataloader_params.libero_panda.persistent_workers=false",
            ]
        )
    return args


def campaign_sources(campaign_id, commit, replacements=None):
    """Pin each suite explicitly when a recovered run replaces a campaign member."""
    sources = (
        replacements
        if replacements is not None
        else {
            suite: {
                "run_id": f"{campaign_id}-{suite.replace('_', '-')}",
                "source_commit": commit,
            }
            for suite in TASKS
        }
    )
    if set(sources) != set(TASKS) or any(
        not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", row.get("run_id", ""))
        or not re.fullmatch(r"[0-9a-f]{40}", row.get("source_commit", ""))
        for row in sources.values()
    ):
        raise ValueError("Campaign requires all suites and immutable run/source pins")
    if len({row["run_id"] for row in sources.values()}) != len(TASKS):
        raise ValueError("Campaign cannot reuse one run for multiple suites")
    return sources


def publish_campaign(
    client, campaign_id, *, commit, epochs, run_sources=None, arc_modes=("joint_dur",)
):
    """The last completed suite publishes scores after checking every receipt."""
    from botocore.exceptions import ClientError

    results, sources, commits = {}, {}, {}
    expected = campaign_sources(campaign_id, commit, run_sources)
    for suite in TASKS:
        run_id = expected[suite]["run_id"]
        prefix = f"experiments/arc-oat-20260919/{run_id}/"

        def read(name):
            return json.loads(
                client.get_object(Bucket="rldb", Key=prefix + name)["Body"].read()
            )

        try:
            status, runtime = read("status.json"), read("runtime.json")
            if status["state"] != "SUITE_COMPLETE":
                return False
            result = read("comparison.json")
        except ClientError as error:
            if error.response["Error"]["Code"] in {"NoSuchKey", "404"}:
                return False
            raise
        if (
            runtime["source_commit"] != expected[suite]["source_commit"]
            or runtime["mode"] != "full"
            or runtime["epochs"] != epochs
            or runtime.get("global_batch_size") != 1024
            or runtime["suite"] != suite
            or result["suite"] != suite
            or not result["complete_protocol"]
            or tuple(runtime.get("arc_modes", ["joint_dur"])) != tuple(arc_modes)
        ):
            raise ValueError("Campaign source, training budget or suite differs")
        if tuple(arc_modes) != ("joint_dur",) and (
            set(result.get("variants", {})) != set(arc_modes)
            or any(
                not row.get("complete_protocol") or row.get("suite") != suite
                for row in result["variants"].values()
            )
        ):
            raise ValueError("Campaign lacks completed results for every ARC mode")
        results[suite] = result
        sources[suite] = "s3://rldb/" + prefix
        commits[suite] = runtime["source_commit"]
    report = {
        "complete_benchmark_suite": True,
        "unique_tasks": sum(map(len, TASKS.values())),
        "source_commit": next(iter(commits.values()))
        if len(set(commits.values())) == 1
        else None,
        "source_commits": commits,
        "training_epochs": epochs,
        "global_batch_size": 1024,
        "full_released_training_budget": epochs == 5001,
        "arc_modes": list(arc_modes),
        "sources": sources,
        "suites": results,
    }
    key = f"experiments/arc-oat-20260919/campaigns/{campaign_id}/comparison.json"
    client.put_object(
        Bucket="rldb",
        Key=key,
        Body=(json.dumps(report, indent=2) + "\n").encode(),
        ContentType="application/json",
    )
    print("CAMPAIGN_COMPLETE", "s3://rldb/" + key, flush=True)
    return True


def arc_checkpoint_settings(config, *, suite):
    """Read codec parameters from the resolved graph saved by ModelWrapper."""
    model = config["model"]
    protocol = model["benchmark_protocol"]
    if protocol.get("suite") != suite or tuple(
        protocol.get(k) for k in ("horizon", "n_obs_steps", "n_action_steps")
    ) != (32, 2, 16):
        raise ValueError("Resumed ARC checkpoint uses a different control protocol")
    stages = model["pipeline"]["stages"]
    codecs = [
        stage
        for stage in stages
        if stage.get("_target_") == "egomimic.pipeline.stages_libero_arc.LiberoArcStage"
    ]
    if len(codecs) != 2 or {stage.get("operation") for stage in codecs} != {
        "encode",
        "decode",
    }:
        raise ValueError("Resumed ARC checkpoint lacks matching encode/decode stages")
    settings = []
    for stage in codecs:
        arc_mode = stage.get("arc_mode", "joint_dur")
        if arc_mode not in ("joint_dur", "stk", "dur") or stage["horizon"] != 32:
            raise ValueError("Resumed ARC checkpoint has incompatible codec settings")
        settings.append(
            {
                "arc_mode": arc_mode,
                "arc_action_dim": 11 if arc_mode == "joint_dur" else 12,
                "arc_waypoints": stage["num_waypoints"],
                "arc_max_translation": stage["max_translation"],
                "arc_max_rotation_degrees": stage["max_rotation_degrees"],
                "arc_velocity_norm_bound": stage.get("velocity_norm_bound", 1.0),
            }
        )
    if settings[0] != settings[1]:
        raise ValueError("Resumed ARC checkpoint encode/decode parameters differ")
    return settings[0]


def restore_checkpoints(
    client,
    source_run,
    evidence,
    *,
    suite,
    mode,
    epochs,
    allow_partial=False,
    methods=("tokenizer", "oat", "arc"),
):
    """Recover completed training checkpoints without reusing partial rollouts."""
    import torch

    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", source_run):
        raise ValueError("Invalid source run ID")
    prefix = f"experiments/arc-oat-20260919/{source_run}/"

    def read(name):
        return json.loads(
            client.get_object(Bucket="rldb", Key=prefix + name)["Body"].read()
        )

    runtime, receipts = read("runtime.json"), read("checkpoint-receipts.json")
    if (runtime["suite"], runtime["mode"], runtime["epochs"]) != (
        suite,
        mode,
        1 if mode == "smoke" else epochs,
    ):
        raise ValueError("Recovery suite, mode or training budget differs")
    if mode == "full" and runtime.get("global_batch_size") != 1024:
        raise ValueError("Recovery does not match the released global training batch")
    restored = {}
    for method in methods:
        relative = f"training/{method}/checkpoints/last.ckpt"
        if allow_partial and relative not in receipts:
            continue
        receipt = receipts[relative]
        uri = receipt["uri"]
        if not uri.startswith("s3://rldb/" + prefix + "checkpoints/"):
            raise ValueError("Checkpoint receipt points outside its source run")
        path = evidence / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        client.download_file("rldb", uri.removeprefix("s3://rldb/"), str(path))
        if digest(path) != receipt["sha256"] or path.stat().st_size != receipt["bytes"]:
            raise ValueError("Recovered checkpoint hash or size differs")
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        completed = payload["loops"]["fit_loop"]["epoch_progress"]["current"][
            "completed"
        ]
        if completed < runtime["epochs"] and not allow_partial:
            raise ValueError("Cannot evaluate an incomplete training checkpoint")
        if allow_partial and mode == "full":
            budget = payload.get("training_budget", {})
            if (
                budget.get("global_batch_size") != 1024
                or budget.get("epochs") != epochs
            ):
                raise ValueError("Partial checkpoint has incompatible training budget")
            if not payload.get("optimizer_states") or "normalizer_state" not in payload:
                raise ValueError(
                    "Partial checkpoint lacks optimizer or normalization state"
                )
        restored[method] = {
            **receipt,
            "epochs_completed": completed,
            "global_step": payload["global_step"],
            "complete": completed >= runtime["epochs"],
        }
        if method.startswith("arc") and allow_partial:
            restored[method]["benchmark"] = arc_checkpoint_settings(
                payload["hyper_parameters"]["config_tree"], suite=suite
            )
        del payload
    if allow_partial and not restored:
        raise ValueError("No training checkpoint is available to resume")
    write_json(
        evidence / "recovered-training.json",
        {"source_run": source_run, "runtime": runtime, "checkpoints": restored},
    )
    return restored


class CalibrationPending(RuntimeError):
    """An unconfirmed codec must not start a full ARC policy run."""


def arc_method_modes(modes):
    modes = tuple(modes)
    if (
        not modes
        or len(set(modes)) != len(modes)
        or any(m not in {"joint_dur", "stk", "dur"} for m in modes)
    ):
        raise ValueError("ARC modes must be unique joint_dur, stk or dur values")
    return {"arc" if mode == "joint_dur" else f"arc_{mode}": mode for mode in modes}


def load_arc_calibration(client, source_run, evidence, *, suite, arc_mode="joint_dur"):
    from botocore.exceptions import ClientError

    if not source_run:
        raise CalibrationPending("ARC requires a confirmed demonstration replay run")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", source_run):
        raise ValueError("Invalid replay run ID")
    prefix = f"experiments/arc-oat-20260919/{source_run}/"

    def read(name):
        return json.loads(
            client.get_object(Bucket="rldb", Key=prefix + name)["Body"].read()
        )

    try:
        status, result, runtime, spec = (
            read("status.json"),
            read("result.json"),
            read("runtime.json"),
            read("spec.json"),
        )
    except ClientError as error:
        if error.response["Error"]["Code"] in {"NoSuchKey", "404"}:
            raise CalibrationPending(
                "Replay confirmation is not available yet"
            ) from error
        raise
    accepts_measured_gap = (
        status.get("state") == "REPLAY_EVALUATED"
        and spec.get("allow_reference_gap") is True
        and spec.get("selection_objective") == "success_then_tokens"
        and result.get("confirmation_complete") is True
    )
    if not accepts_measured_gap and (
        status.get("state") != "CONFIRMED" or result.get("confirmed") is not True
    ):
        raise CalibrationPending(
            "Replay candidate has not passed independent confirmation"
        )
    if result.get("suite") != suite or runtime.get("suite") != suite:
        raise ValueError("Replay calibration suite differs")
    if (spec.get("horizon"), spec.get("execute_steps"), spec.get("dt")) != (
        32,
        16,
        0.05,
    ):
        raise ValueError("Replay execution protocol differs from policy")
    from egomimic.benchmarks.libero.replay import (
        candidate_id,
        rank_candidates,
        validate_controls,
        validate_spec,
    )

    validate_spec(spec)
    spec_hash = hashlib.sha256((json.dumps(spec, indent=2) + "\n").encode()).hexdigest()
    if (
        spec_hash != result.get("spec_sha256")
        or result.get("selected_before_confirmation") is not True
    ):
        raise ValueError("Replay specification/selection provenance differs")
    codec, selected = result["codec"], result["candidate_id"]
    if (
        spec.get("arc_mode", "joint_dur") != arc_mode
        or codec.get("mode", "joint_dur") != arc_mode
        or result.get("arc_mode", "joint_dur") != arc_mode
    ):
        raise ValueError("Replay ARC mode differs from requested policy")
    if candidate_id(codec) != selected:
        raise ValueError("Replay candidate configuration differs")
    validate_controls(result["confirmation"], spec)
    expected_episodes = len(TASKS[suite]) * len(spec["confirmation_demos"])
    if any(
        value.get("episodes") != expected_episodes
        for value in result["confirmation"].values()
    ):
        raise ValueError("Replay confirmation has incomplete episode coverage")
    if selected not in rank_candidates(result["confirmation"], {selected: codec}, spec):
        raise CalibrationPending(
            "Replay result does not satisfy its confirmation criteria"
        )
    # Training and replay may have different orchestration commits. Verify the
    # actual codec bytes, including older replay receipts without a source hash.
    revision = runtime["source_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Replay source is not pinned")
    subprocess.run(
        ["git", "fetch", "--quiet", "--depth", "1", "origin", revision], check=True
    )
    from egomimic.rldb.zarr.libero_arc_timed import codec_source_files

    paths = (
        codec_source_files(arc_mode)
        if arc_mode != "joint_dur"
        else ["egomimic/rldb/zarr/libero_arc.py"]
    )
    hashes = {}
    for path in paths:
        original = subprocess.check_output(["git", "show", f"{revision}:{path}"])
        hashes[path] = hashlib.sha256(original).hexdigest()
        if hashes[path] != digest(Path(__file__).parents[3] / path):
            raise ValueError("ARC codec changed since replay; recalibration required")
    overrides = {
        "arc_waypoints": int(codec["num_waypoints"]),
        "arc_max_translation": codec["max_translation"],
        "arc_max_rotation_degrees": codec["max_rotation_degrees"],
    }
    if arc_mode != "joint_dur":
        overrides.update(arc_mode=arc_mode, arc_action_dim=12)
    if arc_mode == "stk":
        # Three componentwise [-1,1] controller commands have magnitude <=sqrt(3).
        # Include this in resume compatibility checks as well as the saved config.
        overrides["arc_velocity_norm_bound"] = 3**0.5
    write_json(
        evidence
        / (
            "arc-calibration.json"
            if arc_mode == "joint_dur"
            else f"arc-{arc_mode}-calibration.json"
        ),
        {
            "source_run": source_run,
            "runtime": runtime,
            "result": result,
            "verified_codec_sha256": hashes["egomimic/rldb/zarr/libero_arc.py"],
            "verified_codec_sources": hashes,
            "benchmark_overrides": overrides,
        },
    )
    return overrides


def configure_simulator(root):
    import yaml

    package = Path(os.environ["LIBERO_SOURCE_ROOT"]) / "libero/libero"
    paths = {
        "benchmark_root": package,
        "bddl_files": package / "bddl_files",
        "init_states": package / "init_files",
        "assets": package / "assets",
        "datasets": Path(root) / "data",
    }
    if not all(path.is_dir() for path in paths.values()):
        raise FileNotFoundError(
            "Pinned LIBERO simulator assets or data directory missing"
        )
    config = Path(os.environ["LIBERO_CONFIG_PATH"])
    config.mkdir(parents=True, exist_ok=True)
    (config / "config.yaml").write_text(
        yaml.safe_dump({key: str(value) for key, value in paths.items()})
    )


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--suite", choices=TASKS, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--epochs", type=int, default=5001)
    parser.add_argument(
        "--gpus", type=int, default=int(os.environ.get("TRAINING_GPUS", "1"))
    )
    parser.add_argument(
        "--gpu-type",
        choices=tuple(GPU_PLATFORMS),
        default=os.environ.get("BENCHMARK_GPU_TYPE", "L40S"),
    )
    parser.add_argument("--arc-only", action="store_true")
    parser.add_argument("--arc-profile")
    parser.add_argument(
        "--arc-backbone",
        choices=("unet", "oat_dp"),
        default=os.environ.get("ARC_BACKBONE", "unet"),
    )
    parser.add_argument(
        "--oat-reference-run", default=os.environ.get("OAT_REFERENCE_RUN") or None
    )
    parser.add_argument("--campaign-id", default=os.environ.get("CAMPAIGN_ID") or None)
    parser.add_argument(
        "--campaign-runs",
        type=json.loads,
        default=json.loads(os.environ.get("CAMPAIGN_RUNS_JSON") or "null"),
    )
    parser.add_argument(
        "--evaluate-from-run", default=os.environ.get("EVALUATE_FROM_RUN") or None
    )
    parser.add_argument(
        "--resume-from-run", default=os.environ.get("RESUME_FROM_RUN") or None
    )
    parser.add_argument(
        "--arc-replay-run", default=os.environ.get("ARC_REPLAY_RUN") or None
    )
    parser.add_argument(
        "--arc-modes",
        nargs="+",
        default=json.loads(os.environ.get("ARC_MODES_JSON") or '["dur", "stk"]'),
    )
    parser.add_argument(
        "--arc-replay-runs",
        type=json.loads,
        default=json.loads(os.environ.get("ARC_REPLAY_RUNS_JSON") or "{}"),
    )
    args = parser.parse_args()
    layout = training_layout(args.gpus, args.mode)
    arc_methods = arc_method_modes(args.arc_modes)
    methods = (
        tuple(arc_methods) if args.arc_only else ("tokenizer", "oat", *arc_methods)
    )
    policy_methods = tuple(arc_methods) if args.arc_only else ("oat", *arc_methods)
    profile_overrides = {}
    if args.arc_profile:
        from egomimic.benchmarks.libero.arc_sweep import profile_settings

        if not args.arc_only or len(args.arc_modes) != 1 or args.evaluate_from_run:
            raise ValueError("A sweep profile requires exactly one ARC-only mode")
        profile_overrides = profile_settings(args.arc_profile, args.arc_modes[0])
    if args.arc_only and args.campaign_id:
        raise ValueError("ARC-only runs cannot publish an ARC/OAT campaign as complete")
    if args.oat_reference_run and not re.fullmatch(
        r"[a-z0-9][a-z0-9-]{0,62}", args.oat_reference_run
    ):
        raise ValueError("Invalid OAT reference run ID")
    replay_runs = dict(args.arc_replay_runs)
    if args.arc_replay_run:
        if args.arc_modes != ["joint_dur"]:
            raise ValueError("Use mode-specific replay runs for independent ARC clocks")
        replay_runs["joint_dur"] = args.arc_replay_run
    if set(replay_runs) - set(args.arc_modes):
        raise ValueError("Unexpected replay mode")
    if args.evaluate_from_run and args.resume_from_run:
        raise ValueError("Choose evaluation recovery or training resume")
    if args.campaign_runs is not None and not args.campaign_id:
        raise ValueError("Campaign source manifest requires a campaign ID")
    expected_run = (
        campaign_sources(
            args.campaign_id, os.environ["SOURCE_COMMIT"], args.campaign_runs
        )[args.suite]
        if args.campaign_id
        else None
    )
    if args.campaign_id and (
        args.mode != "full"
        or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,44}", args.campaign_id)
        or args.run_id != expected_run["run_id"]
        or os.environ["SOURCE_COMMIT"] != expected_run["source_commit"]
    ):
        raise ValueError("Campaign requires full mode and matching suite run IDs")
    validate_gpu_allocation(args.gpus, args.gpu_type)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if commit != os.environ["SOURCE_COMMIT"]:
        raise RuntimeError("Unexpected source revision")
    evidence = args.root / "evidence"
    evidence.mkdir(parents=True, exist_ok=False)
    uploader = ArtifactUploader(evidence, args.run_id)
    write_json(
        evidence / "runtime.json",
        {
            "source_commit": commit,
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_type": args.gpu_type,
            "suite": args.suite,
            "mode": args.mode,
            "epochs": 1 if args.mode == "smoke" else args.epochs,
            "global_batch_size": layout["global_batch_size"],
            "training_layout": layout,
            "evaluate_from_run": args.evaluate_from_run,
            "resume_from_run": args.resume_from_run,
            "arc_replay_run": args.arc_replay_run,
            "arc_modes": args.arc_modes,
            "arc_replay_runs": replay_runs,
            "arc_only": args.arc_only,
            "arc_profile": args.arc_profile,
            "arc_backbone": args.arc_backbone,
            "evaluation_workers": 5
            if args.arc_backbone == "oat_dp" and args.mode == "full"
            else 1,
            "oat_reference_run": args.oat_reference_run,
            "campaign_id": args.campaign_id,
            "campaign_runs": args.campaign_runs,
            "artifact_prefix": "s3://rldb/" + uploader.prefix,
        },
    )
    uploader.thread.start()
    try:
        execute(["nvidia-smi"], evidence / "nvidia-smi.log")
        execute(["uv", "pip", "freeze"], evidence / "environment.txt")
        write_json(evidence / "status.json", {"state": "STAGING_DATA"})
        dataset = stage_dataset(args.root, args.suite, evidence)
        if not args.evaluate_from_run:
            from egomimic.rldb.zarr.decoded_replay import prepare_decoded_replay
            from egomimic.rldb.zarr.libero_dataset import keymap

            write_json(evidence / "status.json", {"state": "PREPARING_REPLAY_CACHE"})
            cache = prepare_decoded_replay(
                dataset, [info["zarr_key"] for info in keymap().values()]
            )
            write_json(evidence / "decoded-replay-cache.json", cache.manifest)
        configure_simulator(args.root)
        restored = {}
        if args.evaluate_from_run or args.resume_from_run:
            restored = restore_checkpoints(
                uploader.client,
                args.evaluate_from_run or args.resume_from_run,
                evidence,
                suite=args.suite,
                mode=args.mode,
                epochs=args.epochs,
                allow_partial=bool(args.resume_from_run),
                methods=methods,
            )
        for method in () if args.evaluate_from_run else methods:
            if method in arc_methods and method in restored:
                previous_backbone = restored[method]["benchmark"].get(
                    "arc_backbone", "unet"
                )
                if previous_backbone != args.arc_backbone:
                    raise ValueError("Cannot resume ARC with a different backbone")
            overrides = dict(profile_overrides)
            if method in arc_methods and args.mode == "full":
                arc_mode = arc_methods[method]
                calibrated = load_arc_calibration(
                    uploader.client,
                    replay_runs.get(arc_mode),
                    evidence,
                    suite=args.suite,
                    arc_mode=arc_mode,
                )
                if profile_overrides and any(
                    profile_overrides.get(key) != value
                    for key, value in calibrated.items()
                ):
                    raise ValueError("Replay result differs from frozen sweep profile")
                overrides.update(calibrated)
                if method in restored and any(
                    restored[method]["benchmark"].get(key) != value
                    for key, value in overrides.items()
                ):
                    raise ValueError(
                        "Resumed ARC checkpoint uses different replay parameters"
                    )
            if restored.get(method, {}).get("complete"):
                continue
            write_json(
                evidence / "status.json", {"state": "TRAINING", "method": method}
            )
            argv = training_arguments(
                method,
                args.suite,
                dataset,
                evidence,
                args.mode,
                args.epochs,
                gpus=args.gpus,
                arc_backbone=args.arc_backbone,
            )
            argv += [
                f"benchmark.{key}={value if value is not None else 'null'}"
                for key, value in overrides.items()
            ]
            if method in restored:
                argv.append(
                    f"ckpt_path={evidence / f'training/{method}/checkpoints/last.ckpt'}"
                )
            write_json(evidence / f"{method}-arguments.json", argv)
            execute(argv, evidence / f"{method}-training.log")
            checkpoint = evidence / f"training/{method}/checkpoints/last.ckpt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
        for method in policy_methods:
            write_json(
                evidence / "status.json", {"state": "ROLLOUTS", "method": method}
            )
            if args.arc_backbone == "oat_dp" and args.mode == "full":
                from egomimic.benchmarks.libero.evaluate import parallel_rollouts

                parallel_rollouts(
                    evidence / f"training/{method}/checkpoints/last.ckpt",
                    evidence,
                    method,
                    args.suite,
                )
                continue
            argv = [
                sys.executable,
                "-m",
                "egomimic.benchmarks.libero.cli",
                "rollout",
                "--checkpoint",
                str(evidence / f"training/{method}/checkpoints/last.ckpt"),
                "--output",
                str(evidence / method / args.suite),
            ]
            if args.mode == "smoke":
                argv += [
                    "--trials-per-task",
                    "1",
                    "--repetitions",
                    "1",
                    "--max-episode-steps",
                    "4",
                    "--video-trials",
                    "0",
                ]
            execute(argv, evidence / f"{method}-rollout.log")
        records = {
            method: [
                json.loads(line)
                for line in (evidence / method / args.suite / "episodes.jsonl")
                .read_text()
                .splitlines()
            ]
            for method in policy_methods
        }
        paired = [
            [
                (
                    row["task"],
                    row["repetition"],
                    row["trial"],
                    row["seed"],
                    row["initial_state_sha256"],
                )
                for row in records[method]
            ]
            for method in policy_methods
        ]
        if any(row != paired[0] for row in paired[1:]):
            raise RuntimeError("ARC and OAT did not use identical initial states")
        if args.arc_only:
            from egomimic.benchmarks.libero.report import (
                read_run,
                summarize,
                validate_full_protocol,
            )

            results = {}
            for method, arc_mode in arc_methods.items():
                protocol, episodes = read_run(evidence / method / args.suite)
                if args.mode == "full":
                    validate_full_protocol(protocol)
                if (
                    protocol["method"] != "arc"
                    or protocol["representation"]["mode"] != arc_mode
                ):
                    raise ValueError("ARC-only rollout representation differs")
                if profile_overrides and any(
                    protocol["representation"].get(key) != profile_overrides[field]
                    for key, field in {
                        "waypoints": "arc_waypoints",
                        "max_translation": "arc_max_translation",
                        "max_rotation_degrees": "arc_max_rotation_degrees",
                    }.items()
                ):
                    raise ValueError("Rollout checkpoint differs from frozen profile")
                results[arc_mode] = {
                    "protocol": protocol,
                    "metrics": summarize(episodes),
                }
            write_json(
                evidence / "arc-results.json",
                {
                    "suite": args.suite,
                    "profile": args.arc_profile,
                    "complete_protocol": args.mode == "full",
                    "variants": results,
                    "oat_reference_run": args.oat_reference_run,
                    "paired_oat_comparison_complete": False,
                },
            )
            write_json(
                evidence / "status.json",
                {
                    "state": "ARC_SMOKE_PASSED"
                    if args.mode == "smoke"
                    else "ARC_POLICIES_COMPLETE",
                    "episodes_per_method": len(paired[0]),
                    "benchmark_performance": args.mode == "full",
                    "paired_oat_comparison_complete": False,
                },
            )
            return
        argv = [
            sys.executable,
            "-m",
            "egomimic.benchmarks.libero.cli",
            "reconstruct",
            "--suite",
            args.suite,
            "--dataset",
            str(dataset),
            "--checkpoint",
            str(evidence / "training/tokenizer/checkpoints/last.ckpt"),
            "--output",
            str(evidence / "reconstruction.json"),
            "--arc-modes",
            *args.arc_modes,
        ]
        if args.mode == "smoke":
            argv += ["--limit", "16", "--batch-size", "4"]
        execute(argv, evidence / "reconstruction.log")
        from egomimic.benchmarks.libero.report import compare_runs

        variants = {
            mode: compare_runs(
                evidence / method / args.suite,
                evidence / "oat" / args.suite,
                require_full=args.mode == "full",
                arc_mode=mode,
            )
            for method, mode in arc_methods.items()
        }
        comparison = (
            variants["joint_dur"]
            if args.arc_modes == ["joint_dur"]
            else {
                "suite": args.suite,
                "complete_protocol": args.mode == "full",
                "variants": variants,
                "shared_oat_checkpoint": next(iter(variants.values()))["checkpoints"][
                    "oat"
                ],
            }
        )
        write_json(evidence / "comparison.json", comparison)
        write_json(
            evidence / "status.json",
            {
                "state": "SMOKE_PASSED" if args.mode == "smoke" else "SUITE_COMPLETE",
                "paired_episodes_per_method": len(paired[0]),
                "benchmark_performance": args.mode == "full",
            },
        )
        print("BENCHMARK_RESULT", (evidence / "status.json").read_text(), flush=True)
    except CalibrationPending as error:
        write_json(
            evidence / "status.json",
            {
                "state": "AWAITING_ARC_CALIBRATION",
                "reason": str(error),
                "training_preserved": True,
            },
        )
        print("ARC_TRAINING_GATED", str(error), flush=True)
        return
    except BaseException as error:
        write_json(
            evidence / "status.json",
            {"state": "FAILED", "type": type(error).__name__, "error": str(error)},
        )
        raise
    finally:
        uploader.stop.set()
        uploader.thread.join()
        uploader.upload(final=True)
        # Include checkpoint receipts created during the final pass.
        uploader.upload(final=True)
    if args.campaign_id:
        publish_campaign(
            uploader.client,
            args.campaign_id,
            commit=commit,
            epochs=args.epochs,
            run_sources=args.campaign_runs,
            arc_modes=args.arc_modes,
        )


if __name__ == "__main__":
    main()
