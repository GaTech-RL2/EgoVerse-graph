"""Run native LIBERO training/evaluation in an isolated OSMO L40S container."""

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
    for task in TASKS[suite]:
        source = sources[task + "_demo.hdf5"]
        url = f"https://huggingface.co/datasets/{DATA_REPO}/resolve/{DATA_REVISION}/{source['rfilename']}"
        download(url, data / "raw" / source["rfilename"], source["lfs"]["sha256"])
        receipts.append(source)
    replay = data / f"{suite}.zarr"
    convert_suite(data / "raw" / suite, replay, suite)
    write_json(
        evidence / "data-source.json",
        {"repo": DATA_REPO, "revision": DATA_REVISION, "files": receipts},
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


def training_arguments(method, suite, dataset, evidence, mode, epochs):
    experiment = {
        "tokenizer": "libero_oattok",
        "oat": "libero_oatpolicy",
        "arc": "libero_arc_policy",
    }[method]
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
        "trainer.devices=1",
        "trainer.precision=bf16-mixed",
        f"trainer.max_epochs={epochs}",
        "callbacks.model_checkpoint.save_top_k=1",
        "++trainer.enable_progress_bar=false",
    ]
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


def publish_campaign(client, campaign_id, *, commit, epochs):
    """The last completed suite publishes scores after checking every receipt."""
    from botocore.exceptions import ClientError

    results, sources = {}, {}
    for suite in TASKS:
        run_id = f"{campaign_id}-{suite.replace('_', '-')}"
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
            runtime["source_commit"] != commit
            or runtime["mode"] != "full"
            or runtime["epochs"] != epochs
            or runtime.get("global_batch_size") != 1024
            or runtime["suite"] != suite
            or result["suite"] != suite
            or not result["complete_protocol"]
        ):
            raise ValueError("Campaign source, training budget or suite differs")
        results[suite] = result
        sources[suite] = "s3://rldb/" + prefix
    report = {
        "complete_benchmark_suite": True,
        "unique_tasks": sum(map(len, TASKS.values())),
        "source_commit": commit,
        "training_epochs": epochs,
        "global_batch_size": 1024,
        "full_released_training_budget": epochs == 5001,
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


def restore_checkpoints(
    client, source_run, evidence, *, suite, mode, epochs, allow_partial=False
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
    for method in ("tokenizer", "oat", "arc"):
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
        if method == "arc" and allow_partial:
            restored[method]["benchmark"] = payload["hyper_parameters"]["config_tree"][
                "benchmark"
            ]
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


def load_arc_calibration(client, source_run, evidence, *, suite):
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
    if status.get("state") != "CONFIRMED" or result.get("confirmed") is not True:
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
    if candidate_id(codec) != selected:
        raise ValueError("Replay candidate configuration differs")
    validate_controls(result["confirmation"], spec)
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
    path = "egomimic/rldb/zarr/libero_arc.py"
    original = subprocess.check_output(["git", "show", f"{revision}:{path}"])
    if hashlib.sha256(original).hexdigest() != digest(
        Path(__file__).parents[2] / "rldb/zarr/libero_arc.py"
    ):
        raise ValueError("ARC codec changed since replay; recalibration required")
    overrides = {
        "arc_waypoints": int(codec["num_waypoints"]),
        "arc_max_translation": codec["max_translation"],
        "arc_max_rotation_degrees": codec["max_rotation_degrees"],
    }
    write_json(
        evidence / "arc-calibration.json",
        {
            "source_run": source_run,
            "runtime": runtime,
            "result": result,
            "verified_codec_sha256": hashlib.sha256(original).hexdigest(),
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
    parser.add_argument("--campaign-id", default=os.environ.get("CAMPAIGN_ID") or None)
    parser.add_argument(
        "--evaluate-from-run", default=os.environ.get("EVALUATE_FROM_RUN") or None
    )
    parser.add_argument(
        "--resume-from-run", default=os.environ.get("RESUME_FROM_RUN") or None
    )
    parser.add_argument(
        "--arc-replay-run", default=os.environ.get("ARC_REPLAY_RUN") or None
    )
    args = parser.parse_args()
    if args.evaluate_from_run and args.resume_from_run:
        raise ValueError("Choose evaluation recovery or training resume")
    if args.campaign_id and (
        args.mode != "full"
        or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,44}", args.campaign_id)
        or args.run_id != f"{args.campaign_id}-{args.suite.replace('_', '-')}"
    ):
        raise ValueError("Campaign requires full mode and matching suite run IDs")
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or "L40S" not in torch.cuda.get_device_name(0)
    ):
        raise RuntimeError("This workflow requests exactly one L40S GPU")
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
            "suite": args.suite,
            "mode": args.mode,
            "epochs": 1 if args.mode == "smoke" else args.epochs,
            "global_batch_size": 4 if args.mode == "smoke" else 1024,
            "evaluate_from_run": args.evaluate_from_run,
            "resume_from_run": args.resume_from_run,
            "arc_replay_run": args.arc_replay_run,
            "campaign_id": args.campaign_id,
            "artifact_prefix": "s3://rldb/" + uploader.prefix,
        },
    )
    uploader.thread.start()
    try:
        execute(["nvidia-smi"], evidence / "nvidia-smi.log")
        execute(["uv", "pip", "freeze"], evidence / "environment.txt")
        write_json(evidence / "status.json", {"state": "STAGING_DATA"})
        dataset = stage_dataset(args.root, args.suite, evidence)
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
            )
        for method in () if args.evaluate_from_run else ("tokenizer", "oat", "arc"):
            overrides = {}
            if method == "arc" and args.mode == "full":
                overrides = load_arc_calibration(
                    uploader.client, args.arc_replay_run, evidence, suite=args.suite
                )
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
                method, args.suite, dataset, evidence, args.mode, args.epochs
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
        for method in ("oat", "arc"):
            write_json(
                evidence / "status.json", {"state": "ROLLOUTS", "method": method}
            )
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
            for method in ("oat", "arc")
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
            for method in ("oat", "arc")
        ]
        if paired[0] != paired[1]:
            raise RuntimeError("ARC and OAT did not use identical initial states")
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
        ]
        if args.mode == "smoke":
            argv += ["--limit", "16", "--batch-size", "4"]
        execute(argv, evidence / "reconstruction.log")
        from egomimic.benchmarks.libero.report import compare_runs

        write_json(
            evidence / "comparison.json",
            compare_runs(
                evidence / "arc" / args.suite,
                evidence / "oat" / args.suite,
                require_full=args.mode == "full",
            ),
        )
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
            uploader.client, args.campaign_id, commit=commit, epochs=args.epochs
        )


if __name__ == "__main__":
    main()
