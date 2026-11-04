"""Upload finished rollout episodes to R2 and register them in SQL as evals.

The rollout loop never touches the network. When ``episode_upload.enabled`` is
set it starts this module as a separate, low-priority worker process that
watches the episode directory. A take recorded with upload enabled carries an
``upload`` block (the SQL contract) in its manifest. Each such complete take
goes up once, in this order, so the raw_v2 cron can never register it as an
ordinary (training) episode:

1. ``<prefix>/<ms>.hdf5``. With no metadata JSON yet, the cron ignores it.
2. The ``app.episodes`` row: ``is_eval`` true, ``eval_success`` /
   ``eval_score`` from the take's labels (unknown stays NULL).
3. ``<prefix>/<ms>_rollout.json``: the manifest (checkpoint, W&B run,
   station commit) and the labels.
4. ``<prefix>/<ms>_metadata.json``: the usual contract. The cron then finds the
   row already present and skips it.

``<ms>`` is the take's start time in UTC epoch milliseconds, and the SQL
``episode_hash`` is the same instant. Progress lives in
``<episode>.upload.json`` beside the take, so a restart resumes where it
stopped. Saving labels later updates only the eval columns of that row (never
a row that is not ``is_eval``) and re-uploads the rollout JSON. An existing
object or row that this worker did not create is never overwritten.

Run by hand, e.g. to look first or to backfill one take::

    python -m egomimic.robot.rollout_upload status <episode dir>
    python -m egomimic.robot.rollout_upload once <episode dir> --config <rollout yaml>
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import threading
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from egomimic.robot.rollout_episode import (
    is_episode_id,
    list_rollout_episodes,
    read_episode_labels,
)
from egomimic.utils.episode_identity import timestamp_ms_to_episode_hash

UPLOAD_SCHEMA = "rl2_yam.rollout_upload.v1"
EPISODE_UPLOAD_DEFAULTS = {
    "enabled": False,
    "bucket": "rldb",
    "prefix": "raw_v2/yam",
    # SQL contract (as the GELLO uploads of the same task use it).
    "operator": "",
    "lab": "rl2",
    "embodiment": "yam_bimanual",
    "rig_name": "rl2_abc",
    "scene": "rl2 yam setup",
    "task": "",
    "task_description": "",
    "objects": None,
    "poll_seconds": 5.0,
}
# How the GELLO uploads of each RL2 task register them in app.episodes (read
# 2026-11-04); a profile that names one of these tasks inherits the rest.
TASK_CONTRACTS = {
    "organize_stationary": {
        "task_description": "sorting markers and pens",
        "objects": None,
    },
    "fold_towels": {"task_description": "fold towels", "objects": None},
    "wireless_earbuds": {
        "task_description": "place earbuds in earbuds case",
        "objects": ["earbuds"],
    },
}
_CONTRACT = (
    "operator",
    "lab",
    "embodiment",
    "rig_name",
    "scene",
    "task",
    "task_description",
    "objects",
)
_STEPS = ("hdf5", "sql", "rollout_json", "metadata")
_RETRY_SECONDS = (30, 60, 120, 300, 600)


def validate_episode_upload(config: Mapping[str, object] | None) -> dict:
    """Validate the upload policy without touching the network."""
    config = {} if config is None else dict(config)
    unknown = config.keys() - EPISODE_UPLOAD_DEFAULTS.keys()
    if unknown:
        raise ValueError(
            "Unknown episode_upload option(s): " + ", ".join(sorted(unknown))
        )
    result = {**deepcopy(EPISODE_UPLOAD_DEFAULTS), **config}
    if type(result["enabled"]) is not bool:
        raise ValueError("episode_upload.enabled must be a boolean")
    known = TASK_CONTRACTS.get(str(result["task"]).strip())
    if known is not None:
        if not str(result["task_description"]).strip():
            result["task_description"] = known["task_description"]
        if "objects" not in config:
            result["objects"] = deepcopy(known["objects"])
    for name in ("bucket", "prefix", *_CONTRACT[:-1]):
        value = result[name]
        if not isinstance(value, str):
            raise ValueError(f"episode_upload.{name} must be a string")
        result[name] = value.strip()
        if result["enabled"] and not result[name]:
            raise ValueError(f"episode_upload.{name} is required when enabled")
    result["prefix"] = result["prefix"].strip("/")
    objects = result["objects"]
    if objects is not None and not (
        isinstance(objects, str)
        or (
            isinstance(objects, (list, tuple))
            and all(isinstance(item, str) and item for item in objects)
        )
    ):
        raise ValueError("episode_upload.objects must be null, a string or a list")
    result["objects"] = list(objects) if isinstance(objects, tuple) else objects
    poll = result["poll_seconds"]
    if type(poll) not in (int, float) or not 0.5 <= poll <= 300:
        raise ValueError("episode_upload.poll_seconds must be in [0.5, 300]")
    return result


def upload_contract(config: Mapping[str, object]) -> dict:
    """The SQL contract frozen into a take's manifest when it starts recording."""
    return {name: deepcopy(config[name]) for name in _CONTRACT}


def episode_timestamp_ms(episode_id: str) -> int:
    """The take's start (from its ID) as UTC epoch milliseconds."""
    if not is_episode_id(episode_id):
        raise ValueError(f"Not a rollout episode ID: {episode_id!r}")
    started = datetime.strptime(episode_id[len("rollout_") :], "%Y%m%d-%H%M%S-%f")
    started = started.replace(tzinfo=timezone.utc)
    return int(started.timestamp()) * 1000 + started.microsecond // 1000


def eval_fields(episode: Mapping) -> tuple[bool | None, float | None]:
    """(eval_success, eval_score) from the labels, else the recorded outcome."""
    labels = episode.get("labels") or {}
    outcome = labels.get("outcome") or episode.get("recorded_outcome")
    success = {"success": True, "failure": False}.get(outcome)
    score = labels.get("partial_score")
    return success, None if score is None else float(score)


def _sql_objects(objects):
    if isinstance(objects, list):
        return "{" + ",".join(objects) + "}"  # the text app.episodes holds
    return objects


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _upload_path(directory: Path, episode_id: str) -> Path:
    return directory / f"{episode_id}.upload.json"


def read_upload_state(directory: str | Path, episode_id: str) -> dict | None:
    path = _upload_path(Path(directory), episode_id)
    if not path.is_file():
        return None
    try:
        state = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return state if state.get("schema") == UPLOAD_SCHEMA else None


def _write_state(directory: Path, state: dict) -> None:
    state["updated_at"] = _now()
    path = _upload_path(directory, state["id"])
    temporary = directory / f".{state['id']}.upload.partial.json"
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


class S3Store:
    """The R2 bucket through boto3, with the station's ~/.egoverse_env keys."""

    def __init__(self, client=None):
        if client is None:
            from egomimic.utils.aws.aws_data_utils import get_boto3_s3_client

            client = get_boto3_s3_client()
        self.client = client

    def size(self, bucket: str, key: str) -> int | None:
        from botocore.exceptions import ClientError

        try:
            return int(self.client.head_object(Bucket=bucket, Key=key)["ContentLength"])
        except ClientError as error:
            status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            code = str(error.response.get("Error", {}).get("Code", ""))
            if status == 404 or code in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise

    def upload_file(self, path: Path, bucket: str, key: str) -> None:
        from boto3.s3.transfer import TransferConfig

        config = TransferConfig(
            multipart_threshold=64 * 2**20,
            multipart_chunksize=64 * 2**20,
            max_concurrency=4,
            use_threads=True,
        )
        self.client.upload_file(str(path), bucket, key, Config=config)

    def put_json(self, bucket: str, key: str, payload: Mapping) -> None:
        body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        self.client.put_object(
            Bucket=bucket, Key=key, Body=body, ContentType="application/json"
        )


class SqlRegistry:
    """``app.episodes`` through SQLAlchemy Core; touches eval columns only."""

    def __init__(self, engine=None):
        if engine is None:
            from egomimic.utils.aws.aws_sql import create_default_engine

            engine = create_default_engine()
        from sqlalchemy import MetaData, Table

        self.engine = engine
        self.table = Table("episodes", MetaData(), autoload_with=engine, schema="app")

    def get(self, episode_hash: str) -> dict | None:
        from sqlalchemy import select

        t = self.table
        stmt = select(t.c.episode_hash, t.c.is_eval, t.c.task, t.c.operator).where(
            t.c.episode_hash == episode_hash
        )
        with self.engine.connect() as connection:
            row = connection.execute(stmt).first()
        return None if row is None else dict(row._mapping)

    def insert(self, row: Mapping) -> None:
        from sqlalchemy import insert

        with self.engine.begin() as connection:
            connection.execute(insert(self.table).values(**row))

    def update_eval(self, episode_hash: str, success, score) -> int:
        from sqlalchemy import update

        t = self.table
        stmt = (
            update(t)
            .where(t.c.episode_hash == episode_hash, t.c.is_eval.is_(True))
            .values(eval_success=success, eval_score=score)
        )
        with self.engine.begin() as connection:
            return connection.execute(stmt).rowcount


class EpisodeUploader:
    """Advance every eligible take by its upload steps; safe to rerun."""

    def __init__(self, directory, config: Mapping, store=None, registry=None):
        self.directory = Path(directory)
        self.config = validate_episode_upload(config)
        self._store, self._registry = store, registry

    @property
    def store(self):
        if self._store is None:
            self._store = S3Store()
        return self._store

    @property
    def registry(self):
        if self._registry is None:
            self._registry = SqlRegistry()
        return self._registry

    def eligible(self) -> list[dict]:
        """Complete takes recorded with upload requested, oldest first."""
        episodes = [
            episode
            for episode in list_rollout_episodes(self.directory)
            if isinstance(episode.get("upload"), Mapping)
            and episode["upload"].get("requested") is True
            and episode.get("complete") is True
        ]
        return sorted(episodes, key=lambda episode: episode["id"])

    def _keys(self, timestamp_ms: int) -> dict:
        prefix = self.config["prefix"]
        return {
            "hdf5": f"{prefix}/{timestamp_ms}.hdf5",
            "rollout_json": f"{prefix}/{timestamp_ms}_rollout.json",
            "metadata": f"{prefix}/{timestamp_ms}_metadata.json",
        }

    def _new_state(self, episode: Mapping) -> dict:
        timestamp_ms = episode_timestamp_ms(episode["id"])
        contract = dict(episode["upload"]["contract"])
        return {
            "schema": UPLOAD_SCHEMA,
            "id": episode["id"],
            "timestamp_ms": timestamp_ms,
            "episode_hash": timestamp_ms_to_episode_hash(timestamp_ms),
            "bucket": self.config["bucket"],
            "keys": self._keys(timestamp_ms),
            "contract": contract,
            "steps": {},
            "labels_revision_synced": None,
            "eval": None,
            "error": None,
            "attempts": 0,
            "next_attempt_at": 0.0,
        }

    def _rollout_payload(self, episode: Mapping, state: Mapping) -> dict:
        record = read_episode_labels(self.directory, episode["id"])
        manifest = json.loads((self.directory / f"{episode['id']}.json").read_text())
        return {
            "schema": "rl2_yam.rollout_episode.v1",
            "episode_hash": state["episode_hash"],
            "is_eval": True,
            "manifest": manifest,
            "labels": record,
        }

    def _metadata(self, state: Mapping) -> dict:
        contract = state["contract"]
        return {
            **{name: contract[name] for name in _CONTRACT},
            "episode_hash": state["timestamp_ms"],
            "is_eval": True,
        }

    def process(self, episode: Mapping, now: float | None = None) -> str:
        """Advance one take; return 'done', 'waiting', 'synced' or 'error'."""
        now = time.time() if now is None else now
        state = read_upload_state(self.directory, episode["id"])
        if state is None:
            state = self._new_state(episode)
        if state.get("next_attempt_at", 0) > now:
            return "waiting"
        try:
            if any(not state["steps"].get(step) for step in _STEPS):
                self._upload(episode, state)
                result = "done"
            else:
                result = self._sync_labels(episode, state)
        except Exception as error:  # every failure is recorded and retried
            state["attempts"] = int(state.get("attempts", 0)) + 1
            delay = _RETRY_SECONDS[min(state["attempts"], len(_RETRY_SECONDS)) - 1]
            state["error"] = f"{type(error).__name__}: {error}"
            state["next_attempt_at"] = now + delay
            _write_state(self.directory, state)
            print(
                f"Rollout upload {episode['id']} failed ({state['error']}); "
                f"retrying in {delay} s"
            )
            return "error"
        state["error"], state["attempts"], state["next_attempt_at"] = None, 0, 0.0
        _write_state(self.directory, state)
        return result

    def _upload(self, episode: Mapping, state: dict) -> None:
        bucket, keys, steps = state["bucket"], state["keys"], state["steps"]
        path = self.directory / episode["filename"]
        size = path.stat().st_size
        if not steps.get("hdf5"):
            # Never overwrite an object this worker did not create: a takeover
            # of a pre-existing key would replace someone else's episode.
            if self.store.size(bucket, keys["metadata"]) is not None:
                raise RuntimeError(f"{keys['metadata']} already exists")
            existing = self.store.size(bucket, keys["hdf5"])
            if existing is not None and not state.get("hdf5_started"):
                raise RuntimeError(f"{keys['hdf5']} already exists")
            state["hdf5_started"] = True
            _write_state(self.directory, state)
            if existing != size:
                self.store.upload_file(path, bucket, keys["hdf5"])
            uploaded = self.store.size(bucket, keys["hdf5"])
            if uploaded != size:
                raise RuntimeError(f"uploaded size {uploaded} != local {size}")
            steps["hdf5"] = {"size": size, "at": _now()}
            _write_state(self.directory, state)
            print(f"Uploaded {episode['id']} to r2://{bucket}/{keys['hdf5']}")
        success, score = eval_fields(episode)
        if not steps.get("sql"):
            row = self.registry.get(state["episode_hash"])
            ours = state.get("sql_started") and row is not None
            if row is None:
                # Mark before inserting, so a crash after the insert resumes
                # instead of mistaking this worker's own row for a collision.
                state["sql_started"] = True
                _write_state(self.directory, state)
                contract = state["contract"]
                self.registry.insert(
                    {
                        "episode_hash": state["episode_hash"],
                        **{name: contract[name] for name in _CONTRACT},
                        "objects": _sql_objects(contract["objects"]),
                        "num_frames": -1,
                        "is_deleted": False,
                        "is_eval": True,
                        "eval_success": success,
                        "eval_score": score,
                    }
                )
            elif not (
                ours
                and row.get("is_eval") is True
                and row.get("task") == state["contract"]["task"]
                and row.get("operator") == state["contract"]["operator"]
            ):
                raise RuntimeError(
                    f"app.episodes already has {state['episode_hash']} "
                    f"(is_eval={row.get('is_eval')}, task={row.get('task')})"
                )
            else:
                self.registry.update_eval(state["episode_hash"], success, score)
            steps["sql"] = {"at": _now()}
            state["eval"] = {"success": success, "score": score}
            state["labels_revision_synced"] = episode.get("labels_revision", 0)
            _write_state(self.directory, state)
            print(
                f"Registered {episode['id']} as eval episode {state['episode_hash']} "
                f"(task {state['contract']['task']}, success {success}, score {score})"
            )
        if not steps.get("rollout_json"):
            self.store.put_json(
                bucket, keys["rollout_json"], self._rollout_payload(episode, state)
            )
            steps["rollout_json"] = {"at": _now()}
            _write_state(self.directory, state)
        if not steps.get("metadata"):
            self.store.put_json(bucket, keys["metadata"], self._metadata(state))
            steps["metadata"] = {"at": _now()}

    def _sync_labels(self, episode: Mapping, state: dict) -> str:
        revision = episode.get("labels_revision", 0)
        if revision == state.get("labels_revision_synced"):
            return "done"
        success, score = eval_fields(episode)
        if self.registry.update_eval(state["episode_hash"], success, score) != 1:
            raise RuntimeError(f"no is_eval row {state['episode_hash']} to update")
        self.store.put_json(
            state["bucket"],
            state["keys"]["rollout_json"],
            self._rollout_payload(episode, state),
        )
        state["eval"] = {"success": success, "score": score}
        state["labels_revision_synced"] = revision
        print(
            f"Synced labels of {episode['id']} (revision {revision}): "
            f"success {success}, score {score}"
        )
        return "synced"

    def plan(self, episode: Mapping) -> dict:
        """Read-only: where a take would go and whether its keys and row are free."""
        timestamp_ms = episode_timestamp_ms(episode["id"])
        episode_hash = timestamp_ms_to_episode_hash(timestamp_ms)
        keys = self._keys(timestamp_ms)
        upload = episode.get("upload")
        contract = (
            upload["contract"]
            if isinstance(upload, Mapping) and upload.get("contract")
            else upload_contract(self.config)
        )
        success, score = eval_fields(episode)
        return {
            "id": episode["id"],
            "episode_hash": episode_hash,
            "requested": isinstance(upload, Mapping)
            and upload.get("requested") is True,
            "complete": episode.get("complete"),
            "contract": contract,
            "eval_success": success,
            "eval_score": score,
            "r2": {
                key: self.store.size(self.config["bucket"], key)
                for key in keys.values()
            },
            "sql_row": self.registry.get(episode_hash),
        }

    def run_once(self, now: float | None = None) -> dict:
        counts: dict[str, int] = {}
        for episode in self.eligible():
            result = self.process(episode, now=now)
            counts[result] = counts.get(result, 0) + 1
        return counts

    def pending(self) -> bool:
        for episode in self.eligible():
            state = read_upload_state(self.directory, episode["id"])
            if state is None or any(not state["steps"].get(s) for s in _STEPS):
                return True
            if episode.get("labels_revision", 0) != state.get("labels_revision_synced"):
                return True
        return False


def upload_status(directory: str | Path) -> list[dict]:
    """One row per take recorded with upload requested, newest first."""
    rows = []
    for episode in list_rollout_episodes(directory):
        upload = episode.get("upload")
        if not isinstance(upload, Mapping) or upload.get("requested") is not True:
            continue
        state = read_upload_state(directory, episode["id"]) or {}
        steps = state.get("steps", {})
        done = all(steps.get(step) for step in _STEPS)
        rows.append(
            {
                "id": episode["id"],
                "complete": episode.get("complete"),
                "episode_hash": state.get("episode_hash"),
                "state": (
                    "skipped (incomplete take)"
                    if not episode.get("complete")
                    else "uploaded"
                    if done
                    else "error"
                    if state.get("error")
                    else "pending"
                ),
                "error": state.get("error"),
                "eval": state.get("eval"),
            }
        )
    return rows


def _stdin_watcher(gone: threading.Event) -> None:
    try:
        while sys.stdin.buffer.read(4096):
            pass
    except (OSError, ValueError):
        pass
    gone.set()


DRAIN_SECONDS = 1800.0


def run_worker(directory: str | Path, config: Mapping, watch_parent: bool) -> None:
    """Upload until the rollout exits (stdin closes) and nothing is pending.

    After the rollout exits it keeps draining for at most ``DRAIN_SECONDS``;
    whatever is still pending then resumes with the next rollout's worker.
    """
    directory = Path(directory)
    config = validate_episode_upload(config)
    try:
        os.nice(10)
    except OSError:
        pass
    directory.mkdir(parents=True, exist_ok=True)
    gone = threading.Event()
    if watch_parent:
        threading.Thread(target=_stdin_watcher, args=(gone,), daemon=True).start()
    with open(directory / ".rollout_upload.lock", "w") as lock:
        # A worker from the previous session may still be draining; wait for it.
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if gone.is_set():
                    return
                time.sleep(1.0)
        uploader = EpisodeUploader(directory, config)
        print(f"Rollout upload worker watching {directory} (pid {os.getpid()})")
        gone_at = None
        while True:
            try:
                uploader.run_once()
                pending = uploader.pending()
            except Exception as error:  # e.g. credentials; keep the rollout alive
                print(f"Rollout upload worker error: {type(error).__name__}: {error}")
                pending = True
            if gone.is_set():
                gone_at = time.monotonic() if gone_at is None else gone_at
                if not pending:
                    return
                if time.monotonic() - gone_at > DRAIN_SECONDS:
                    print("Rollout upload worker: stopping with uploads pending")
                    return
                time.sleep(min(config["poll_seconds"], 5.0))
            else:
                gone.wait(config["poll_seconds"])


def start_worker(directory: str | Path, config: Mapping):
    """Start the worker beside a rollout; it drains after the rollout exits."""
    import subprocess

    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "egomimic.robot.rollout_upload",
            "worker",
            str(directory),
            "--config-json",
            json.dumps(validate_episode_upload(config)),
        ],
        stdin=subprocess.PIPE,
        close_fds=True,
    )


def _load_config(path: str) -> dict:
    from omegaconf import OmegaConf

    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    return config.get("episode_upload") or {}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status", help="show upload state; no network")
    status.add_argument("directory")
    once = sub.add_parser("once", help="advance every eligible take once")
    once.add_argument("directory")
    once.add_argument("--config", required=True, help="rollout YAML")
    plan = sub.add_parser("plan", help="read-only dry run against R2 and SQL")
    plan.add_argument("directory")
    plan.add_argument("--config", required=True, help="rollout YAML")
    plan.add_argument("--episode", action="append", help="only these take IDs")
    worker = sub.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("directory")
    worker.add_argument("--config-json", required=True)
    args = parser.parse_args(argv)
    if args.command == "status":
        for row in upload_status(args.directory):
            print(json.dumps(row))
    elif args.command == "plan":
        uploader = EpisodeUploader(args.directory, _load_config(args.config))
        for episode in list_rollout_episodes(args.directory):
            if args.episode and episode["id"] not in args.episode:
                continue
            if not args.episode and episode not in uploader.eligible():
                continue
            print(json.dumps(uploader.plan(episode), default=str))
    elif args.command == "once":
        print(
            json.dumps(
                EpisodeUploader(args.directory, _load_config(args.config)).run_once()
            )
        )
    else:
        run_worker(args.directory, json.loads(args.config_json), watch_parent=True)


if __name__ == "__main__":
    main()
