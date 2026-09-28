"""Full-state, immutable HPT checkpoints and the fixed imitation learner recipe."""

import copy
import json
import os
import random
import sqlite3
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils._pytree import tree_map

from egomimic.experiments.astra_push.artifacts import (
    file_hash,
    named_seed,
    publish_json,
)
from egomimic.experiments.astra_push.data import (
    EpisodeBalancedReplay,
    EpisodeWindows,
    ProprioceptionStats,
    collate_windows,
)
from egomimic.experiments.astra_push.init_audit import (
    MODEL_CONFIG,
    construct,
    set_seed,
    tensor_hash,
)
from egomimic.experiments.astra_push.partitions import config
from egomimic.experiments.astra_push.schemas import canonical_hash


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    with temporary.open("xb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()
    receipt = {
        "schema_version": "astrapush-checkpoint-1",
        "path": str(path),
        "sha256": file_hash(path),
        "phase": state["phase"],
        "phase_step": state["phase_step"],
        "total_updates": state["total_updates"],
        "model_state_sha256": tensor_hash(state["model"]),
        "parent_checkpoint": state["parent_checkpoint"],
        "config_sha256": state["config_sha256"],
        "data_manifest_hash": canonical_hash(state["records"]),
        "normalization_hash": canonical_hash(state["normalization"]),
        "source_commit": state["source_commit"],
        "training_operation_id": state.get("training_block", {}).get("operation_id"),
    }
    publish_json(path.with_suffix(".json"), receipt)
    return receipt


def read_checkpoint(receipt):
    if isinstance(receipt, (str, Path)):
        receipt = json.loads(Path(receipt).read_text())
    if file_hash(receipt["path"]) != receipt["sha256"]:
        raise ValueError("Checkpoint differs from its committed content hash")
    # Only this experiment's own hash-verified checkpoints, including Python/NumPy RNGs.
    state = torch.load(receipt["path"], map_location="cpu", weights_only=False)
    if (
        tensor_hash(state["model"]) != receipt["model_state_sha256"]
        or canonical_hash(state["records"]) != receipt["data_manifest_hash"]
    ):
        raise ValueError("Checkpoint model/data provenance differs")
    if state.get("ledger_snapshot"):
        record = state["ledger_snapshot"]
        if file_hash(record["path"]) != record["sha256"]:
            raise ValueError("Checkpoint ledger snapshot changed")
    return state


def recover_training_blocks(output, phase, ledger):
    output = Path(output)
    published = [json.loads(p.read_text()) for p in output.glob("step-*.json")]
    by_operation = {p["training_operation_id"]: p for p in published}
    for row in ledger.connection.execute(
        "SELECT * FROM operations WHERE phase=? AND kind='training_block' AND status='reserved'",
        (phase,),
    ).fetchall():
        receipt = by_operation.get(row["id"])
        if receipt:
            state = read_checkpoint(receipt)
            block = state["training_block"]
            record = {**block, "checkpoint": receipt}
            path = output / f"block-{block['start']:04d}-{block['end']:04d}.json"
            if path.exists():
                if json.loads(path.read_text()) != record:
                    raise ValueError("Committed training log changed")
            else:
                publish_json(path, record)
            ledger.complete(
                row["id"],
                status="succeeded",
                result={
                    "updates": block["end"] - block["start"],
                    "checkpoint_sha256": receipt["sha256"],
                    "seconds": block["seconds"],
                },
            )
        else:
            ledger.complete(
                row["id"],
                status="abandoned",
                result={
                    "reason": "interrupted_unpublished_block",
                    "maximum_updates": config()["checkpoint_interval"],
                },
            )


class Learner:
    def __init__(self, normalization, *, source_commit):
        if not torch.cuda.is_available():
            raise RuntimeError("Production HPT training requires an allocated CUDA GPU")
        self.graph, self.cfg = construct()
        self.initial_hash = tensor_hash(self.graph.nets.state_dict())
        self.source_commit = source_commit
        self.graph.device = torch.device("cuda")
        self.graph.nets.to("cuda")
        self.optimizer = instantiate(self.cfg.optimizer)(self.graph.nets.parameters())
        self.scheduler = instantiate(self.cfg.scheduler)(self.optimizer)
        self.normalization = copy.deepcopy(normalization)
        self.stats = ProprioceptionStats(
            normalization["mean"], normalization["std"], normalization["source_hashes"]
        )
        self.total_updates = 0
        self.parent_checkpoint = None
        self.loaded = None
        self.phase, self.phase_step = "initial", 0
        self.dataset = self.replay = None

    def restore(self, receipt):
        state = read_checkpoint(receipt)
        if (
            state["config_sha256"] != file_hash(MODEL_CONFIG)
            or state["normalization"] != self.normalization
        ):
            raise ValueError(
                "Checkpoint model configuration or frozen normalization changed"
            )
        self.graph.nets.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        restore_rng(state["rng"])
        self.total_updates = state["total_updates"]
        self.phase, self.phase_step = state["phase"], state["phase_step"]
        self.parent_checkpoint = receipt["sha256"]
        self.loaded = state

    def configure_data(self, records, phase, *, arm=None, round_index=None):
        self.dataset = EpisodeWindows(records, self.stats)
        self.replay = EpisodeBalancedReplay(
            self.dataset,
            seed=named_seed(f"sampler:{phase}"),
            arm=arm,
            round_index=round_index,
        )
        if self.loaded and self.loaded["phase"] == phase:
            self.replay.load_state_dict(self.loaded["sampler"])
        else:
            self.phase, self.phase_step = phase, 0
            # Named phase streams keep future training independent of evaluation RNG.
            set_seed(named_seed(f"training:{phase}"))

    def batch(self, indices):
        batch = collate_windows([self.dataset[i] for i in indices])["libero_push"]
        return tree_map(
            lambda x: x.to("cuda") if isinstance(x, torch.Tensor) else x, batch
        )

    def update(self, batch):
        self.graph.nets.train()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.graph.forward_training({"libero_push": batch})
            loss = self.graph.compute_losses(output, {"libero_push": batch})["loss"]
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite production loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.graph.nets.parameters(),
            config()["training"]["gradient_clip_norm"],
            error_if_nonfinite=True,
        )
        self.optimizer.step()
        self.scheduler.step()
        self.total_updates += 1
        self.phase_step += 1
        return {
            "step": self.phase_step,
            "total_updates": self.total_updates,
            "loss": float(loss.detach()),
            "gradient_norm": float(norm),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    def snapshot(self, path, *, ledger, training_block=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path = path.with_suffix(".ledger.sqlite")
        if ledger_path.exists():
            raise ValueError("Checkpoint ledger snapshot already exists")
        with sqlite3.connect(ledger_path) as backup:
            ledger.connection.backup(backup)
        state = {
            "schema_version": "astrapush-learner-state-1",
            "model": self.graph.nets.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "rng": rng_state(),
            "sampler": self.replay.state_dict() if self.replay else None,
            "normalization": self.normalization,
            "records": self.dataset.records if self.dataset else [],
            "phase": self.phase,
            "phase_step": self.phase_step,
            "total_updates": self.total_updates,
            "parent_checkpoint": self.parent_checkpoint,
            "config_sha256": file_hash(MODEL_CONFIG),
            "resolved_model_config": OmegaConf.to_container(self.cfg, resolve=True),
            "source_commit": self.source_commit,
            "initial_state_sha256": self.initial_hash,
            "ledger_summary": ledger.summary(),
            "ledger_snapshot": {
                "path": str(ledger_path),
                "sha256": file_hash(ledger_path),
            },
            "training_block": training_block or {},
        }
        receipt = save_checkpoint(path, state)
        self.parent_checkpoint = receipt["sha256"]
        return receipt

    def train(self, output, *, updates, ledger):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        interval = config()["checkpoint_interval"]
        recover_training_blocks(output, self.phase, ledger)
        latest = None
        while self.phase_step < updates:
            start_step = self.phase_step
            end_step = min(updates, start_step + interval)
            ordinal = ledger.connection.execute(
                "SELECT COUNT(*) FROM operations WHERE phase=? AND kind='training_block'",
                (self.phase,),
            ).fetchone()[0]
            op = f"{self.phase}-training-{ordinal:04d}"
            ledger.reserve(
                op,
                self.phase,
                "training_block",
                {
                    "start": start_step,
                    "end": end_step,
                    "parent": self.parent_checkpoint,
                },
            )
            started = time.monotonic()
            logs = []
            for _ in range(start_step, end_step):
                indices = self.replay.next_indices()
                value = self.update(self.batch(indices))
                value["sample_indices"] = indices
                logs.append(value)
                if self.phase_step % 25 == 0:
                    print(
                        json.dumps(
                            {
                                "event": "training",
                                "phase": self.phase,
                                **{
                                    k: v
                                    for k, v in value.items()
                                    if k != "sample_indices"
                                },
                            }
                        ),
                        flush=True,
                    )
            torch.cuda.synchronize()
            block = {
                "operation_id": op,
                "start": start_step,
                "end": end_step,
                "updates": logs,
                "seconds": time.monotonic() - started,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            }
            latest = self.snapshot(
                output / f"step-{end_step:04d}-{op}.pt",
                ledger=ledger,
                training_block=block,
            )
            publish_json(
                output / f"block-{start_step:04d}-{end_step:04d}.json",
                {**block, "checkpoint": latest},
            )
            ledger.complete(
                op,
                status="succeeded",
                result={
                    "updates": end_step - start_step,
                    "checkpoint_sha256": latest["sha256"],
                    "seconds": block["seconds"],
                },
            )
        if latest is None:
            candidates = [
                json.loads(p.read_text())
                for p in output.glob(f"step-{updates:04d}-*.json")
            ]
            if len(candidates) != 1:
                raise ValueError(
                    "Expected exactly one committed final phase checkpoint"
                )
            latest = candidates[0]
        return latest
