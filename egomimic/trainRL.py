"""Configuration-driven offline RL using PipelineAlgo and ModelWrapper.

Example: python -m egomimic.trainRL --config /path/to/resolved-run.yaml
This adapter supplies transition replay and environment evaluation to the same
graph/Lightning runtime used by supervised robot policies.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import hydra
import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CSVLogger, WandbLogger
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from egomimic.eval.goal_rollout import evaluate_goals
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.goal_replay import ShardedGoalReplay, sha256_file
from egomimic.utils.experiment_artifacts import ArtifactWriter


class RLReceipts(Callback):
    def __init__(self, cfg, replay, writer):
        self.cfg, self.replay, self.writer = cfg, replay, writer
        self.start_time, self.start_step = time.monotonic(), replay.start_step
        self.resume_rng = None
        self.evaluated_steps = set()
        self.local_checkpoints = []

    def on_save_checkpoint(self, trainer, module, checkpoint):
        checkpoint["replay_next_update"] = trainer.global_step
        checkpoint["rl_rng"] = {"torch": torch.get_rng_state(),
                                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

    def on_load_checkpoint(self, trainer, module, checkpoint):
        self.resume_rng = checkpoint["rl_rng"]

    def on_train_start(self, trainer, module):
        if self.resume_rng:
            torch.set_rng_state(self.resume_rng["torch"].cpu())
            if self.resume_rng["cuda"]:
                torch.cuda.set_rng_state_all([v.cpu() for v in self.resume_rng["cuda"]])

    def checkpoint(self, trainer):
        step = trainer.global_step
        path = self.writer.directory / "checkpoints" / f"step-{step:09d}.ckpt"
        path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(path)
        receipt = self.writer.publish(path)
        self.writer.json("latest-checkpoint.json", {"step": step, **receipt})
        self.local_checkpoints.append((path, receipt))
        keep = int(self.cfg.get("local_checkpoint_keep", 2))
        if self.writer.client and keep > 0:
            # These are only scratch files this callback created and uploaded.
            # Keep every remote checkpoint and leave pre-existing files alone.
            for previous, published in self.local_checkpoints[:-keep]:
                key = self.writer.prefix + "/" + str(previous.relative_to(self.writer.directory))
                remote = self.writer.client.head_object(Bucket=self.writer.bucket, Key=key)
                if (previous.is_symlink() or sha256_file(previous) != published["sha256"]
                        or remote["ContentLength"] != published["bytes"]
                        or remote.get("Metadata", {}).get("sha256") != published["sha256"]):
                    raise RuntimeError("refusing to evict a checkpoint without its verified durable copy")
                previous.unlink()
            self.local_checkpoints = self.local_checkpoints[-keep:]

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        step = trainer.global_step
        if not torch.isfinite(outputs["loss"]).all():
            raise FloatingPointError("nonfinite RL loss")
        if step == self.start_step + 1 or step % self.cfg.log_interval == 0:
            elapsed = time.monotonic() - self.start_time
            rate = (step - self.start_step) / elapsed
            status = {"state": "TRAINING", "step": step, "loss": float(outputs["loss"]),
                      "updates_per_second": rate, "elapsed_seconds": elapsed,
                      "estimated_training_seconds_remaining": (self.cfg.steps - step) / max(rate, 1e-9),
                      "loaded_shards": self.replay.receipts}
            self.writer.json("status.json", status)
            print(json.dumps({k: v for k, v in status.items() if k != "loaded_shards"}), flush=True)
        if step % self.cfg.checkpoint_interval == 0 or step == self.cfg.steps:
            self.checkpoint(trainer)
        if self.cfg.eval_interval and (step % self.cfg.eval_interval == 0 or step == self.cfg.steps):
            self.evaluate(trainer, module)

    def evaluate(self, trainer, module):
        import ogbench
        step = trainer.global_step
        env = ogbench.make_env_and_datasets(self.cfg.env_name, env_only=True)
        try:
            directory = self.writer.directory / "evaluation" / f"step-{step:09d}"
            scores = evaluate_goals(module.model, env, directory,
                episodes=self.cfg.eval_episodes, seed_start=self.cfg.eval_seed_start,
                task_ids=self.cfg.get("eval_task_ids"), video_episodes=self.cfg.video_episodes)
            for path in sorted(directory.iterdir()):
                self.writer.publish(path)
            for logger in trainer.loggers:
                logger.log_metrics({"evaluation/success": scores["success"],
                    **{f"evaluation/task{k}": v for k, v in scores["tasks"].items()}}, step=step)
            self.evaluated_steps.add(step)
        finally:
            env.close()


def train(cfg):
    L.seed_everything(cfg.seed, workers=True)
    torch.set_num_threads(int(cfg.get("torch_threads", 4)))
    torch.set_float32_matmul_precision(cfg.get("matmul_precision", "highest"))
    writer = ArtifactWriter(cfg.output_dir, **OmegaConf.to_container(cfg.artifacts))
    resume = cfg.get("resume")
    start = 0
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        start = int(checkpoint["global_step"])
        if checkpoint["replay_next_update"] != start:
            raise ValueError("resume replay/optimizer alignment mismatch")
        del checkpoint
    replay = ShardedGoalReplay(cfg.env_name, cfg.shards, cfg.cache_dir, cfg.seed,
                               cfg.batch_size, cfg.steps, cfg.replace_interval,
                               start_step=start, backup_horizon=cfg.backup_horizon,
                               discount=cfg.discount, cache_keep_shards=cfg.get("cache_keep_shards", 128),
                               data_registry=cfg.get("data_registry"),
                               include_future_observations=cfg.get("include_future_observations", False))
    first = replay.load_shard((start // cfg.replace_interval) % len(cfg.shards))
    example = first.sample(1, np.random.RandomState(0))
    del first
    cfg.observation_dim = int(example["observations"].shape[-1])
    cfg.goal_dim = int(example["high_value_goals"].shape[-1])
    cfg.action_dim = int(example["high_value_action_chunks"].shape[-1])
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.get("require_gpu", True) and cfg.device != "cuda":
        raise RuntimeError("GPU training requires an allocated GPU")
    resolved = OmegaConf.to_container(cfg, resolve=True)
    config_path = writer.directory / "resolved-config.yaml"
    OmegaConf.save(OmegaConf.create(resolved), config_path)
    config_receipt = writer.publish(config_path)
    # Data are already normalized by the environment: no learned dataset stats.
    writer.json("normalizer.json", {"mode": "environment_action_bounds",
        "native_action_low": -1, "native_action_high": 1,
        "clip_training_epsilon": 1e-5, "codec": resolved["codec"],
        "observation_transform": "identity", "goal_transform": "ogbench_oracle_rep"})
    source_commit = os.environ.get("SOURCE_COMMIT")
    if source_commit is None:
        source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    writer.json("runtime.json", {"source_commit": source_commit,
        "source_archive_sha256": os.environ.get("SOURCE_SHA256"), "config": config_receipt,
        "torch": torch.__version__, "lightning": L.__version__, "device": cfg.device,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "resume_step": start, "resolved_native_backup_horizon": cfg.backup_horizon,
        "backup_mode": resolved["model"]["pipeline"]["stages"][0].get("backup_mode", "fixed")})
    model = ModelWrapper(config_tree=cfg, train_log_on_step=True, enable_grad_norm=False)
    loggers = [CSVLogger(str(writer.directory), name="csv")]
    if cfg.wandb.enabled:
        loggers.append(WandbLogger(project=cfg.wandb.project, entity=cfg.wandb.entity,
            id=cfg.run_id, name=cfg.run_id, group=cfg.wandb.group, resume="allow",
            config=resolved, save_dir=str(writer.directory), log_model=False))
    callback = RLReceipts(cfg, replay, writer)
    trainer = L.Trainer(accelerator=cfg.device, devices=1, max_steps=cfg.steps,
        max_epochs=-1, precision="32-true", logger=loggers, callbacks=[callback],
        enable_checkpointing=False, enable_progress_bar=False, num_sanity_val_steps=0,
        limit_val_batches=0, log_every_n_steps=cfg.log_interval)
    loader_rng = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(replay, batch_size=None, num_workers=0, generator=loader_rng)
    try:
        trainer.fit(model, train_dataloaders=loader, ckpt_path=resume)
        # A preemption after the final checkpoint but during evaluation must
        # resume evaluation, rather than silently declaring training complete.
        if cfg.eval_interval and trainer.global_step not in callback.evaluated_steps:
            callback.evaluate(trainer, model)
        writer.json("status.json", {"state": "COMPLETED", "step": trainer.global_step,
                                    "loaded_shards": replay.receipts,
                                    "evaluated_steps_this_attempt": sorted(callback.evaluated_steps)})
    except BaseException as error:
        writer.json("failure.json", {"type": type(error).__name__, "message": str(error),
                                     "step": trainer.global_step})
        raise
    finally:
        if cfg.wandb.enabled:
            import wandb
            wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(OmegaConf.load(args.config))
