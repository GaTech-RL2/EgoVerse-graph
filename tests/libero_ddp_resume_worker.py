"""Subprocess fixture for a real one-process to two-process Lightning resume."""

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf, open_dict

from egomimic.trainHydra import train
from tests.test_libero_benchmark import make_replay
from tests.test_oat_training import config_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.root.mkdir(parents=True, exist_ok=True)
    replay = args.root / "replay.zarr"
    if not args.resume:
        make_replay(replay)
    cfg = config_for("libero_oattok", replay, args.root / "train")
    spec = cfg.model.pipeline.stages[0].tokenizer
    spec.emb_dim, spec.head_dim = 32, 8
    spec.encoder_depth, spec.decoder_depth, spec.num_registers = 1, 1, 4
    cfg.trainer.limit_train_batches = 1.0
    cfg.trainer.accumulate_grad_batches = 2 if args.resume else 4
    cfg.trainer.devices = 2 if args.resume else 1
    cfg.trainer.max_epochs = 2 if args.resume else 1
    cfg.callbacks.batch_budget = OmegaConf.create(
        {
            "_target_": "egomimic.pl_utils.oat_training.OATBatchBudgetCallback",
            "global_batch_size": 8,
        }
    )
    with open_dict(cfg.trainer):
        cfg.trainer.enable_progress_bar = False
        if args.resume:
            cfg.trainer.strategy = "ddp_find_unused_parameters_true"
    if args.resume:
        cfg.ckpt_path = str(args.root / "train/checkpoints/last.ckpt")
    _, objects = train(cfg)
    if objects["trainer"].is_global_zero:
        payload = torch.load(
            args.root / "train/checkpoints/last.ckpt",
            map_location="cpu",
            weights_only=False,
        )
        report = {
            "global_step": payload["global_step"],
            "ema_num_updates": payload["ema_num_updates"],
            "training_budget": payload["training_budget"],
            "optimizer_states": bool(payload["optimizer_states"]),
        }
        (args.root / ("resumed.json" if args.resume else "initial.json")).write_text(
            json.dumps(report)
        )


if __name__ == "__main__":
    main()
