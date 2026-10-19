"""Real-data joint-MoT smoke evidence; never attached to production training."""

import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_once(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def weights_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def image_parameter_state(network):
    """Select the common image expert, whose bank index is always zero."""
    return {
        key: value
        for key, value in network.state_dict().items()
        if key.startswith(
            (
                "inputs.image.",
                "outputs.image.",
                "camera.",
                "noise_embeddings.0.",
                "norms.0.",
                "final_adaln.0.",
            )
        )
        or (
            key.startswith("blocks.")
            and any(
                f".{bank}.0." in key
                for bank in ("norm1", "qkv", "out", "norm2", "ffn", "adaln")
            )
        )
    }


def paired_image_digest(world, values, levels):
    """Bound RGB data, geometry, masks and actual corruption; consumes no RNG."""
    state = dict(world["image"])
    state.update(noised_value=values["image"], noise_level=levels["image"])
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{key}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_smoke_configs(output):
    """Join Hydra's full run snapshot with the checkpoint's model-only tree.

    trainHydra deliberately excludes data/trainer settings from model hparams.
    Validate both snapshots together rather than treating model hparams as the
    full experiment or weakening the strict data/context restoration checks.
    """
    output = Path(output)
    cfg = OmegaConf.load(output / ".hydra" / "config.yaml")
    model_config = OmegaConf.load(output / "smoke-config.yaml")
    require(
        all(key in cfg for key in ("model", "data", "trainer", "paths"))
        and "model" in model_config,
        "Missing full Hydra run snapshot or model construction snapshot",
    )
    require(
        OmegaConf.to_container(cfg.model, resolve=True)
        == OmegaConf.to_container(model_config.model, resolve=True),
        "Hydra model config differs from the optimized checkpoint construction",
    )
    require(
        OmegaConf.to_container(
            OmegaConf.create(cfg.get("run_provenance", {})), resolve=True
        )
        == OmegaConf.to_container(
            OmegaConf.create(model_config.get("run_provenance", {})), resolve=True
        ),
        "Hydra run provenance differs from the model snapshot",
    )
    require(
        Path(cfg.paths.output_dir).resolve() == output.resolve(),
        "Hydra snapshot belongs to another smoke output",
    )
    return cfg, model_config


class JointSmokeProof(Callback):
    def __init__(self, output, world_size, expected_steps=50, warmup_steps=10):
        require(world_size in (2, 4), "Joint pilots use two/four dedicated GPUs")
        require(0 < warmup_steps < expected_steps, "Need warmed optimization timing")
        self.output, self.world_size = Path(output), world_size
        self.expected_steps, self.warmup_steps = expected_steps, warmup_steps

    def on_fit_start(self, trainer, pl_module):
        require(
            trainer.world_size == self.world_size and trainer.precision == "bf16-mixed",
            "Smoke topology/precision differs from production",
        )
        require(
            trainer.global_step == 0 and not trainer.ckpt_path,
            "Joint pilots must initialize from scratch",
        )
        require(
            all(not optimizer.state for optimizer in trainer.optimizers),
            "Fresh training cannot restore optimizer moments",
        )
        self.output.mkdir(parents=True, exist_ok=True)
        self.fit_start = time.perf_counter()
        self.fp32_losses, self.bf16_outputs = 0, 0
        self.gradients_checked = False
        self.network = pl_module.model.pipeline.stage_by_id("joint_denoiser").model
        require(
            all(
                p.requires_grad and p.dtype == torch.float32
                for p in self.network.parameters()
            ),
            "All active experts must be trainable FP32 stored weights",
        )
        require(
            all(
                math.isclose(group["lr"], 5e-6 / 3000, rel_tol=1e-12)
                for optimizer in trainer.optimizers
                for group in optimizer.param_groups
            ),
            "Expected first update LR5e-6/3000",
        )
        require(
            pl_module.hparams.config_tree.model.scheduler.warmup_steps == 3000
            and len(trainer.lr_scheduler_configs) == 1
            and trainer.lr_scheduler_configs[0].interval == "step"
            and trainer.lr_scheduler_configs[0].frequency == 1,
            "Both pilots require a3000 optimizer-update warmup",
        )
        self.optimizer_lrs = []
        digest = weights_digest(pl_module.state_dict())
        initial = [None] * self.world_size
        torch.distributed.all_gather_object(initial, digest)
        require(len(set(initial)) == 1, "Fresh DDP initial weights differ")
        self.initial_weights = initial
        self.initial_image_weights = weights_digest(image_parameter_state(self.network))
        self.paired_rgb_batches = []

        def observe(module, args, outputs):
            require(
                set(outputs) == set(self.network.streams)
                and all(value.dtype == torch.bfloat16 for value in outputs.values()),
                "Actual active-stream predictions must execute in BF16",
            )
            if pl_module.training and torch.is_grad_enabled():
                self.bf16_outputs += 1
                if len(self.paired_rgb_batches) < 3:
                    self.paired_rgb_batches.append(
                        paired_image_digest(args[0], args[2], args[3])
                    )

        self.precision_hook = self.network.register_forward_hook(observe)

    def on_before_backward(self, trainer, pl_module, loss):
        require(
            loss.dtype == torch.float32 and bool(torch.isfinite(loss)),
            "Joint MSE must be finite FP32",
        )
        self.fp32_losses += 1

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        expected_lr = 5e-6 * min((trainer.global_step + 1) / 3000, 1.0)
        require(
            all(
                math.isclose(g["lr"], expected_lr, rel_tol=1e-12)
                for g in optimizer.param_groups
            ),
            "Actual optimizer LR does not follow the3000-update warmup",
        )
        self.optimizer_lrs.append(expected_lr)
        if self.gradients_checked:
            return
        parameters = list(self.network.parameters())
        require(
            all(
                p.grad is not None and p.grad.dtype == torch.float32 for p in parameters
            ),
            "Every expert requires an FP32 gradient",
        )
        require(
            bool(torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all()),
            "Nonfinite expert gradient",
        )
        self.gradients_checked = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step in (self.warmup_steps, self.expected_steps):
            torch.cuda.synchronize(pl_module.device)
            torch.distributed.barrier()
            if trainer.global_step == self.warmup_steps:
                torch.cuda.reset_peak_memory_stats(pl_module.device)
                self.wall_start = time.perf_counter()
            else:
                self.wall_seconds = time.perf_counter() - self.wall_start
                self.peak_allocated = torch.cuda.max_memory_allocated(pl_module.device)
                self.peak_reserved = torch.cuda.max_memory_reserved(pl_module.device)

    def on_validation_start(self, trainer, pl_module):
        self.validation_start = time.perf_counter()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if trainer.global_rank != 0 or batch_idx != 0:
            return
        sample = batch["libero_gripper"]
        world = {
            name: {key: value.detach().cpu() for key, value in stream.items()}
            for name, stream in sample["world"].items()
        }
        for stream in world.values():
            stream["value"] = torch.where(
                stream["observed"][..., None], stream["value"], float("nan")
            )
        torch.save(
            {"world": world, "embodiment": "libero_gripper"},
            self.output / "sample-input.pt",
        )

    def on_train_end(self, trainer, pl_module):
        self.precision_hook.remove()
        evaluator = pl_module.evaluator
        require(
            trainer.global_step == self.expected_steps
            and self.fp32_losses == self.bf16_outputs == self.expected_steps
            and self.gradients_checked,
            "Missing real optimizer/precision evidence",
        )
        require(
            evaluator.has_completed_validation(self.expected_steps)
            and evaluator.global_validation_batches == 100
            and evaluator._validation_batches == 100 // self.world_size,
            "All100 scheduled validation anchors must complete after optimization",
        )
        metrics = {
            str(k): float(v)
            for k, v in trainer.callback_metrics.items()
            if str(k).startswith("val/")
        }
        expected = {
            f"val/libero_gripper/{kind}/{metric}"
            for kind in ("forecast", "completion")
            for metric in (
                {
                    "image": "image_normalized_mse",
                    "proprio": "proprio_normalized_mse",
                    "action": "action_command_mse",
                }[name]
                for name in self.network.streams
            )
        }
        require(
            set(metrics) == expected
            and all(math.isfinite(v) for v in metrics.values()),
            "Missing/nonfinite held-out image/state/command metrics",
        )
        require(
            trainer.datamodule.consumed_batches == self.expected_steps,
            "Sampling cursor differs from completed optimizer updates",
        )
        final = weights_digest(pl_module.state_dict())
        weights = [None] * self.world_size
        torch.distributed.all_gather_object(weights, final)
        require(
            len(set(weights)) == 1 and weights != self.initial_weights,
            "Weights did not optimize identically across all ranks",
        )
        # Diagnostic terminal save only. All ranks participate to preserve RNG.
        checkpoint_name = (
            f"smoke-epoch{trainer.current_epoch:04d}-step{trainer.global_step:09d}.ckpt"
        )
        trainer.save_checkpoint(self.output / checkpoint_name)
        if trainer.global_rank == 0:
            with (self.output / "smoke-config.yaml").open("x") as stream:
                stream.write(
                    OmegaConf.to_yaml(pl_module.hparams.config_tree, resolve=True)
                )
        record = dict(
            schema="joint-mot-fit-v1",
            status="FIT_PASS",
            job_id=os.environ["SLURM_JOB_ID"],
            rank=trainer.global_rank,
            world_size=self.world_size,
            global_step=self.expected_steps,
            validation_batches=100,
            local_validation_batches=100 // self.world_size,
            weights_sha256=final,
            all_rank_weights=weights,
            initial_weights=self.initial_weights,
            active_streams=list(self.network.streams),
            initial_image_weights_sha256=self.initial_image_weights,
            paired_rgb_batch_sha256=self.paired_rgb_batches,
            bf16_training_outputs=self.bf16_outputs,
            fp32_losses=self.fp32_losses,
            fp32_gradients_verified=self.gradients_checked,
            lr_warmup_steps=3000,
            optimizer_update_lrs=self.optimizer_lrs,
            metrics=metrics,
            gpu_uuid=str(torch.cuda.get_device_properties(pl_module.device).uuid),
            measured_updates=self.expected_steps - self.warmup_steps,
            warm_wall_seconds=self.wall_seconds,
            peak_allocated_bytes=self.peak_allocated,
            peak_reserved_bytes=self.peak_reserved,
            validation_seconds=time.perf_counter() - self.validation_start,
            data_context_sha256=pl_module.data_context.fingerprint(),
            checkpoint_name=checkpoint_name,
        )
        write_once(self.output / f"fit-rank{trainer.global_rank}.json", record)
