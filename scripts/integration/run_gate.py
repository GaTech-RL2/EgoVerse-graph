"""Five-step integration gate driven by explicit recipe and source declarations."""

import argparse
import hashlib
import json
import math
import os
import subprocess
from copy import deepcopy
from pathlib import Path

import lightning.pytorch as pl
import torch
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from egomimic.pipeline.inference_session import InferenceSession
from egomimic.pl_utils.pl_data_utils import annotation_collate
from egomimic.trainHydra import train

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def parameter_probe(model):
    checksum = hashlib.sha256()
    for name, value in model.named_parameters():
        if value.requires_grad:
            checksum.update(name.encode())
            checksum.update(
                value.detach().flatten()[:64].float().cpu().numpy().tobytes()
            )
    return checksum.hexdigest()


class StepReceipt(pl.Callback):
    """Require actual finite backward/optimizer work and validation metrics."""

    def __init__(self):
        self.losses, self.gradients = [], []
        self.initial_probe = None

    def on_train_start(self, trainer, pl_module):
        self.initial_probe = parameter_probe(pl_module)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        norms = [
            torch.linalg.vector_norm(p.grad.detach(), dtype=torch.float32)
            for p in pl_module.parameters()
            if p.grad is not None
        ]
        if not norms:
            raise AssertionError("No gradient reached any model parameter")
        norm = torch.linalg.vector_norm(torch.stack(norms)).item()
        if not math.isfinite(norm):
            raise AssertionError("Nonfinite gradient norm in integration gate")
        self.gradients.append(
            {"step": trainer.global_step, "norm": norm, "parameter_tensors": len(norms)}
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        value = float(loss.detach().cpu())
        if not math.isfinite(value):
            raise AssertionError("Nonfinite training loss in integration gate")
        self.losses.append({"step": trainer.global_step, "loss": value})


def configured_case(matrix_path, name, inputs, output):
    matrix = OmegaConf.load(matrix_path)
    spec = matrix.cases[name]
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "egomimic/hydra_configs")
    ):
        cfg = compose(
            config_name=spec.config,
            overrides=list(spec.get("overrides", [])),
            return_hydra_config=True,
        )
    for key, value in spec.get("runtime_overrides", {}).items():
        OmegaConf.update(cfg, key, value, force_add=True)
    with open_dict(cfg):
        cfg.hydra.runtime.output_dir = str(output)
        cfg.hydra.job.id = name
        cfg.hydra.job.num = 0
        cfg.paths.output_dir = str(output)
        cfg.norm_stats.save_cache_dir = str(output)
        cfg.norm_stats.num_workers = 0
        cfg.norm_stats.sample_frac = 1.0
        cfg.logger = None
        cfg.callbacks = {
            "receipt": {"_target_": "scripts.integration.run_gate.StepReceipt"}
        }
        cfg.val_at_start = False
        cfg.model.enable_grad_norm = False
        for key, value in matrix.trainer.items():
            cfg.trainer[key] = value
        for key, value in spec.get("trainer", {}).items():
            cfg.trainer[key] = value
        cfg.evaluator.viz_every_n_epochs = 0
        cfg.evaluator.rkl_samples = 1
        cfg.inference_config.output_path = str(
            output / "checkpoints/inference-config.yaml"
        )
    HydraConfig.instance().set_config(cfg)
    data_manifest = json.loads((inputs / "data-manifest.json").read_text())
    datasets = {}
    for domain, source in spec.sources.items():
        original = cfg.data.train_datasets[domain]
        for split in ("train", "valid"):
            ids = [
                row["episode_hash"]
                for row in data_manifest["episodes"]
                if row["source"] == source and row["split"] == split
            ]
            if not ids:
                raise ValueError(f"No pinned {split} episodes for {source}")
            ds = deepcopy(original)
            with open_dict(ds):
                ds.resolver._target_ = (
                    "egomimic.rldb.zarr.zarr_dataset_multi.LocalEpisodeResolver"
                )
                ds.resolver.folder_path = str(inputs / "data" / source)
                ds.filters = {
                    "_target_": "egomimic.rldb.filters.DatasetFilter",
                    "episode_hashes": ids,
                }
                ds.mode = "total"
            datasets.setdefault(split, {})[domain] = ds
    with open_dict(cfg.data):
        cfg.data.train_datasets = datasets["train"]
        cfg.data.valid_datasets = datasets["valid"]
        for key in ("train_dataloader_params", "valid_dataloader_params"):
            cfg.data[key] = {
                domain: {"batch_size": int(spec.get("batch_size", 1)), "num_workers": 0}
                for domain in spec.sources
            }
    # Save a fully resolved runtime config, not Hydra's launcher internals.
    saved = OmegaConf.masked_copy(cfg, [key for key in cfg if key != "hydra"])
    saved = OmegaConf.create(
        OmegaConf.to_container(saved, resolve=True, throw_on_missing=True)
    )
    return cfg, saved, spec


def training_gate(matrix, name, inputs, output):
    cfg, saved, spec = configured_case(matrix, name, inputs, output)
    metrics, objects = train(cfg)
    trainer, module, dm = objects["trainer"], objects["model"], objects["datamodule"]
    from scripts.integration.run_gate import StepReceipt as ReceiptType

    receipt = next(
        callback for callback in trainer.callbacks if isinstance(callback, ReceiptType)
    )
    if (
        trainer.global_step != 5
        or len(receipt.gradients) != 5
        or len(receipt.losses) != 5
    ):
        raise AssertionError(
            "Integration gate did not execute exactly five optimizer updates"
        )
    validation = {
        key: float(value.detach().cpu())
        for key, value in metrics.items()
        if key.startswith("Valid/") and torch.is_tensor(value)
    }
    if not validation or not all(math.isfinite(v) for v in validation.values()):
        raise AssertionError("Gate requires real finite validation metrics")
    final_probe = parameter_probe(module)
    if final_probe == receipt.initial_probe:
        raise AssertionError("No sampled trainable parameter changed")
    checkpoint = output / "checkpoints/model.ckpt"
    trainer.save_checkpoint(checkpoint)
    if not trainer.is_global_zero:
        return
    identity = int(
        spec.get(
            "inference_identity",
            saved.model.inference.compatibility.normalizer_schema.get("embodiment"),
        )
    )
    for _, source, dataset in dm.iter_valid_datasets():
        sample = dataset[0]
        if int(sample["embodiment"]) == identity:
            observations = annotation_collate([sample])
            selected = {
                key: observations[key] for key in saved.model.inference.input["keys"]
            }
            torch.save(selected, output / "observations.pt")
            break
    else:
        raise AssertionError(
            "No held-out observation matches the selected inference profile"
        )
    OmegaConf.save(saved, output / "resolved.yaml")
    contexts = module.data_context
    write(
        output / "training-receipt.json",
        {
            "status": "passed",
            "case": name,
            "steps": trainer.global_step,
            "scope": "five-step real-data integration smoke; not a performance score",
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "devices": trainer.num_devices,
            "world_size": trainer.world_size,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "torch_version": torch.__version__,
            "inference_identity": identity,
            "sources": OmegaConf.to_container(spec.sources),
            "losses": receipt.losses,
            "gradients": receipt.gradients,
            "validation": validation,
            "initial_parameter_probe": receipt.initial_probe,
            "final_parameter_probe": final_probe,
            "resolved_config_sha256": digest(output / "resolved.yaml"),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest(checkpoint),
            "data_context_sha256": contexts.fingerprint(),
            "normalizer_sha256": contexts.state["sha256"],
            "data_manifest_sha256": digest(inputs / "data-manifest.json"),
            "input_receipt_sha256": digest(inputs / "input-receipt.json"),
            "normalizer_identities": sorted(map(str, contexts.sample_schema)),
            "observations_sha256": digest(output / "observations.pt"),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
    )
    print(
        "TRAINING_RECEIPT", (output / "training-receipt.json").read_text(), flush=True
    )


def verification_gate(output, device="cuda"):
    receipt = json.loads((output / "training-receipt.json").read_text())
    checkpoint = output / "checkpoints/model.ckpt"
    if digest(checkpoint) != receipt["checkpoint_sha256"]:
        raise ValueError("Training checkpoint changed before strict restore")
    config = OmegaConf.load(output / "resolved.yaml")
    identity = receipt["inference_identity"]
    session = InferenceSession.load(
        config,
        checkpoint_path=checkpoint,
        context_path=output / "checkpoints/data-context.json",
        identity=identity,
        device=device,
        normalize_inputs=False,
    )
    observations = torch.load(
        output / "observations.pt", map_location="cpu", weights_only=False
    )
    controls = session.inference_controls()
    results = []
    for steps, prefix in ((1, 1), (2, 2)):
        overrides = {"replan_every": prefix}
        if "inference_steps" in controls:
            overrides["inference_steps"] = steps
        session.apply_inference_overrides(overrides)
        torch.manual_seed(42)
        actions = session.predict(observations)
        executed = session.execution_plan(actions)
        assert torch.isfinite(actions).all() and executed.shape[1] == prefix
        results.append(
            {
                "controls": overrides,
                "output_shape": list(actions.shape),
                "executed_shape": list(executed.shape),
            }
        )
    write(
        output / "inference-receipt.json",
        {
            "status": "passed",
            "checkpoint_sha256": receipt["checkpoint_sha256"],
            "data_context_sha256": session.context.fingerprint(),
            "cases": results,
            "scope": "strict checkpoint restore through shared graph without robot/dataset construction",
        },
    )
    print(
        "INFERENCE_RECEIPT", (output / "inference-receipt.json").read_text(), flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["train", "verify"], required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Real integration gates require an allocated OSMO GPU")
    os.environ["SMOKE_INPUT_ROOT"] = str(args.inputs)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "train":
        training_gate(args.matrix, args.case, args.inputs, args.output)
    else:
        verification_gate(args.output)
