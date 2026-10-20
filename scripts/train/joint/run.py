"""Request-bound joint training service, promoted from the accepted Lambda drivers.

One implementation handles fresh smoke and full requests. Scheduler bindings and
immutable evidence remain outside the source; this driver never submits jobs.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.train.joint.contract import (
    bound,
    compare_configs,
    need,
    sha,
    verify_evidence,
    write_once,
)


def credentials():
    if os.environ.get("WANDB_API_KEY"):
        return
    import netrc

    try:
        values = netrc.netrc(os.environ["PIXEL_WANDB_NETRC"]).authenticators(
            "api.wandb.ai"
        )
    except Exception:
        raise ValueError("Could not read the explicit W&B credential binding") from None
    need(values is not None and bool(values[2]), "Missing W&B credentials")
    os.environ["WANDB_API_KEY"] = values[2]


def main():
    path = Path(sys.argv[1])
    request = json.loads(path.read_text())
    root, source = path.parent.parent, Path(request["source"])
    need(
        request["schema"] == "joint-training-v2"
        and request["phase"] in ("smoke", "full"),
        "Wrong training request",
    )
    smoke = request["phase"] == "smoke"
    need(
        request["target_steps"] == (50 if smoke else 100000),
        "Wrong authorized training budget",
    )
    from scripts.train.joint.allocation import admit

    allocation = admit(request)
    need(
        sha(request["source_manifest"]) == request["source_manifest_sha256"],
        "Changed source manifest",
    )
    for line in Path(request["source_manifest"]).read_text().splitlines():
        expected, name = line.split("  ", 1)
        need(sha(source / name) == expected, "Changed source: " + name)
    need(
        sha(request["data_manifest"]) == request["data_manifest_sha256"],
        "Changed data manifest",
    )
    preflight = None
    if not smoke:
        _, preflight = verify_evidence(request)
        controls = bound(request["factorial_controls"])
        need(
            controls["status"] == "PASS"
            and controls["source_revision"] == request["source_revision"]
            and controls["experiments"] == request["factorial_experiments"],
            "Missing paired four-arm GPU controls",
        )
        need(
            any(
                row["experiment"] == request["experiment"]
                and row["smoke_request_sha256"] == request["smoke"]["request"]["sha256"]
                and row["smoke_completion_sha256"]
                == request["smoke"]["completion"]["sha256"]
                for row in controls["evidence"]
            ),
            "Paired evidence belongs to another smoke",
        )
        credentials()
        from scripts.train.joint.check_wandb import inspect_run

        need(
            inspect_run(request["run_id"])["exists"] is False,
            "Fresh W&B identity exists; inspect rather than reset",
        )
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    with initialize_config_dir(
        config_dir=str(source / "egomimic/hydra_configs"), version_base=None
    ):
        cfg = compose(
            config_name="train_zarr_cartesian", overrides=request["overrides"]
        )
    resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    world = request["world_size"]
    need(
        cfg.seed == 42
        and cfg.trainer.devices == cfg.data.distributed_world_size == world,
        "Seed/topology mismatch",
    )
    need(
        cfg.data.batch_size == request["per_rank_batch"]
        and cfg.data.batch_size * world == 32
        and cfg.model.optimizer.lr == 5e-6
        and cfg.model.optimizer.weight_decay == 0.01,
        "Batch/optimizer mismatch",
    )
    need(
        cfg.model.scheduler._target_
        == "egomimic.utils.schedulers.linear_warmup_constant_scheduler"
        and cfg.model.scheduler.warmup_steps == 3000
        and cfg.model.scheduler_interval == "step"
        and cfg.model.scheduler_frequency == 1,
        "Scheduler contract changed",
    )
    need(
        request["lr_policy"]
        == dict(
            peak_lr=5e-6,
            warmup_optimizer_steps=3000,
            schedule="linear_warmup_then_constant",
        ),
        "Unbound LR policy",
    )
    need(
        cfg.ckpt_path is None
        and not cfg.model.pipeline.get("initialization")
        and not cfg.model.pipeline.get("trainability"),
        "Fresh step0/trainable experts required",
    )
    need(
        cfg.data.couple_state_observations
        and not cfg.data.image_only
        and cfg.data.task == "visual_state_action",
        "Paired full data/noising contract changed",
    )
    need(
        cfg.trainer.precision == "bf16-mixed"
        and not cfg.model.enable_grad_norm
        and cfg.runtime.disable_tf32,
        "Precision/clipping policy changed",
    )
    if smoke:
        need(
            cfg.trainer.max_steps == cfg.trainer.val_check_interval == 50
            and cfg.trainer.num_sanity_val_steps == 0,
            "Scheduled optimization and validation required",
        )
        need(
            "wandb" not in cfg.logger
            and cfg.evaluator.pixel_diagnostics.log_wandb is False,
            "Smoke cannot write W&B",
        )
        need("joint_smoke" in cfg.callbacks, "Missing all-rank smoke instrumentation")
    else:
        smoke_cfg = OmegaConf.to_container(
            OmegaConf.create(bound(request["smoke"]["resolved"], False)), resolve=True
        )
        compare_configs(resolved, smoke_cfg, request)
    with torch.device("meta"):
        network = instantiate(
            cfg.model.pipeline.stages[cfg.model.pipeline.stage_ids.joint_denoiser].model
        )
    counts = network.expert_parameter_counts()
    need(
        list(network.streams) == request["active_streams"]
        and counts == request["expert_parameters"]
        and sum(p.numel() for p in network.parameters()) == request["parameters"]
        and all(p.requires_grad for p in network.parameters())
        and list(network.expert_widths) == request["expert_widths"],
        "Active expert count/width/trainability mismatch",
    )
    need(
        counts["image"] == 88178688
        and all(v <= 20000000 for k, v in counts.items() if k != "image"),
        "Expert budget mismatch",
    )
    del network
    data = instantiate(cfg.data)
    context = data.prepare_context(
        mode="train", normalization="prepared", normalizer=None
    )
    if preflight is not None:
        need(
            context.fingerprint() == preflight["provenance"]["data_context_sha256"],
            "Context changed after smoke",
        )
    need(
        {k: len(v.episodes) for k, v in data.datasets.items()}
        == {"train": 450, "val": 50}
        and len(data.val_dataloader()) == 100,
        "Wrong split or validation anchors",
    )
    need(
        shutil.disk_usage(root).free > request["reserved_checkpoint_bytes"],
        "Insufficient full retention space",
    )
    output = Path(request["output"])
    need(
        str(cfg.paths.output_dir) == str(output) and not output.exists(),
        "Output collision/mismatch",
    )
    provenance = dict(
        source_revision=request["source_revision"],
        source_manifest_sha256=request["source_manifest_sha256"],
        data_manifest_sha256=request["data_manifest_sha256"],
        data_context_sha256=context.fingerprint(),
    )
    if not smoke:
        provenance["run_id"] = request["run_id"]
    if allocation is None:
        from gpu_health_set import admit_gpu_set

        admit_gpu_set(
            root
            / "status"
            / (path.stem + "-" + os.environ["SLURM_JOB_ID"] + "-gpu.json"),
            world,
        )
    write_once(
        path.with_suffix(".preflight.json"),
        dict(
            status="PREFLIGHT_PASS" if smoke else "FULL_ADMISSION_PASS",
            job_id=os.environ["SLURM_JOB_ID"],
            request_sha256=sha(path),
            provenance=provenance,
            parameters=request["parameters"],
            expert_parameters=counts,
            expert_widths=request["expert_widths"],
            active_streams=request["active_streams"],
            world_size=world,
            global_batch=32,
            allocation_attestation=allocation,
        ),
    )
    with path.with_suffix(".resolved.yaml").open("x") as stream:
        stream.write(OmegaConf.to_yaml(cfg, resolve=True))
    if smoke:
        from scripts.world_model.check_joint_capacity import check_capacity

        check_capacity(cfg, data, path.with_suffix(".capacity.json"))
    output.mkdir(parents=True)
    (output / ".training-admitted").write_text(sha(path) + "\n")
    extra = (
        "+run_provenance={"
        + ",".join(k + ":" + json.dumps(v) for k, v in provenance.items())
        + "}"
    )
    bootstrap = root / "runtime_bootstrap.py"
    print(
        json.dumps(
            dict(
                status="SMOKE_ADMISSION_PASS" if smoke else "FULL_ADMISSION_PASS",
                experiment=request["experiment"],
                job_id=os.environ["SLURM_JOB_ID"],
            )
        ),
        flush=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(bootstrap),
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=" + str(world),
            "--max-restarts=0",
            str(bootstrap),
            str(source / "egomimic/trainHydra.py"),
            *request["overrides"],
            extra,
        ],
        check=True,
        cwd=source,
    )
    verifier = source / (
        "scripts/world_model/verify_joint_smoke.py"
        if smoke
        else "scripts/train/joint/verify_full.py"
    )
    arguments = [str(output), "--world-size", str(world)] if smoke else [str(path)]
    subprocess.run(
        [sys.executable, str(bootstrap), str(verifier), *arguments],
        check=True,
        cwd=source,
    )


if __name__ == "__main__":
    main()
