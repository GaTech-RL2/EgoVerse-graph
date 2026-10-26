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

from scripts.train.joint.lr_policy import from_config, validate
from scripts.train.joint.contract import (
    bound,
    compare_configs,
    need,
    sha,
    verify_evidence,
    verify_model_execution,
    verify_capacity_profile,
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


def validate_resolved_request(request, cfg):
    """Check pure resolved science/mode guards before any allocation or construction.

    The runtime and metadata preflight must use this same gate. Configuration
    group selection alone may leave experiment-inherited logger dictionaries.
    """
    from omegaconf import OmegaConf

    need(request["schema"] == "joint-training-v2" and request["phase"] in ("smoke", "full"),
         "Wrong training request")
    smoke = request["phase"] == "smoke"
    need(request["target_steps"] in ((50,) if smoke else (100000, 300000)),
         "Wrong authorized training budget")
    resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    expected_data_task = verify_model_execution(request, resolved)
    from scripts.train.joint import native_subtasks
    if native_subtasks.selected(request):
        need(cfg.data.get("_convert_") == "all",
             "Native subtask data requires Hydra _convert_=all before construction")
    world = request["world_size"]
    need(
        cfg.seed == 42
        and cfg.trainer.devices == cfg.data.distributed_world_size == world,
        "Seed/topology mismatch",
    )
    need(
        cfg.data.batch_size == request["per_rank_batch"]
        and cfg.data.batch_size * world == 32
        and cfg.model.optimizer.lr == validate(request["lr_policy"])["peak_lr"]
        and cfg.model.optimizer.weight_decay == 0.01,
        "Batch/optimizer mismatch",
    )
    need(from_config(cfg.model) == validate(request["lr_policy"]), "Unbound LR policy")
    need(
        cfg.ckpt_path is None
        and not cfg.model.pipeline.get("initialization")
        and not cfg.model.pipeline.get("trainability"),
        "Fresh step0/trainable experts required",
    )
    need(
        (
            (expected_data_task == "visual_diffusion"
             and request["active_streams"] == ["image"]
             and cfg.data.image_only and not cfg.data.couple_state_observations)
            or (cfg.data.couple_state_observations and not cfg.data.image_only)
        )
        and cfg.data.task == expected_data_task,
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
            and (cfg.evaluator.pixel_diagnostics is None or cfg.evaluator.pixel_diagnostics.log_wandb is False),
            "Smoke cannot write W&B",
        )
        need("joint_smoke" in cfg.callbacks, "Missing all-rank smoke instrumentation")
    return resolved


def main():
    path = Path(sys.argv[1])
    request = json.loads(path.read_text())
    root, source = path.parent.parent, Path(request["source"])
    need(
        request["schema"] == "joint-training-v2"
        and request["phase"] in ("smoke", "full"),
        "Wrong training request",
    )
    from scripts.train.joint import native_subtasks
    native_subtask = native_subtasks.selected(request)
    smoke = request["phase"] == "smoke"
    need(
        request["target_steps"] in ((50,) if smoke else (100000, 300000)),
        "Wrong authorized training budget",
    )
    from scripts.train.joint.allocation import admit

    allocation = admit(request, status_root=root/"status", bootstrap=root/"runtime_bootstrap.py")
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
    if native_subtask:
        native_subtasks.verify_dataset(request)
    preflight = None
    start, stop = 0, request["target_steps"]
    if not smoke:
        completion, preflight = verify_evidence(request)
        from scripts.train.joint.retention import check_storage

        from scripts.train.joint.continuation import allocation_steps

        start, stop = allocation_steps(request)
        if request.get("interruption") is not None:
            from scripts.train.joint.interruption_admission import live_admission
            current_recovery = live_admission(request, path)
            write_once(path.with_suffix(".interruption-live-admission.json"), dict(
                status="CURRENT_INTENT_CLAIMS_QUEUE_STORAGE_PASS", job_id=os.environ["SLURM_JOB_ID"],
                request_sha256=sha(path), **current_recovery))
        check_storage(request, completion)
        from scripts.train.joint.verify_temporal_pair import admission_mode

        mode = "native_subtask" if native_subtask else admission_mode(request)
        if mode == "native_subtask":
            native_subtasks.verify_dataset(request)
        elif mode == "temporal_comparison":
            from scripts.train.joint.verify_temporal_pair import admit as admit_temporal_pair
            admit_temporal_pair(request)
        elif mode == "pixel_recipe_comparison":
            from scripts.train.joint.verify_temporal_pair import admit_track_recipe
            admit_track_recipe(request, spatial="image")
        elif mode == "track_recipe_comparison":
            from scripts.train.joint.verify_temporal_pair import admit_track_recipe
            admit_track_recipe(request)
        elif mode == "modality_comparison":
            from scripts.train.joint.contract import verify_modality_extension
            verify_modality_extension(request, completion)
        elif mode == "optimizer_comparison":
            from scripts.train.joint.contract import verify_optimizer_comparison
            verify_optimizer_comparison(request, completion)
        elif mode == "capacity_comparison":
            from scripts.train.joint.contract import verify_capacity_extension
            verify_capacity_extension(request, completion)
        elif mode in ("gap_comparison", "noise_comparison"):
            from scripts.train.joint.contract import verify_gap_extension
            verify_gap_extension(request)
        else:
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

        need(inspect_run(request["run_id"])["exists"] == bool(start),
             "W&B identity does not match fresh/continued allocation")
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
    resolved = validate_resolved_request(request, cfg)
    world = request["world_size"]
    if not smoke:
        smoke_cfg = OmegaConf.to_container(
            OmegaConf.create(bound(request["smoke"]["resolved"], False)), resolve=True
        )
        compare_configs(resolved, smoke_cfg, request)
        from scripts.train.joint.continuation import configure_allocation

        configure_allocation(cfg, request, start, stop)
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
        counts[request["active_streams"][0]] == verify_capacity_profile(request, resolved)
        and all(v <= request["vector_expert_parameter_budget"] if native_subtask else v <= 20000000
                for k, v in counts.items() if k != request["active_streams"][0]),
        "Expert budget mismatch",
    )
    del network
    data = instantiate(cfg.data)
    context = data.prepare_context(
        mode="train", normalization="prepared", normalizer=None
    )
    from scripts.train.joint import boundary_uncertainty
    boundary_uncertainty.context(resolved,context.snapshot(),data.state_dict())
    boundary_contract=request.get('boundary_sampling_contract')
    if boundary_contract is not None:
        census=bound(boundary_contract['eligibility']['report'])
        need(census.get('data_context_sha256')==context.fingerprint(),
             'Actual training context differs from allocated boundary census')
    if preflight is not None:
        need(
            context.fingerprint() == preflight["provenance"]["data_context_sha256"],
            "Context changed after smoke",
        )
    from scripts.train.joint.dataset_contract import verify_dataset
    expected_split, expected_validation = verify_dataset(request)
    need(
        {k: len(v.episodes) for k, v in data.datasets.items()} == expected_split
        and len(data.val_dataloader()) == expected_validation,
        "Wrong split or validation anchors",
    )
    if smoke:
        need(shutil.disk_usage(root).free > request["reserved_checkpoint_bytes"],
             "Insufficient smoke retention space")
    output = Path(request["output"])
    fresh_output = not start or request.get("interruption") is not None
    need(
        str(cfg.paths.output_dir) == str(output)
        and output.exists() == (not fresh_output)
        and (fresh_output or (output / ".training-admitted").is_file()),
        "Output collision/mismatch",
    )
    provenance = dict(
        source_revision=request["source_revision"],
        source_manifest_sha256=request["source_manifest_sha256"],
        data_manifest_sha256=request["data_manifest_sha256"],
        data_context_sha256=context.fingerprint(),
    )
    if boundary_contract is not None:
        provenance.update(boundary_policy_sha256=boundary_uncertainty.digest(boundary_uncertainty.POLICY),
            boundary_objective_sha256=boundary_uncertainty.digest(boundary_uncertainty.OBJECTIVE),
            boundary_eligibility_report_sha256=boundary_contract['eligibility']['report']['sha256'])
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
    from scripts.train.joint.service_bootstrap import command
    if smoke:
        if native_subtask:
            # The same source-owned verifier owns actual H100/Flash4 maxshape.
            # A40/pilot or historical two-camera evidence never substitutes.
            bootstrap = root / "runtime_bootstrap.py"
            subprocess.run([
                sys.executable, str(bootstrap), "-m", "torch.distributed.run",
                "--standalone", "--nnodes=1", "--nproc-per-node=" + str(world),
                "--max-restarts=0",
                *command(request, path, bootstrap,
                    [str(source / "scripts/world_model/verify_joint_smoke.py"),
                     str(output), "--capacity-config", str(path.with_suffix(".resolved.yaml")),
                     "--capacity-output", str(path.with_suffix(".capacity.json")),
                     "--capacity-source-manifest-sha256", request["source_manifest_sha256"],
                     "--capacity-data-manifest-sha256", request["data_manifest_sha256"]])[1:]],
                check=True, cwd=source)
        else:
            from scripts.world_model.check_joint_capacity import check_capacity
            check_capacity(cfg, data, path.with_suffix(".capacity.json"))
    if fresh_output:
        output.mkdir(parents=True)
        (output / ".training-admitted").write_text(sha(path) + "\n")
    extra = (
        "+run_provenance={"
        + ",".join(k + ":" + json.dumps(v) for k, v in provenance.items())
        + "}"
    )
    bootstrap = root / "runtime_bootstrap.py"
    from scripts.train.joint.service_bootstrap import command
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
    training_arguments = request["overrides"]
    if not smoke:
        # The fully resolved file includes only the strictly compared science
        # plus the guarded native checkpoint/callback additions above.
        training_arguments = [
            "--config-path", str(path.parent),
            "--config-name", path.with_suffix(".resolved.yaml").stem,
            "hydra.run.dir="+str(output/"attempts"/f"step{start:09d}"),
            "hydra.job.chdir=false",
        ]
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
            *command(request, path, bootstrap, ["-m", "egomimic.trainHydra"])[1:],
            *training_arguments,
            extra,
        ],
        check=True,
        cwd=source,
    )
    verifier = (source / "scripts/world_model/verify_joint_smoke.py"
                if smoke else Path(__file__).with_name("verify_full.py"))
    arguments = [str(output), "--world-size", str(world), *(["--multimodal"] if native_subtask else [])] if smoke else [str(path)]
    subprocess.run(
        command(request, path, bootstrap, [str(verifier), *arguments]),
        check=True,
        cwd=source,
    )


if __name__ == "__main__":
    main()
