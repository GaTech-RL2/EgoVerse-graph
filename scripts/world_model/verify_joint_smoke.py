"""Allocated-compute postflight; a fit receipt alone never authorizes full training."""

# ruff: noqa: E402

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import torch
from hydra.utils import instantiate

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from egomimic.pipeline.checkpoint_binding import validate_checkpoint_binding
from egomimic.pipeline.construction import checkpoint_construction
from egomimic.pl_utils.joint_smoke import (
    load_smoke_configs,
    _check_pilot_adam_usage,
    require,
    weights_digest,
    write_once,
)
from egomimic.pl_utils.pl_model import ModelWrapper
from scripts.train.joint.lr_policy import from_config, update_lr, verify_scheduler
from scripts.train.joint import boundary_uncertainty


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--world-size", type=int, choices=(2, 4))
    parser.add_argument("--pilot-config", type=Path, help="One-GPU four-update TRAIN-pilot code validation; distinct from production smoke")
    parser.add_argument("--pilot-postflight-only", action="store_true", help="Resume only verification of existing immutable step4 checkpoint; zero optimizer updates")
    parser.add_argument("--multimodal", action="store_true", help="Native robot/human camera/language DDP smoke contract")
    parser.add_argument("--capacity-config",type=Path,help="Distributed native C64/3camera/rankbatch maximum-shape backward; zero optimizer updates")
    parser.add_argument("--capacity-output",type=Path,help="Immutable sibling capacity receipt; smoke output remains absent")
    parser.add_argument("--capacity-source-manifest-sha256",help="Exact staged source manifest SHA256")
    parser.add_argument("--capacity-data-manifest-sha256",help="Exact request-bound prepared data manifest SHA256")
    args = parser.parse_args()
    if args.capacity_config is not None:
        require(not args.multimodal and args.pilot_config is None and args.world_size is None,
                'Capacity mode is separate from optimizer smoke/pilot')
        from egomimic.pl_utils.joint_smoke import validate_multimodal_capacity
        require(args.capacity_output is not None and args.capacity_source_manifest_sha256 and args.capacity_data_manifest_sha256,
                "Capacity requires exact independent output/source binding")
        validate_multimodal_capacity(args.capacity_config,args.capacity_output,
            source_manifest_sha256=args.capacity_source_manifest_sha256,
            data_manifest_sha256=args.capacity_data_manifest_sha256)
        return
    require(args.capacity_output is None and args.capacity_source_manifest_sha256 is None
            and args.capacity_data_manifest_sha256 is None,
            'Capacity-only bindings require --capacity-config')
    if args.pilot_config is not None:
        require(not args.multimodal, "Pilot cannot certify distributed multimodal smoke")
        require(args.world_size is None, "Pilot mode owns its explicit single-GPU topology")
        from egomimic.pl_utils.joint_smoke import validate_multimodal_pilot
        if args.pilot_postflight_only:
            validate_multimodal_pilot(args.pilot_config, args.output, postflight_only=True)
        else:
            validate_multimodal_pilot(args.pilot_config, args.output)
        return
    require(not args.pilot_postflight_only, "Postflight-only requires --pilot-config")
    require(args.world_size in (2,4), "Historical smoke requires --world-size")
    root, world = args.output, args.world_size
    job = os.environ["SLURM_JOB_ID"]
    ranks = [
        json.loads((root / f"fit-rank{rank}.json").read_text()) for rank in range(world)
    ]
    cfg, model_config = load_smoke_configs(root)
    policy = from_config(cfg.model)
    peak, warmup = policy["peak_lr"], policy["warmup_optimizer_steps"]
    for rank, record in enumerate(ranks):
        require(
            record["schema"] == ("joint-multimodal-fit-v1" if args.multimodal else "joint-mot-fit-v1")
            and record["status"] == "FIT_PASS"
            and record["job_id"] == job
            and record["world_size"] == world
            and record["rank"] == rank,
            "Fit proof belongs to a different allocation/topology",
        )
        require(
            record["global_step"]
            == record["bf16_training_outputs"]
            == record["fp32_losses"]
            == 50
            and record["validation_batches"] == 100
            and record["local_validation_batches"] == 100 // world
            and record["fp32_gradients_verified"],
            "Incomplete real optimization/validation proof",
        )
        require(
            record["lr_warmup_steps"] == warmup
            and record["lr_policy"] == policy
            and len(record["optimizer_update_lrs"]) == 50
            and all(
                math.isclose(lr, update_lr(policy, step), rel_tol=1e-12)
                for step, lr in enumerate(record["optimizer_update_lrs"])
            ),
            "Actual optimizer updates did not use the configured LR warmup",
        )
    require(
        len({r["weights_sha256"] for r in ranks}) == 1
        and len({r["gpu_uuid"] for r in ranks}) == world,
        "DDP ranks need identical weights and distinct GPUs",
    )
    require(
        all(r["metrics"] == ranks[0]["metrics"] for r in ranks),
        "Validation reduction differs across ranks",
    )
    checkpoint_name = ranks[0]["checkpoint_name"]
    require(
        re.fullmatch(r"smoke-epoch[0-9]{4,}-step000000050\.ckpt", checkpoint_name)
        and all(r["checkpoint_name"] == checkpoint_name for r in ranks),
        "Diagnostic checkpoint must retain its true epoch and global step",
    )
    checkpoint_path = root / checkpoint_name
    with checkpoint_path.open("rb") as stream:
        checkpoint_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    # This exact checkpoint was generated by our own bound smoke in this allocation.
    checkpoint = torch.load(
        checkpoint_path, weights_only=False, map_location="cpu", mmap=True
    )
    cfg, model_config = load_smoke_configs(root)
    require(
        checkpoint["global_step"] == 50
        and (cfg.data.image_only or cfg.data.couple_state_observations)
        and cfg.trainer.devices == world
        and cfg.data.batch_size * world == 32,
        "Checkpoint/config scientific contract mismatch",
    )
    require(
        weights_digest(checkpoint["state_dict"]) == ranks[0]["weights_sha256"],
        "Checkpoint weights differ from actual optimized ranks",
    )
    require(
        all(
            torch.isfinite(t).all() and t.dtype == torch.float32
            for t in checkpoint["state_dict"].values()
        ),
        "Stored model weights must be finite FP32",
    )
    optimizers = checkpoint.get("optimizer_states", [])
    require(
        len(optimizers) == 1 and optimizers[0]["state"], "Missing complete AdamW state"
    )
    require(
        all(
            math.isclose(g["lr"], update_lr(policy, 50), rel_tol=1e-12)
            and (not warmup or g["initial_lr"] == peak)
            for g in optimizers[0]["param_groups"]
        ),
        "Wrong next-update LR after50 completed updates",
    )
    schedulers = checkpoint.get("lr_schedulers", [])
    verify_scheduler(policy, schedulers, 50)
    for state in optimizers[0]["state"].values():
        if not args.multimodal:
            require(float(state["step"]) == 50, "AdamW update count mismatch")
        require(
            all(
                state[k].dtype == torch.float32 and torch.isfinite(state[k]).all()
                for k in ("exp_avg", "exp_avg_sq")
            ),
            "Invalid AdamW moments",
        )
    rng = checkpoint.get("mot_rng_state", {})
    require(
        rng.get("schema") == "mot_rng_state.distributed.v1"
        and rng.get("world_size") == world
        and len(rng.get("ranks", [])) == world,
        "Missing all-rank RNG state",
    )
    require(
        all(
            set(r) == {"torch_cpu", "torch_cuda", "numpy", "python", "cuda_scope"}
            and r["cuda_scope"] == "current_device"
            and len(r["torch_cuda"]) == 1
            for r in rng["ranks"]
        ),
        "Incomplete rank RNG state",
    )
    sampler = checkpoint.get("PixelProprioDataModule", {})
    require(
        sampler.get("schema") == "mot-sampling-state-v1"
        and sampler.get("consumed_batches") == 50,
        "Missing exact consumed sampler state",
    )
    require(
        sampler.get("distributed_sampling", {}).get("world_size") == world
        and sampler["distributed_sampling"]["global_batch_size"] == 32,
        "Wrong checkpoint sampling topology",
    )
    data = instantiate(cfg.data)
    context = data.prepare_context(
        mode="eval",
        normalization="prepared",
        normalizer=None,
        restored_state=checkpoint["data_context"],
    )
    require(
        context.fingerprint() == ranks[0]["data_context_sha256"],
        "Restored data context changed",
    )
    boundary_uncertainty.context(cfg,checkpoint['data_context'],sampler)
    boundary_proof=None
    if args.multimodal:
        boundary_proof=boundary_uncertainty.fit_rows(ranks,
            selected=boundary_uncertainty.sampling_config(cfg),world=world,
            per_rank_batch=cfg.data.batch_size,embodiments=cfg.joint.pose_dims,
            context_sha256=context.fingerprint())
    else:
        require(cfg.data.get('boundary_sampling') is None,'Boundary uncertainty requires actual multimodal smoke')
    validate_checkpoint_binding(checkpoint, model_config, context)
    with checkpoint_construction():
        wrapper = (
            ModelWrapper(config_tree=model_config, enable_grad_norm=False).cuda().eval()
        )
        context.bind(wrapper.model)
    wrapper.data_context = context
    wrapper.load_state_dict(checkpoint["state_dict"], strict=True)
    restored_optimizer = instantiate(cfg.model.optimizer, params=wrapper.parameters())
    if callable(restored_optimizer):
        restored_optimizer = restored_optimizer()
    restored_scheduler = instantiate(cfg.model.scheduler, optimizer=restored_optimizer)
    restored_optimizer.load_state_dict(optimizers[0])
    if restored_scheduler is not None:
        restored_scheduler.load_state_dict(schedulers[0])
    require(
        (restored_scheduler.state_dict() == schedulers[0] if warmup else restored_scheduler is None)
        and all(
            math.isclose(g["lr"], update_lr(policy, 50), rel_tol=1e-12)
            for g in restored_optimizer.param_groups
        ),
        "Strict optimizer/scheduler restoration changed warmup state",
    )
    if args.multimodal:
        usage=ranks[0]['parameter_update_counts']
        require(all(r['parameter_update_counts']==usage for r in ranks),
                'Conditional optimizer usage differs across ranks')
        require(set(usage)==set(dict(wrapper.named_parameters())),'Optimizer parameter identity differs')
        _check_pilot_adam_usage(restored_optimizer,wrapper.named_parameters(),usage)
        require(all(r.get('private_bank_optimizer_steps_verified') and r.get('human_action_absent')
                    and all(r.get(key) for key in ('language_conditioning_verified','camera_targets_verified'))
                    and r.get('attention_backend')=='flash4' and r.get('token_execution')=='packed_valid'
                    and 'H100' in r.get('gpu_name','') and r.get('gpu_capability',[0])[0]>=9 for r in ranks),
                'Missing actual H100/multimodal optimizer proof')
    require(all(p.requires_grad for p in wrapper.parameters()), "An expert was frozen")
    network = wrapper.model.pipeline.stage_by_id("joint_denoiser").model
    require(
        all(
            r.get("active_streams") == list(network.streams)
            and len(r.get("paired_rgb_batch_sha256", [])) == 3
            for r in ranks
        ),
        "Missing active-stream or paired RGB controls",
    )
    require(
        len({r["initial_image_weights_sha256"] for r in ranks}) == 1,
        "Initial image experts differ across ranks",
    )
    expert_parameters = network.expert_parameter_counts()
    require(
        sum(expert_parameters.values()) == sum(p.numel() for p in wrapper.parameters()),
        "Per-expert counts do not account for every trained parameter",
    )
    if args.multimodal:
        samples={emb:torch.load(root/f'sample-input-{emb}.pt',map_location='cpu',weights_only=False)
                 for emb in network.pose_dims}
        if boundary_proof is not None:boundary_uncertainty.strict_validation_samples(samples)
        require(all(r['active_streams_by_embodiment']=={e:list(network.active_streams(e)) for e in network.pose_dims}
                    for r in ranks),'Actual embodiment streams differ')
    else:
        samples={'libero_gripper':torch.load(root/'sample-input.pt',map_location='cpu',weights_only=True)}
    wrapper.model.device = next(wrapper.parameters()).device
    sample_seconds=0.0
    for embodiment,sample in samples.items():
        groups=wrapper.model.process_batch_for_training({embodiment:sample})
        torch.manual_seed(42)
        torch.cuda.synchronize()
        start=time.perf_counter()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            output=wrapper.model.forward_eval(groups)[embodiment]['image_proprio_prediction']
        torch.cuda.synchronize()
        sample_seconds+=time.perf_counter()-start
        if args.multimodal:
            require(set(output)==set(network.active_streams(embodiment)), 'Unexpected native sampled streams')
            require('language_features' in sample and 'pixel_metadata' in sample,
                    'Missing actual subtask/language sample provenance')
        for name,value in output.items():
            source=groups[embodiment]['world'][name]
            require(value.shape==source['value'].shape and value.dtype==torch.float32
                    and torch.isfinite(value).all(),'Strictly reloaded sample is invalid')
            require(torch.equal(value[source['observed']],source['value'][source['observed']]),
                    'Sampling changed observed state/commands')
    sampler = wrapper.model.pipeline.stage_by_id("joint_sampler").sampler
    sample_state_steps = sampler.state_steps
    sample_action_steps = getattr(sampler, "action_steps", 0) if "action" in network.streams else 0
    require(sample_state_steps == 20 and ("action" not in network.streams or sample_action_steps == 20),
            "Actual configured sampler steps differ from the smoke contract")
    execution = None
    if hasattr(network, "image_model"):
        require(network.image_model.compile_mode == "default" and network.image_model.compile_dynamic is True
                and network.image_model.compile_backend == "inductor",
                "Native image compile policy differs from historical training")
        execution = {"implementation": "native_image_joint", "compile_mode": "default",
                     "compile_dynamic": True, "compile_backend": "inductor"}
    receipt = dict(
        schema="joint-mot-smoke-v1",
        model_execution=execution,
        status="PASS",
        job_id=job,
        world_size=world,
        active_streams=list(network.streams),
        expert_widths=dict(zip(network.streams, network.expert_widths)),
        expert_parameters=expert_parameters,
        global_step=50,
        validation_batches=100,
        strict_full_wrapper_reload=True,
        optimizer_sampler_all_rank_rng_verified=True,
        lr_scheduler_strictly_restored=True,
        lr_warmup_steps=warmup,
        lr_policy=policy,
        checkpoint_sha256=checkpoint_sha,
        all_rank_weights_agree=True,
        sample_finite=True,
        sample_state_steps=sample_state_steps,
        sample_action_steps=sample_action_steps,
        sample_seconds=sample_seconds,
        ranks=ranks,
    )
    if args.multimodal:
        require(network.image_model.token_execution=='packed_valid'
                and network.image_model.blocks[0].attention_backend=='flash4'
                and cfg.evaluator.active_streams_from_data,
                'Production multimodal execution/evaluator contract differs')
        require(len(network.pose_dims)==2 and len(network.action_dims)==1
                and network.separate_proprio_experts and network.camera_dims,
                'Missing native private human/robot expert contract')
        receipt.update(schema='joint-multimodal-smoke-v1',
            private_bank_optimizer_steps_verified=True,human_action_absent=True,
            subtask_boundaries_verified=bool(cfg.data.subtask_manifest_sha256),
            language_conditioning_verified=network.language_feature_dim==1024,
            camera_targets_verified=bool(cfg.data.camera_pose),
            active_streams_by_embodiment={e:list(network.active_streams(e)) for e in network.pose_dims},
            attention_backend='flash4',token_execution='packed_valid',
            data_context_sha256=context.fingerprint())
    if boundary_proof is not None:
        receipt.update(subtask_boundaries_verified=False,
            original_subtask_annotation_identity_verified=True,
            subtask_boundary_verification_policy=boundary_uncertainty.POLICY['policy'],
            boundary_uncertainty=boundary_proof,
            boundary_policy_sha256=boundary_uncertainty.digest(boundary_uncertainty.POLICY),
            boundary_objective_sha256=boundary_uncertainty.digest(boundary_uncertainty.OBJECTIVE))
    write_once(root / "smoke-completion.json", receipt)
    print(
        json.dumps({"status": "PASS", "receipt": str(root / "smoke-completion.json")})
    )


if __name__ == "__main__":
    main()
