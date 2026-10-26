"""Prove real native BF16/DDP continuation from an already verified joint smoke.

This diagnostic uses the checkpoint-bound model and the existing trainHydra
loader. It does not submit jobs, change model/data settings or authorize full
training by itself. Its short cadence is diagnostic, never a production recipe.
"""

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch
from omegaconf import OmegaConf, open_dict

from egomimic.pl_utils.joint_smoke import JointSmokeProof, weights_digest
from scripts.train.joint.lr_policy import from_config, verify_scheduler
from scripts.train.joint.contract import bound, canonical_uuid, need, sha, write_once, verify_allocation_relocation


def equal_state(actual, saved):
    if isinstance(saved, torch.Tensor):
        return (isinstance(actual, torch.Tensor) and actual.dtype == saved.dtype
                and actual.shape == saved.shape and torch.equal(actual.detach().cpu(), saved.cpu()))
    if isinstance(saved, dict):
        return (isinstance(actual, dict) and actual.keys() == saved.keys()
                and all(equal_state(actual[k], v) for k, v in saved.items()))
    if isinstance(saved, (list, tuple)):
        return (type(actual) is type(saved) and len(actual) == len(saved)
                and all(equal_state(a, b) for a, b in zip(actual, saved)))
    # NumPy's RNG state includes arrays, without requiring a pickle workaround.
    import numpy as np
    if isinstance(saved, np.ndarray):
        return isinstance(actual, np.ndarray) and np.array_equal(actual, saved)
    return type(actual) is type(saved) and actual == saved


def quota_free_bytes(text, mount):
    """Read the aggregate Lustre user row; filesystem free alone ignores quota."""
    rows = [line.split() for line in text.splitlines() if line.split()[:1] == [mount]]
    need(len(rows) == 1 and len(rows[0]) >= 4, "Missing unambiguous live Lustre quota")
    used, soft, hard = [int(v.rstrip("*")) for v in rows[0][1:4]]
    limits = [v for v in (soft, hard) if v > 0]
    need(limits, "Lustre quota limit unavailable")
    return max(0, min(limits) - used) * 1024


class ResumeOptimizerProof(JointSmokeProof):
    """Reuse smoke precision/LR/gradient hooks, checking restored native state."""
    def __init__(self, *, output, world_size, start_step, stop_step, checkpoint, multimodal=False):
        super().__init__(output, world_size, expected_steps=stop_step, warmup_steps=start_step,
                         multimodal=multimodal)
        self.start_step, self.checkpoint = start_step, checkpoint
        self.rng_cleanup = None

    def on_fit_start(self, trainer, pl_module):
        # Lightning restores optimizer/loop state after on_fit_start.
        pass

    def on_train_start(self, trainer, pl_module):
        need(trainer.world_size == self.world_size and trainer.precision == "bf16-mixed",
             "Recovery must use the original BF16/DDP topology")
        need(trainer.global_step == self.start_step, "Wrong restored step")
        saved = torch.load(self.checkpoint, map_location="cpu", weights_only=False, mmap=True)
        need(saved["global_step"] == self.start_step, "Wrong input checkpoint")
        from scripts.train.joint.native_recovery import checkpoint_science
        checkpoint_science(saved["hyper_parameters"]["config_tree"], pl_module.hparams.config_tree)
        need(equal_state(pl_module.state_dict(), saved["state_dict"]), "Weights not strictly restored")
        need(equal_state(trainer.optimizers[0].state_dict(), saved["optimizer_states"][0]),
             "AdamW moments/groups not strictly restored")
        self.lr_policy = from_config(pl_module.hparams.config_tree.model)
        restored_schedulers = [c.scheduler.state_dict() for c in trainer.lr_scheduler_configs]
        need(equal_state(restored_schedulers, saved.get("lr_schedulers", [])),
             "LR scheduler not strictly restored")
        verify_scheduler(self.lr_policy, restored_schedulers, self.start_step)
        need(equal_state(pl_module.training_behavior._pending_rng, saved["mot_rng_state"]),
             "Native per-rank RNG restoration was not armed")
        need(trainer.datamodule.consumed_batches == self.start_step, "Sampler cursor not restored")
        need(equal_state(trainer.datamodule.state_dict(), saved["PixelProprioDataModule"]),
             "Complete sampler continuation state differs")
        need(equal_state(pl_module.data_context.snapshot(), saved["data_context"]),
             "Checkpoint data context was not strictly restored")
        self.initial_weights = weights_digest(pl_module.state_dict())
        self.network = pl_module.model.pipeline.stage_by_id("joint_denoiser").model
        need(all(p.requires_grad and p.dtype == torch.float32 for p in self.network.parameters()),
             "Stored weight dtype/trainability changed")
        self.output.mkdir(parents=True, exist_ok=True)
        self.fp32_losses = self.bf16_outputs = 0
        self.optimizer_lrs, self.validation_steps = [], []
        self.gradients_checked = False
        if self.multimodal:
            from egomimic.pl_utils.joint_smoke import _check_pilot_interfaces
            from scripts.train.joint.native_recovery import restored_conditional_counts, install_rng_observer

            _check_pilot_interfaces(self.network, set(self.network.pose_dims), set(self.network.action_dims))
            need(set(self.network.pose_dims) == {"eva_bimanual", "human_bimanual"}
                 and set(self.network.action_dims) == {"eva_bimanual"}
                 and "camera" in self.network.streams, "Wrong native Elmo interfaces")
            need(torch.distributed.get_backend() == "nccl", "Native recovery requires NCCL")
            prop = torch.cuda.get_device_properties(pl_module.device)
            need("H100" in prop.name and prop.major >= 9, "Native recovery requires actual H100")
            image = self.network.image_model
            need(image.token_execution == "packed_valid" and image.blocks[0].attention_backend == "flash4",
                 "Native recovery changed Flash4/packed execution")
            need(trainer.limit_val_batches == 100 // self.world_size
                 and len(trainer.datamodule.val_dataloader()) >= 100 // self.world_size,
                 "Native recovery needs100 global validation anchors")
            self.parameter_usage, self.usage_callback = restored_conditional_counts(trainer, pl_module, saved)
            self.initial_parameter_usage = dict(self.parameter_usage)
            self.embodiment_batches = {emb: 0 for emb in self.network.pose_dims}
            self.current_embodiment = None
            self.rng_observation, self.rng_cleanup = install_rng_observer(
                pl_module, saved["mot_rng_state"], equal_state)

        def observe(module, args, outputs):
            need(set(outputs) == set(self.network.active_streams(args[1]) if self.multimodal else self.network.streams)
                 and all(v.dtype == torch.bfloat16 for v in outputs.values()),
                 "Native active-stream forward did not use BF16")
            if pl_module.training and torch.is_grad_enabled():
                self.bf16_outputs += 1
                if self.multimodal:
                    self.current_embodiment = args[1]
        self.precision_hook = self.network.register_forward_hook(observe)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        # Check every resumed update, retaining the maintained smoke checks.
        self.gradients_checked = False
        super().on_before_optimizer_step(trainer, pl_module, optimizer)
        need(pl_module.training_behavior._pending_rng is None, "Native RNG was not restored on training")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        pass

    def on_validation_start(self, trainer, pl_module):
        self.validation_steps.append(trainer.global_step)
        super().on_validation_start(trainer, pl_module)

    def on_validation_batch_end(self, *args, **kwargs):
        if self.multimodal:
            super().on_validation_batch_end(*args, **kwargs)

    def teardown(self, trainer, pl_module, stage):
        if self.rng_cleanup is not None:
            self.rng_cleanup()

    def on_train_end(self, trainer, pl_module):
        self.precision_hook.remove()
        count = self.expected_steps - self.start_step
        need(trainer.global_step == self.expected_steps
             and self.fp32_losses == self.bf16_outputs == len(self.optimizer_lrs) == count,
             "Missing actual resumed optimizer/precision updates")
        need(self.validation_steps == list(range(self.start_step + 2, self.expected_steps + 1, 2)),
             "Continuation repeated or skipped validation events")
        need(trainer.datamodule.consumed_batches == self.expected_steps
             and pl_module.evaluator.has_completed_validation(self.expected_steps)
             and pl_module.evaluator.global_validation_batches == 100
             and pl_module.evaluator._validation_batches == 100 // self.world_size,
             "Missing resumed sampling cursor or all100 validation anchors")
        metrics = {str(k): float(v) for k, v in trainer.callback_metrics.items() if str(k).startswith("val/")}
        import math
        need(metrics and all(math.isfinite(v) for v in metrics.values()), "Nonfinite resumed validation")
        final = weights_digest(pl_module.state_dict())
        gathered = [None] * self.world_size
        torch.distributed.all_gather_object(gathered, final)
        need(len(set(gathered)) == 1 and final != self.initial_weights, "Resumed weights did not update identically")
        extra = {}
        if self.multimodal:
            from scripts.train.joint.native_recovery import conditional_record, expected_metrics, sample_both_embodiments

            need(set(metrics) == expected_metrics(self.network), "Missing native embodiment/stream validation metrics")
            record = conditional_record(trainer.optimizers[0], pl_module.named_parameters(),
                                        self.parameter_usage, self.expected_steps)
            need(self.usage_callback.counts == self.parameter_usage,
                 "Maintained optimizer counters differ from observed resumed updates")
            need(self.rng_observation["observed"], "Actual rank RNG restore was never observed")
            sample_proof = sample_both_embodiments(pl_module, self.network, self.output)
            extra = dict(multimodal=True, conditional_optimizer_record=record,
                         initial_parameter_usage=self.initial_parameter_usage,
                         embodiment_optimizer_batches=self.embodiment_batches,
                         active_streams_by_embodiment={e: list(self.network.active_streams(e)) for e in self.network.pose_dims},
                         actual_rank_rng_restore=self.rng_observation,
                         hidden_label_samples=sample_proof,
                         private_bank_optimizer_steps_verified=True, human_action_absent=True)
        else:
            need(all(float(s["step"]) == self.expected_steps for s in trainer.optimizers[0].state.values()),
                 "Optimizer restarted instead of continuing")
        verify_scheduler(self.lr_policy, [c.scheduler.state_dict() for c in trainer.lr_scheduler_configs], self.expected_steps)
        from scripts.clusters.pace.gpu_admission import cuda_uuid

        write_once(self.output / f"resume-{self.start_step}-{self.expected_steps}-rank{trainer.global_rank}.json", dict(
            status="RESUME_PASS", job_id=os.environ["SLURM_JOB_ID"], rank=trainer.global_rank,
            world_size=self.world_size, start_step=self.start_step, global_step=self.expected_steps,
            optimizer_updates=count, fp32_losses=self.fp32_losses, bf16_outputs=self.bf16_outputs,
            optimizer_update_lrs=self.optimizer_lrs, validation_steps=self.validation_steps,
            strict_start_weights_optimizer_scheduler_rng_cursor=True,
            fp32_gradients_verified=self.gradients_checked, weights_sha256=final,
            initial_weights_sha256=self.initial_weights, validation_metrics=metrics,
            gpu_uuid=cuda_uuid(torch.cuda.get_device_properties(pl_module.device)),
            **extra,
        ))


def phase_config(original, *, output, checkpoint, start_step, stop_step, service_output, world):
    need((start_step, stop_step) in ((50, 52), (52, 56)) and world in (2, 4),
         "This diagnostic proves exactly50->52->56; it is not production crash admission")
    cfg = OmegaConf.create(copy.deepcopy(OmegaConf.to_container(original, resolve=True)))
    original_model, original_data = copy.deepcopy(cfg.model), copy.deepcopy(cfg.data)
    multimodal = bool(cfg.callbacks.joint_smoke.get("multimodal", False))
    with open_dict(cfg):
        cfg.ckpt_path = checkpoint
        cfg.paths.output_dir = output
        cfg.val_at_start = cfg.val_at_end = False
        cfg.trainer.max_steps = 56
        cfg.trainer.val_check_interval = 2
        cfg.callbacks.pop("joint_smoke")
        cb = cfg.callbacks.model_checkpoint
        cb.every_n_train_steps, cb.every_n_epochs = 2, 0
        cb.save_top_k, cb.save_last = -1, "link"
        cb.dirpath = output + "/checkpoints"
        cb.filename = "allocation-epoch{epoch:04d}-step{step:09d}"
        cfg.callbacks.resume_proof = dict(
            _target_="scripts.train.joint.recovery_check.ResumeOptimizerProof",
            output=service_output, world_size=world, start_step=start_step,
            stop_step=stop_step, checkpoint=checkpoint,
            multimodal=multimodal,
        )
        if stop_step < 56:
            cfg.callbacks.allocation_boundary = dict(
                _target_="scripts.train.joint.allocation_boundary.ScheduledAllocationBoundary",
                start_step=start_step, stop_step=stop_step, target_step=56,
                cadence=2, output=service_output,
            )
        cfg.logger.csv.save_dir = output + "/logs"
        if cfg.evaluator.get("pixel_diagnostics") is not None:
            cfg.evaluator.pixel_diagnostics.artifact_dir = service_output + f"/validation-{stop_step}"
    if multimodal:
        need(any(c.get("_target_") == "egomimic.pl_utils.joint_smoke.ConditionalOptimizerUsage"
                 for c in cfg.callbacks.values()), "Missing maintained conditional optimizer callback")
    need(OmegaConf.to_container(cfg.model, resolve=True) == OmegaConf.to_container(original_model, resolve=True)
         and OmegaConf.to_container(cfg.data, resolve=True) == OmegaConf.to_container(original_data, resolve=True),
         "Recovery changed model/data semantics")
    return cfg


def finish_proof(request_path, q, original, admission, training_job_id, repaired=False):
    output = Path(q["output"])
    evidence = output/"proof"
    boundary = json.loads((evidence/"allocation-step000000052.json").read_text())
    checkpoint = boundary["checkpoint"]["path"]
    need(boundary["status"] == "PASS" and boundary["checkpoint_saved_after_validation"]
         and sha(checkpoint) == boundary["checkpoint"]["sha256"], "Changed boundary recovery")
    phases = []
    for start, stop in ((50,52),(52,56)):
        ranks = [json.loads((evidence/f"resume-{start}-{stop}-rank{r}.json").read_text())
                 for r in range(original["world_size"])]
        need(all(r["status"] == "RESUME_PASS" and r["job_id"] == training_job_id
                 and r["rank"] == i and r["world_size"] == original["world_size"]
                 and r["start_step"] == start and r["global_step"] == stop
                 and r["optimizer_updates"] == r["fp32_losses"] == r["bf16_outputs"] == stop-start
                 and r["fp32_gradients_verified"] and r["strict_start_weights_optimizer_scheduler_rng_cursor"]
                 and r["validation_steps"] == list(range(start+2,stop+1,2))
                 for i,r in enumerate(ranks)), "Missing all-rank actual optimizer recovery proof")
        phases.append(ranks)
    need(all(a["weights_sha256"] == b["initial_weights_sha256"] for a,b in zip(*phases)),
         "Second process did not restore first-process weights")
    from scripts.train.joint.native_subtasks import selected
    extra = {}
    if selected(original):
        from scripts.train.joint.native_recovery import summarize_native_phases
        extra = summarize_native_phases(phases)
    if admission is not None:
        attestation = bound(admission)
        need(all([canonical_uuid(r["gpu_uuid"]) for r in ranks]
                 == [canonical_uuid(r["uuid"]) for r in attestation["gpu_probe"]["ranks"]]
                 for ranks in phases), "Resumed optimizer ranks differ from admitted physical GPUs")
    write_once(output/"recovery-completion.json", dict(
        schema="joint-recovery-check-v1", status="PASS", job_id=training_job_id,
        postflight_job_id=os.environ["SLURM_JOB_ID"], postflight_repaired=repaired,
        request_sha256=sha(request_path), smoke_completion_sha256=q["smoke_completion"]["sha256"],
        source_revision=original["source_revision"], data_manifest_sha256=original["data_manifest_sha256"],
        active_streams=original["active_streams"], world_size=original["world_size"], global_batch=32,
        resumed_from=50, boundary_step=52, resumed_to=56, actual_updates=6,
        model_data_unchanged=True, all_rank_full_state_restored=True,
        allocation_policy=q["allocation"], allocation_relocation=q.get("allocation_relocation"),
        no_extra_validation=True, allocation_attestation=admission,
        boundary_checkpoint_sha256=sha(checkpoint),
        native_gpu_status="FAILED_POSTFLIGHT_UUID_FORMAT" if repaired else "postflight_complete",
        **extra,
    ))


def main():
    request_path = Path(sys.argv[1])
    q = json.loads(request_path.read_text())
    if q.get("schema") == "native-elmo-production-recovery-request-v1":
        need(len(sys.argv) == 2, "Production recovery does not support legacy postflight repair")
        from scripts.train.joint.production_recovery import main as production_recovery_main
        return production_recovery_main(request_path)
    need(q["schema"] == "joint-recovery-check-v1", "Wrong recovery request")
    original = bound(q["smoke_request"])
    verify_allocation_relocation(q, original, q["smoke_request"]["sha256"])
    execution = copy.deepcopy(original)
    execution["allocation"] = q["allocation"]
    if q.get("operational_service"):
        execution["operational_service"] = q["operational_service"]
    completion = bound(q["smoke_completion"])
    bound(q["resolved"], False)
    cfg = OmegaConf.load(q["resolved"]["path"])
    need(completion["status"] == "PASS" and completion["global_step"] == 50
         and completion["world_size"] == original["world_size"], "Missing exact GPU smoke")
    from scripts.train.joint.native_subtasks import selected
    if selected(original):
        need(completion.get("schema") == "joint-multimodal-smoke-v1"
             and completion.get("private_bank_optimizer_steps_verified") is True
             and cfg.callbacks.joint_smoke.get("multimodal") is True,
             "Native recovery requires the matching actual multimodal smoke")
    need(sha(q["checkpoint"]["path"]) == completion["checkpoint_sha256"] == q["checkpoint"]["sha256"],
         "Changed smoke checkpoint")
    for line in Path(original["source_manifest"]).read_text().splitlines():
        digest, relative = line.split("  ", 1)
        need(sha(Path(original["source"]) / relative) == digest, "Changed checkpoint-bound source")
    need(sha(original["source_manifest"]) == original["source_manifest_sha256"]
         and sha(original["data_manifest"]) == original["data_manifest_sha256"], "Changed source/data manifest")
    if len(sys.argv) == 4 and sys.argv[2] == "--postflight":
        job = sys.argv[3]
        need(job.isdecimal(), "Exact native training job required")
        root = request_path.parent.parent
        raw = subprocess.check_output(["sacct","-X","-n","-P","-j",job,
            "-o","JobIDRaw,State,ExitCode,User,UID,NodeList,Account,QOS"],text=True)
        rows=[line.split("|") for line in raw.splitlines() if line.split("|")[0] == job]
        need(len(rows)==1 and rows[0][1:3]==["FAILED","1:0"], "Unexpected original job termination")
        native=rows[0]
        policy=execution["allocation"]
        need(native[3:5]==[policy["user"],str(policy["uid"])]
             and native[6:8]==[policy["account"],policy["qos"]], "Wrong native GPU job owner")
        error=(root/"logs"/f"recovery-{job}.err").read_text()
        need(error.count("ValueError: Resumed optimizer ranks differ from admitted physical GPUs")==1
             and error.count("ValueError:")==1, "Failure was not solely UUID-format postflight")
        attestation_path=root/"status"/f"native-admission-{job}"/"attestation.json"
        admission=dict(path=str(attestation_path),sha256=sha(attestation_path))
        attestation=bound(admission)
        need(attestation["job_id"]==job and attestation["node"]==native[5], "Wrong recorded allocation")
        from scripts.train.joint.allocation import verify_pace

        verify_pace(execution,attestation,dict(SLURM_JOB_ID=job,SLURMD_NODENAME=native[5],SLURM_RESTART_COUNT="0"))
        (Path(q["output"])/"proof"/f"native-{job}.txt").write_text(raw)
        finish_proof(request_path,q,original,admission,job,repaired=True)
        return
    output = Path(q["output"])
    required = q["reserved_checkpoint_bytes"]
    need(type(required) is int and required >= 4 * Path(q["checkpoint"]["path"]).stat().st_size + 500000000,
         "Four retained recovery checkpoints plus artifact headroom are required")
    quota = q["quota"]
    if execution.get("allocation") is None:
        from scripts.train.joint.lambda_runtime import filesystem_recovery_reserve

        live_quota = filesystem_recovery_reserve(q)
    else:
        need(output.is_relative_to(quota["mount"]) and quota["user"] == execution["allocation"]["user"],
             "Quota is not bound to the actual output owner")
        live_quota = subprocess.check_output(["lfs", "quota", "-u", quota["user"], quota["mount"]], text=True)
        need(quota_free_bytes(live_quota, quota["mount"]) > required, "Insufficient live user quota")
    need(not output.exists() and shutil.disk_usage(output.parent).free > q["reserved_checkpoint_bytes"],
         "Recovery output collision or insufficient retention space")
    output.mkdir()
    (output/"quota-before.txt").write_text(live_quota)
    root = request_path.parent.parent
    from scripts.train.joint.allocation import admit
    admission = admit(execution, status_root=root/"status", bootstrap=root/"runtime_bootstrap.py")
    evidence = output / "proof"
    evidence.mkdir()
    config_dir = output / "configs"
    config_dir.mkdir()
    checkpoint = q["checkpoint"]["path"]
    from scripts.train.joint.service_bootstrap import command as service_command
    for start, stop in ((50, 52), (52, 56)):
        phase = phase_config(cfg, output=str(output), checkpoint=checkpoint,
                             start_step=start, stop_step=stop,
                             service_output=str(evidence), world=original["world_size"])
        path = config_dir / f"resume-{start}-{stop}.yaml"
        path.write_text(OmegaConf.to_yaml(phase, resolve=True))
        command = [sys.executable, str(root/"runtime_bootstrap.py"), "-m", "torch.distributed.run",
                   "--standalone", "--nnodes=1", "--nproc-per-node="+str(original["world_size"]),
                   "--max-restarts=0",
                   *service_command(q, request_path, root/"runtime_bootstrap.py",
                                    ["-m", "egomimic.trainHydra"])[1:],
                   "--config-path", str(config_dir), "--config-name", path.stem,
                   "hydra.run.dir="+str(output/f"hydra-{stop}"), "hydra.job.chdir=false"]
        subprocess.run(command, check=True, cwd=original["source"])
        if stop == 52:
            boundary = json.loads((evidence/"allocation-step000000052.json").read_text())
            need(boundary["status"] == "PASS" and boundary["checkpoint_saved_after_validation"],
                 "Missing post-validation boundary recovery")
            checkpoint = boundary["checkpoint"]["path"]
            need(sha(checkpoint) == boundary["checkpoint"]["sha256"], "Changed boundary checkpoint")
    finish_proof(request_path,q,original,admission,os.environ["SLURM_JOB_ID"])


if __name__ == "__main__":
    main()
