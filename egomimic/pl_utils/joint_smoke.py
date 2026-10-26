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
from scripts.train.joint.lr_policy import from_config, update_lr, verify_scheduler
from egomimic.rldb.boundary_sampling import validate_boundary_record, validate_boundary_sampling


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


def state_control_parameter_state(state):
    """Project only the optional named action bank out of a native wrapper.

    Retaining full wrapper keys lets the actual three-stream initialization be
    compared with an immutable image/proprio smoke without constructing it again.
    """
    result = {key: value for key, value in state.items() if "action" not in key.split(".")}
    require(result and len(result) < len(state), "Missing optional action bank")
    return result


def image_parameter_state(network):
    """Select the model-owned image state or the existing joint bank zero.

    An unsupported namespace must fail, never certify an empty image digest.
    """
    provider = getattr(network, "image_parameter_state", None)
    if callable(provider):
        state = provider()
        require(bool(state) and all(torch.is_tensor(v) for v in state.values()),
                "Model-owned image state is empty or invalid")
        return state
    state = {
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
    require(bool(state), "Unsupported image parameter namespace; empty digest rejected")
    return state


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


def check_conditional_adam_state(optimizer_state, usage_record, expected_step):
    """Verify actual bank-wise Adam counts using the preserved parameter order."""
    require(usage_record.get('schema')=='native-conditional-optimizer-usage-v1'
            and usage_record.get('global_step')==expected_step,
            'Missing exact conditional optimizer usage checkpoint')
    names=usage_record['parameter_names'];counts=usage_record['parameter_update_counts']
    ids=[key for group in optimizer_state['param_groups'] for key in group['params']]
    require(len(ids)==len(names)==len(set(names)) and set(counts)==set(names),
            'Conditional optimizer parameter order/identity differs')
    for name,key in zip(names,ids):
        count=counts[name];state=optimizer_state['state'].get(key)
        require(type(count) is int and 0<=count<=expected_step,
                'Invalid conditional optimizer count')
        require((state is None)==(count==0),'Unexpected Adam state for unused parameter')
        if state is not None:
            require(float(state['step'])==count,'Conditional Adam update count differs')
            require(all(state[k].dtype==torch.float32 and torch.isfinite(state[k]).all()
                        for k in ('exp_avg','exp_avg_sq')),'Invalid conditional Adam moments')


class ConditionalOptimizerUsage(Callback):
    """Opt-in production checkpoint accounting for genuinely absent branches."""
    def __init__(self):
        self.counts=None
        self.restored=None

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        record=checkpoint.get('native_conditional_optimizer_usage')
        require(record is not None,'Conditional training resume requires exact optimizer usage')
        require(len(checkpoint.get('optimizer_states',[]))==1,'Expected one Adam optimizer')
        check_conditional_adam_state(checkpoint['optimizer_states'][0],record,checkpoint['global_step'])
        self.restored=record

    def on_fit_start(self, trainer, pl_module):
        names=[name for name,_ in pl_module.named_parameters()]
        if self.restored is None:
            require(trainer.global_step==0,'Missing restored conditional optimizer usage')
            self.counts={name:0 for name in names}
        else:
            require(names==self.restored['parameter_names'],'Conditional parameter order changed on resume')
            self.counts=dict(self.restored['parameter_update_counts'])

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        require(self.counts is not None and len(trainer.optimizers)==1,'Conditional usage callback not initialized')
        for name,param in pl_module.named_parameters():
            self.counts[name]+=int(param.grad is not None)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        names=[name for name,_ in pl_module.named_parameters()]
        record=dict(schema='native-conditional-optimizer-usage-v1',global_step=int(trainer.global_step),
                    parameter_names=names,parameter_update_counts=dict(self.counts))
        require(len(checkpoint.get('optimizer_states',[]))==1,'Expected full Adam state on save')
        check_conditional_adam_state(checkpoint['optimizer_states'][0],record,trainer.global_step)
        checkpoint['native_conditional_optimizer_usage']=record


def _check_multimodal_sample(sample, *, boundary_policy=None):
    boundary_policy=validate_boundary_sampling(boundary_policy)
    features=sample.get('language_features')
    require(torch.is_tensor(features) and features.ndim==2 and features.shape[1]==1024
            and torch.isfinite(features).all(),'Missing finite actual Qwen language condition')
    metadata=sample.get('pixel_metadata',[])
    require(len(metadata)==features.shape[0],'One metadata record required per sample')
    for i,record in enumerate(metadata):
        unit=record.get('subtask',{})
        require(unit.get('interval_semantics')=='half_open' and unit.get('annotation_provenance'),
                'Missing reviewed subtask provenance')
        frames=record.get('frame_indices',[])
        chunks=record.get('proprio_frame_indices',[])
        indices=frames+[v for chunk in chunks for v in chunk]
        for key in ('action_frame_indices','camera_frame_indices'):
            if key in record:
                for item in record[key]:indices.extend(item if isinstance(item,list) else [item])
        if boundary_policy is None:
            require('boundary_sampling' not in record,'Legacy/held-out smoke cannot admit boundary uncertainty metadata')
        else:
            require(record.get('boundary_sampling',{}).get('policy')==boundary_policy,
                    'Boundary sample differs from explicit expected training policy')
        if 'boundary_sampling' in record:
            packet=validate_boundary_record(record,unit,indices)
            require(type(sample.get('sampling_cursor')) is int
                    and packet['rng_identity']['cursor']==sample['sampling_cursor']-1,
                    'Boundary RNG cursor differs from checkpointed sample draw')
            require(record['chunk_size']==1 and chunks==[[v] for v in frames],
                    'Boundary image/state selection must share native timestamps')
            for name in ('action','camera'):
                if name in sample['world']:
                    require(record.get(name+'_frame_indices')==[[v] for v in frames],
                            'Boundary action/camera selection must share native timestamps')
            physical=torch.tensor(record['timestamps_seconds'],dtype=torch.float64)
            relative=(physical-float(record['time_origin_seconds'])).float()
            repeats={'image':len(record['camera_names'])*(record['image_size'][0]//record['patch_size'])*(record['image_size'][1]//record['patch_size']),
                     'proprio':1,'action':1,'camera':len(record['camera_names'])}
            for name,stream in sample['world'].items():
                expected=relative.repeat_interleave(repeats[name])
                actual=stream['time'][i,:len(expected)].detach().cpu()
                require(actual.shape==expected.shape and torch.equal(actual,expected),
                        'Boundary stream timestamp differs from shared recorded selection')
        else:
            require(indices and all(unit['start_idx']<=v<unit['end_idx'] for v in indices),
                    'Native sample crosses reviewed subtask boundary')
        require(sample['task_id'][i]==unit['task_id'] and sample['language_instruction'][i]==unit['instruction'],
                'Actual language condition differs from reviewed subtask')
        require(record.get('camera_reference_policy')=='first_selected_native_frame_per_camera'
                and record.get('camera_reference_frame_index')==frames[0],
                'Camera reference must be actual first sampled native frame')
    require('camera' in sample['world'],'Actual camera targets are missing')
    stream=sample['world']['camera']
    target=stream['valid']&stream.get('target_valid',stream['valid'])
    require(target.any() and torch.isfinite(stream['value'][target]).all(),
            'Actual measured camera targets must be finite and nonempty')


def _boundary_sample_counts(sample):
    """Count actual checked training realizations; never a corpus census."""
    counts=dict(samples=0,requested_boundary=0,realized_boundary=0,fallback=0,legacy_uniform=0,
                outward_samples=0,start_inward=0,start_outward=0,end_inward=0,end_outward=0)
    for record in sample['pixel_metadata']:
        packet=record.get('boundary_sampling')
        require(packet is not None,'Opt-in smoke requires boundary provenance on every training record')
        counts['samples']+=1
        for key,value in packet['mixture_counters'].items():counts[key]+=value
        counts['outward_samples']+=int(bool(packet['selected_outside_original']))
        if packet['pair_indices'] is not None:
            start,end=packet['pair_offsets_frames']
            counts['start_inward']+=int(start>0);counts['start_outward']+=int(start<0)
            counts['end_inward']+=int(end<0);counts['end_outward']+=int(end>0)
    return counts


class JointSmokeProof(Callback):
    def __init__(self, output, world_size, expected_steps=50, warmup_steps=10, multimodal=False):
        require(world_size in (2, 4), "Joint pilots use two/four dedicated GPUs")
        require(0 < warmup_steps < expected_steps, "Need warmed optimization timing")
        self.output, self.world_size = Path(output), world_size
        self.expected_steps, self.warmup_steps = expected_steps, warmup_steps
        self.multimodal = bool(multimodal)
        self.boundary_policy = None

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
        if self.multimodal:
            _check_pilot_interfaces(self.network,set(self.network.pose_dims),set(self.network.action_dims))
            require('camera' in self.network.streams and len(self.network.pose_dims)==2,
                    'Multimodal smoke requires robot/human plus camera')
            require(torch.distributed.get_backend()=='nccl', 'Multimodal production smoke requires NCCL')
            prop=torch.cuda.get_device_properties(pl_module.device)
            require('H100' in prop.name and prop.major>=9,'Production smoke requires actual H100')
            image=self.network.image_model
            require(image.token_execution=='packed_valid' and image.blocks[0].attention_backend=='flash4',
                    'Production smoke requires Flash4 packed_valid')
            self.parameter_usage={name:0 for name,_ in pl_module.named_parameters()}
            self.embodiment_batches={emb:0 for emb in self.network.pose_dims}
            self.current_embodiment=None
            self.boundary_policy=validate_boundary_sampling(trainer.datamodule.settings.get('boundary_sampling'))
            if self.boundary_policy is not None:
                self.boundary_counts={emb:_boundary_sample_counts({'pixel_metadata':[]}) for emb in self.network.pose_dims}
            require(len(trainer.datamodule.val_dataloader())>=100//self.world_size,
                    'Held-out loader cannot provide100 global validation batches')
            require(trainer.limit_val_batches==100//self.world_size,
                    'Distributed smoke must cap exactly100 global held-out batches')
        self.lr_policy = from_config(pl_module.hparams.config_tree.model)
        require(all(math.isclose(group["lr"], update_lr(self.lr_policy, 0), rel_tol=1e-12)
                    for optimizer in trainer.optimizers for group in optimizer.param_groups),
                "Wrong first optimizer update LR")
        verify_scheduler(self.lr_policy,
                         [c.scheduler.state_dict() for c in trainer.lr_scheduler_configs], 0)
        self.optimizer_lrs = []
        digest = weights_digest(pl_module.state_dict())
        initial = [None] * self.world_size
        torch.distributed.all_gather_object(initial, digest)
        require(len(set(initial)) == 1, "Fresh DDP initial weights differ")
        self.initial_weights = initial
        self.initial_image_weights = weights_digest(image_parameter_state(self.network))
        self.initial_state_control_weights = (
            weights_digest(state_control_parameter_state(pl_module.state_dict()))
            if "action" in self.network.streams else None
        )
        self.paired_rgb_batches = []

        def observe(module, args, outputs):
            require(
                set(outputs) == set(self.network.active_streams(args[1]) if self.multimodal else self.network.streams)
                and all(value.dtype == torch.bfloat16 for value in outputs.values()),
                "Actual active-stream predictions must execute in BF16",
            )
            if pl_module.training and torch.is_grad_enabled():
                self.bf16_outputs += 1
                if self.multimodal:self.current_embodiment=args[1]
                if len(self.paired_rgb_batches) < 3:
                    self.paired_rgb_batches.append(
                        paired_image_digest(args[0], args[2], args[3])
                    )

        self.precision_hook = self.network.register_forward_hook(observe)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.multimodal:
            require(len(batch)==1,'Native training batches must be homogeneous by embodiment')
            for emb,sample in batch.items():
                _check_multimodal_sample(sample,boundary_policy=self.boundary_policy)
                if self.boundary_policy is not None:
                    for record in sample['pixel_metadata']:
                        require(record.get('boundary_sampling',{}).get('policy')==self.boundary_policy,
                                'Actual boundary policy differs from context-bound opt-in')
                    counts=_boundary_sample_counts(sample)
                    for key,value in counts.items():self.boundary_counts[emb][key]+=value
                else:
                    require(all('boundary_sampling' not in record for record in sample['pixel_metadata']),
                            'Legacy smoke cannot silently admit boundary uncertainty')

    def on_before_backward(self, trainer, pl_module, loss):
        require(
            loss.dtype == torch.float32 and bool(torch.isfinite(loss)),
            "Joint MSE must be finite FP32",
        )
        self.fp32_losses += 1

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        expected_lr = update_lr(self.lr_policy, trainer.global_step)
        require(
            all(
                math.isclose(g["lr"], expected_lr, rel_tol=1e-12)
                for g in optimizer.param_groups
            ),
            "Actual optimizer LR does not follow the3000-update warmup",
        )
        self.optimizer_lrs.append(expected_lr)
        if self.multimodal:
            emb=self.current_embodiment
            require(emb in self.embodiment_batches,'No actual training embodiment recorded')
            self.embodiment_batches[emb]+=1
            active=[p for p in self.network.parameters() if p.grad is not None]
            require(active and all(p.grad.dtype==torch.float32 and torch.isfinite(p.grad).all() for p in active),
                    'Active multimodal gradients must be finite FP32')
            for other,bank in self.network.proprio_banks.items():
                if other!=emb:require(all(p.grad is None for p in bank.parameters()),'Inactive private proprio gradient')
            if emb not in self.network.action_dims:
                require(all(p.grad is None for p in self.network.action.parameters()),'Human action bank must be unused')
            for name,param in pl_module.named_parameters():self.parameter_usage[name]+=int(param.grad is not None)
            self.gradients_checked=True
            return
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
        if self.multimodal:
            # Exact contiguous validation shards may put an embodiment entirely
            # on nonzero ranks. Elect one actual owner per current embodiment;
            # every rank participates even after its sample is already saved.
            rank_embodiments = [None] * self.world_size
            torch.distributed.all_gather_object(rank_embodiments, sorted(batch))
            writers = {}
            for rank, names in enumerate(rank_embodiments):
                for name in names:
                    writers.setdefault(name, rank)
            for emb,sample in batch.items():
                if writers[emb] != trainer.global_rank:continue
                _check_multimodal_sample(sample)
                path=self.output/f'sample-input-{emb}.pt'
                if path.exists():continue
                def cpu(value):
                    if torch.is_tensor(value):return value.detach().cpu()
                    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
                    if isinstance(value,list):return [cpu(v) for v in value]
                    if isinstance(value,tuple):return tuple(cpu(v) for v in value)
                    return value
                sample=cpu(sample)
                require('language_features' in sample and 'pixel_metadata' in sample,
                        'Need real subtask/language camera sample provenance')
                for stream in sample['world'].values():
                    stream['value']=torch.where(stream['observed'][...,None],stream['value'],float('nan'))
                torch.save(sample,path)
            # Publish completed files before any rank checks both embodiments
            # at train end. This never moves a held-out example between ranks.
            torch.distributed.barrier()
            return
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
                }.get(name,name+"_normalized_mse")
                for name in self.network.streams
            )
        }
        if self.multimodal:
            expected={f'val/{emb}/{kind}/{"action_command_mse" if stream=="action" else stream+"_normalized_mse"}'
                      for emb in self.network.pose_dims for kind in ('forecast','completion')
                      for stream in self.network.active_streams(emb)}
            require(all(count>0 for count in self.embodiment_batches.values()),'Both embodiments must optimize')
            _check_pilot_adam_usage(trainer.optimizers[0],pl_module.named_parameters(),self.parameter_usage)
            require(all((self.output/f'sample-input-{emb}.pt').is_file() for emb in self.network.pose_dims),
                    'Both embodiments need actual validation samples')
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
            initial_state_control_weights_sha256=self.initial_state_control_weights,
            paired_rgb_batch_sha256=self.paired_rgb_batches,
            bf16_training_outputs=self.bf16_outputs,
            fp32_losses=self.fp32_losses,
            fp32_gradients_verified=self.gradients_checked,
            lr_warmup_steps=self.lr_policy["warmup_optimizer_steps"],
            lr_policy=self.lr_policy,
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
        if self.multimodal:
            record.update(schema='joint-multimodal-fit-v1',multimodal=True,
                parameter_update_counts=self.parameter_usage,embodiment_optimizer_batches=self.embodiment_batches,
                active_streams_by_embodiment={e:list(self.network.active_streams(e)) for e in self.network.pose_dims},
                private_bank_optimizer_steps_verified=True,human_action_absent=True,
                subtask_boundaries_verified=True,language_conditioning_verified=True,camera_targets_verified=True,
                gpu_name=torch.cuda.get_device_properties(pl_module.device).name,
                gpu_capability=list(torch.cuda.get_device_capability(pl_module.device)),
                attention_backend='flash4',token_execution='packed_valid')
        if self.boundary_policy is not None:
            all_counts=[None]*self.world_size
            torch.distributed.all_gather_object(all_counts,self.boundary_counts)
            totals={emb:{key:sum(rank_counts[emb][key] for rank_counts in all_counts)
                         for key in self.boundary_counts[emb]} for emb in self.boundary_counts}
            require(all(counts['samples']>0 for counts in totals.values())
                    and sum(counts['outward_samples'] for counts in totals.values())>0,
                    'Changed sampling smoke must exercise both embodiments and actual bounded outward frames')
            # Explicitly false: this arm does not claim every target is within
            # the original reviewed interval. New admission must read the policy.
            record.update(subtask_boundaries_verified=False,
                original_subtask_annotation_identity_verified=True,
                subtask_boundary_verification_policy='annotation_boundary_uncertainty',
                boundary_uncertainty=dict(schema='boundary-uncertainty-smoke-realizations-v1',
                    policy=self.boundary_policy,all_rank_counts=all_counts,global_counts_by_embodiment=totals,
                    synchronized_native_streams_verified=True,original_instruction_retained=True,
                    neighbor_instruction_equivalence_asserted=False,validation_sampling_unchanged=True))
        write_once(self.output / f"fit-rank{trainer.global_rank}.json", record)


def _check_pilot_interfaces(net, expected, action_embodiments):
    require(set(net.pose_dims) == set(expected) and set(net.camera_dims) == set(expected),
            'Extra/missing embodiment banks in resolved model')
    require(set(net.action_dims) == set(action_embodiments),
            'Extra/missing action banks in resolved model')
    require(net.separate_proprio_experts, 'Full private proprio banks required')


def _check_pilot_adam_usage(optimizer, named_parameters, usage):
    for name,p in named_parameters:
        item=optimizer.state.get(p)
        require((item is None)==(usage[name]==0),'Adam state exists for unused parameter')
        if item is not None:
            require(int(item['step'])==usage[name], 'Private-bank Adam update count mismatch')
            require(all(item[k].dtype==torch.float32 and torch.isfinite(item[k]).all() for k in ('exp_avg','exp_avg_sq')),
                    'Adam moments must be finite FP32')


def validate_multimodal_pilot(config_path, output, *, optimizer_updates=4, postflight_only=False):
    # Multiple embodiment/dtype/autograd decode specializations are expected.
    # Keep compilation mandatory and bounded; never fall back to eager execution.
    with torch._dynamo.config.patch(recompile_limit=64, accumulated_recompile_limit=256,
                                    suppress_errors=False):
        return _validate_multimodal_pilot(config_path, output,
            optimizer_updates=optimizer_updates, postflight_only=postflight_only)


def _validate_multimodal_pilot(config_path, output, *, optimizer_updates=4, postflight_only=False):
    """Explicit one-GPU code validation on the reviewed TRAIN pilot.

    Reuses the declared pipeline, optimizer, sampler and prepared-data loader.
    This is not a production training launcher, held-out score, maximum-shape
    capacity claim, or the historical 50-update/100-anchor DDP smoke contract.
    """
    import inspect
    import random
    import numpy as np
    from hydra.utils import instantiate
    from egomimic.rldb.pixel_proprio import PreparedPixelProprioDataset, collate_pixel_proprio

    require(optimizer_updates == 4, 'Pilot contract is exactly four diagnostic optimizer updates')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            'Pilot validation requires exactly one visible allocated GPU')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    require(not (root/'pilot-validation.json').exists(), 'Completed pilot validation is immutable')
    checkpoint=root/'validation-epoch0000-step000000004.ckpt'
    if postflight_only:
        require(checkpoint.is_file(), 'Postflight requires the existing optimizer checkpoint')
    else:
        require(not checkpoint.exists(),
                'Completed optimizer checkpoint exists; resume postflight, never repeat optimizer updates')
    config_bytes=Path(config_path).read_bytes()
    config_sha256=hashlib.sha256(config_bytes).hexdigest()
    cfg = OmegaConf.load(config_path)
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    accepted = set(inspect.signature(PreparedPixelProprioDataset).parameters)
    settings = {k:v for k,v in data_cfg.items() if k in accepted}
    settings.update(split='train')
    data = PreparedPixelProprioDataset(**settings)
    batches,offset = {},0
    for (episode,unit),count in zip(data.sample_units,data.counts):
        emb = episode['embodiment']
        if emb not in batches:
            sample=data[(offset,0,0)]
            require(not any(s['observed'].any() for s in sample['world'].values()),
                    'Pilot must use the selected independent-noise training contract')
            batches[emb] = collate_pixel_proprio([sample])
        offset += count
    expected = {e['embodiment'] for e in data.manifest['episodes']}
    require(set(batches) == expected and len(expected) == 2,
            'Both native robot and human temporal units are required')
    torch.manual_seed(int(cfg.seed));np.random.seed(int(cfg.seed));random.seed(int(cfg.seed))

    def construct():
        algo=instantiate(cfg.model.pipeline, device='cuda')
        net=algo.pipeline.stage_by_id('joint_denoiser').model
        _check_pilot_interfaces(net, expected, cfg.data.action_embodiments)
        require(net.image_model.compile_mode == 'default' and net.image_model.compile_dynamic,
                'Pilot must exercise native dynamic Inductor compilation')
        require(all(p.dtype == torch.float32 and p.requires_grad for p in algo.nets.parameters()),
                'Pilot stored weights must be trainable FP32')
        return algo,net

    algo,net=construct()
    optimizer=instantiate(cfg.model.optimizer, params=algo.nets.parameters())
    if callable(optimizer):optimizer=optimizer()
    scheduler=instantiate(cfg.model.scheduler, optimizer=optimizer) if cfg.model.scheduler else None
    policy=from_config(cfg.model)
    order=sorted(expected)
    usage={name:0 for name,_ in algo.nets.named_parameters()}
    rows=[]
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    if not postflight_only:
        for step in range(optimizer_updates):
            emb=order[step % len(order)]
            algo.nets.train();optimizer.zero_grad(set_to_none=True)
            batch=algo.process_batch_for_training(batches[emb])
            with torch.autocast('cuda',dtype=torch.bfloat16):
                predicted=algo.forward_training(batch)
                losses=algo.compute_losses(predicted,batch)
            loss=losses['loss']
            require(loss.dtype==torch.float32 and torch.isfinite(loss),'Diagnostic loss must be finite FP32')
            # The model-owned denoiser stage records the actual active predictions.
            sample=predicted[emb] if emb in predicted else predicted
            outputs=sample['joint/prediction']
            require(set(outputs)==set(net.active_streams(emb)) and
                    all(v.dtype==torch.bfloat16 for v in outputs.values()),
                    'Active expert outputs did not execute in BF16')
            loss.backward()
            active=[p for p in algo.nets.parameters() if p.grad is not None]
            require(active and all(p.grad.dtype==torch.float32 and torch.isfinite(p.grad).all() for p in active),
                    'Nonfinite/non-FP32 active gradients')
            for other,bank in net.proprio_banks.items():
                if other != emb:require(all(p.grad is None for p in bank.parameters()),'Inactive private bank got gradients')
            if emb not in net.action_dims:
                require('action' not in outputs and all(p.grad is None for p in net.action.parameters()),
                        'Human action path must be absent and unused')
            for name,p in algo.nets.named_parameters():usage[name]+=int(p.grad is not None)
            actual_lr=optimizer.param_groups[0]['lr']
            require(all(math.isclose(g['lr'],update_lr(policy,step),rel_tol=1e-12) for g in optimizer.param_groups),
                    'Diagnostic optimizer uses a different LR policy')
            optimizer.step()
            if scheduler is not None:scheduler.step()
            rows.append(dict(step=step+1,embodiment=emb,loss=float(loss.detach()),lr=actual_lr,
                             active_streams=list(outputs),active_gradient_parameters=sum(p.numel() for p in active)))
        state=optimizer.state_dict()
        _check_pilot_adam_usage(optimizer, algo.nets.named_parameters(), usage)
        checkpoint=root/'validation-epoch0000-step000000004.ckpt'
        require(not checkpoint.exists(),'Diagnostic checkpoint may not be overwritten')
        rng=dict(torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state(),
                 numpy=np.random.get_state(),python=random.getstate())
        torch.save(dict(schema='joint-multimodal-pilot-validation-v1',global_step=4,epoch=0,
            state_dict=algo.nets.state_dict(),optimizer_state=state,scheduler_state=None if scheduler is None else scheduler.state_dict(),
            rng=rng,parameter_update_counts=usage,optimizer_rows=rows,config_raw_sha256=config_sha256,config=OmegaConf.to_container(cfg,resolve=True)),checkpoint)
    loaded=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    require(loaded['schema']=='joint-multimodal-pilot-validation-v1' and
            loaded['global_step']==4 and loaded['epoch']==0,
            'Postflight checkpoint is not the completed four-update diagnostic')
    require(loaded['config']==OmegaConf.to_container(cfg,resolve=True),
            'Postflight resolved config differs from checkpoint')
    if 'config_raw_sha256' in loaded:
        require(loaded['config_raw_sha256']==config_sha256, 'Postflight config bytes differ')
    if postflight_only:
        algo.nets.load_state_dict(loaded['state_dict'],strict=True)
        optimizer.load_state_dict(loaded['optimizer_state'])
        if scheduler is not None:scheduler.load_state_dict(loaded['scheduler_state'])
        usage=loaded['parameter_update_counts']
        require(set(usage)==set(dict(algo.nets.named_parameters())), 'Checkpoint parameter usage identity differs')
        require(all(count in (0,2,4) for count in usage.values()) and 4 in usage.values(),
                'Checkpoint optimizer update history is invalid')
        _check_pilot_adam_usage(optimizer,algo.nets.named_parameters(),usage)
        rows=loaded.get('optimizer_rows',[])
    restored,restored_net=construct()
    restored.nets.load_state_dict(loaded['state_dict'],strict=True)
    require(weights_digest(algo.nets.state_dict())==weights_digest(restored.nets.state_dict()),'Strict model reload changed weights')
    ro=instantiate(cfg.model.optimizer,params=restored.nets.parameters())
    if callable(ro):ro=ro()
    rs=instantiate(cfg.model.scheduler,optimizer=ro) if cfg.model.scheduler else None
    ro.load_state_dict(loaded['optimizer_state'])
    if rs is not None:rs.load_state_dict(loaded['scheduler_state'])
    require((None if rs is None else rs.state_dict())==loaded['scheduler_state'],'Scheduler reload differs')
    for left,right in zip(optimizer.state.values(),ro.state.values()):
        require(set(left)==set(right),'Optimizer reload keys differ')
        for key,value in left.items():
            require(torch.equal(value.cpu(),right[key].cpu()) if torch.is_tensor(value) else value==right[key],
                    'Optimizer reload value differs')
    torch.set_rng_state(loaded['rng']['torch_cpu']);torch.cuda.set_rng_state(loaded['rng']['torch_cuda'])
    np.random.set_state(loaded['rng']['numpy']);random.setstate(loaded['rng']['python'])
    require(torch.equal(torch.get_rng_state(),loaded['rng']['torch_cpu']) and
            torch.equal(torch.cuda.get_rng_state(),loaded['rng']['torch_cuda']) and
            random.getstate()==loaded['rng']['python'] and
            np.array_equal(np.random.get_state()[1],loaded['rng']['numpy'][1]),'RNG restoration differs')
    validation=[]
    for emb in order:
        algo.nets.eval();restored.nets.eval()
        batch=algo.process_batch_for_training(batches[emb])
        restored_batch=restored.process_batch_for_training(batches[emb])
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            torch.manual_seed(1701);original=algo.forward_training(batch)
            torch.manual_seed(1701);replayed=restored.forward_training(restored_batch)
            a=original[emb] if emb in original else original
            b=replayed[emb] if emb in replayed else replayed
            for key,value in a['joint/prediction'].items():
                require(torch.equal(value,b['joint/prediction'][key]),'Reload prediction changed under identical noise RNG')
            metrics=restored.compute_losses(replayed,restored_batch)
        require(all(torch.isfinite(v) for v in metrics.values() if torch.is_tensor(v)),'Nonfinite diagnostic validation')
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            query=restored.process_batch_for_training(batches[emb])
            for stream in query[emb]['world'].values():
                stream['value']=torch.where(stream['observed'][...,None],stream['value'],float('nan'))
            torch.manual_seed(1701)
            sampled=restored.forward_eval(query)[emb]['image_proprio_prediction']
        require(set(sampled)==set(restored_net.active_streams(emb)) and
                all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in sampled.values()),
                'Strictly restored native sampler produced invalid active streams')
        sampler=restored.pipeline.stage_by_id('joint_sampler').sampler
        require(sampler.state_steps==20 and (emb not in net.action_dims or sampler.action_steps==20),
                'Configured native sampler schedule changed')
        validation.append(dict(embodiment=emb,loss=float(metrics['loss']),split='TRAIN_pilot_diagnostic',
                               native_sampler_steps=20,sampled_streams=list(sampled)))
    torch.cuda.synchronize()
    with checkpoint.open('rb') as checkpoint_stream:
        checkpoint_sha=hashlib.file_digest(checkpoint_stream,'sha256').hexdigest()
    result=dict(schema='joint-multimodal-pilot-validation-v1',status='PASS',optimizer_updates=4,
        production_training=False,held_out_benchmark=False,full_shape_capacity=False,
        config_raw_sha256=config_sha256,config_path=str(config_path),
        validation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rows=rows,optimizer_telemetry_available=bool(rows),postflight_only=postflight_only,
        optimizer_updates_this_invocation=0 if postflight_only else 4,
        dynamo_recompile_limit=64,dynamo_accumulated_recompile_limit=256,
        validation=validation,strict_model_optimizer_scheduler_rng_reload=True,
        parameter_update_counts=usage,parameters_total=sum(p.numel() for p in algo.nets.parameters()),
        parameters_trainable=sum(p.numel() for p in algo.nets.parameters() if p.requires_grad),
        lr_policy=policy,compile_mode=net.image_model.compile_mode,compile_dynamic=net.image_model.compile_dynamic,
        attention_backend=net.image_model.blocks[0].attention_backend,token_execution=net.image_model.token_execution,
        elapsed_seconds=time.perf_counter()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        checkpoint=dict(path=str(checkpoint),bytes=checkpoint.stat().st_size,
            sha256=checkpoint_sha),torch_version=torch.__version__)
    write_once(root/'pilot-validation.json',result)
    return result


def _capacity_manifest_sha256(cfg, expected):
    require(isinstance(expected,str) and len(expected)==64
            and all(c in '0123456789abcdef' for c in expected),
            'Capacity requires exact request data manifest SHA256')
    actual=hashlib.sha256(Path(cfg.data.manifest_path).read_bytes()).hexdigest()
    require(actual==expected,'Capacity data manifest differs from bound request')
    return actual


def _optimizer_capacity_reserve_bytes(parameters):
    parameters=list(parameters)
    require(parameters and all(p.dtype==torch.float32 for p in parameters),
            'Optimizer capacity reserve requires FP32 parameters')
    return 3*sum(p.numel()*p.element_size() for p in parameters)


def validate_multimodal_capacity(config_path, output, *, source_manifest_sha256, data_manifest_sha256):
    """One real maximum-shape distributed forward/backward, zero optimizer steps.

    Invoke through the existing GPU allocation with torchrun, matching the
    production rank batch. Uses an actual eligible robot subtask and all three
    native RGB cameras; no synthetic all-valid camera-pose labels are invented.
    """
    import inspect
    from hydra.utils import instantiate
    from egomimic.rldb.pixel_proprio import PreparedPixelProprioDataset, collate_pixel_proprio
    from torch.nn.parallel import DistributedDataParallel
    world=int(os.environ['WORLD_SIZE']);rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    require(world in (2,4),'Capacity check requires production two/four ranks')
    require(torch.cuda.device_count()==world,
            'Capacity visible CUDA device count must equal WORLD_SIZE before device or model access')
    torch.cuda.set_device(local)
    created=not torch.distributed.is_initialized()
    if created:torch.distributed.init_process_group('nccl')
    destination=Path(output);root=destination.parent;root.mkdir(parents=True,exist_ok=True)
    require(not destination.exists(),'Capacity proof is immutable')
    require(len(source_manifest_sha256)==64 and all(c in '0123456789abcdef' for c in source_manifest_sha256),
            'Capacity requires exact source manifest SHA256')
    cfg=OmegaConf.load(config_path)
    data_manifest_sha256=_capacity_manifest_sha256(cfg,data_manifest_sha256)
    require(cfg.data.mode in ('discrete','continuous') and cfg.data.batch_size*world==32
            and cfg.data.distributed_world_size==world,'Wrong capacity temporal/global batch contract')
    settings=OmegaConf.to_container(cfg.data,resolve=True)
    accepted=set(inspect.signature(PreparedPixelProprioDataset).parameters)
    settings={k:v for k,v in settings.items() if k in accepted}
    frames=64 if cfg.data.mode=='continuous' else 16
    require(cfg.data.mode!='discrete' or cfg.data.discrete_frames==16,'Discrete capacity requires native16anchors')
    settings.update(split='train')
    if cfg.data.mode=='continuous':settings.update(min_frames=64,max_frames=64)
    data=PreparedPixelProprioDataset(**settings)
    offset=0;sample=None
    for (episode,unit),count in zip(data.sample_units,data.counts):
        if episode['embodiment'] in cfg.data.action_embodiments and len(episode['cameras'])==3:
            sample=collate_pixel_proprio([data[(offset,0,rank*cfg.data.batch_size+i)]
                                         for i in range(cfg.data.batch_size)])
            break
        offset+=count
    require(sample is not None,'No eligible actual64-frame three-camera robot subtask')
    emb=next(iter(sample));item=sample[emb]
    _check_multimodal_sample(item,boundary_policy=settings.get('boundary_sampling'))
    require(all(len(m['frame_indices'])==frames and len(m['camera_names'])==3
                and m['image_size']==[160,160] and m['patch_size']==16 for m in item['pixel_metadata']),
            'Capacity sample is not actual64x3x160x160 native RGB')
    require(item['world']['image']['value'].shape[:2]==(cfg.data.batch_size,frames*3*100)
            and item['world']['image']['valid'].all(),'Missing actual maximum RGB tokens')
    torch.manual_seed(int(cfg.seed))
    algo=instantiate(cfg.model.pipeline,device=f'cuda:{local}')
    network=algo.pipeline.stage_by_id('joint_denoiser').model
    image=network.image_model;props=torch.cuda.get_device_properties(local)
    require('H100' in props.name and props.major>=9 and image.compile_mode=='default'
            and image.compile_dynamic and image.compile_backend=='inductor'
            and image.token_execution=='packed_valid' and image.blocks[0].attention_backend=='flash4',
            'Capacity requires exact H100/Flash4/packed/compiled execution')
    class CapacityPass(torch.nn.Module):
        def __init__(self):
            super().__init__();self.nets=algo.nets
        def forward(self,batch):
            predictions=algo.forward_training(batch)
            require(all(value.dtype==torch.bfloat16 for result in predictions.values()
                        for value in result['joint/prediction'].values()),
                    'Maximum-shape actual expert outputs must be BF16')
            return algo.compute_losses(predictions,batch)['loss']
    require(all(p.dtype==torch.float32 and p.requires_grad for p in algo.nets.parameters()),
            'Maximum-shape stored expert weights must be trainableFP32')
    wrapped=DistributedDataParallel(CapacityPass(),device_ids=[local],find_unused_parameters=True)
    batch=algo.process_batch_for_training(sample)
    # Conservative declared memory reserve: two FP32 Adam moments plus one
    # FP32 foreach workspace. These are memory-only buffers, never optimizer state.
    optimizer_reserve_bytes=_optimizer_capacity_reserve_bytes(wrapped.parameters())
    optimizer_memory_reserve=torch.empty(optimizer_reserve_bytes//4,device=f'cuda:{local}',dtype=torch.float32)
    optimizer_memory_reserve.zero_()
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    try:
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=wrapped(batch)
        require(loss.dtype==torch.float32 and torch.isfinite(loss),'Capacity loss must be finiteFP32')
        loss.backward()
        active=[p for p in wrapped.parameters() if p.grad is not None]
        require(active and all(p.dtype==torch.float32 and p.grad.dtype==torch.float32
                and torch.isfinite(p.grad).all() for p in active),'Capacity active gradients must be finiteFP32')
        torch.cuda.synchronize()
        record=dict(schema='native-multimodal-capacity-v1',status='PASS',rank=rank,world_size=world,
            job_id=os.environ['SLURM_JOB_ID'],global_batch_size=32,rank_batch_size=int(cfg.data.batch_size),
            frames=frames,cameras=3,image_size=[160,160],patch_size=16,image_tokens_per_sample=frames*300,
            per_rank_batch=int(cfg.data.batch_size),backend='flash4',compiled_bf16_forward_backward=True,
            peak_memory_bytes=torch.cuda.max_memory_allocated(),device_memory_bytes=props.total_memory,
            optimizer_memory_reserve_bytes=optimizer_reserve_bytes,
            optimizer_memory_reserve_policy='2FP32AdamMoments+1FP32ForeachWorkspace; nooptimizerstateorupdates',
            source_manifest_sha256=source_manifest_sha256,
            actual_embodiment=emb,optimizer_updates=0,attention_backend='flash4',token_execution='packed_valid',
            compile_mode='default',compile_dynamic=True,bf16_forward=True,fp32_weights_gradients=True,
            loss=float(loss.detach()),gpu_uuid=str(props.uuid),gpu_name=props.name,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            seconds=time.perf_counter()-started,config_raw_sha256=hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
            config_sha256=hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
            data_manifest_sha256=data_manifest_sha256)
        write_once(root/f'{destination.stem}-rank{rank}.json',record)
        ranks=[None]*world;torch.distributed.all_gather_object(ranks,record)
        require(len({r['gpu_uuid'] for r in ranks})==world,'Capacity ranks must use distinct GPUs')
        if rank==0:write_once(destination,dict(record,ranks=ranks))
    finally:
        if created:torch.distributed.destroy_process_group()
