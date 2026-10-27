"""Explicit ImageNet image-generation admission; temporal contracts stay exact.

This module owns the independent-image source semantics, not model execution or
Slurm submission. Unknown request kinds still fail in the legacy selectors.
"""
from __future__ import annotations
import copy
import math
from pathlib import Path

KIND = "native-imagenet-world-model-v1"
STREAMS = ["image", "proprio", "action", "camera"]
COUNTS = dict(image=158875392, proprio=95083666, action=30068372,
              camera=30068370, language_shared=131200)
INTERFACES = dict(pose_dims={"eva_bimanual":20,"human_bimanual":126},
                  action_dims={"eva_bimanual":20},
                  camera_dims={"eva_bimanual":9,"human_bimanual":9})
POLICY = dict(implementation="native_imagenet_joint", attention_backend="flash4",
              token_execution="packed_valid", compile_mode="default", compile_dynamic=True,
              compile_backend="inductor", recompile_limit=64, accumulated_recompile_limit=256)
METRIC = "val/imagenet/image_generation/image_normalized_mse"


def selected(request):
    return request.get("kind") == KIND


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def verify_model_execution(request, cfg):
    from scripts.train.joint.lr_policy import from_config, validate
    _need(selected(request), "Explicit ImageNet request kind required")
    _need(request["phase"] in ("smoke","full") and request["arm"] == "image_generation",
          "ImageNet is an independent-image generation arm")
    _need(request.get("executed_streams") == ["image"]
          and request["active_streams"] == STREAMS and request["expert_parameters"] == COUNTS
          and request["parameters"] == sum(COUNTS.values()) == 314227000
          and request["expert_widths"] == [1024,512,384,384],
          "ImageNet must retain the exact full joint architecture and dormant banks")
    _need(request.get("model_execution") == POLICY, "Unbound ImageNet execution policy")
    _need(validate(request["lr_policy"]) == dict(peak_lr=5e-6,warmup_optimizer_steps=3000,
          schedule="linear_warmup_then_constant"), "ImageNet optimizer policy is exact 5e-6/3000-warmup then constant")
    _need(cfg["model"]["scheduler"] == dict(_target_="egomimic.utils.schedulers.linear_warmup_constant_scheduler",warmup_steps=3000),
          "The maintained exact warmup scheduler is required")
    pipeline = cfg["model"]["pipeline"]
    ids, stages = pipeline["stage_ids"], pipeline["stages"]
    _need(pipeline["_target_"] == "egomimic.pipeline.algo.PipelineAlgo"
          and set(pipeline) == {"_target_","stage_ids","stages"}
          and ids == dict(joint_noising=0,joint_denoiser=1,image_loss=2,joint_sampler=3)
          and len(stages) == 4,
          "ImageNet uses only the maintained noising/denoising/image-loss/sampler graph")
    noiser, denoiser, loss, sampler = stages
    _need(noiser == dict(_target_="egomimic.pipeline.stages_joint_state_action.JointNoisingStage",
                         noising=dict(_target_="egomimic.models.joint_image_proprio.CoupledImageProprioNoising",
                                      maximum=1.0,active_streams=STREAMS)),
          "ImageNet corruption must remain native uniform noise; no fused weighting")
    _need(denoiser["_target_"] == "egomimic.pipeline.stages_joint_state_action.JointDenoiserStage"
          and denoiser.get("language_key") == "language_features"
          and set(denoiser) == {"_target_","language_key","model"}, "Language denoiser binding changed")
    net = denoiser["model"]; core = net["image_model"]
    _need(net["_target_"] == "egomimic.models.joint_image_proprio.JointImageProprioModel"
          and net["active_streams"] == STREAMS and net["separate_proprio_experts"] is True
          and net["proprio_width"] == 512 and net["action_width"] == net["camera_width"] == 384
          and net["mlp_ratio"] == 4 and net["language_feature_dim"] == 1024
          and net["language_width"] == 128 and net["camera_max_count"] == 3
          and all(net[k] == v for k,v in INTERFACES.items()), "Joint parameter interfaces changed")
    expected = dict(_target_="egomimic.models.image_proprio.ImageProprioModel",objective="visual_only",
        require_observed_context=False,phase_a_condition_on_proprio=False,pose_dims=None,action_dims=None,
        image_dim=768,width=1024,depth=8,heads=8,mlp_ratio=4,max_cameras=3,
        use_camera_embedding=True,use_modality_embedding=False,use_embodiment_embedding=False,
        use_phase_embedding=False,noise_conditioning="adaln_zero",position_encoding="spatiotemporal_rope",
        rope_axis_dims=[64,32,32],spatial_grid_size=[10,10],max_tokens=19520,time_mode="continuous",
        attention_backend="flash4",attention_dtype="bfloat16",strict_fp32=False,
        attention_query_chunk_size=0,activation_checkpointing=False,allow_patch_compression=False,
        loss_reduction="mean_per_example",token_execution="packed_valid",compile_mode="default",
        compile_dynamic=True,compile_backend="inductor",compile_recompile_limit=64,
        compile_accumulated_recompile_limit=256,qk_norm=True,use_rmsnorm=True,use_swiglu=True,
        swiglu_multiple_of=64)
    _need(core == expected, "ImageNet image core differs from the exact QK/RMS/SwiGLU constructor")
    _need(loss == dict(_target_="egomimic.pipeline.stages_joint_state_action.JointLossStage",
                       stream="image",weight=1.0,loss=dict(_target_="egomimic.models.pixel_diffusion.PlainPixelMSE",
                                                         reduction="mean_per_example")),
          "Only unweighted plain FP32 image x0 MSE is admitted")
    _need(sampler == dict(_target_="egomimic.pipeline.stages_joint_state_action.JointSamplerStage",
                         language_key="language_features",sampler=dict(
                         _target_="egomimic.models.joint_image_proprio.JointImageProprioSampler",
                         state_steps=20,action_steps=20)), "Actual image generation requires20 native DDIM steps")
    _need(cfg["model"]["_target_"] == "egomimic.pl_utils.pl_model.ModelWrapper"
          and cfg["model"]["optimizer"] == dict(_target_="torch.optim.AdamW",_partial_=True,lr=5e-6,weight_decay=0.01)
          and cfg["model"]["training_behavior"] == dict(
              _target_="egomimic.pl_utils.world_model_behavior.WorldModelTrainingBehavior",distributed_noise_seed=42),
          "The shared wrapper, exact AdamW and rank RNG behavior are required")
    data, trainer = cfg["data"], cfg["trainer"]
    _need(data["_target_"] == "egomimic.rldb.imagenet_imagefolder.ImageNetImageFolderDataModule"
          and data["expected_classes"] == 1000 and data["expected_train_images"] == 1281167
          and data["expected_val_images"] == 50000 and data["validation_batch_size"] == 1
          and data["distributed_world_size"] == trainer["devices"] == request["world_size"] == 4
          and data["batch_size"] == request["per_rank_batch"] == 8 and request["global_batch"] == 32
          and data["seed"] == cfg["seed"] == 42,
          "Exact official ImageNet source and4x8 topology are required")
    contract = request["image_generation_contract"]
    _need(contract["schema"] == "native-imagenet-dataset-v1", "Exact ImageNet source contract required")
    for field,key in (("manifest","data_manifest"),("class_language",None),
                      ("language_features",None),("validation_anchor",None)):
        binding = contract[field]
        cfgfield = {"manifest":"manifest","class_language":"class_language",
                    "language_features":"language_features","validation_anchor":"validation_anchor"}[field]
        _need(data[cfgfield+"_path"] == binding["path"] and data[cfgfield+"_sha256"] == binding["sha256"],
              "Unbound ImageNet data input: "+field)
        if key is not None:
            _need(request[key] == binding["path"] and request[key+"_sha256"] == binding["sha256"],
                  "Request/adapter manifest differs")
    _need(data["data_root"] == request["data_cache"] and from_config(cfg["model"]) == validate(request["lr_policy"]),
          "Unbound physical cache or optimizer schedule")
    _need(trainer["strategy"] == dict(_target_="lightning.pytorch.strategies.DDPStrategy",find_unused_parameters=True)
          and trainer["accelerator"] == "gpu" and trainer["precision"] == "bf16-mixed" and trainer["num_nodes"] == 1
          and trainer["accumulate_grad_batches"] == 1 and trainer["use_distributed_sampler"] is False
          and trainer.get("fast_dev_run",False) is False and trainer.get("overfit_batches",0.0) == 0.0
          and trainer.get("enable_checkpointing",True) is True and trainer.get("barebones",False) is False
          and trainer.get("gradient_clip_val") is None and not cfg["model"]["enable_grad_norm"]
          and cfg["runtime"]["disable_tf32"] is True and cfg["runtime"]["slurm_requeue_owner"] == "none",
          "Conditional DDP/precision/clipping/runtime changed")
    _need(cfg["normalizer"] is None and cfg["norm_stats"] is None and cfg["ckpt_path"] is None
          and cfg["mode"] == "train" and cfg.get("norm_stats_only",False) is False and cfg["val_at_end"] is False
          and cfg.get("val_at_start",False) is False,
          "Fresh image-only fixed normalization and scheduled validation are required")
    evaluator = cfg["evaluator"]
    _need(evaluator["_target_"] == "egomimic.eval.image_proprio_eval.ImageProprioEval"
          and evaluator["objective"] == "image_generation" and evaluator["sample_seed"] == 42
          and evaluator["active_streams_from_data"] is False
          and evaluator["pixel_diagnostics"]["write_video"] is False
          and evaluator["pixel_diagnostics"]["max_samples_per_source_kind"] == 32
          and evaluator["pixel_diagnostics"]["contact_sheet_frames"] == 1,
          "ImageNet must validate class-conditioned generation with fixed32 PNG previews")
    _need(cfg["logger"]["csv"]["_target_"] == "lightning.pytorch.loggers.CSVLogger",
          "The maintained CSV producer is required")
    callbacks = cfg["callbacks"]
    allowed_callbacks={"egomimic.pl_utils.joint_smoke.ConditionalOptimizerUsage",
        "lightning.pytorch.callbacks.ModelCheckpoint","egomimic.pl_utils.imagenet_smoke.ImageNetSmokeProof"}
    _need(all(v is None or isinstance(v,dict) and v.get("_target_") in allowed_callbacks for v in callbacks.values()),
          "Unreviewed callbacks cannot alter the fixed ImageNet training contract")
    usage = [v for v in callbacks.values() if isinstance(v,dict) and
             v.get("_target_") == "egomimic.pl_utils.joint_smoke.ConditionalOptimizerUsage"]
    _need(len(usage) == 1, "Exactly one conditional optimizer usage callback is required")
    _need(type(trainer["limit_val_batches"]) is float and trainer["limit_val_batches"] == 1.0
          and type(trainer["limit_train_batches"]) is float and trainer["limit_train_batches"] == 1.0
          and trainer["max_epochs"] == -1 and trainer.get("max_time") is None
          and trainer["min_epochs"] == 0 and trainer["num_sanity_val_steps"] == 0
          and trainer["check_val_every_n_epoch"] is None,
          "Every scheduled pass must consume all1000 frozen anchors without sanity/epoch validation")
    _need(not request.get("continuation") and not request.get("interruption")
          and request.get("allocation_boundaries",[300000]) == [300000]
          and request.get("archive_policy") is None,
          "This ImageNet admission is fresh-only; no unrelated recovery proof is inherited")
    checkpoint_callbacks = [v for v in callbacks.values() if isinstance(v,dict) and
                            v.get("_target_") == "lightning.pytorch.callbacks.ModelCheckpoint"]
    smoke_callbacks = [v for v in callbacks.values() if isinstance(v,dict) and
                       v.get("_target_") == "egomimic.pl_utils.imagenet_smoke.ImageNetSmokeProof"]
    if request["phase"] == "smoke":
        _need(not checkpoint_callbacks and len(smoke_callbacks) == 1,
              "Smoke admits exactly one instrumentation callback and no production checkpoint callback")
        cb = callbacks["joint_smoke"]
        _need(request["target_steps"] == trainer["max_steps"] == trainer["val_check_interval"] == 50
              and cb == dict(_target_="egomimic.pl_utils.imagenet_smoke.ImageNetSmokeProof",
                             output=request["output"],expected_steps=50,warmup_steps=10)
              and callbacks.get("model_checkpoint") is None,
              "Smoke must optimize50 then validate1000, with only its diagnostic save")
        _need("wandb" not in cfg["logger"] and evaluator["pixel_diagnostics"]["log_wandb"] is False,
              "Smoke cannot write a production W&B identity")
    else:
        _need(len(checkpoint_callbacks) == 1 and not smoke_callbacks,
              "Full admits exactly one production checkpoint callback and no smoke callback")
        checkpoint = callbacks["model_checkpoint"]
        _need(request["target_steps"] == trainer["max_steps"] == 300000
              and trainer["val_check_interval"] == 20000 and "joint_smoke" not in callbacks,
              "Full ImageNet budget is300k total with scheduled20k validation")
        _need(checkpoint["_target_"] == "lightning.pytorch.callbacks.ModelCheckpoint"
              and checkpoint["every_n_train_steps"] == 20000 and checkpoint["every_n_epochs"] == 0
              and checkpoint["save_top_k"] == -1 and checkpoint["save_last"] == "link"
              and checkpoint["save_on_train_epoch_end"] is False and not checkpoint.get("train_time_interval")
              and checkpoint["filename"] == "epoch_{epoch:04d}-step_{step:09d}"
              and checkpoint.get("auto_insert_metric_name",True) is True,
              "Full retains all15 immutable scheduled checkpoints and last symlink")
        wb = cfg["logger"]["wandb"]
        _need(wb["_target_"] == "lightning.pytorch.loggers.wandb.WandbLogger"
              and wb["id"] == request["run_id"] and wb["resume"] == "never"
              and wb["entity"] == "rl2-group" and wb["project"] == "mot-world-model"
              and wb["log_model"] is False and set(cfg["logger"]) == {"csv","wandb"},
              "Fresh same-identity W&B/source history is required")
    return "image_generation"


def compare_configs(production, smoke, request):
    """Keep exact data/context and model; permit only explicit diagnostics/budget/log paths."""
    _need(selected(request), "ImageNet comparison kind required")
    full, diagnostic = copy.deepcopy(production), copy.deepcopy(smoke)
    def move(value):
        if isinstance(value,dict):return {k:move(v) for k,v in value.items()}
        if isinstance(value,list):return [move(v) for v in value]
        if isinstance(value,str):return value.replace(request["output"],smoke["paths"]["output_dir"]).replace(
            request["local_logs"],smoke["logger"]["csv"]["save_dir"])
        return value
    full = move(full)
    for key in ("model","data","seed","runtime","world_model_attention","world_model_execution",
                "normalizer","norm_stats","inference_config","val_at_end","mode","ckpt_path","data_context_loader"):
        _need(full.get(key) == diagnostic.get(key), "ImageNet science/context changed after smoke: "+key)
    for cfg in (full,diagnostic):
        cfg["trainer"].pop("max_steps");cfg["trainer"].pop("val_check_interval")
        cfg["callbacks"].pop("model_checkpoint",None)
        cfg["callbacks"].pop("joint_smoke",None)
        cfg["evaluator"]["pixel_diagnostics"].pop("artifact_dir")
        cfg["evaluator"]["pixel_diagnostics"].pop("log_wandb")
    for key in ("trainer","callbacks","evaluator"):
        _need(full[key] == diagnostic[key], "ImageNet smoke/full behavior differs: "+key)


def precheck_evidence(evidence):
    """Bound every declared smoke evidence file before the shared reader hashes it."""
    pending=[evidence]; count=0
    while pending:
        value=pending.pop(); count+=1
        _need(count <= 512, "Oversized evidence binding tree")
        if isinstance(value,dict):
            if "path" in value and "sha256" in value:
                _small_text(value["path"])
            else:
                pending.extend(value.values())
        elif isinstance(value,list):
            pending.extend(value)


def verify_completion(request, smoke, completion, preflight, capacity):
    """Pure interpretation of authenticated actual producers, after common native admission."""
    from scripts.train.joint.contract import canonical_uuid
    from scripts.train.joint.lr_policy import update_lr
    for key in ("kind","executed_streams","image_generation_contract"):
        _need(request.get(key) == smoke.get(key), "ImageNet source/semantics differ from smoke: "+key)
    job = str(request["smoke"]["job_id"])
    _need(completion.get("schema") == "imagenet-native-smoke-v1" and completion.get("status") == "PASS"
          and completion.get("job_id") == job and completion.get("world_size") == 4
          and completion.get("global_step") == 50 and completion.get("validation_batches") == 1000
          and completion.get("local_validation_batches") == 250
          and completion.get("rank_batch_size") == 8 and completion.get("global_batch_size") == 32
          and completion.get("distributed_backend") == "nccl" and completion.get("trainer_precision") == "bf16-mixed"
          and completion.get("parameters") == 314227000
          and completion.get("executed_parameters") == 159006592
          and completion.get("expert_parameters") == COUNTS and completion.get("active_streams") == ["image"]
          and completion.get("data_context_sha256") == preflight["provenance"]["data_context_sha256"],
          "Missing actual ImageNet50-update/1000-anchor smoke")
    for flag in ("strict_full_wrapper_reload","optimizer_sampler_all_rank_rng_verified",
                 "lr_scheduler_strictly_restored","all_rank_weights_agree","all_training_bf16_outputs",
                 "fp32_stored_model_and_adam","sample_finite","language_conditioning_verified",
                 "inactive_banks_unchanged","all50_lr_verified","all_rank_rng_payload_restorable"):
        _need(completion.get(flag) is True, "Missing actual ImageNet smoke proof: "+flag)
    _need(completion.get("sample_state_steps") == 20 and completion.get("sample_action_steps") == 0
          and completion.get("diagnostic_checkpoint_is_production_initializer") is False
          and completion.get("actual_denoiser_calls") == 20
          and completion.get("hidden_rgb_values_all_nan") is True
          and completion.get("genuine_finite_language_retained") is True,
          "Diagnostic checkpoint cannot initialize production")
    rows = completion["ranks"]
    _need(len(rows) == 4 and len({canonical_uuid(row["gpu_uuid"]) for row in rows}) == 4,
          "Four distinct actual smoke GPUs are required")
    first = rows[0]
    for rank,row in enumerate(rows):
        _need(row.get("schema") == "imagenet-joint-fit-v1" and row.get("status") == "FIT_PASS"
              and row.get("job_id") == job and row.get("rank") == rank and row.get("world_size") == 4
              and row.get("rank_batch_size") == 8 and row.get("global_batch_size") == 32
              and row.get("distributed_backend") == "nccl" and row.get("trainer_precision") == "bf16-mixed"
              and row.get("global_step") == row.get("bf16_training_outputs") == row.get("fp32_losses") == 50
              and row.get("validation_batches") == 1000 and row.get("local_validation_batches") == 250
              and row.get("validation_batch_indices") == list(range(rank*250,(rank+1)*250)),
              "Missing exact actual rank optimization/validation coverage")
        _need(row["expert_parameters"] == COUNTS and row["data_context_sha256"] == completion["data_context_sha256"]
              and row["metrics"] == first["metrics"] and set(row["metrics"]) == {METRIC}
              and all(type(v) in (float,int) and math.isfinite(v) for v in row["metrics"].values())
              and "H100" in row["gpu_name"] and row["gpu_capability"][0] >= 9,
              "ImageNet architecture/H100/finite validation evidence differs")
        for flag in ("fp32_gradients_verified","all_training_bf16_outputs","all50_lr_verified",
                     "inactive_banks_unchanged","language_conditioning_verified"):
            _need(row.get(flag) is True, "Missing actual ImageNet rank proof: "+flag)
        _need(row["lr_policy"] == request["lr_policy"] and len(row["optimizer_update_lrs"]) == 50
              and all(type(v) in (float,int) and math.isfinite(v)
                      and math.isclose(v,update_lr(request["lr_policy"],step),rel_tol=1e-12)
                      for step,v in enumerate(row["optimizer_update_lrs"])), "Actual50 optimizer LRs differ")
        names, partition, usage = row["parameter_names"], row["parameter_partition"], row["parameter_update_counts"]
        _need(names == first["parameter_names"] and partition == first["parameter_partition"]
              and usage == first["parameter_update_counts"] and len(names) == len(set(names))
              and set(names) == set(partition) == set(usage)
              and set(partition.values()) == {"active","inactive"}
              and all(type(usage[n]) is int and usage[n] == (50 if partition[n] == "active" else 0) for n in names),
              "Inactive vector banks must remain unused; active image/language must optimize50 times")
        _need(row["inactive_initial_sha256"] == row["inactive_final_sha256"] == first["inactive_initial_sha256"]
              and row["weights_sha256"] == first["weights_sha256"]
              and row["initial_weights_sha256"] == first["initial_weights_sha256"] != row["weights_sha256"]
              and row["all_rank_weights"] == [row["weights_sha256"]]*4
              and row["all_rank_initial_weights"] == [row["initial_weights_sha256"]]*4,
              "ImageNet all-rank weights diverged or inactive banks changed")
        _need(row["measured_updates"] == 40 and type(row["warm_wall_seconds"]) in (float,int)
              and math.isfinite(row["warm_wall_seconds"]) and row["warm_wall_seconds"] > 0,
              "Actual instrumented warm40-update timing is required")
    _need(completion["inactive_weights_sha256"] == first["inactive_final_sha256"],
          "Strict checkpoint inactive weights differ from optimized producers")
    _need(capacity.get("schema") == "imagenet-native-capacity-v1" and capacity.get("status") == "PASS"
          and capacity.get("job_id") == job and capacity.get("world_size") == 4
          and capacity.get("config_sha256") == request["smoke"]["resolved"]["sha256"]
          and capacity.get("source_manifest_sha256") == request["source_manifest_sha256"]
          and capacity.get("data_manifest_sha256") == request["data_manifest_sha256"]
          and capacity.get("data_context_sha256") == completion["data_context_sha256"]
          and capacity.get("rank_batch_size") == 8 and capacity.get("global_batch_size") == 32
          and capacity.get("frames") == capacity.get("cameras") == 1 and capacity.get("patches") == 100
          and capacity.get("parameters") == 314227000 and capacity.get("executed_parameters") == 159006592
          and capacity.get("expert_parameters") == COUNTS and capacity.get("optimizer_updates") == 0
          and capacity.get("compiled_backward_fp32finite") is True
          and capacity.get("bf16_forward") is True and capacity.get("fp32_weights_gradients") is True
          and capacity.get("inactive_gradients_absent") is True
          and capacity.get("attention_backend") == "flash4" and capacity.get("token_execution") == "packed_valid"
          and capacity.get("compile_mode") == "default" and capacity.get("compile_dynamic") is True,
          "Actual ImageNet maximum T1/C1/rank8 compiled backward proof is required")
    _need(capacity["optimizer_memory_reserve_bytes"] > 0
          and capacity["peak_reserved_bytes"] < capacity["device_memory_bytes"]
          and len(capacity["ranks"]) == 4
          and len({canonical_uuid(r["gpu_uuid"]) for r in capacity["ranks"]}) == 4,
          "Actual optimizer memory reserve and four capacity ranks are required")
    for rank,record in enumerate(capacity["ranks"]):
        _need(record.get("rank") == rank and record.get("gpu_uuid") == rows[rank]["gpu_uuid"]
              and "H100" in record.get("gpu_name","") and record.get("gpu_capability",[0])[0] >= 9,
              "Actual capacity GPU/rank binding differs from optimization")
        for key in ("schema","status","job_id","world_size","rank_batch_size","global_batch_size","frames","cameras",
                    "patches","parameters","executed_parameters","expert_parameters","data_context_sha256",
                    "source_manifest_sha256","data_manifest_sha256","config_sha256","optimizer_updates",
                    "compiled_backward_fp32finite","bf16_forward","fp32_weights_gradients","inactive_gradients_absent",
                    "attention_backend","token_execution","compile_mode","compile_dynamic"):
            _need(record.get(key) == capacity[key], "Missing per-rank actual capacity proof: "+key)
        _need(record.get("optimizer_memory_reserve_bytes",0) > 0
              and 0 < record.get("peak_reserved_bytes",0) < record.get("device_memory_bytes",0)
              and type(record.get("loss")) in (int,float) and math.isfinite(record["loss"]),
              "Actual rank capacity reserve/loss evidence is invalid")
    _need(completion["warm_wall_seconds"] == max(row["warm_wall_seconds"] for row in rows)
          and completion["warm_wall_seconds_by_rank"] == [row["warm_wall_seconds"] for row in rows],
          "Published timing must equal actual instrumented rank measurements")
    verify_dataset(request, allocated=False)
    return completion, preflight


def _small_text(path):
    path=Path(path)
    _need(path.is_file() and not path.is_symlink() and path.stat().st_size <= 4 << 20,
          "Unsafe or oversized proof metadata")
    return path.read_text()


def _small_json(path):
    import json
    def unique(pairs):
        result={}
        for key,value in pairs:
            _need(key not in result, "Duplicate proof JSON key")
            result[key]=value
        return result
    return json.loads(_small_text(path),object_pairs_hook=unique,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite proof JSON")))


def _bounded_bound(binding):
    """Read only small JSON evidence; never import ML or touch model/data payloads."""
    import json
    import re
    from scripts.train.joint.contract import sha
    path=Path(binding["path"])
    _need(path.is_absolute() and ".." not in path.parts and path.is_file() and not path.is_symlink()
          and path.stat().st_size <= 4 << 20 and re.fullmatch(r"[0-9a-f]{64}",binding.get("sha256", ""))
          and sha(path) == binding["sha256"], "Unbound or oversized metadata")
    def unique(pairs):
        value={}
        for key,item in pairs:
            _need(key not in value, "Duplicate JSON metadata key")
            value[key]=item
        return value
    return _small_json(path)


def _manifest_entries(path, maximum_members=128):
    import re
    path=Path(path)
    _need(path.is_file() and not path.is_symlink() and path.stat().st_size <= 4 << 20, "Unsafe manifest metadata")
    result={}
    for line in path.read_text().splitlines():
        parts=line.split("  ",1)
        _need(len(parts)==2, "Noncanonical file manifest")
        digest,name=parts
        _need(re.fullmatch(r"[0-9a-f]{64}",digest) and name not in result and name
              and not Path(name).is_absolute() and ".." not in Path(name).parts, "Invalid/duplicate manifest member")
        result[name]=digest
        _need(len(result) <= maximum_members, "Too many manifest metadata members")
    _need(bool(result), "Empty source/evidence manifest")
    return result


def _cpu_version(q):
    prefix="/coc/flash7/paphiwetsa3/tasks/elmo-qknorm-swiglu-rmsnorm-imagenet-20261026-v1/imagenet-cpu-"
    _need(q["root"] in (prefix+"v1",prefix+"v2",prefix+"v3",prefix+"v4",prefix+"v5"), "Unreviewed raw CPU producer namespace")
    return q["root"][len(prefix):]


def _cpu_native(raw, q, job, state, request_sha, *, observer=False, parent=None):
    """Pure reader for the exact published raw/Qwen producer's native contract."""
    import re
    pairs = re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9:/_-]*)=(\S+)',raw)
    _need(len(pairs) == len(dict(pairs)), "Duplicate raw CPU native fields")
    f = dict(pairs); r = q["capture_resources"] if observer else q["resources"]
    suffix = _cpu_version(q)
    name = "elmo-native-imagenet-cpu-"+suffix if observer else "elmo-imagenet-cpu-preparation-"+suffix
    cpus = str(r["cpus"]); mem = str(r["memory_mib"]//1024)+"G"
    comment = "imagenet-cpu-observer:"+request_sha+":"+str(parent) if observer else "imagenet-cpu:"+request_sha
    _need(f.get("JobId") == str(job) and f.get("JobName") == name
          and f.get("UserId") == "paphiwetsa3("+str(q["uid"])+")"
          and f.get("Account") == q["account"] and f.get("Partition") == q["partition"]
          and f.get("QOS") == q["qos"] and f.get("ReqNodeList") == q["node"]
          and f.get("NumNodes") in (("1","1-1") if state == "PENDING" else ("1",))
          and f.get("NumTasks") == "1" and f.get("NumCPUs") == cpus
          and f.get("MinMemoryNode") in (mem,str(r["memory_mib"]),str(r["memory_mib"])+"M")
          and f.get("TimeLimit") == r["time_limit"] and f.get("Requeue") == "0"
          and f.get("Restarts") == "0" and f.get("JobState") == state
          and f.get("Command") == str(Path(q["root"])/("capture.sbatch" if observer else "producer.sbatch"))
          and f.get("WorkDir") == q["root"] and f.get("Comment") == comment
          and f.get("SubmitTime") not in (None,"Unknown","N/A"), "Changed raw CPU native identity/resources/request")
    for key in ("ReqTRES","AllocTRES"):
        value=f.get(key,"")
        _need("gpu" not in value.lower(), "GPU TRES forbidden in raw CPU producer")
        if key == "ReqTRES" or state != "PENDING":
            parts=dict(v.split("=",1) for v in value.split(",") if "=" in v)
            _need(len(parts)==len(value.split(",")) and parts.get("cpu")==cpus and parts.get("node")=="1",
                  "Missing exact raw CPU native TRES")
            if _cpu_version(q) in ("v3","v4","v5"):
                from scripts.train.joint.imagenet_memory_policy import POLICY
                _need(q.get("memory_policy") == POLICY and (parts.get("mem") == mem if key == "ReqTRES" else
                      "mem" not in parts or parts["mem"] == mem) and f.get("Mem","0") == "0",
                      "Unreviewed Skynet CR_CPU memory allocation semantics")
            else:
                _need(parts.get("mem") == mem, "Missing actual reserved memory for historical CPU producer")
    for key,value in f.items():
        if key.lower().startswith(("gres","job_gres","tresper")):
            _need(value.lower() in ("","(null)","n/a","none","0") or
                  key == "TresPerTask" and value in ("cpu:"+cpus,"cpu="+cpus), "GPU/GRES raw producer field forbidden")
    if state == "PENDING":
        _need(f.get("Reason") == ("Dependency" if observer else "JobHeldUser"), "Wrong raw CPU acceptance state")
        _need(bool(re.fullmatch(r"afterany:"+re.escape(str(parent))+r"(?:\(unfulfilled\))?",f.get("Dependency","")))
              if observer else f.get("Dependency") == "(null)", "Wrong original CPU afterany binding")
    else:
        _need(f.get("NodeList") == q["node"] and f.get("StartTime") not in (None,"Unknown","N/A"),
              "Raw CPU producer lacks actual start/node")
    if state == "COMPLETED":
        _need(f.get("ExitCode") == "0:0" and f.get("EndTime") not in (None,"Unknown","N/A")
              and f["EndTime"] >= f["StartTime"], "Raw CPU native completion is not successful")
    return f


def _cpu_argv(q, request_sha, files_sha, *, parent=None):
    observer=parent is not None; r=q["capture_resources"] if observer else q["resources"]
    root=Path(q["root"]); suffix=_cpu_version(q)
    result=[q["native"]["sbatch"],"--parsable","--no-requeue","--account="+q["account"],
        "--partition="+q["partition"],"--qos="+q["qos"],
        "--job-name="+("elmo-native-imagenet-cpu-"+suffix if observer else "elmo-imagenet-cpu-preparation-"+suffix),
        "--nodelist="+q["node"],"--nodes=1","--ntasks=1","--cpus-per-task="+str(r["cpus"]),
        "--mem="+str(r["memory_mib"])+"M","--gres=none","--time="+r["time_limit"],"--export=NONE","--chdir="+str(root),
        "--comment="+("imagenet-cpu-observer:"+request_sha+":"+parent if observer else "imagenet-cpu:"+request_sha),
        "--output="+str(root/"logs/%x-%j.out"),"--error="+str(root/"logs/%x-%j.err")]
    result += ["--dependency=afterany:"+parent] if observer else ["--hold"]
    return result+[str(root/("capture.sbatch" if observer else "producer.sbatch")),str(root),request_sha,files_sha]


def verify_dataset(request, *, allocated=False):
    """Read native producer envelopes; large payload/adapter checks are allocated-only.

    Gateway callers get a bounded stdlib proof interpretation. The actual driver
    independently checks every payload and adapter context after allocation.
    """
    from scripts.train.joint.contract import bound, sha
    import json
    import re
    contract = request["image_generation_contract"]
    _need(contract.get("schema") == "native-imagenet-dataset-v1", "Unknown ImageNet data contract")
    packet = contract["raw_preparation"]
    original = Path(packet["original_root"]); relocated = Path(packet["evidence_root"])
    _need(original.is_absolute() and relocated.is_absolute() and ".." not in original.parts
          and ".." not in relocated.parts, "CPU evidence relocation requires explicit absolute roots")
    def reopen(binding, *, parsed=True):
        old = Path(binding["path"])
        _need(old.is_relative_to(original) and ".." not in old.parts, "CPU evidence escaped its immutable root")
        actual = relocated/old.relative_to(original)
        _need(actual.is_file() and not actual.is_symlink() and actual.resolve().is_relative_to(relocated.resolve())
              and actual.stat().st_size <= 4 << 20
              and sha(actual) == binding["sha256"], "Changed/oversized imported CPU metadata: "+str(old))
        return _small_json(actual) if parsed else _small_text(actual)
    q = _bounded_bound(packet["request"]); completion = _bounded_bound(packet["completion"])
    qsha = packet["request"]["sha256"]
    _need(q.get("schema") == "imagenet-allocated-cpu-request-v1" and q.get("state") == "BOUND_CPU_ONLY"
          and q["root"] == str(original) and _cpu_version(q) in ("v1","v2","v3","v4","v5")
          and q["uid"] == 3346154 and q["node"] in ("sky1","sky2")
          and q["account"] in ("rl2-lab","hoffman-lab") and q["partition"] == q["account"] and q["qos"] == "short"
          and all(q["native"][n] == "/opt/slurm/Ubuntu-24.04/current/bin/"+n for n in ("sbatch","scontrol","squeue"))
          and q["train_root"] == "/coc/dataset/ImageNet/train"
          and q["val_root"] == "/coc/dataset/ImageNet/val"
          and q["expected_counts"] == dict(classes=1000,train=1281167,val=50000)
          and q["resources"] == dict(cpus=4,gpus=0,memory_mib=24576,time_limit="04:00:00")
          and q["capture_resources"] == dict(cpus=2,gpus=0,memory_mib=1024,time_limit="00:10:00")
          and q["runtime"]["pillow_version"] == "13.0.0"
          and q["source"]["revision"] == "e9bff72c82618e74175886d16983f3cf90780fb0"
          and q["training_authorized_by_this_packet"] is False, "Unbound exact CPU data producer")
    _need(completion.get("schema") == "imagenet-raw-cpu-completion-v1" and completion.get("status") == "PASS"
          and completion["request_sha256"] == qsha and completion["resources"] == q["resources"]
          and completion["node"] == q["node"] and completion["uid"] == q["uid"]
          and completion["counts"] == dict(train=1281167,val=50000)
          and completion["training_admitted"] is False, "Actual allocated data completion is missing")
    files_path = relocated/"FILES.sha256"
    files = _manifest_entries(files_path)
    _need(sha(files_path) == completion["files_manifest_sha256"] == packet["files_manifest_sha256"],
          "Imported CPU wrapper manifest differs from the reviewed request binding")
    _need({"request.json","guard.py","producer.sbatch","capture.sbatch","run_allocated.py","observe.py"} <= set(files),
          "Incomplete actual CPU producer code closure")
    large_inputs = {str(Path(q["source"]["archive"]["path"]).relative_to(original)):q["source"]["archive"]["sha256"],
                    str(Path(q["pillow_wheel"]["path"]).relative_to(original)):q["pillow_wheel"]["sha256"]}
    for name,digest in files.items():
        _need(not Path(name).is_absolute() and ".." not in Path(name).parts, "Unsafe CPU producer manifest path")
        if not allocated and name in large_inputs:
            _need(digest == large_inputs[name], "CPU large-input binding differs")
        else:
            _need((relocated/name).is_file() and not (relocated/name).is_symlink()
                  and (allocated or (relocated/name).stat().st_size <= 4 << 20)
                  and sha(relocated/name) == digest, "Changed CPU producer wrapper code/request")
    _need(files["request.json"] == qsha, "CPU wrapper request differs from native accepted request")
    job = str(completion["job_id"])
    _need(re.fullmatch(r"[1-9][0-9]*",job) is not None, "Missing actual CPU parent job")
    _need(reopen(completion["host_exit"],parsed=False).strip() == "0", "Raw preparation host did not succeed")
    result = reopen(completion["producer_result"])
    _need(result.get("schema") == "imagenet-raw-cpu-producer-result-v1"
          and result.get("status") == "PASS_DATA_PREPARATION_ONLY" and result["job_id"] == job
          and result["request_sha256"] == qsha and result["files_manifest_sha256"] == completion["files_manifest_sha256"]
          and result["resources"] == q["resources"] and result["uid"] == q["uid"]
          and result["source_revision"] == completion["source_revision"] == q["source"]["revision"]
          and result["counts"] == completion["counts"] and result["classes"] == 1000
          and result["training_admitted"] is False, "Unbound raw/Qwen producer result")
    capture = reopen(completion["native_capture"])
    _need(capture.get("schema") == "imagenet-cpu-native-capture-v1"
          and capture["parent_job_id"] == job and capture["request_sha256"] == qsha
          and capture["files_manifest_sha256"] == completion["files_manifest_sha256"]
          and [r["kind"] for r in capture["queries"]] == ["job","steps"],
          "Actual original afterany capture provenance is missing")
    terminal = None
    for query in capture["queries"]:
        expected = [q["native"]["scontrol"],"show","job",job,"-o","-dd"] if query["kind"] == "job" else [q["native"]["scontrol"],"show","step","-o"]
        _need(query["argv"] == expected and query["exit_code"] == 0, "CPU native query failed or changed")
        reopen(query["stderr"],parsed=False)
        raw = reopen(query["stdout"],parsed=False)
        if query["kind"] == "job":
            pairs = re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9:/_-]*)=(\S+)',raw)
            _need(len(pairs) == len(dict(pairs)), "Duplicate native parent fields")
            terminal = dict(pairs)
        else:
            for line in raw.splitlines():
                if not line.strip() or line.strip() in ("No job steps in the system","No job steps in the system."):continue
                pairs=re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9:/_-]*)=(\S+)',line)
                _need(len(pairs) == len(dict(pairs)), "Duplicate native step fields")
                fields = dict(pairs)
                step = fields.get("StepId",fields.get("StepID"))
                _need(step is not None and not (step == job or step.startswith(job+".") or step.startswith(job+"_")),
                      "Owned CPU producer steps remain or full native inventory is unreadable")
    _need(terminal is not None, "Missing actual raw CPU native terminal record")
    terminal = _cpu_native(reopen(completion["native_terminal"],parsed=False),q,job,"COMPLETED",qsha)
    _need(terminal["JobName"] == completion["job_name"] and terminal["StartTime"] == completion["native_start"]
          and terminal["EndTime"] == completion["native_end"], "Raw CPU terminal interval/name differs")
    _need(completion["native_terminal"] == capture["queries"][0]["stdout"], "Wrong native terminal binding")
    canonical={"raw_inventory":"raw-inventory.json","features":"qwen-features.json","class_manifest":"classes.json",
               "manifest":"imagefolder-manifest.json","join":"join-receipt.json","anchors":"validation-anchors.json",
               "runtime":"runtime.json","qwen_fresh_inputs":"qwen-fresh-inputs.json","source_stage":"source-stage.json",
               "native_running":"native-running.txt"}
    _need(completion["producer_result"]["path"] == str(original/"output/producer-result.json")
          and completion["host_exit"]["path"] == str(original/("host-exit-"+job+".txt"))
          and completion["native_capture"]["path"] == str(original/("native-"+job+".capture.json")),
          "Noncanonical original CPU producer outputs")
    for key,name in canonical.items():
        _need(result[key]["path"] == str(original/"output"/name), "Noncanonical original CPU output: "+key)
        if key not in ("raw_inventory","features","manifest"):
            reopen(result[key],parsed=False)
    if _cpu_version(q) in ("v4","v5"):
        # This pure stdlib reader replays the reviewed exact offline wheel
        # closure.  It does not import a producer-side module or reopen an
        # image, archive, wheel or extracted native library on the gateway.
        from scripts.train.joint.imagenet_wheel_overlay import validate_runtime
        policy=dict(python="/coc/flash7/paphiwetsa3/projects/EgoVerse-refactor/.venv/bin/python",
                   python_real="/nethome/paphiwetsa3/flash/uv-python/cpython-3.11.14-linux-x86_64-gnu/bin/python3.11",
                   python_version="3.11.14",pillow_version="13.0.0")
        if _cpu_version(q) == "v5":
            policy["python_physical"]="/coc/flash7/paphiwetsa3/uv-python/cpython-3.11.14-linux-x86_64-gnu/bin/python3.11"
        _need(files.get("wheel_overlay.py") == "8aa731d672b1a034cdfd17e9356cc409a36e5ee68fe8c22e479a6eb1fa36f631"
              and q["runtime"] == policy
              and q["pillow_wheel"] == dict(path=str(original/"inputs/pillow-13.0.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"),
                   sha256="a625df9a2a5597fa5e5266e01aefc82604bb85d291e3547981e863560d75da6c"),
              "Source CPU exact reviewed decoder/interpreter contract changed")
        actual_runtime=reopen(result["runtime"])
        if _cpu_version(q) == "v5":
            _need(actual_runtime.get("configured_executable") == policy["python"]
                  and actual_runtime.get("configured_target") == policy["python_real"]
                  and actual_runtime.get("physical_executable_expected") == policy["python_physical"],
                  "Source CPU-v5 configured/physical interpreter provenance changed")
        validate_runtime(actual_runtime,q["pillow_wheel"],original/"output/pillow-overlay",
                         policy.get("python_physical",policy["python_real"]),profile="cp311")
    _need(not Path(str(relocated/"output/raw-inventory.json")+".failure.json").exists(), "Raw producer recorded a failure")
    claim = relocated/"claims"/qsha
    observer = str(capture["observer_job_id"])
    _need(observer.isdecimal() and observer != job and _small_text(claim/"PARENT_JOB_ID").strip() == job
          and _small_text(claim/"CAPTURE_JOB_ID").strip() == observer,
          "Missing original parent/observer acceptance ledger")
    prepared = _small_json(claim/"PREPARED.json")
    _need(prepared["request_sha256"] == qsha and prepared["files_manifest_sha256"] == completion["files_manifest_sha256"]
          and prepared["parent_argv"] == _cpu_argv(q,qsha,completion["files_manifest_sha256"]), "Changed raw CPU PREPARED claim")
    submit_times={}
    for name,identity in (("parent",job),("capture",observer)):
        _need(_small_text(claim/(name+"-sbatch.stdout")).strip() == identity
              and _small_text(claim/(name+"-sbatch.exit")).strip() == "0"
              and _small_json(claim/(name+"-sbatch.argv.json")) ==
                  _cpu_argv(q,qsha,completion["files_manifest_sha256"],parent=job if name == "capture" else None),
              "Unknown or changed original CPU submission")
    for name,identity,is_observer in (("HELD_NATIVE.txt",job,False),("CAPTURE_PENDING_NATIVE.txt",observer,True),
                 ("RELEASE_HELD_NATIVE.txt",job,False),("RELEASE_CAPTURE_PENDING_NATIVE.txt",observer,True)):
        f=_cpu_native(_small_text(claim/name),q,identity,"PENDING",qsha,observer=is_observer,parent=job)
        _need(identity not in submit_times or submit_times[identity] == f["SubmitTime"], "Original CPU acceptance identity changed")
        submit_times[identity]=f["SubmitTime"]
    release = _small_json(claim/"RELEASE_ATTEMPT.json")
    _need(release["parent"] == job and release["capture"] == observer and release["request_sha256"] == qsha,
          "Missing exact once-only producer release ledger")
    _need(_small_text(relocated/"request-claims"/q["request_id"]/"REQUEST_SHA256").strip() == qsha,
          "Missing original request consumption claim")
    for name in ("root-review.json","deployment-readback.json"):
        proof=_small_json(relocated/name)
        _need(proof.get("status") == "PASS" and proof.get("root") == str(original)
              and proof.get("request_sha256") == qsha and proof.get("files_manifest_sha256") == completion["files_manifest_sha256"],
              "Actual CPU deployment/review binding is missing")
    if _cpu_version(q) in ("v3","v4","v5"):
        from scripts.train.joint.imagenet_memory_policy import POLICY,check_observation,check_limit
        _need(q.get("memory_policy") == POLICY and completion.get("memory_semantics") == "requested_not_consumable_reserved"
              and completion.get("reserved_memory_mib") is None and "memory_policy.py" in files,
              "Actual Skynet data producer must disclose requested, unreserved RAM")
        for binding,identity,is_observer in ((result["memory_policy"],job,False),(capture["memory_policy"],observer,True)):
            _need(binding["path"] == str(original/("memory-policy-"+identity+".json"))
                  and binding == completion["observer_memory_policy" if is_observer else "memory_policy"],
                  "Changed canonical allocated memory proof binding")
            memory=reopen(binding)
            _need(memory["schema"] == "allocated-imagenet-memory-policy-v1" and memory["job_id"] == identity
                  and memory["request_sha256"] == qsha and memory["capture"] is is_observer,
                  "Actual source allocation memory proof identity differs")
            summary=check_observation(q,memory["observation"],capture=is_observer)
            _need(summary == memory["summary"], "Allocated RAM summary differs from actual observation")
            check_limit(memory["cgroup"],q["capture_resources" if is_observer else "resources"]["memory_mib"])
        check_observation(q,prepared["memory"])
    running=_cpu_native(reopen(result["native_running"],parsed=False),q,job,"RUNNING",qsha)
    _need(running["SubmitTime"] == terminal["SubmitTime"] == submit_times[job]
          and running["StartTime"] == terminal["StartTime"], "Original CPU running/terminal job differs")
    _need(capture["observer_running"]["path"] == str(original/("capture-running-"+observer+".txt")),
          "Noncanonical original observer running proof")
    observer_native=_cpu_native(reopen(capture["observer_running"],parsed=False),q,observer,"RUNNING",qsha,observer=True,parent=job)
    _need(observer_native["SubmitTime"] == submit_times[observer], "Original observer acceptance/running differs")
    for query in capture["queries"]:
        target = str(original/("native-"+job+(".txt" if query["kind"] == "job" else ".steps.txt")))
        _need(query["stdout"]["path"] == target and query["stderr"]["path"] == target+".stderr", "Noncanonical original native capture output")
    _need(result["raw_inventory"] == completion["raw_inventory"], "Wrong decoded raw output")
    source_manifest = reopen(q["source"]["manifest"],parsed=False)
    components = _manifest_entries(relocated/Path(q["source"]["manifest"]["path"]).relative_to(original),8192)
    # Dataset science is component-bound: later operational commits do not
    # silently invalidate or relabel a genuine completed1.33M-image decode.
    required = {"egomimic/rldb/imagenet_imagefolder.py","egomimic/rldb/imagenet_preparation.py",
                "egomimic/rldb/pixel_language.py","egomimic/rldb/pixel_proprio.py",
                "egomimic/scripts/embedding_process/qwen3_embedding.py"}
    _need(set(contract["preparation_components"]) == required, "Incomplete exact data-preparation code closure")
    for name,digest in contract["preparation_components"].items():
        current=Path(request["source"])/name
        _small_text(current)
        _need(components.get(name) == digest and sha(current) == digest,
              "Data preparation code changed after actual CPU producer: "+name)
    if not allocated:
        # The bound native CPU result proves which payloads were produced. Their
        # large hashes, schema projection and real adapter loading are repeated
        # by the actual allocated driver, never on the gateway.
        for target,key in (("manifest","manifest"),("class_language","class_manifest"),
                           ("language_features","features"),("validation_anchor","anchors")):
            _need(contract[target]["sha256"] == result[key]["sha256"], "Data envelope differs from CPU result")
        verify_cache_native(contract["cache_native"],contract["cache_readback"],request,allocated=False)
        return dict(train=1281167,val=50000),1000
    from egomimic.rldb.imagenet_preparation import validate_raw_inventory
    raw_path = relocated/Path(result["raw_inventory"]["path"]).relative_to(original)
    raw_report = validate_raw_inventory(raw_path,result["raw_inventory"]["sha256"])
    runtime = reopen(result["runtime"]); staged = reopen(result["source_stage"])
    _need(runtime["pillow_version"] == q["runtime"]["pillow_version"] == "13.0.0"
          and runtime["pillow_wheel"] == q["pillow_wheel"]
          and staged.get("status") == "PASS" and staged["revision"] == q["source"]["revision"]
          and staged["manifest"] == q["source"]["manifest"] and staged["files"] == q["source"]["entries"],
          "Allocated source/decoder runtime differs from the authentic raw producer")
    _need(raw_report["producer"]["job_id"] == job
          and raw_report["content_identity_sha256"] == result["content_identity_sha256"] == completion["content_identity_sha256"]
          and raw_report["physical_sources"] == dict(train=q["train_root"],val=q["val_root"])
          and raw_report["decoder"] == result["decoder"], "Raw content/actual CPU source/decoder changed")
    manifest = bound(contract["manifest"])
    classes = bound(contract["class_language"]); features = bound(contract["language_features"])
    anchors = bound(contract["validation_anchor"]); join = reopen(result["join"])
    fresh_qwen = reopen(result["qwen_fresh_inputs"])
    _need(fresh_qwen.get("schema") == "allocated-qwen-fresh-inputs-v1" and fresh_qwen.get("job_id") == job
          and fresh_qwen.get("all_hashes_computed_inside_allocation") is True
          and fresh_qwen.get("weights") == features["encoder"].get("weights")
          and len(fresh_qwen["weights"]) == 1
          and fresh_qwen["weights"][0]["sha256"] == features["encoder"]["weights_sha256"],
          "Class vectors lack genuine actual allocated Qwen weight provenance")

    _need(contract["manifest"]["sha256"] == result["manifest"]["sha256"]
          and contract["class_language"]["sha256"] == result["class_manifest"]["sha256"]
          and contract["language_features"]["sha256"] == result["features"]["sha256"]
          and contract["validation_anchor"]["sha256"] == result["anchors"]["sha256"]
          and join["raw_report"] == result["raw_inventory"] and join["manifest"] == result["manifest"]
          and join["raw_content_identity_sha256"] == raw_report["content_identity_sha256"]
          and join["images_rehashed_or_decoded"] == 0 and join["data_admitted"] is False,
          "Raw/Class/Qwen/source-anchor join differs")
    from egomimic.rldb.imagenet_imagefolder import load_class_language, _validate_manifest, _validate_anchors
    class_value,_ = load_class_language(contract["class_language"]["path"],contract["class_language"]["sha256"],
                                       contract["language_features"]["path"],contract["language_features"]["sha256"])
    _validate_manifest(manifest,class_value,contract["class_language"]["sha256"],contract["language_features"]["sha256"],1000)
    for split in ("train","val"):
        source_rows = raw_report["portable_content"]["splits"][split]["images"]
        target_rows = manifest["splits"][split]["images"]
        _need(len(source_rows) == len(target_rows) and all(
            target == {key:raw[key] for key in ("path","wnid","bytes","sha256")}
            for raw,target in zip(source_rows,target_rows)), "Actual source raw-to-language manifest projection differs")
    _validate_anchors(anchors,manifest,contract["manifest"]["sha256"],1000)
    _need(classes["instruction_catalog_sha256"] == q["catalog"]["sha256"]
          and features["source_catalog_sha256"] == q["catalog"]["sha256"]
          and {s:manifest["splits"][s]["count"] for s in ("train","val")} == dict(train=1281167,val=50000),
          "Wrong actual ImageNet official source/class-name identity")
    # Final destination proof is separately produced after transfer. Its native
    # envelope is intentionally explicit; source decode never certifies a copy.
    cache = _bounded_bound(contract["cache_readback"])
    _need(cache.get("schema") == "imagenet-cache-readback-v2"
          and cache.get("status") == "EXTERNAL_COMPLETION_REQUIRED" and cache.get("data_admitted") is False
          and cache.get("data_root") == request["data_cache"]
          and cache.get("raw_content_identity_sha256") == raw_report["content_identity_sha256"]
          and cache.get("source_manifest_sha256") == result["raw_inventory"]["sha256"]
          and cache.get("counts") == dict(train=1281167,val=50000) and cache.get("classes") == 1000
          and cache.get("all_image_bytes_hashed") is True and cache.get("all_images_decoded") is True
          and cache.get("decoder") == raw_report["decoder"]
          and cache["storage"].get("file_fsync") is True and cache["storage"].get("directory_fsync") is True,
          "A source decode is insufficient: exact destination byte/decode/fsync proof is required")
    _need(all(cache.get(key) == "EXTERNAL_REQUIRED" for key in
              ("host_completion","native_completion","afterany_completion")),
          "Payload must leave actual host/native/afterany completion independently bound")
    destination = validate_raw_inventory(cache["destination_inventory"]["path"],cache["destination_inventory"]["sha256"])
    _need(destination["content_identity_sha256"] == raw_report["content_identity_sha256"]
          and destination["portable_content"] == raw_report["portable_content"]
          and destination["producer"]["job_id"] == str(cache["job_id"])
          and destination["physical_sources"] == dict(train=str(Path(request["data_cache"])/"train"),
                                                       val=str(Path(request["data_cache"])/"val")),
          "Actual complete destination inventory differs from the genuine source byte/decode identity")
    # A fresh allocated cache producer must itself be admitted by its exact
    # reviewed wrapper. The payload deliberately cannot certify completion.
    done = verify_cache_native(contract["cache_native"], contract["cache_readback"], request, allocated=True)
    _need(str(done["job_id"]) == str(cache["job_id"]) and done["request_sha256"] == cache["request_sha256"],
          "Cache payload and actual completed producer differ")
    return dict(train=1281167,val=50000),1000


def _cache_argv(q, digest, files_sha, *, parent=None):
    _need(q.get("schema") == "imagenet-target-cpu-request-v2" and q.get("wrapper_protocol") ==
          "REQUIRES_REVIEWED_LAMBDA_CPU_WRAPPER_V2", "Exact target-v2 submission protocol required")
    observer=parent is not None; r=q["capture_resources"] if observer else q["resources"]
    root=Path(q["root"])
    args=[q["native"]["sbatch"],"--parsable","--no-requeue","--account="+q["account"],
        "--partition="+q["partition"],"--qos="+q["qos"],
        "--job-name="+q["capture_name" if observer else "job_name"],"--nodelist="+q["node"],
        "--nodes=1","--ntasks=1","--cpus-per-task="+str(r["cpus"]),"--mem="+str(r["memory_mib"])+"M",
        "--gres=none","--time="+r["time_limit"],"--export=NONE","--chdir="+str(root),
        "--comment="+("imagenet-target-observer:"+digest+":"+parent if observer else "imagenet-target-cpu:"+digest),
        "--output="+str(root/"logs/%x-%j.out"),"--error="+str(root/"logs/%x-%j.err")]
    args += ["--dependency=afterany:"+parent] if observer else ["--hold"]
    return args+[str(root/("capture.sbatch" if observer else "producer.sbatch")),str(root),digest,files_sha]


def _cache_phase_chain(q, done, report, training_request, proofs, reopen, native, running, terminal):
    """Replay target-v2 metadata only; never execute producer/service inputs.

    The enclosing dataset reader already verifies the genuine source-CPU chain.
    This reader ties the exact same source identity to the host all-byte commit,
    isolated container native queries, private cp312 decoder activation and the
    subsequent maintained Torch import, without opening corpus/wheel/ELF bytes.
    """
    import hashlib
    import json
    import re
    from datetime import datetime, timezone
    from scripts.train.joint.contract import sha
    from scripts.train.joint.imagenet_wheel_overlay import validate_installation
    original=Path(q["root"]); out=original/"output"; job=str(done["job_id"]); digest=done["request_sha256"]
    files_sha=done["files_manifest_sha256"]
    def canonical(item,name,*,parsed=True):
        _need(item["path"]==str(original/name), "Noncanonical target-v2 phase evidence: "+name)
        return reopen(item,parsed=parsed)
    def content_sha(value):
        return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    def time(value,*,aware=False):
        _need(isinstance(value,str) and value not in ("Unknown","None","N/A",""), "Missing actual native/phase timestamp")
        result=datetime.fromisoformat(value.replace("Z","+00:00"))
        if aware:
            _need(result.tzinfo is not None and result.utcoffset()==timezone.utc.utcoffset(result), "Phase timestamp must be UTC-aware")
        return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)
    start,end=time(terminal["StartTime"]),time(terminal["EndTime"])
    _need(time(terminal["SubmitTime"])<=start<end, "Invalid actual target native interval")
    def interval(item,name):
        actual=native(canonical(item,name,parsed=False),job,"RUNNING")
        _need(all(actual[k]==running[k] for k in ("JobId","SubmitTime","StartTime","Command","Comment")),
              "Target phase escaped the actual allocated interval")
        return actual
    def namespaces(value):
        _need(isinstance(value,dict) and set(value)=={"mnt","pid","user","uts","net"}
              and all(isinstance(v,str) and re.fullmatch(re.escape(k)+r":\[[1-9][0-9]*\]",v) for k,v in value.items()),
              "Incomplete actual container namespace identity")
        return value
    _need(q.get("training_admitted") is False and q["source"]["revision"]=="e9bff72c82618e74175886d16983f3cf90780fb0"
          and q["source"]["root"]==str(out/"source-cache/source")
          and q["source"]["archive"]["path"]==str(original/"inputs/source.tar.gz")
          and q["source"]["manifest"]["path"]==str(original/"inputs/source.files.sha256"), "Target source authority changed")
    source_publication=canonical(q["source"]["publication"],"inputs/publication-receipt.json")
    _need(source_publication.get("status")=="PASS" and source_publication.get("upstream_verified") is True
          and source_publication.get("revision")==q["source"]["revision"]
          and source_publication.get("archive_sha256")==q["source"]["archive"]["sha256"]
          and source_publication.get("manifest_sha256")==q["source"]["manifest"]["sha256"],
          "Actual source-I publication binding differs from staged bytes")
    _need(q.get("native_client")==dict(schema="native-elmo-client-closure-input-v1",scontrol="/usr/local/bin/scontrol",
          squeue="/usr/local/bin/squeue",config="/etc/slurm/slurm.conf",munge_socket="/run/munge/munge.socket.2"),
          "Unreviewed target native authentication sources")
    # Source identity is the same producer chain verified by verify_dataset,
    # irrespective of its explicit physical metadata import/mirror directory.
    raw=training_request["image_generation_contract"]["raw_preparation"]
    source_q=_bounded_bound(raw["request"]); source_done=_bounded_bound(raw["completion"])
    _need(_cpu_version(source_q)=="v5" and source_done.get("schema")=="imagenet-raw-cpu-completion-v1"
          and source_done.get("status")=="PASS" and source_done["request_sha256"]==raw["request"]["sha256"]
          and q["source_completion"]["sha256"]==raw["completion"]["sha256"]
          and q["source_raw_inventory"]["sha256"]==source_done["raw_inventory"]["sha256"],
          "Target source differs from the genuine source-CPU-v5 chain")
    _need(q["source_completion"]["path"]==str(Path(q["source_evidence_root"])/"completion.json")
          and q["source_raw_inventory"]["path"]==str(Path(q["source_evidence_root"])/"output/raw-inventory.json"),
          "Noncanonical imported source-CPU evidence")
    host_item=report["host_preparation"]; auth_item=report["container_authentication"]
    _need(done["host_preparation"]==host_item and done["container_authentication"]==auth_item,
          "Completion does not bind the real two-phase target commit")
    host=canonical(host_item,"output/host-preparation.json"); auth=canonical(auth_item,"output/container-authentication.json")
    _need(host.get("schema")=="imagenet-target-host-preparation-v2" and host.get("status")=="ALL_BYTES_HASHED_BEFORE_ML_IMPORT"
          and host.get("job_id")==job and host.get("request_sha256")==digest and host.get("data_root")==q["data_root"]
          and host.get("source")==q["source"] and host.get("source_completion")==q["source_completion"]
          and host.get("source_raw_inventory")==q["source_raw_inventory"] and host.get("no_ml_modules_loaded") is True
          and host.get("data_admitted") is False and host.get("counts")==q["counts"] and host.get("classes")==1000,
          "Actual all-byte host preparation before ML is missing")
    interval(host["native_running"],"output/native-running.txt")
    for key in ("source_revalidation","assembly","storage"):
        _need(report[key]==host[key], "Cache report changed the committed host proof")
    plan=canonical(q["plan"],"inputs/plan/plan.json"); selection=canonical(q["selections"],"inputs/selections.json")
    identity=host["raw_content_identity_sha256"]
    _need(re.fullmatch("[0-9a-f]{64}",identity) and plan.get("schema")=="imagenet-mac-stream-transfer-plan-v1"
          and plan["content_identity_sha256"]==source_done["content_identity_sha256"]==identity==report["raw_content_identity_sha256"]
          and plan["raw_inventory_sha256"]==host["source_manifest_sha256"]==report["source_manifest_sha256"]
          ==q["source_raw_inventory"]["sha256"] and plan["source_completion_sha256"]==q["source_completion"]["sha256"]
          and plan["total_files"]==1331167 and type(plan["total_raw_bytes"]) is int and plan["total_raw_bytes"]>0
          and selection.get("schema")=="imagenet-transfer-selection-v1" and selection["plan_sha256"]==q["plan"]["sha256"],
          "Host byte assembly does not bind the actual complete source inventory")
    version=q["verification_id"]
    _need(re.fullmatch("verify-[A-Za-z0-9-]{16,80}",version) and q["data_root"]==str(original.parent/"imagenet-staging-v1"/identity/"verified"/version/"data"),
          "Target corpus is not the fresh immutable content/verification namespace")
    assembly=canonical(host["assembly"],"output/assembly.json")
    _need(assembly.get("schema")=="imagenet-target-assembly-v1" and assembly.get("status")=="ALL_BYTES_HASHED_BEFORE_ML_IMPORT"
          and assembly.get("data_root")==q["data_root"] and assembly.get("images")==1331167
          and assembly.get("raw_bytes")==plan["total_raw_bytes"] and assembly.get("plan_sha256")==q["plan"]["sha256"]
          and assembly.get("selections_sha256")==q["selections"]["sha256"] and assembly.get("shared_inode_assembly") is True
          and assembly.get("data_admitted") is False, "Incomplete actual byte-hashed target assembly")
    selected={row["shard_id"]:row for row in selection["shards"]}; shards={row["shard_id"]:row for row in plan["shards"]}
    attempts={row["shard_id"]:row for row in assembly["selected_attempts"]}
    _need(len(selected)==len(selection["shards"])==len(shards)==len(plan["shards"])
          ==len(attempts)==len(assembly["selected_attempts"]) and set(selected)==set(shards)==set(attempts)
          and sum(row["files"] for row in shards.values())==1331167
          and sum(row["bytes"] for row in shards.values())==plan["total_raw_bytes"], "Selected target shard closure differs")
    for key,row in attempts.items():
        _need(re.fullmatch("attempt-[A-Za-z0-9-]{16,80}",row["attempt"]) and row["attempt"]==selected[key]["attempt"]
              and all(re.fullmatch("[0-9a-f]{64}",row[k]) for k in ("intent_sha256","exit_sha256")), "Unknown selected shard completion")
    chain=canonical(host["source_revalidation"],"output/source-chain-revalidation/receipt.json")
    reader={"guard.py":"44f02f1364c7e474440e38d2b494ad5c0b90d203407e3617874cb212755f9f16",
        "submit_once.py":"eaae1bd0297f80948e36eaef0196da23dce3db6c4a97af83726047a355d87d5a",
        "finalize.py":"210ff464bb42bf71c32205865e8f2e5072394bfb2173e8544cca111ec8bbad60",
        "memory_policy.py":"a9ef364c6714e35be0a1937fef46cab2325b4f3e221ddfd2186aa133983518d9",
        "wheel_overlay.py":"8aa731d672b1a034cdfd17e9356cc409a36e5ee68fe8c22e479a6eb1fa36f631"}
    _need(chain.get("schema")=="imagenet-source-completion-revalidation-v1" and chain.get("status")=="PASS"
          and chain.get("completion_sha256")==q["source_completion"]["sha256"] and chain.get("source_request_sha256")==raw["request"]["sha256"]
          and chain.get("source_files_sha256")==source_done["files_manifest_sha256"] and chain.get("reader_files")==reader
          and chain.get("source_evidence_root")==q["source_evidence_root"] and chain.get("image_corpus_copied") is False,
          "Actual source-v5 chain was not revalidated before target image work")
    _need(host["source_decoder"]==report["decoder"] and report["decoder"]["version"]=="13.0.0"
          and content_sha(report["decoder"])==source_done["decoder_binding_sha256"], "Destination decoder differs from the authentic source")
    _need(type(host["storage"]["free_bytes_before_assembly"]) is int and host["storage"]["free_bytes_before_assembly"]>=8*1024**3
          and host["storage"]["file_fsync"] is True and host["storage"]["directory_fsync"] is True, "Host actual storage/fsync commit missing")
    # Exact maintained transfer code is witnessed by the allocated producer.
    transfer=proofs["transfer-code-readback.json"]; transfer_root=original.parent/"imagenet-transfer-controller-v3"
    expected_payload="19f90890eecd51334c7247037f99f30ff01a119f93ee331b9a1b68979f8a10b6"
    transfer_files={"common.py":"b202a5bdff12c2ca3cd89062323e301876d111fc9cf32cdedcf087e7f423004c",
        "inspect_shard.sh":"6151c88ed2ecf070b26b16cd31968d73292ed91fe129450b3a20ccded34be615",
        "intake.sh":"2ac7f12e1c6f7a84f50f291e048e1a30796ada3e6625a9a2565f5dd8c2a00260",
        "plan.py":"73a6d811f56a8179651ec3f8c8c5e53419c17bb859153096a7cad13f0b035925",
        "prepare_destination.sh":"8ed7d41cec26f4f060fe5559616b4d329a6ae7cde02578d62adbabf6ba4a9a2a",
        "relay.py":"c68c51a04552c47c3365980797532c1a323c71ada8c20b34eae4da858b10574c",
        "source_reader.py":"9462529fc6337f7185831f11927ce18a7245927acf6e14fca67f191473ffe1fe",
        "stream.py":"3f240f68ea0c151b77440de0ea7abfe18826f3bf071565479e5ac571fd16236d",
        "target_wheel_overlay.py":"5a6c57071d11fbd0a8efb8ec0cde4d4648af8f9d1264b4e533ceb9ba54257034",
        "verify_allocated.py":expected_payload}
    _need(q["transfer"]["root"]==str(transfer_root) and q["transfer"]["manifest"]==dict(path=str(transfer_root/"FILES.sha256"),sha256="b900a219ee035760d40818876b75ee71dcf1253ddf05696d8293f6e24c0fc31d")
          and q["transfer"]["payload"]==dict(path=str(transfer_root/"verify_allocated.py"),sha256=expected_payload)
          and transfer["files"]==transfer_files,
          "Target did not use the exact reviewed transfer-v3/source-v5 phases")
    pre=proofs["container-preflight.json"]; imported=proofs["container-runtime.json"]
    torch_origin=Path(imported.get("torch_origin",""))
    _need(torch_origin.is_absolute() and ".." not in torch_origin.parts
          and torch_origin.is_relative_to(Path(q["lambda_runtime"]["fa4_root"])/"torch-runtime"),
          "Actual Torch import escaped the pinned private runtime")
    _need(pre["files_manifest_sha256"]==files_sha and pre["host_native"]==host["native_running"], "Pre-Torch native proof names another host/FILES")
    capsule=pre["capsule"]; logical="/run/elmo-slurm-client"; directory=Path(q["source"]["root"])/"scripts/train/joint/native_elmo_ops"
    implementations={"native_client.py":"41a3af46c6ce2acafc3ce71219815925903a6543e0559c6da1d32e0fe8c45d60",
        "wrapper_common.py":"4dc45077e7f94cc1e896152b454936fb1f2651e495b88db8529e84ba5c3f701c",
        "native_scontrol.sh":"5b4604db7643907444cafaa68d4a5c93f8f2c88618af8d947c8bb6a0d6afbca7",
        "native_squeue.sh":"7e8e24b21ac12c8d8eba73c90de5cbf01bc8e4d107aa6b1e13b00510ab9a161b"}
    _need(capsule.get("schema")=="imagenet-target-native-capsule-v1" and capsule.get("status")=="STAGED_CONTAINER_AUTH_PENDING"
          and capsule.get("job_id")==job and capsule.get("request_sha256")==digest and capsule.get("source_revision")==q["source"]["revision"]
          and capsule.get("logical_root")==logical and capsule.get("members_verified") is True
          and capsule.get("container_native_query_certified") is False and capsule.get("munge_keys_read_or_copied") is False
          and set(capsule["implementation"])==set(implementations), "Container native capsule provenance is incomplete")
    for name,value in implementations.items():
        item=capsule["implementation"][name]
        _need(item==dict(path=str(directory/name),sha256=value), "Changed exact published native helper binding")
        reopen(item,parsed=False)
    stage=canonical(capsule["stage"],"output/native-client-stage.json")
    manifest=canonical(capsule["manifest_readback"],"output/native-client-files.sha256",parsed=False)
    capsule_host=str(Path(q["lambda_runtime"]["scratch_root"])/"native-clients"/job)
    _need(stage.get("schema")=="native-elmo-client-stage-v1" and stage.get("phase")=="imagenet_target_cpu"
          and stage.get("job_id")==job and stage.get("parent_request")==dict(path=str(original/"request.json"),sha256=digest)
          and stage.get("capsule")==capsule_host and stage.get("logical_container_root")==logical
          and stage.get("manifest_sha256")==capsule["manifest_sha256"]==capsule["manifest_readback"]["sha256"]
          and stage.get("container_native_query_certified") is False and stage.get("munge_keys_read_or_copied") is False,
          "Actual native capsule stage/manifest linkage differs")
    expected_mounts=",".join((capsule_host+":"+logical+":ro",str(directory/"native_scontrol.sh")+":/usr/local/bin/scontrol:ro",
        str(directory/"native_squeue.sh")+":/usr/local/bin/squeue:ro","/run/munge:/run/munge:ro"))
    _need(capsule["exports"]==dict(ELMO_NATIVE_CLIENT_HOST_ROOT=capsule_host,ELMO_NATIVE_CLIENT_ROOT=logical,
          ELMO_NATIVE_CLIENT_MANIFEST_SHA256=capsule["manifest_sha256"],ELMO_NATIVE_MOUNTS=expected_mounts), "Native capsule mounts/libraries escaped isolated children")
    seen=set(); members=stage["members"]
    for line in manifest.splitlines():
        pair=line.split("  ",1); _need(len(pair)==2,"Malformed captured native manifest")
        value,name=pair; _need(re.fullmatch("[0-9a-f]{64}",value) and name and name not in seen and len(seen)<256
            and not Path(name).is_absolute() and ".." not in Path(name).parts and members.get(name,{}).get("sha256")==value,
            "Unsafe/changed native capsule member manifest")
        seen.add(name)
    _need(seen==set(members) and {"bin/scontrol","bin/squeue","etc/slurm.conf","lib/ld-linux-x86-64.so.2"}<=seen,
          "Captured native ELF/plugin closure is incomplete")
    for key,stem,argv in (("native_query","container-native-running",["/usr/local/bin/scontrol","show","job",job,"-o","-dd"]),
        ("native_queue","container-native-queue",["/usr/local/bin/squeue","--noheader","--jobs",job,"-o","%i|%j|%u|%T|%N|%k"])):
        query=pre[key]
        _need(query.get("argv")==argv and type(query.get("exit_code")) is int and query["exit_code"]==0
              and start<=time(query.get("observed_at"),aware=True)<=end, "Live pre-Torch native authentication failed or escaped the interval")
        _need(not canonical(query["stderr"],"output/"+stem+".stderr",parsed=False).strip(), "Container native stderr requires diagnosis")
        native_raw=canonical(query["stdout"],"output/"+stem+".txt",parsed=False)
        if key=="native_query": interval(query["stdout"],"output/"+stem+".txt")
        else: _need([line.split("|") for line in native_raw.splitlines() if line.strip()]==[[job,q["job_name"],q["user"],"RUNNING",q["node"],"imagenet-target-cpu:"+digest]], "Live native own queue differs")
    _need(auth.get("schema")=="imagenet-target-container-authentication-v2"
          and auth.get("status")=="NATIVE_AND_DECODER_AUTHENTICATED_BEFORE_TORCH"
          and auth.get("job_id")==job and auth.get("request_sha256")==digest and auth.get("host_preparation")==host_item
          and auth.get("torch_not_imported") is True and auth.get("data_admitted") is False,
          "Actual same-container decoder authentication before Torch is missing")
    interval(auth["native_running"],"output/container-payload-native-running.txt")
    ns=namespaces(auth["namespaces"]); step=auth["step_id"]
    _need(isinstance(step,str) and re.fullmatch("[0-9]+",step) and pre["step_id"]==step and namespaces(pre["namespaces"])==ns,
          "Container auth changed namespace/numeric step")
    wheel=q["decoder_wheel"]; overlay=out/"pillow-overlay"
    _need(wheel==dict(path=str(original/"inputs/pillow-13.0.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"),
          sha256="42d63be3af4f9b951b199b0a9d23365d37ab14687e6d77b74940432c8515c4ca"), "Wrong exact target cp312 decoder wheel")
    service=q["decoder_service"]; service_root=original/"inputs/decoder-service"; helper_sha="d5167548d4982c9554a40fdb246f5ba0d301a5f2c1d23efbe61a147205031caf"
    _need(service["root"]==str(service_root) and service["helper"]==dict(path=str(service_root/"imagenet_wheel_overlay.py"),sha256=helper_sha)
          and re.fullmatch("[0-9a-f]{40}",service["revision"]), "Unbound published decoder service/helper")
    manifest=canonical(service["manifest"],"inputs/decoder-service/SERVICE_FILES.sha256",parsed=False); entries={}
    for line in manifest.splitlines():
        pair=line.split("  ",1);_need(len(pair)==2,"Malformed published decoder service manifest")
        value,name=pair;_need(re.fullmatch("[0-9a-f]{64}",value) and name and name not in entries and len(entries)<128
            and not Path(name).is_absolute() and ".." not in Path(name).parts,"Unsafe decoder service code member")
        entries[name]=value;reopen(dict(path=str(service_root/name),sha256=value),parsed=False)
    _need(entries.get("imagenet_wheel_overlay.py")==helper_sha,"Published maintained decoder helper differs")
    publication=canonical(service["publication"],"inputs/decoder-service-publication.json")
    _need(publication.get("status")=="PASS" and publication.get("upstream_verified") is True
          and publication.get("revision")==service["revision"] and publication.get("manifest_sha256")==service["manifest"]["sha256"],
          "Decoder service publication/readback binding is missing")
    decoder=auth["decoder_runtime"]
    expected_interpreter=dict(implementation="cpython",major_minor=[3,12],physical_executable="/usr/bin/python3.12")
    measured=pre["python"]
    _need(q.get("decoder_interpreter")==expected_interpreter and isinstance(measured,dict)
          and set(measured)=={"physical_executable","version","implementation","system","machine"}
          and measured["physical_executable"]==expected_interpreter["physical_executable"]
          and measured["version"]==decoder["python_version"] and measured["implementation"]==decoder["python_implementation"]
          and measured["system"]==decoder["system"] and measured["machine"]==decoder["machine"]
          and decoder["executable"]==measured["physical_executable"], "Actual pre-Torch interpreter/ABI measurement differs")
    _need(decoder.get("schema")=="imagenet-target-decoder-runtime-v1" and decoder.get("pillow_version")=="13.0.0"
          and decoder.get("wheel")==wheel and decoder.get("python_implementation")=="cpython" and decoder.get("system")=="Linux"
          and decoder.get("machine")=="x86_64" and isinstance(decoder.get("python_version"),str)
          and re.fullmatch(r"3\.12\.[0-9]+",decoder["python_version"])
          and decoder.get("pillow_origin")==str(overlay/"PIL/__init__.py")
          and decoder.get("native_decoder_origin")==str(overlay/"PIL/_imaging.cpython-312-x86_64-linux-gnu.so")
          and decoder.get("executable")=="/usr/bin/python3.12", "Actual isolated cp312 Pillow/native import origin differs")
    validate_installation(decoder.get("installation",{}),wheel["sha256"],overlay,profile="cp312")
    pre_item=done["decoder_prebootstrap"]; bootstrap=canonical(pre_item,"output/decoder-prebootstrap.json")
    _need(bootstrap.get("schema")=="imagenet-target-decoder-prebootstrap-v1"
          and bootstrap.get("status")=="EXACT_DECODER_ACTIVE_BEFORE_MAINTAINED_TORCH_BOOTSTRAP"
          and bootstrap.get("job_id")==job and bootstrap.get("request_sha256")==digest and bootstrap.get("files_manifest_sha256")==files_sha
          and bootstrap.get("host_preparation")==host_item and bootstrap.get("container_authentication")==auth_item
          and bootstrap.get("decoder_runtime")==decoder and bootstrap.get("payload")==q["transfer"]["payload"]
          and bootstrap.get("torch_not_imported") is True and bootstrap.get("base_environment_mutated") is False
          and bootstrap.get("step_id")==step and namespaces(bootstrap.get("namespaces"))==ns, "Decoder activation before maintained Torch bootstrap is unproven")
    interval(bootstrap["native_running"],"output/prebootstrap-native-running.txt")
    query=bootstrap["native_query"]
    _need(set(query)=={"argv","exit_code","stdout","stderr","observed_at","failure","record"}
          and query["argv"]==["/usr/local/bin/scontrol","show","job",job,"-o","-dd"]
          and type(query["exit_code"]) is int and query["exit_code"]==0 and query["failure"] is None
          and query["stdout"]==bootstrap["native_running"]
          and start<=time(query["observed_at"],aware=True)<=end,
          "Fresh native authentication immediately before decoder/Torch activation is missing")
    _need(not canonical(query["stderr"],"output/prebootstrap-native-running.stderr",parsed=False).strip()
          and canonical(query["record"],"output/prebootstrap-native-running.query.json")
          =={k:v for k,v in query.items() if k!="record"}, "Prebootstrap native query raw/record binding differs")
    _need(imported.get("host_preparation")==host_item and imported.get("container_authentication")==auth_item
          and imported.get("decoder_prebootstrap")==pre_item and imported.get("preflight")==done["producer_proofs"]["container-preflight.json"]
          and imported.get("step_id")==step and namespaces(imported.get("namespaces"))==ns
          and imported.get("pillow_origin")==str(overlay/"PIL/__init__.py")
          and imported.get("real_executable")==decoder["executable"]
          and all(imported.get(k)==decoder[k] for k in ("python_version","python_implementation","system","machine")),
          "Torch/decode phase left the authenticated container/proof chain")
    _need(report["destination_inventory"]["path"]==str(out/"destination-raw.json")
          and re.fullmatch("[0-9a-f]{64}",report["destination_inventory"]["sha256"]), "Noncanonical full destination inventory")
    # Its large body is reopened only by verify_dataset(allocated=True).


def verify_cache_native(packet, report_binding, training_request, *, allocated=False):
    """Authenticate real allocated cache readback; a PASS summary cannot substitute."""
    import json
    import re
    from scripts.train.joint.contract import bound, sha
    _need(packet.get("schema") == "imagenet-cache-native-evidence-v1", "Explicit cache-native evidence protocol required")
    q = _bounded_bound(packet["request"]); done = _bounded_bound(packet["completion"])
    digest = packet["request"]["sha256"]
    original = Path(q["root"]); evidence = Path(packet["evidence_root"])
    _need(original.is_absolute() and evidence.is_absolute() and ".." not in original.parts
          and ".." not in evidence.parts, "Cache evidence requires an explicit immutable root")
    def reopen(binding, *, parsed=True):
        path = Path(binding["path"])
        _need(isinstance(binding,dict) and set(binding)=={"path","sha256"}
              and isinstance(binding["sha256"],str) and re.fullmatch("[0-9a-f]{64}",binding["sha256"])
              and path.is_relative_to(original) and ".." not in path.parts, "Cache evidence escaped its producer root")
        actual = evidence/path.relative_to(original)
        _need(actual.is_file() and not actual.is_symlink() and actual.resolve().is_relative_to(evidence.resolve())
              and actual.stat().st_size <= 4 << 20
              and sha(actual) == binding["sha256"], "Changed/oversized cache producer metadata")
        return _small_json(actual) if parsed else _small_text(actual)
    _need(q.get("schema") == "imagenet-target-cpu-request-v2" and q.get("state") == "BOUND_TARGET_CPU"
          and q["root"] == "/workspace/users/paphiwetsa3/tasks/elmo-qknorm-swiglu-rmsnorm-imagenet-20261026-v1/imagenet-target-cpu-v2"
          and q["output"] == str(original/"output")
          and q["uid"] == 41003 and q["user"] == "paphiwetsa3" and q["node"] == "gpu21"
          and q["account"] == "loaner" and q["partition"] == "rl2" and q["qos"] == "loaner-normal"
          and q["job_name"] == "elmo-imagenet-target-cache-v2"
          and q["capture_name"] == "elmo-native-imagenet-target-cache-v2"
          and q["wrapper_protocol"] == "REQUIRES_REVIEWED_LAMBDA_CPU_WRAPPER_V2"
          and all(q["native"][n] == "/usr/local/bin/"+n for n in ("sbatch","scontrol","squeue"))
          and q["data_root"] == training_request["data_cache"]
          and q["resources"] == dict(cpus=4,gpus=0,memory_mib=24576,time_limit="04:00:00")
          and q["capture_resources"] == dict(cpus=2,gpus=0,memory_mib=1024,time_limit="00:10:00")
          and q["counts"] == dict(train=1281167,val=50000) and q["classes"] == 1000, "Wrong actual ImageNet cache producer request")
    _need(done.get("schema") == "imagenet-cache-readback-completion-v2" and done.get("status") == "PASS"
          and done.get("request_sha256") == digest and done.get("cache_report_sha256") == report_binding["sha256"]
          and done.get("data_root") == q["data_root"], "Missing cache completion bound to actual copied bytes")
    job = str(done["job_id"]); cap = reopen(done["native_capture"])
    _need(re.fullmatch(r"[1-9][0-9]*",job) and cap.get("schema") == "imagenet-cache-native-capture-v1"
          and cap["parent_job_id"] == job and cap["request_sha256"] == digest
          and cap["files_manifest_sha256"] == done["files_manifest_sha256"], "Wrong actual cache observer")
    parent_files = _manifest_entries(evidence/"FILES.sha256")
    _need(sha(evidence/"FILES.sha256") == done["files_manifest_sha256"]
          and {"request.json","guard.py","submit_once.py","producer.sbatch","capture.sbatch","run_allocated.py",
               "observe.py","finalize.py","runtime_prereq.py","native_capsule.py","authenticate_decoder.py",
               "decoder_runtime_bootstrap.py","payload_dispatch.py"} <= set(parent_files),
          "Incomplete cache wrapper code closure")
    declared_inputs={}
    pending=[q]
    while pending:
        value=pending.pop()
        if isinstance(value,dict):
            if set(value) == {"path","sha256"}:
                target=Path(value["path"])
                if target.is_relative_to(original) and ".." not in target.parts:
                    declared_inputs[str(target.relative_to(original))]=value["sha256"]
            else:pending.extend(value.values())
        elif isinstance(value,list):pending.extend(value)
    for name,expected in parent_files.items():
        _need(not Path(name).is_absolute() and ".." not in Path(name).parts, "Unsafe cache wrapper manifest path")
        if not allocated and name in declared_inputs and not name.endswith((".py",".sh",".sbatch")):
            _need(expected == declared_inputs[name], "Changed declared cache input binding")
        else:
            _need((evidence/name).is_file() and not (evidence/name).is_symlink()
                  and (allocated or (evidence/name).stat().st_size <= 4 << 20)
                  and sha(evidence/name) == expected, "Changed/oversized cache wrapper code/request")
    _need(parent_files["request.json"] == digest, "Cache wrapper ran another actual request")
    claims = evidence/"claims"/digest
    observer = str(cap["observer_job_id"])
    _need(re.fullmatch(r"[1-9][0-9]*",observer) and observer != job, "Wrong actual observer ID")
    prepared=_small_json(claims/"PREPARED.json")
    _need(prepared["request_sha256"] == digest and prepared["files_manifest_sha256"] == done["files_manifest_sha256"]
          and prepared["parent_argv"] == _cache_argv(q,digest,done["files_manifest_sha256"]), "Wrong accepted cache PREPARED claim")
    _need(_small_text(claims/"PARENT_JOB_ID").strip() == job
          and _small_text(claims/"CAPTURE_JOB_ID").strip() == observer,
          "Wrong accepted cache parent/observer")
    for kind,identity in (("parent",job),("capture",observer)):
        _need(_small_text(claims/(kind+"-sbatch.stdout")).strip() == identity
              and _small_text(claims/(kind+"-sbatch.exit")).strip() == "0"
              and _small_json(claims/(kind+"-sbatch.argv.json")) ==
                  _cache_argv(q,digest,done["files_manifest_sha256"],parent=job if kind == "capture" else None),
              "Unknown or changed cache submission outcome")
    def native(raw, identity, state, *, capture=False):
        pairs = re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9:/_-]*)=(\S+)',raw)
        _need(len(dict(pairs)) == len(pairs), "Duplicate cache native fields")
        f = dict(pairs); resources = q["capture_resources"] if capture else q["resources"]
        mem = str(resources["memory_mib"]//1024)+"G"
        expected_comment = ("imagenet-target-observer:"+digest+":"+job) if capture else "imagenet-target-cpu:"+digest
        _need(f.get("JobId") == identity and f.get("JobName") == q["capture_name" if capture else "job_name"]
              and f.get("UserId") == q["user"]+"("+str(q["uid"])+")"
              and f.get("JobState") == state and f.get("Account") == q["account"]
              and f.get("Partition") == q["partition"] and f.get("QOS") == q["qos"]
              and f.get("NumNodes") in (("1","1-1") if state == "PENDING" else ("1",))
              and f.get("NumCPUs") == str(resources["cpus"]) and f.get("NumTasks") == "1"
              and f.get("MinMemoryNode") in (mem,str(resources["memory_mib"]),str(resources["memory_mib"])+"M")
              and f.get("TimeLimit") == resources["time_limit"] and f.get("Requeue") == "0"
              and f.get("Restarts") == "0" and f.get("ReqNodeList") == q["node"]
              and f.get("Command") == str(original/("capture.sbatch" if capture else "producer.sbatch"))
              and f.get("WorkDir") == str(original) and f.get("Comment") == expected_comment
              and f.get("SubmitTime") not in (None,"Unknown","N/A"),
              "Cache native owner/resources/actual request differs")
        for key in ("ReqTRES","AllocTRES"):
            value=f.get(key,"")
            _need("gpu" not in value.lower(), "GPU cache readback is forbidden")
            if key == "ReqTRES" or state != "PENDING":
                parts=[item.split("=",1) for item in value.split(",")]
                _need(all(len(item)==2 for item in parts) and len(dict(parts))==len(parts)
                      and dict(parts).get("cpu")==str(resources["cpus"])
                      and dict(parts).get("mem")==mem and dict(parts).get("node")=="1",
                      "Missing exact actual cache native TRES")
        for key,value in f.items():
            if key.lower().startswith(("gres","job_gres","tresper")):
                _need(value.lower() in ("","(null)","n/a","none","0") or key == "TresPerTask"
                      and value in ("cpu:"+str(resources["cpus"]),"cpu="+str(resources["cpus"])),
                      "Unexpected cache GRES assignment")
        if state == "PENDING":
            _need(f.get("Reason") == ("Dependency" if capture else "JobHeldUser"), "Missing genuine held/dependency acceptance")
            _need((re.fullmatch(r"afterany:"+re.escape(job)+r"(?:\(unfulfilled\))?",f.get("Dependency","")) is not None)
                  if capture else f.get("Dependency") == "(null)", "Wrong cache afterany parent")
        else:
            _need(f.get("NodeList") == q["node"] and f.get("StartTime") not in (None,"Unknown","N/A"),
                  "Cache producer lacks an allocated node/start")
        if state == "COMPLETED":
            _need(f.get("ExitCode") == "0:0" and f.get("EndTime") not in (None,"Unknown","N/A")
                  and f.get("DerivedExitCode","0:0")=="0:0" and f["EndTime"] >= f["StartTime"], "Cache parent did not complete successfully")
        return f
    accepted_times={}
    for name,identity,capture_flag in (("HELD_NATIVE.txt",job,False),("CAPTURE_PENDING_NATIVE.txt",observer,True),
                                       ("RELEASE_HELD_NATIVE.txt",job,False),("RELEASE_CAPTURE_PENDING_NATIVE.txt",observer,True)):
        f=native(_small_text(claims/name),identity,"PENDING",capture=capture_flag)
        _need(identity not in accepted_times or accepted_times[identity]==f["SubmitTime"], "Cache held identity changed before release")
        accepted_times[identity]=f["SubmitTime"]
    release = _small_json(claims/"RELEASE_ATTEMPT.json")
    _need(release["parent"] == job and release["capture"] == observer and release["request_sha256"] == digest,
          "Missing durable once-only cache release")
    accepted=_small_json(claims/"RELEASE_ACCEPTED.json")
    _need(accepted.get("parent")==job and accepted.get("capture")==observer
          and accepted.get("status")=="RELEASE_ACCEPTED_NOT_COMPLETION" and accepted.get("stdout")=="",
          "Cache release outcome is unknown or incomplete")
    _need([r["kind"] for r in cap["queries"]] == ["job","steps"], "Missing complete cache native inventory")
    for query in cap["queries"]:
        argv = [q["native"]["scontrol"],"show","job",job,"-o","-dd"] if query["kind"] == "job" else [q["native"]["scontrol"],"show","step","-o"]
        _need(query["argv"] == argv and type(query["exit_code"]) is int and query["exit_code"] == 0, "Cache native query did not succeed")
        target=str(original/("native-"+job+(".txt" if query["kind"] == "job" else ".steps.txt")))
        _need(query["stdout"]["path"] == target and query["stderr"]["path"] == target+".stderr",
              "Noncanonical cache terminal query")
        _need(not reopen(query["stderr"],parsed=False).strip(), "Cache native stderr requires diagnosis")
        raw = reopen(query["stdout"],parsed=False)
        if query["kind"] == "job":
            native(raw,job,"COMPLETED")
            _need(done["native_terminal"] == query["stdout"], "Changed cache terminal binding")
        else:
            for line in raw.splitlines():
                if not line.strip() or line.strip() in ("No job steps in the system","No job steps in the system."):continue
                pairs=re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9:/_-]*)=(\S+)',line)
                _need(len(pairs) == len(dict(pairs)), "Duplicate native cache step fields")
                f = dict(pairs);step=f.get("StepId",f.get("StepID"))
                _need(step is not None and not (step==job or step.startswith(job+".") or step.startswith(job+"_")),
                      "Cache parent still owns worker steps or full inventory is unknown")
    observed = native(reopen(cap["observer_running"],parsed=False),observer,"RUNNING",capture=True)
    accepted_observer = native(_small_text(claims/"CAPTURE_PENDING_NATIVE.txt"),observer,"PENDING",capture=True)
    _need(observed["SubmitTime"] == accepted_observer["SubmitTime"], "Cache observer identity changed after acceptance")
    _need(reopen(done["host_exit"],parsed=False).strip() == "0", "Cache host exit was not zero")
    _need(report_binding["path"] == str(original/"output/cache-readback.json")
          and done["host_exit"]["path"] == str(original/("host-exit-"+job+".txt"))
          and done["native_capture"]["path"] == str(original/("native-"+job+".capture.json")),
          "Noncanonical cache producer evidence")
    for key in ("root-review.json","deployment-readback.json"):
        proof=_small_json(evidence/key)
        _need(proof.get("status") == "PASS" and proof.get("root") == str(original)
              and proof.get("request_sha256") == digest and proof.get("files_manifest_sha256") == done["files_manifest_sha256"],
              "Missing actual cache review/deployment proof")
    _need(isinstance(q["request_id"],str) and q["request_id"] and "/" not in q["request_id"]
          and q["request_id"] not in (".","..")
          and _small_text(evidence/"request-claims"/q["request_id"]/"REQUEST_SHA256").strip() == digest,
          "Missing actual cache once-only request claim")
    _need(not (evidence/"output/cache-readback.json.failure.json").exists()
          and not (evidence/"output/destination-raw.json.failure.json").exists(), "Target producer recorded an explicit failure")
    report = reopen(report_binding)
    _need(report.get("schema") == "imagenet-cache-readback-v2"
          and report.get("status") == "EXTERNAL_COMPLETION_REQUIRED" and report.get("data_admitted") is False
          and report.get("request_sha256") == digest and str(report.get("job_id")) == job
          and report.get("data_root") == q["data_root"] and report.get("counts") == q["counts"]
          and report.get("classes") == 1000 and report.get("all_image_bytes_hashed") is True
          and report.get("all_images_decoded") is True and report["storage"].get("file_fsync") is True
          and report["storage"].get("directory_fsync") is True
          and report["source_raw_inventory"] == q["source_raw_inventory"]
          and report["source_completion"] == q["source_completion"]
          and report["source"] == q["source"]
          and all(report.get(k) == "EXTERNAL_REQUIRED" for k in
                  ("host_completion","native_completion","afterany_completion")),
          "Actual cache completion does not close the exact non-admitting payload")
    _need(report["native_running"]["path"] == str(original/"output/native-running.txt"),
          "Noncanonical cache producer running proof")
    running = native(reopen(report["native_running"],parsed=False),job,"RUNNING")
    held = native(_small_text(claims/"HELD_NATIVE.txt"),job,"PENDING")
    terminal = native(reopen(done["native_terminal"],parsed=False),job,"COMPLETED")
    _need(running["SubmitTime"] == terminal["SubmitTime"] == held["SubmitTime"]
          == accepted_times[job] and running["StartTime"] == terminal["StartTime"]
          and observed["StartTime"] >= terminal["EndTime"]
          and done["native_start"] == terminal["StartTime"] and done["native_end"] == terminal["EndTime"],
          "Cache producer accepted/running/terminal identity differs")
    names={"prior-runtime-proof.json","runtime-current.json","container-preflight.json","container-runtime.json",
           "source-stage-binding.json","transfer-code-readback.json","filesystem-check.json"}
    _need(set(done["producer_proofs"]) == names and done.get("training_admitted") is False,
          "Cache completion lacks the real producer prerequisite closure")
    proofs={}
    for name,binding in done["producer_proofs"].items():
        _need(binding["path"] == str(original/"output"/name), "Noncanonical cache producer prerequisite")
        proofs[name]=reopen(binding)
    prior=proofs["prior-runtime-proof.json"]; reference=q["runtime_prerequisite"]
    _need(prior.get("status") == "PASS_PRIOR_RUNTIME_SCOPE_ONLY"
          and prior["prior_request_sha256"] == reference["request"]["sha256"]
          and prior["prior_files_manifest_sha256"] == reference["files_manifest"]["sha256"]
          and prior["completion_sha256"] == reference["completion"]["sha256"]
          and prior.get("ImageNet_source_data_model_admission") is False and prior.get("GPU_or_training_admission") is False,
          "A failed prior runtime attempt cannot certify the target")
    current=proofs["runtime-current.json"]; pre=proofs["container-preflight.json"]; imported=proofs["container-runtime.json"]
    for item in (current,pre,imported):
        _need(item["job_id"] == job and item["request_sha256"] == digest, "Cache producer runtime identity differs")
    _need(current.get("status") == "ACTUAL_CURRENT_RUNTIME_PASS_CONTAINER_IMPORT_PENDING"
          and current.get("all_regular_files_freshly_hashed") is True and current.get("exact_inventory") is True
          and current.get("regular_files") == 72878 and current.get("symlinks") == 10
          and current.get("regular_bytes") == 17361459887 and current.get("runtime_mutated") is False
          and pre.get("status") == "AUTHENTICATED_BEFORE_TORCH" and pre.get("zero_gpu") is True
          and pre["source_revision"] == q["source"]["revision"] and pre["source_manifest_sha256"] == q["source"]["manifest"]["sha256"]
          and imported.get("status") == "CPU_IMPORT_PASS" and imported.get("zero_gpu") is True
          and imported.get("torch_version") == "2.9.1+cu126" and imported.get("pillow_version") == "13.0.0"
          and imported.get("payload") == q["transfer"]["payload"], "Actual target runtime/source/container checks are incomplete")
    stage=proofs["source-stage-binding.json"]; transfer=proofs["transfer-code-readback.json"]; filesystem=proofs["filesystem-check.json"]
    _need(stage["source"] == q["source"] and stage.get("source_inputs_freshly_hashed") is True
          and stage.get("source_stage_verified") is True and transfer["manifest"] == q["transfer"]["manifest"]
          and transfer["payload"] == q["transfer"]["payload"] and filesystem.get("status") == "PASS"
          and filesystem.get("runtime_mounts_verified") is True and filesystem.get("write_fsync_verified") is True
          and filesystem.get("unix_socket_verified") is True, "Target source/payload/storage authentication differs")
    _need(stage["source_stage"]["path"] == str(original/"output/source-stage-native.json")
          and stage["projection"]["path"] == str(original/"output/source-stage-projection.json"), "Wrong actual source stage")
    staged=reopen(stage["source_stage"])
    native_source=str(Path(q["lambda_runtime"]["scratch_root"])/"source"/job/"source")
    _need(staged.get("status") == "PASS" and staged.get("schema")=="joint-node-local-source-v2"
          and str(staged["job_id"]) == job and staged["node"] == q["node"] and staged["source"]==native_source
          and staged["archive_sha256"]==q["source"]["archive"]["sha256"]
          and staged["manifest_sha256"] == q["source"]["manifest"]["sha256"]
          and staged.get("all_manifest_files_verified") is True, "Actual source stage is not certified")
    projection=reopen(stage["projection"])
    _need(projection.get("schema")=="imagenet-source-physical-projection-v1"
          and projection.get("status")=="SOURCE_PROJECTION_FRESH_MANIFEST_READBACK_PASS"
          and projection.get("job_id")==job and projection.get("request_sha256")==digest
          and projection.get("native_stage")==stage["source_stage"] and projection.get("native_source")==native_source
          and projection.get("projected_source")==q["source"]["root"] and projection.get("source_manifest")==q["source"]["manifest"]
          and type(projection.get("files")) is int and projection["files"]==pre["source_files"]>0
          and all(projection.get(k) is True for k in ("all_manifest_files_verified","physical_source","all_files_readonly",
              "all_members_owned","files_and_directories_fsynced")) and projection.get("source_bytes_changed") is False,
          "Exact read-only physical source projection is missing")
    _cache_phase_chain(q,done,report,training_request,proofs,reopen,native,running,terminal)
    return done


def run(request_path, request):
    """First fresh ImageNet allocation through the maintained runtime and trainer."""
    import json
    import os
    import shutil
    import subprocess
    import sys
    from scripts.train.joint.contract import bound, sha, write_once, verify_evidence
    from scripts.train.joint.allocation import admit
    path = Path(request_path); root = path.parent.parent; source = Path(request["source"])
    _need(selected(request) and request["schema"] == "joint-training-v2", "Wrong ImageNet request")
    _need(request["phase"] in ("smoke","full"), "Unknown ImageNet allocation phase")
    _need(not request.get("continuation") and not request.get("interruption")
          and request.get("allocation_stop_step",300000) == 300000,
          "Fresh ImageNet driver cannot reuse unrelated continuation/crash admission")
    allocation = admit(request,status_root=root/"status",bootstrap=root/"runtime_bootstrap.py")
    _need(sha(request["source_manifest"]) == request["source_manifest_sha256"], "Changed ImageNet source manifest")
    for line in Path(request["source_manifest"]).read_text().splitlines():
        expected,name = line.split("  ",1)
        _need(not Path(name).is_absolute() and ".." not in Path(name).parts
              and sha(source/name) == expected, "Changed published ImageNet source: "+name)
    _need(sha(request["data_manifest"]) == request["data_manifest_sha256"], "Changed ImageNet data manifest")
    verify_dataset(request, allocated=True)
    smoke = request["phase"] == "smoke"
    completion = preflight = None
    if not smoke:
        completion,preflight = verify_evidence(request)
        from scripts.train.joint.retention import check_storage
        check_storage(request,completion)
        from scripts.train.joint.run import credentials
        from scripts.train.joint.check_wandb import inspect_run
        credentials()
        _need(inspect_run(request["run_id"])["exists"] is False, "Fresh ImageNet identity already exists")
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    import torch
    with initialize_config_dir(config_dir=str(source/"egomimic/hydra_configs"),version_base=None):
        cfg = compose(config_name="train_zarr_cartesian",overrides=request["overrides"])
    resolved = OmegaConf.to_container(cfg,resolve=True,throw_on_missing=True)
    verify_model_execution(request,resolved)
    if not smoke:
        smoke_cfg = OmegaConf.to_container(OmegaConf.create(
            bound(request["smoke"]["resolved"],False)),resolve=True)
        compare_configs(resolved,smoke_cfg,request)
        from scripts.train.joint.continuation import configure_allocation
        configure_allocation(cfg,request,0,300000)
    with torch.device("meta"):
        network = instantiate(cfg.model.pipeline.stages[1].model)
    counts = network.expert_parameter_counts()
    _need(counts == COUNTS and sum(p.numel() for p in network.parameters()) == 314227000
          and list(network.expert_widths) == [1024,512,384,384]
          and network.active_streams("imagenet") == ("image",)
          and all(p.requires_grad for p in network.parameters()), "Constructed full/image-only ImageNet architecture changed")
    del network
    data = instantiate(cfg.data)
    context = data.prepare_context(mode="train",normalization=None,normalizer=None)
    _need(len(data.datasets["train"]) == 1281167 and len(data.datasets["val"]) == 1000
          and len(data.val_dataloader()) == 1000, "Wrong official TRAIN or full frozen class-balanced anchors")
    if preflight is not None:
        _need(context.fingerprint() == preflight["provenance"]["data_context_sha256"], "Actual full data context differs from smoke")
    _need(shutil.disk_usage(root).free > request["reserved_checkpoint_bytes"], "Insufficient actual checkpoint destination space")
    output = Path(request["output"])
    _need(str(cfg.paths.output_dir) == str(output) and not output.exists(), "Fresh ImageNet output collision")
    provenance = dict(source_revision=request["source_revision"],source_manifest_sha256=request["source_manifest_sha256"],
                      data_manifest_sha256=request["data_manifest_sha256"],data_context_sha256=context.fingerprint())
    if not smoke:provenance["run_id"] = request["run_id"]
    if allocation is None:
        from gpu_health_set import admit_gpu_set
        admit_gpu_set(root/"status"/(path.stem+"-"+os.environ["SLURM_JOB_ID"]+"-gpu.json"),4)
    write_once(path.with_suffix(".preflight.json"),dict(status="PREFLIGHT_PASS" if smoke else "FULL_ADMISSION_PASS",
        job_id=os.environ["SLURM_JOB_ID"],request_sha256=sha(path),provenance=provenance,
        parameters=314227000,expert_parameters=counts,expert_widths=[1024,512,384,384],
        active_streams=STREAMS,executed_streams=["image"],world_size=4,global_batch=32,allocation_attestation=allocation))
    with path.with_suffix(".resolved.yaml").open("x") as stream:stream.write(OmegaConf.to_yaml(cfg,resolve=True))
    bootstrap = root/"runtime_bootstrap.py"
    from scripts.train.joint.service_bootstrap import command
    if smoke:
        subprocess.run([sys.executable,str(bootstrap),"-m","torch.distributed.run","--standalone","--nnodes=1",
            "--nproc-per-node=4","--max-restarts=0",*command(request,path,bootstrap,[
            str(source/"scripts/world_model/verify_imagenet_smoke.py"),str(output),
            "--capacity-config",str(path.with_suffix(".resolved.yaml")),
            "--capacity-output",str(path.with_suffix(".capacity.json")),
            "--capacity-source-manifest-sha256",request["source_manifest_sha256"],
            "--capacity-data-manifest-sha256",request["data_manifest_sha256"]])[1:]],check=True,cwd=source)
    output.mkdir(parents=True)
    (output/".training-admitted").write_text(sha(path)+"\n")
    extra = "+run_provenance={"+",".join(k+":"+json.dumps(v) for k,v in provenance.items())+"}"
    training_arguments = request["overrides"] if smoke else ["--config-path",str(path.parent),
        "--config-name",path.with_suffix(".resolved.yaml").stem,"hydra.run.dir="+str(output/"attempts/step000000000"),
        "hydra.job.chdir=false"]
    subprocess.run([sys.executable,str(bootstrap),"-m","torch.distributed.run","--standalone","--nnodes=1",
        "--nproc-per-node=4","--max-restarts=0",*command(request,path,bootstrap,["-m","egomimic.trainHydra"])[1:],
        *training_arguments,extra],check=True,cwd=source)
    verifier = source/"scripts/world_model/verify_imagenet_smoke.py" if smoke else source/"scripts/train/joint/verify_full.py"
    subprocess.run(command(request,path,bootstrap,[str(verifier),str(output) if smoke else str(path)]),check=True,cwd=source)


def retained_milestones(output):
    """Inventory canonical scheduled files only; this does not load checkpoints."""
    import re
    output=Path(output); directory=output/"checkpoints"; milestones={}
    for path in directory.glob("*.ckpt"):
        if path.name == "last.ckpt":continue
        match=re.fullmatch(r"epoch_epoch=[0-9]{4,}-step_step=([0-9]{9})\.ckpt",path.name)
        _need(match is not None and path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
              "Unexpected, empty or noncanonical production checkpoint")
        step=int(match[1]);_need(step not in milestones, "Duplicate production checkpoint milestone")
        milestones[step]=path
    _need(set(milestones) == set(range(20000,300001,20000)), "All15 immutable scheduled checkpoints must remain local")
    latest=directory/"last.ckpt"
    _need(latest.is_symlink() and latest.resolve() == milestones[300000].resolve(),
          "last.ckpt must link to300k production milestone")
    return milestones,latest


def validation_history(paths, milestones):
    """Reject duplicates/nonfinite events before claiming15 scheduled passes."""
    import csv
    validation={}
    for csv_path in paths:
        segment={}
        with Path(csv_path).open() as stream:
            for row in csv.DictReader(stream):
                metrics={k:float(v) for k,v in row.items() if k.startswith("val/imagenet/") and v}
                if metrics:
                    _need(set(metrics) == {METRIC} and all(math.isfinite(v) for v in metrics.values()),
                          "Nonfinite/unexpected ImageNet validation history")
                    scalar=float(row["step"])
                    _need(math.isfinite(scalar) and scalar.is_integer() and scalar >= 0, "Invalid validation optimizer step")
                    step=int(scalar)
                    _need(step not in segment, "Duplicated scheduled ImageNet validation within one log")
                    segment[step]=metrics
        _need(not validation.keys() & segment.keys(), "Duplicated scheduled validation across logs")
        validation.update(segment)
    _need(len(validation) == 15 and all(step in validation or step-1 in validation for step in milestones),
          "Missing or extra scheduled ImageNet validation events")
    return validation


def verify_full(request_path, request):
    """Allocated terminal audit of the15 retained production milestones; no save."""
    import csv
    import json
    import os
    import re
    from scripts.train.joint.contract import sha,write_once,verify_evidence
    from scripts.train.joint.continuation import log_segments
    _,preflight = verify_evidence(request)
    _need(request["phase"] == "full" and request["target_steps"] == 300000
          and not request.get("continuation"), "ImageNet terminal verification is fresh300k-only")
    output = Path(request["output"]); directory = output/"checkpoints"
    milestones,latest = retained_milestones(output)
    import torch
    from omegaconf import OmegaConf
    from egomimic.pl_utils.imagenet_smoke import validate_imagenet_checkpoint,sample_imagenet_hidden
    checkpoint = torch.load(latest.resolve(),map_location="cpu",weights_only=False,mmap=True)
    cfg = OmegaConf.load(Path(request_path).with_suffix(".resolved.yaml"))
    model_config = OmegaConf.create(checkpoint["hyper_parameters"]["config_tree"])
    _need(OmegaConf.to_container(cfg.model,resolve=True) == OmegaConf.to_container(model_config.model,resolve=True)
          and model_config.run_provenance.run_id == request["run_id"], "Terminal checkpoint source/run/model differs")
    wrapper,data,optimizer,proof = validate_imagenet_checkpoint(checkpoint,cfg,model_config,
        expected_step=300000,expected_context_sha256=preflight["provenance"]["data_context_sha256"])
    sample = next(iter(data.val_dataloader()))["imagenet"]
    sample["world"]["image"]["value"].fill_(float("nan"))
    sampled = sample_imagenet_hidden(wrapper,sample)
    validation = validation_history(log_segments(request),milestones)
    records = {str(step):dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)) for step,path in sorted(milestones.items())}
    write_once(output/"full-completion.json",dict(schema="imagenet-native-full-v1",status="PASS",
        request_sha256=sha(request_path),job_id=os.environ["SLURM_JOB_ID"],run_id=request["run_id"],
        global_step=300000,world_size=4,checkpoint=str(latest.resolve()),checkpoint_sha256=records["300000"]["sha256"],
        strict_full_wrapper_optimizer_scheduler_reload=True,sampler_all_rank_rng_verified=True,
        retained_milestones=15,validation_events=15,validation_anchors_per_event=1000,
        **sampled,checkpoint_milestones=records,proof=proof,
        host_and_native_completion="EXTERNAL_REQUIRED"))
