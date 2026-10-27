"""Synthetic pure service contracts; no actual producer/native/cache PASS proof.

Only verify_dataset is patched in completion-unit fixtures. The actual
source/cache/native evidence closure must pass separately and is never
represented by the synthetic dictionaries below.
"""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from scripts.train.joint import imagenet
from scripts.train.joint.lr_policy import from_config, update_lr


def service_fixture(phase="full"):
    source = Path(imagenet.__file__).resolve().parents[3]
    with initialize_config_dir(version_base=None, config_dir=str(source / "egomimic/hydra_configs")):
        cfg = compose(config_name="train_zarr_cartesian", overrides=[
            "+experiment=pixel_proprio/native_joint/imagenet_image_generation",
            "paths.output_dir=/fixture-only/" + phase,
        ])
    OmegaConf.set_struct(cfg, False)
    cfg.paths.work_dir = "/fixture-only/work"
    for key in ("manifest", "class_language", "language_features", "validation_anchor"):
        cfg.data[key+"_path"] = "/fixture-only/shared/" + key + ".json"
        cfg.data[key+"_sha256"] = "1" * 64
    cfg.data.data_root = "/fixture-only/node-cache"
    for key in list(cfg.callbacks):
        value = cfg.callbacks[key]
        if isinstance(value, (dict, type(cfg))) and value.get("_target_") == "egomimic.pl_utils.joint_smoke.ConditionalOptimizerUsage":
            del cfg.callbacks[key]
    cfg.callbacks["conditional_usage"] = dict(_target_="egomimic.pl_utils.joint_smoke.ConditionalOptimizerUsage")
    cfg.logger["csv"] = dict(_target_="lightning.pytorch.loggers.CSVLogger", save_dir="/fixture-only/"+phase+"-logs",name="csv")
    cfg.logger.wandb.id = "fixture-only-no-actual-wandb-run"
    cfg.logger.wandb.resume = "never"
    if phase == "smoke":
        cfg.trainer.max_steps = cfg.trainer.val_check_interval = 50
        cfg.callbacks.model_checkpoint = None
        cfg.callbacks["joint_smoke"] = dict(_target_="egomimic.pl_utils.imagenet_smoke.ImageNetSmokeProof",
            output=str(cfg.paths.output_dir),expected_steps=50,warmup_steps=10)
        cfg.evaluator.pixel_diagnostics.log_wandb = False
        del cfg.logger["wandb"]
    plain = OmegaConf.to_container(cfg,resolve=True,throw_on_missing=True)
    contract = dict(schema="native-imagenet-dataset-v1", **{
        key:dict(path=plain["data"][key+"_path"],sha256=plain["data"][key+"_sha256"])
        for key in ("manifest","class_language","language_features","validation_anchor")})
    request = dict(kind=imagenet.KIND,phase=phase,arm="image_generation",active_streams=list(imagenet.STREAMS),
        executed_streams=["image"],expert_parameters=dict(imagenet.COUNTS),parameters=314227000,
        expert_widths=[1024,512,384,384],model_execution=dict(imagenet.POLICY),world_size=4,per_rank_batch=8,
        global_batch=32,image_generation_contract=contract,data_cache=plain["data"]["data_root"],
        data_manifest=plain["data"]["manifest_path"],data_manifest_sha256=plain["data"]["manifest_sha256"],
        lr_policy=from_config(plain["model"]),target_steps=50 if phase=="smoke" else 300000,
        output=plain["paths"]["output_dir"],local_logs=plain["logger"]["csv"]["save_dir"],
        run_id="fixture-only-no-actual-wandb-run",allocation_boundaries=[300000],
        source_manifest_sha256="2"*64,
        smoke=dict(job_id="900001",resolved=dict(sha256="7"*64)))
    return request,plain


def completion_fixture():
    request,cfg=service_fixture()
    smoke,_=service_fixture("smoke")
    context="4"*64;rows=[]
    for rank in range(4):
        rows.append(dict(schema="imagenet-joint-fit-v1",status="FIT_PASS",job_id="900001",rank=rank,
            world_size=4,global_step=50,rank_batch_size=8,global_batch_size=32,
            distributed_backend="nccl",trainer_precision="bf16-mixed",bf16_training_outputs=50,fp32_losses=50,
            validation_batches=1000,local_validation_batches=250,
            validation_batch_indices=list(range(rank*250,(rank+1)*250)),expert_parameters=dict(imagenet.COUNTS),
            data_context_sha256=context,metrics={imagenet.METRIC:.25},gpu_name="synthetic H100 fixture label",
            gpu_capability=[9,0],gpu_uuid="GPU-00000000-0000-0000-0000-"+f"{rank+1:012d}",
            fp32_gradients_verified=True,all_training_bf16_outputs=True,all50_lr_verified=True,
            inactive_banks_unchanged=True,language_conditioning_verified=True,lr_policy=request["lr_policy"],
            optimizer_update_lrs=[update_lr(request["lr_policy"],step) for step in range(50)],
            parameter_names=["synthetic_image","synthetic_unused"],
            parameter_partition={"synthetic_image":"active","synthetic_unused":"inactive"},
            parameter_update_counts={"synthetic_image":50,"synthetic_unused":0},
            inactive_initial_sha256="3"*64,inactive_final_sha256="3"*64,weights_sha256="6"*64,
            initial_weights_sha256="5"*64,all_rank_weights=["6"*64]*4,all_rank_initial_weights=["5"*64]*4,
            measured_updates=40,warm_wall_seconds=10.0+rank))
    completion=dict(schema="imagenet-native-smoke-v1",status="PASS",job_id="900001",world_size=4,global_step=50,
        rank_batch_size=8,global_batch_size=32,distributed_backend="nccl",trainer_precision="bf16-mixed",
        validation_batches=1000,local_validation_batches=250,parameters=314227000,executed_parameters=159006592,
        expert_parameters=dict(imagenet.COUNTS),active_streams=["image"],data_context_sha256=context,
        sample_state_steps=20,sample_action_steps=0,diagnostic_checkpoint_is_production_initializer=False,
        actual_denoiser_calls=20,hidden_rgb_values_all_nan=True,genuine_finite_language_retained=True,
        warm_wall_seconds=13.0,warm_wall_seconds_by_rank=[10.0,11.0,12.0,13.0],
        inactive_weights_sha256="3"*64,ranks=rows)
    for flag in ("strict_full_wrapper_reload","optimizer_sampler_all_rank_rng_verified","lr_scheduler_strictly_restored",
        "all_rank_weights_agree","all_training_bf16_outputs","fp32_stored_model_and_adam","sample_finite",
        "language_conditioning_verified","inactive_banks_unchanged","all50_lr_verified","all_rank_rng_payload_restorable"):
        completion[flag]=True
    capacity=dict(schema="imagenet-native-capacity-v1",status="PASS",job_id="900001",world_size=4,
        config_sha256="7"*64,source_manifest_sha256=request["source_manifest_sha256"],
        data_manifest_sha256=request["data_manifest_sha256"],data_context_sha256=context,rank_batch_size=8,
        global_batch_size=32,frames=1,cameras=1,patches=100,parameters=314227000,executed_parameters=159006592,
        expert_parameters=dict(imagenet.COUNTS),optimizer_updates=0,compiled_backward_fp32finite=True,
        bf16_forward=True,fp32_weights_gradients=True,inactive_gradients_absent=True,attention_backend="flash4",
        token_execution="packed_valid",compile_mode="default",compile_dynamic=True,optimizer_memory_reserve_bytes=100,
        peak_reserved_bytes=200,device_memory_bytes=400)
    capacity["ranks"]=[dict(copy.deepcopy(capacity),rank=rank,gpu_uuid=row["gpu_uuid"],
        gpu_name=row["gpu_name"],gpu_capability=row["gpu_capability"],loss=.25) for rank,row in enumerate(rows)]
    preflight=dict(provenance=dict(data_context_sha256=context))
    return request,smoke,completion,preflight,capacity


def test_genuinely_composed_full_and_smoke_are_admitted_without_data_reads():
    for phase in ("full","smoke"):
        request,cfg=service_fixture(phase)
        assert imagenet.verify_model_execution(request,cfg)=="image_generation"
    full_request,full=service_fixture()
    _,smoke=service_fixture("smoke")
    imagenet.compare_configs(full,smoke,full_request)


@pytest.mark.parametrize("phase",["full","smoke"])
@pytest.mark.parametrize("field",["limit_val_batches","limit_train_batches"])
@pytest.mark.parametrize("value",[1,True,.25])
def test_validation_full_fraction_is_type_sensitive_not_one_batch(phase,field,value):
    request,cfg=service_fixture(phase);cfg["trainer"][field]=value
    with pytest.raises(ValueError,match="Every scheduled pass"):
        imagenet.verify_model_execution(request,cfg)


@pytest.mark.parametrize("damage",["drop_bank","only_image_arch","image_width","no_qk","no_rms","no_swiglu",
    "compression","optimizer_betas","optimizer_lr","freeze","initialization","extra_loss","weighted_loss",
    "noise","sampler","language_width","no_conditional","duplicate_conditional","clip","wrong_topology",
    "wrong_train_count","wrong_val_count","preview_count","ckpt_init","continuation","interruption",
    "short_boundary","observed_context","wrong_task"])
def test_exact_full_architecture_science_and_freshness_cannot_be_changed(damage):
    request,cfg=service_fixture();pipeline=cfg["model"]["pipeline"];net=pipeline["stages"][1]["model"]
    if damage=="drop_bank": net["pose_dims"].pop("human_bimanual")
    if damage=="only_image_arch": request["active_streams"]=["image"]
    if damage=="image_width": net["image_model"]["width"]=16
    if damage=="no_qk": net["image_model"]["qk_norm"]=False
    if damage=="no_rms": net["image_model"]["use_rmsnorm"]=False
    if damage=="no_swiglu": net["image_model"]["use_swiglu"]=False
    if damage=="compression": net["image_model"]["allow_patch_compression"]=True
    if damage=="optimizer_betas": cfg["model"]["optimizer"]["betas"]=[.5,.9]
    if damage=="optimizer_lr": cfg["model"]["optimizer"]["lr"]=1e-4
    if damage=="freeze": pipeline["trainability"]={"frozen":["image"]}
    if damage=="initialization": pipeline["initialization"]={"path":"/fixture-only/smoke.ckpt"}
    if damage=="extra_loss": pipeline["stages"].append(copy.deepcopy(pipeline["stages"][2]))
    if damage=="weighted_loss": pipeline["stages"][2]["loss"]["_target_"]="different.WeightedMSE"
    if damage=="noise": pipeline["stages"][0]["noising"]["maximum"]=.5
    if damage=="sampler": pipeline["stages"][3]["sampler"]["state_steps"]=10
    if damage=="language_width": net["language_width"]=64
    if damage=="no_conditional": cfg["callbacks"].pop("conditional_usage")
    if damage=="duplicate_conditional": cfg["callbacks"]["duplicate"]=copy.deepcopy(cfg["callbacks"]["conditional_usage"])
    if damage=="clip": cfg["trainer"]["gradient_clip_val"]=1
    if damage=="wrong_topology": cfg["data"]["distributed_world_size"]=2
    if damage=="wrong_train_count": cfg["data"]["expected_train_images"]=1000
    if damage=="wrong_val_count": cfg["data"]["expected_val_images"]=1000
    if damage=="preview_count": cfg["evaluator"]["pixel_diagnostics"]["max_samples_per_source_kind"]=16
    if damage=="ckpt_init": cfg["ckpt_path"]="/fixture-only/smoke.ckpt"
    if damage=="continuation": request["continuation"]={"status":"synthetic"}
    if damage=="interruption": request["interruption"]={"status":"synthetic"}
    if damage=="short_boundary": request["allocation_boundaries"]=[60000,300000]
    if damage=="observed_context": net["image_model"]["require_observed_context"]=True
    if damage=="wrong_task": request["arm"]="continuous"
    with pytest.raises(ValueError): imagenet.verify_model_execution(request,cfg)


@pytest.mark.parametrize("damage",["save_top1","no_last_link","epoch_save","terminal_interval","warm_resume",
    "wrong_identity","extra_smoke_callback","smoke_normal_checkpoint","smoke_wandb","smoke_short","smoke_old_callback"])
def test_full_retention_and_separate_fresh_smoke_saves(damage):
    phase="smoke" if damage.startswith("smoke_") else "full"
    request,cfg=service_fixture(phase)
    checkpoint=cfg["callbacks"]["model_checkpoint"]
    if damage=="save_top1": checkpoint["save_top_k"]=1
    if damage=="no_last_link": checkpoint["save_last"]=True
    if damage=="epoch_save": checkpoint["save_on_train_epoch_end"]=True
    if damage=="terminal_interval": checkpoint["train_time_interval"]={"hours":1}
    if damage=="warm_resume": cfg["logger"]["wandb"]["resume"]="must"
    if damage=="wrong_identity": cfg["logger"]["wandb"]["id"]="other"
    if damage=="extra_smoke_callback": cfg["callbacks"]["joint_smoke"]={"_target_":"synthetic.unexpected"}
    if damage=="smoke_normal_checkpoint": cfg["callbacks"]["model_checkpoint"]={"_target_":"lightning.pytorch.callbacks.ModelCheckpoint"}
    if damage=="smoke_wandb": cfg["logger"]["wandb"]={"_target_":"lightning.pytorch.loggers.wandb.WandbLogger"}
    if damage=="smoke_short": cfg["trainer"]["max_steps"]=2
    if damage=="smoke_old_callback": cfg["callbacks"]["joint_smoke"]["_target_"]="egomimic.pl_utils.joint_smoke.JointSmokeProof"
    with pytest.raises(ValueError): imagenet.verify_model_execution(request,cfg)


@pytest.mark.parametrize("damage",["accelerator_cpu","initial_validation","duplicate_checkpoint",
    "full_aliased_smoke","smoke_aliased_checkpoint"])
def test_phase_wide_callback_and_runtime_guards_are_not_bypassed_by_aliases(damage):
    phase="smoke" if damage=="smoke_aliased_checkpoint" else "full"
    request,cfg=service_fixture(phase)
    if damage=="accelerator_cpu": cfg["trainer"]["accelerator"]="cpu"
    if damage=="initial_validation": cfg["val_at_start"]=True
    if damage=="duplicate_checkpoint": cfg["callbacks"]["other_key"]=copy.deepcopy(cfg["callbacks"]["model_checkpoint"])
    if damage=="full_aliased_smoke": cfg["callbacks"]["other_key"]=dict(
        _target_="egomimic.pl_utils.imagenet_smoke.ImageNetSmokeProof",output=request["output"],expected_steps=50,warmup_steps=10)
    if damage=="smoke_aliased_checkpoint": cfg["callbacks"]["other_key"]=dict(
        _target_="lightning.pytorch.callbacks.ModelCheckpoint",every_n_train_steps=1)
    with pytest.raises(ValueError): imagenet.verify_model_execution(request,cfg)


@pytest.mark.parametrize("field,value",[("max_epochs",1),("max_time","00:02:00:00")])
def test_matching_smoke_full_configs_cannot_hide_an_early_full_training_stop(field,value):
    request,full=service_fixture();smoke_request,smoke=service_fixture("smoke")
    full["trainer"][field]=value;smoke["trainer"][field]=value
    # Equality alone is insufficient: a50-update smoke can finish within an
    # epoch/time cap that would prevent the intended300k total full budget.
    for req,cfg in ((request,full),(smoke_request,smoke)):
        with pytest.raises(ValueError): imagenet.verify_model_execution(req,cfg)


@pytest.mark.parametrize("phase",["full","smoke"])
@pytest.mark.parametrize("damage",["peak_lr","warmup_steps","constant_without_warmup"])
def test_matching_request_and_config_cannot_change_declared_lr_policy(phase,damage):
    request,cfg=service_fixture(phase)
    if damage=="peak_lr":
        request["lr_policy"]["peak_lr"]=cfg["model"]["optimizer"]["lr"]=1e-4
    elif damage=="warmup_steps":
        request["lr_policy"]["warmup_optimizer_steps"]=1000
        cfg["model"]["scheduler"]["warmup_steps"]=1000
    else:
        request["lr_policy"]=dict(peak_lr=5e-6,warmup_optimizer_steps=0,schedule="constant")
        cfg["model"]["scheduler"]=None
    assert from_config(cfg["model"])==request["lr_policy"]
    with pytest.raises(ValueError,match="exact 5e-6/3000-warmup"):
        imagenet.verify_model_execution(request,cfg)


@pytest.mark.parametrize("phase",["full","smoke"])
@pytest.mark.parametrize("field,value",[("fast_dev_run",True),("overfit_batches",.25),
    ("barebones",True),("enable_checkpointing",False)])
def test_trainer_shortcuts_cannot_replace_actual_training_or_validation(phase,field,value):
    request,cfg=service_fixture(phase);cfg["trainer"][field]=value
    with pytest.raises(ValueError,match="DDP/precision/clipping/runtime"):
        imagenet.verify_model_execution(request,cfg)


@pytest.mark.parametrize("phase",["full","smoke"])
def test_unknown_callback_target_cannot_bypass_phase_guards(phase):
    request,cfg=service_fixture(phase)
    cfg["callbacks"]["unreviewed"]={"_target_":"fixture_only.TrainingMutator"}
    with pytest.raises(ValueError,match="Unreviewed callbacks"):
        imagenet.verify_model_execution(request,cfg)


def test_checkpoint_name_formatter_must_retain_metric_names():
    request,cfg=service_fixture()
    cfg["callbacks"]["model_checkpoint"]["auto_insert_metric_name"]=False
    with pytest.raises(ValueError,match="all15 immutable"):
        imagenet.verify_model_execution(request,cfg)


@pytest.mark.parametrize("damage",["context_anchors","raw_manifest","full_val_count","preview_count","noise","loss","topology","rank_batch","seed"])
def test_same_complete_context_required_in_smoke_and_full(damage):
    request,full=service_fixture();_,smoke=service_fixture("smoke")
    if damage=="context_anchors": smoke["data"]["validation_anchor_sha256"]="f"*64
    if damage=="raw_manifest": smoke["data"]["manifest_sha256"]="f"*64
    if damage=="full_val_count": smoke["data"]["expected_val_images"]=1000
    if damage=="preview_count": smoke["evaluator"]["pixel_diagnostics"]["max_samples_per_source_kind"]=16
    if damage=="noise": smoke["model"]["pipeline"]["stages"][0]["noising"]["maximum"]=.5
    if damage=="loss": smoke["model"]["pipeline"]["stages"][2]["weight"]=.5
    if damage=="topology": smoke["data"]["distributed_world_size"]=2
    if damage=="rank_batch": smoke["data"]["batch_size"]=4
    if damage=="seed": smoke["seed"]=43
    with pytest.raises(ValueError): imagenet.compare_configs(full,smoke,request)


def test_unknown_kind_still_rejected_by_legacy_selector():
    from scripts.train.joint import contract,native_subtasks
    request,cfg=service_fixture();request["kind"]="unknown-optional-request-v999"
    assert not imagenet.selected(request)
    with pytest.raises(ValueError,match="Unknown optional"):
        native_subtasks.selected(request)
    with pytest.raises(ValueError,match="Unknown optional"):
        contract.verify_model_execution(request,cfg)


def test_completion_unit_uses_dataset_gate_and_exact_thousand_anchors(monkeypatch):
    evidence=completion_fixture();calls=[]
    monkeypatch.setattr(imagenet,"verify_dataset",lambda request,**kwargs:calls.append(request))
    completion,preflight=imagenet.verify_completion(*evidence)
    assert completion is evidence[2] and preflight is evidence[3] and calls==[evidence[0]]


@pytest.mark.parametrize("damage",["old100","different_context","false_reload","nan_metric","wrong_lr","missing_lr",
    "inactive_updates","inactive_changed","same_gpu","missing_anchor","diagnostic_init","one_step_sample",
    "wrong_capacity_source","capacity_optimizer_update","capacity_c64","capacity_no_reserve","wrong_source_semantics"])
def test_completion_unit_rejects_missing_or_mismatched_producer_evidence(monkeypatch,damage):
    request,smoke,done,preflight,capacity=completion_fixture()
    monkeypatch.setattr(imagenet,"verify_dataset",lambda request,**kwargs:None)
    if damage=="old100": done["validation_batches"]=100
    if damage=="different_context": done["data_context_sha256"]="f"*64
    if damage=="false_reload": done["strict_full_wrapper_reload"]=False
    if damage=="nan_metric": done["ranks"][0]["metrics"][imagenet.METRIC]=float("nan")
    if damage=="wrong_lr": done["ranks"][0]["optimizer_update_lrs"][0]=1e-4
    if damage=="missing_lr": done["ranks"][0]["optimizer_update_lrs"].pop()
    if damage=="inactive_updates": done["ranks"][0]["parameter_update_counts"]["synthetic_unused"]=1
    if damage=="inactive_changed": done["ranks"][0]["inactive_final_sha256"]="f"*64
    if damage=="same_gpu": done["ranks"][1]["gpu_uuid"]=done["ranks"][0]["gpu_uuid"]
    if damage=="missing_anchor": done["ranks"][0]["validation_batch_indices"].pop()
    if damage=="diagnostic_init": done["diagnostic_checkpoint_is_production_initializer"]=True
    if damage=="one_step_sample": done["sample_state_steps"]=1
    if damage=="wrong_capacity_source": capacity["source_manifest_sha256"]="f"*64
    if damage=="capacity_optimizer_update": capacity["optimizer_updates"]=1
    if damage=="capacity_c64": capacity["frames"]=64
    if damage=="capacity_no_reserve": capacity["optimizer_memory_reserve_bytes"]=0
    if damage=="wrong_source_semantics": smoke["image_generation_contract"]["manifest"]["sha256"]="f"*64
    with pytest.raises(ValueError): imagenet.verify_completion(request,smoke,done,preflight,capacity)


@pytest.mark.parametrize("scope",["summary","rank"])
@pytest.mark.parametrize("field,value",[("rank_batch_size",4),("global_batch_size",16),
    ("distributed_backend","gloo"),("trainer_precision","32-true")])
def test_completion_requires_real_batch_backend_and_precision_at_both_levels(monkeypatch,scope,field,value):
    evidence=completion_fixture();done=evidence[2]
    monkeypatch.setattr(imagenet,"verify_dataset",lambda request,**kwargs:None)
    target=done if scope=="summary" else done["ranks"][3]
    target[field]=value
    with pytest.raises(ValueError):imagenet.verify_completion(*evidence)


@pytest.mark.parametrize("field,value",[("schema","other-capacity-v1"),("status","FAILED"),
    ("job_id","900002"),("rank",0),("world_size",2),("rank_batch_size",4),("global_batch_size",16),
    ("frames",64),("cameras",3),("patches",99),("source_manifest_sha256","f"*64),
    ("data_manifest_sha256","f"*64),("config_sha256","f"*64),("data_context_sha256","f"*64),
    ("compiled_backward_fp32finite",False),("bf16_forward",False),("fp32_weights_gradients",False),
    ("inactive_gradients_absent",False),("optimizer_updates",1),("optimizer_memory_reserve_bytes",0),
    ("peak_reserved_bytes",0),("device_memory_bytes",100),("loss",float("nan")),
    ("gpu_uuid","GPU-00000000-0000-0000-0000-000000000005")])
def test_each_capacity_rank_must_supply_its_own_complete_proof(monkeypatch,field,value):
    evidence=completion_fixture();capacity=evidence[4]
    monkeypatch.setattr(imagenet,"verify_dataset",lambda request,**kwargs:None)
    # Mutate the final rank only: an intact aggregate and rank0 are insufficient.
    capacity["ranks"][3][field]=value
    with pytest.raises(ValueError):imagenet.verify_completion(*evidence)


@pytest.mark.parametrize("field,value",[("actual_denoiser_calls",1),("hidden_rgb_values_all_nan",False),
    ("genuine_finite_language_retained",False)])
def test_completion_requires_actual_hidden_rgb_sample_and_genuine_language(monkeypatch,field,value):
    evidence=completion_fixture();evidence[2][field]=value
    monkeypatch.setattr(imagenet,"verify_dataset",lambda request,**kwargs:None)
    with pytest.raises(ValueError):imagenet.verify_completion(*evidence)


@pytest.mark.parametrize("damage",["aggregate_not_max","rank_times_reordered","rank_time_nan","rank_time_zero"])
def test_reported_training_time_must_equal_actual_ordered_rank_measurements(monkeypatch,damage):
    evidence=completion_fixture();done=evidence[2]
    monkeypatch.setattr(imagenet,"verify_dataset",lambda request,**kwargs:None)
    if damage=="aggregate_not_max":done["warm_wall_seconds"]=sum(done["warm_wall_seconds_by_rank"])
    if damage=="rank_times_reordered":done["warm_wall_seconds_by_rank"].reverse()
    if damage=="rank_time_nan":done["ranks"][3]["warm_wall_seconds"]=float("nan")
    if damage=="rank_time_zero":done["ranks"][3]["warm_wall_seconds"]=0.0
    with pytest.raises(ValueError):imagenet.verify_completion(*evidence)


def test_pure_service_import_and_contract_execute_without_torch_in_child():
    request,cfg=service_fixture()
    code='''import importlib.abc,json,sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=="torch" or fullname.startswith("torch."):
            raise AssertionError("Pure service imported Torch")
sys.meta_path.insert(0,NoTorch())
from scripts.train.joint import imagenet
request,cfg,smoke_cfg,evidence=json.loads(sys.stdin.read())
assert imagenet.verify_model_execution(request,cfg)=="image_generation"
imagenet.compare_configs(cfg,smoke_cfg,request)
# Only the pure completion interpretation is in scope in this child. Native
# source/cache producer closure has separate tests and must not import Torch
# on a login node merely to import this standard-library service.
imagenet.verify_dataset=lambda request,**kwargs:None
imagenet.verify_completion(*evidence)
assert "torch" not in sys.modules
print("PASS_STDLIB_CONTRACT_ONLY")
'''
    source=Path(imagenet.__file__).resolve().parents[3]
    _,smoke_cfg=service_fixture("smoke")
    result=subprocess.run([sys.executable,"-c",code],input=json.dumps([request,cfg,smoke_cfg,completion_fixture()]),text=True,
        capture_output=True,env=dict(os.environ,PYTHONPATH=str(source)),cwd=source)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=="PASS_STDLIB_CONTRACT_ONLY"


def cpu_request_fixture():
    # Exact relevant request/record shape from the immutable CPU preparation
    # packet's test_preparation.py native() fixture. These are never submitted.
    return dict(root="/coc/flash7/paphiwetsa3/tasks/elmo-qknorm-swiglu-rmsnorm-imagenet-20261026-v1/imagenet-cpu-v1",
        uid=3346154,account="rl2-lab",partition="rl2-lab",qos="short",node="sky1",
        resources=dict(cpus=4,gpus=0,memory_mib=24576,time_limit="04:00:00"),
        capture_resources=dict(cpus=2,gpus=0,memory_mib=1024,time_limit="00:10:00"),
        native=dict(sbatch="/opt/slurm/Ubuntu-24.04/current/bin/sbatch"))


def cpu_native_fixture(q,*,job="123",state="PENDING",observer=False,parent="123"):
    r=q["capture_resources"] if observer else q["resources"];cpus=r["cpus"];mem=str(r["memory_mib"]//1024)+"G"
    fields=dict(JobId=job,JobName="elmo-native-imagenet-cpu-v1" if observer else "elmo-imagenet-cpu-preparation-v1",
        UserId="paphiwetsa3(3346154)",Account=q["account"],Partition=q["partition"],QOS=q["qos"],ReqNodeList=q["node"],
        NumNodes="1-1" if state=="PENDING" else "1",NumTasks="1",NumCPUs=str(cpus),MinMemoryNode=mem,
        TimeLimit=r["time_limit"],Requeue="0",Restarts="0",JobState=state,
        Command=str(Path(q["root"])/("capture.sbatch" if observer else "producer.sbatch")),WorkDir=q["root"],
        Comment="imagenet-cpu-observer:"+"a"*64+":"+parent if observer else "imagenet-cpu:"+"a"*64,
        Reason="Dependency" if observer else "JobHeldUser",Dependency="afterany:"+parent+"(unfulfilled)" if observer else "(null)",
        NodeList=q["node"] if state!="PENDING" else "(null)",StartTime="2026-10-26T23:00:00" if state!="PENDING" else "Unknown",
        EndTime="2026-10-26T23:30:00" if state=="COMPLETED" else "Unknown",SubmitTime="2026-10-26T22:59:00",ExitCode="0:0",
        ReqTRES=f"cpu={cpus},mem={mem},node=1,billing={cpus}",
        AllocTRES=f"cpu={cpus},mem={mem},node=1,billing={cpus}" if state!="PENDING" else "",TresPerTask="cpu:"+str(cpus))
    return " ".join(key+"="+value for key,value in fields.items())


@pytest.mark.parametrize("state",["PENDING","RUNNING","COMPLETED"])
@pytest.mark.parametrize("observer",[False,True])
def test_exact_original_cpu_guard_native_shapes_are_compatible(state,observer):
    q=cpu_request_fixture();raw=cpu_native_fixture(q,state=state,observer=observer)
    actual=imagenet._cpu_native(raw,q,"123",state,"a"*64,observer=observer,parent="123")
    assert actual["JobState"]==state and actual["NumCPUs"]==("2" if observer else "4")
    assert actual["NumNodes"]==("1-1" if state=="PENDING" else "1")


@pytest.mark.parametrize("old,new",[("NumCPUs=4","NumCPUs=8"),("24G","25G"),("Restarts=0","Restarts=1"),
    ("Requeue=0","Requeue=1"),("04:00:00","04:00:01"),("paphiwetsa3(3346154)","paphiwetsa3(41003)"),
    ("JobHeldUser","Priority"),("SubmitTime=2026-10-26T22:59:00","SubmitTime=Unknown"),
    ("NumTasks=1","NumTasks=2"),("Partition=rl2-lab","Partition=other"),("QOS=short","QOS=long"),
    ("WorkDir=/coc/flash7","WorkDir=/tmp"),("ReqNodeList=sky1","ReqNodeList=sky2")])
def test_cpu_native_reader_rejects_changed_owner_resources_or_acceptance(old,new):
    q=cpu_request_fixture();raw=cpu_native_fixture(q).replace(old,new)
    with pytest.raises(ValueError): imagenet._cpu_native(raw,q,"123","PENDING","a"*64)


@pytest.mark.parametrize("suffix",[" JOB_GRES=gpu:h100:1"," Gres=gpu:a40:1"," TresPerNode=gres/gpu:1"," JobId=124"])
def test_cpu_native_reader_rejects_gpu_fields_or_duplicate_native_keys(suffix):
    q=cpu_request_fixture()
    with pytest.raises(ValueError): imagenet._cpu_native(cpu_native_fixture(q)+suffix,q,"123","PENDING","a"*64)


@pytest.mark.parametrize("damage",["gpu_tres","missing_tres","wrong_dependency","wrong_observer_comment",
    "missing_start","bad_exit","missing_end","end_before_start","allocated_range"])
def test_cpu_native_reader_rejects_unknown_or_failed_running_terminal_proof(damage):
    q=cpu_request_fixture();observer=damage in ("wrong_dependency","wrong_observer_comment")
    state="PENDING" if observer or damage in ("gpu_tres","missing_tres") else "COMPLETED"
    raw=cpu_native_fixture(q,state=state,observer=observer)
    if damage=="gpu_tres": raw=raw.replace("ReqTRES=cpu=4,","ReqTRES=gres/gpu=1,cpu=4,")
    if damage=="missing_tres": raw=raw.replace("ReqTRES=cpu=4,mem=24G,node=1,billing=4","ReqTRES=cpu=4")
    if damage=="wrong_dependency": raw=raw.replace("afterany:123","afterany:124")
    if damage=="wrong_observer_comment": raw=raw.replace("imagenet-cpu-observer:","imagenet-cpu:")
    if damage=="missing_start": raw=raw.replace("StartTime=2026-10-26T23:00:00","StartTime=Unknown")
    if damage=="bad_exit": raw=raw.replace("ExitCode=0:0","ExitCode=0:9")
    if damage=="missing_end": raw=raw.replace("EndTime=2026-10-26T23:30:00","EndTime=Unknown")
    if damage=="end_before_start": raw=raw.replace("EndTime=2026-10-26T23:30:00","EndTime=2026-10-25T23:30:00")
    if damage=="allocated_range": raw=raw.replace("NumNodes=1 ","NumNodes=1-1 ")
    with pytest.raises(ValueError): imagenet._cpu_native(raw,q,"123",state,"a"*64,observer=observer,parent="123")


@pytest.mark.parametrize("parent",[None,"123"])
def test_cpu_submission_argv_matches_actual_guard_cpu_only_single_use_protocol(parent):
    q=cpu_request_fixture();argv=imagenet._cpu_argv(q,"a"*64,"b"*64,parent=parent)
    assert argv[0]==q["native"]["sbatch"] and argv[-3:]==[q["root"],"a"*64,"b"*64]
    assert "--parsable" in argv and "--no-requeue" in argv and "--gres=none" in argv and "--export=NONE" in argv
    assert ("--hold" in argv)==(parent is None)
    assert ("--dependency=afterany:123" in argv)==(parent is not None)
    assert ("--cpus-per-task=2" if parent else "--cpus-per-task=4") in argv
    assert ("--time=00:10:00" if parent else "--time=04:00:00") in argv
    assert ("--mem=1024M" if parent else "--mem=24576M") in argv
    assert argv[-4]==str(Path(q["root"])/("capture.sbatch" if parent else "producer.sbatch"))


def test_bounded_manifest_accepts_exactly_128_unique_relative_members(tmp_path):
    path=tmp_path/"fixture-manifest.sha256"
    expected={f"files/member-{index}.py":"a"*64 for index in range(128)}
    path.write_text("".join(f"{digest}  {name}\n" for name,digest in expected.items()))
    assert imagenet._manifest_entries(path)==expected


@pytest.mark.parametrize("damage",["duplicate","traversal","absolute","bad_digest","one_space","empty",
    "too_many","symlink","oversized"])
def test_bounded_manifest_rejects_unsafe_or_ambiguous_members(tmp_path,damage):
    path=tmp_path/"fixture-manifest.sha256";line="a"*64+"  files/one.py\n"
    if damage=="duplicate": line+=line
    if damage=="traversal": line="a"*64+"  files/../one.py\n"
    if damage=="absolute": line="a"*64+"  /absolute/one.py\n"
    if damage=="bad_digest": line="z"*64+"  files/one.py\n"
    if damage=="one_space": line="a"*64+" files/one.py\n"
    if damage=="empty": line=""
    if damage=="too_many": line="".join("a"*64+f"  files/{index}.py\n" for index in range(129))
    path.write_text(line)
    if damage=="symlink":
        link=tmp_path/"fixture-link.sha256";link.symlink_to(path);path=link
    if damage=="oversized":
        with path.open("wb") as stream: stream.truncate((4<<20)+1)
    with pytest.raises(ValueError): imagenet._manifest_entries(path)


def bound_metadata_fixture(tmp_path,text='{"fixture_only":true}'):
    path=tmp_path/"fixture-metadata.json";path.write_text(text)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_bound_small_metadata_accepts_exact_bytes_without_ml_imports(tmp_path):
    binding=bound_metadata_fixture(tmp_path)
    assert imagenet._bounded_bound(binding)=={"fixture_only":True}


@pytest.mark.parametrize("text",['{"a":1,"a":2}','{"a":{"nested":1,"nested":2}}',
    '{"a":NaN}','{"a":Infinity}','{"a":-Infinity}'])
def test_bound_metadata_rejects_duplicate_keys_and_nonfinite_json(tmp_path,text):
    binding=bound_metadata_fixture(tmp_path,text)
    with pytest.raises(ValueError): imagenet._bounded_bound(binding)


@pytest.mark.parametrize("damage",["bad_hash","changed_bytes","relative","traversal","symlink","oversized"])
def test_bound_metadata_rejects_unbound_or_relocated_or_oversized_files(tmp_path,damage):
    binding=bound_metadata_fixture(tmp_path);path=Path(binding["path"])
    if damage=="bad_hash": binding["sha256"]="z"*64
    if damage=="changed_bytes": path.write_text('{"fixture_only":false}')
    if damage=="relative": binding["path"]=path.name
    if damage=="traversal":
        (tmp_path/"child").mkdir();binding["path"]=str(tmp_path/"child"/".."/path.name)
    if damage=="symlink":
        link=tmp_path/"fixture-metadata-link.json";link.symlink_to(path);binding["path"]=str(link)
    if damage=="oversized":
        with path.open("wb") as stream: stream.truncate((4<<20)+1)
    with pytest.raises(ValueError): imagenet._bounded_bound(binding)
