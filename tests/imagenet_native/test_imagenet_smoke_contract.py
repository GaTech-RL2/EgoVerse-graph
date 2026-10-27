"""Pure CPU smoke-admission fixtures; never actual CUDA/producer evidence."""

import copy
import hashlib
import random
import sys
from pathlib import Path

import hydra
from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from egomimic.pl_utils.imagenet_smoke import (
    EXECUTED_PARAMETERS, METRIC, TOTAL_PARAMETERS, ImageNetSmokeProof,
    _gpu_runtime, _validate_sample, check_parameter_updates, partition_parameters,
    validate_fit_rows, validate_imagenet_capacity, validate_imagenet_checkpoint,
    validate_imagenet_config, validate_rng_payload, verify_imagenet_smoke,
)
from egomimic.pl_utils.joint_smoke import (
    ConditionalOptimizerUsage, _optimizer_capacity_reserve_bytes,
)
from scripts.train.joint.lr_policy import from_config, update_lr
from scripts.train.joint.native_recovery import conditional_record


@pytest.fixture(autouse=True)
def forbid_cuda_in_cpu_fixtures(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU fixture attempted a CUDA call")
    for name in ("is_available", "device_count", "current_device", "set_device", "get_device_properties",
                 "get_device_capability", "synchronize", "mem_get_info", "get_rng_state", "set_rng_state"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)


def bound_config(*, smoke=True):
    source = Path(__import__("egomimic").__file__).parent
    with initialize_config_dir(version_base=None, config_dir=str(source / "hydra_configs")):
        cfg = compose(config_name="train_zarr_cartesian", overrides=[
            "+experiment=pixel_proprio/native_joint/imagenet_image_generation",
            "paths.output_dir=/fixture-only-imagenet-no-actual-data",
        ])
    OmegaConf.set_struct(cfg, False)
    for key in ("manifest_path", "data_root", "class_language_path", "language_features_path", "validation_anchor_path"):
        cfg.data[key] = "/fixture-only/" + key
    for key in ("manifest_sha256", "class_language_sha256", "language_features_sha256", "validation_anchor_sha256"):
        cfg.data[key] = "1" * 64
    cfg.callbacks["conditional_optimizer_usage"] = dict(_target_="egomimic.pl_utils.joint_smoke.ConditionalOptimizerUsage")
    if smoke:
        cfg.trainer.max_steps = cfg.trainer.val_check_interval = 50
        cfg.trainer.limit_val_batches = 250
        cfg.callbacks.model_checkpoint = None
        cfg.callbacks["joint_smoke"] = dict(_target_="egomimic.pl_utils.imagenet_smoke.ImageNetSmokeProof",
                                            output=cfg.paths.output_dir, expected_steps=50, warmup_steps=10)
    return cfg


def fit_rows():
    policy = from_config(bound_config().model)
    rows = []
    for rank in range(4):
        rows.append(dict(schema="imagenet-joint-fit-v1", status="FIT_PASS", job_id="900001", rank=rank,
            world_size=4, global_step=50, validation_batches=1000, local_validation_batches=250,
            rank_batch_size=8, global_batch_size=32, distributed_backend="nccl", trainer_precision="bf16-mixed",
            validation_batch_indices=list(range(rank*250, (rank+1)*250)),
            parameters=TOTAL_PARAMETERS, executed_parameters=EXECUTED_PARAMETERS,
            parameter_names=["image", "unused"], parameter_partition={"image":"active", "unused":"inactive"},
            parameter_update_counts={"image":50,"unused":0},
            expert_parameters={"synthetic_image":EXECUTED_PARAMETERS,
                               "synthetic_unused":TOTAL_PARAMETERS-EXECUTED_PARAMETERS},
            bf16_training_outputs=50, fp32_losses=50, fp32_gradients_verified=True,
            all_training_bf16_outputs=True, all50_lr_verified=True,
            optimizer_update_lrs=[update_lr(policy, step) for step in range(50)], lr_policy=policy,
            inactive_banks_unchanged=True, language_conditioning_verified=True,
            inactive_initial_sha256="3"*64, inactive_final_sha256="3"*64,
            weights_sha256="2"*64, initial_weights_sha256="1"*64,
            all_rank_weights=["2"*64]*4, all_rank_initial_weights=["1"*64]*4,
            data_context_sha256="4"*64, metrics={METRIC: .25}, active_streams=["image"],
            attention_backend="flash4", token_execution="packed_valid", compile_mode="default",
            gpu_name="synthetic fixture H100 label; not actual hardware", gpu_capability=[9,0],
            gpu_uuid="synthetic-only-"+str(rank), measured_updates=40, warm_wall_seconds=12.0+rank,
            peak_allocated_bytes=100, peak_reserved_bytes=200, validation_checkpoint_endcheck_seconds=30.0,
            checkpoint_name="smoke-epoch0000-step000000050.ckpt"))
    return rows, policy


def test_config_exact_smoke_and_full_defaults_are_distinct():
    assert validate_imagenet_config(bound_config())["warmup_optimizer_steps"] == 3000
    assert validate_imagenet_config(bound_config(smoke=False), smoke=False)["peak_lr"] == 5e-6
    with pytest.raises(ValueError): validate_imagenet_config(bound_config(smoke=False))
    with pytest.raises(ValueError): validate_imagenet_config(bound_config(), smoke=False)


@pytest.mark.parametrize("path,value", [
    ("seed", 43), ("data.expected_train_images", 1000), ("data.expected_val_images", 1000),
    ("data.expected_classes", 10), ("data.batch_size", 4), ("data.validation_batch_size", 2),
    ("data.distributed_world_size", 2), ("data.manifest_sha256", "UNBOUND"),
    ("data.validation_anchor_sha256", ""), ("data.data_root", "relative"),
    ("trainer.devices", 2), ("trainer.precision", "32-true"), ("trainer.accelerator", "cpu"),
    ("trainer.accumulate_grad_batches", 2), ("trainer.num_sanity_val_steps", 1),
    ("trainer.gradient_clip_val", 1), ("trainer.limit_val_batches", 25),
    ("trainer.val_check_interval", 10), ("trainer.max_steps", 49),
    ("trainer.strategy.find_unused_parameters", False), ("evaluator.objective", "visual_only"),
    ("evaluator.active_streams_from_data", True), ("model.optimizer.lr", 1e-4),
    ("model.optimizer.weight_decay", .1), ("model.scheduler.warmup_steps", 0),
    ("model.optimizer.betas", [.5,.99]),
    ("model.pipeline.initialization", {"path":"/fixture-only/not-a-fresh-run.ckpt"}),
    ("model.pipeline.trainability", {"frozen":["image"]}),
    ("model.pipeline.stages.1.model.image_model.width", 16),
    ("model.pipeline.stages.1.model.image_model.allow_patch_compression", True),
    ("model.pipeline.stages.1.model.image_model.attention_backend", "sdpa"),
    ("model.pipeline.stages.1.model.image_model.compile_mode", "none"),
    ("model.pipeline.stages.1.model.image_model.token_execution", "padded"),
    ("model.pipeline.stages.1.model.image_model.qk_norm", False),
    ("model.pipeline.stages.1.model.image_model.use_swiglu", False),
    ("model.pipeline.stages.1.model.image_model.use_rmsnorm", False),
    ("model.pipeline.stages.1.model.language_feature_dim", 128),
    ("model.pipeline.stages.3.sampler.state_steps", 10), ("val_at_end", True),
    ("callbacks.conditional_optimizer_usage", None),
])
def test_changed_science_or_execution_config_rejected(path, value):
    cfg = bound_config()
    OmegaConf.update(cfg, path, value)
    with pytest.raises(ValueError): validate_imagenet_config(cfg)


def test_full_model_meta_partition_includes_every_inactive_interface():
    cfg = bound_config()
    with torch.device("meta"):
        network = hydra.utils.instantiate(cfg.model.pipeline.stages[1].model)
    partition = partition_parameters(network, network)
    assert sum(p.numel() for p in network.parameters()) == TOTAL_PARAMETERS
    assert sum(p.numel() for n,p in network.named_parameters() if partition[n]=="active") == EXECUTED_PARAMETERS
    assert partition["camera_identity.weight"] == "inactive"
    assert all(partition[n] == "inactive" for n in partition if n.startswith(("proprio_banks.", "action.", "camera.",
        "language_fusions.proprio__", "language_fusions.action.", "language_fusions.camera.")))
    # Meta tensor storage is declared FP32, so this is a byte arithmetic test,
    # not proof that the memory reserve was actually allocated on any GPU.
    assert _optimizer_capacity_reserve_bytes(network.parameters()) == TOTAL_PARAMETERS * 4 * 3


def test_pure_all_four_rank_rows_accept_exact_image_contract():
    rows, policy = fit_rows()
    assert validate_fit_rows(rows, policy, "900001") is rows[0]


@pytest.mark.parametrize("damage", ["old100", "rank", "job", "same_gpu", "nan_metric", "wrong_lr", "missing_lr",
    "unused_updates", "inactive_digest", "wrong_weights", "missing_anchor", "duplicate_anchor", "old_namespace",
    "false_precision", "wrong_count", "wrong_experts", "wrong_checkpoint", "missing_compile", "no_timing"])
def test_pure_rank_proof_rejects_historical_or_incomplete_evidence(damage):
    rows, policy = fit_rows(); row = rows[1]
    if damage == "old100": row["validation_batches"] = 100
    if damage == "rank": row["rank"] = 0
    if damage == "job": row["job_id"] = "another"
    if damage == "same_gpu": row["gpu_uuid"] = rows[0]["gpu_uuid"]
    if damage == "nan_metric": row["metrics"][METRIC] = float("nan")
    if damage == "wrong_lr": row["optimizer_update_lrs"][49] = 1e-4
    if damage == "missing_lr": row["optimizer_update_lrs"].pop()
    if damage == "unused_updates": row["parameter_update_counts"]["unused"] = 1
    if damage == "inactive_digest": row["inactive_final_sha256"] = "f"*64
    if damage == "wrong_weights": row["weights_sha256"] = "f"*64
    if damage == "missing_anchor": row["validation_batch_indices"].pop()
    if damage == "duplicate_anchor": row["validation_batch_indices"][0] = 0
    if damage == "old_namespace": row["metrics"] = {"val/imagenet/forecast/image_normalized_mse":.1}
    if damage == "false_precision": row["all_training_bf16_outputs"] = False
    if damage == "wrong_count": row["parameters"] -= 1
    if damage == "wrong_experts": row["expert_parameters"]["synthetic_unused"] -= 1
    if damage == "wrong_checkpoint": row["checkpoint_name"] = "last.ckpt"
    if damage == "missing_compile": row["compile_mode"] = "none"
    if damage == "no_timing": row["warm_wall_seconds"] = 0
    with pytest.raises(ValueError): validate_fit_rows(rows, policy, "900001")


def rng_payload():
    generator = torch.Generator(device="cpu").manual_seed(7)
    ranks = [dict(torch_cpu=generator.get_state(), torch_cuda=[torch.arange(8,dtype=torch.uint8)],
                  numpy=np.random.RandomState(rank).get_state(), python=random.Random(rank).getstate(),
                  cuda_scope="current_device") for rank in range(4)]
    return dict(schema="mot_rng_state.distributed.v1", world_size=4, ranks=ranks)


def test_all_rank_rng_schema_is_validated_without_cuda_or_global_rng_change():
    payload = rng_payload()
    before = (torch.get_rng_state().clone(), np.random.get_state(), random.getstate())
    assert validate_rng_payload(payload) == payload["ranks"]
    assert torch.equal(before[0], torch.get_rng_state())
    assert np.array_equal(before[1][1], np.random.get_state()[1]) and before[2] == random.getstate()


@pytest.mark.parametrize("damage", ["old_schema", "missing_rank", "no_cuda", "cuda_scope", "bad_cpu", "bad_numpy", "bad_python"])
def test_missing_or_corrupt_all_rank_rng_payload_rejected(damage):
    payload = rng_payload()
    if damage == "old_schema": payload["schema"] = "single"
    if damage == "missing_rank": payload["ranks"].pop()
    if damage == "no_cuda": payload["ranks"][1]["torch_cuda"] = []
    if damage == "cuda_scope": payload["ranks"][1]["cuda_scope"] = "all_devices"
    if damage == "bad_cpu": payload["ranks"][1]["torch_cpu"] = torch.ones(4)
    if damage == "bad_numpy": payload["ranks"][1]["numpy"] = ("not_rng",)
    if damage == "bad_python": payload["ranks"][1]["python"] = ("not_rng",)
    with pytest.raises((ValueError, TypeError, RuntimeError)): validate_rng_payload(payload)


def test_actual_conditional_optimizer_keeps_unused_parameter_unchanged_and_stateless():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.image = torch.nn.Parameter(torch.ones(2)); self.unused = torch.nn.Parameter(torch.ones(3))
    model = Tiny(); opt = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=.01)
    callback = ConditionalOptimizerUsage()
    trainer = type("FixtureTrainer", (), dict(global_step=0, optimizers=[opt]))()
    callback.on_fit_start(trainer, model); inactive = model.unused.detach().clone()
    for step in range(3):
        opt.zero_grad(set_to_none=True); model.image.square().sum().backward()
        callback.on_before_optimizer_step(trainer, model, opt); opt.step(); trainer.global_step += 1
    record = conditional_record(opt, model.named_parameters(), callback.counts, 3)
    assert record["parameter_update_counts"] == {"image":3,"unused":0}
    check_parameter_updates(record["parameter_names"], {"image":"active","unused":"inactive"},
                            record["parameter_update_counts"], expected_step=3)
    assert torch.equal(inactive, model.unused) and opt.state.get(model.unused) is None
    clone = Tiny(); restored = torch.optim.AdamW(clone.parameters(), lr=5e-6)
    restored.load_state_dict(opt.state_dict())
    assert conditional_record(restored, clone.named_parameters(), callback.counts, 3) == record
    restored.state[clone.unused] = copy.deepcopy(restored.state[clone.image])
    with pytest.raises(ValueError): conditional_record(restored, clone.named_parameters(), callback.counts, 3)


def test_no_allocation_fails_before_any_cuda_access(tmp_path):
    with pytest.raises(ValueError, match="actual allocated"):
        _gpu_runtime(torch.device("cuda",0), distributed=False)
    with pytest.raises(ValueError, match="actual allocated"):
        verify_imagenet_smoke(tmp_path)
    with pytest.raises(ValueError, match="actual allocated"):
        validate_imagenet_checkpoint({}, {}, {}, expected_step=50, expected_context_sha256="1"*64)
    with pytest.raises(ValueError, match="actual allocated"):
        validate_imagenet_capacity(tmp_path/"unbound.yaml", tmp_path/"unbound.json",
                                   source_manifest_sha256="1"*64,data_manifest_sha256="2"*64)
    assert not list(tmp_path.iterdir())


def test_smoke_lengths_are_not_configurable_downward(tmp_path):
    ImageNetSmokeProof(tmp_path)
    for kwargs in (dict(expected_steps=2),dict(warmup_steps=1),dict(expected_steps=True)):
        with pytest.raises(ValueError): ImageNetSmokeProof(tmp_path, **kwargs)


def sample_fixture(batch_size=1, *, hidden=False):
    coordinate = (torch.arange(10,dtype=torch.float32)+.5)/5-1
    y,x = torch.meshgrid(coordinate,coordinate,indexing="ij")
    features = torch.stack([torch.full((1024,),1/32,dtype=torch.float32) * (-1 if row%2 else 1)
                            for row in range(batch_size)])
    metadata = [dict(sample_id=f"val/n{row+1:08d}/image.png", wnid=f"n{row+1:08d}",
        class_name=f"synthetic class{row}", source_sha256="1"*64,
        class_feature_sha256=hashlib.sha256(features[row].numpy().astype("<f4").tobytes()).hexdigest(),
        frame_indices=[0],timestamps_seconds=[0.0],time_origin_seconds=0.0,camera_names=["image"],
        image_size=[160,160],patch_size=16,normalization="uint8_to_minus_one_one",patch_order="row_major_hwc")
        for row in range(batch_size)]
    stream=dict(value=torch.full((batch_size,100,768),float("nan") if hidden else 0,dtype=torch.float32),
        valid=torch.ones(batch_size,100,dtype=torch.bool),target_valid=torch.ones(batch_size,100,dtype=torch.bool),
        observed=torch.zeros(batch_size,100,dtype=torch.bool),time=torch.zeros(batch_size,100),
        index=torch.zeros(batch_size,100),camera=torch.zeros(batch_size,100,dtype=torch.long),
        xy=torch.stack((x,y),-1).reshape(1,100,2).expand(batch_size,-1,-1).clone())
    return dict(embodiment="imagenet",world={"image":stream},language_features=features,
        pixel_metadata=metadata,task_id=[m["wnid"] for m in metadata],
        language_instruction=[m["class_name"] for m in metadata],task_kind=["image_generation"]*batch_size)


def test_hidden_sample_retains_real_language_but_removes_every_rgb_target():
    clean=sample_fixture(2)
    _validate_sample(clean,batch_size=2)
    hidden=sample_fixture(2,hidden=True)
    assert torch.equal(hidden["language_features"],clean["language_features"])
    _validate_sample(hidden,batch_size=2,hidden=True)
    with pytest.raises(ValueError): _validate_sample(hidden,batch_size=2)


@pytest.mark.parametrize("damage", ["partial_hidden", "observed", "target_mask", "time", "second_camera", "extra_stream",
    "language_nan", "language_swapped", "task_swapped", "text_swapped", "feature_hash", "forecast", "patch_geometry"])
def test_hidden_sample_rejects_leakage_or_wrong_row_language_provenance(damage):
    sample=sample_fixture(2,hidden=True);image=sample["world"]["image"]
    if damage=="partial_hidden": image["value"][0,0,0]=0
    if damage=="observed": image["observed"][0,0]=True
    if damage=="target_mask": image["target_valid"][0,0]=False
    if damage=="time": image["time"][0,0]=1
    if damage=="second_camera": image["camera"][0,0]=1
    if damage=="extra_stream": sample["world"]["proprio"]={}
    if damage=="language_nan": sample["language_features"][0,0]=float("nan")
    if damage=="language_swapped": sample["language_features"]=sample["language_features"].flip(0)
    if damage=="task_swapped": sample["task_id"].reverse()
    if damage=="text_swapped": sample["language_instruction"].reverse()
    if damage=="feature_hash": sample["pixel_metadata"][0]["class_feature_sha256"]="f"*64
    if damage=="forecast": sample["task_kind"][0]="forecast"
    if damage=="patch_geometry": image["xy"][0,0]=0
    with pytest.raises(ValueError): _validate_sample(sample,batch_size=2,hidden=True)


@pytest.mark.parametrize("damage", ["wrong_lr","missing_active_grad","nan_active_grad","inactive_zero_grad","inactive_adam_state"])
def test_actual_callback_rejects_wrong_lr_or_active_inactive_gradient_state(tmp_path,damage):
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.image=torch.nn.Parameter(torch.ones(2));self.unused=torch.nn.Parameter(torch.ones(2))
    model=Tiny();optimizer=torch.optim.AdamW(model.parameters(),lr=update_lr(from_config(bound_config().model),0))
    proof=ImageNetSmokeProof(tmp_path);proof.partition={"image":"active","unused":"inactive"}
    proof.usage={"image":0,"unused":0};proof.optimizer_lrs=[];proof.policy=from_config(bound_config().model)
    model.image.square().sum().backward()
    if damage=="wrong_lr": optimizer.param_groups[0]["lr"]=1e-4
    if damage=="missing_active_grad": model.image.grad=None
    if damage=="nan_active_grad": model.image.grad[0]=float("nan")
    if damage=="inactive_zero_grad": model.unused.grad=torch.zeros_like(model.unused)
    if damage=="inactive_adam_state": optimizer.state[model.unused]={"step":torch.tensor(0.)}
    trainer=type("FixtureTrainer",(),dict(global_step=0))()
    with pytest.raises(ValueError): proof.on_before_optimizer_step(trainer,model,optimizer)


def test_cli_keeps_capacity_output_and_postflight_separate(monkeypatch,tmp_path,capsys):
    from scripts.world_model import verify_imagenet_smoke as cli
    calls=[]
    def postflight(output):
        calls.append(("postflight",output));return dict(status="PASS",schema="fixture_only",job_id="not_actual")
    def capacity(config,output,**kwargs):
        calls.append(("capacity",config,output,kwargs));return None
    monkeypatch.setattr(cli,"verify_imagenet_smoke",postflight)
    monkeypatch.setattr(cli,"validate_imagenet_capacity",capacity)
    monkeypatch.setattr(sys,"argv",["verify_imagenet_smoke",str(tmp_path/"smoke")])
    cli.main();assert calls[0]==("postflight",tmp_path/"smoke")
    capsys.readouterr()
    monkeypatch.setattr(sys,"argv",["verify_imagenet_smoke",str(tmp_path/"smoke"),
        "--capacity-config",str(tmp_path/"config.yaml"),"--capacity-output",str(tmp_path/"capacity.json"),
        "--capacity-source-manifest-sha256","1"*64,"--capacity-data-manifest-sha256","2"*64])
    cli.main()
    assert calls[1][0:3]==("capacity",tmp_path/"config.yaml",tmp_path/"capacity.json")
    assert calls[1][3]==dict(source_manifest_sha256="1"*64,data_manifest_sha256="2"*64)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("argv", [["--capacity-output","elsewhere"],["--capacity-config","config.yaml"]])
def test_cli_incomplete_capacity_bindings_rejected_before_dispatch(monkeypatch,tmp_path,argv):
    from scripts.world_model import verify_imagenet_smoke as cli
    monkeypatch.setattr(cli,"verify_imagenet_smoke",lambda *a,**k:pytest.fail("must not dispatch"))
    monkeypatch.setattr(cli,"validate_imagenet_capacity",lambda *a,**k:pytest.fail("must not dispatch"))
    monkeypatch.setattr(sys,"argv",["verify_imagenet_smoke",str(tmp_path/"smoke"),*argv])
    with pytest.raises(SystemExit) as exc: cli.main()
    assert exc.value.code==2
