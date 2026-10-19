"""Scheduled evaluation/strict checkpoint and launch-contract regressions."""

import copy
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf, open_dict

from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.trainHydra import _build_model_config_tree
from scripts.train.joint.contract import compare_configs
from tests.test_joint_modality_ablation import CELLS
from tests.test_pixel_proprio_data import pixel_manifest


def configuration(cell, output, smoke=False):
    overrides = [
        f"+experiment=pixel_proprio/ablation/discrete_{cell}",
        "hydra/launcher=basic",
        "data.manifest_path=/fixture/manifest.json",
        "paths.work_dir=/fixture/source",
        f"paths.output_dir={output}",
        f"logger.csv.save_dir=/fixture/logs/{Path(output).name}",
        "evaluator.pixel_diagnostics.log_wandb=false",
        "+trainer.enable_progress_bar=false",
        "+trainer.enable_model_summary=false",
    ]
    if smoke:
        overrides += ["+joint_execution=smoke", "logger=[csv]", "~logger.wandb"]
    else:
        overrides += ["logger=[wandb,csv]"]
    with initialize_config_dir(
        config_dir=str(Path("egomimic/hydra_configs").resolve()), version_base=None
    ):
        return compose(config_name="train_zarr_cartesian", overrides=overrides)


@pytest.mark.parametrize("cell", CELLS)
def test_exact_full_config_matches_its_smoke_and_rejects_scientific_changes(cell):
    full = OmegaConf.to_container(configuration(cell, "/fixture/full"), resolve=True)
    smoke = OmegaConf.to_container(
        configuration(cell, "/fixture/smoke", True), resolve=True
    )
    request = dict(
        output="/fixture/full",
        local_logs="/fixture/logs/full",
        target_steps=100000,
        world_size=2,
        per_rank_batch=16,
        run_id=full["logger"]["wandb"]["id"],
    )
    compare_configs(full, smoke, request)
    for section, key, value in [
        ("model", "optimizer", {"lr": 5e-5}),
        ("data", "batch_size", 8),
        ("seed", None, 43),
        ("trainer", "max_steps", 90000),
        ("evaluator", "prediction_streams", ["image", "action"]),
    ]:
        damaged = copy.deepcopy(full)
        if key is None:
            damaged[section] = value
        else:
            damaged[section][key] = value
        if damaged == full:
            continue
        with pytest.raises(ValueError):
            compare_configs(damaged, smoke, request)
    damaged = copy.deepcopy(full)
    damaged["callbacks"]["model_checkpoint"]["save_top_k"] = 1
    with pytest.raises(ValueError):
        compare_configs(damaged, smoke, request)


@pytest.mark.parametrize("cell,streams", CELLS.items())
def test_scheduled_validation_and_strict_full_checkpoint_with_removed_streams(
    tmp_path, cell, streams
):
    cfg = configuration(cell, str(tmp_path))
    manifest = pixel_manifest(tmp_path, include_sharpa=False, frames=12, actions=True)
    with open_dict(cfg):
        cfg.data.manifest_path = str(manifest)
        cfg.data.discrete_frames = 4
        cfg.data.stride = 2
        cfg.data.batch_size = 1
        cfg.data.distributed_world_size = 1
        cfg.data.num_workers = 0
        cfg.data.persistent_workers = False
        cfg.data.prefetch_factor = None
        cfg.data.balance_training_work = False
        cfg.data.validation_discrete_frames = 4
        cfg.data.validation_stride = 2
        cfg.joint.pose_dims = {"libero_gripper": 18}
        cfg.joint.action_dims = {"libero_gripper": 14}
        cfg.joint.max_tokens = 100
        cfg.joint.compile_mode = "none"
        net = cfg.modality_ablation.nodes.joint_denoiser.model
        net.image_dim = 12
        net.width = 16
        net.proprio_width = 5
        net.action_width = 7
        net.depth = 1
        net.heads = 2
        net.rope_axis_dims = [4, 2, 2]
        net.spatial_grid_size = [2, 2]
        net.attention_backend = "sdpa"
        cfg.modality_ablation.nodes.joint_sampler.sampler.state_steps = 2
        cfg.modality_ablation.nodes.joint_sampler.sampler.action_steps = 2
        cfg.evaluator.pixel_diagnostics = None
    data = instantiate(cfg.data)
    context = data.prepare_context(
        mode="train", normalization="prepared", normalizer=None
    )
    model_cfg = _build_model_config_tree(cfg)
    evaluator = instantiate(cfg.evaluator)
    model = ModelWrapper(
        config_tree=model_cfg, evaluator=evaluator, enable_grad_norm=False
    )
    model.data_context = context
    context.bind(model.model, evaluator)
    data.configure_evaluation(evaluator.data_requirements())
    cb = ModelCheckpoint(
        dirpath=tmp_path / "checkpoints",
        every_n_train_steps=2,
        save_top_k=-1,
        save_last="link",
        save_on_train_epoch_end=False,
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        logger=False,
        max_steps=2,
        max_epochs=-1,
        min_epochs=0,
        limit_val_batches=2,
        val_check_interval=2,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[cb],
        default_root_dir=tmp_path,
    )
    trainer.fit(model, datamodule=data)
    assert evaluator.has_completed_validation(2) and data.consumed_batches == 2
    metrics = {
        k: v for k, v in trainer.callback_metrics.items() if k.startswith("val/")
    }
    assert len(metrics) == 2 * len(streams) and all(
        torch.isfinite(v) for v in metrics.values()
    )
    checkpoint = torch.load(cb.last_model_path, map_location="cpu", weights_only=False)
    assert (
        checkpoint["global_step"] == 2
        and checkpoint["data_context"] == context.snapshot()
    )
    assert checkpoint["lr_schedulers"][0]["last_epoch"] == 2
    restored = ModelWrapper(config_tree=model_cfg, enable_grad_norm=False)
    restored.load_state_dict(checkpoint["state_dict"], strict=True)
    assert (
        restored.model.pipeline.stage_by_id("joint_denoiser").model.streams == streams
    )


@pytest.mark.parametrize(
    "damage", [None, "initial_weights", "batch_noise", "topology", "missing_arm"]
)
def test_factorial_gpu_gate_requires_paired_actual_controls(
    tmp_path, monkeypatch, damage
):
    import scripts.train.joint.verify_factorial as gate
    from scripts.train.joint.contract import sha

    requests = []
    for cell in CELLS:
        cfg = configuration(cell, f"/fixture/{cell}", True)
        file = tmp_path / (cell + ".yaml")
        file.write_text(OmegaConf.to_yaml(cfg, resolve=True))
        request = tmp_path / (cell + ".json")
        request.write_text(
            __import__("json").dumps(
                dict(output=f"/fixture/{cell}", local_logs=f"/fixture/logs/{cell}")
            )
        )
        ranks = [
            dict(
                data_context_sha256="context",
                initial_image_weights_sha256="weights",
                paired_rgb_batch_sha256=["a", "b", "c"],
            )
            for _ in range(2)
        ]
        requests.append(
            dict(
                experiment=cell,
                source_revision="one-source",
                completion=dict(ranks=ranks),
                smoke=dict(
                    resolved=dict(path=str(file), sha256=sha(file)),
                    request=dict(path=str(request), sha256=sha(request)),
                    completion=dict(sha256="completion-" + cell),
                    job_id=1,
                ),
            )
        )
    monkeypatch.setattr(
        gate, "verify_evidence", lambda request: (request["completion"], {})
    )
    if damage == "initial_weights":
        requests[-1]["completion"]["ranks"][1]["initial_image_weights_sha256"] = "other"
    elif damage == "batch_noise":
        requests[-1]["completion"]["ranks"][0]["paired_rgb_batch_sha256"][2] = "other"
    elif damage == "topology":
        binding = requests[-1]["smoke"]["resolved"]
        cfg = OmegaConf.load(binding["path"])
        cfg.trainer.devices = 4
        OmegaConf.save(cfg, binding["path"])
        binding["sha256"] = sha(binding["path"])
    elif damage == "missing_arm":
        requests.pop()
    if damage:
        with pytest.raises(ValueError):
            gate.verify(requests)
    else:
        assert gate.verify(requests)["status"] == "PASS"
