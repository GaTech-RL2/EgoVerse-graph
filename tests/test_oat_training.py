"""Exercise the real shared training entrypoint and self-contained checkpoints."""

from pathlib import Path

import pytest
import torch

from tests.test_libero_benchmark import make_replay


def config_for(experiment, path, output):
    from hydra import compose, initialize_config_dir

    root = Path(__file__).parents[1] / "egomimic/hydra_configs"
    with initialize_config_dir(version_base=None, config_dir=str(root)):
        return compose(
            config_name="train_zarr_cartesian",
            overrides=[
                f"+experiment=oat/{experiment}",
                f"benchmark.dataset={path}",
                "benchmark.batch_size=2",
                "benchmark.horizon=8",
                "benchmark.n_action_steps=4",
                f"paths.output_dir={output}",
                f"paths.work_dir={output}",
                f"norm_stats.save_cache_dir={output}",
                "trainer.accelerator=cpu",
                "trainer.precision=32-true",
                "trainer.devices=1",
                "trainer.max_epochs=1",
                "trainer.limit_train_batches=2",
                "trainer.limit_val_batches=1",
                "trainer.accumulate_grad_batches=1",
                "callbacks.batch_budget=null",
                "trainer.check_val_every_n_epoch=1",
                "callbacks.model_checkpoint.every_n_epochs=1",
                "data.train_dataloader_params.libero_panda.num_workers=0",
                "data.train_dataloader_params.libero_panda.persistent_workers=false",
                "data.valid_dataloader_params.libero_panda.num_workers=0",
                "data.valid_dataloader_params.libero_panda.persistent_workers=false",
            ],
        )


@pytest.mark.parametrize("precision", ["32-true", "bf16-mixed"])
def test_shared_training_tokenizer_then_policy_reload_and_resume(tmp_path, precision):
    from egomimic.benchmarks.libero.rollout import load_policy
    from egomimic.models.oat.factory import load_tokenizer
    from egomimic.trainHydra import train

    path = tmp_path / "replay.zarr"
    make_replay(path)
    tokenizer_cfg = config_for("libero_oattok", path, tmp_path / "tokenizer")
    tokenizer_cfg.trainer.precision = precision
    spec = tokenizer_cfg.model.pipeline.stages[0].tokenizer
    spec.emb_dim, spec.head_dim = 32, 8
    spec.encoder_depth, spec.decoder_depth, spec.num_registers = 1, 1, 4
    metrics, objects = train(tokenizer_cfg)
    assert objects["trainer"].global_step == 2
    checkpoint = tmp_path / "tokenizer/checkpoints/last.ckpt"
    tokenizer = load_tokenizer(checkpoint)
    assert tokenizer.latent_horizon == 4
    assert "Valid/reconst_mse" in metrics
    policy_cfg = config_for("libero_oatpolicy", path, tmp_path / "policy")
    policy_cfg.trainer.precision = precision
    policy_cfg.benchmark.tokenizer_checkpoint = str(checkpoint)
    policy_cfg.model.pipeline.stages[0].policy.embed_dim = 16
    policy_cfg.model.pipeline.stages[0].policy.n_layers = 1
    policy_cfg.model.pipeline.stages[0].policy.n_heads = 2
    _, policy_objects = train(policy_cfg)
    assert policy_objects["trainer"].global_step == 2
    policy_checkpoint = tmp_path / "policy/checkpoints/last.ckpt"
    # Resume through shared Lightning, including optimizer/EMA state.
    policy_cfg.ckpt_path = str(policy_checkpoint)
    policy_cfg.trainer.max_epochs = 2
    _, resumed = train(policy_cfg)
    assert resumed["trainer"].global_step == 4
    checkpoint.unlink()
    restored, protocol = load_policy(policy_checkpoint, device="cpu")
    assert protocol["horizon"] == 8
    observation = {
        "agentview_rgb": torch.zeros(128, 128, 3).numpy(),
        "robot0_eye_in_hand_rgb": torch.zeros(128, 128, 3).numpy(),
        "robot0_eef_pos": torch.zeros(3).numpy(),
        "robot0_eef_quat": torch.tensor([0, 0, 0, 1]).numpy(),
        "robot0_gripper_qpos": torch.zeros(2).numpy(),
        "task_uid": torch.tensor([30]).numpy(),
    }
    restored.reset(observation)
    actions = restored.predict()
    assert actions.shape == (4, 7)
    assert torch.isfinite(torch.from_numpy(actions)).all()


@pytest.mark.parametrize("precision", ["32-true", "bf16-mixed"])
def test_arc_policy_uses_shared_training_graph_and_can_infer_without_targets(
    tmp_path, precision
):
    from egomimic.benchmarks.libero.rollout import load_policy
    from egomimic.trainHydra import _load_eval_checkpoint, train

    path = tmp_path / "replay.zarr"
    make_replay(path)
    cfg = config_for("libero_arc_policy", path, tmp_path / "arc")
    cfg.trainer.precision = precision
    cfg.benchmark.arc_waypoints = 8
    denoiser = cfg.model.pipeline.stages[3]
    denoiser.policy.model.down_dims = [32, 64]
    denoiser.policy.model.diffusion_step_embed_dim = 32
    denoiser.policy.num_inference_steps = 2
    _, objects = train(cfg)
    assert objects["trainer"].global_step == 2
    policy, _ = load_policy(tmp_path / "arc/checkpoints/last.ckpt", device="cpu")
    dataset = objects["datamodule"].train_datasets["libero_panda"]
    values = dataset[0]
    values.pop("actions")
    batch = {
        key: value.unsqueeze(0)
        for key, value in values.items()
        if torch.is_tensor(value)
    }
    result = policy.algo.forward_eval({"libero_panda": batch})
    assert result["libero_panda"]["pred_action"].shape == (1, 8, 7)
    payload = torch.load(
        tmp_path / "arc/checkpoints/last.ckpt", map_location="cpu", weights_only=False
    )
    _load_eval_checkpoint(objects["model"], payload, cfg)
    context = payload["normalizer_state"]["benchmark_context"]
    context["observations_sha256"] = "changed-camera-replay"
    with pytest.raises(ValueError, match="different benchmark"):
        objects["model"].on_load_checkpoint(payload)
    with pytest.raises(ValueError, match="different benchmark"):
        _load_eval_checkpoint(objects["model"], payload, cfg)


def test_final_checkpoint_includes_epoch_after_periodic_checkpoint(tmp_path):
    from egomimic.trainHydra import train

    path = tmp_path / "replay.zarr"
    make_replay(path)
    cfg = config_for("libero_oattok", path, tmp_path / "tokenizer")
    spec = cfg.model.pipeline.stages[0].tokenizer
    spec.emb_dim, spec.head_dim = 32, 8
    spec.encoder_depth, spec.decoder_depth, spec.num_registers = 1, 1, 4
    cfg.trainer.max_epochs = 3
    cfg.callbacks.model_checkpoint.every_n_epochs = 2
    _, objects = train(cfg)
    payload = torch.load(
        tmp_path / "tokenizer/checkpoints/last.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    assert payload["global_step"] == objects["trainer"].global_step == 6
    assert payload["ema_num_updates"] == 6
    assert payload["loops"]["fit_loop"]["epoch_progress"]["current"]["completed"] == 3


def test_accumulation_preserves_global_batch_drop_last_steps_and_ema(tmp_path):
    from omegaconf import OmegaConf

    from egomimic.trainHydra import train

    path = tmp_path / "replay.zarr"
    make_replay(path)  # 90 train examples: 11 full batches of 8, two dropped.
    cfg = config_for("libero_oattok", path, tmp_path / "tokenizer")
    spec = cfg.model.pipeline.stages[0].tokenizer
    spec.emb_dim, spec.head_dim = 32, 8
    spec.encoder_depth, spec.decoder_depth, spec.num_registers = 1, 1, 4
    cfg.trainer.limit_train_batches = 1.0
    cfg.trainer.accumulate_grad_batches = 4
    cfg.callbacks.batch_budget = OmegaConf.create(
        {
            "_target_": "egomimic.pl_utils.oat_training.OATBatchBudgetCallback",
            "global_batch_size": 8,
            "report_path": str(tmp_path / "training-budget.json"),
        }
    )
    _, objects = train(cfg)
    payload = torch.load(
        tmp_path / "tokenizer/checkpoints/last.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    assert objects["trainer"].num_training_batches == 44
    assert payload["global_step"] == payload["ema_num_updates"] == 11
    assert payload["training_budget"]["total_optimizer_steps"] == 11
    assert payload["training_budget"]["dropped_examples_per_epoch"] == 2
    cfg.ckpt_path = str(tmp_path / "tokenizer/checkpoints/last.ckpt")
    cfg.trainer.max_epochs = 2
    _, resumed = train(cfg)
    payload = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
    assert resumed["trainer"].global_step == payload["ema_num_updates"] == 22
    assert payload["training_budget"]["total_optimizer_steps"] == 22
