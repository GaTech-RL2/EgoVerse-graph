"""Step-limited pilot provenance and weights-only callback metadata."""

from types import SimpleNamespace

import pytest
import torch

from egomimic.pl_utils.oat_training import (
    OATBatchBudgetCallback,
    OATEMACallback,
    OATTrainingBehavior,
)


@pytest.mark.parametrize(
    "epochs,max_steps,expected",
    [(3, -1, 30), (3, 17, 17), (-1, 17, 17), (3, 45, 30)],
)
def test_step_budget_honors_first_limit(epochs, max_steps, expected):
    trainer = SimpleNamespace(
        datamodule=SimpleNamespace(
            train_datasets={"x": range(83)},
            train_dataloader_params={"x": {"batch_size": 2, "drop_last": True}},
        ),
        accumulate_grad_batches=4,
        world_size=1,
        limit_train_batches=1.0,
        max_epochs=epochs,
        max_steps=max_steps,
        is_global_zero=False,
    )
    callback = OATBatchBudgetCallback(global_batch_size=8)
    callback.on_fit_start(trainer, None)
    assert callback.budget["total_optimizer_steps"] == expected
    assert callback.budget["max_steps"] == max_steps
    assert callback.budget["dropped_examples_per_epoch"] == 3
    assert trainer.limit_train_batches == 40


def test_joint_budget_matches_actual_cycling_loader_and_preserves_both_sources():
    from torch.utils.data import DataLoader
    from lightning.pytorch.utilities.combined_loader import CombinedLoader

    datasets = {"original": range(83), "sharpa": range(37)}
    params = {name: {"batch_size": 16, "drop_last": True} for name in datasets}
    trainer = SimpleNamespace(
        datamodule=SimpleNamespace(train_datasets=datasets,
            train_dataloader_params=params, train_loader_mode="max_size_cycle"),
        accumulate_grad_batches=1, world_size=1, limit_train_batches=1.0,
        max_epochs=-1, max_steps=120000, is_global_zero=False,
    )
    callback = OATBatchBudgetCallback(global_batch_size=32,
        expected_source_batches={"original": 16, "sharpa": 16})
    callback.on_fit_start(trainer, None)
    loader = CombinedLoader({name: DataLoader(data, **params[name])
        for name, data in datasets.items()}, "max_size_cycle")
    seen = list(loader)
    assert trainer.limit_train_batches == len(seen) == 5
    assert all({name: len(batch) for name, batch in item[0].items()}
               == {"original": 16, "sharpa": 16} for item in seen)
    assert callback.budget["total_optimizer_steps"] == 120000
    assert callback.budget["global_batch_size"] == 32
    params["sharpa"]["batch_size"] = 8
    trainer.limit_train_batches = 1.0
    with pytest.raises(ValueError, match="preserve batch size"):
        callback.on_fit_start(trainer, None)


@pytest.mark.parametrize("weights_only", [True, False])
def test_weights_only_retains_ema_and_budget_without_optimizer(
    monkeypatch, weights_only
):
    from egomimic.models.oat import checkpoint as contract

    monkeypatch.setattr(contract, "validate_input_representation", lambda stages: {})
    stage = SimpleNamespace(normalizer_state={"scale": 1}, data_context={"split": 42})

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([2.0]))
            self.model = SimpleNamespace(pipeline=SimpleNamespace(stages=[stage]))

    model = Model()
    budget = OATBatchBudgetCallback()
    budget.budget = {"total_optimizer_steps": 10000}
    ema = OATEMACallback()
    trainer = SimpleNamespace(callbacks=[budget, ema], global_step=0)
    model.trainer = trainer
    ema.on_fit_start(trainer, model)
    ema._num_updates = 3
    ema._shadow["weight"].fill_(1.0)
    behavior = OATTrainingBehavior()
    behavior.bind(model)
    checkpoint = {"state_dict": model.state_dict(), "global_step": 3}
    if not weights_only:
        checkpoint["optimizer_states"] = [{"sentinel": True}]
        for callback in trainer.callbacks:
            callback.on_save_checkpoint(trainer, model, checkpoint)
        # Full-state model hook must not call callbacks a second time.
        monkeypatch.setattr(
            ema, "on_save_checkpoint", lambda *args: pytest.fail("twice")
        )
    behavior.on_save_checkpoint(checkpoint)
    assert checkpoint["ema_num_updates"] == 3
    assert checkpoint["ema_state_dict"]["weight"].item() == 1.0
    assert checkpoint["state_dict"]["weight"].item() == 2.0
    assert checkpoint["training_budget"]["total_optimizer_steps"] == 10000
    assert checkpoint["normalizer_state"] == {"scale": 1}
    assert ("optimizer_states" not in checkpoint) == weights_only


@pytest.mark.parametrize("basis", ["uniform", "fourier", "chebyshev"])
@pytest.mark.parametrize("mode", ["smoke", "full"])
@pytest.mark.parametrize("gpus", [1, 4])
def test_pilot_arguments_compose_matched_budget_and_checkpoint_policy(
    tmp_path, basis, mode, gpus
):
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from egomimic.benchmarks.libero.cluster import global_basis_training_arguments

    args = global_basis_training_arguments(
        basis, tmp_path / "data.zarr", tmp_path, mode, f"test-{basis}-{mode}", gpus=gpus
    )
    with initialize_config_dir(
        config_dir=str(Path(__file__).parents[1] / "egomimic/hydra_configs"),
        version_base="1.3",
    ):
        cfg = compose(config_name="train_zarr_cartesian", overrides=args[3:])
    assert cfg.benchmark.arc_basis == basis
    assert cfg.trainer.max_steps == (16 if mode == "smoke" else 10000)
    assert cfg.trainer.max_epochs == -1
    assert cfg.trainer.devices == gpus
    assert cfg.model.train_log_on_step
    assert cfg.trainer.check_val_every_n_epoch is None
    cp = cfg.callbacks.model_checkpoint
    assert cp.every_n_train_steps == (8 if mode == "smoke" else 2500)
    assert (
        cfg.trainer.val_check_interval
        == cp.every_n_train_steps * cfg.trainer.accumulate_grad_batches
    )
    assert cp.every_n_epochs == 0
    assert cp.save_weights_only and not cp.save_last and cp.save_top_k == -1
    assert not cp.save_on_train_epoch_end
    assert cfg.callbacks.ema.final_checkpoint_path is None
    assert "{epoch:" in cp.filename and "{step:" in cp.filename
    instantiate(cp)  # Real callback argument validation, no training.
    expected_batch = 4 * gpus if mode == "smoke" else 1024
    assert cfg.callbacks.batch_budget.global_batch_size == expected_batch
    assert (
        cfg.benchmark.batch_size * gpus * cfg.trainer.accumulate_grad_batches
        == expected_batch
    )
    assert cfg.logger.wandb.resume == "never" and not cfg.logger.wandb.offline


def test_early_recovery_save_has_full_state_and_refuses_overwrite(tmp_path):
    from types import SimpleNamespace
    from egomimic.pl_utils.oat_training import EarlyRecoveryCheckpoint
    import torch
    import json
    callback = EarlyRecoveryCheckpoint(str(tmp_path), first_step=2)
    state = {'global_step': 2, 'ema_num_updates': 2,
             'state_dict': {'weight': torch.ones(2)},
             'optimizer_states': [{'state': {0: {'moment': torch.zeros(2)}}}],
             'lr_schedulers': [{'last_epoch': 2}], 'data_context': {'bound': True}}
    def save(path, weights_only):
        assert weights_only is False
        torch.save(state, path)
    trainer = SimpleNamespace(global_step=1, current_epoch=0, world_size=1, save_checkpoint=save)
    callback.on_train_batch_end(trainer, None, None, None, 0)
    assert not list(tmp_path.iterdir())
    trainer.global_step = 2
    callback.on_train_batch_end(trainer, None, None, None, 1)
    path = next((tmp_path / 'checkpoints').glob('*.ckpt'))
    proof = json.loads(path.with_suffix('.ckpt.recovery.json').read_text())
    assert proof['global_step'] == 2 and proof['full_state'] and proof['bytes'] == path.stat().st_size
    callback.on_train_batch_end(trainer, None, None, None, 1)
    with pytest.raises(FileExistsError, match='overwrite'):
        EarlyRecoveryCheckpoint(str(tmp_path), 2).on_train_batch_end(trainer, None, None, None, 1)
