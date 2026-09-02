from types import SimpleNamespace

import torch
from lightning import LightningModule

from egomimic.utils.ema_callback import EMACallback


class _TinyModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1, bias=False)
        self.logged = {}

    def log(self, name, value, **kwargs):
        self.logged[name] = float(value)


def test_ema_updates_swaps_validates_and_serializes_online_weights():
    module = _TinyModule()
    trainer = SimpleNamespace(global_step=0)
    callback = EMACallback(decay=0.5, validate_with_ema=True)
    callback.on_fit_start(trainer, module)
    initial = module.layer.weight.detach().clone()

    with torch.no_grad():
        module.layer.weight.add_(2.0)
    online = module.layer.weight.detach().clone()
    trainer.global_step = 1
    callback.on_train_batch_end(trainer, module, None, None, 0)
    expected_ema = 0.5 * initial + 0.5 * online

    callback.on_validation_start(trainer, module)
    torch.testing.assert_close(module.layer.weight, expected_ema)
    checkpoint = {"state_dict": module.state_dict()}
    callback.on_save_checkpoint(trainer, module, checkpoint)
    torch.testing.assert_close(checkpoint["state_dict"]["layer.weight"], online)
    torch.testing.assert_close(
        checkpoint["ema_state_dict"]["layer.weight"], expected_ema
    )
    assert checkpoint["ema_decay"] == 0.5
    assert checkpoint["ema_num_updates"] == 1

    callback.on_validation_end(trainer, module)
    torch.testing.assert_close(module.layer.weight, online)
    assert module.logged["log/unite_ema_decay"] == 0.5
    assert module.logged["log/unite_ema_num_updates"] == 1.0


def test_ema_resume_rejects_missing_or_mismatched_state():
    module = _TinyModule()
    trainer = SimpleNamespace(global_step=3)
    callback = EMACallback(decay=0.9978)
    try:
        callback.on_load_checkpoint(trainer, module, {"global_step": 3})
    except RuntimeError as exc:
        assert "without EMA" in str(exc)
    else:
        raise AssertionError("EMA resume accepted a checkpoint without EMA")

    try:
        callback.on_load_checkpoint(
            trainer,
            module,
            {
                "global_step": 3,
                "ema_decay": 0.9,
                "ema_state_dict": {"layer.weight": module.layer.weight},
            },
        )
    except RuntimeError as exc:
        assert "decay mismatch" in str(exc)
    else:
        raise AssertionError("EMA resume accepted a different decay")


def test_ema_resume_baselines_step_before_lightning_restores_loop_progress():
    module = _TinyModule()
    callback = EMACallback(decay=0.9978)
    shadow = {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
    }
    callback.on_load_checkpoint(
        SimpleNamespace(global_step=0),
        module,
        {
            "global_step": 20_000,
            "ema_decay": 0.9978,
            "ema_num_updates": 20_000,
            "ema_state_dict": shadow,
        },
    )

    # Lightning calls fit-start hooks before its loop progress has been restored.
    trainer = SimpleNamespace(global_step=0)
    callback.on_fit_start(trainer, module)
    trainer.global_step = 20_001
    callback.on_train_batch_end(trainer, module, None, None, 0)

    assert callback.num_updates == 20_001


def test_ema_resume_rejects_update_count_drift():
    module = _TinyModule()
    callback = EMACallback(decay=0.9978)
    try:
        callback.on_load_checkpoint(
            SimpleNamespace(global_step=0),
            module,
            {
                "global_step": 20_000,
                "ema_decay": 0.9978,
                "ema_num_updates": 19_999,
                "ema_state_dict": {
                    name: parameter.detach().clone()
                    for name, parameter in module.named_parameters()
                },
            },
        )
    except RuntimeError as exc:
        assert "must equal checkpoint global_step" in str(exc)
    else:
        raise AssertionError("EMA resume accepted a drifted update count")
