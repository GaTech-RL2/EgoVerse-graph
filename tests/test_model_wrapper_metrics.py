from collections import OrderedDict
import csv

from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf
import pytest
import torch
from torch.utils.data import DataLoader

from egomimic.pipeline.stages_planar import PlanarActionMSELoss
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.trainHydra import _instantiate_model_wrapper


class _MetricPipeline:
    def __init__(self):
        self.nets = torch.nn.ModuleDict({"anchor": torch.nn.Linear(1, 1)})

    @staticmethod
    def process_batch_for_training(batch):
        return batch

    @staticmethod
    def forward_training(batch):
        return OrderedDict(
            (
                source,
                {
                    "loss/test": torch.tensor(1.0, requires_grad=True),
                    "log/MSE": values["metric"],
                    "prediction": values.get("prediction", torch.ones(2)),
                },
            )
            for source, values in batch.items()
        )

    @staticmethod
    def compute_losses(predictions, _batch):
        return {
            "loss": torch.stack(
                [result["loss/test"] for result in predictions.values()]
            ).mean()
        }

    @staticmethod
    def log_info(_info):
        return {}


def _wrapper_with_log_capture(monkeypatch):
    wrapper = ModelWrapper(pipeline=_MetricPipeline())
    logged = {}
    monkeypatch.setattr(
        wrapper,
        "log",
        lambda name, value, **kwargs: logged.setdefault(name, (value, kwargs)),
    )
    return wrapper, logged


def test_training_logs_each_opaque_source_and_equal_source_macro(monkeypatch):
    wrapper, logged = _wrapper_with_log_capture(monkeypatch)
    batch = OrderedDict(
        (
            ("pushshapes_sim_u_socket", {"metric": torch.tensor(2.0)}),
            ("another_source", {"metric": 6.0}),
        )
    )

    wrapper.training_step(batch, batch_idx=0)

    expected = {
        "Train/MSE/pushshapes_sim_u_socket": 2.0,
        "Train/MSE/another_source": 6.0,
        "Train/MSE": 4.0,
    }
    assert expected.keys() <= logged.keys()
    for name, expected_value in expected.items():
        value, kwargs = logged[name]
        assert float(value) == pytest.approx(expected_value)
        assert kwargs == {
            "sync_dist": True,
            "on_step": False,
            "on_epoch": True,
        }


def test_training_rejects_non_scalar_log_metric_but_ignores_other_predictions(
    monkeypatch,
):
    wrapper, _logged = _wrapper_with_log_capture(monkeypatch)
    batch = {"source": {"metric": torch.tensor([1.0, 2.0])}}

    with pytest.raises(TypeError, match="must be scalar"):
        wrapper.training_step(batch, batch_idx=0)


def test_training_rejects_non_finite_log_metric(monkeypatch):
    wrapper, _logged = _wrapper_with_log_capture(monkeypatch)
    batch = {"source": {"metric": torch.tensor(float("nan"))}}

    with pytest.raises(RuntimeError, match="Non-finite pipeline metric"):
        wrapper.training_step(batch, batch_idx=0)


def test_planar_action_loss_exports_canonical_mse_metric():
    stage = PlanarActionMSELoss()
    result = stage(
        {
            "pred_action": torch.zeros(2, 3, 5),
            "target": torch.ones(2, 3, 5),
        }
    )

    assert stage.writes == ("loss/action", "log/MSE")
    assert float(result["loss/action"]) == pytest.approx(1.0)
    assert float(result["log/MSE"]) == pytest.approx(1.0)
    assert "log/action_mse" not in result


@pytest.mark.parametrize("on_step", [None, False, True])
def test_entrypoint_logging_control_reaches_real_csv_rows(tmp_path, monkeypatch, on_step):
    """Config-only assertions miss controls dropped by wrapper construction."""
    class TrainMetricPipeline(_MetricPipeline):
        def forward_training(self, batch):
            loss = self.nets["anchor"](batch).square().mean()
            return {"source": {"loss/test": loss, "log/MSE": loss.detach()}}

    monkeypatch.setattr(ModelWrapper, "_instantiate_model", lambda *args: TrainMetricPipeline())
    cfg = OmegaConf.create({"model": {
        "pipeline": {}, "enable_grad_norm": False,
        "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.01},
    }})
    if on_step is not None:
        cfg.model.train_log_on_step = on_step
    wrapper = _instantiate_model_wrapper(cfg)
    assert wrapper.train_log_on_step is (on_step is True)
    logger = CSVLogger(tmp_path, name="metrics")
    trainer = Trainer(
        accelerator="cpu", devices=1, max_steps=16, max_epochs=1,
        logger=logger, log_every_n_steps=1, enable_checkpointing=False,
        enable_model_summary=False, enable_progress_bar=False,
    )
    trainer.fit(wrapper, DataLoader(torch.ones(16, 1), batch_size=1))
    with (tmp_path / "metrics/version_0/metrics.csv").open() as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("Train/MSE")]
    assert trainer.global_step == 16
    assert [int(r["step"]) for r in rows] == (list(range(16)) if on_step else [15])
    assert all(torch.isfinite(torch.tensor(float(r["Train/MSE"]))) for r in rows)
