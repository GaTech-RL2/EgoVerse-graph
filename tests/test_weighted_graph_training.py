"""Real Lightning/DDP integration for the graph's weighted source loader."""

import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from lightning import Callback, Trainer
from lightning.pytorch.strategies import DDPStrategy
from torch import nn
from torch.utils.data import Dataset

from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pipeline.core import Stage, resolve_homogeneous_scalar
from egomimic.pl_utils.pl_data_utils import MultiDataModuleWrapper
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.pl_utils.training_behavior import TrainingBehavior
from egomimic.rldb.weighted_dataset import WeightedDatasetSampler


class Samples(Dataset):
    def __init__(self, domain):
        self.domain = domain

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return {"actions": torch.tensor([[index / 8.0]]), "domain": self.domain}


class RoutedHead(Stage):
    reads = ("actions", "domain")
    writes = ("loss/fit", "log/*")

    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleDict({name: nn.Linear(1, 1) for name in ("a", "b")})
        self.seen = []

    def forward(self, batch):
        domain = resolve_homogeneous_scalar(batch["domain"])
        self.seen.append(domain)
        loss = self.heads[domain](batch["actions"]).square().mean()
        batch["loss/fit"] = loss
        batch["log/" + domain] = loss.detach()
        return batch


class OptimizerBehavior(TrainingBehavior):
    def configure_optimizers(self):
        return torch.optim.SGD(self.context.parameters(), lr=0.01)


class SamplerTrace(Callback):
    def __init__(self, output):
        self.output = str(output)

    def on_train_epoch_end(self, trainer, module):
        sampler = trainer.train_dataloader.sampler
        assert isinstance(sampler, WeightedDatasetSampler)
        assert sampler.num_replicas == 2 and sampler.rank == trainer.global_rank
        path = (
            Path(self.output)
            / f"rank{trainer.global_rank}-epoch{trainer.current_epoch}.json"
        )
        path.write_text(
            json.dumps(
                {
                    "epoch": sampler.epoch,
                    "indices": list(sampler),
                    "seen": module.model.pipeline.stages[0].seen,
                }
            )
        )


@pytest.mark.skipif(
    not torch.distributed.is_gloo_available(), reason="CPU DDP requires Gloo"
)
def test_weighted_graph_ddp_handles_missing_sources_and_advances_sampler(tmp_path):
    data = MultiDataModuleWrapper(
        {"eva_bimanual": Samples("a"), "human_bimanual": Samples("b")},
        {},
        {},
        {},
        dataset_weights={"eva_bimanual": 1, "human_bimanual": 3},
        weighted_dataloader_params={"batch_size": 1, "num_workers": 0},
        samples_per_epoch=12,
        sampling_seed=0,
    )
    model = ModelWrapper(
        pipeline=PipelineAlgo([RoutedHead()], device="cpu"),
        training_behavior=OptimizerBehavior(),
        enable_grad_norm=False,
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=2,
        strategy=DDPStrategy(start_method="spawn", find_unused_parameters=True),
        max_epochs=2,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[SamplerTrace(tmp_path)],
        default_root_dir=str(tmp_path),
    )
    trainer.fit(model, datamodule=data)
    traces = {
        (rank, epoch): json.loads(
            (tmp_path / f"rank{rank}-epoch{epoch}.json").read_text()
        )
        for rank in range(2)
        for epoch in range(2)
    }
    assert traces[0, 0]["seen"] != traces[1, 0]["seen"]
    for rank in range(2):
        assert traces[rank, 0]["epoch"] == 0 and traces[rank, 1]["epoch"] == 1
        assert traces[rank, 0]["indices"] != traces[rank, 1]["indices"]
    assert torch.isfinite(trainer.callback_metrics["Train/Loss"])


def test_weighted_overlay_uses_graph_runtime_and_composes():
    config_dir = Path(__file__).parents[1] / "egomimic/hydra_configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(
            config_name="train_zarr_cartesian_pi",
            overrides=[
                "hydra/launcher=basic",
                "data=pi05/cotrain_pi_lang_6d",
                "model=pi05/pi0.5_cotrain_eva_aria_6d",
                "+experiment=weighted_cotrain",
            ],
        )
    assert cfg.model.pipeline.homogeneous_training
    assert cfg.data.dataset_weights.human_bimanual == 3
    assert cfg.data.weighted_dataloader_params.batch_size == 64
    assert cfg.trainer.strategy == "ddp_find_unused_parameters_true"
    assert not cfg.trainer.sync_batchnorm
