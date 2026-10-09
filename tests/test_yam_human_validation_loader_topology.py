"""Prevent nested CombinedLoader tuples at the two-domain validation boundary."""

from collections.abc import Mapping

import pytest
import torch
from lightning import LightningModule, Trainer
from torch.utils.data import Dataset

from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pl_utils.pl_data_utils import ProportionalMultiDataModuleWrapper


class _TinyDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {"value": torch.tensor(index, dtype=torch.float32)}


class _ValidationProbe(LightningModule):
    def __init__(self):
        super().__init__()
        self.seen = []

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        PipelineAlgo._validate_groups(batch)
        self.seen.append((dataloader_idx, tuple(batch), batch_idx))


@pytest.mark.parametrize("workers", [0, 2])
def test_two_validation_groups_reach_lightning_as_flat_source_mappings(workers):
    data = ProportionalMultiDataModuleWrapper(
        train_datasets={},
        valid_datasets={
            "yam": {"yam_bimanual": _TinyDataset()},
            "human": {"human_bimanual": _TinyDataset()},
        },
        train_dataloader_params={},
        valid_dataloader_params={
            "yam": {"yam_bimanual": {"batch_size": 4, "num_workers": workers, "shuffle": False}},
            "human": {"human_bimanual": {"batch_size": 4, "num_workers": workers, "shuffle": False}},
        },
        proportional_train_batch_size=8,
        proportional_train_num_workers=0,
    )
    loaders = data.val_dataloader()
    assert len(loaders) == 2
    for source, loader in zip(("yam_bimanual", "human_bimanual"), loaders):
        batch = next(iter(loader))
        assert isinstance(batch, Mapping)
        PipelineAlgo._validate_groups(batch)
        assert tuple(batch) == (source,)
    probe = _ValidationProbe()
    trainer = Trainer(
        accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
        enable_model_summary=False, limit_val_batches=1,
    )
    trainer.validate(probe, dataloaders=loaders)
    assert probe.seen == [
        (0, ("yam_bimanual",), 0),
        (1, ("human_bimanual",), 0),
    ]
