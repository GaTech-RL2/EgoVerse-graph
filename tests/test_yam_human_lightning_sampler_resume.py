"""Checkpointed Lightning fit must replay and skip exact co-train batches."""

import lightning as L
import pytest
import torch
from torch.utils.data import Dataset

from egomimic.pl_utils.pl_data_utils import ProportionalMultiDataModuleWrapper


class _IndexedDataset(Dataset):
    def __init__(self, count: int):
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return {"index": torch.tensor(index, dtype=torch.float32)}


class _RecordingModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.seen = []

    def training_step(self, batch, batch_idx):
        assert len(batch) == 1
        source, values = next(iter(batch.items()))
        self.seen.append((source, tuple(int(x) for x in values["index"].tolist())))
        return self.scale * (values["index"].mean() + 1)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.001)


def _data(num_workers: int):
    return ProportionalMultiDataModuleWrapper(
        train_datasets={"yam_bimanual": _IndexedDataset(17), "human_bimanual": _IndexedDataset(11)},
        valid_datasets={"valid": {"yam_bimanual": _IndexedDataset(2)}},
        train_dataloader_params={},
        valid_dataloader_params={},
        proportional_train_batch_size=4,
        proportional_train_num_workers=num_workers,
        proportional_train_seed=42,
    )


def _trainer(max_steps: int, root):
    return L.Trainer(
        accelerator="cpu", devices=1, max_steps=max_steps,
        default_root_dir=root, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
        limit_val_batches=0, num_sanity_val_steps=0,
    )


@pytest.mark.parametrize("num_workers", [0, 2])
def test_mid_epoch_full_state_resume_preserves_exact_source_order(tmp_path, num_workers):
    uninterrupted = _RecordingModel()
    _trainer(4, tmp_path / "reference").fit(uninterrupted, datamodule=_data(num_workers))
    assert len(uninterrupted.seen) == 4

    first = _RecordingModel()
    first_trainer = _trainer(2, tmp_path / "first")
    first_trainer.fit(first, datamodule=_data(num_workers))
    checkpoint = tmp_path / "mid_epoch.ckpt"
    first_trainer.save_checkpoint(checkpoint)
    assert first.seen == uninterrupted.seen[:2]

    resumed = _RecordingModel()
    _trainer(4, tmp_path / "resumed").fit(
        resumed, datamodule=_data(num_workers), ckpt_path=checkpoint
    )
    assert resumed.seen == uninterrupted.seen[2:4]
