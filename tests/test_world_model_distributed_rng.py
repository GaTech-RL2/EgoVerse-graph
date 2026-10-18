"""Rank-specific random streams survive a real distributed checkpoint/resume."""

import json
import os
import random
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import egomimic.pl_utils.world_model_behavior as behavior


@pytest.fixture(autouse=True)
def preserve_rng():
    state = behavior._capture_rng_state()
    yield
    behavior._restore_rng_state(state)


def test_single_process_checkpoint_keeps_legacy_schema_and_random_streams():
    checkpoint = {}
    behavior.WorldModelTrainingBehavior().on_save_checkpoint(checkpoint)
    saved = checkpoint["mot_rng_state"]
    assert set(saved) == {"torch_cpu", "torch_cuda", "numpy", "python"}
    expected = (torch.rand(3), np.random.rand(3), random.random())
    behavior._restore_rng_state(behavior._rank_rng_state(saved))
    assert torch.equal(torch.rand(3), expected[0])
    assert np.array_equal(np.random.rand(3), expected[1])
    assert random.random() == expected[2]


def test_distributed_resume_rejects_legacy_or_changed_world_size(monkeypatch):
    local = behavior._capture_rng_state(current_cuda_only=True)
    monkeypatch.setattr(behavior, "_distributed_world", lambda: (2, 1))
    with pytest.raises(ValueError, match="per-rank RNG"):
        behavior._rank_rng_state(local)
    saved = {
        "schema": behavior._DISTRIBUTED_RNG_SCHEMA,
        "world_size": 4,
        "ranks": [local] * 4,
    }
    with pytest.raises(ValueError, match="saved RNG world size"):
        behavior._rank_rng_state(saved)


@pytest.mark.parametrize(
    "update,match",
    [
        ({"schema": "unknown"}, "Unknown"),
        ({"ranks": []}, "one state per rank"),
        ({"ranks": [{}, {}]}, "current CUDA device"),
    ],
)
def test_corrupt_distributed_rng_fails_closed(monkeypatch, update, match):
    local = behavior._capture_rng_state(current_cuda_only=True)
    monkeypatch.setattr(behavior, "_distributed_world", lambda: (2, 1))
    saved = {
        "schema": behavior._DISTRIBUTED_RNG_SCHEMA,
        "world_size": 2,
        "ranks": [local, local],
    }
    saved.update(update)
    with pytest.raises(ValueError, match=match):
        behavior._rank_rng_state(saved)


def test_cuda_capture_and_restore_touch_only_current_device(monkeypatch):
    state = torch.get_rng_state()
    restored = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: state)
    monkeypatch.setattr(
        torch.cuda, "get_rng_state_all", lambda: pytest.fail("all devices")
    )
    monkeypatch.setattr(
        torch.cuda, "set_rng_state", lambda value: restored.append(value)
    )
    monkeypatch.setattr(
        torch.cuda, "set_rng_state_all", lambda _: pytest.fail("all devices")
    )
    saved = behavior._capture_rng_state(current_cuda_only=True)
    assert saved["cuda_scope"] == "current_device"
    behavior._restore_rng_state(saved)
    assert len(restored) == 1 and torch.equal(restored[0], state)


class _Context:
    global_step = 0
    trainer = SimpleNamespace(datamodule=None)

    def _default_training_step(self, batch, batch_idx):
        return (float(torch.rand(())), float(np.random.rand()), random.random())


def _bound_behavior(seed):
    context = _Context()
    instance = behavior.WorldModelTrainingBehavior(distributed_noise_seed=seed)
    instance.bind(context)
    return instance, context  # Retain the behavior's weakly referenced context.


def test_fresh_training_seeds_ranks_once_without_changing_initialization(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    first_draws = []
    for rank in range(4):
        monkeypatch.setattr(behavior, "_distributed_world", lambda: (4, rank))
        initial_state = torch.get_rng_state().clone()
        instance, context = _bound_behavior(42)
        assert torch.equal(torch.get_rng_state(), initial_state)
        generator = torch.Generator().manual_seed(42 + rank)
        numpy_rng, python_rng = (
            np.random.RandomState(42 + rank),
            random.Random(42 + rank),
        )
        for step in range(2):
            expected = (
                float(torch.rand((), generator=generator)),
                float(numpy_rng.rand()),
                python_rng.random(),
            )
            actual = instance.training_step({"probe": {}}, step)
            assert actual == expected
            if step == 0:
                first_draws.append(actual)
        del context
    assert len(set(first_draws)) == 4


def test_saved_rank_rng_takes_precedence_over_fresh_noise_seed(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(behavior, "_distributed_world", lambda: (2, 1))
    torch.manual_seed(987)
    np.random.seed(987)
    random.seed(987)
    local = behavior._capture_rng_state(current_cuda_only=True)
    expected = _Context()._default_training_step({}, 0)
    instance, context = _bound_behavior(42)
    instance.on_load_checkpoint(
        {
            "mot_rng_state": {
                "schema": behavior._DISTRIBUTED_RNG_SCHEMA,
                "world_size": 2,
                "ranks": [local, local],
            }
        }
    )
    assert instance.training_step({"probe": {}}, 0) == expected
    assert instance._pending_rng is None and instance._training_rng_initialized
    del context


def test_fresh_seed_targets_current_cuda_device_only(monkeypatch):
    seeds = []
    monkeypatch.setattr(behavior, "_distributed_world", lambda: (4, 2))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed", seeds.append)
    monkeypatch.setattr(
        torch.cuda, "manual_seed_all", lambda _: pytest.fail("all devices")
    )
    instance, context = _bound_behavior(42)
    instance.training_step({"probe": {}}, 0)
    instance.training_step({"probe": {}}, 1)
    assert seeds == [44]
    del context


def test_default_does_not_reseed_legacy_training(monkeypatch):
    monkeypatch.setattr(behavior, "_distributed_world", lambda: (4, 2))
    state = behavior._capture_rng_state()
    expected = _Context()._default_training_step({}, 0)
    behavior._restore_rng_state(state)
    instance, context = _bound_behavior(None)
    assert instance.training_step({"probe": {}}, 0) == expected
    del context


@pytest.mark.parametrize("seed", [-1, 2**32, True, "42"])
def test_distributed_noise_seed_rejects_invalid_values(seed):
    with pytest.raises(ValueError, match="distributed_noise_seed"):
        behavior.WorldModelTrainingBehavior(distributed_noise_seed=seed)


_WORKER = r"""
import json
import os
from pathlib import Path
import random
import sys
import numpy as np
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf
from egomimic.pl_utils.world_model_behavior import WorldModelTrainingBehavior
from egomimic.trainHydra import _instantiate_trainer_plugins

class Probe(LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.4))
        self.behavior = WorldModelTrainingBehavior(distributed_noise_seed=42)
        self.behavior.bind(self)
        self.draws = []
    def on_train_start(self):
        # The ordinary entrypoint starts all ranks from the same global seed.
        seed = 1234
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    def training_step(self, batch, idx):
        return self.behavior.training_step({'probe': {'value': batch[0]}}, idx)
    def _default_training_step(self, batch, idx):
        draws = [float(torch.rand(())), float(np.random.rand()), random.random()]
        self.draws.append({'step': self.global_step, 'draws': draws})
        return (self.weight - sum(draws)).square()
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.01)
    def on_save_checkpoint(self, checkpoint):
        self.behavior.on_save_checkpoint(checkpoint)
    def on_load_checkpoint(self, checkpoint):
        self.behavior.on_load_checkpoint(checkpoint)

out, mode = Path(sys.argv[1]), sys.argv[2]
cfg = OmegaConf.create({'mmap_checkpoint': False, 'runtime': {
    'cluster_environment': 'torch_elastic', 'slurm_requeue_owner': 'none'}})
model = Probe()
trainer = Trainer(accelerator='cpu', devices=2, num_nodes=1,
    strategy=DDPStrategy(process_group_backend='gloo', find_unused_parameters=False),
    plugins=_instantiate_trainer_plugins(cfg), logger=False,
    enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
    max_steps=2 if mode == 'partial' else 4, max_epochs=-1,
    num_sanity_val_steps=0, limit_val_batches=0, default_root_dir=out)
loader = DataLoader(TensorDataset(torch.ones(8, 1)), batch_size=1, shuffle=False)
trainer.fit(model, train_dataloaders=loader,
    ckpt_path=str(out / 'partial.ckpt') if mode == 'resume' else None,
    weights_only=False)
trainer.save_checkpoint(out / (mode + '.ckpt'))
(out / f'{mode}-rank{trainer.global_rank}.json').write_text(json.dumps(model.draws))
"""


def test_real_torchrun_checkpoint_resume_preserves_each_rank(tmp_path):
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_gloo_available()
    ):
        pytest.skip("CPU distributed backend unavailable")
    worker = tmp_path / "worker.py"
    worker.write_text(textwrap.dedent(_WORKER))
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "",
            # A one-task Slurm parent must not override torchrun's actual world.
            "SLURM_JOB_ID": "999",
            "SLURM_NTASKS": "1",
            "SLURM_PROCID": "0",
            "SLURM_LOCALID": "0",
        }
    )
    for key in ("ICE_REQUEUE_OWNER", "ICE_CHILD_REQUEUE_DISABLED"):
        env.pop(key, None)
    for mode in ("full", "partial", "resume"):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                "--nproc-per-node=2",
                str(worker),
                str(tmp_path),
                mode,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    first = []
    for rank in range(2):
        full = json.loads((tmp_path / f"full-rank{rank}.json").read_text())
        partial = json.loads((tmp_path / f"partial-rank{rank}.json").read_text())
        resume = json.loads((tmp_path / f"resume-rank{rank}.json").read_text())
        assert full == partial + resume
        first.append(full[0]["draws"])
    assert first[0] != first[1]
    full = torch.load(tmp_path / "full.ckpt", weights_only=False)
    resume = torch.load(tmp_path / "resume.ckpt", weights_only=False)
    assert full["global_step"] == resume["global_step"] == 4
    assert torch.equal(full["state_dict"]["weight"], resume["state_dict"]["weight"])
    for name, value in full["optimizer_states"][0]["state"][0].items():
        assert torch.equal(value, resume["optimizer_states"][0]["state"][0][name])
    saved = resume["mot_rng_state"]
    assert saved["schema"] == behavior._DISTRIBUTED_RNG_SCHEMA
    assert saved["world_size"] == len(saved["ranks"]) == 2
    assert not torch.equal(
        saved["ranks"][0]["torch_cpu"], saved["ranks"][1]["torch_cpu"]
    )
