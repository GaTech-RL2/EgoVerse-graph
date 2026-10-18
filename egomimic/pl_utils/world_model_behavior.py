"""Use the shared Lightning wrapper; schedule follows actual optimizer steps."""

import random

import numpy as np
import torch

from egomimic.pl_utils.training_behavior import TrainingBehavior

_DISTRIBUTED_RNG_SCHEMA = "mot_rng_state.distributed.v1"


def _distributed_world():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(), torch.distributed.get_rank()
    return 1, 0


def _capture_rng_state(*, current_cuda_only=False):
    cuda = []
    if torch.cuda.is_available():
        cuda = ([torch.cuda.get_rng_state()] if current_cuda_only
                else torch.cuda.get_rng_state_all())
    state = {
        "torch_cpu": torch.get_rng_state(), "torch_cuda": cuda,
        "numpy": np.random.get_state(), "python": random.getstate(),
    }
    if current_cuda_only:
        state["cuda_scope"] = "current_device"
    return state


def _rank_rng_state(saved):
    """Select at the first train step, after distributed initialization."""
    world_size, rank = _distributed_world()
    if "schema" not in saved:
        if world_size != 1:
            raise ValueError("Exact MoT DDP resume requires per-rank RNG state")
        return saved
    if saved.get("schema") != _DISTRIBUTED_RNG_SCHEMA:
        raise ValueError("Unknown MoT RNG checkpoint schema")
    if saved.get("world_size") != world_size:
        raise ValueError("Exact MoT resume requires the saved RNG world size")
    ranks = saved.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != world_size:
        raise ValueError("MoT RNG checkpoint must contain one state per rank")
    state = ranks[rank]
    if not isinstance(state, dict) or state.get("cuda_scope") != "current_device":
        raise ValueError("Distributed MoT RNG state must bind the current CUDA device")
    if len(state.get("torch_cuda", [])) > 1:
        raise ValueError("Distributed MoT RNG state contains multiple CUDA devices")
    return state


def _restore_rng_state(state):
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("Cannot restore CUDA RNG on a CPU training runtime")
        if state.get("cuda_scope") == "current_device":
            torch.cuda.set_rng_state(state["torch_cuda"][0].cpu())
        else:
            torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


class WorldModelTrainingBehavior(TrainingBehavior):
    def __init__(self, distributed_noise_seed=None):
        super().__init__()
        if distributed_noise_seed is not None and (
            type(distributed_noise_seed) is not int
            or not 0 <= distributed_noise_seed < 2**32
        ):
            raise ValueError("distributed_noise_seed must be a uint32 integer or None")
        self.distributed_noise_seed = distributed_noise_seed
        self._training_rng_initialized = False

    def on_validation_epoch_end(self):
        if self.context.evaluator is not None:
            self.context.evaluator.log_epoch_metrics()

    def training_step(self, batch, batch_idx):
        pending = getattr(self, "_pending_rng", None)
        if pending is not None:
            _restore_rng_state(_rank_rng_state(pending))
            self._pending_rng = None
            self._training_rng_initialized = True
        elif not self._training_rng_initialized:
            world_size, rank = _distributed_world()
            if self.distributed_noise_seed is not None and world_size > 1:
                # Model construction remains identically seeded on every rank.
                # Fresh training then gets distinct streams; resume above wins.
                seed = (self.distributed_noise_seed + rank) % (2**32)
                torch.random.default_generator.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)  # Current rank's device only.
                np.random.seed(seed)
                random.seed(seed)
            self._training_rng_initialized = True
        data = self.context.trainer.datamodule
        if getattr(data, "checkpointable_sampling", False):
            cursors = {values["sampling_cursor"] for values in batch.values()}
            if len(cursors) != 1:
                raise ValueError("Expected one consumed MoT batch cursor")
            data.acknowledge_batch(cursors.pop())
        batch = {
            name: {**values, "optimizer_step": int(self.context.global_step)}
            for name, values in batch.items()
        }
        return self.context._default_training_step(batch, batch_idx)

    def on_save_checkpoint(self, checkpoint):
        world_size, _ = _distributed_world()
        local = _capture_rng_state(current_cuda_only=world_size > 1)
        if world_size == 1:
            checkpoint["mot_rng_state"] = local
            return
        # Lightning calls model checkpoint hooks on every rank before its
        # strategy limits the file write to rank zero. This collective must
        # remain outside a rank-zero guard.
        ranks = [None] * world_size
        torch.distributed.all_gather_object(ranks, local)
        checkpoint["mot_rng_state"] = {
            "schema": _DISTRIBUTED_RNG_SCHEMA,
            "world_size": world_size,
            "ranks": ranks,
        }

    def on_load_checkpoint(self, checkpoint):
        self._pending_rng = checkpoint.get("mot_rng_state")
        if (getattr(self.context.trainer.datamodule, "checkpointable_sampling", False)
                and self._pending_rng is None):
            raise ValueError("Exact MoT resume requires saved model RNG state")
