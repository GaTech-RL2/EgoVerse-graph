"""Regional compilation must preserve ordinary distributed gradient reduction."""

import copy
import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from egomimic.models.compilation import compile_region


class RegionModel(torch.nn.Module):
    def __init__(self, *, compiled=False):
        super().__init__()
        self.norm = torch.nn.LayerNorm(4)
        self.linear = torch.nn.Linear(4, 3)
        self.execute = compile_region(
            self.region, mode="default" if compiled else "none", dynamic=True,
            backend="pixel_region_ddp_test",
        )

    def region(self, value):
        return self.linear(self.norm(value)).square()

    def forward(self, value):
        return self.execute(value)


def _input(rank):
    return torch.linspace(-1, 2, (rank + 2) * 3 * 4).reshape(rank + 2, 3, 4)


def _worker(rank, directory):
    torch.set_num_threads(1)
    from torch._dynamo.backends.registry import lookup_backend, register_backend

    policies = []
    backend = lookup_backend("aot_eager")

    def checked_backend(graph, inputs):
        policies.append(torch._dynamo.config.optimize_ddp)
        return backend(graph, inputs)

    register_backend(checked_backend, name="pixel_region_ddp_test")
    dist.init_process_group(
        "gloo", init_method=Path(directory, "rendezvous").as_uri(), rank=rank, world_size=2,
    )
    try:
        torch.manual_seed(42)
        network = RegionModel(compiled=True)
        reference = RegionModel()
        reference.load_state_dict(copy.deepcopy(network.state_dict()), strict=True)
        expected = sum(reference(_input(r)).mean() for r in range(2)) / 2
        expected.backward()
        wrapped = torch.nn.parallel.DistributedDataParallel(network)
        original_policy = torch._dynamo.config.optimize_ddp
        loss = wrapped(_input(rank)).mean()
        loss.backward()
        assert policies and all(value is False for value in policies)
        assert torch._dynamo.config.optimize_ddp == original_policy
        for actual, wanted in zip(network.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, wanted.grad, rtol=1e-5, atol=1e-6)
        torch.optim.SGD(network.parameters(), lr=0.01).step()
        state = torch.cat([parameter.detach().reshape(-1) for parameter in network.parameters()])
        ranks = [torch.empty_like(state) for _ in range(2)]
        dist.all_gather(ranks, state)
        assert torch.equal(ranks[0], ranks[1])
        Path(directory, f"rank{rank}.json").write_text(json.dumps({
            "rank": rank, "gradient_average_verified": True,
            "compiled_regions": len(policies), "identical_updated_weights": True,
        }))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_regional_compile_preserves_two_rank_ddp_gradient_average(tmp_path):
    torch.multiprocessing.spawn(_worker, args=(str(tmp_path),), nprocs=2, join=True)
    assert all(json.loads((tmp_path / f"rank{rank}.json").read_text())[
        "gradient_average_verified"
    ] for rank in range(2))


def test_region_restores_ddp_policy_after_backend_failure(monkeypatch):
    def fail(*args, **kwargs):
        assert torch._dynamo.config.optimize_ddp is False
        raise RuntimeError("test compiler failure")

    monkeypatch.setattr(torch, "compile", lambda *args, **kwargs: fail)
    original = torch._dynamo.config.optimize_ddp
    region = compile_region(lambda x: x + 1, mode="default")
    with pytest.raises(RuntimeError, match="test compiler failure"):
        region(torch.ones(1))
    assert torch._dynamo.config.optimize_ddp == original
