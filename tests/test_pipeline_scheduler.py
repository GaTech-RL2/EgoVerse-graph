import pytest
import torch

from egomimic.utils.schedulers import (
    linear_warmup_constant_scheduler,
    warmup_cosine_scheduler,
)


def test_linear_warmup_uses_exact_optimizer_update_boundaries():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=5e-6)
    scheduler = linear_warmup_constant_scheduler(optimizer, warmup_steps=3000)
    rates = []
    for _ in range(3002):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert rates[0] == pytest.approx(5e-6 / 3000)
    assert rates[1499] == pytest.approx(2.5e-6)
    assert rates[2998] == pytest.approx(5e-6 * 2999 / 3000)
    assert rates[2999:] == pytest.approx([5e-6] * 3)


@pytest.mark.parametrize("completed", [50, 2998, 2999, 3000])
def test_linear_warmup_full_state_resume(completed):
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))], lr=5e-6)
    scheduler = linear_warmup_constant_scheduler(optimizer, warmup_steps=3000)
    for _ in range(completed):
        optimizer.step()
        scheduler.step()
    optimizer_state, scheduler_state = optimizer.state_dict(), scheduler.state_dict()
    restored_optimizer = torch.optim.AdamW(
        [torch.nn.Parameter(torch.ones(()))], lr=5e-6
    )
    restored_scheduler = linear_warmup_constant_scheduler(
        restored_optimizer, warmup_steps=3000
    )
    restored_optimizer.load_state_dict(optimizer_state)
    restored_scheduler.load_state_dict(scheduler_state)
    assert restored_scheduler.state_dict() == scheduler_state
    assert scheduler_state["lr_lambdas"] == [{"warmup_steps": 3000}]
    for _ in range(4):
        assert (
            restored_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
        )
        optimizer.step()
        scheduler.step()
        restored_optimizer.step()
        restored_scheduler.step()


@pytest.mark.parametrize("steps", [0, -1, 1.5, True])
def test_linear_warmup_rejects_invalid_length(steps):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=5e-6)
    with pytest.raises(ValueError, match="positive integer"):
        linear_warmup_constant_scheduler(optimizer, warmup_steps=steps)


def test_warmup_cosine_scheduler_reaches_peak_then_eta_min():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = warmup_cosine_scheduler(
        optimizer,
        max_steps=6,
        warmup_steps=2,
        warmup_start_factor=0.1,
        eta_min=0.2,
    )
    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(6):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])

    assert rates[:3] == pytest.approx([0.1, 0.55, 1.0])
    assert rates[-1] == pytest.approx(0.2)
