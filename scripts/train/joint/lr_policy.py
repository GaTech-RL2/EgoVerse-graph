"""Declared optimizer policy shared by admission, smoke and restart checks.

This validates configuration and observed state; it never changes an optimizer.
"""
import math


def validate(policy):
    if not isinstance(policy, dict) or set(policy) != {"peak_lr", "warmup_optimizer_steps", "schedule"}:
        raise ValueError("Explicit complete LR policy required")
    rate, warmup = policy["peak_lr"], policy["warmup_optimizer_steps"]
    if type(rate) not in (float, int) or not math.isfinite(rate) or rate <= 0 or type(warmup) is not int:
        raise ValueError("Invalid LR or warmup")
    if not ((policy["schedule"] == "constant" and warmup == 0)
            or (policy["schedule"] == "linear_warmup_then_constant" and warmup > 0)):
        raise ValueError("Unsupported LR schedule")
    return dict(policy)


def from_config(model):
    scheduler = model.get("scheduler")
    warmup = 0
    if scheduler is not None:
        if (scheduler.get("_target_") != "egomimic.utils.schedulers.linear_warmup_constant_scheduler"
                or model.get("scheduler_interval") != "step" or model.get("scheduler_frequency") != 1):
            raise ValueError("Unsupported configured scheduler")
        warmup = scheduler.get("warmup_steps")
    return validate(dict(peak_lr=model["optimizer"]["lr"], warmup_optimizer_steps=warmup,
                         schedule="constant" if scheduler is None else "linear_warmup_then_constant"))


def update_lr(policy, completed_steps):
    policy = validate(policy)
    if type(completed_steps) is not int or completed_steps < 0:
        raise ValueError("Actual nonnegative optimizer step required")
    warmup = policy["warmup_optimizer_steps"]
    return policy["peak_lr"] * (min((completed_steps + 1) / warmup, 1.0) if warmup else 1.0)


def verify_scheduler(policy, states, completed_steps):
    policy = validate(policy)
    warmup = policy["warmup_optimizer_steps"]
    if not warmup:
        if states != []:
            raise ValueError("Constant LR must not restore a scheduler")
    elif (len(states) != 1 or states[0]["last_epoch"] != completed_steps
          or states[0]["base_lrs"] != [policy["peak_lr"]]
          or states[0]["lr_lambdas"] != [{"warmup_steps": warmup}]):
        raise ValueError("Wrong exact scheduler continuation state")
    return True
