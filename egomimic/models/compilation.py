"""Explicit compilation of tensor-only regions, without eager fallback."""

from functools import wraps

import torch
from torch._functorch import config as _aot_config


def compile_region(function, *, mode="none", dynamic=False, backend="inductor"):
    if mode not in ("none", "default", "reduce-overhead", "max-autotune"):
        raise ValueError("Unknown compile mode")
    if type(dynamic) is not bool:
        raise ValueError("compile dynamic must be a boolean")
    if not isinstance(backend, str) or not backend:
        raise ValueError("compile backend must be a nonempty string")
    if mode == "none":
        return function
    if torch._dynamo.config.suppress_errors:
        raise RuntimeError("Compilation requires suppress_errors=False; no eager fallback")
    compiled = torch.compile(
        function, backend=backend, mode=mode, dynamic=dynamic, fullgraph=True
    )

    @wraps(function)
    def execute(*args, **kwargs):
        # Compilation is lazy, so the policy must enclose invocation, including
        # retracing, rather than only the torch.compile call above. Lightning's
        # standard AMP backward runs outside the forward autocast context.
        # These are already bounded tensor regions inside the ordinary DDP
        # wrapper. DDPOptimizer must not repartition each region against the
        # whole model's gradient buckets (Torch 2.9 can emit invalid SymInt
        # subgraph outputs). This affects compilation only; DDP's reducer and
        # gradient synchronization remain active. Restore the caller's policy.
        with torch._dynamo.config.patch(optimize_ddp=False):
            if hasattr(_aot_config, "backward_pass_autocast"):
                with _aot_config.patch(backward_pass_autocast="off"):
                    return compiled(*args, **kwargs)
            # Older runtimes can still compile ordinary FP32 regions, but cannot
            # express the required AOT backward policy for our CPU/CUDA mixed mode.
            if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
                raise RuntimeError(
                    "Compiled autocast requires a PyTorch runtime with "
                    "torch._functorch.config.backward_pass_autocast; "
                    "no unsafe mixed-precision fallback"
                )
            return compiled(*args, **kwargs)

    return execute
