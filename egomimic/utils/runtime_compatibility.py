"""Explicit, opt-in historical execution semantics (not a precision policy)."""


def math_sdpa_context():
    """Select only math SDPA using the runtime's supported public API."""
    import torch
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ModuleNotFoundError as exc:
        if exc.name != "torch.nn.attention":
            raise
        return torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False
        )
    return sdpa_kernel(SDPBackend.MATH)


def autocast_noise_dtype(device_type: str):
    """Query existing autocast without enabling a context or changing RNG."""
    import torch
    try:
        enabled = torch.is_autocast_enabled(device_type)
    except TypeError:
        if device_type == "cpu":
            enabled = torch.is_autocast_cpu_enabled()
        elif device_type == "cuda":
            enabled = torch.is_autocast_enabled()
        else:
            enabled = False
    if not enabled:
        return torch.get_default_dtype()
    getter = getattr(torch, "get_autocast_dtype", None)
    if getter is not None:
        return getter(device_type)
    if device_type == "cpu":
        return torch.get_autocast_cpu_dtype()
    if device_type == "cuda":
        return torch.get_autocast_gpu_dtype()
    raise ValueError(f"Unsupported autocast device {device_type!r}")


def validate_compatibility_mode(mode: str) -> str:
    if mode not in ("current", "legacy_c12"):
        raise ValueError("compatibility_mode must be 'current' or 'legacy_c12'")
    return mode
