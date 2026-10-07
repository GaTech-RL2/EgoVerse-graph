"""Canonical tensor action boundary for non-hardware inference consumers."""

import torch


class CanonicalSequenceAdapter:
    def __init__(self, shape, decoder=None):
        self.shape = tuple(shape)
        if len(self.shape) != 2 or any(type(n) is not int or n < 1 for n in self.shape):
            raise ValueError("Canonical sequence shape must be [horizon, width]")
        self.decoder = decoder

    def __call__(self, native):
        output = self.decoder(native) if self.decoder is not None else native
        output = torch.as_tensor(output)
        if (
            output.ndim != 3
            or tuple(output.shape[1:]) != self.shape
            or not torch.isfinite(output).all()
        ):
            raise ValueError(
                f"Canonical action sequence must be finite (B, {self.shape[0]}, {self.shape[1]})"
            )
        return output
