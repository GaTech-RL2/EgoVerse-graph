"""Explicit uniform time-grid conversion for interpolated training targets."""

import numpy as np
import torch


class UniformTimeGridDecoder:
    def __init__(
        self,
        native_shape,
        native_dt=None,
        output_horizon=None,
        output_dt=None,
        angular_indices=(),
        *,
        native_duration=None,
    ):
        self.native_shape = tuple(native_shape)
        self.output_horizon = output_horizon
        self.angular_indices = tuple(angular_indices)
        if (
            len(self.native_shape) != 2
            or any(type(n) is not int or n < 1 for n in self.native_shape)
            or type(self.output_horizon) is not int
            or self.output_horizon < 1
        ):
            raise ValueError("Declare positive native and output sequence dimensions")
        if (native_dt is None) == (native_duration is None):
            raise ValueError("Declare exactly one native_dt or native_duration")
        if native_duration is not None:
            if self.native_shape[0] < 2:
                raise ValueError("A duration requires at least two native rows")
            native_dt = float(native_duration) / (self.native_shape[0] - 1)
        self.native_dt, self.output_dt = float(native_dt), float(output_dt)
        if not all(np.isfinite(v) and v > 0 for v in (self.native_dt, self.output_dt)):
            raise ValueError("Time-grid periods must be finite and positive")
        if any(
            type(i) is not int or not 0 <= i < self.native_shape[1]
            for i in self.angular_indices
        ):
            raise ValueError("Angular channel indices must be inside the native action")
        self.native_time = np.arange(self.native_shape[0]) * self.native_dt
        self.output_time = np.arange(self.output_horizon) * self.output_dt
        if self.output_time[-1] > self.native_time[-1] + 1e-8:
            raise ValueError(
                "Output time grid exceeds the trained trajectory; extrapolation is unsupported"
            )

    def __call__(self, actions):
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)
        if actions.ndim != 3 or tuple(actions.shape[1:]) != self.native_shape:
            raise ValueError(
                f"Expected native (B,{self.native_shape}), got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("Native actions must be finite before resampling")
        values = actions.astype(np.float64, copy=True)
        values[..., self.angular_indices] = np.unwrap(
            values[..., self.angular_indices], axis=1
        )
        output = np.empty((len(values), self.output_horizon, self.native_shape[1]))
        for b, sequence in enumerate(values):
            for channel in range(self.native_shape[1]):
                output[b, :, channel] = np.interp(
                    self.output_time, self.native_time, sequence[:, channel]
                )
        output[..., self.angular_indices] = (
            output[..., self.angular_indices] + np.pi
        ) % (2 * np.pi) - np.pi
        return output


class SequentialActionDecoder:
    """Compose configured representation and time-grid conversions in order."""

    def __init__(self, decoders):
        self.decoders = tuple(decoders)
        if not self.decoders or not all(callable(step) for step in self.decoders):
            raise ValueError("Declare a nonempty sequence of callable decoders")

    def __call__(self, actions):
        for decoder in self.decoders:
            actions = decoder(actions)
        return actions
