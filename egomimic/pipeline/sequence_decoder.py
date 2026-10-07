"""Explicit uniform time-grid conversion for interpolated training targets."""

import numpy as np
import torch


def _sequence_shape(shape, name):
    try:
        shape = tuple(shape)
    except TypeError as error:
        raise ValueError(f"{name} must be [horizon, width]") from error
    if len(shape) != 2 or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError(f"{name} must contain positive sequence dimensions")
    return shape


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

    @property
    def output_shape(self):
        return self.output_horizon, self.native_shape[1]

    def validate_inference_contract(self, native_shape, canonical_shape):
        if _sequence_shape(native_shape, "native_shape") != self.native_shape:
            raise ValueError(
                "Uniform time-grid native shape differs from the model contract"
            )
        if _sequence_shape(canonical_shape, "canonical_shape") != self.output_shape:
            raise ValueError(
                "Uniform time-grid output shape differs from the model contract"
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

    def validate_inference_contract(self, native_shape, canonical_shape):
        current = _sequence_shape(native_shape, "native_shape")
        canonical = _sequence_shape(canonical_shape, "canonical_shape")
        for index, decoder in enumerate(self.decoders):
            declared_input = getattr(decoder, "native_shape", None)
            if declared_input is None:
                declared_input = getattr(decoder, "shape", None)
            if declared_input is not None and current != _sequence_shape(
                declared_input, "decoder native_shape"
            ):
                raise ValueError(
                    f"Sequential decoder {index} has an incompatible input shape"
                )
            output = getattr(decoder, "output_shape", None)
            if output is None:
                if index == len(self.decoders) - 1:
                    output = canonical
                else:
                    next_decoder = self.decoders[index + 1]
                    output = getattr(next_decoder, "native_shape", None)
                    if output is None:
                        output = getattr(next_decoder, "shape", None)
                    if output is None:
                        raise ValueError(
                            f"Sequential decoder {index} needs a declared intermediate shape"
                        )
            output = _sequence_shape(output, "decoder output_shape")
            validate = getattr(decoder, "validate_inference_contract", None)
            if callable(validate):
                validate(current, output)
            current = output
        if current != canonical:
            raise ValueError(
                "Sequential decoder output shape differs from the model contract"
            )

    def __call__(self, actions):
        for decoder in self.decoders:
            actions = decoder(actions)
        return actions
