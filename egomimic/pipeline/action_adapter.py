"""Canonical tensor action boundary and explicit native decoder context."""

from collections.abc import Mapping

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def decoder_context_keys(decoder):
    """A decoder declares its required context; never infer it from its family."""
    keys = tuple(getattr(decoder, "context_keys", ()))
    if any(not isinstance(key, str) or not key for key in keys) or len(
        set(keys)
    ) != len(keys):
        raise ValueError("Decoder context_keys must be distinct nonempty strings")
    return keys


def decode_actions(decoder, native, *, context=None):
    if decoder is None:
        return native
    keys = decoder_context_keys(decoder)
    context = {} if context is None else context
    if missing := set(keys) - context.keys():
        raise ValueError(f"Missing native decoder context: {sorted(missing)}")
    return decoder(native, **{key: context[key] for key in keys})


def validate_decode_context(decoder, bindings):
    bindings = {} if bindings is None else bindings
    if not isinstance(bindings, Mapping):
        raise TypeError("decode_context must map context names to explicit providers")
    if set(bindings) != set(decoder_context_keys(decoder)):
        raise ValueError(
            "decode_context must bind exactly the decoder's declared context_keys"
        )
    for provider in bindings.values():
        if not callable(provider) or not getattr(provider, "input_keys", ()):
            raise TypeError(
                "Decode context providers must be callable and declare input_keys"
            )
        if any(not isinstance(key, str) or not key for key in provider.input_keys):
            raise ValueError("Decode context input_keys must be nonempty strings")
    return dict(bindings)


def resolve_decode_context(bindings, observations):
    for provider in bindings.values():
        if missing := set(provider.input_keys) - observations.keys():
            raise ValueError(f"Missing native context observations: {sorted(missing)}")
    return {name: provider(observations) for name, provider in bindings.items()}


class CartesianCurrentAction:
    """Convert native measured proprio to the explicitly declared action frame.

    EEF-relative current XYZ/rotation is identity, not the model-frame proprio
    position. Grippers always retain their measured opening. Model-frame inputs
    are already in that frame (station calibration belongs to the robot adapter).
    This provider never normalizes data or substitutes a predicted first target.
    """

    def __init__(self, key, action_frame, rotation_mode="euler"):
        if not isinstance(key, str) or not key:
            raise ValueError(
                "Current-action context requires an explicit observation key"
            )
        if action_frame not in ("eef_frame", "model_frame"):
            raise ValueError("Current-action context requires eef_frame or model_frame")
        if rotation_mode not in ("euler", "6D"):
            raise ValueError("Current-action context requires euler or 6D rotation")
        self.key, self.action_frame, self.rotation_mode = (
            key,
            action_frame,
            rotation_mode,
        )
        self.input_keys = (key,)

    def __call__(self, observations):
        values = observations[self.key]
        if torch.is_tensor(values):
            values = values.detach().cpu().numpy()
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 3:
            values = values[:, -1]  # latest observation in an assembled history
        width = 14 if self.rotation_mode == "euler" else 20
        if (
            values.ndim != 2
            or values.shape[1] != width
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"Native current pose must be finite (B,{width})")
        result = np.zeros((len(values), 14), dtype=np.float64)
        half = width // 2
        for arm, offset in enumerate((0, 7)):
            pose = values[:, arm * half : (arm + 1) * half]
            result[:, offset + 6] = pose[:, -1]
            if self.action_frame == "eef_frame":
                continue
            result[:, offset : offset + 3] = pose[:, :3]
            if self.rotation_mode == "euler":
                result[:, offset + 3 : offset + 6] = pose[:, 3:6]
            else:
                a, b = pose[:, 3:6].copy(), pose[:, 6:9].copy()
                lengths = np.linalg.norm(a, axis=1, keepdims=True)
                if np.any(lengths < 1e-8):
                    raise ValueError("Degenerate current 6D rotation")
                a /= lengths
                b -= a * np.sum(a * b, axis=1, keepdims=True)
                lengths = np.linalg.norm(b, axis=1, keepdims=True)
                if np.any(lengths < 1e-8):
                    raise ValueError("Degenerate current 6D rotation")
                b /= lengths
                matrix = np.stack((a, b, np.cross(a, b)), axis=-1)
                result[:, offset + 3 : offset + 6] = Rotation.from_matrix(
                    matrix
                ).as_euler("ZYX")
        return result


def execution_prefix(prediction, *, decoder=None, replan_every=None, time_axis=0):
    """Select a prefix using a declared decoder capability, without family dispatch."""
    steps = getattr(decoder, "execution_steps", None)
    count = steps() if callable(steps) else None
    count = replan_every if count is None else count
    if count is None:
        return prediction
    if type(count) is not int or count <= 0:
        raise ValueError("Executable action prefix must be a positive integer")
    indices = [slice(None)] * prediction.ndim
    indices[time_axis] = slice(0, min(count, prediction.shape[time_axis]))
    return prediction[tuple(indices)]


class CanonicalSequenceAdapter:
    def __init__(self, shape, decoder=None, decode_context=None):
        self.shape = tuple(shape)
        if len(self.shape) != 2 or any(type(n) is not int or n < 1 for n in self.shape):
            raise ValueError("Canonical sequence shape must be [horizon, width]")
        self.decoder = decoder
        self.decode_context = validate_decode_context(decoder, decode_context)

    @property
    def context_input_keys(self):
        return tuple(
            dict.fromkeys(
                key
                for provider in self.decode_context.values()
                for key in provider.input_keys
            )
        )

    def __call__(self, native, *, observations=None):
        context = resolve_decode_context(self.decode_context, observations or {})
        output = decode_actions(self.decoder, native, context=context)
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
