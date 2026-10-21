"""Fixed-window planar decoding using the shared canonical ARC clocks."""

import numpy as np
import torch

from egomimic.rldb.zarr.planar_arc_clock import PlanarArcClockCodec


class PlanarArcClockNativeDecoder:
    preserves_decoded_timing = True

    def __init__(self, action_horizon=80, **codec_kwargs):
        self.codec = PlanarArcClockCodec(horizon=action_horizon, **codec_kwargs)
        self.action_horizon = self.codec.horizon
        self.native_action_dim = self.codec.native_action_dim

    def decode(self, actions, context=None):
        del context
        tensor = torch.is_tensor(actions)
        value = (
            actions.detach().cpu().double().numpy() if tensor else np.asarray(actions)
        )
        if value.ndim < 2 or value.shape[-2:] != self.codec.shape:
            raise ValueError(f"Expected (...,{self.codec.shape}) ARC predictions")
        leading = value.shape[:-2]
        native = np.stack(
            [self.codec.decode(row) for row in value.reshape(-1, *self.codec.shape)]
        ).reshape(*leading, self.action_horizon, self.native_action_dim)
        if tensor:
            return torch.as_tensor(native, device=actions.device, dtype=actions.dtype)
        return native.astype(value.dtype if value.dtype.kind == "f" else np.float32)

    def waypoint_clocks(self, actions):
        value = (
            actions.detach().cpu().double().numpy()
            if torch.is_tensor(actions)
            else actions
        )
        return self.codec.waypoint_clocks(value)

    def representation_context(self):
        return self.codec.representation_context()

    __call__ = decode
