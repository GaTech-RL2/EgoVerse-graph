"""Decode native ARC policy outputs before robot frame transforms.

Ported from aidan/shorts-extreme (55932d99), rollout-arc.py. This module has no
robot I/O. Call it after action unnormalization, then apply the controller's
camera/base frame transforms to the returned canonical poses.
"""

import numpy as np
import torch

from egomimic.rldb.zarr.arc_length_tokenizer import TokenizeBimanualArcLengthCartesian
from egomimic.rldb.zarr.e1_arc_tokenizer import TokenizeBimanualArcLengthE1

E1_VELOCITY_MODE = {"e1_dur": "dur", "e1_logdur": "logdur", "e1_profile": "profile"}
ARC_TOKEN_LAYOUTS = ("lab", *E1_VELOCITY_MODE)


class BimanualArcDecoder:
    def __init__(self, token_layout="lab", min_distance_unit=0.4,
                 resampled_vector_length=100, dt=1/30, action_horizon=100):
        if token_layout not in ARC_TOKEN_LAYOUTS:
            raise ValueError(f"token_layout must be one of {ARC_TOKEN_LAYOUTS}")
        self.M = int(resampled_vector_length)
        self.action_horizon = int(action_horizon)
        if self.M < 2 or self.action_horizon < 1 or dt <= 0 or min_distance_unit <= 0:
            raise ValueError("Invalid ARC distance, time, horizon or waypoint count")
        self.shape = (self.M + 1, 14) if token_layout == "lab" else (self.M, 16)
        kwargs = dict(min_distance_unit=min_distance_unit,
                      resampled_vector_length=self.M, dt=dt)
        self.codec = (TokenizeBimanualArcLengthCartesian(**kwargs) if token_layout == "lab"
                      else TokenizeBimanualArcLengthE1(**kwargs, velocity_norm="path",
                                                     velocity_mode=E1_VELOCITY_MODE[token_layout]))

    def __call__(self, native_tokens):
        if torch.is_tensor(native_tokens):
            native_tokens = native_tokens.detach().double().cpu().numpy()
        values = np.asarray(native_tokens, dtype=np.float64)
        if values.ndim == 2:
            values = values[None]
        if values.ndim != 3 or values.shape[1:] != self.shape:
            raise ValueError(f"Expected native tokens (B,{self.shape[0]},{self.shape[1]}), got {values.shape}")
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("Native tokens must be nonempty and finite")
        return np.stack([self.codec.detokenize(row, action_horizon=self.action_horizon)
                         for row in values])


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Decode saved native ARC tokens; no robot I/O")
    parser.add_argument("tokens", help="Input .npy file, already unnormalized")
    parser.add_argument("output", help="New .npy file for (B,H,14) canonical poses")
    parser.add_argument("--arc-token-layout", choices=ARC_TOKEN_LAYOUTS, default="lab")
    parser.add_argument("--arc-min-distance-unit", type=float, default=0.4)
    parser.add_argument("--arc-resampled-vector-length", type=int, default=100)
    parser.add_argument("--arc-dt", type=float, default=1/30)
    parser.add_argument("--arc-rollout-horizon", type=int, default=100)
    args = parser.parse_args()
    decoder = BimanualArcDecoder(args.arc_token_layout, args.arc_min_distance_unit,
                                args.arc_resampled_vector_length, args.arc_dt, args.arc_rollout_horizon)
    result = decoder(np.load(args.tokens, allow_pickle=False))
    with open(args.output, "xb") as output:
        np.save(output, result, allow_pickle=False)


if __name__ == "__main__":
    main()
