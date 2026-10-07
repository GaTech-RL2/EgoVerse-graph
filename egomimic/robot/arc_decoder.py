"""Decode native ARC policy outputs before robot frame transforms.

Ported from aidan/shorts-extreme (55932d99), rollout-arc.py. This module has no
robot I/O. Call it after action unnormalization, then apply the controller's
camera/base frame transforms to the returned canonical poses.
"""

import numpy as np
import torch

from egomimic.rldb.zarr.arc_length_tokenizer import (
    TokenizeBimanualArcLengthCartesian,
    bimanual_arc_token_shape,
)
from egomimic.rldb.zarr.e1_arc_tokenizer import (
    DEFAULT_ROTATION_DISTANCE_UNIT,
    E1_ARCDURHYB_DIM,
    TokenizeBimanualArcLengthE1,
)

E1_VELOCITY_MODE = {
    "e1_dur": "dur",
    "e1_logdur": "logdur",
    "e1_profile": "profile",
    "e1_durhyb": "durhyb",
    "e1_profhyb": "profhyb",
}
E1_HYBRID_LAYOUTS = {"e1_durhyb", "e1_profhyb"}
ARC_TOKEN_LAYOUTS = ("lab", *E1_VELOCITY_MODE)


class BimanualArcDecoder:
    def __init__(
        self,
        token_layout="lab",
        min_distance_unit=0.4,
        resampled_vector_length=100,
        dt=1 / 30,
        action_horizon=100,
        rotation_distance_unit=None,
    ):
        if token_layout not in ARC_TOKEN_LAYOUTS:
            raise ValueError(f"token_layout must be one of {ARC_TOKEN_LAYOUTS}")
        self.M = int(resampled_vector_length)
        self.action_horizon = int(action_horizon)
        if self.M < 2 or self.action_horizon < 1 or dt <= 0 or min_distance_unit <= 0:
            raise ValueError("Invalid ARC distance, time, horizon or waypoint count")
        self.shape = (
            (self.M + 1, 14)
            if token_layout == "lab"
            else (self.M, E1_ARCDURHYB_DIM if token_layout in E1_HYBRID_LAYOUTS else 16)
        )
        kwargs = dict(
            min_distance_unit=min_distance_unit, resampled_vector_length=self.M, dt=dt
        )
        if token_layout in E1_HYBRID_LAYOUTS:
            rotation_distance_unit = float(
                DEFAULT_ROTATION_DISTANCE_UNIT
                if rotation_distance_unit is None
                else rotation_distance_unit
            )
            if not np.isfinite(rotation_distance_unit) or rotation_distance_unit <= 0:
                raise ValueError(
                    "Hybrid ARC rotation distance must be finite and positive"
                )
            kwargs["rotation_distance_unit"] = rotation_distance_unit
        elif rotation_distance_unit is not None:
            raise ValueError("A rotation distance requires an E1 hybrid token layout")
        self.codec = (
            TokenizeBimanualArcLengthCartesian(**kwargs)
            if token_layout == "lab"
            else TokenizeBimanualArcLengthE1(
                **kwargs,
                velocity_norm="path",
                velocity_mode=E1_VELOCITY_MODE[token_layout],
            )
        )

    def __call__(self, native_tokens):
        if torch.is_tensor(native_tokens):
            native_tokens = native_tokens.detach().double().cpu().numpy()
        values = np.asarray(native_tokens, dtype=np.float64)
        if values.ndim == 2:
            values = values[None]
        if values.ndim != 3 or values.shape[1:] != self.shape:
            raise ValueError(
                f"Expected native tokens (B,{self.shape[0]},{self.shape[1]}), got {values.shape}"
            )
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("Native tokens must be nonempty and finite")
        return np.stack(
            [
                self.codec.detokenize(row, action_horizon=self.action_horizon)
                for row in values
            ]
        )


class BimanualIntervalArcDecoder:
    """Decode an explicitly declared interval layout using the training codec."""

    def __init__(
        self,
        velocity_mode,
        min_distance_unit,
        resampled_vector_length,
        dt,
        action_horizon,
        velocity_layout="stacked",
        rotation_distance_unit=None,
        arc_chunking_mode=None,
    ):
        if velocity_mode not in {"per_waypoint", "duration"}:
            raise ValueError(
                "Deployment requires per-waypoint timing, not chunk-mean timing"
            )
        self.action_horizon = int(action_horizon)
        self.M = int(resampled_vector_length)
        if self.action_horizon < 1 or self.M < 2:
            raise ValueError("Invalid ARC output horizon or waypoint count")
        self.codec = TokenizeBimanualArcLengthCartesian(
            min_distance_unit=min_distance_unit,
            resampled_vector_length=self.M,
            dt=dt,
            velocity_mode=velocity_mode,
            velocity_layout=velocity_layout,
            rotation_distance_unit=rotation_distance_unit,
            arc_chunking_mode=arc_chunking_mode,
        )

        self.shape = bimanual_arc_token_shape(self.M, velocity_mode, velocity_layout)

    def __call__(self, native_tokens):
        if torch.is_tensor(native_tokens):
            native_tokens = native_tokens.detach().cpu().numpy()
        values = np.asarray(native_tokens)
        if (
            values.ndim != 3
            or values.shape[1:] != self.shape
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                f"Expected finite (B,{self.shape[0]},{self.shape[1]}) interval ARC tokens"
            )
        return np.stack(
            [
                self.codec.detokenize(row, action_horizon=self.action_horizon)
                for row in values
            ]
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Decode saved native ARC tokens; no robot I/O"
    )
    parser.add_argument("tokens", help="Input .npy file, already unnormalized")
    parser.add_argument("output", help="New .npy file for (B,H,14) canonical poses")
    parser.add_argument("--arc-token-layout", choices=ARC_TOKEN_LAYOUTS, default="lab")
    parser.add_argument("--arc-min-distance-unit", type=float, default=0.4)
    parser.add_argument("--arc-resampled-vector-length", type=int, default=100)
    parser.add_argument("--arc-dt", type=float, default=1 / 30)
    parser.add_argument("--arc-rollout-horizon", type=int, default=100)
    parser.add_argument("--arc-rotation-distance-unit", type=float, default=None)
    args = parser.parse_args()
    decoder = BimanualArcDecoder(
        args.arc_token_layout,
        args.arc_min_distance_unit,
        args.arc_resampled_vector_length,
        args.arc_dt,
        args.arc_rollout_horizon,
        rotation_distance_unit=args.arc_rotation_distance_unit,
    )
    result = decoder(np.load(args.tokens, allow_pickle=False))
    with open(args.output, "xb") as output:
        np.save(output, result, allow_pickle=False)


if __name__ == "__main__":
    main()
