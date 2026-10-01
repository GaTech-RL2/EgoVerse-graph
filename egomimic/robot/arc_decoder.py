"""Decode native ARC policy outputs before robot frame transforms.

Ported from aidan/shorts-extreme (55932d99), rollout-arc.py. This module has no
robot I/O. Call it after action unnormalization, then apply the controller's
camera/base frame transforms to the returned canonical poses.

Replay tempo
------------
An arc token is a path plus a clock, so the path can be replayed at a tempo the
training data never contained without touching the geometry. ``speed`` multiplies
the tempo of the MOVING phases and ``hold_speed`` the tempo of the slow phases
(holds, grasp dwell, careful placement: every instant at which both arms' decoded
path speed is under ``hold_threshold`` m/s). 1.0 / 1.0 is the demonstrated tempo
and takes the unmodified decode path. The two are separate because a hold is where
the gripper physically closes, and that time does not shrink with the arm's.

The warp is ONE monotone map from wall time to token-clock time, applied to both
arms, so bimanual coordination is exactly what the token encoded; only the rate
at which the pair advances changes. It acts on the clock, before the codec's own
arc-length resampling, so rotation still goes through the codec's SLERP. Tokens
without a per-waypoint clock (lab and the cartesian layouts) take a uniform speed
only, as a codec whose control period is scaled.

Token shapes (M = resampled_vector_length):
  e1_dur / e1_logdur / e1_profile  (M, 16)  E1 arcdur / arclogdur / arcvel
  e1_durhyb                        (M, 18)  E1 hybrid arcdur: independent per-arm
                                            translation and rotation clocks
  lab                              (M+1, 14)
  cartesian_per_waypoint/duration  (2M, 14) station codec (Elmo's race-hybrid)
  lab_pw_wide                      (M, 28)  PR #193 lab codec, per-waypoint
  lab_pw_stacked                   (2M, 14)   velocity beside / under waypoints
The two lab_pw layouts decode with the vendored PR #193 codec
(``arc_length_tokenizer_pr193``) that trained the Elmo+Aidan lab runs, not the
station's drifted copy.

``TimeChunkRetimer`` gives a time-indexed policy (no token, no decoder) the same
control: its (H, 14) chunk is a path on a uniform clock, read through the same warp.
speed == hold_speed is the naive baseline speed-up (2x = every other row).
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from egomimic.rldb.zarr.arc_length_tokenizer import (
    TokenizeBimanualArcLengthCartesian,
    cumulative_arc_length,
)
from egomimic.rldb.zarr import arc_length_tokenizer_pr193 as pr193
from egomimic.rldb.zarr.e1_arc_tokenizer import (
    ARM_LAYOUT,
    E1_HYBRID_DIM,
    TokenizeBimanualArcLengthE1,
)
from egomimic.robot.arc_speed import ARC_SPEED_RANGE, validate_arc_speed  # noqa: F401

E1_VELOCITY_MODE = {"e1_dur": "dur", "e1_logdur": "logdur", "e1_profile": "profile",
                    "e1_durhyb": "durhyb"}
ARC_CARTESIAN_VELOCITY_MODE = {
    "cartesian_per_waypoint": "per_waypoint",
    "cartesian_duration": "duration",
}
# velocity_layout of the PR #193 lab codec, per_waypoint timing.
LAB_PW_LAYOUT = {"lab_pw_wide": "wide", "lab_pw_stacked": "stacked"}
ARC_TOKEN_LAYOUTS = (
    "lab",
    *E1_VELOCITY_MODE,
    *ARC_CARTESIAN_VELOCITY_MODE,
    *LAB_PW_LAYOUT,
)
# Token-clock seconds the tempo eases over at a hold <-> moving transition, so the
# commanded speed does not step when an arm crosses ``hold_threshold``.
RATE_RAMP_S = 0.2
_WARP_OVERSAMPLE = 4  # warp grid points per control step


class BimanualArcRoundTrip:
    """Temporary ARC encode/decode around a canonical Cartesian chunk.

    The input is an ordinary baseline ``(B, H, 14)`` prediction.  The codec
    uses M=100 and computes D from that prediction on every call.  Because the
    non-hybrid tokenizer applies one cap independently to each arm, the scalar
    D is the larger of the two arm path lengths; the shorter arm is clipped at
    its own exact path length and then represented with trailing holds.
    """

    def __init__(
        self,
        resampled_vector_length=100,
        dt=1 / 30,
        velocity_mode="per_waypoint",
        distance_mode="predicted_span",
    ):
        if int(resampled_vector_length) != 100:
            raise ValueError("The temporary ARC round trip requires M=100")
        if velocity_mode != "per_waypoint":
            raise ValueError(
                "The temporary ARC round trip requires per_waypoint velocities"
            )
        if distance_mode != "predicted_span":
            raise ValueError("Unsupported temporary ARC round-trip distance mode")
        if not np.isfinite(float(dt)) or float(dt) <= 0:
            raise ValueError("ARC round-trip dt must be positive and finite")
        self.M = 100
        self.dt = float(dt)
        self.velocity_mode = velocity_mode
        self.distance_mode = distance_mode
        self.last_distances = ()

    @staticmethod
    def _distance(row):
        left = cumulative_arc_length(row[:, 0:3])[-1]
        right = cumulative_arc_length(row[:, 7:10])[-1]
        return max(float(left), float(right), 1e-6)

    def __call__(self, actions):
        if torch.is_tensor(actions):
            actions = actions.detach().double().cpu().numpy()
        values = np.asarray(actions, dtype=np.float64)
        if values.ndim == 2:
            values = values[None]
        if values.ndim != 3 or values.shape[-1] != 14 or values.shape[1] < 2:
            raise ValueError(
                "ARC round trip expects canonical actions shaped (B,H,14)"
            )
        if not np.isfinite(values).all():
            raise ValueError("ARC round-trip input must be finite")

        decoded = []
        distances = []
        for row in values:
            distance = self._distance(row)
            codec = TokenizeBimanualArcLengthCartesian(
                min_distance_unit=distance,
                resampled_vector_length=self.M,
                dt=self.dt,
                velocity_mode=self.velocity_mode,
            )
            tokens = codec.transform({"actions_cartesian": row.copy()})[
                "actions_cartesian"
            ]
            decoded.append(codec.detokenize(tokens, action_horizon=len(row)))
            distances.append(distance)
        self.last_distances = tuple(distances)
        return np.stack(decoded)


class ReplayTempo:
    """speed / hold_speed state and the wall-time -> clock warp shared by the ARC
    decoder and the time-chunk retimer. Subclasses set dt, action_horizon,
    hold_threshold and ``holds`` (whether hold_speed can act on their output)."""

    def _init_tempo(self, speed, hold_speed, hold_threshold):
        if (isinstance(hold_threshold, bool) or not isinstance(hold_threshold, (int, float))
                or not np.isfinite(hold_threshold) or hold_threshold < 0):
            raise ValueError("ARC hold_threshold must be a finite speed >= 0 in m/s")
        self.hold_threshold = float(hold_threshold)
        self.last_stats = None
        self.set_speed(speed, hold_speed)

    def set_speed(self, speed, hold_speed=None):
        """Set the tempo multipliers used from the NEXT decoded token on.

        ``hold_speed=None`` keeps holds in step with ``speed`` (a uniform speed-up).
        Both values are validated before either is stored.
        """
        speed = validate_arc_speed(speed, "speed")
        hold_speed = speed if hold_speed is None else validate_arc_speed(hold_speed, "hold_speed")
        self.speed, self.hold_speed = speed, hold_speed

    # Integer percent views for the rollout profile's typed inference controls.
    @property
    def speed_percent(self):
        return int(round(self.speed * 100))

    @speed_percent.setter
    def speed_percent(self, value):
        self.set_speed(value / 100, self.hold_speed)

    @property
    def hold_speed_percent(self):
        return int(round(self.hold_speed * 100))

    @hold_speed_percent.setter
    def hold_speed_percent(self, value):
        self.set_speed(self.speed, value / 100)

    def _warp(self, clocks, cums, end=np.inf):
        """Token-clock time to read each of the H control ticks at, and the share of
        the decoded span that was a hold. rate = d(clock) / d(wall). ``end`` is the
        clock time at which the token's path runs out: nothing after it is a hold,
        it is just the endpoint being repeated, so it is left out of the share."""
        wall = self.dt * np.arange(self.action_horizon, dtype=np.float64)
        top = max(self.speed, self.hold_speed)
        n = int(np.ceil(self.action_horizon * top * _WARP_OVERSAMPLE)) + _WARP_OVERSAMPLE + 1
        tau = (self.dt / _WARP_OVERSAMPLE) * np.arange(n, dtype=np.float64)  # covers wall[-1] * top
        speed = np.zeros(n - 1)
        for clock, cum in zip(clocks, cums):
            if cum[-1] >= 1e-9:  # a stationary arm never makes the pair "moving"
                speed = np.maximum(speed, np.diff(np.interp(tau, clock, cum)) / np.diff(tau))
        moving = speed >= self.hold_threshold
        rate = np.where(moving, self.speed, self.hold_speed)
        width = max(1, int(round(RATE_RAMP_S / (self.dt / _WARP_OVERSAMPLE))))
        if width > 1 and self.speed != self.hold_speed:
            padded = np.pad(rate, (width // 2, width - 1 - width // 2), mode="edge")
            rate = np.convolve(padded, np.ones(width) / width, mode="valid")
        elapsed = np.concatenate(([0.0], np.cumsum(np.diff(tau) / rate)))  # wall time at each tau
        times = np.interp(wall, elapsed, tau)
        used = tau[1:] <= min(times[-1], end)
        return times, (float(1.0 - moving[used].mean()) if used.any() else None)


class BimanualArcDecoder(ReplayTempo):
    def __init__(self, token_layout="lab", min_distance_unit=0.4,
                 resampled_vector_length=100, dt=1/30, action_horizon=100,
                 rotation_distance_unit=None, arc_chunking_mode=None,
                 speed=1.0, hold_speed=1.0, hold_threshold=0.05):
        if token_layout not in ARC_TOKEN_LAYOUTS:
            raise ValueError(f"token_layout must be one of {ARC_TOKEN_LAYOUTS}")
        self.token_layout, self.dt = token_layout, float(dt)
        self.holds = token_layout in E1_VELOCITY_MODE  # only a per-waypoint clock separates holds
        self._init_tempo(speed, hold_speed, hold_threshold)
        self.M = int(resampled_vector_length)
        self.action_horizon = int(action_horizon)
        if self.M < 2 or self.action_horizon < 1 or dt <= 0 or min_distance_unit <= 0:
            raise ValueError("Invalid ARC distance, time, horizon or waypoint count")
        if token_layout == "lab":
            self.shape = (self.M + 1, 14)
        elif token_layout in ARC_CARTESIAN_VELOCITY_MODE:
            self.shape = (2 * self.M, 14)
        elif token_layout in LAB_PW_LAYOUT:
            self.shape = pr193.bimanual_arc_token_shape(
                self.M, "per_waypoint", LAB_PW_LAYOUT[token_layout])
        elif token_layout == "e1_durhyb":
            self.shape = (self.M, E1_HYBRID_DIM)
        else:
            self.shape = (self.M, 16)
        kwargs = dict(min_distance_unit=min_distance_unit,
                      resampled_vector_length=self.M, dt=dt)
        self._uniform_codec = None  # (speed, codec) for a uniform speed-up
        self._uniform_cls = TokenizeBimanualArcLengthCartesian
        if token_layout in LAB_PW_LAYOUT:
            if rotation_distance_unit is not None or arc_chunking_mode is not None:
                raise ValueError("lab_pw layouts are the plain (no R) PR #193 codec")
            kwargs.update(velocity_mode="per_waypoint",
                          velocity_layout=LAB_PW_LAYOUT[token_layout])
            self._uniform_kwargs = kwargs
            self._uniform_cls = pr193.TokenizeBimanualArcLengthCartesian
            self.codec = self._uniform_cls(**kwargs)
        elif token_layout in E1_VELOCITY_MODE:
            self.codec = TokenizeBimanualArcLengthE1(
                **kwargs,
                velocity_norm="path",
                velocity_mode=E1_VELOCITY_MODE[token_layout],
            )
        else:
            if token_layout in ARC_CARTESIAN_VELOCITY_MODE:
                kwargs.update(
                    velocity_mode=ARC_CARTESIAN_VELOCITY_MODE[token_layout],
                    rotation_distance_unit=rotation_distance_unit,
                    arc_chunking_mode=arc_chunking_mode,
                )
            self._uniform_kwargs = kwargs
            self.codec = TokenizeBimanualArcLengthCartesian(**kwargs)

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
        return np.stack([self._decode(row) for row in values])

    def _decode(self, row):
        h = self.action_horizon
        if self.token_layout not in E1_VELOCITY_MODE:
            # No per-waypoint clock to warp: holds cannot be told apart, so
            # ``hold_speed`` has nothing to act on. A uniform tempo is exactly a
            # codec whose control period is scaled, rebuilt only when ``speed`` moves.
            self.last_stats = {"speed": self.speed, "hold_speed": None, "horizon": h,
                               "valid_steps": None, "hold_fraction": None}
            if self.speed == 1.0:
                return self.codec.detokenize(row, action_horizon=h)
            if self._uniform_codec is None or self._uniform_codec[0] != self.speed:
                kwargs = {**self._uniform_kwargs, "dt": self.dt * self.speed}
                self._uniform_codec = (self.speed, self._uniform_cls(**kwargs))
            return self._uniform_codec[1].detokenize(row, action_horizon=h)
        clocks = self.codec.clock_at_waypoints(row)
        cums = [cumulative_arc_length(row[:, off : off + 3]) for off, *_ in ARM_LAYOUT]
        # Past the last moving arm's final waypoint the codec holds the endpoint:
        # rows beyond ``valid_steps`` command no motion, so a replan interval
        # above it spends control ticks standing still and gives the speed-up back.
        if self.token_layout == "e1_durhyb":
            # Four streams: a wrist still turning, or a held arm's gripper still
            # closing, keeps the token live after translation has finished.
            ends = self.codec.hybrid_stream_ends(row)
        else:
            ends = [float(clock[-1]) for clock, cum in zip(clocks, cums) if cum[-1] >= 1e-9]
        end = max(ends) if ends else 0.0
        times, hold_fraction = self._warp(clocks, cums, end)
        valid = int(np.searchsorted(times, end, side="right")) if ends else 0
        self.last_stats = {"speed": self.speed, "hold_speed": self.hold_speed, "horizon": h,
                           "valid_steps": min(valid, h), "hold_fraction": hold_fraction}
        if self.speed == 1.0 and self.hold_speed == 1.0:
            return self.codec.detokenize(row, action_horizon=h)  # the unmodified path
        return self.codec.detokenize(row, action_horizon=h, times=times)


class TimeChunkRetimer(ReplayTempo):
    """Replay tempo for a time-indexed policy's canonical (H, 14) Euler chunk.

    Row k is the pose at k * dt, so the chunk is a path on a uniform clock and the
    ARC decoder's warp applies unchanged: speed == hold_speed is the naive uniform
    speed-up (2x replays every other row), hold_speed < speed keeps holds at the
    demonstrated tempo. Past the last row the final pose is held, as an ARC path's
    endpoint is. Rotation is SLERPed in intrinsic ZYX, the convention pose_matrix
    reads. 1.0 / 1.0 returns the chunk untouched.
    """

    holds = True

    def __init__(self, action_horizon=100, dt=1/30, speed=1.0, hold_speed=1.0, hold_threshold=0.05):
        self.action_horizon, self.dt = int(action_horizon), float(dt)
        if self.action_horizon < 2 or not self.dt > 0:
            raise ValueError("Invalid chunk horizon or control period")
        self.shape = (self.action_horizon, 14)
        self._init_tempo(speed, hold_speed, hold_threshold)

    def __call__(self, actions):
        h = self.action_horizon
        if self.speed == 1.0 and self.hold_speed == 1.0:
            self.last_stats = {"speed": 1.0, "hold_speed": 1.0, "horizon": h,
                               "valid_steps": h, "hold_fraction": None}
            return actions  # the unmodified path, byte for byte
        if torch.is_tensor(actions):
            actions = actions.detach().double().cpu().numpy()
        values = np.asarray(actions, dtype=np.float64)
        if values.ndim == 2:
            values = values[None]
        if values.ndim != 3 or values.shape[1:] != self.shape or not np.isfinite(values).all():
            raise ValueError(f"Expected a finite time chunk (B,{h},14), got {values.shape}")
        return np.stack([self._retime(row) for row in values])

    def _retime(self, row):
        h = self.action_horizon
        clock = self.dt * np.arange(h, dtype=np.float64)
        cums = [cumulative_arc_length(row[:, off : off + 3]) for off, *_ in ARM_LAYOUT]
        times, hold_fraction = self._warp([clock, clock], cums, clock[-1])
        valid = int(np.searchsorted(times, clock[-1], side="right"))
        self.last_stats = {"speed": self.speed, "hold_speed": self.hold_speed, "horizon": h,
                           "valid_steps": min(valid, h), "hold_fraction": hold_fraction}
        times = np.minimum(times, clock[-1])
        out = np.empty_like(row)
        for col in range(14):
            out[:, col] = np.interp(times, clock, row[:, col])
        for _xyz, ypr, _grip, _vsl in ARM_LAYOUT:
            spin = Slerp(clock, Rotation.from_euler("ZYX", row[:, ypr : ypr + 3]))
            out[:, ypr : ypr + 3] = spin(times).as_euler("ZYX")
        return out


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
    parser.add_argument("--arc-speed", type=float, default=1.0,
                        help="tempo multiplier for moving phases (1 = demonstrated)")
    parser.add_argument("--arc-hold-speed", type=float, default=None,
                        help="tempo multiplier for holds; default follows --arc-speed")
    parser.add_argument("--arc-hold-threshold", type=float, default=0.05,
                        help="path speed in m/s under which both arms count as holding")
    args = parser.parse_args()
    decoder = BimanualArcDecoder(args.arc_token_layout, args.arc_min_distance_unit,
                                args.arc_resampled_vector_length, args.arc_dt, args.arc_rollout_horizon,
                                hold_threshold=args.arc_hold_threshold)
    decoder.set_speed(args.arc_speed, args.arc_hold_speed)
    result = decoder(np.load(args.tokens, allow_pickle=False))
    with open(args.output, "xb") as output:
        np.save(output, result, allow_pickle=False)


if __name__ == "__main__":
    main()
