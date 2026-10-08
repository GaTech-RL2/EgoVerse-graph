"""Explicit physical-time retiming, before normalization/frame transforms.

No images are interpolated. Callers retain the observation anchor and request
sufficient raw futures. Out-of-range requests fail instead of silently padding.
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

from egomimic.utils.pose_utils import wxyz_to_xyzw, xyzw_to_wxyz


def retime_stream(values, timestamps_s, *, start_s, rate, output_dt_s,
                  horizon, kind="linear", embodiment="human"):
    t = np.asarray(timestamps_s, dtype=np.float64)
    x = np.asarray(values)
    if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or (np.diff(t) <= 0).any():
        raise ValueError("timestamps must be strictly increasing finite seconds")
    if x.ndim < 1 or len(x) != len(t) or not np.isfinite(x).all():
        raise ValueError("invalid stream or time axis")
    if not np.isfinite([start_s, rate, output_dt_s]).all() or rate <= 0 or output_dt_s <= 0:
        raise ValueError("invalid retiming clock/rate")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 2:
        raise ValueError("horizon must be an integer >= 2")
    if embodiment not in ("human", "robot") or (embodiment == "robot" and rate != 1):
        raise ValueError("robot augmentation rate must remain exactly 1")
    query = start_s + rate * np.arange(horizon) * output_dt_s
    if query[0] < t[0] or query[-1] > t[-1] + 1e-12:
        raise ValueError("insufficient real future: no padding/extrapolation allowed")
    query = np.minimum(query, t[-1])
    if kind == "hold":
        return x[np.searchsorted(t, query, side="right") - 1].copy()
    if kind == "linear":
        return interp1d(t, x, axis=0, bounds_error=True)(query)
    if kind != "pose_wxyz" or x.shape != (len(t), 7):
        raise ValueError("expected linear, hold, or (T,7) pose_wxyz")
    if (np.linalg.norm(x[:, 3:], axis=1) < 1e-8).any():
        raise ValueError("zero quaternion")
    xyz = interp1d(t, x[:, :3], axis=0, bounds_error=True)(query)
    quat = Slerp(t, Rotation.from_quat(wxyz_to_xyzw(x[:, 3:])))(query).as_quat()
    return np.concatenate([xyz, xyzw_to_wxyz(quat)], axis=1)


def chunk_speed(positions, output_dt_s):
    """Mean path speed per arm; caller explicitly binds native metric units."""
    x = np.asarray(positions, dtype=np.float64)
    if x.ndim not in (2, 3) or x.shape[-1] != 3 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("expected finite (H,3) or (H,arms,3)")
    if not np.isfinite(output_dt_s) or output_dt_s <= 0:
        raise ValueError("invalid output clock")
    return np.linalg.norm(np.diff(x, axis=0), axis=-1).mean(axis=0) / output_dt_s


class PhysicalWindowRetiming:
    """Deterministic uniform virtual views with recorded physical-time queries.

    Every source uses the same view count, so virtual views do not change
    source proportions. Counts use complete native raw windows, never padded
    tails. Observations remain at the original anchor.
    """
    def __init__(self, rates, fields, pose_keys, horizon, stride=1,
                 embodiment="human", sample_views=5, timestamp_key=None):
        self.rates = tuple(float(r) for r in rates)
        self.fields = dict(fields)
        self.pose_keys = tuple(pose_keys)
        self.required_frames = int(horizon)
        self.stride = int(stride)
        self.embodiment = embodiment
        self.sample_views = int(sample_views)
        self.timestamp_key = timestamp_key
        if (not self.rates or not np.isfinite(self.rates).all()
                or any(r <= 0 or r > 1 for r in self.rates)
                or len(set(self.rates)) != len(self.rates)
                or self.sample_views < 1 or self.sample_views % len(self.rates)
                or self.required_frames < 2 or not 1 <= self.stride < self.required_frames
                or len(self.pose_keys) != 2 or any(k not in self.fields for k in self.pose_keys)
                or any(kind not in {"linear", "hold", "pose_wxyz"} for kind in self.fields.values())):
            raise ValueError("invalid physical retiming contract")
        if embodiment not in {"human", "robot"} or (embodiment == "robot" and self.rates != (1.0,)):
            raise ValueError("robot trajectory must remain at1x")
        if embodiment == "human" and timestamp_key is None:
            raise ValueError("human recorded timestamps are mandatory")
        self.required_keys = tuple(self.fields) + ((timestamp_key,) if timestamp_key else ())
        self.fps = None

    def bind_episode(self, metadata, key_map):
        for key in self.required_keys:
            if key not in key_map or key_map[key].get("horizon") != self.required_frames:
                raise ValueError("every retimed field/clock needs its full native horizon")
        if self.embodiment == "robot":
            fps = float(metadata.get("fps", -1))
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError("robot metadata clock absent")
            if self.fps is not None and self.fps != fps:
                raise ValueError("shared robot transform requires one verified metadata clock")
            self.fps = fps

    def transform(self, batch):
        view = int(batch.pop("_retiming_view"))
        if not 0 <= view < self.sample_views:
            raise ValueError("invalid virtual view")
        rate = self.rates[view % len(self.rates)]
        if self.timestamp_key:
            stamp = np.asarray(batch.pop(self.timestamp_key))
            if stamp.shape != (self.required_frames,) or not np.issubdtype(stamp.dtype, np.integer):
                raise ValueError("recorded timestamps must be integer nanoseconds")
            clock = (stamp - stamp[0]).astype(np.float64) * 1e-9
        else:
            if self.fps is None:
                raise ValueError("episode clock not bound")
            clock = np.arange(self.required_frames, dtype=np.float64) / self.fps
        if not np.isfinite(clock).all() or np.any(np.diff(clock) <= 0):
            raise ValueError("physical timestamps must be strictly increasing")
        query = rate * clock
        if query[-1] > clock[-1] or query[0] < clock[0]:
            raise ValueError("physical retiming cannot extrapolate")
        for key, kind in self.fields.items():
            value = np.asarray(batch[key])
            if len(value) != self.required_frames or not np.isfinite(value).all():
                raise ValueError("retiming refuses padding or nonfinite native futures")
            if kind == "pose_wxyz":
                if value.shape != (self.required_frames, 7) or np.any(np.linalg.norm(value[:, 3:], axis=1) < 1e-8):
                    raise ValueError("invalid pose quaternion")
            if rate == 1.0:
                batch[key] = value.copy()  # exact identity, preserving native values
            elif kind == "hold":
                batch[key] = value[np.searchsorted(clock, query, side="right") - 1].copy()
            elif kind == "linear":
                batch[key] = interp1d(clock, value, axis=0, bounds_error=True)(query)
            else:
                xyz = interp1d(clock, value[:, :3], axis=0, bounds_error=True)(query)
                quaternion = Slerp(clock, Rotation.from_quat(wxyz_to_xyzw(value[:, 3:])))(query).as_quat()
                batch[key] = np.concatenate([xyz, xyzw_to_wxyz(quaternion)], axis=1)
        offsets = np.arange(0, self.required_frames, self.stride)
        duration = clock[offsets[-1]] - clock[0]
        speeds = [np.linalg.norm(np.diff(batch[k][offsets, :3], axis=0), axis=-1).sum() / duration
                  for k in self.pose_keys]
        speed = float(np.mean(speeds))
        if not np.isfinite(speed) or speed < 0:
            raise ValueError("invalid physical requested speed")
        batch["requested_speed"] = np.asarray([speed], dtype=np.float32)
        batch["requested_speed_value"] = np.asarray(speed, dtype=np.float32)
        batch["retiming_rate"] = np.asarray(rate, dtype=np.float32)
        batch["retiming_view"] = np.asarray(view, dtype=np.int64)
        batch["physical_window_duration_s"] = np.asarray(duration, dtype=np.float32)
        return batch


def prepend_window_transform(window_transform, native_transforms):
    """Retiming precedes every native frame/rotation/shape transform."""
    if any(hasattr(t, "sample_views") for t in native_transforms):
        raise ValueError("native transform list already has virtual views")
    return [window_transform, *native_transforms]


def extend_window_key_map(base_key_map, extra_key_map, norm_mode=False):
    if set(base_key_map) & set(extra_key_map):
        raise ValueError("extra clock key must not replace a native key")
    key_map = {**base_key_map, **extra_key_map}
    if norm_mode:
        # Match native keymap normalization semantics, retaining clock metadata
        # required by physical retiming before native transforms.
        key_map = {k: v for k, v in key_map.items()
                   if v.get("key_type") not in ("camera_keys", "annotation_keys")}
    return key_map
