"""Read-only Zarr joint replay through the same rollout loop as graph policies."""

import numpy as np

from egomimic.robot.interface import ARM_OFFSET, joint_vector


class ZarrReplayPolicy:
    action_type = "joints"

    def __init__(
        self, path, keys=None, action_key=None, chunk_size=1, start=0, stop=None
    ):
        import zarr

        self.store = zarr.open_group(str(path), mode="r")
        self.keys, self.action_key = dict(keys or {}), action_key
        if bool(self.keys) == bool(action_key):
            raise ValueError("Choose per-arm keys or one bimanual action_key")
        if self.keys and not set(self.keys) <= ARM_OFFSET.keys():
            raise ValueError("Replay arm keys must be left/right")
        arrays = (
            [self.store[action_key]]
            if action_key
            else [
                self.store[key] for spec in self.keys.values() for key in spec.values()
            ]
        )
        if not arrays:
            raise ValueError("Replay needs action arrays")
        # EgoVerse Zarr arrays may have chunk padding beyond total_frames.
        length = int(
            self.store.attrs.get(
                "total_frames", min(array.shape[0] for array in arrays)
            )
        )
        if length <= 0 or any(array.shape[0] < length for array in arrays):
            raise ValueError("Invalid Zarr frame count")
        if action_key and arrays[0].shape[1:] != (14,):
            raise ValueError("Bimanual joint replay requires an (N, 14) array")
        for arm, spec in self.keys.items():
            if (
                set(spec) != {"joints", "gripper"}
                or self.store[spec["joints"]].shape[1:] != (6,)
                or self.store[spec["gripper"]].shape[1:] not in ((), (1,))
            ):
                raise ValueError(
                    f"Replay requires six joints and one gripper per row: {arm}"
                )
        self.cursor, self.stop, self.chunk_size = (
            int(start),
            length if stop is None else int(stop),
            int(chunk_size),
        )
        if not 0 <= self.cursor < self.stop <= length or self.chunk_size < 1:
            raise ValueError("Invalid replay bounds or chunk size")

    def predict(self, obs):
        if self.cursor >= self.stop:
            raise StopIteration
        end = min(self.stop, self.cursor + self.chunk_size)
        if self.action_key:
            rows = np.asarray(
                self.store[self.action_key][self.cursor : end], dtype=float
            )
        else:
            # Hold arms missing from the replay at their measured positions.
            rows = np.tile(obs["joint_positions"], (end - self.cursor, 1)).astype(float)
            for arm, spec in self.keys.items():
                offset = ARM_OFFSET[arm]
                rows[:, offset : offset + 6] = self.store[spec["joints"]][
                    self.cursor : end
                ]
                rows[:, offset + 6] = np.asarray(
                    self.store[spec["gripper"]][self.cursor : end]
                ).reshape(-1)
        for row in rows:
            for offset in ARM_OFFSET.values():
                joint_vector(row[offset : offset + 7])
        self.cursor = end
        return rows
