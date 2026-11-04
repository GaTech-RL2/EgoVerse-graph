"""Opt-in shared control clock for datasets recorded at different frame rates.

All observation cameras, proprioception and actions use the same nearest native
frame at each target timestamp. Source stores are never rewritten. This is for
visual/proprioceptive recipes; annotation spans remain native and are rejected.
"""

from functools import partial

import numpy as np

from egomimic.rldb.zarr.zarr_dataset_multi import S3EpisodeResolver, ZarrDataset


def control_frame_indices(total_frames, source_fps, target_fps=30.0):
    source_fps, target_fps = float(source_fps), float(target_fps)
    if (
        not np.isfinite(source_fps)
        or not np.isfinite(target_fps)
        or min(source_fps, target_fps) <= 0
    ):
        raise ValueError("Source and target FPS must be positive and finite")
    total_frames = int(total_frames)
    if total_frames < 1:
        raise ValueError("Episode must contain at least one frame")
    count = int(np.floor((total_frames - 1) * target_fps / source_fps + 1e-9)) + 1
    return np.minimum(
        np.floor(np.arange(count) * source_fps / target_fps + 0.5).astype(np.int64),
        total_frames - 1,
    )


class ControlRateEpisode:
    def __init__(self, source, target_fps):
        self.source = source
        self._store = source._store
        self.indices = control_frame_indices(
            source.metadata["total_frames"], source.metadata["fps"], target_fps
        )
        self.metadata = dict(
            source.metadata,
            source_fps=source.metadata["fps"],
            fps=float(target_fps),
            total_frames=len(self.indices),
        )

    @property
    def intrinsics(self):
        return self.source.intrinsics

    def _collect_keys(self):
        return self.source._collect_keys()

    def read(self, keys_with_ranges):
        result = {}
        for key, (start, end) in keys_with_ranges.items():
            if end is None:
                result[key] = self.source.read({key: (int(self.indices[start]), None)})[
                    key
                ]
                continue
            indices = self.indices[start:end]
            if len(indices) == 0:
                result[key] = self._store[key][0:0]
                continue
            first, stop = int(indices[0]), int(indices[-1]) + 1
            native = self.source.read({key: (first, stop)})[key]
            result[key] = native[indices - first]
        return result

    def __len__(self):
        return len(self.indices)


class ControlRateZarrDataset(ZarrDataset):
    def __init__(self, *args, control_fps=30.0, **kwargs):
        self.control_fps = float(control_fps)
        keymap = kwargs.get("key_map") or {}
        if any(spec.get("key_type") == "annotation_keys" for spec in keymap.values()):
            raise ValueError(
                "Control-rate visual datasets do not support native annotation spans"
            )
        super().__init__(*args, **kwargs)

    def init_episode(self):
        super().init_episode()
        self.episode_reader = ControlRateEpisode(self.episode_reader, self.control_fps)
        self.metadata = self.episode_reader.metadata
        self.total_frames = self.metadata["total_frames"]
        self.keys_dict = {
            key: (0, self.total_frames) for key in self.episode_reader._collect_keys()
        }


class ControlRateS3EpisodeResolver(S3EpisodeResolver):
    def __init__(self, *args, control_fps=30.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._dataset_class = partial(ControlRateZarrDataset, control_fps=control_fps)

    def _load_zarr_datasets(self, search_path, valid_folder_names):
        datasets = super()._load_zarr_datasets(search_path, valid_folder_names)
        missing = set(valid_folder_names) - set(datasets)
        if missing:
            raise ValueError(
                f"Refusing incomplete control-rate dataset: {len(missing)} missing episodes"
            )
        return datasets
