"""Recorded-data visualization through explicit renderers and shared video IO."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from egomimic.eval.video import EvalVideo


def _component(value):
    value = str(value)
    return (
        value
        if value not in {".", ".."} and value and Path(value).name == value
        else hashlib.sha256(value.encode()).hexdigest()
    )


class DatasetVideo(EvalVideo):
    """No model or synthetic prediction: visualize the selected recorded samples."""

    def __init__(
        self,
        renderers,
        *,
        source_fps=30,
        max_episodes=3,
        sample_id_key="episode_hash",
        frame_index_key="frame_index",
        complete_episodes=True,
        max_episode_frames=100000,
        sample_every=50,
        max_saved_samples=12,
        artifact_keys=(),
        artifact_transforms=None,
    ):
        if not renderers or not all(callable(value) for value in renderers.values()):
            raise ValueError("Every source needs a configured callable renderer")
        if type(sample_every) is not int or sample_every < 1:
            raise ValueError("sample_every must be a positive integer")
        if type(max_saved_samples) is not int or max_saved_samples < 0:
            raise ValueError("max_saved_samples must be a nonnegative integer")
        self.viz_func = renderers
        self.viz_every_n_epochs, self.viz_max_batches = 1, None
        self.video_output_dir = None
        self.max_episode_frames, self.frames_per_file = (
            max_episode_frames,
            max_episode_frames,
        )
        self.max_episodes = max_episodes
        self.sample_every, self.max_saved_samples = sample_every, max_saved_samples
        self.artifact_keys = tuple(artifact_keys)
        self.artifact_transforms = dict(artifact_transforms or {})
        if not all(callable(value) for value in self.artifact_transforms.values()):
            raise TypeError("Artifact transforms must be configured batch callables")
        self._validation_group = "valid"
        self.configure_video(
            source_fps=source_fps,
            sample_id_key=sample_id_key,
            frame_index_key=frame_index_key,
            complete_episodes=complete_episodes,
        )

    def data_requirements(self):
        return replace(super().data_requirements(), max_episodes=self.max_episodes)

    def bind_data_context(self, *, normalizer):
        self.normalizer = normalizer

    def on_validation_start(self):
        super().on_validation_start()
        self._saved, self._samples = set(), []

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        del batch_idx, dataloader_idx
        for source, values in batch.items():
            if source not in self.viz_func:
                raise ValueError(f"No recorded-data renderer configured for {source!r}")
            frames = torch.as_tensor(self.viz_func[source](values))
            if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != torch.uint8:
                raise ValueError("A dataset renderer must return B,H,W,3 uint8 frames")
            group = self._validation_group
            self._record_video_frames((group, source), frames, values)
            transform = self.artifact_transforms.get(source)
            export_values = transform(dict(values)) if transform else values
            for i, (episode, index) in enumerate(
                zip(
                    values[self.sample_id_key],
                    values[self.frame_index_key],
                    strict=True,
                )
            ):
                index = int(index)
                identity = (group, source, str(episode), index)
                if (
                    identity in self._saved
                    or index % self.sample_every
                    or len(self._saved) >= self.max_saved_samples
                ):
                    continue
                arrays = {}
                for key in self.artifact_keys:
                    value = export_values[key][i]
                    if torch.is_tensor(value):
                        value = value.detach().cpu().numpy()
                    value = np.asarray(value)
                    if value.dtype.hasobject:
                        raise ValueError(
                            f"Artifact {key} must be a numeric/string array, not a Python object"
                        )
                    arrays[key] = value
                root = (
                    Path(self.root_dir())
                    / "samples"
                    / _component(group)
                    / _component(source)
                    / _component(episode)
                )
                root.mkdir(parents=True, exist_ok=True)
                stem = root / f"frame_{index:08d}"
                png, numeric = stem.with_suffix(".png"), stem.with_suffix(".npz")
                if png.exists() or numeric.exists():
                    raise FileExistsError(
                        f"Refusing to replace visualization artifacts: {stem}"
                    )
                Image.fromarray(frames[i].cpu().numpy()).save(png)
                with numeric.open("xb") as handle:
                    np.savez(handle, **arrays)
                self._samples.append(
                    {
                        "group": group,
                        "source": source,
                        "episode": str(episode),
                        "frame": index,
                        "png": str(png),
                        "arrays": str(numeric),
                    }
                )
                self._saved.add(identity)
        return {}

    def on_validation_end(self):
        super().on_validation_end()
        path = Path(self.root_dir()) / "visualization-receipt.json"
        with path.open("x") as handle:
            json.dump(
                {
                    "kind": "recorded-data-visualization",
                    "samples": self._samples,
                    "videos": [str(row[2]) for row in self._written_paths],
                },
                handle,
                indent=2,
            )
