"""PI metadata adapters without modifying stored episode attributes."""

from pathlib import Path
import logging

from egomimic.campaigns.pi05.embodiment import get_embodiment, get_embodiment_id
from egomimic.rldb.zarr.zarr_dataset_multi import (
    LocalEpisodeResolver as GraphLocalEpisodeResolver,
    S3EpisodeResolver as GraphS3EpisodeResolver,
    SafeS3EpisodeResolver as GraphSafeS3EpisodeResolver,
    ZarrDataset as GraphZarrDataset,
)

logger = logging.getLogger(__name__)


class ZarrDataset(GraphZarrDataset):
    def init_episode(self):
        super().init_episode()
        self.embodiment = get_embodiment(get_embodiment_id(self.embodiment)).lower()


class S3EpisodeResolver(GraphS3EpisodeResolver):
    _dataset_class = ZarrDataset


class SafeS3EpisodeResolver(GraphSafeS3EpisodeResolver):
    _dataset_class = ZarrDataset


class LocalEpisodeResolver(GraphLocalEpisodeResolver):
    _dataset_class = ZarrDataset

# Annotation-cutoff bodies preserved from d5f72068.
class ZarrAnnotationCutoffDataset(ZarrDataset):
    """ZarrDataset that clamps action chunks at the end of the enclosing annotation.

    Standard chunking from the start frame, but action reads stop at EOS+1 of the
    annotation span containing the start frame. The chunk is then padded out to
    ``horizon`` via the base ``_pad_sequences`` (repeat-last), so frames beyond
    EOS become the last action of the interval rather than crossing into the
    next annotation.

    If the start frame is not inside any annotation, behaves like the base class.
    """

    def init_episode(self):
        super().init_episode()
        self._frame_to_ann_end: dict[int, int] | None = None

    def _build_frame_to_ann_end(self) -> dict[int, int]:
        """Map ``frame_idx -> ann_end`` (exclusive) for every frame inside an
        annotation span. Annotations use half-open ``[start_idx, end_idx)``.
        """
        mapping: dict[int, int] = {}
        n_spans = 0
        for ann in self._load_annotations():
            start_idx = int(ann.get("start_idx", -1))
            end_idx = int(ann.get("end_idx", -1))
            if start_idx < 0 or end_idx <= start_idx:
                continue
            n_spans += 1
            for idx in range(start_idx, end_idx):
                mapping[idx] = end_idx
        # One-time per-episode visibility into annotation-cutoff usage: if
        # spans/frames_covered are 0 the cutoff is a no-op (episode has no usable
        # annotations); >0 confirms action chunks are being clamped at EOS.
        ep = Path(self.episode_path).name
        logger.info(
            "[AnnotationCutoff] ep=%s spans=%d frames_covered=%d/%d",
            ep,
            n_spans,
            len(mapping),
            self.total_frames,
        )
        return mapping

    def _chunk_end_idx(self, start_idx: int, horizon: int, key_type: str | None) -> int:
        end_idx = super()._chunk_end_idx(start_idx, horizon, key_type)
        if key_type != "action_keys":
            return end_idx
        if self._frame_to_ann_end is None:
            self._frame_to_ann_end = self._build_frame_to_ann_end()
        ann_end = self._frame_to_ann_end.get(start_idx)
        if ann_end is None:
            return end_idx
        return min(end_idx, ann_end)

def _episode_has_annotation_spans(ds: "ZarrDataset") -> bool:
    """True if the episode has at least one usable ``[start_idx, end_idx)`` span.

    Many Scale-"completed" episodes have an empty (or span-less) zarr
    ``annotations`` array because the annotation-injection step lagged; the
    AnnotationCutoff is a no-op for those, so they should be dropped when the
    point of the run is to clamp chunks at annotation boundaries.
    """
    try:
        anns = ds._load_annotations()
    except Exception:
        return False
    return any(
        isinstance(a, dict)
        and 0 <= int(a.get("start_idx", -1)) < int(a.get("end_idx", -1))
        for a in anns
    )

class S3AnnotationCutoffEpisodeResolver(S3EpisodeResolver):
    """S3EpisodeResolver that loads ZarrAnnotationCutoffDataset instances.

    When ``require_annotations`` is set (default), episodes whose zarr
    ``annotations`` array has no usable span are dropped — otherwise the
    annotation cutoff would silently no-op on them.
    """

    _dataset_class = ZarrAnnotationCutoffDataset

    def __init__(self, *args, require_annotations: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.require_annotations = require_annotations

    def resolve(self, filters=None):
        datasets = super().resolve(filters=filters)
        if not self.require_annotations:
            return datasets
        kept = {
            h: ds for h, ds in datasets.items() if _episode_has_annotation_spans(ds)
        }
        dropped = sorted(set(datasets) - set(kept))
        if dropped:
            logger.warning(
                "[AnnotationCutoff] dropped %d/%d episodes with no usable "
                "annotation spans (e.g. %s)",
                len(dropped),
                len(datasets),
                dropped[:5],
            )
        logger.info(
            "[AnnotationCutoff] kept %d/%d episodes with usable annotations",
            len(kept),
            len(datasets),
        )
        if not kept:
            raise ValueError(
                "[AnnotationCutoff] no resolved episodes contain usable annotation "
                "spans — check the filter / annotation injection for this dataset."
            )
        return kept

class LocalAnnotationCutoffEpisodeResolver(LocalEpisodeResolver):
    """LocalEpisodeResolver that loads ZarrAnnotationCutoffDataset instances."""

    _dataset_class = ZarrAnnotationCutoffDataset
