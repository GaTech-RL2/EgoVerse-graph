"""Val overlay must forward language annotations into viz_gt_preds."""

from functools import partial

import numpy as np

from egomimic.eval.bimanual_cartesian_eval import (
    overlay_annotation_fields,
    viz_annotation_key,
)
from egomimic.utils.viz_utils import _viz_annotations


def _viz(*, annotation_key=None):
    def _fn(*, predictions, batch):
        del predictions, batch
        return None

    return partial(_fn, annotation_key=annotation_key)


def test_viz_annotation_key_reads_partial_keywords():
    assert viz_annotation_key(_viz(annotation_key="annotations")) == "annotations"
    assert viz_annotation_key(_viz(annotation_key=None)) is None
    assert viz_annotation_key(_viz(annotation_key="null")) is None


def test_overlay_annotation_fields_copies_list_valued_prompts():
    batch = {"annotations": [["fold the shirt"], ["align the sleeves"]]}
    copied = overlay_annotation_fields(_viz(annotation_key="annotations"), batch)
    assert copied == {"annotations": [["fold the shirt"], ["align the sleeves"]]}


def test_overlay_annotation_fields_skips_when_key_missing_or_unset():
    batch = {"annotations": [["fold the shirt"]]}
    assert overlay_annotation_fields(_viz(annotation_key=None), batch) == {}
    assert overlay_annotation_fields(_viz(annotation_key="annotations"), {}) == {}


def test_viz_annotations_paints_text_on_the_frame():
    blank = np.zeros((120, 240, 3), dtype=np.uint8)
    drawn = _viz_annotations(blank, ["fold the long sleeve shirt"])
    assert drawn.shape == blank.shape
    assert int((drawn != blank).sum()) > 0
