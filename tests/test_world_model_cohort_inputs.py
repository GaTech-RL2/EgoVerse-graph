import hashlib
import json

import pytest
from PIL import Image

from egomimic.scripts.embedding_process.dinov3_correspondence import load_manifest
from egomimic.scripts.embedding_process.world_model_pose_probe import _reference_pairs


def test_explicit_pairs_preserve_selected_horizons():
    records = [
        dict(episode="e", frame=i, obs_rgb_timestamps_ns=i * 100) for i in [0, 1, 5, 10]
    ]
    explicit = [dict(episode="e", source_frame=0, target_frame=i) for i in [1, 5, 10]]
    assert _reference_pairs(records, explicit) == [(0, 1), (0, 2), (0, 3)]
    with pytest.raises(ValueError, match="positive physical time"):
        _reference_pairs(records, [dict(episode="e", source_frame=5, target_frame=1)])
    with pytest.raises(ValueError, match="unique"):
        _reference_pairs(records, explicit + explicit[:1])


def test_unlabelled_manifest_only_for_features(tmp_path):
    path = tmp_path / "frame.png"
    Image.new("RGB", (32, 32)).save(path)
    manifest = dict(
        model="official",
        revision="a" * 40,
        frames=[
            dict(
                id="a",
                path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                points={},
                episode="e",
            )
        ],
        pairs=[],
    )
    f = tmp_path / "manifest.json"
    f.write_text(json.dumps(manifest))
    assert len(load_manifest(f, features_only=True)[1]) == 1
    with pytest.raises(ValueError, match="At least one pair"):
        load_manifest(f)


def test_timestamp_selection_rejects_filtered_gaps():
    import numpy as np

    from egomimic.scripts.embedding_process.world_model_pose_probe import (
        select_contiguous_anchors,
        validate_raw_aria_wrist,
    )

    timestamps = np.arange(1200, dtype=np.int64) * 33333333
    timestamps[500:] += 8_000_000_000
    anchors = select_contiguous_anchors(timestamps)
    assert len(anchors) == 4
    for a, targets in anchors:
        assert not (a < 500 <= targets[-1])
        assert np.allclose(
            (timestamps[targets] - timestamps[a]) * 1e-9, [0.1, 0.5, 1], atol=0.02
        )
    points = np.zeros((2, 21, 3))
    wrists = np.zeros((2, 3))
    assert validate_raw_aria_wrist(points, wrists) == 0
    points[:, 5, 0] = 0.1
    with pytest.raises(ValueError, match="index5 mismatch"):
        validate_raw_aria_wrist(points, wrists)
