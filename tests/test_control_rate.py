import numpy as np
import pytest

from egomimic.rldb.zarr.control_rate import ControlRateEpisode, control_frame_indices


@pytest.mark.parametrize("fps", [30, 59, 60, 29, 15])
def test_control_clock_preserves_duration_and_has_bounded_sampling_error(fps):
    indices = control_frame_indices(10 * fps + 1, fps, 30)
    assert len(indices) == 301
    assert indices[0] == 0 and indices[-1] == 10 * fps
    assert np.max(np.abs(indices / fps - np.arange(301) / 30)) <= 0.5 / fps + 1e-10


def test_thirty_hz_is_identity():
    assert np.array_equal(control_frame_indices(1128, 30), np.arange(1128))


def test_reader_uses_same_clock_for_action_chunks_and_images():
    class Source:
        metadata = dict(total_frames=601, fps=60)
        _store = {"pose": np.arange(601)[:, None], "image": np.arange(601)}

        def read(self, ranges):
            return {
                key: self._store[key][start]
                if end is None
                else self._store[key][start:end]
                for key, (start, end) in ranges.items()
            }

    reader = ControlRateEpisode(Source(), 30)
    result = reader.read({"pose": (30, 130), "image": (30, None)})
    assert result["image"] == 60
    assert np.array_equal(result["pose"][:, 0], np.arange(60, 260, 2))
    assert reader.metadata["total_frames"] == 301
    assert len(reader.read({"pose": (290, 400)})["pose"]) == 11


@pytest.mark.parametrize("source,target", [(0, 30), (30, 0), (np.nan, 30)])
def test_reject_invalid_rates(source, target):
    with pytest.raises(ValueError):
        control_frame_indices(100, source, target)
