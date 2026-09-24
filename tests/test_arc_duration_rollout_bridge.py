import numpy as np

from egomimic.eval.core.ckpt_loading import _chunk_seam_event


def test_duration_arc_chunk_seam_aligns_previous_unused_tail_to_fresh_head():
    old_chunk = np.arange(120, dtype=np.float32).reshape(40, 3)
    fresh_chunk = (1000 + np.arange(120, dtype=np.float32)).reshape(40, 3)

    event = _chunk_seam_event(
        old_chunk,
        consumed=8,
        new_chunk=fresh_chunk,
        timestep=8,
        embodiment_id=19,
        episode_index=2,
    )

    assert event is not None
    assert event["t"] == 8
    assert event["embodiment_id"] == 19
    assert event["episode_index"] == 2
    np.testing.assert_array_equal(event["previous_tail"], old_chunk[8:])
    np.testing.assert_array_equal(event["new_head"], fresh_chunk[:32])
    np.testing.assert_array_equal(event["executed_head"], fresh_chunk[:32])


def test_duration_arc_chunk_seam_skips_the_initial_or_exhausted_chunk():
    fresh_chunk = np.zeros((40, 3), dtype=np.float32)
    assert (
        _chunk_seam_event(
            None,
            consumed=0,
            new_chunk=fresh_chunk,
            timestep=0,
            embodiment_id=19,
            episode_index=0,
        )
        is None
    )
    assert (
        _chunk_seam_event(
            fresh_chunk,
            consumed=40,
            new_chunk=fresh_chunk,
            timestep=40,
            embodiment_id=19,
            episode_index=0,
        )
        is None
    )
