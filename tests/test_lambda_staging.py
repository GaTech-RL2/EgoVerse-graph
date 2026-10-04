from pathlib import Path
import runpy
import pytest

sync_lines = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/clusters/lambda/stage_data.py"))["sync_lines"]


def test_stage_always_syncs_existing_episode(tmp_path):
    (tmp_path / "abc").mkdir()
    assert sync_lines([("processed_v3/human/abc/", "abc")], tmp_path) == [
        f'sync "s3://rldb/processed_v3/human/abc/*" "{tmp_path}/abc/"'
    ]


def test_absolute_s3_source_and_space_are_preserved(tmp_path):
    assert sync_lines([("s3://rldb/with space/xyz", "xyz")], tmp_path)[0].startswith(
        'sync "s3://rldb/with space/xyz/*" '
    )


@pytest.mark.parametrize("episode", ["", ".", "..", "../abc", "/root/abc", "abc/def"])
def test_unsafe_episode_identifier_is_rejected(tmp_path, episode):
    with pytest.raises(ValueError, match="Unsafe episode"):
        sync_lines([("source", episode)], tmp_path)
