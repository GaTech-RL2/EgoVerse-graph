"""Sync the 2026-09-17 rl2 YAM stationery re-upload (lab='rl2',
task='organize_stationary_updated', rig_name='rl2_abc') from the SQL-indexed
bucket into the shared PACE mirror.

Background: the original lab='rl2' task='organize_stationary' rows were hard
deleted on 2026-09-16 and re-uploaded on 09-17 under the '_updated' task names
by operator aniketh. The new processed zarrs exist in S3 but none were on the
Phoenix mirror, so every data config resolving against the mirror silently saw
ABC episodes only. Same shape as scripts/pi05/sync_abc_eva.py; skips episodes
already present, safe to re-run.
"""

from pathlib import Path

from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.zarr_dataset_multi import S3EpisodeResolver
from egomimic.utils.aws.aws_data_utils import load_env

MIRROR = Path("/storage/project/r-dxu345-0/shared/egoverseS3ZarrDatasets")
TASK = "organize_stationary_updated"


def main():
    load_env()
    filters = DatasetFilter(
        filter_lambdas=[
            "lambda row: str(row.get('embodiment', '')) == 'yam_bimanual'",
            "lambda row: str(row.get('lab', '')) == 'rl2'",
            f"lambda row: str(row.get('task', '')) == {TASK!r}",
            "lambda row: str(row.get('rig_name', '')) == 'rl2_abc'",
            "lambda row: row.get('is_deleted', False) == False",
            "lambda row: str(row.get('zarr_processed_path', '')) != ''",
        ]
    )
    paths = S3EpisodeResolver.sync_from_filters(
        bucket_name="rldb",
        filters=filters,
        local_dir=MIRROR,
        numworkers=20,
    )
    print(f"[sync_rl2_stationery] synced/verified {len(paths)} episodes", flush=True)


if __name__ == "__main__":
    main()
