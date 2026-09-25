"""Mirror the fold-towel data for BOTH labs into the shared PACE dataset folder.

  abc  task='fold and stack the towels'          (245 in SQL; 5 were missing from the mirror)
  rl2  task='fold_towels' rig='rl2_abc'  (199 in SQL, 198 processed, 1 'Zero Frames'; renamed from fold_towels_updated 09-18;
                                                  none on the mirror -- withheld on 09-17 by
                                                  instruction, reversed 09-18)
Same sync_from_filters path as sync_abc_eva / sync_rl2_stationery; skips present episodes."""
from pathlib import Path
from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.zarr_dataset_multi import S3EpisodeResolver
from egomimic.utils.aws.aws_data_utils import load_env

MIRROR = Path("/storage/project/r-dxu345-0/shared/egoverseS3ZarrDatasets")
BASE = ["lambda row: str(row.get('embodiment', '')) == 'yam_bimanual'",
        "lambda row: row.get('is_deleted', False) == False",
        "lambda row: str(row.get('zarr_processed_path', '')) != ''"]
SETS = {
  "abc fold and stack the towels": BASE + ["lambda row: str(row.get('lab','')) == 'abc'",
      "lambda row: str(row.get('task','')) == 'fold and stack the towels'"],
  "rl2 fold_towels":       BASE + ["lambda row: str(row.get('lab','')) == 'rl2'",
      "lambda row: str(row.get('task','')) == 'fold_towels'",
      "lambda row: str(row.get('rig_name','')) == 'rl2_abc'"],
}
def main():
    load_env()
    for name, lams in SETS.items():
        paths = S3EpisodeResolver.sync_from_filters(bucket_name="rldb",
                    filters=DatasetFilter(filter_lambdas=lams), local_dir=MIRROR, numworkers=20)
        print(f"[sync_towels] {name}: synced/verified {len(paths)} episodes", flush=True)
if __name__ == "__main__": main()
