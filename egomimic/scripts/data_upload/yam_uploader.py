"""Upload collected Yam HDF5 demos with the existing RLDB metadata/S3 workflow."""

import argparse
import asyncio
from pathlib import Path


def collect_files(local_dir):
    return sorted(
        path
        for path in Path(local_dir).iterdir()
        if path.is_file() and path.suffix == ".hdf5"
    )


def yam_uploader():
    from egomimic.scripts.data_upload.abstract_upload import Uploader

    uploader = Uploader(
        embodiment="yam_bimanual",
        datatype=".hdf5",
        collect_files=collect_files,
        defaults={"rig_name": "yam"},
    )
    # Keep the established raw object prefix while using the canonical
    # downstream embodiment name in uploader-generated metadata.
    uploader.s3_base_prefix = "raw_v2/yam/"
    return uploader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        type=Path,
        metavar="DIRECTORY",
        help="List input demos without credentials or uploading",
    )
    args = parser.parse_args()
    if args.list is not None:
        for path in collect_files(args.list):
            print(path)
        return
    asyncio.run(yam_uploader().run())


if __name__ == "__main__":
    main()
