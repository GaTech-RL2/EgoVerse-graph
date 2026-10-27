"""Allocated ImageNet-only smoke postflight or distinct zero-update capacity."""

import argparse
import json
import sys
from pathlib import Path

# Match the maintained world-model CLIs when invoked by file path from an
# immutable source checkout, without assuming an editable package install.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from egomimic.pl_utils.imagenet_smoke import (
    validate_imagenet_capacity, verify_imagenet_smoke,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--capacity-config", type=Path)
    parser.add_argument("--capacity-output", type=Path)
    parser.add_argument("--capacity-source-manifest-sha256")
    parser.add_argument("--capacity-data-manifest-sha256")
    args = parser.parse_args()
    if args.capacity_config is not None:
        if not all((args.capacity_output, args.capacity_source_manifest_sha256, args.capacity_data_manifest_sha256)):
            parser.error("Capacity requires its separate output and exact source/data manifest bindings")
        receipt = validate_imagenet_capacity(args.capacity_config, args.capacity_output,
            source_manifest_sha256=args.capacity_source_manifest_sha256,
            data_manifest_sha256=args.capacity_data_manifest_sha256)
    else:
        if any((args.capacity_output, args.capacity_source_manifest_sha256, args.capacity_data_manifest_sha256)):
            parser.error("Capacity-only arguments require --capacity-config")
        receipt = verify_imagenet_smoke(args.output)
    if receipt is not None:
        print(json.dumps(dict(status=receipt["status"], schema=receipt["schema"],
                             job_id=receipt["job_id"]), sort_keys=True))


if __name__ == "__main__":
    main()
