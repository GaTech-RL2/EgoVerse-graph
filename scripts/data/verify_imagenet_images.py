"""Allocated CPU only: complete raw ImageNet decode, or later language join."""

import argparse
import json
import signal

from egomimic.rldb import imagenet_imagefolder as adapter
from egomimic.rldb.imagenet_preparation import join_raw_inventory, verify_images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    raw = modes.add_parser("inventory-decode")
    raw.add_argument("--train-root", required=True)
    raw.add_argument("--val-root", required=True)
    raw.add_argument("--train-sidecars", required=True,
                     help="Exact exclusion-receipt JSON object containing an explicit exclusions list")
    raw.add_argument("--train-sidecars-sha256", required=True)
    raw.add_argument("--output", required=True)
    raw.add_argument("--workers", type=int, default=4)
    raw.add_argument("--max-inflight", type=int, default=16)
    join = modes.add_parser("join-language")
    join.add_argument("--raw-report", required=True)
    join.add_argument("--raw-report-sha256", required=True)
    join.add_argument("--class-language", required=True)
    join.add_argument("--class-language-sha256", required=True)
    join.add_argument("--language-features", required=True)
    join.add_argument("--language-features-sha256", required=True)
    join.add_argument("--output", required=True)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError("Allocated producer received signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    if args.mode == "inventory-decode":
        # adapter._read_json deliberately requires an object; the input receipt
        # is an object containing an explicit list, not an unbound CLI glob.
        excluded = adapter._read_json(args.train_sidecars, args.train_sidecars_sha256,
                                      maximum=64 * 1024)
        if set(excluded) != {"schema", "exclusions"} or excluded["schema"] != "imagenet-train-sidecar-exclusions-v1":
            raise ValueError("Exact explicit train sidecar exclusion receipt required")
        receipt = verify_images(args.train_root, args.val_root, args.output,
                    workers=args.workers, max_inflight=args.max_inflight,
                    train_sidecar_exclusions=excluded["exclusions"])
    else:
        receipt = join_raw_inventory(args.raw_report, args.raw_report_sha256,
                    args.class_language, args.class_language_sha256,
                    args.language_features, args.language_features_sha256, args.output)
    print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
