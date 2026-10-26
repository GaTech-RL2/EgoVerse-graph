"""Allocated-CPU preparation of exact ImageNet manifests; no downloads or training."""
import argparse
import json

from egomimic.rldb.imagenet_imagefolder import (
    build_class_language_manifest, build_imagefolder_manifest,
    build_validation_anchor_manifest,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    classes = sub.add_parser("class-language")
    classes.add_argument("--class-catalog", required=True)
    classes.add_argument("--class-catalog-sha256", required=True)
    classes.add_argument("--language-features", required=True)
    classes.add_argument("--language-features-sha256", required=True)
    classes.add_argument("--output", required=True)
    inventory = sub.add_parser("image-inventory")
    inventory.add_argument("--data-root", required=True)
    inventory.add_argument("--class-language", required=True)
    inventory.add_argument("--class-language-sha256", required=True)
    inventory.add_argument("--language-features", required=True)
    inventory.add_argument("--language-features-sha256", required=True)
    inventory.add_argument("--output", required=True)
    anchors = sub.add_parser("validation-anchors")
    anchors.add_argument("--manifest", required=True)
    anchors.add_argument("--manifest-sha256", required=True)
    anchors.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.mode == "class-language":
        receipt = build_class_language_manifest(
            args.class_catalog, args.class_catalog_sha256,
            args.language_features, args.language_features_sha256, args.output)
    elif args.mode == "image-inventory":
        receipt = build_imagefolder_manifest(
            args.data_root, args.class_language, args.class_language_sha256,
            args.language_features, args.language_features_sha256, args.output)
    else:
        receipt = build_validation_anchor_manifest(
            args.manifest, args.manifest_sha256, args.output)
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
