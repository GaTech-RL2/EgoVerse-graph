"""Rebuild token-analysis CSVs and plots without loading model weights or data."""

import argparse

from egomimic.eval.latent_archive import rebuild_archives


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument(
        "--methods", nargs="+", default=["pca", "umap", "pca_umap", "tsne2d"]
    )
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--pca-for-downstream", action="store_true")
    parser.add_argument(
        "--color-by", choices=["embodiment", "hash"], default="embodiment"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rebuild_archives(
        args.source,
        args.destination,
        methods=args.methods,
        pca_components=args.pca_components,
        pca_for_downstream=args.pca_for_downstream,
        color_by=args.color_by,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
