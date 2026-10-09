"""Slow-pace 366 hybrid / tri recipes (robot-only BC + RL2 Aria cotrain) with a chosen rotation cap.

build_slowpace366.py [--rotation-deg 72] [--variants ...]           regenerate data + experiment leaves
build_slowpace366.py --check [--rotation-deg 72] [--variants ...]   compose each recipe, resolve its episodes
    (SQL + Phoenix mirror; missing zarrs sync from R2) and assert 366 / 24 robot and 38 Aria episodes, load one
    real sample per dataset and check the token width. CPU node only. Normally run via launch_slowpace366.sh.
"""

import argparse
import math

from recipe_builders import (
    CONFIGS,
    HYBRID_LAYOUTS,
    build_slowpace366,
    require_compute_node,
)


def names(variants, rotation_deg):
    tag = f"r{rotation_deg:g}"
    return [
        (
            k,
            v,
            f"{'bc_rl2_stattempo_slowpace366' if k == 'bc' else 'cotrain_rl2_stattempo_slowpace366_aria'}_{v}_{tag}",
        )
        for k in ("bc", "cotrain")
        for v in variants
    ]


def check(variants, rotation_deg):
    import numpy as np
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import instantiate

    expect = {
        ("train_datasets", "yam_bimanual"): 366,
        ("valid_datasets", "yam_bimanual"): 24,
        ("train_datasets", "human_bimanual"): 38,
    }
    for kind, v, name in names(variants, rotation_deg):
        with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
            cfg = compose(
                "train_zarr_cartesian",
                overrides=[f"+experiment=yam_arc_grid/{name}"],
                return_hydra_config=True,
            )
        HydraConfig.instance().set_config(cfg)
        width = 20 if v.endswith("tri") else 18
        assert cfg.model.action_token_dim == width and math.isclose(
            cfg.model.rotation_distance_unit, math.radians(rotation_deg)
        )
        for group in ("train_datasets", "valid_datasets"):
            for emb, leaf in cfg.data[group].items():
                assert math.isclose(
                    leaf.resolver.transform_list.rotation_distance_unit,
                    math.radians(rotation_deg),
                )
                ds = instantiate(leaf)
                n = len(ds.datasets)
                assert (
                    n == expect[(group, emb)]
                ), f"{name} {group}/{emb}: {n} episodes, expected {expect[(group, emb)]}"
                a = np.asarray(ds[len(ds) // 2]["actions_cartesian"])
                assert a.shape == (100, width), f"{name} {group}/{emb}: token {a.shape}"
                print(
                    f"   {name} {group}/{emb}: {n} eps, {len(ds)} samples, token {a.shape}",
                    flush=True,
                )
    print("CHECK_OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--rotation-deg",
        type=float,
        default=72.0,
        help="hybrid / tri rotation stream budget (degrees)",
    )
    p.add_argument(
        "--variants",
        nargs="+",
        default=list(HYBRID_LAYOUTS),
        choices=list(HYBRID_LAYOUTS),
    )
    p.add_argument("--check", action="store_true")
    p.add_argument(
        "--names", action="store_true", help="print the experiment names and exit"
    )
    a = p.parse_args()
    if a.names:
        # Marked: under srun the Slurm prolog / epilog lines share stdout.
        print("NAMES " + " ".join(n for _, _, n in names(a.variants, a.rotation_deg)))
    elif a.check:
        require_compute_node()
        check(a.variants, a.rotation_deg)
    else:
        build_slowpace366(a.variants, a.rotation_deg)
