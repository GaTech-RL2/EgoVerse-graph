"""Slow-pace 366 hybrid / tri / event-stream / reference recipes (robot-only BC + RL2 Aria cotrain) with a chosen
rotation cap.

build_slowpace366.py [--rotation-deg 72] [--variants ...]           regenerate data + experiment leaves
build_slowpace366.py --check [--rotation-deg 72] [--variants ...]   compose each recipe, resolve its episodes
    (SQL + Phoenix mirror; missing zarrs sync from R2) and assert 366 / 24 robot and 38 Aria episodes, load one
    real sample per dataset and check the token width. CPU node only. Normally run via launch_slowpace366.sh.

No arguments = the committed r72 recipes (four hybrid / tri variants, BC + cotrain, W&B stamp 20261009). Further flags:
--kinds bc|cotrain ..., --stamp <date in the W&B id>, --fixes (2026-10-11 evaluator options: the tick-prefix regime,
and speed floors for arcvelhyb / arcveltri). arcdurevt / arcvelevt are BC only. The 2026-10-11 relaunch:
build_slowpace366.py --rotation-deg 360 --kinds bc --stamp 20261011 --fixes \
    --variants arcdurhyb arcvelhyb arcdurtri arcveltri arcdurevt arcvelevt
arcdurref / arcvelref (Ryan's two-clock token with rotation6d, fixed-range normalization) are BC only as well:
build_slowpace366.py --rotation-deg 360 --kinds bc --stamp 20261011 --variants arcdurref arcvelref
"""

import argparse
import math

from recipe_builders import (
    CONFIGS,
    EVT_LAYOUTS,
    HYBRID_LAYOUTS,
    REF_LAYOUTS,
    build_slowpace366,
    require_compute_node,
)


def names(variants, rotation_deg, kinds=("bc", "cotrain")):
    tag = f"r{rotation_deg:g}"
    return [
        (
            k,
            v,
            f"{'bc_rl2_stattempo_slowpace366' if k == 'bc' else 'cotrain_rl2_stattempo_slowpace366_aria'}_{v}_{tag}",
        )
        for k in kinds
        for v in variants
    ]


def check(variants, rotation_deg, kinds=("bc", "cotrain")):
    import numpy as np
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import instantiate

    expect = {
        ("train_datasets", "yam_bimanual"): 366,
        ("valid_datasets", "yam_bimanual"): 24,
        ("train_datasets", "human_bimanual"): 38,
    }
    for kind, v, name in names(variants, rotation_deg, kinds):
        with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
            cfg = compose(
                "train_zarr_cartesian",
                overrides=[f"+experiment=yam_arc_grid/{name}"],
                return_hydra_config=True,
            )
        HydraConfig.instance().set_config(cfg)
        width = 24 if v in REF_LAYOUTS else 18 if v.endswith("hyb") else 20
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
        default=list(HYBRID_LAYOUTS),  # pinned: a bare run never writes evt recipes
        choices=[*HYBRID_LAYOUTS, *EVT_LAYOUTS, *REF_LAYOUTS],
    )
    p.add_argument(
        "--kinds", nargs="+", default=["bc", "cotrain"], choices=["bc", "cotrain"]
    )
    p.add_argument("--stamp", default="20261009", help="date in the W&B id")
    p.add_argument(
        "--fixes",
        action="store_true",
        help="2026-10-11 evaluator options: tick-prefix regime, speed floors for the speed variants",
    )
    p.add_argument("--check", action="store_true")
    p.add_argument(
        "--names", action="store_true", help="print the experiment names and exit"
    )
    a = p.parse_args()
    if "cotrain" in a.kinds and set(a.variants) & {*EVT_LAYOUTS, *REF_LAYOUTS}:
        p.error(
            "arcdurevt / arcvelevt / arcdurref / arcvelref are BC only: pass --kinds bc"
        )
    if a.names:
        # Marked: under srun the Slurm prolog / epilog lines share stdout.
        print(
            "NAMES "
            + " ".join(n for _, _, n in names(a.variants, a.rotation_deg, a.kinds))
        )
    elif a.check:
        require_compute_node()
        check(a.variants, a.rotation_deg, a.kinds)
    else:
        build_slowpace366(a.variants, a.rotation_deg, a.kinds, a.stamp, a.fixes)
