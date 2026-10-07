"""Data configs for the slow-pace YAM + RL2 Aria cotrain runs (Aidan, 2026-10-01).

Robot side: the slow-pace pool exactly as scratch_rl2_stattempo_slowpace_{time,arcdur} trains on it (Elmo + Aidan,
217 train eps) and validates on it (the shared 24-episode held-out robot set). Human side: every RL2 Aria episode
(SQL lab=rl2, embodiment in --embodiments, frames > 0, zarr registered) whose task is in --tasks and whose hash date
is on/after --since. The selection is frozen into the generated configs as an explicit hash list, plus a manifest. Elmo's first upload (2026-10-01) registered as
embodiment='aria', task='organize stationary' (space) before conversion; both spellings and both embodiment labels are
accepted by default.

    build_slowpace_aria_cotrain.py --list [--since D]          rl2 Aria tasks/operators recorded since D, then exit
    build_slowpace_aria_cotrain.py --tasks T [T ...] [...]     write data/abc_arc/stationery_slowpace_aria_cotrain_<variant>.yaml
                                                               for --variants (default: all five below)
    build_slowpace_aria_cotrain.py --check                     compose both experiments and load real samples (CPU node;
                                                               syncs any missing Aria zarrs from R2 into the mirror)
Normally run through scripts/e1/launch_slowpace_aria_cotrain.sh.
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

from recipe_builders import ROTATION_DISTANCE_UNIT, require_compute_node, robot_data

CONS = Path(__file__).resolve().parents[2]
H = CONS / "egomimic/hydra_configs"
MIRROR = "/storage/project/r-dxu345-0/shared/egoverseS3ZarrDatasets"
VARIANTS = ("time", "arcdur", "arcdurhyb", "arcvel", "arcvelhyb")
OUT = {
    v: H / f"data/abc_arc/stationery_slowpace_aria_cotrain_{v}.yaml" for v in VARIANTS
}


def robot_base(v):
    """The slow-pace robot leaves for variant v: the time / arcdur data configs as generated for the robot-only runs; the
    other arc variants are the arcdur config with only the transform's variant changed (same window, D, M, episodes)."""
    from omegaconf import OmegaConf

    return OmegaConf.create(robot_data("slowpace", v))


MANIFEST = CONS / "scripts/e1/stationery_slowpace_aria_manifest.json"


def rl2_aria(since, embodiments=("human_bimanual", "aria")):
    from egomimic.utils.aws.aws_data_utils import load_env
    from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df

    load_env()
    df = episode_table_to_df(create_default_engine())
    ok = (
        (df["lab"] == "rl2")
        & df["embodiment"].isin(embodiments)
        & ~df["is_deleted"].astype(bool)
        & (df["num_frames"].fillna(-1) > 0)
        & (df["zarr_processed_path"].fillna("") != "")
        & (df["episode_hash"] >= since)
    )
    return df[ok]


def human_leaf(variant, hashes, embodiments):
    # 100 source frames = 3.33 s at the Aria 30 fps, the same window the YAM leaf uses; stride 1 so arc length is
    # measured on undecimated samples. Aria zarrs say attrs.embodiment=aria_bimanual, which has no embodiment id:
    # the override registers them as human_bimanual (3), the model's second domain.
    s = ",".join(f"'{h}'" for h in sorted(hashes))
    e = ", ".join(f"'{x}'" for x in sorted(embodiments))
    lam = (
        "lambda row, S=frozenset({"
        + s
        + "}): row['embodiment'] in ("
        + e
        + ",) and row['lab'] == 'rl2' "
        "and row['zarr_processed_path'] != '' and row['is_deleted'] == False and row['episode_hash'] in S"
    )
    return {
        "_target_": "egomimic.rldb.zarr.zarr_dataset_multi.MultiDataset._from_resolver",
        "resolver": {
            "_target_": "egomimic.rldb.zarr.e1_resolvers.S3EpisodeResolverWithEmbodimentOverride",
            "embodiment_override": "human_bimanual",
            "folder_path": MIRROR,
            "image_hw": [480, 640],
            "key_map": {
                "_target_": "egomimic.rldb.embodiment.bimanual_arc.get_keymap",
                "horizon": 100,
                "embodiment": "human",
                "drop_wrist_images": True,
            },
            "transform_list": {
                "_target_": "egomimic.rldb.embodiment.bimanual_arc.get_transform_list",
                "variant": variant,
                "chunk_length": 100,
                "time_rows": 100,
                "min_distance_unit": 0.4,
                "resampled_vector_length": 100,
                "stride": 1,
                "rotation_mode": "euler",
                "velocity_norm": "path",
                "embodiment": "human",
                "progress_smooth_hz": None,
                "fixed_spacing": False,
                **(
                    {"rotation_distance_unit": ROTATION_DISTANCE_UNIT}
                    if variant.endswith("hyb")
                    else {}
                ),
            },
        },
        "filters": {
            "_target_": "egomimic.rldb.filters.DatasetFilter",
            "filter_lambdas": [lam],
        },
        "mode": "total",
        "valid_ratio": 0.0,
        "bounds_check": False,
    }


def build(args):
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text())
        hashes = manifest["episodes"]
        if (
            len(hashes) != manifest["n"]
            or len(set(hashes)) != len(hashes)
            or not hashes
        ):
            raise ValueError(
                "Frozen Aria manifest must contain its declared unique episode set"
            )
        if args.limit:
            raise ValueError(
                "--limit cannot change a frozen manifest; use the exact recorded selection"
            )
        hours, who = manifest["hours"], manifest["operators"]
        embodiments, rigs = set(manifest["embodiment"]), manifest["rig_name"]
    else:
        df = rl2_aria(args.since, args.embodiments)
        sel = df[df["task"].isin(args.tasks)]
        if args.operators:
            sel = sel[sel["operator"].isin(args.operators)]
        sel = sel.sort_values("episode_hash")
        if args.limit:
            sel = sel.head(args.limit)
        if sel.empty:
            sys.exit(
                f"NO rl2 Aria episodes with frames match tasks={args.tasks} since={args.since} operators={args.operators or 'any'}"
                " -- not ingested/converted yet? See --list."
            )
        hashes = list(sel["episode_hash"])
        hours = float(sel["num_frames"].sum()) / 30 / 3600
        who, rigs = (
            sel["operator"].value_counts().to_dict(),
            sel["rig_name"].value_counts().to_dict(),
        )
        embodiments = set(sel["embodiment"])
        manifest = {
            "tasks": sel["task"].value_counts().to_dict(),
            "rig_name": rigs,
            "embodiment": sel["embodiment"].value_counts().to_dict(),
        }
    from omegaconf import OmegaConf

    for v in args.variants:
        cfg = robot_base(v)
        cfg.train_datasets.human_bimanual = human_leaf(v, hashes, embodiments)
        cfg.train_dataloader_params.human_bimanual = {
            "batch_size": args.human_batch,
            "num_workers": 6,
            "persistent_workers": True,
        }
        head = (
            f"# GENERATED {datetime.datetime.now():%Y-%m-%d %H:%M} by scripts/e1/build_slowpace_aria_cotrain.py -- rebuild, don't edit.\n"
            f"# Robot: data/abc_arc/stationery_tempo_slowpace_{'time' if v == 'time' else 'arcdur'}.yaml, variant {v} (slow-pace pool, 217 train, shared 24-ep val).\n"
            f"# Human: {len(hashes)} RL2 Aria eps / {hours:.2f} h, tasks {args.tasks}, since {args.since}, operators {who}.\n"
            f"# Per step: robot batch 32 + human batch {args.human_batch}; validation is robot-only.\n"
        )
        OUT[v].write_text(head + OmegaConf.to_yaml(cfg))
    MANIFEST.write_text(
        json.dumps(
            {
                **manifest,
                "created": datetime.datetime.now().isoformat(timespec="seconds"),
                "query": vars(args),
                "variants": list(args.variants),
                "n": len(hashes),
                "hours": round(hours, 3),
                "operators": who,
                "episodes": hashes,
            },
            indent=1,
        )
    )
    print(f"human: {len(hashes)} eps, {hours:.2f} h, operators {who}, rigs {rigs}")
    print("robot: slow-pace pool, 217 train eps (~2.2 h), shared 24-ep val")
    for v in args.variants:
        print("wrote", OUT[v].relative_to(CONS))


def check():
    import numpy as np
    import zarr
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import instantiate

    man = json.loads(MANIFEST.read_text())
    for v in man.get("variants", ["time", "arcdur"]):
        with initialize_config_dir(config_dir=str(H), version_base=None):
            cfg = compose(
                "train_zarr_cartesian",
                overrides=[
                    f"+experiment=yam_arc_grid/cotrain_rl2_stattempo_slowpace_aria_{v}"
                ],
                return_hydra_config=True,
            )
        HydraConfig.instance().set_config(cfg)
        print(
            f"== {v}: train {list(cfg.data.train_datasets)}, valid {list(cfg.data.valid_datasets)}, "
            f"model domains {cfg.model.pipeline.stages[1].domains}",
            flush=True,
        )
        for emb in ("human_bimanual", "yam_bimanual"):
            ds = instantiate(
                cfg.data.train_datasets[emb]
            )  # resolver: SQL filter, R2 sync of missing zarrs, zarr load
            item = ds[len(ds) // 2]
            shapes = {
                k: tuple(getattr(x, "shape", ()))
                for k, x in item.items()
                if k.startswith(("actions", "observations"))
            }
            print(
                f"   {emb}: {len(ds.datasets)} eps, {len(ds)} samples; {shapes}",
                flush=True,
            )
            if emb == "human_bimanual" and v == "time":
                fps = {
                    zarr.open_group(f"{MIRROR}/{h}", mode="r").attrs.get("fps")
                    for h in man["episodes"]
                }
                assert (
                    fps <= {30, "30"}
                ), f"Aria fps {fps}: the 100-frame window assumes 30 fps -- set horizon/chunk_length"
                a = np.asarray(item["actions_time"])
                print(
                    f"   human chunk: xyz travel L {np.linalg.norm(np.diff(a[:, :3], axis=0), axis=1).sum():.3f} m,"
                    f" R {np.linalg.norm(np.diff(a[:, 7:10], axis=0), axis=1).sum():.3f} m over 3.33 s; grip cols {a[0, [6, 13]]}"
                )
    print("CHECK_OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=["organize stationary", "organize_stationary"],
        help="SQL task names of the Aria episodes",
    )
    p.add_argument(
        "--embodiments",
        nargs="+",
        default=["human_bimanual", "aria"],
        help="SQL embodiment labels of RL2 Aria rows",
    )
    p.add_argument(
        "--since", default="2026-10-01", help="earliest episode-hash date (YYYY-MM-DD)"
    )
    p.add_argument("--operators", nargs="*", help="restrict to these SQL operators")
    p.add_argument("--human-batch", type=int, default=32)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    p.add_argument("--limit", type=int, help="first N episodes only (smoke tests)")
    p.add_argument(
        "--manifest", help="rebuild from an exact frozen manifest without querying SQL"
    )
    p.add_argument("--list", action="store_true")
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    require_compute_node()
    if a.list:
        df = rl2_aria(a.since, a.embodiments)
        print(
            f"rl2 Aria ({'/'.join(a.embodiments)}) episodes with frames since {a.since}: {len(df)}"
        )
        if len(df):
            print(
                df.groupby(["embodiment", "task", "operator", "rig_name"])
                .agg(
                    n=("episode_hash", "size"),
                    frames=("num_frames", "sum"),
                    first=("episode_hash", "min"),
                    last=("episode_hash", "max"),
                )
                .to_string()
            )
    elif a.check:
        check()
    else:
        build(a)
