"""Data configs for the slow-pace YAM + RL2 Aria cotrain runs (Aidan, 2026-10-01) and their robot-only (BC) twins (10-05).

Robot side: the slow-pace pool by its rule (2a9f02b7: every rl2 YAM organize_stationary episode by Elmo or Aidan, minus
the shared 24-episode held-out robot set), re-evaluated at build time so new uploads join; 217 train eps when the rule
was made, and the build refuses if any of those 217 drops out. Validation is that held-out set. Same leaves as
scratch_rl2_stattempo_slowpace_{time,arcdur}. Human side: every RL2 Aria episode
(SQL lab=rl2, embodiment in --embodiments, frames > 0, zarr registered) whose task is in --tasks and whose hash date
is on/after --since. The selection is frozen into the generated configs as an explicit hash list, plus a manifest. Elmo's first upload (2026-10-01) registered as
embodiment='aria', task='organize stationary' (space) before conversion; both spellings and both embodiment labels are
accepted by default.

    build_slowpace_aria_cotrain.py --list [--since D]          rl2 Aria tasks/operators recorded since D, then exit
    build_slowpace_aria_cotrain.py --tasks T [T ...] [...]     write data/abc_arc/stationery_slowpace_aria_cotrain_<variant>.yaml
                                                               and the robot-only data/abc_arc/stationery_slowpace_bc_<variant>.yaml
                                                               for --variants (default: all five below)
    build_slowpace_aria_cotrain.py --check                     compose the cotrain + BC experiments and load real samples
                                                               (CPU node; syncs any missing zarrs from R2 into the mirror)
    build_slowpace_aria_cotrain.py --lambda                    write *_lambda twins of the generated data configs and both
                                                               experiments for the rl2-lambda loaner (no SQL there): local
                                                               resolver over LAMBDA_ROOT with resolver pins, rolling last.ckpt
                                                               every 5k, W&B resume allow. --check --lambda checks them there.
Normally run through scripts/e1/launch_slowpace_aria_cotrain.sh.
"""
import argparse, datetime, json, os, sys
from pathlib import Path

CONS = Path(__file__).resolve().parents[2]
H = CONS / "egomimic/hydra_configs"
MIRROR = "/storage/project/r-dxu345-0/shared/egoverseS3ZarrDatasets"
VARIANTS = ("time", "arcdur", "arcdurhyb", "arcvel", "arcvelhyb")
OUT = {v: H / f"data/abc_arc/stationery_slowpace_aria_cotrain_{v}.yaml" for v in VARIANTS}
OUT_BC = {v: H / f"data/abc_arc/stationery_slowpace_bc_{v}.yaml" for v in VARIANTS}
TEMPO = CONS / "scripts/e1/stationery_tempo_manifest.json"
LAMBDA_ROOT = "/workspace/users/agao81/stattempo/data/egoverseS3ZarrDatasets"


def robot_base(v, hashes):
    """The slow-pace robot leaves for variant v: the composed time / arcdur data config of the robot-only runs (the other
    arc variants change only the transform's variant: same window, D, M), training on `hashes`."""
    import re
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(H), version_base=None):
        cfg = compose("train_zarr_cartesian", overrides=[f"data=abc_arc/stationery_tempo_slowpace_{'time' if v == 'time' else 'arcdur'}"]).data
    cfg = OmegaConf.create(OmegaConf.to_container(cfg))  # unstructured, so the human leaf can be added
    for group in ("train_datasets", "valid_datasets"):
        cfg[group].yam_bimanual.resolver.transform_list.variant = v
    lams = cfg.train_datasets.yam_bimanual.filters.filter_lambdas
    S = "frozenset({" + ",".join(f"'{h}'" for h in sorted(hashes)) + "})"
    lams[0], n = re.subn(r"frozenset\(\{[^}]*\}\)", lambda m: S, lams[0])
    assert n == 1, "slow-pace train filter has no frozenset to replace"
    return cfg
MANIFEST = CONS / "scripts/e1/stationery_slowpace_aria_manifest.json"


def rl2_episodes():
    from egomimic.utils.aws.aws_data_utils import load_env
    from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df
    load_env()
    df = episode_table_to_df(create_default_engine())
    return df[(df["lab"] == "rl2") & ~df["is_deleted"].astype(bool) & (df["num_frames"].fillna(-1) > 0)
              & (df["zarr_processed_path"].fillna("") != "")]


def rl2_aria(df, since, embodiments=("human_bimanual", "aria")):
    return df[df["embodiment"].isin(embodiments) & (df["episode_hash"] >= since)]


def robot_pool(df):
    sets = json.loads(TEMPO.read_text())["sets"]
    r = df[(df["embodiment"] == "yam_bimanual") & (df["task"] == "organize_stationary") & (df["rig_name"] == "rl2_abc")
           & df["operator"].astype(str).str.lower().isin(["elmo", "aidan"]) & ~df["episode_hash"].isin(sets["val"]["episodes"])]
    lost = set(sets["train_slowpace"]["episodes"]) - set(r["episode_hash"])
    if lost:
        sys.exit(f"{len(lost)} of the original 217 slow-pace episodes no longer match the rule (deleted/relabelled?): {sorted(lost)[:3]}")
    return r.sort_values("episode_hash")


def human_leaf(variant, hashes, embodiments):
    # 100 source frames = 3.33 s at the Aria 30 fps, the same window the YAM leaf uses; stride 1 so arc length is
    # measured on undecimated samples. Aria zarrs say attrs.embodiment=aria_bimanual, which has no embodiment id:
    # the override registers them as human_bimanual (3), the model's second domain.
    s = ",".join(f"'{h}'" for h in sorted(hashes))
    e = ", ".join(f"'{x}'" for x in sorted(embodiments))
    lam = ("lambda row, S=frozenset({" + s + "}): row['embodiment'] in (" + e + ",) and row['lab'] == 'rl2' "
           "and row['zarr_processed_path'] != '' and row['is_deleted'] == False and row['episode_hash'] in S")
    return {
        "_target_": "egomimic.rldb.zarr.zarr_dataset_multi.MultiDataset._from_resolver",
        "resolver": {
            "_target_": "egomimic.rldb.zarr.e1_resolvers.S3EpisodeResolverWithEmbodimentOverride",
            "embodiment_override": "human_bimanual",
            "folder_path": MIRROR,
            "image_hw": [480, 640],
            "key_map": {"_target_": "egomimic.rldb.embodiment.bimanual_arc.get_keymap", "horizon": 100, "embodiment": "human"},
            "transform_list": {
                "_target_": "egomimic.rldb.embodiment.bimanual_arc.get_transform_list",
                "variant": variant, "chunk_length": 100, "time_rows": 100, "min_distance_unit": 0.4,
                "resampled_vector_length": 100, "stride": 1, "rotation_mode": "euler", "velocity_norm": "path",
                "embodiment": "human",
            },
        },
        "filters": {"_target_": "egomimic.rldb.filters.DatasetFilter", "filter_lambdas": [lam]},
        "mode": "total", "valid_ratio": 0.0, "bounds_check": False,
    }


def build(args):
    df = rl2_episodes()
    robot = robot_pool(df)
    sel = rl2_aria(df, args.since, args.embodiments)
    sel = sel[sel["task"].isin(args.tasks)]
    if args.operators:
        sel = sel[sel["operator"].isin(args.operators)]
    sel = sel.sort_values("episode_hash")
    if args.limit:
        sel = sel.head(args.limit)
    if sel.empty:
        sys.exit(f"NO rl2 Aria episodes with frames match tasks={args.tasks} since={args.since} operators={args.operators or 'any'}"
                 " -- not ingested/converted yet? See --list.")
    from omegaconf import OmegaConf
    hashes = list(sel["episode_hash"])
    hours = float(sel["num_frames"].sum()) / 30 / 3600
    who = sel["operator"].value_counts().to_dict()
    r_hashes = list(robot["episode_hash"])
    r_hours = float(robot["num_frames"].sum()) / 30 / 3600
    r_who = robot["operator"].value_counts().to_dict()
    for v in args.variants:
        cfg = robot_base(v, r_hashes)
        stamp = f"# GENERATED {datetime.datetime.now():%Y-%m-%d %H:%M} by scripts/e1/build_slowpace_aria_cotrain.py -- rebuild, don't edit.\n"
        rn = (f"# Robot: slow-pace pool (Elmo + Aidan organize_stationary minus the shared 24-ep val) {len(r_hashes)} train eps / {r_hours:.2f} h,"
              f" operators {r_who}; leaves of data/abc_arc/stationery_tempo_slowpace_{'time' if v == 'time' else 'arcdur'}.yaml, variant {v}.\n")
        OUT_BC[v].write_text(stamp + rn + "# Robot-only (BC) twin of the cotrain config: no human leaf.\n" + OmegaConf.to_yaml(cfg))
        cfg.train_datasets.human_bimanual = human_leaf(v, hashes, set(sel["embodiment"]))
        cfg.train_dataloader_params.human_bimanual = {"batch_size": args.human_batch, "num_workers": 6, "persistent_workers": True}
        head = (stamp + rn +
                f"# Human: {len(hashes)} RL2 Aria eps / {hours:.2f} h, tasks {args.tasks}, since {args.since}, operators {who}.\n"
                f"# Per step: robot batch 32 + human batch {args.human_batch}; validation is robot-only.\n")
        OUT[v].write_text(head + OmegaConf.to_yaml(cfg))
    MANIFEST.write_text(json.dumps({
        "created": datetime.datetime.now().isoformat(timespec="seconds"), "query": vars(args), "variants": list(args.variants), "n": len(hashes),
        "hours": round(hours, 3), "operators": who, "tasks": sel["task"].value_counts().to_dict(),
        "rig_name": sel["rig_name"].value_counts().to_dict(), "embodiment": sel["embodiment"].value_counts().to_dict(),
        "episodes": hashes,
        "robot": {"n": len(r_hashes), "hours": round(r_hours, 3), "operators": r_who, "episodes": r_hashes}}, indent=1))
    print(f"human: {len(hashes)} eps, {hours:.2f} h, operators {who}, rigs {sel['rig_name'].value_counts().to_dict()}")
    print(f"robot: slow-pace pool, {len(r_hashes)} train eps ({len(r_hashes) - 217} beyond the original 217), {r_hours:.2f} h,"
          f" operators {r_who}, shared 24-ep val")
    for v in args.variants:
        print("wrote", OUT[v].relative_to(CONS), "and", OUT_BC[v].relative_to(CONS))


def lambda_twins(variants):
    """Loaner twins: same leaves, episodes and recipe; LocalEpisodeResolverWithEmbodimentOverride over LAMBDA_ROOT with a hash-only
    filter (local rows carry zarr attrs + episode_hash, not SQL lab/task) and a pin on episode count + name sha256."""
    import re
    from omegaconf import OmegaConf
    from egomimic.rldb.zarr.zarr_dataset_multi import episode_names_sha256
    man = json.loads(MANIFEST.read_text())
    R, N = man["robot"]["n"], man["n"]
    for v in variants:
        for kind, src in (("cotrain", OUT[v]), ("bc", OUT_BC[v])):
            cfg = OmegaConf.load(src)
            for g in ("train_datasets", "valid_datasets"):
                for leaf in cfg[g].values():
                    hashes = sorted(re.findall(r"'(\d{4}(?:-\d\d){5}-\d{6})'", leaf.filters.filter_lambdas[0]))
                    r = leaf.resolver
                    r._target_ = "egomimic.rldb.zarr.zarr_dataset_multi.LocalEpisodeResolverWithEmbodimentOverride"
                    r.folder_path = LAMBDA_ROOT
                    r.expected_episode_count = len(hashes)
                    r.expected_episode_names_sha256 = episode_names_sha256(hashes)
                    leaf.filters.filter_lambdas = ["lambda row, S=frozenset({" + ",".join(f"'{h}'" for h in hashes) +
                                                   "}): str(row['episode_hash']) in S"]
            n = {g: {e: l.resolver.expected_episode_count for e, l in cfg[g].items()} for g in ("train_datasets", "valid_datasets")}
            assert n["train_datasets"]["yam_bimanual"] == R and n["valid_datasets"]["yam_bimanual"] == 24, n
            assert kind == "bc" or n["train_datasets"]["human_bimanual"] == N, n
            head = "".join(l + "\n" for l in src.read_text().splitlines() if l.startswith("#"))
            src.with_name(src.stem + "_lambda.yaml").write_text(
                head + f"# LAMBDA twin (--lambda): local resolver over {LAMBDA_ROOT} + resolver pins, no SQL.\n" + OmegaConf.to_yaml(cfg))
            base = f"cotrain_rl2_stattempo_slowpace_aria_{v}" if kind == "cotrain" else f"bc_rl2_stattempo_slowpace_{v}"
            run = f"stattempo_slowpace{R}_aria{N}_cotrain_{v}" if kind == "cotrain" else f"stattempo_slowpace{R}_bc_{v}"
            (H / f"experiment/yam_arc_grid/{base}_lambda.yaml").write_text(f"""# @package _global_
# LAMBDA twin of {base}, GENERATED by scripts/e1/build_slowpace_aria_cotrain.py --lambda: local data under {LAMBDA_ROOT}
# with resolver pins (the loaner has no SQL), a rolling last.ckpt every 5k steps and W&B resume: allow so a requeued job resumes.
# Model, recipe, evaluator and episodes are the Phoenix experiment's ({R} robot{f' + {N} Aria' if kind == 'cotrain' else ''} train eps).
defaults:
- /experiment/yam_arc_grid/{base}
- _self_
- override /data: abc_arc/{src.stem}_lambda
name: {base}_lambda
description: h640t8_d384x10_cos10k_240k_s42_{base}_lambda
paths:
  dataset_dir: {LAMBDA_ROOT}
callbacks:
  model_checkpoint:
    save_last: false
  resume_last:
    _target_: lightning.pytorch.callbacks.ModelCheckpoint
    dirpath: ${{paths.output_dir}}/checkpoints
    save_last: true
    save_top_k: 0
    every_n_train_steps: 5000
logger:
  wandb:
    id: {run}_lambda_20261005_s42
    resume: allow
""")
            print("wrote", src.stem + "_lambda.yaml", "and experiment", base + "_lambda")


def check(variants, suffix="", root=MIRROR):
    import numpy as np, zarr
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    man = json.loads(MANIFEST.read_text())

    def dims(node):  # every action_dim / act_dim value in a resolved model config
        if isinstance(node, dict):
            return {f"{k}={v}" for k, v in node.items() if k in ("action_dim", "act_dim")} | {d for v in node.values() for d in dims(v)}
        return {d for v in node for d in dims(v)} if isinstance(node, list) else set()
    for v in variants:
        with initialize_config_dir(config_dir=str(H), version_base=None):
            cfg = compose("train_zarr_cartesian", overrides=[f"+experiment=yam_arc_grid/cotrain_rl2_stattempo_slowpace_aria_{v}{suffix}"],
                          return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)
        print(f"== {v}: train {list(cfg.data.train_datasets)}, valid {list(cfg.data.valid_datasets)}, "
              f"model domains {cfg.model.pipeline.stages[1].domains}", flush=True)
        for emb in ("human_bimanual", "yam_bimanual"):
            ds = instantiate(cfg.data.train_datasets[emb])  # resolver: SQL filter, R2 sync of missing zarrs, zarr load
            item = ds[len(ds) // 2]
            shapes = {k: tuple(getattr(x, "shape", ())) for k, x in item.items() if k.startswith(("actions", "observations"))}
            print(f"   {emb}: {len(ds.datasets)} eps, {len(ds)} samples; {shapes}", flush=True)
            if emb == "human_bimanual" and v == "time":
                fps = {zarr.open_group(f"{root}/{h}", mode="r").attrs.get("fps") for h in man["episodes"]}
                assert fps <= {30, "30"}, f"Aria fps {fps}: the 100-frame window assumes 30 fps -- set horizon/chunk_length"
                a = np.asarray(item["actions_time"])
                print(f"   human chunk: xyz travel L {np.linalg.norm(np.diff(a[:, :3], axis=0), axis=1).sum():.3f} m,"
                      f" R {np.linalg.norm(np.diff(a[:, 7:10], axis=0), axis=1).sum():.3f} m over 3.33 s; grip cols {a[0, [6, 13]]}")
            if emb == "yam_bimanual" and "robot" in man:
                assert len(ds.datasets) == man["robot"]["n"], f"robot leaf has {len(ds.datasets)} eps, manifest {man['robot']['n']}"
        # BC twin: same robot leaves and action dims, no human leaf or domain
        with initialize_config_dir(config_dir=str(H), version_base=None):
            bc = compose("train_zarr_cartesian", overrides=[f"+experiment=yam_arc_grid/bc_rl2_stattempo_slowpace_{v}{suffix}"])
        assert list(bc.data.train_datasets) == ["yam_bimanual"], list(bc.data.train_datasets)
        assert list(bc.model.pipeline.stages[1].domains) == ["yam_bimanual"], bc.model.pipeline.stages[1].domains
        for g in ("train_datasets", "valid_datasets"):
            assert OmegaConf.to_container(bc.data[g].yam_bimanual) == OmegaConf.to_container(cfg.data[g].yam_bimanual), f"bc {g} differs"
        d_bc, d_co = dims(OmegaConf.to_container(bc.model, resolve=True)), dims(OmegaConf.to_container(cfg.model, resolve=True))
        assert d_bc == d_co, (d_bc, d_co)
        assert OmegaConf.to_container(bc.evaluator) == OmegaConf.to_container(cfg.evaluator), "bc evaluator differs"
        print(f"   bc twin: robot leaves + evaluator identical, domains {list(bc.model.pipeline.stages[1].domains)}, {sorted(d_bc)}", flush=True)
    print("CHECK_OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", default=["organize stationary", "organize_stationary"], help="SQL task names of the Aria episodes")
    p.add_argument("--embodiments", nargs="+", default=["human_bimanual", "aria"], help="SQL embodiment labels of RL2 Aria rows")
    p.add_argument("--since", default="2026-10-01", help="earliest episode-hash date (YYYY-MM-DD)")
    p.add_argument("--operators", nargs="*", help="restrict to these SQL operators")
    p.add_argument("--human-batch", type=int, default=32)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    p.add_argument("--limit", type=int, help="first N episodes only (smoke tests)")
    p.add_argument("--list", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--lambda", dest="lam", action="store_true", help="write the *_lambda twins (with --check: check them)")
    a = p.parse_args()
    if a.list:
        df = rl2_aria(rl2_episodes(), a.since, a.embodiments)
        print(f"rl2 Aria ({'/'.join(a.embodiments)}) episodes with frames since {a.since}: {len(df)}")
        if len(df):
            print(df.groupby(["embodiment", "task", "operator", "rig_name"]).agg(n=("episode_hash", "size"), frames=("num_frames", "sum"),
                  first=("episode_hash", "min"), last=("episode_hash", "max")).to_string())
    elif a.check:
        check(a.variants, *(("_lambda", LAMBDA_ROOT) if a.lam else ()))
    elif a.lam:
        lambda_twins(a.variants)
    else:
        build(a)
