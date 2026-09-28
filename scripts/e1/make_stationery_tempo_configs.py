#!/usr/bin/env python3
"""Write the data + experiment configs for the rl2 sort-stationery tempo-bucket runs from the split manifest.

  data/abc_arc/stationery_tempo_{slow,medium,fast,mixed}_{time,arcdur}.yaml   train = train_<b>, valid = val (all 24 held out)
  data/abc_arc/stationery_tempo_eval_{time,arcdur}_{val_slow,val_medium,val_fast,val}.yaml   EVAL-ONLY, train = valid = <set>
  experiment/yam_arc_grid/scratch_rl2_stattempo_{b}_{variant}.yaml

Loader and recipe are the towels394 cell's (scratch_rl2_towels394_*): one loader path for both variants (bimanual_arc keymap +
transform list, variant time | arcdur), Elmo's h640t8_d384x10, 240k steps, warmup 10k, seed 42, BimanualTempoEval. Only the
episode selection differs: an explicit hash frozenset per split (rl2 only; both stationery task names).

With --lambda it also writes *_lambda twins for the rl2-lambda loaner (no SQL there): LocalEpisodeResolverWithEmbodimentOverride over
/workspace/users/agao81/datasets/egoverseS3ZarrDatasets with a resolver pin (episode count + name sha256) and a hash-only filter (the local
resolver's rows carry zarr attrs + episode_hash, not SQL lab/task); a rolling last.ckpt every 5k steps and W&B resume: allow, because the
loaner jobs run at a preemptible QOS and requeue. Everything else (pool, recipe, 1 GPU x batch 32) is identical.

usage: make_stationery_tempo_configs.py --manifest <manifest.json> --repo <worktree> [--lambda]
"""

import argparse
import json
import os

MIRROR = "/storage/project/r-dxu345-0/shared/egoverseS3ZarrDatasets"
LAMBDA_ROOT = "/workspace/users/agao81/datasets/egoverseS3ZarrDatasets"
BUCKETS = ("slow", "medium", "fast", "mixed")
VARIANTS = ("time", "arcdur")
EVAL_SETS = ("val_slow", "val_medium", "val_fast", "val")

LEAF = """  yam_bimanual:
    _target_: egomimic.rldb.zarr.zarr_dataset_multi.MultiDataset._from_resolver
    resolver:
      _target_: egomimic.rldb.zarr.e1_resolvers.S3EpisodeResolverWithEmbodimentOverride
      embodiment_override: yam_bimanual
      folder_path: {mirror}
      image_hw:
      - 480
      - 640
      key_map:
        _target_: egomimic.rldb.embodiment.bimanual_arc.get_keymap
        horizon: 200
        embodiment: yam
        drop_wrist_images: false
      transform_list:
        _target_: egomimic.rldb.embodiment.bimanual_arc.get_transform_list
        variant: {variant}
        chunk_length: 200
        time_rows: 100
        min_distance_unit: 0.4
        resampled_vector_length: 100
        stride: 1
        rotation_mode: euler
        velocity_norm: path
        embodiment: yam
    filters:
      _target_: egomimic.rldb.filters.DatasetFilter
      filter_lambdas:
      - "{lam}"
    mode: total
    valid_ratio: 0.0
    bounds_check: false
"""

TAIL = """train_dataloader_params:
  yam_bimanual:
    batch_size: 32
    num_workers: 7
    persistent_workers: true
valid_dataloader_params:
  yam_bimanual:
    batch_size: 32
    num_workers: 7
    persistent_workers: true
"""


def lam(hashes):
    s = ",".join(f"'{h}'" for h in sorted(hashes))
    return ("lambda row, S=frozenset({" + s + "}): row['embodiment'] == 'yam_bimanual' and row['lab'] == 'rl2' "
            "and row['task'] in ('organize_stationary', 'organize_stationary_updated') and row['rig_name'] == 'rl2_abc' "
            "and row['zarr_processed_path'] != '' and row['is_deleted'] == False and row['episode_hash'] in S")


def lam_local(hashes):
    s = ",".join(f"'{h}'" for h in sorted(hashes))
    return "lambda row, S=frozenset({" + s + "}): str(row['episode_hash']) in S"


def lambda_leaf(variant, hashes):
    from egomimic.rldb.zarr.zarr_dataset_multi import episode_names_sha256
    t = LEAF.format(mirror=LAMBDA_ROOT, variant=variant, lam=lam_local(hashes))
    t = t.replace("egomimic.rldb.zarr.e1_resolvers.S3EpisodeResolverWithEmbodimentOverride",
                  "egomimic.rldb.zarr.zarr_dataset_multi.LocalEpisodeResolverWithEmbodimentOverride")
    pin = (f"      expected_episode_count: {len(hashes)}\n"
           f"      expected_episode_names_sha256: {episode_names_sha256(sorted(hashes))}\n")
    return t.replace("        embodiment: yam\n    filters:", "        embodiment: yam\n" + pin + "    filters:")


def lambda_data_cfg(man, variant, train, valid, what):
    sets = man["sets"]
    return (header(man, variant, what).replace("# persistent_workers", "# LAMBDA twin: local resolver + resolver pins, no SQL.\n# persistent_workers")
            + "train_datasets:\n" + lambda_leaf(variant, sets[train]["episodes"])
            + "valid_datasets:\n" + lambda_leaf(variant, sets[valid]["episodes"]) + TAIL)


def lambda_experiment(t, bucket, variant):
    new = f"scratch_rl2_stattempo_{bucket}_{variant}"
    t = t.replace(f"abc_arc/stationery_tempo_{bucket}_{variant}\n", f"abc_arc/stationery_tempo_{bucket}_{variant}_lambda\n")
    t = t.replace(f"name: {new}\n", f"name: {new}_lambda\n").replace(f"_240k_s42_{new}\n", f"_240k_s42_{new}_lambda\n")
    t = t.replace(f"id: {new}_20260925_s42", f"id: {new}_lambda_20260925_s42")
    t = t.replace("  dataset_dir: /storage/project/r-dxu345-0/shared/egoverseS3ZarrDatasets", "  dataset_dir: " + LAMBDA_ROOT)
    t = t.replace("    save_last: true\n", "    save_last: false\n")
    t = t.replace("    auto_insert_metric_name: false\n", "    auto_insert_metric_name: false\n  resume_last:\n"
                  "    _target_: lightning.pytorch.callbacks.ModelCheckpoint\n    dirpath: ${paths.output_dir}/checkpoints\n"
                  "    save_last: true\n    save_top_k: 0\n    every_n_train_steps: 5000\n", 1)
    t = t.replace("    resume: never\n", "    resume: allow\n").replace(f"    - tempo_{bucket}\n", f"    - tempo_{bucket}\n    - lambda\n")
    t = t.replace("# towels394 time cell by scripts/e1/make_stationery_tempo_configs.py.",
                  "# towels394 time cell by scripts/e1/make_stationery_tempo_configs.py. LAMBDA twin: local data, rolling last.ckpt, W&B resume allow.")
    for k in ("_lambda\n", "resume: allow", "resume_last:", "save_last: false", LAMBDA_ROOT):
        assert k in t, k
    return t


def header(man, variant, what):
    s = man["sets"]
    brief = lambda k: f"{k} {s[k]['n']} eps / {s[k]['hours']} h {s[k]['buckets']}"  # noqa: E731
    return ("# GENERATED by scripts/e1/make_stationery_tempo_configs.py from scripts/e1/stationery_tempo_manifest.json -- do not edit by hand.\n"
            f"# rl2 sort-stationery tempo buckets ({man['version']}), {variant}: {what}\n"
            f"# Pool: {man['pool']['episodes']} scored rl2 episodes ({', '.join(man['pool']['tasks'])}), tau tertile cuts "
            f"{man['pool']['tertile_cuts_m_per_s'][0]} / {man['pool']['tertile_cuts_m_per_s'][1]} m/s.\n"
            f"# {brief('train_slow')}; {brief('train_medium')}; {brief('train_fast')}; {brief('train_mixed')}; {brief('val')}.\n"
            "# persistent_workers true: default re-forks loader workers each epoch and this stack deadlocks there (job 13307279).\n"
            "_target_: egomimic.pl_utils.pl_data_utils.MultiDataModuleWrapper\n")


def data_cfg(man, variant, train, valid, what):
    sets = man["sets"]
    return (header(man, variant, what)
            + "train_datasets:\n" + LEAF.format(mirror=MIRROR, variant=variant, lam=lam(sets[train]["episodes"]))
            + "valid_datasets:\n" + LEAF.format(mirror=MIRROR, variant=variant, lam=lam(sets[valid]["episodes"])) + TAIL)


def experiment(template, bucket, variant, n_eps=133):
    t = template
    old = "scratch_rl2_towels394_time"
    new = f"scratch_rl2_stattempo_{bucket}_{variant}"
    assert t.count(old) >= 3
    t = t.replace("abc_arc/towels_rl2_394_time", f"abc_arc/stationery_tempo_{bucket}_time")
    t = t.replace(old, new).replace("scratch_rl2_stattempo_", "scratch_rl2_stattempo_")
    t = t.replace("id: " + new + "_20260925_s42", "id: " + new + "_20260925_s42")
    t = t.replace("    - towels\n    - towels394\n", f"    - stationery\n    - stattempo\n    - tempo_{bucket}\n")
    first = t.index("\n", t.index("\n", t.index("\n") + 1) + 1)  # replace the 2-line header comment
    t = ("# @package _global_\n"
         f"# FROM SCRATCH, RL2-only sort stationery, tempo bucket '{bucket}' ({n_eps} eps; see the data config header), variant {variant}.\n"
         "# One of the stationery tempo cells {slow, medium, fast, mixed, slowpace, all} x {time, arcdur} sharing loader path, recipe (Elmo h640t8_d384x10, 240k, warmup 10k,\n"
         "# seed 42), evaluator (BimanualTempoEval) and the same 24-episode held-out validation set (8 per bucket). GENERATED from the\n"
         "# towels394 time cell by scripts/e1/make_stationery_tempo_configs.py." + t[first:])
    if variant == "arcdur":
        t = t.replace("override /model: e1/hpt_flow_wrists_ft\n", "override /model: e1/hpt_flow_wrists_ft_arcdur\n")
        t = t.replace(f"abc_arc/stationery_tempo_{bucket}_time", f"abc_arc/stationery_tempo_{bucket}_arcdur")
        t = t.replace("  variant: time\n", "  variant: arcdur\n").replace("    - time\n", "    - arcdur\n")
        t = t.replace("  action_dim: 14\n", "  action_dim: 16\n")
        assert t.count("action_dim: 16") == 2 and "variant: arcdur" in t
    return t


def write(path, text):
    with open(path, "w") as f:
        f.write(text)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--lambda", dest="lam", action="store_true", help="also write the *_lambda twins")
    ap.add_argument("--pools", default=",".join(BUCKETS), help="comma list of train_<pool> sets to write (default: the 4 tempo pools)")
    args = ap.parse_args()
    man = json.load(open(args.manifest))
    dd = os.path.join(args.repo, "egomimic/hydra_configs/data/abc_arc")
    ed = os.path.join(args.repo, "egomimic/hydra_configs/experiment/yam_arc_grid")
    template = open(os.path.join(ed, "scratch_rl2_towels394_time.yaml")).read()
    for v in VARIANTS:
        for b in args.pools.split(","):
            write(os.path.join(dd, f"stationery_tempo_{b}_{v}.yaml"),
                  data_cfg(man, v, f"train_{b}", "val", f"training on train_{b}, in-training validation on val (8 per bucket)"))
            exp = experiment(template, b, v, man["sets"][f"train_{b}"]["n"])
            write(os.path.join(ed, f"scratch_rl2_stattempo_{b}_{v}.yaml"), exp)
            if args.lam:
                write(os.path.join(dd, f"stationery_tempo_{b}_{v}_lambda.yaml"),
                      lambda_data_cfg(man, v, f"train_{b}", "val", f"training on train_{b}, in-training validation on val (8 per bucket)"))
                write(os.path.join(ed, f"scratch_rl2_stattempo_{b}_{v}_lambda.yaml"), lambda_experiment(exp, b, v))
        for es in EVAL_SETS:
            write(os.path.join(dd, f"stationery_tempo_eval_{v}_{es}.yaml"),
                  data_cfg(man, v, es, es, f"EVAL-ONLY on '{es}'; train and valid both point at it, mode=eval never trains"))


if __name__ == "__main__":
    main()
