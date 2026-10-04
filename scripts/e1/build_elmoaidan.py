# 2026-09-30: Elmo (200) + Aidan (29) organize_stationary, fixed 5 % validation split (11 eps, seed 42), shared by three runs:
#   arcdur E1 (100, 16)   -> consolidated worktree (PR #160 branch), existing E1 path
#   lab wide (100, 28)    -> stattempo-lab-layouts worktree (Aniketh's #193 stack + velocity_layout pass-through)
#   lab stacked (200, 14) -> same
# The split is written out as explicit episode lists so both codebases train/validate on identical episodes.
import csv, json, os, re, sys
import numpy as np
G = "/storage/project/r-dxu345-0/agao81/EgoVerse-graph/.claude/worktrees"
CONS, LAB = f"{G}/arc-bc-consolidated", f"{G}/stattempo-lab-layouts"
MIRROR = "/storage/project/r-dxu345-0/shared/egoverseS3ZarrDatasets"
rows = [r for r in csv.DictReader(open(os.path.expanduser("~/rl2_stat_rows_2026-09-27.csv")))
        if r["task"] == "organize_stationary" and float(r["num_frames"] or -1) > 0 and r["zarr_processed_path"]]
assert len(rows) == 229, len(rows)
eps = sorted(r["episode_hash"] for r in rows); op = {r["episode_hash"]: r["operator"] for r in rows}
nfr = {r["episode_hash"]: float(r["num_frames"]) for r in rows}
perm = list(np.random.default_rng(42).permutation(eps)); n_val = round(0.05 * len(eps))
val, train = sorted(perm[:n_val]), sorted(perm[n_val:])
def summ(v): return {"n": len(v), "hours": round(sum(nfr[h] for h in v) / 30 / 3600, 3), "operators": {o: sum(op[h] == o for h in v) for o in sorted(set(op.values()))}}
split = {"created": "2026-09-30", "pool": "rl2 organize_stationary with frames (Elmo 200 on 9-24 + Aidan 29 on 9-19)", "seed": 42, "valid_fraction": 0.05,
         "train": {**summ(train), "episodes": train}, "val": {**summ(val), "episodes": val}}
for root in (CONS, LAB):
    json.dump(split, open(f"{root}/scripts/e1/stationery_elmoaidan_split.json", "w"), indent=1)
print("train", summ(train), "val", summ(val))

# ---- arcdur (consolidated): reuse the tempo generator's writers
sys.path.insert(0, f"{CONS}/scripts/e1"); import make_stationery_tempo_configs as g
man = json.load(open(f"{CONS}/scripts/e1/stationery_tempo_manifest.json"))
for k, v in (("train_elmoaidan", train), ("val_elmoaidan", val)):
    man["sets"][k] = {**summ(v), "tau_median": None, "buckets": {}, "episodes": v}
json.dump(man, open(f"{CONS}/scripts/e1/stationery_tempo_manifest.json", "w"), indent=1)
open(f"{CONS}/egomimic/hydra_configs/data/abc_arc/stationery_tempo_elmoaidan_arcdur.yaml", "w").write(
    g.data_cfg(man, "arcdur", "train_elmoaidan", "val_elmoaidan", "training on train_elmoaidan (Elmo + Aidan, 218), validation on its own 5 % split (11)"))
tmpl = open(f"{CONS}/egomimic/hydra_configs/experiment/yam_arc_grid/scratch_rl2_towels394_time.yaml").read()
exp = g.experiment(tmpl, "elmoaidan", "arcdur", len(train)).replace(
    "and the same 24-episode held-out validation set (8 per bucket)", "and the fixed 11-episode (5 %) validation split of the Elmo + Aidan pool")
open(f"{CONS}/egomimic/hydra_configs/experiment/yam_arc_grid/scratch_rl2_stattempo_elmoaidan_arcdur.yaml", "w").write(exp)

# ---- lab model: the ft arcdur model with only the flow head's rows/cols following the token (state stem stays 14)
m = open(f"{CONS}/egomimic/hydra_configs/model/e1/hpt_flow_wrists_ft_arcdur.yaml").read()
body = m[m.index("_target_"):]
for a, b in (("    action_horizon: 100\n", "    action_horizon: ${stt.token_rows}\n"), ("    action_dim: 16\n", "    action_dim: ${stt.token_cols}\n"),
             ("      act_dim: 16\n", "      act_dim: ${stt.token_cols}\n"), ("      act_seq: 100\n", "      act_seq: ${stt.token_rows}\n")):
    assert body.count(a) >= 1, a; body = body.replace(a, b)
assert "16\n" not in re.sub(r"crossattn_latent: 16\n", "", body)
os.makedirs(f"{LAB}/egomimic/hydra_configs/model/e1", exist_ok=True)
open(f"{LAB}/egomimic/hydra_configs/model/e1/hpt_flow_wrists_ft_lab.yaml", "w").write(
    "# Aidan's stationery fine-tune model (h640 / t8 / flow d384x10, warmup-cosine over ${ft.max_steps}), copied from\n"
    "# model/e1/hpt_flow_wrists_ft_arcdur.yaml on aidan/arc-bc-consolidated. Only the flow head follows the lab token:\n"
    "# rows = ${stt.token_rows}, cols = ${stt.token_cols} (wide (100, 28) or stacked (200, 14)); the ee_pose state stem stays 14.\n" + body)

# ---- lab data + experiments
def lam(v):
    s = ",".join(f"'{h}'" for h in v)
    return ("lambda row, S=frozenset({" + s + "}): row['embodiment'] == 'yam_bimanual' and row['lab'] == 'rl2' "
            "and row['task'] == 'organize_stationary' and row['zarr_processed_path'] != '' and row['is_deleted'] == False and row['episode_hash'] in S")
def leaf(v, layout):
    return f"""  yam_bimanual:
    _target_: egomimic.rldb.zarr.zarr_dataset_multi.MultiDataset._from_resolver
    resolver:
      _target_: egomimic.rldb.zarr.zarr_dataset_multi.S3EpisodeResolver
      folder_path: {MIRROR}
      image_hw: [480, 640]
      key_map:
        _target_: egomimic.rldb.embodiment.yam.Yam.get_keymap
        keymap_mode: arc_tokenizer_cartesian
      transform_list:
        _target_: egomimic.rldb.embodiment.yam.Yam.get_transform_list
        action_mode: arc_tokenizer_cartesian
        coord_frame: eef_frame
        rotation_mode: euler
        min_distance_unit: 0.40
        resampled_vector_length: 100
        velocity_mode: per_waypoint
        velocity_layout: {layout}
    filters:
      _target_: egomimic.rldb.filters.DatasetFilter
      filter_lambdas:
      - "{lam(v)}"
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
DIMS = {"wide": (100, 28), "stacked": (200, 14)}
arcdur_exp = exp
for lay, (R, C) in DIMS.items():
    name = f"scratch_rl2_stattempo_elmoaidan_lab_{lay}"
    open(f"{LAB}/egomimic/hydra_configs/data/abc_arc/stationery_elmoaidan_lab_{lay}.yaml", "w").write(
        f"# GENERATED by scripts/e1/build_elmoaidan.py. Elmo + Aidan organize_stationary, lab arc token D=0.40 M=100 per_waypoint,\n"
        f"# velocity_layout={lay} -> ({R}, {C}). Train {len(train)} / valid {len(val)} = the fixed split in scripts/e1/stationery_elmoaidan_split.json.\n"
        "# persistent_workers true: default re-forks loader workers each epoch and this stack deadlocks there (job 13307279).\n"
        "_target_: egomimic.pl_utils.pl_data_utils.MultiDataModuleWrapper\ntrain_datasets:\n" + leaf(train, lay) + "valid_datasets:\n" + leaf(val, lay) + TAIL)
    t = arcdur_exp
    t = t.replace("override /model: e1/hpt_flow_wrists_ft_arcdur\n", "override /model: e1/hpt_flow_wrists_ft_lab\n")
    t = t.replace("override /data: abc_arc/stationery_tempo_elmoaidan_arcdur\n", f"override /data: abc_arc/stationery_elmoaidan_lab_{lay}\n")
    t = t.replace("- override /callbacks: checkpoints\n", "- override /callbacks: checkpoints\n- override /evaluator: eval_open_loop_sim\n")
    t = t.replace("scratch_rl2_stattempo_elmoaidan_arcdur", name)
    t = t.replace("    - arcdur\n", f"    - lab_{lay}\n    - m100x{C}\n" if lay == "wide" else f"    - lab_{lay}\n    - 2mx14\n")
    t = t.replace("  action_dim: 16\n", f"  action_dim: {C}\n").replace("  action_horizon: 100\n", f"  action_horizon: {R}\n")
    t = t.replace("  limit_val_batches: 80\n", "  limit_val_batches: 1.0\n")
    t = t[:t.index("\nevaluator:\n") + 1] + (
        f"stt:\n  token_rows: {R}\n  token_cols: {C}\n"
        "evaluator:\n  action_mode: arc\n  ground_truth_action_key: actions_cartesian_untokenized\n  velocity_mode: per_waypoint\n"
        "  min_distance_unit: 0.4\n  resampled_vector_length: 100\n  execute_fraction: 0.25\n  distance_dtw_enabled: false\n"
        "  viz_every_n_epochs: 0\n  limit_val_episodes: null\n")
    hdr = (f"# @package _global_\n# FROM SCRATCH, RL2-only sort stationery, Elmo + Aidan pool ({len(train)} train / {len(val)} valid, fixed 5 % split),\n"
           f"# lab arc token velocity_layout={lay} ({R}, {C}). Aidan's h640t8_d384x10 recipe (240k, warmup 10k, seed 42) -- the arcdur twin is\n"
           "# scratch_rl2_stattempo_elmoaidan_arcdur on aidan/arc-bc-consolidated. In-training eval: open-loop sim (execute 25 %, no DTW).\n")
    t = hdr + t[t.index("defaults:"):]
    os.makedirs(f"{LAB}/egomimic/hydra_configs/experiment/yam_arc_grid", exist_ok=True)
    open(f"{LAB}/egomimic/hydra_configs/experiment/yam_arc_grid/{name}.yaml", "w").write(t)
    print("wrote", name)
os.system(f"cp {CONS}/scripts/e1/stationery_ft.sbatch {LAB}/scripts/e1/stationery_ft.sbatch")
os.system(f"cp {__file__} {LAB}/scripts/e1/build_elmoaidan.py; cp {__file__} {CONS}/scripts/e1/build_elmoaidan.py")
