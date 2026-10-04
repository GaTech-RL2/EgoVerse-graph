# 2026-09-30: HPT-300M twins of the Elmo + Aidan time / arcdur (M=16) runs, same fixed 218/11 split.
import json, os, sys
CONS = "/storage/project/r-dxu345-0/agao81/EgoVerse-graph/.claude/worktrees/arc-bc-consolidated"
H = f"{CONS}/egomimic/hydra_configs"
sys.path.insert(0, f"{CONS}/scripts/e1"); import make_stationery_tempo_configs as g
man = json.load(open(f"{CONS}/scripts/e1/stationery_tempo_manifest.json"))
# time base (180M) data + experiment for the same pool, so the 300M time twin can inherit it like the arcdur one
open(f"{H}/data/abc_arc/stationery_tempo_elmoaidan_time.yaml", "w").write(
    g.data_cfg(man, "time", "train_elmoaidan", "val_elmoaidan", "training on train_elmoaidan (Elmo + Aidan, 218), validation on its own 5 % split (11)"))
tmpl = open(f"{H}/experiment/yam_arc_grid/scratch_rl2_towels394_time.yaml").read()
open(f"{H}/experiment/yam_arc_grid/scratch_rl2_stattempo_elmoaidan_time.yaml", "w").write(
    g.experiment(tmpl, "elmoaidan", "time", man["sets"]["train_elmoaidan"]["n"]).replace(
        "and the same 24-episode held-out validation set (8 per bucket)", "and the fixed 11-episode (5 %) validation split of the Elmo + Aidan pool"))
# 300M time model = 300M arcdur model with the three action dims back at 14 (the 180M time/arcdur pair differs the same way)
m = open(f"{H}/model/e1/hpt300_flow_wrists_arcdur.yaml").read()
n = m.count("    action_dim: 16\n") + m.count("      act_dim: 16\n")
assert n == 3, n
m = m.replace("    action_dim: 16\n", "    action_dim: 14\n").replace("      act_dim: 16\n", "      act_dim: 14\n")
open(f"{H}/model/e1/hpt300_flow_wrists_time.yaml", "w").write(
    "# HPT-300M time twin of hpt300_flow_wrists_arcdur.yaml: identical except the three action dims (FlowNoising, FlowDenoiser,\n"
    "# CrossTransformer act_dim) are 14 for the (100, 14) time chunk, exactly as hpt_flow_wrists_ft differs from _ft_arcdur.\n" + m)
for v in ("time", "arcdur"):
    base = f"scratch_rl2_stattempo_elmoaidan_{v}"; name = f"{base}_hpt300"
    open(f"{H}/experiment/yam_arc_grid/{name}.yaml", "w").write(f"""# @package _global_
# HPT-300M twin of {base} (Elmo + Aidan stationery, 218 train / 11 valid fixed split): same data, recipe and eval, only the
# model is the lab HPT-300M dims (embed 840, trunk 19 blocks x 10 heads, flow CrossTransformer 6 x 320), as the towels394
# hpt300 cell (scratch_rl2_towels394_arcdur_hpt300_lambda) did.
defaults:
- /experiment/yam_arc_grid/{base}
- _self_
- override /model: e1/hpt300_flow_wrists_{v}
name: {name}
description: hpt300_cos10k_240k_s42_{name}
logger:
  wandb:
    id: {name}_20260930_s42
    tags:
    - rl2
    - stationery
    - elmoaidan
    - scratch
    - {v}
    - hpt300
hpt:
  embed_dim: 840
  num_blocks: 19
  num_heads: 10
  stem_specs:
    cross_attn:
      crossattn_heads: 10
      crossattn_dim_head: 84
      modality_embed_dim: 840
""")
    print("wrote", name)
