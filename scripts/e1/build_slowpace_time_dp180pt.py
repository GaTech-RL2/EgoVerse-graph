# 2026-09-30: DP-180M time baseline on the slowpace pool (Elmo + Aidan slow-pace stationery, 217 train eps, shared 24-ep
# held-out val): the DP twin of scratch_rl2_stattempo_slowpace_time for the rollout speed-up comparison (Ryan, group DM
# 9-30: "does baseline 2x speed work?"). Model = dp300pt_wrists_arcdur with the time action (100, 14) and the UNet
# narrowed to match the HPT run's parameter count (180.4 M).
CONS = "/storage/project/r-dxu345-0/agao81/EgoVerse-graph/.claude/worktrees/arc-bc-consolidated"
H = f"{CONS}/egomimic/hydra_configs"
m = open(f"{H}/model/e1/dp300pt_wrists_arcdur.yaml").read()
body = m[m.index("_target_: egomimic.pl_utils"):]
for a, b, n in (("    action_dim: 16\n", "    action_dim: 14\n", 2), ("        input_dim: 16\n", "        input_dim: 14\n", 1),
                ("        - 512\n        - 1024\n        - 2048\n", "        - 488\n        - 976\n        - 1952\n", 1)):
    assert body.count(a) == n, (a, body.count(a)); body = body.replace(a, b)
assert "16\n" not in body.replace("action_horizon: 16", "")
open(f"{H}/model/e1/dp180pt_wrists_time.yaml", "w").write(
    "# dp180pt time: model/e1/dp300pt_wrists_arcdur.yaml (graph Diffusion Policy, one ImageNet-pretrained ResNet-18\n"
    "# VisualCore encoder per camera, unfrozen, stock BatchNorm; ConditionalUnet1D; DDIM 100; epsilon loss) with two changes:\n"
    "# the time action (100, 14) instead of arcdur (100, 16), and UNet down_dims [488, 976, 1952] instead of\n"
    "# [512, 1024, 2048] so the model has the HPT h640t8 stationery runs' parameter count (~180 M; the 'dp300'\n"
    "# widths come to 195 M here).\n" + body)
open(f"{H}/experiment/yam_arc_grid/scratch_rl2_stattempo_slowpace_time_dp180pt.yaml", "w").write("""# @package _global_
# FROM SCRATCH, RL2-only sort stationery, tempo pool 'slowpace' (217 eps), variant time, DP-180M graph Diffusion Policy
# (3 cams, pretrained ResNet-18 encoders, UNet [488, 976, 1952], DDIM 100). Twin of scratch_rl2_stattempo_slowpace_time
# (same data, shared 24-episode held-out val, 240k / warmup 10k / seed 42 recipe, BimanualTempoEval); only the model differs.
defaults:
- /experiment/yam_arc_grid/scratch_rl2_stattempo_slowpace_time
- _self_
- override /model: e1/dp180pt_wrists_time
name: scratch_rl2_stattempo_slowpace_time_dp180pt
description: dp180pt_cos10k_240k_s42_scratch_rl2_stattempo_slowpace_time_dp180pt
logger:
  wandb:
    id: scratch_rl2_stattempo_slowpace_time_dp180pt_20260930_s42
    tags:
    - rl2
    - stationery
    - stattempo
    - tempo_slowpace
    - scratch
    - time
    - dp180
""")
print("ok")
