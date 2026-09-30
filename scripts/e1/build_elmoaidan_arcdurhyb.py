# 2026-09-30: arcdurhyb (E1 duration token + independent per-arm rotation clock, (100, 18)) on the Elmo + Aidan split, 180M recipe.
import json, sys
CONS = "/storage/project/r-dxu345-0/agao81/EgoVerse-graph/.claude/worktrees/arc-bc-consolidated"
H = f"{CONS}/egomimic/hydra_configs"
sys.path.insert(0, f"{CONS}/scripts/e1"); import make_stationery_tempo_configs as g
man = json.load(open(f"{CONS}/scripts/e1/stationery_tempo_manifest.json"))
d = g.data_cfg(man, "arcdurhyb", "train_elmoaidan", "val_elmoaidan", "arcdurhyb (100, 18): translation + rotation duration clocks per arm, R = 24 deg")
assert d.count("        variant: arcdurhyb\n") == 2
d = d.replace("        variant: arcdurhyb\n", "        variant: arcdurhyb\n        rotation_distance_unit: 0.4188790204786391\n")
open(f"{H}/data/abc_arc/stationery_tempo_elmoaidan_arcdurhyb.yaml", "w").write(d)
m = open(f"{H}/model/e1/hpt_flow_wrists_ft_arcdur.yaml").read()
assert m.count("    action_dim: 16\n") == 2 and m.count("      act_dim: 16\n") == 1
m = m.replace("    action_dim: 16\n", "    action_dim: 18\n").replace("      act_dim: 16\n", "      act_dim: 18\n")
open(f"{H}/model/e1/hpt_flow_wrists_ft_arcdurhyb.yaml", "w").write(
    "# arcdurhyb twin of hpt_flow_wrists_ft_arcdur.yaml: identical except the three action dims are 18 for the\n"
    "# (100, 18) hybrid duration token [14 canonical | translation dt L, R | rotation dt L, R].\n" + m)
e = open(f"{H}/experiment/yam_arc_grid/scratch_rl2_stattempo_elmoaidan_arcdur.yaml").read()
for a, b in (("override /model: e1/hpt_flow_wrists_ft_arcdur\n", "override /model: e1/hpt_flow_wrists_ft_arcdurhyb\n"),
             ("override /data: abc_arc/stationery_tempo_elmoaidan_arcdur\n", "override /data: abc_arc/stationery_tempo_elmoaidan_arcdurhyb\n"),
             ("scratch_rl2_stattempo_elmoaidan_arcdur", "scratch_rl2_stattempo_elmoaidan_arcdurhyb"),
             ("  variant: arcdur\n", "  variant: arcdurhyb\n"), ("    - arcdur\n", "    - arcdurhyb\n    - rotclock\n")):
    assert e.count(a) >= 1, a; e = e.replace(a, b)
assert e.count("  action_dim: 16\n") == 2; e = e.replace("  action_dim: 16\n", "  action_dim: 18\n")
e = e.replace("variant arcdur.\n", "variant arcdurhyb: E1 duration token with an independent per-arm rotation clock, (100, 18).\n", 1)
open(f"{H}/experiment/yam_arc_grid/scratch_rl2_stattempo_elmoaidan_arcdurhyb.yaml", "w").write(e)
print("ok")
