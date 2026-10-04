# 2026-09-30: arcvel (E1 velocity-profile token, (100, 16)) on the Elmo + Aidan split, 180M recipe; the arcdur run with only
# `variant` changed, exactly as the towels394 arcvel cell twins its arcdur cell.
import json, sys
CONS = "/storage/project/r-dxu345-0/agao81/EgoVerse-graph/.claude/worktrees/arc-bc-consolidated"
H = f"{CONS}/egomimic/hydra_configs"
sys.path.insert(0, f"{CONS}/scripts/e1"); import make_stationery_tempo_configs as g
man = json.load(open(f"{CONS}/scripts/e1/stationery_tempo_manifest.json"))
d = g.data_cfg(man, "arcvel", "train_elmoaidan", "val_elmoaidan", "training on train_elmoaidan (Elmo + Aidan, 218), validation on its own 5 % split (11)")
assert d.count("        variant: arcvel\n") == 2
open(f"{H}/data/abc_arc/stationery_tempo_elmoaidan_arcvel.yaml", "w").write(d)
e = open(f"{H}/experiment/yam_arc_grid/scratch_rl2_stattempo_elmoaidan_arcdur.yaml").read()
for a, b in (("override /data: abc_arc/stationery_tempo_elmoaidan_arcdur\n", "override /data: abc_arc/stationery_tempo_elmoaidan_arcvel\n"),
             ("scratch_rl2_stattempo_elmoaidan_arcdur", "scratch_rl2_stattempo_elmoaidan_arcvel"),
             ("  variant: arcdur\n", "  variant: arcvel\n"), ("    - arcdur\n", "    - arcvel\n"), ("variant arcdur.\n", "variant arcvel.\n")):
    assert e.count(a) >= 1, a; e = e.replace(a, b)
open(f"{H}/experiment/yam_arc_grid/scratch_rl2_stattempo_elmoaidan_arcvel.yaml", "w").write(e)
print("ok")
