"""Fold-towels 'new290' BC runs (Aidan 2026-10-08 evening), built like build_towels_bc.py from the stationery BC Lambda configs.

new290 = the 211 fold_towels takes recorded after the towels394 pool + the 79 Aidan recorded 10-08 afternoon; the last 100
pool takes of the 10-08 'new' set are dropped. Val 24 = the 16 episodes of the 10-08 towels val that are in this set + 8 of
the 79 (seed 42). Variants time, arcdur, arcvel, arcdurhyb_r24, arcdurtri_r24; the hybrid / tri ones also set
norm_stats.widen_degenerate_quantiles, because their start-delay rows are ~always 0 on towels (the 10-08 arcdurtri runs diverged).
Manifest: scripts/e1/towels_sets_new290_20261008.json (SQL on eva).

    python scripts/e1/build_towels_new290_bc.py <hydra_configs dir>
"""
import hashlib, json, sys
from pathlib import Path
import yaml

H = Path(sys.argv[1]); ROOT = "/workspace/users/agao81/towels/data/egoverseS3ZarrDatasets"
man = json.load(open(Path(__file__).with_name("towels_sets_new290_20261008.json")))
S = man["sets"]; n = len(S["train"]); WIDEN = {"arcdurhyb_r24", "arcdurtri_r24"}
def pin(names): return hashlib.sha256("".join(f"{n}\n" for n in sorted(names)).encode()).hexdigest()
def lam(names): return "lambda row, S=frozenset({" + ",".join(f"'{h}'" for h in sorted(names)) + "}): str(row['episode_hash']) in S"
def leaf(l, names):
    r = l["resolver"]; r["folder_path"] = ROOT; r["expected_episode_count"] = len(names); r["expected_episode_names_sha256"] = pin(names)
    l["filters"]["filter_lambdas"] = [lam(names)]
class D(yaml.SafeDumper): pass
D.add_representer(str, lambda d, s: d.represent_scalar("tag:yaml.org,2002:str", s, style="'" if s.startswith("lambda ") else None))
for v in ("time", "arcdur", "arcvel", "arcdurhyb_r24", "arcdurtri_r24"):
    src = H / f"data/abc_arc/stationery_slowpace_bc_{v}_lambda.yaml"
    cfg = yaml.safe_load(src.read_text())
    assert list(cfg["train_datasets"]) == ["yam_bimanual"] and list(cfg["valid_datasets"]) == ["yam_bimanual"]
    leaf(cfg["train_datasets"]["yam_bimanual"], S["train"]); leaf(cfg["valid_datasets"]["yam_bimanual"], S["val"])
    name = f"towels_new290_bc_{v}_lambda"; exp = f"bc_rl2_towels_new290_{v}_lambda"; base = f"bc_rl2_stattempo_slowpace_{v}_lambda"
    (H / f"data/abc_arc/{name}.yaml").write_text(
        f"# GENERATED 2026-10-08 by scripts/e1/build_towels_new290_bc.py from {src.name} -- rebuild, don't edit.\n"
        f"# Fold towels new290: {n} train eps ({man['hours']['train']} h, {man['operators']['train']}), 24-ep val; folder {ROOT}.\n"
        + yaml.dump(cfg, Dumper=D, sort_keys=False, width=10**9))
    widen = "norm_stats:\n  widen_degenerate_quantiles: true\n" if v in WIDEN else ""
    (H / f"experiment/yam_arc_grid/{exp}.yaml").write_text(f"""# @package _global_
# Fold-towels new290 twin of {base} (Aidan 2026-10-08 evening): same model, recipe and open-loop validation as the stationery
# BC run; episodes: 211 post-pool takes + 79 of 10-08, {n} train, 24 val (scripts/e1/build_towels_new290_bc.py).
defaults:
- /experiment/yam_arc_grid/{base}
- _self_
- override /data: abc_arc/{name}
name: {exp}
description: h640t8_d384x10_cos10k_240k_s42_{exp}
paths:
  dataset_dir: {ROOT}
{widen}logger:
  wandb:
    id: towels_new290_train{n}_bc_{v}_lambda_20261008_s42
    tags:
    - rl2
    - towels
    - towels_new290
    - bc
    - {v}
    - openloop_val
""")
    print("wrote", exp, n, "widen" if widen else "")
