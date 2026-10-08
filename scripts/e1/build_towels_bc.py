"""Fold-towels BC runs (Aidan 2026-10-08), derived from the stationery BC Lambda configs -- no new recipe.

"new towels" = the last 100 takes of the towels394 pool (Aidan, 9-19 / 9-24) + every fold_towels take recorded after it (211);
"all towels" = every usable rl2 fold_towels take (605). One 24-episode val (stratified over old100 / new Aidan / new Elmo, seed 42)
is held out of both. Manifest: scripts/e1/towels_sets_20261008.json (built from SQL on eva).

For each variant this copies data/abc_arc/stationery_slowpace_bc_<v>_lambda.yaml (arcdurtri: the _r24 one) and changes only the
folder and the two episode sets + resolver pins, and writes an experiment that inherits bc_rl2_stattempo_slowpace_<v>_lambda and
overrides data / name / W&B id. Model, steps, batch, evaluator and transforms are the stationery ones.

    python scripts/e1/build_towels_bc.py <hydra_configs dir>
"""
import hashlib, json, sys
from pathlib import Path
import yaml

H = Path(sys.argv[1]); ROOT = "/workspace/users/agao81/towels/data/egoverseS3ZarrDatasets"
man = json.load(open(Path(__file__).with_name("towels_sets_20261008.json")))
S = man["sets"]
VARIANTS = {"time": "time", "arcdur": "arcdur", "arcvel": "arcvel", "arcdurtri_r24": "arcdurtri_r24"}
def pin(names): return hashlib.sha256("".join(f"{n}\n" for n in sorted(names)).encode()).hexdigest()
def lam(names): return "lambda row, S=frozenset({" + ",".join(f"'{h}'" for h in sorted(names)) + "}): str(row['episode_hash']) in S"
def leaf(l, names):
    r = l["resolver"]; r["folder_path"] = ROOT; r["expected_episode_count"] = len(names); r["expected_episode_names_sha256"] = pin(names)
    l["filters"]["filter_lambdas"] = [lam(names)]
class D(yaml.SafeDumper): pass
D.add_representer(str, lambda d, s: d.represent_scalar("tag:yaml.org,2002:str", s, style="'" if s.startswith("lambda ") else None))
for key, v in VARIANTS.items():
    src = H / f"data/abc_arc/stationery_slowpace_bc_{v}_lambda.yaml"
    for setname, train in (("new", S["train_new"]), ("all", S["train_all"])):
        cfg = yaml.safe_load(src.read_text())
        assert list(cfg["train_datasets"]) == ["yam_bimanual"] and list(cfg["valid_datasets"]) == ["yam_bimanual"]
        leaf(cfg["train_datasets"]["yam_bimanual"], train); leaf(cfg["valid_datasets"]["yam_bimanual"], S["val"])
        n = len(train); name = f"towels_{setname}_bc_{key}_lambda"
        head = (f"# GENERATED 2026-10-08 by scripts/e1/build_towels_bc.py from {src.name} -- rebuild, don't edit.\n"
                f"# Fold towels '{setname}' set: {n} train eps ({man['hours']['train_' + setname]} h, {man['operators']['train_' + setname]}), "
                f"shared 24-ep towels val; folder {ROOT}. Everything else is the stationery BC config.\n")
        (H / f"data/abc_arc/{name}.yaml").write_text(head + yaml.dump(cfg, Dumper=D, sort_keys=False, width=10**9))
        base = f"bc_rl2_stattempo_slowpace_{v}_lambda"; run = f"towels_{setname}{n}_bc_{key}"
        exp = f"bc_rl2_towels_{setname}_{key}_lambda"
        (H / f"experiment/yam_arc_grid/{exp}.yaml").write_text(f"""# @package _global_
# Fold-towels twin of {base} (Aidan 2026-10-08): same model, recipe and open-loop validation as the stationery BC run;
# only the episodes differ: '{setname}' towels set, {n} train eps, shared 24-episode towels val (scripts/e1/build_towels_bc.py).
defaults:
- /experiment/yam_arc_grid/{base}
- _self_
- override /data: abc_arc/{name}
name: {exp}
description: h640t8_d384x10_cos10k_240k_s42_{exp}
paths:
  dataset_dir: {ROOT}
logger:
  wandb:
    id: {run}_lambda_20261008_s42
    tags:
    - rl2
    - towels
    - towels_{setname}
    - bc
    - {key}
    - openloop_val
""")
        print("wrote", name, n)
