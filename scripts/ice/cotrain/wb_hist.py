import wandb, json, sys
api = wandb.Api(timeout=120)
ent = "rl2-group"
runs = {
 "ctA":   ("pushshapes-flow-transfer", "aidan-ct2-ctA-h384-240k-09091542"),
 "ctA768":("pushshapes-flow-transfer", "aidan-ct2-ctA768-h768-240k-"),
 "ctB":   ("pushshapes-flow-transfer", "aidan-ct2-ctB-h384-240k-"),
 "uniteus":("pushshapes-flow-transfer", "aidan-ct2-unite-usocket-h384-240k-"),
 "unitech":("pushshapes-flow-transfer", "aidan-ct2-unite-chain-points6-h384-240k-"),
 "dpct":  ("pushshapes-planar-v2", "aidan-ct2-dp_paper-cotrain-points6-240k-"),
 "dpus":  ("pushshapes-planar-v2", "aidan-ct2-dp_paper-usocket-240k-"),
 "dpch":  ("pushshapes-planar-v2", "aidan-ct2-dp_paper-chain-points6-240k-"),
}
out = {}
for name, (proj, prefix) in runs.items():
    try:
        if prefix.endswith("-"):
            cands = [r for r in api.runs(f"{ent}/{proj}", filters={"display_name": {"$regex": prefix.replace("aidan-", "") + ".*"}} ) ]
            if not cands:
                cands = [r for r in api.runs(f"{ent}/{proj}") if r.name.startswith(prefix.replace("aidan-", "")) or r.id.startswith(prefix)]
            r = cands[0]
        else:
            r = api.run(f"{ent}/{proj}/{prefix}")
    except Exception as e:
        print("ERR", name, str(e)[:100]); continue
    keys = [k for k in r.summary.keys() if k.startswith(("Train/", "Valid/", "Optimizer/")) and not k.endswith("_epoch")]
    keys = [k for k in keys if "/pushshapes_sim" not in k or "MSE" in k or "EnergyScore@32" in k]
    hist = r.history(keys=keys + ["trainer/global_step"], samples=400, pandas=False)
    out[name] = {"id": r.id, "name": r.name, "state": r.state, "keys": keys, "hist": hist}
    print(name, r.id, r.name, r.state, len(hist), "rows;", len(keys), "keys")
json.dump(out, open("wb_hist.json", "w"))
