import wandb, json
api = wandb.Api(timeout=120)
ent = "rl2-group"
ids = json.load(open("wb_hist.json"))
out = {}
for name, d in ids.items():
    proj = "pushshapes-planar-v2" if name.startswith("dp") else "pushshapes-flow-transfer"
    r = api.run(f"{ent}/{proj}/{d['id']}")
    series = {}
    for k in d["keys"]:
        try:
            h = r.history(keys=[k, "trainer/global_step"], samples=300, pandas=False)
        except Exception as e:
            continue
        pts = [(row.get("trainer/global_step", row.get("_step")), row[k]) for row in h if row.get(k) is not None]
        if pts:
            series[k] = pts
    out[name] = series
    print(name, {k: len(v) for k, v in series.items()})
json.dump(out, open("wb_series.json", "w"))
