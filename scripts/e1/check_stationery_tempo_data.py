# 2026-09-25: construct the train/valid datasets of every stattempo config and check episode sets against the manifest
import json, sys
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import os
man = json.load(open("scripts/e1/stationery_tempo_manifest.json"))["sets"]
cfgdir = os.path.abspath("egomimic/hydra_configs")
bad = 0
for b in sys.argv[1].split(",") if len(sys.argv) > 1 else ("slow", "medium", "fast", "mixed"):
    for v in ("time", "arcdur"):
        with initialize_config_dir(config_dir=cfgdir, version_base=None):
            cfg = compose(config_name="train_zarr_cartesian", overrides=[f"+experiment=yam_arc_grid/scratch_rl2_stattempo_{b}_{v}", "paths.output_dir=/tmp/x", "hydra.run.dir=/tmp/x"], return_hydra_config=False)
        for split, want in (("train_datasets", man[f"train_{b}"]["episodes"]), ("valid_datasets", man["val"]["episodes"])):
            ds = instantiate(cfg.data[split]["yam_bimanual"])
            names = set()
            for attr in ("datasets", "episode_datasets"):
                if hasattr(ds, attr):
                    d = getattr(ds, attr); names = set(d.keys()) if isinstance(d, dict) else set(getattr(x, "episode_hash", getattr(x, "name", str(x))) for x in d); break
            ok = names == set(want)
            bad += not ok
            print(f"{b:6s} {v:6s} {split:15s} eps={len(names):4d} want={len(want):4d} len={len(ds):7d} match={ok}", flush=True)
            if not ok: print("  missing", sorted(set(want) - names)[:5], "extra", sorted(names - set(want))[:5])
print("BAD", bad)
