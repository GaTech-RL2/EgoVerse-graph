"""End-to-end check of group_balance_sampler on the REAL datamodule of an experiment.

Ground truth for an episode s lab is the SQL snapshot (NOT the episode-list file the sampler reads),
so a wrong or stale list shows up as a wrong realised share.
usage: python scripts/e1/test_group_balance.py abc_arc/scratch_mix5050_towels_time [n_draws]
"""
import os, sys, time, collections
import numpy as np, pandas as pd
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

exp = sys.argv[1]; n_draws = int(sys.argv[2]) if len(sys.argv) > 2 else 320000
cfgdir = os.path.abspath("egomimic/hydra_configs")
with initialize_config_dir(config_dir=cfgdir, version_base="1.3"):
    cfg = compose(config_name="train_zarr_cartesian", overrides=[f"+experiment={exp}"])
t0 = time.time(); dm = instantiate(cfg.data); print(f"datamodule built in {time.time()-t0:.0f}s", flush=True)
train = dm.train_datasets["yam_bimanual"]; valid = dm.valid_datasets["yam_bimanual"]
tr_eps = {str(n) for n, _ in train.index_map}; va_eps = {str(n) for n, _ in valid.index_map}
assert not (tr_eps & va_eps), "train/valid episode overlap"
df = pd.read_pickle(os.path.expanduser("~/episodes_snapshot_latest.pkl"))
lab = dict(zip(df.episode_hash.astype(str), df.lab.astype(str)))
missing = [e for e in tr_eps | va_eps if e not in lab]; assert not missing, f"{len(missing)} episodes not in SQL snapshot"
print("TRAIN episodes by lab (SQL):", dict(collections.Counter(lab[e] for e in tr_eps)), "| VALID:", dict(collections.Counter(lab[e] for e in va_eps)))
frames = collections.Counter(str(n) for n, _ in train.index_map)
nat = sum(c for e, c in frames.items() if lab[e] == "rl2") / len(train.index_map); print(f"natural rl2 frame share in train split: {nat:.4f}")

combined = dm.train_dataloader()
loader = combined.iterables["yam_bimanual"] if isinstance(combined.iterables, dict) else combined.iterables
sampler = loader.sampler; print("sampler type:", type(sampler).__name__, "| shuffle-free:", loader.batch_sampler.sampler is sampler)
assert type(sampler).__name__ == "GroupBalanceSampler", "the real loader is NOT using the balance sampler"
names = np.array([str(n) for n, _ in train.index_map]); is_rl2 = np.array([lab[n] == "rl2" for n in names])
res = []
for ep in range(2):
    t0 = time.time(); it = iter(sampler); idx = np.fromiter((next(it) for _ in range(n_draws)), dtype=np.int64, count=n_draws)
    assert idx.min() >= 0 and idx.max() < len(train.index_map)
    share = float(is_rl2[idx].mean()); res.append(idx); print(f"epoch {ep}: {n_draws} draws in {time.time()-t0:.1f}s, realised rl2 share (SQL truth) = {share:.4f}")
    assert 0.49 < share < 0.51, "realised share outside 49-51 %"
assert (res[0][:2000] != res[1][:2000]).any(), "two epochs drew identical indices (generator not advancing)"
# within-lab uniformity over frames: an episode s share of its lab s draws should equal its share of the lab s frames
idx = np.concatenate(res)
for g, mask in (("rl2", is_rl2), ("abc", ~is_rl2)):
    drawn = collections.Counter(names[idx[mask[idx]]]); eps = [e for e in frames if (lab[e] == "rl2") == (g == "rl2")]
    f = np.array([frames[e] for e in eps], float); d = np.array([drawn.get(e, 0) for e in eps], float)
    exp_d = f / f.sum() * d.sum(); z = (d - exp_d) / np.sqrt(np.maximum(exp_d, 1))
    print(f"{g}: {len(eps)} episodes, never-drawn {int((d==0).sum())}, corr(draws, frames) = {np.corrcoef(f, d)[0,1]:.4f}, z-score std = {z.std():.2f} (1.0 = pure sampling noise), max |z| = {np.abs(z).max():.1f}")
# the real loader path: a few real batches through workers + collate
if os.environ.get("SKIP_BATCHES"):  # login nodes OOM-kill 7 loader workers; the GPU job itself exercises this path
    print("PASS (sampler-level; batch fetch skipped)"); sys.exit(0)
t0 = time.time(); itl = iter(loader); nb = 0
for _ in range(5):
    b = next(itl); nb += 1
bs = len(b["embodiment"]); keys = sorted(b.keys())[:8]
print(f"{nb} real batches through the DataLoader in {time.time()-t0:.0f}s; keys: {keys}; batch size {bs}")
print("PASS")
