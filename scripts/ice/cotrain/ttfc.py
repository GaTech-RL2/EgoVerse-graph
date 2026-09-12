"""Time-to-first-contact / speed analysis of traced 1800-step rollouts."""
import json, sys, numpy as np
def analyze(path, budget):
    d = json.load(open(path)); rows = []
    for e in d["episodes"]:
        tr = e.get("trace")
        if not tr: continue
        cov = np.array([s["coverage"] for s in tr]); t = np.array([s["t"] for s in tr])
        pos = np.array([s["agent_pos"] for s in tr], dtype=float)
        c0 = cov[0]
        gain = cov - c0
        i_first = int(np.argmax(gain >= 0.05)) if (gain >= 0.05).any() else None
        i_half = int(np.argmax(cov >= 0.5)) if (cov >= 0.5).any() else None
        i_peak = int(np.argmax(cov)); peak = float(cov.max())
        peak_in_budget = float(cov[t <= budget].max())
        dist_pre = float(np.linalg.norm(np.diff(pos[: (i_first or len(pos))], axis=0), axis=1).sum()) if (i_first or 0) > 1 else 0.0
        speed_pre = dist_pre / max(1, (i_first or len(pos)) - 1)
        # fraction of steps with pusher speed < 0.5 px (idle) before first contact
        sp = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        idle_pre = float((sp[: (i_first or len(sp))] < 0.5).mean()) if (i_first or len(sp)) > 0 else float("nan")
        rows.append(dict(seed=e["seed"], first=t[i_first] if i_first is not None else None, half=t[i_half] if i_half is not None else None,
                         peak_t=int(t[i_peak]), peak=peak, peak_budget=peak_in_budget, speed_pre=speed_pre, idle_pre=idle_pre))
    return rows
def summ(label, rows, budget):
    n = len(rows)
    firsts = [r["first"] for r in rows if r["first"] is not None]
    halves = [r["half"] for r in rows if r["half"] is not None]
    peaks = np.array([r["peak"] for r in rows]); pb = np.array([r["peak_budget"] for r in rows])
    print(f"== {label} (n={n}, budget {budget})")
    print(f"   mean peak @1800 = {peaks.mean():.3f}   mean peak within budget = {pb.mean():.3f}   never-moved(<0.05 gain) = {n-len(firsts)}")
    print(f"   first contact: median {np.median(firsts):.0f} steps, p75 {np.percentile(firsts,75):.0f}, p90 {np.percentile(firsts,90):.0f}; share > budget = {(np.array(firsts)>budget).mean():.2f}")
    if halves: print(f"   reach 0.5: median {np.median(halves):.0f}, share within budget = {(np.array(halves)<=budget).mean():.2f} (of {len(halves)} that reach it)")
    print(f"   peak time: median {np.median([r['peak_t'] for r in rows]):.0f}; share of peaks reached after budget = {np.mean([r['peak_t']>budget for r in rows]):.2f}")
    print(f"   pusher speed before first contact: median {np.median([r['speed_pre'] for r in rows]):.2f} px/step; idle share {np.nanmedian([r['idle_pre'] for r in rows]):.2f}")
budget = int(sys.argv[3]) if len(sys.argv) > 3 else 318
for lab, p in ((sys.argv[1], sys.argv[2]),):
    summ(lab, analyze(p, budget), budget)
