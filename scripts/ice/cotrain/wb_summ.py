import json, numpy as np
S = json.load(open("wb_series.json"))
def binned(pts, edges):
    pts = [(s, v) for s, v in pts if s is not None]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        vals = [v for s, v in pts if a <= s < b]
        out.append(np.median(vals) if vals else None)
    return out
edges = [0, 15000, 30000, 60000, 90000, 120000, 150000, 180000, 210000, 240001]
lab = ["<15k", "15-30k", "30-60k", "60-90k", "90-120k", "120-150k", "150-180k", "180-210k", "210-240k"]
def fmt(v, p=4): return "  -   " if v is None else f"{v:.{p}f}"
print("TRAIN losses (median per step-bin)")
print("row      key                      " + " ".join(f"{l:>8}" for l in lab))
for row, key in [("ctA","Train/UNITE/FlowLoss_step"),("ctA","Train/UNITE/ReconstructionLoss_step"),("ctA768","Train/UNITE/FlowLoss_step"),("ctA768","Train/UNITE/ReconstructionLoss_step"),("ctB","Train/UNITE/FlowLoss_step"),("uniteus","Train/UNITE/FlowLoss_step"),("uniteus","Train/UNITE/ReconstructionLoss_step"),("unitech","Train/UNITE/FlowLoss_step"),("dpct","Train/MSE_step"),("dpct","Train/MSE/pushshapes_sim_u_socket_step"),("dpct","Train/MSE/pushshapes_sim_chain_gripper_step"),("dpus","Train/MSE_step"),("dpch","Train/MSE_step")]:
    if row in S and key in S[row]:
        print(f"{row:8} {key.split('/')[-1][:24]:24} " + " ".join(f"{fmt(v):>8}" for v in binned(S[row][key], edges)))
print("\nVALID (each logged point: step value)")
for row in ["ctA","ctA768","ctB","uniteus","unitech","dpct","dpus","dpch"]:
    if row not in S: continue
    for key in sorted(S[row]):
        if not key.startswith("Valid/"): continue
        if key in ("Valid/EnergyScore@32","Valid/EnergyScoreAccuracy@32","Valid/EnergyScoreDiversity@32","Valid/MSE","Valid/Native_MSE") or "UNITE" in key: 
            pts = sorted((s, v) for s, v in S[row][key] if s is not None)
            print(f"{row:8} {key[6:]:36} " + " ".join(f"{int(s/1000)}k:{v:.4f}" for s, v in pts))
print("\nLR (step: lr)")
for row in ["ctA","dpct","uniteus","dpus"]:
    pts = sorted((s, v) for s, v in S[row]["Optimizer/param_group_0_lr"] if s is not None)
    print(f"{row:8} " + " ".join(f"{int(s/1000)}k:{v:.1e}" for s, v in pts[::max(1,len(pts)//8)]))
