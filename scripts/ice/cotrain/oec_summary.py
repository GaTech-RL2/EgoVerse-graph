import csv, collections
rows=list(csv.DictReader(open("/home/hice1/agao81/scratch/rollouts/CT2-rev3/summary.tsv"), delimiter="\t"))
unite=("ctA","ctA768","s3ctA","s3ctA768","uniteus","ctAc","ctB","ctAema","ctAann","s3unitech","unitech","uniteusema","uniteusann")
oec=collections.defaultdict(list)
for r in rows:
    if not r.get("row") or r.get("level") in (None, "0") or r.get("step") is None: continue
    if r["variant"]=="cfgemb" and r["row"] in unite: continue
    oec[(r["row"],r["emb"],int(r["step"]))].append((int(r["level"]),float(r["mean_peak"]),r["sr80"],r["never_moved"]))
print("== OEC-56 (gen) results: mean peak over the levels landed so far")
for k in sorted(oec):
    v=oec[k]; m=sum(x[1] for x in v)/len(v); sr=sum(int(x[2].split("/")[0]) for x in v); nm=sum(int(x[3]) for x in v)
    print("%-8s %-8s %4dk levels=%2d mean_peak=%.3f sr80=%d/%d never_moved=%d/%d" % (k[0],k[1],k[2]//1000,len(v),m,sr,5*len(v),nm,5*len(v)))
print("\n== variants (level 0, cfg1)")
for r in rows:
    if r.get("level") is not None and r["row"] in ("ctAema","ctAann","uniteusema","uniteusann","ctA768","s3ctA","s3ctA768","s3unitech") and r["level"]=="0" and r["variant"]!="cfgemb":
        print("%-10s %7s %-8s pooled=%s (n=%s) canon=%s sr95=%s nm=%s" % (r["row"],r["step"],r["emb"],r["mean_peak"],r["n"],r["seeds0-39_mean"],r["sr95"],r["never_moved"]))
