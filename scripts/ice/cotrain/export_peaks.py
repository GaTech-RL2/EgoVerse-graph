#!/usr/bin/env python3
"""Collect per-episode peak coverage from protocol rollout JSONs into one file.

Usage: export_peaks.py <results_dir> <out.json>
  results_dir: rollout_ct3.sh / monitor_ct3.py output (default ~/scratch/rollouts/CT2-rev3)
  out.json:    {"row|step|emb|level|variant": {seed: peak}}, the input of plot_protocol.py

File names follow the monitor's convention:
  <row>-<step>k-<emb>-L0[-s<seedblock>]-<cfgemb|cfg1>.json       (level 0 seed blocks)
  <row>-<step>k-<emb>-OEC<a>-<b>-<cfgemb|cfg1>-L<level>.json     (OEC-56 obstacle levels)
"""
import glob, json, os, re, sys

src = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/scratch/rollouts/CT2-rev3")
dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(src, "peaks_export.json")
L0 = re.compile(r"([A-Za-z0-9]+)-(\d+)k-(usocket|chain)-L(\d+)(?:-s(\d+))?-(cfgemb|cfg1)\.json$")
OEC = re.compile(r"([A-Za-z0-9]+)-(\d+)k-(usocket|chain)-OEC\d+-\d+-(cfgemb|cfg1)-L(\d+)\.json$")
out = {}
for path in glob.glob(os.path.join(src, "*.json")):
    name = os.path.basename(path)
    m = L0.match(name)
    if m:
        row, stepk, emb, level, _, var = m.groups()
    else:
        m = OEC.match(name)
        if not m:
            continue
        row, stepk, emb, var, level = m.groups()
    try:
        d = json.load(open(path))
    except Exception:
        continue
    pr = d.get("protocol") or {}
    if pr.get("horizon_revision") != 3:
        continue  # never pool horizon revisions
    key = f"{row}|{int(stepk) * 1000}|{emb}|{int(level)}|{var}"
    bucket = out.setdefault(key, {})
    for e in d.get("episodes", []):
        bucket.setdefault(str(e["seed"]), float(e["peak"]))
json.dump(out, open(dst, "w"))
print(len(out), "keys ->", dst)
