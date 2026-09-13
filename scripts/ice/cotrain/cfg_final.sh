#!/bin/bash
# Final protocol evaluation of the best DP cotrain and UNITE cotrain checkpoints on all four graphs,
# with a CFG grid for UNITE. Protocol sets only: level 0 seeds 0-39; OEC-56 30 levels x 5 seeds.
set -uo pipefail
cd ~/scratch/autoresearch/orchestrator-260909-1520
C=/storage/cedar/cedar0/cedarp-dxu345-0/agao81/runs
OEC_CHUNKS="1-5 6-10 11-15 16-20 21-25 26-30"
n=0
sub() { ./rollout_ct3.sh "$@" >/dev/null 2>&1 && n=$((n+1)) || echo "FAILED: $*"; }
# --- DP cotrain, sweep 2 (240k) and sweep 3 (240k): embedded sampler, all four graphs
for spec in "unite-cotrain-2/dp_paper-cotrain-points6-240k-09091540:240000:dpct:s2" "unite-cotrain-3/s3-dp_paper-cotrain-chaingen-240k-09100200:240000:s3dpct:s3"; do
  IFS=: read d st row bset <<<"$spec"; ck=$(ls $C/$d/checkpoints | grep -a "^epoch.*step=$st.ckpt$")
  for emb in usocket chain; do
    for rep in 1 2 3; do sub "$C/$d" "$ck" $emb 0 "${row}-$((st/1000))k-${emb}-L0-s0-cfgemb-final-rep${rep}" --budget-set $bset --seed-block 0; done
    for ch in $OEC_CHUNKS; do sub "$C/$d" "$ck" $emb $ch "${row}-$((st/1000))k-${emb}-OEC${ch}-cfgemb-final" --budget-set $bset; done
  done
done
# --- UNITE cotrain, sweep 2 (120k) and sweep 3 (180k): CFG grid
for spec in "unite-cotrain-2/ctA-h384-240k-09091542:120000:ctA:s2" "unite-cotrain-3/s3-ctA-chaingen-240k-r150k-09110412:180000:s3ctA:s3"; do
  IFS=: read d st row bset <<<"$spec"; ck=$(ls $C/$d/checkpoints | grep -a "^epoch.*step=$st.ckpt$")
  for emb in usocket chain; do
    for cfg in 1.0 1.5 2.0 3.0 4.0 6.0; do
      for rep in 1 2 3; do sub "$C/$d" "$ck" $emb 0 "${row}-$((st/1000))k-${emb}-L0-s0-cfg${cfg}-final-rep${rep}" --cfg $cfg --budget-set $bset --seed-block 0; done
    done
    for cfg in 1.0 2.0 4.0 6.0; do
      for ch in $OEC_CHUNKS; do sub "$C/$d" "$ck" $emb $ch "${row}-$((st/1000))k-${emb}-OEC${ch}-cfg${cfg}-final" --cfg $cfg --budget-set $bset; done
    done
  done
done
echo "submitted $n jobs at $(date)"
