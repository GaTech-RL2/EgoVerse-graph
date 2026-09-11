#!/bin/bash
# Two extra protocol replicates (rep2, rep3) of the bar checkpoints and the incumbent, both seed blocks.
set -euo pipefail
cd ~/scratch/autoresearch/orchestrator-260909-1520
R2=/storage/cedar/cedar0/cedarp-dxu345-0/agao81/runs/unite-cotrain-2
sub() { # run_dir ckpt emb tag-prefix var(cfg1|cfgemb) blk rep
  local rd=$1 ck=$2 emb=$3 pre=$4 var=$5 blk=$6 rep=$7 extra=""
  [ "$var" = cfg1 ] && extra="--cfg 1.0"
  ./rollout_ct3.sh "$rd" "$ck" "$emb" 0 "${pre}-${emb}-L0-s${blk}-${var}-rep${rep}" $extra --seed-block "$blk" | tail -1
}
for rep in 2 3; do for blk in 0 40; do
  sub $R2/dp_paper-cotrain-points6-240k-09091540 "epoch-epoch=16-step-step=240000.ckpt" usocket dpct-240k cfgemb $blk $rep
  sub $R2/dp_paper-cotrain-points6-240k-09091540 "epoch-epoch=16-step-step=240000.ckpt" chain   dpct-240k cfgemb $blk $rep
  sub $R2/dp_paper-usocket-240k-09091540         "epoch-epoch=22-step-step=180000.ckpt" usocket dpus-180k cfgemb $blk $rep
  sub $R2/dp_paper-chain-points6-240k-09091540   "epoch-epoch=16-step-step=240000.ckpt" chain   dpch-240k cfgemb $blk $rep
  sub $R2/ctA-h384-240k-09091542                 "epoch-epoch=8-step-step=120000.ckpt" usocket ctA-120k  cfg1   $blk $rep
  sub $R2/ctA-h384-240k-09091542                 "epoch-epoch=8-step-step=120000.ckpt" chain   ctA-120k  cfg1   $blk $rep
done; done
