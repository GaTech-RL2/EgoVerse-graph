#!/bin/bash
# ABC sim multitask BC on Phoenix: time / arcdur / arcvel on the 13 single-prompt abc_sim tasks
# (data: egomimic/hydra_configs/data/abc_arc/abcsim13.yaml), 180M h640t8, 240k steps, 1 GPU each.
#   scripts/e1/launch_abcsim13.sh                 submit all three
#   MODE=smoke ...                                300-step W&B-offline smokes instead
#   VARIANTS="time arcvel" ...                    a subset
set -euo pipefail
WT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel); cd "$WT"
PY=${PY:-$WT/.venv/bin/python}
OUT=${OUT:-$HOME/scratch/runs/abc_sim}
MODE=${MODE:-launch}
EXCLUDE=$(cat scripts/e1/phoenix_exclude_nodes.txt)
mkdir -p "$OUT"
for V in ${VARIANTS:-time arcdur arcvel}; do
  EXP=multitask_abcsim13_$V
  ARM=abcsim13_$V
  if [ "$MODE" = smoke ]; then
    X=(--time=1:30:00 --export=ALL,ARM=smk_$ARM,EXPERIMENT=yam_arc_grid/$EXP,PY=$PY,OUT=$OUT/smoke,WANDB_MODE=offline,"EXTRA=ft.max_steps=300 ft.warmup_steps=50 trainer.check_val_every_n_epoch=2 evaluator.limit_val_episodes=2")
  else
    X=(--time=${WALL:-36:00:00} --export=ALL,ARM=$ARM,EXPERIMENT=yam_arc_grid/$EXP,PY=$PY,OUT=$OUT)
  fi
  J=$(sbatch --parsable --job-name=$ARM --cpus-per-task=12 --mem=160G --exclude=$EXCLUDE --output=$OUT/%x_%j.log "${X[@]}" scripts/e1/stationery_ft.sbatch)
  echo "$ARM -> job $J  (W&B id ${ARM}_20260917_s42_j$J, log $OUT/${ARM}_$J.log)"
done
