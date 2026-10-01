#!/bin/bash
# Slow-pace YAM (Elmo + Aidan) + RL2 Aria cotrain on Phoenix, arcdur and time, 180M h640t8 recipe, 240k steps (Aidan 2026-10-01).
# Run from a Phoenix login node once the Aria episodes are ingested (SQL rows with frames, zarr on R2):
#   scripts/e1/launch_slowpace_aria_cotrain.sh --list                         which rl2 Aria tasks exist since --since
#   scripts/e1/launch_slowpace_aria_cotrain.sh --tasks organize_stationary    build + data check + submit both runs
#   MODE=dry   ...   build + data check only        MODE=smoke ...   300-step W&B-offline smoke of both instead
# Builder flags (--tasks --since --operators --human-batch --limit): scripts/e1/build_slowpace_aria_cotrain.py -h
set -euo pipefail
WT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel); cd "$WT"
PY=${PY:-/storage/project/r-dxu345-0/agao81/EgoVerse-graph/.venv/bin/python}
OUT=${OUT:-$HOME/scratch/runs/stat_tempo}
MODE=${MODE:-launch}
export PYTHONPATH=$WT
# The resolver fetches episodes missing from the mirror with s5cmd, which lives next to the venv python;
# exported so the CPU check and (via --export=ALL) the training jobs both find it.
export PATH="$(dirname "$PY"):$PATH"
$PY scripts/e1/build_slowpace_aria_cotrain.py "$@"
for a in "$@"; do [ "$a" = --list ] && exit 0; done
# Data check on a CPU node (login-node Lustre reads fail intermittently): compose both experiments, sync any
# missing Aria zarrs into the mirror, load a sample per embodiment, assert 30 fps.
srun -A gts-dxu345-rl2 -q inferno -p cpu-small -N1 -c8 --mem=64G -t 3:00:00 \
  $PY scripts/e1/build_slowpace_aria_cotrain.py --check | grep -v -i warn
[ "$MODE" = dry ] && exit 0
# H100/H200 only; excluded: non-Hopper and known-throttled nodes (same list as the stattempo runs).
EXCLUDE=$(cat scripts/e1/phoenix_exclude_nodes.txt)
for v in time arcdur; do
  if [ "$MODE" = smoke ]; then
    ARM=smk_aria_$v; X=(--time=1:00:00 --export=ALL,ARM=$ARM,EXPERIMENT=yam_arc_grid/cotrain_rl2_stattempo_slowpace_aria_$v,PY=$PY,OUT=$OUT/smoke_aria,WANDB_MODE=offline,"EXTRA=ft.max_steps=300 ft.warmup_steps=50 trainer.check_val_every_n_epoch=2")
  else
    ARM=stattempo_slowpace_aria_cotrain_$v; X=(--time=72:00:00 --export=ALL,ARM=$ARM,EXPERIMENT=yam_arc_grid/cotrain_rl2_stattempo_slowpace_aria_$v,PY=$PY,OUT=$OUT)
  fi
  J=$(sbatch --parsable --job-name=$ARM --cpus-per-task=12 --mem=160G --exclude=$EXCLUDE --output=$OUT/%x_%j.log "${X[@]}" scripts/e1/stationery_ft.sbatch)
  echo "$ARM -> job $J  (W&B id ${ARM}_20260917_s42_j$J, log $OUT/${ARM}_$J.log)"
done
