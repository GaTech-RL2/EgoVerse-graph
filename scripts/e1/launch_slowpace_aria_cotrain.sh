#!/bin/bash
# Slow-pace YAM (Elmo + Aidan) + RL2 Aria cotrain on Phoenix, 180M h640t8 recipe, 240k steps (Aidan 2026-10-01).
# Variants: time arcdur arcdurhyb arcvel arcvelhyb by default; --variants picks a subset (submitted in that order). KINDS
# (default "cotrain bc") also submits each variant's robot-only BC twin (experiment bc_rl2_stattempo_slowpace_<variant>,
# same robot data, recipe and open-loop val). Run names carry the robot and Aria episode counts
# (stattempo_slowpace<R>_aria<N>_cotrain_<variant>, stattempo_slowpace<R>_bc_<variant>) so data versions stay apart in W&B.
# Run from a Phoenix login node once the Aria episodes are ingested (SQL rows with frames, zarr on R2):
#   scripts/e1/launch_slowpace_aria_cotrain.sh --list                         which rl2 Aria tasks exist since --since
#   scripts/e1/launch_slowpace_aria_cotrain.sh --tasks "organize stationary"  build + data check + submit the runs
#   MODE=dry   ...   build + data check only        MODE=smoke ...   300-step W&B-offline smokes instead
# Builder flags (--tasks --since --operators --variants --human-batch --limit): scripts/e1/build_slowpace_aria_cotrain.py -h
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
# Data check on a CPU node (login-node Lustre reads fail intermittently): compose every variant, sync any
# missing Aria zarrs into the mirror, load a sample per embodiment, assert 30 fps.
CHECK=($PY scripts/e1/build_slowpace_aria_cotrain.py --check)
if [ -n "${SLURM_JOB_ID:-}" ]; then "${CHECK[@]}"   # already on a compute node (e.g. a poller job)
else srun -A gts-dxu345-rl2 -q inferno -p cpu-small -N1 -c8 --mem=64G -t 3:00:00 "${CHECK[@]}"; fi 2>&1 | grep -v -i warn
[ "$MODE" = dry ] && exit 0
# H100/H200 only; excluded: non-Hopper and known-throttled nodes (same list as the stattempo runs).
# WALL (default 32 h; 240k took ~23-27 h plus ~1 h of in-training open-loop validation): a job whose wall time would overlap a PACE maintenance reservation
# (scontrol show reservation) cannot start until after it -- keep WALL short enough to finish before one.
EXCLUDE=$(cat scripts/e1/phoenix_exclude_nodes.txt)
N=$($PY -c "import json; print(json.load(open('scripts/e1/stationery_slowpace_aria_manifest.json'))['n'])")
R=$($PY -c "import json; print(json.load(open('scripts/e1/stationery_slowpace_aria_manifest.json'))['robot']['n'])")
VARIANTS=$($PY -c "import json; print(' '.join(json.load(open('scripts/e1/stationery_slowpace_aria_manifest.json'))['variants']))")
for v in $VARIANTS; do for k in ${KINDS:-cotrain bc}; do
  case $k in
    cotrain) EXP=yam_arc_grid/cotrain_rl2_stattempo_slowpace_aria_$v; NAME=stattempo_slowpace${R}_aria${N}_cotrain_$v ;;
    bc)      EXP=yam_arc_grid/bc_rl2_stattempo_slowpace_$v;          NAME=stattempo_slowpace${R}_bc_$v ;;
    *) echo "unknown kind $k (KINDS: cotrain bc)" >&2; exit 2 ;;
  esac
  if [ "$MODE" = smoke ]; then
    ARM=smk_$NAME; X=(--time=1:00:00 --export=ALL,ARM=$ARM,EXPERIMENT=$EXP,PY=$PY,OUT=$OUT/smoke_aria,WANDB_MODE=offline,"EXTRA=ft.max_steps=300 ft.warmup_steps=50 trainer.check_val_every_n_epoch=2")
  else
    ARM=$NAME; X=(--time=${WALL:-32:00:00} --export=ALL,ARM=$ARM,EXPERIMENT=$EXP,PY=$PY,OUT=$OUT)
  fi
  J=$(sbatch --parsable --job-name=$ARM --cpus-per-task=12 --mem=160G --exclude=$EXCLUDE --output=$OUT/%x_%j.log "${X[@]}" scripts/e1/stationery_ft.sbatch)
  echo "$ARM -> job $J  (W&B id ${ARM}_20260917_s42_j$J, log $OUT/${ARM}_$J.log)"
done; done
