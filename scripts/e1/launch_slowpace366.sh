#!/bin/bash
# Slow-pace 366 hybrid / tri runs on Phoenix (Aidan 2026-10-09: retrain the hybrid / tri with a 72 deg rotation cap
# instead of 24): arcdurhyb arcvelhyb arcdurtri arcveltri x robot-only BC + RL2 Aria38 cotrain, 180M h640t8, 240k.
# Same frozen episodes as the 10-05 slowpace366 runs (scripts/e1/stationery_slowpace366_manifest.json).
#   scripts/e1/launch_slowpace366.sh                 build (no-op if committed) + data check + submit all 8
#   MODE=dry ...   build + data check only            MODE=smoke ...   300-step W&B-offline smokes instead
# Builder flags (--rotation-deg --variants): scripts/e1/build_slowpace366.py -h
set -euo pipefail
WT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel); cd "$WT"
PY=${PY:-$WT/.venv/bin/python}
OUT=${OUT:-$HOME/scratch/runs/stat_tempo}
MODE=${MODE:-launch}
export PYTHONPATH=$WT
# The resolver fetches episodes missing from the mirror with s5cmd, which lives next to the venv python.
export PATH="$(dirname "$PY"):$PATH"
run_cpu() {
  if [ -n "${SLURM_JOB_ID:-}" ]; then srun --ntasks=1 "$@"
  else srun -A gts-dxu345-rl2 -q inferno -p cpu-small -N1 -c8 --mem=64G -t 3:00:00 "$@"; fi
}
run_cpu "$PY" scripts/e1/build_slowpace366.py "$@"
git diff --quiet -- egomimic/hydra_configs || { echo "regenerated configs differ from the commit -- review and commit first" >&2; exit 1; }
run_cpu "$PY" scripts/e1/build_slowpace366.py --check "$@" 2>&1 | grep -v -i warn
[ "$MODE" = dry ] && exit 0
EXCLUDE=$(cat scripts/e1/phoenix_exclude_nodes.txt)
mkdir -p "$OUT"
# Only the marked line: srun also prints the Slurm prolog (10-09 the unmarked list submitted 22 bogus jobs).
NAMES=$(run_cpu "$PY" scripts/e1/build_slowpace366.py --names "$@" | sed -n 's/^NAMES //p')
for EXP in $NAMES; do [[ $EXP =~ ^(bc|cotrain)_rl2_stattempo_slowpace366_[a-z_]+_r[0-9.]+$ ]] || { echo "bad experiment name: $EXP" >&2; exit 1; }; done
[ -n "$NAMES" ] || { echo "no experiment names" >&2; exit 1; }
for EXP in $NAMES; do
  ARM=${EXP/_rl2_stattempo_/_}
  if [ "$MODE" = smoke ]; then
    X=(--time=1:00:00 --export=ALL,ARM=smk_$ARM,EXPERIMENT=yam_arc_grid/$EXP,PY=$PY,OUT=$OUT/smoke_slowpace366,WANDB_MODE=offline,"EXTRA=ft.max_steps=300 ft.warmup_steps=50 trainer.check_val_every_n_epoch=2")
  else
    X=(--time=${WALL:-32:00:00} --export=ALL,ARM=$ARM,EXPERIMENT=yam_arc_grid/$EXP,PY=$PY,OUT=$OUT)
  fi
  J=$(sbatch --parsable --job-name=$ARM --cpus-per-task=12 --mem=160G --exclude=$EXCLUDE --output=$OUT/%x_%j.log "${X[@]}" scripts/e1/stationery_ft.sbatch)
  echo "$ARM -> job $J  (W&B id ${ARM}_20260917_s42_j$J, log $OUT/${ARM}_$J.log)"
done
