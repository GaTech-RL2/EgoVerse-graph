#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -p gpu-h100,gpu-h200
#SBATCH -N1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --gres=gpu:1
#SBATCH -t 6:00:00
#SBATCH --job-name=stattempo-eval
#SBATCH --output=/storage/project/r-dxu345-0/agao81/runs/stat_tempo_eval/logs/%x_%j.log
# Stationery tempo buckets: open-loop eval-only scoring of checkpoints on each bucket's held-out episodes.
# Same evaluator settings the stationery mid-tempo holdout was scored with (eval_open_loop_sim, execute_fraction 0.25,
# distance DTW off, no videos) so the numbers are comparable; whole ordered episodes (limit_val_batches 1.0).
#
#   sbatch --export=ALL,RUN=<bucket>_<variant>,SETS="val_slow val_medium val_fast",CKPTS="<ckpt> ..." scripts/e1/stattempo_eval.sbatch
#
# Writes $OUT_ROOT/<RUN>/step<N>_<set>/open_loop_sim.json (+ ckpt_path.txt). Norm stats = the training run's own.
set -euo pipefail
: "${RUN:?}" "${CKPTS:?}"
SETS=${SETS:-"val_slow val_medium val_fast"}
WT=${WT:-$(git rev-parse --show-toplevel)}
cd "$WT"
PY=${PY:-/storage/project/r-dxu345-0/agao81/EgoVerse-graph/.venv/bin/python}
OUT_ROOT=${OUT_ROOT:-/storage/project/r-dxu345-0/agao81/runs/stat_tempo_eval}
NORM=${NORM:?set NORM to the run norm_stats.json}
VARIANT=${RUN##*_}
export PYTHONPATH="$WT" HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${SLURM_JOB_ID} TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qE 'H100|H200' || { echo "WRONG GPU CLASS" >&2; exit 42; }
echo "=== RUN=$RUN VARIANT=$VARIANT NORM=$NORM commit=$(git rev-parse --short HEAD) host=$(hostname) $(date) ==="
for CKPT in $CKPTS; do
  STEP=$("$PY" -c "import sys,torch; print(int(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['global_step']))" "$CKPT")
  for SET in $SETS; do
    OUT="$OUT_ROOT/$RUN/step${STEP}_${SET}"
    [ -s "$OUT/open_loop_sim.json" ] && { echo "skip $OUT (done)"; continue; }
    echo "=== EVAL step $STEP on $SET -> $OUT ($(date)) ==="
    "$PY" egomimic/trainHydra.py --config-name=train_zarr_cartesian \
      +experiment=yam_arc_grid/stattempo_eval_${VARIANT} seed=42 mode=eval \
      "ckpt_path=\"$CKPT\"" \
      data=abc_arc/stationery_tempo_eval_${VARIANT}_${SET} \
      "evaluator.results_path=$OUT/open_loop_sim.json" \
      trainer.limit_val_batches=1.0 logger.wandb.offline=true \
      "norm_stats.precomputed_norm_path=$NORM" \
      "paths.output_dir=$OUT" "hydra.run.dir=$OUT"
    test -s "$OUT/open_loop_sim.json" || { echo "missing $OUT/open_loop_sim.json" >&2; exit 1; }
    printf '%s\n' "$CKPT" > "$OUT/ckpt_path.txt"
  done
done
echo "=== DONE $(date) ==="
