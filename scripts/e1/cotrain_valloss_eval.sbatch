#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -p gpu-h100,gpu-h200
#SBATCH -N1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --gres=gpu:1
#SBATCH -t 4:00:00
#SBATCH --job-name=cotrain-valloss
#SBATCH --output=/storage/project/r-dxu345-0/agao81/runs/aria_cotrain_valloss/logs/%x_%j.log
# Validation FLOW LOSS of slow-pace + Aria cotrain checkpoints on the shared 24-episode robot val: the training objective
# on validation batches, fixed noise per batch (evaluator=flow_loss, egomimic/eval/flow_loss_eval.py).
#   sbatch --export=ALL,RUNS="<variant>:<run dir> ...",STEPS="90000 ..." scripts/e1/cotrain_valloss_eval.sbatch
# STEPS empty = each run's newest step checkpoint. Uses each run's OWN norm stats and the data config on disk (must be the
# config the run trained with -- the committed generated config). Writes $OUT_ROOT/<run dir name>/step<N>/flow_loss.json.
set -euo pipefail
: "${RUNS:?}"
WT=${WT:-$(git rev-parse --show-toplevel)}; cd "$WT"
PY=${PY:-/storage/project/r-dxu345-0/agao81/EgoVerse-graph/.venv/bin/python}
OUT_ROOT=${OUT_ROOT:-/storage/project/r-dxu345-0/agao81/runs/aria_cotrain_valloss}
export PATH="$(dirname "$PY"):$PATH" PYTHONPATH="$WT" HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${SLURM_JOB_ID} TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qE 'H100|H200' || { echo "WRONG GPU CLASS" >&2; exit 42; }
echo "=== commit=$(git rev-parse --short HEAD) host=$(hostname) $(date) ==="
for spec in $RUNS; do
  v=${spec%%:*}; RD=${spec#*:}
  steps=${STEPS:-$(ls "$RD"/checkpoints/epoch-*-step-*.ckpt | sed -E 's/.*step-0*([0-9]+)\.ckpt/\1/' | sort -n | tail -1)}
  for STEP in $steps; do
    CK=$(ls "$RD"/checkpoints/epoch-*-step-$(printf %08d $STEP).ckpt)
    OUT="$OUT_ROOT/$(basename $RD)/step$STEP"
    [ -s "$OUT/flow_loss.json" ] && { echo "skip $OUT (done)"; continue; }
    echo "=== $v step $STEP: $CK -> $OUT ($(date)) ==="
    "$PY" egomimic/trainHydra.py --config-name=train_zarr_cartesian \
      +experiment=yam_arc_grid/cotrain_rl2_stattempo_slowpace_aria_${v} seed=42 mode=eval \
      "ckpt_path=$CK" \
      "norm_stats.precomputed_norm_path=$RD/norm_stats/norm_stats.json" \
      evaluator=flow_loss "evaluator.results_path=$OUT/flow_loss.json" \
      "paths.output_dir=$OUT" "hydra.run.dir=$OUT" 2>&1 | grep -E "FLOW_LOSS|Traceback|Error" | tail -5
    "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print('RESULT', sys.argv[2], sys.argv[3], 'flow_loss', round(d['flow_loss'], 6), 'samples', d['samples'])" "$OUT/flow_loss.json" "$v" "$STEP"
  done
done
echo EVAL_DONE
