#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -p gpu-h200
#SBATCH -N1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --gres=gpu:h200:1
#SBATCH -t 6:00:00
#SBATCH --exclude=atl1-1-02-012-16-0,atl1-1-02-014-9-0,atl1-1-02-012-2-0
#SBATCH --job-name=midtempo-eval
#SBATCH --output=/storage/project/r-dxu345-0/agao81/runs/stationery_midtempo/logs/%x_%j.log
# Stationery mid-tempo holdout: open-loop eval-only scoring of checkpoints on whole sets.
#
#   sbatch --export=ALL,ROW=time|arcdur,SETS="val test_mid test_in",CKPTS="<ckpt> [<ckpt> ...]"[,RUN=<dir>] \
#     scripts/e1/stationery_midtempo_eval.sbatch
#
# Each (checkpoint, set) pair writes $RUN/eval/step<N>_<set>/open_loop_sim.json. Step N is read
# from the checkpoint's own global_step so no '=' from epoch_epoch=49.ckpt reaches a Hydra path.
# Normalization comes from the training run's cache (norm_stats.precomputed_norm_path), never
# recomputed on the eval set.
set -euo pipefail
G=/storage/project/r-dxu345-0/agao81/EgoVerse-graph
ROW=${ROW:?set ROW=time or ROW=arcdur}
RUN=${RUN:-/storage/project/r-dxu345-0/agao81/runs/stationery_midtempo/${ROW}_s42}
SETS=${SETS:-"val"}
CKPTS=${CKPTS:?set CKPTS to one or more checkpoint paths}
DATA_PREFIX=${DATA_PREFIX:-stationery_midtempo}   # stationery_midtempo_smoke for the smoke subset
cd "$G"
export PYTHONPATH="$G" HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 WANDB_MODE=disabled
export TORCH_HOME=/storage/project/r-dxu345-0/agao81/.torch-home
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${SLURM_JOB_ID} TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}
PY="$G/.venv/bin/python"
NORM=$(find "$RUN/train" -maxdepth 2 -name "norm_stats*.json" | head -1)
[ -n "$NORM" ] || { echo "no norm stats cache under $RUN/train" >&2; exit 1; }
echo "=== ROW=$ROW RUN=$RUN NORM=$NORM host=$(hostname) job=$SLURM_JOB_ID $(date) ==="
echo "=== graph HEAD $(git -C "$G" rev-parse HEAD); dirty: $(git -C "$G" status --porcelain | wc -l) ==="
for CKPT in $CKPTS; do
  STEP=$("$PY" -c "import sys,torch; print(int(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['global_step']))" "$CKPT")
  for SET in $SETS; do
    OUT="$RUN/eval/step${STEP}_${SET}"
    echo "=== EVAL step $STEP on $SET -> $OUT ($(date)) ==="
    srun "$PY" egomimic/trainHydra.py --config-name=train_zarr_cartesian \
      +experiment=abc_arc/stationery_midtempo_${ROW} seed=42 mode=eval \
      "ckpt_path=\"$CKPT\"" \
      data=abc_arc/${DATA_PREFIX}_eval_${ROW}_${SET} \
      evaluator.limit_val_episodes=null \
      "norm_stats.precomputed_norm_path=$(dirname "$NORM")" \
      "hydra.run.dir=$OUT"
    test -s "$OUT/open_loop_sim.json" || { echo "missing $OUT/open_loop_sim.json" >&2; exit 1; }
    printf '%s\n' "$CKPT" > "$OUT/ckpt_path.txt"
  done
done
echo "=== DONE $(date) ==="
