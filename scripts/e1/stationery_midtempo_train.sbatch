#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -p gpu-h200
#SBATCH -N1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --gres=gpu:h200:1
#SBATCH -t 14:00:00
#SBATCH --exclude=atl1-1-02-012-16-0,atl1-1-02-014-9-0,atl1-1-02-012-2-0
#SBATCH --job-name=midtempo-train
#SBATCH --output=/storage/project/r-dxu345-0/agao81/runs/stationery_midtempo/logs/%x_%j.log
# Stationery mid-tempo holdout: train one row on the graph runtime (1 x H200).
#
#   sbatch --job-name=midtempo-<row> --export=ALL,ROW=time|arcdur[,RUN=<dir>][,EXTRA="<hydra overrides>"] \
#     scripts/e1/stationery_midtempo_train.sbatch
#
# RUN defaults to runs/stationery_midtempo/<row>_s42; Hydra writes to $RUN/train
# (checkpoints/, norm stats cache, open_loop_sim.json from the in-training evals).
set -euo pipefail
G=/storage/project/r-dxu345-0/agao81/EgoVerse-graph
ROW=${ROW:?set ROW=time or ROW=arcdur}
RUN=${RUN:-/storage/project/r-dxu345-0/agao81/runs/stationery_midtempo/${ROW}_s42}
EXTRA=${EXTRA:-}
cd "$G"
export PYTHONPATH="$G" HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export TORCH_HOME=/storage/project/r-dxu345-0/agao81/.torch-home
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${SLURM_JOB_ID} TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}
if curl -fsS -o /dev/null --max-time 15 https://api.wandb.ai/healthz; then
  export WANDB_MODE=online
else
  export WANDB_MODE=offline
  echo "W&B unreachable from $(hostname) -- offline fallback" >&2
fi
echo "=== ROW=$ROW RUN=$RUN host=$(hostname) job=$SLURM_JOB_ID $(date) ==="
echo "=== graph HEAD $(git -C "$G" rev-parse HEAD) ($(git -C "$G" rev-parse --abbrev-ref HEAD)); dirty: $(git -C "$G" status --porcelain | wc -l) ==="
nvidia-smi --query-gpu=name,memory.total,clocks.sm,temperature.gpu --format=csv
mkdir -p "$RUN"
# shellcheck disable=SC2086
srun "$G/.venv/bin/python" egomimic/trainHydra.py --config-name=train_zarr_cartesian \
  +experiment=yam_arc_grid/stationery_midtempo_${ROW} seed=42 \
  "hydra.run.dir=$RUN/train" $EXTRA
echo "=== DONE $(date) ==="
