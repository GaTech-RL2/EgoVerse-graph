#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=16:00:00
#SBATCH --job-name=statft
#SBATCH --output=ft_%x_%j.log
# Throttled H200 units -- these pin SM ~350 MHz at >90 C with SW thermal slowdown
# and cost 3-4x in wall time (provenance: scripts/pi05/train_abc_stationery_pi.sbatch).
#SBATCH --exclude=atl1-1-02-012-16-0,atl1-1-02-014-9-0,atl1-1-02-012-2-0
#
# Stationery fine-tune off Elmo's epoch-2099-step-00210000.ckpt, 60k steps.
# Usage: sbatch --export=ALL,ARM=time   scripts/e1/stationery_ft.sbatch
#        sbatch --export=ALL,ARM=arcdur scripts/e1/stationery_ft.sbatch
#
# ONE GPU on purpose. These rows are input-bound (see the 09-16 arc-tokenizer
# budget note), so extra ranks mostly idle, and DDP would multiply the effective
# batch past the 32 the checkpoint was converging under. num_workers is raised
# 6 -> 12 in the data configs instead -- that attacks the real bottleneck and
# changes no maths.

set -euo pipefail
ARM=${ARM:?set ARM=time or ARM=arcdur}
WT=${WT:-$(git rev-parse --show-toplevel)}
cd "$WT"
PY=${PY:-$WT/.venv/bin/python}
RUN=${OUT:-$HOME/scratch/runs/stationery_ft}/${ARM}_s42

export PATH="$WT/.venv/bin:$PATH" PYTHONPATH="$WT"
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
# W&B online on Phoenix, always (PACE.md §4); entity rl2-group is pinned in the config.
export WANDB_MODE=${WANDB_MODE:-online}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Loader workers do small numpy ops; without this each spawns a full BLAS pool.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_$SLURM_JOB_ID TRITON_CACHE_DIR=/tmp/tr_$SLURM_JOB_ID

nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu --format=csv,noheader
echo "=== FT $ARM -> $RUN ($(date)) commit=$(git rev-parse --short HEAD) ==="
$PY egomimic/trainHydra.py --config-name=train_zarr_cartesian \
  +experiment=abc_arc/stationery_ft_${ARM} \
  hydra.run.dir=$RUN
