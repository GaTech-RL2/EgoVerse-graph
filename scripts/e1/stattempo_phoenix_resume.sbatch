#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -p gpu-h100,gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --requeue
#
# Resume one stationery tempo-bucket run on Phoenix from a checkpoint it wrote elsewhere (the Lambda loaner,
# 2026-09-25, where the loaner-low QOS kept getting preempted). Continuity is the point:
#   - same W&B run (WANDB_ID, resume: allow) -- unlike stationery_ft.sbatch, which mints a job-scoped id;
#   - the run's OWN norm stats (norm_stats.precomputed_norm_path), never recomputed;
#   - Lightning resume from the newest RUN/checkpoints/last*.ckpt (optimizer, scheduler, step);
#   - a rolling last.ckpt every 5k steps so a node failure or requeue loses little.
# The Phoenix experiment (SQL + mirror resolver) selects the same episode sets as the Lambda twin
# (check_stationery_tempo_data.py: 16/16 against the manifest).
#
# Usage: sbatch --job-name=<n> --export=ALL,RUN=<dir with checkpoints/last.ckpt + norm_stats/>,\
#          EXPERIMENT=yam_arc_grid/scratch_rl2_stattempo_<b>_<v>,WANDB_ID=<id>,PY=<python> scripts/e1/stattempo_phoenix_resume.sbatch
set -euo pipefail
: "${RUN:?}" "${EXPERIMENT:?}" "${WANDB_ID:?}"
WT=${WT:-$(git rev-parse --show-toplevel)}
cd "$WT"
PY=${PY:-$WT/.venv/bin/python}
# Resume from the NEWEST last*.ckpt: a pre-existing last.ckpt (the relayed Lambda copy) makes Lightning's rolling
# checkpoint save as last-v1.ckpt, so resuming from last.ckpt alone would silently rewind (found 2026-09-26).
CKPT=$(ls -t "$RUN"/checkpoints/last*.ckpt 2>/dev/null | head -1)
test -s "$CKPT" && test -s "$RUN/norm_stats/norm_stats.json"
export PYTHONPATH="$WT" HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 WANDB_MODE=${WANDB_MODE:-online}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_$SLURM_JOB_ID TRITON_CACHE_DIR=/tmp/tr_$SLURM_JOB_ID
nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu --format=csv,noheader
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qE 'H100|H200' || { echo "WRONG GPU CLASS" >&2; exit 42; }
echo "=== RESUME $EXPERIMENT from $CKPT wandb=$WANDB_ID ($(date)) commit=$(git rev-parse --short HEAD) restart=${SLURM_RESTART_COUNT:-0} ==="
$PY egomimic/trainHydra.py --config-name=train_zarr_cartesian \
  +experiment=${EXPERIMENT} \
  logger.wandb.id=${WANDB_ID} logger.wandb.resume=allow \
  ckpt_path=$CKPT \
  norm_stats.precomputed_norm_path=$RUN/norm_stats \
  callbacks.model_checkpoint.save_last=false \
  "+callbacks.resume_last={_target_:lightning.pytorch.callbacks.ModelCheckpoint,dirpath:$RUN/checkpoints,save_last:true,save_top_k:0,every_n_train_steps:5000}" \
  paths.output_dir=$RUN \
  norm_stats.save_cache_dir=$RUN \
  hydra.run.dir=$RUN \
  ${EXTRA:-}
