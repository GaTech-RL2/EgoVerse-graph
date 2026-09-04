#!/bin/bash
#SBATCH --job-name=synthetic-codec10k
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=bishop
#SBATCH --output=/coc/flash7/paphiwetsa3/experiments/synthetic-gaussian-torus-codec10k-v1/logs/%A_%a.out
#SBATCH --error=/coc/flash7/paphiwetsa3/experiments/synthetic-gaussian-torus-codec10k-v1/logs/%A_%a.err

set -euo pipefail
SOURCE=/coc/flash7/paphiwetsa3/experiments/synthetic-gaussian-torus-codec10k-v1/source/EgoVerse
PYTHON=/coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/python
: "${EXPECTED_SOURCE_COMMIT:?set EXPECTED_SOURCE_COMMIT}"
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$EXPECTED_SOURCE_COMMIT"
test -z "$(git -C "$SOURCE" status --porcelain)"
case "$SLURM_ARRAY_TASK_ID" in
  0) CONFIG=$SOURCE/egomimic/hydra_configs/experiment/synthetic/unite_gaussian_torus_latent2_codec10k.json ;;
  1) CONFIG=$SOURCE/egomimic/hydra_configs/experiment/synthetic/vfm_gaussian_torus_latent2_codec10k.json ;;
  2) CONFIG=$SOURCE/egomimic/hydra_configs/experiment/synthetic/unite_gaussian_torus_latent4_codec10k.json ;;
  3) CONFIG=$SOURCE/egomimic/hydra_configs/experiment/synthetic/vfm_gaussian_torus_latent4_codec10k.json ;;
  *) exit 2 ;;
esac
cd "$SOURCE"
export PYTHONPATH="$SOURCE" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export WANDB_DIR=/coc/flash7/paphiwetsa3/experiments/synthetic-gaussian-torus-codec10k-v1/wandb/task-$SLURM_ARRAY_TASK_ID
mkdir -p "$WANDB_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec "$PYTHON" scripts/train/train_synthetic_manifold.py --config "$CONFIG"
