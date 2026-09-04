#!/bin/bash
#SBATCH --job-name=synthetic-torus
#SBATCH --account=hoffman-lab
#SBATCH --partition=hoffman-lab
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=bishop
#SBATCH --output=/coc/flash7/paphiwetsa3/experiments/synthetic-gaussian-torus-v1/logs/%A_%a.out
#SBATCH --error=/coc/flash7/paphiwetsa3/experiments/synthetic-gaussian-torus-v1/logs/%A_%a.err

set -euo pipefail
SOURCE=/coc/flash7/paphiwetsa3/experiments/synthetic-gaussian-torus-v1/source/EgoVerse
PYTHON=/coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/python
: "${EXPECTED_SOURCE_COMMIT:?set EXPECTED_SOURCE_COMMIT}"
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$EXPECTED_SOURCE_COMMIT"
test -z "$(git -C "$SOURCE" status --porcelain)"
case "$SLURM_ARRAY_TASK_ID" in
  0) CONFIG=$SOURCE/egomimic/hydra_configs/experiment/synthetic/unite_gaussian_torus.json ;;
  1) CONFIG=$SOURCE/egomimic/hydra_configs/experiment/synthetic/vfm_gaussian_torus.json ;;
  *) exit 2 ;;
esac
cd "$SOURCE"
export PYTHONPATH="$SOURCE" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec "$PYTHON" scripts/train/train_synthetic_manifold.py --config "$CONFIG"
