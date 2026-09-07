#!/bin/bash
# ICE CPU-array launcher for the synthetic UNITE sweep (tiny MLPs; no GPU needed).
# Usage (from the checkout, on a login node):
#   SOURCE=$PWD PYTHON=/path/to/.venv/bin/python CONFIG_LIST=/path/configs.txt \
#   EXPECTED_SOURCE_COMMIT=$(git rev-parse HEAD) \
#   sbatch --array=0-$((N-1))%32 scripts/train/launch_synthetic_sweep_ice.sbatch
#SBATCH --job-name=synth-unite-sweep
#SBATCH --partition=ice-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/home/hice1/agao81/scratch/logs/ss1-%A_%a.log

set -euo pipefail
: "${SOURCE:?set SOURCE (checkout root)}"
: "${PYTHON:?set PYTHON (venv interpreter)}"
: "${CONFIG_LIST:?set CONFIG_LIST (one config path per line)}"
: "${EXPECTED_SOURCE_COMMIT:?set EXPECTED_SOURCE_COMMIT}"
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$EXPECTED_SOURCE_COMMIT"
test -z "$(git -C "$SOURCE" status --porcelain)"
CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG_LIST")
test -n "$CONFIG" && test -f "$CONFIG"
cd "$SOURCE"
export PYTHONPATH="$SOURCE" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "task=${SLURM_ARRAY_TASK_ID} config=${CONFIG} host=$(hostname) commit=${EXPECTED_SOURCE_COMMIT} start=$(date -Is)"
"$PYTHON" scripts/train/train_synthetic_manifold.py --config "$CONFIG"
echo "end=$(date -Is)"
