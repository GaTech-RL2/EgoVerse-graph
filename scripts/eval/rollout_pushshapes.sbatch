#!/bin/bash
# Closed-loop PushShapes rollouts on coe-gpu. Env: ROLLOUT_ARGS (driver args), SOURCE (repo
# whose egomimic + scripts run), PYTHON, SIM_ROOT (dir containing the Tsimulation/ package to
# use; chain gripper needs the 2133a92 sim, U-Socket the rollout branch's).
#SBATCH --partition=coe-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=H100|H200
#SBATCH --time=04:00:00
#SBATCH --output=/home/hice1/agao81/scratch/logs/rollout-%j.log
set -euo pipefail
: "${ROLLOUT_ARGS:?}" "${SOURCE:?}" "${PYTHON:?}" "${SIM_ROOT:?}"
cd "$SOURCE"
test -d "$SIM_ROOT/Tsimulation" || { echo "SIM_ROOT has no Tsimulation/"; exit 64; }
export PYTHONPATH="$SOURCE:$SIM_ROOT" SIM_ROOT PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy TSIM_VERSION=sim_v2
echo "job=$SLURM_JOB_ID host=$(hostname) commit=$(git rev-parse --short HEAD) sim=$SIM_ROOT start=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader
eval "$PYTHON" scripts/eval/rollout_pushshapes.py $ROLLOUT_ARGS
echo "end=$(date -Is)"
