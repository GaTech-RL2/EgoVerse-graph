#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -p gpu-h200
#SBATCH -N1 --ntasks=1 --cpus-per-task=4 --mem=48G
#SBATCH --gres=gpu:h200:1
#SBATCH -t 1:30:00
#SBATCH --exclude=atl1-1-02-012-16-0,atl1-1-02-014-9-0,atl1-1-02-012-2-0
#SBATCH --job-name=midtempo-replay
#SBATCH --output=/storage/project/r-dxu345-0/agao81/runs/stationery_midtempo/logs/%x_%j.log
# Offline replay check of a packaged stationery mid-tempo rollout artefact (no robot I/O).
#
#   sbatch --export=ALL,ROBOT_YAML=<package>/yam_rollout_<row>.yaml,EPISODES="<hash> [<hash> ...]",OUT=<dir>[,EVAL_JSON=<open_loop_sim.json>][,MAX_SEGMENTS=N] \
#     scripts/e1/stationery_midtempo_replay.sbatch
#
# Writes $OUT/replay_<hash>.json per episode (see scripts/e1/replay_check_stationery_midtempo.py).
set -euo pipefail
G=/storage/project/r-dxu345-0/agao81/EgoVerse-graph
MIRROR=/storage/project/r-dxu345-0/shared/egoverseS3ZarrDatasets
ROBOT_YAML=${ROBOT_YAML:?set ROBOT_YAML}
EPISODES=${EPISODES:?set EPISODES to one or more episode hashes}
OUT=${OUT:?set OUT}
cd "$G"
export PYTHONPATH="$G" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export TORCH_HOME=/storage/project/r-dxu345-0/agao81/.torch-home
mkdir -p "$OUT"
echo "=== graph HEAD $(git -C "$G" rev-parse HEAD); dirty: $(git -C "$G" status --porcelain | wc -l); yaml $ROBOT_YAML ($(date)) ==="
for EP in $EPISODES; do
  ARGS=(--robot-yaml "$ROBOT_YAML" --episode "$MIRROR/$EP" --max-segments "${MAX_SEGMENTS:-0}" --out "$OUT/replay_${EP}.json")
  if [ -n "${EVAL_JSON:-}" ]; then ARGS+=(--eval-json "$EVAL_JSON"); fi
  srun "$G/.venv/bin/python" scripts/e1/replay_check_stationery_midtempo.py "${ARGS[@]}"
done
echo "=== DONE $(date) ==="
