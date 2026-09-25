#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -p cpu-small
#SBATCH -N1 --ntasks=1 --cpus-per-task=8 --mem=32G
#SBATCH -t 0:45:00
#SBATCH --job-name=midtempo-tests
#SBATCH --output=/storage/project/r-dxu345-0/agao81/runs/stationery_midtempo/logs/%x_%j.log
# CPU unit tests for the stationery mid-tempo build on the graph runtime:
# open_loop_sim (incl. the E1 wide (M,16) decode), episode-limited validation,
# the robot graph policy / arc decoder, and the ported arc campaigns.
set -uo pipefail
G=/storage/project/r-dxu345-0/agao81/EgoVerse-graph
cd "$G"
export PYTHONPATH="$G" OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
echo "=== graph HEAD $(git rev-parse HEAD); dirty: $(git status --porcelain | wc -l) ==="
"$G/.venv/bin/python" -m pytest -q -p no:cacheprovider \
  tests/test_open_loop_sim_eval.py \
  tests/test_episode_limited_validation.py \
  tests/test_robot_graph_policy.py \
  tests/test_robot_arc_campaigns.py
echo "=== PYTEST_EXIT $? ==="
