#!/bin/bash
# usage: grid_submit.sh <ARM> <EXPERIMENT> [walltime] [EXTRA hydra overrides...]
# One submit path for the stationery/towels grid: either H100 or H200, never anything else
# (exclude list + in-job GPU guard), throttled H200 units excluded.
set -euo pipefail
ARM=$1; EXP=$2; WT=${3:-48:00:00}; shift 3 || shift $#
EXTRA="$*"
R=/storage/project/r-dxu345-0/agao81/EgoVerse-graph
EX="$(cat $HOME/nonH_gpu_nodes.txt),atl1-1-02-012-16-0,atl1-1-02-014-9-0,atl1-1-02-012-2-0"
cd $R
sbatch --job-name="$ARM" --time="$WT" --exclude="$EX" \
  --export=ALL,ARM="$ARM",EXPERIMENT="$EXP",EXTRA="$EXTRA" scripts/e1/stationery_ft.sbatch | tail -1
