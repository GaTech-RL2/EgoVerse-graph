#!/bin/bash
#SBATCH --job-name=rollout-multi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/hice1/agao81/scratch/logs/rollout-%j.log
# Several driver invocations in one GPU job (e.g. five OEC-56 levels). Env: ROLLOUT_ARGS_LIST
# (invocations separated by ' ;; '), SOURCE, PYTHON, SIM_ROOT — same contract as rollout_pushshapes.sbatch.
set -euo pipefail
: "${ROLLOUT_ARGS_LIST:?}" "${SOURCE:?}" "${PYTHON:?}" "${SIM_ROOT:?}"
cd "$SOURCE"
test -d "$SIM_ROOT/Tsimulation" || { echo "SIM_ROOT has no Tsimulation/"; exit 64; }
export PYTHONPATH="$SOURCE:$SIM_ROOT" SIM_ROOT PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy TSIM_VERSION=sim_v2
echo "job=$SLURM_JOB_ID host=$(hostname) commit=$(git rev-parse --short HEAD) sim=$SIM_ROOT start=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader
IFS_SAVE=$IFS
rc=0
while IFS= read -r -d $'\x1e' one || [ -n "$one" ]; do
  one=${one## }; one=${one%% }
  [ -z "$one" ] && continue
  echo "[multi] $(date -Is) :: $one" | cut -c1-300
  eval "$PYTHON" scripts/eval/rollout_pushshapes.py $one || { echo "[multi] invocation FAILED (rc=$?)"; rc=1; }
done < <(printf '%s' "$ROLLOUT_ARGS_LIST" | sed 's/ ;; /\x1e/g'; printf '\x1e')
echo "end=$(date -Is)"
exit $rc
