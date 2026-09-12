#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --job-name=sync_abc_eva
#SBATCH --output=pi05_%x_%j.log

# Stage 1 of the abc-eva finetune chain: mirror the 18,456 abc eva episodes
# into the shared dataset folder (skips already-present episodes).

set -euo pipefail
WT=${WT:-$(git rev-parse --show-toplevel)}
cd "$WT"
PY=${PY:-python}

"$PY" -u scripts/pi05/sync_abc_eva.py
