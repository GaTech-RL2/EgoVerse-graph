#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --job-name=norm_mecka_all_6d
#SBATCH --output=pi05_%x_%j.log

# CPU-only: compute 0.1-frac quantile norm stats over all SQL lab=mecka
# episodes (shared PACE mirror already holds the data; the resolver syncs
# only the handful of missing episodes). cartesian_6d: 18D action + proprio.
#
# Output: /storage/project/r-dxu345-0/agao81/norm_stats/mecka_all_6d/norm_stats/norm_stats.json
# Training uses:
#   norm_stats.precomputed_norm_path=/storage/project/r-dxu345-0/agao81/norm_stats/mecka_all_6d/norm_stats

set -euo pipefail
WT=${WT:-$(git rev-parse --show-toplevel)}
cd "$WT"
PY=${PY:-python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

"$PY" scripts/pi05/precompute_norm_stats.py \
  --data pi05/mecka_all_pi_6d \
  --model pi05/pi0.5_bc_mecka_6d \
  --sample-frac 0.1 \
  --num-workers 22 \
  --out /storage/project/r-dxu345-0/agao81/norm_stats/mecka_all_6d
