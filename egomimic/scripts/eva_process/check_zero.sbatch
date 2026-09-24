#!/usr/bin/env bash
# Retained CPU-only launcher. Active integration validation uses OSMO.
#SBATCH --job-name=egoverse-data-audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
audit_repo="${CHECK_DATA_REPO:?Set CHECK_DATA_REPO to the checkout}"
cd "$audit_repo"
if [[ -f emimic/bin/activate ]]; then
    source emimic/bin/activate
elif [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
else
    echo "No project Python environment found" >&2
    exit 1
fi
exec python -m egomimic.scripts.check_data "$@"
