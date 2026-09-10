#!/usr/bin/env bash
set -euo pipefail

die() {
  echo "launch_abc_pace: $*" >&2
  exit 2
}

REPO=${ICE_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PYTHON=${ICE_PYTHON:-python}
DATASET=${EGOVERSE_ABC_DATASET_DIR:?set EGOVERSE_ABC_DATASET_DIR to the shared ARC/ABC dataset root}
OUTPUT_ROOT=${ICE_OUTPUT_ROOT:?set ICE_OUTPUT_ROOT to a persistent shared output directory}
EXPERIMENT=${ICE_EXPERIMENT:?set ICE_EXPERIMENT, for example abc_arc/abc_fstshirt_arc_bc}
LAUNCHER=${ICE_PACE_LAUNCHER:-submitit_pace_l40s}

case "$EXPERIMENT" in
  abc/*|abc_arc/*) ;;
  *) die "ICE_EXPERIMENT must name an abc/* or abc_arc/* experiment: $EXPERIMENT" ;;
esac

test -d "$REPO" || die "ICE_REPO is not a directory: $REPO"
test -d "$DATASET" || die "EGOVERSE_ABC_DATASET_DIR is not a directory: $DATASET"
mkdir -p "$OUTPUT_ROOT"

ARCHIVER_SCRIPT="$REPO/scripts/ice/archive_logs_to_skynet.sbatch"
ARCHIVER_JOB_NAME=${ICE_LOG_ARCHIVER_JOB_NAME:-ice-log-archive}
ARCHIVER_HOST=${ICE_LOG_SKYNET_HOST:-sky2.cc.gatech.edu}
ARCHIVER_ROOT=${ICE_LOG_SKYNET_ROOT:-/coc/flash7/acheluva3/ice_runs}

ensure_log_archiver() {
  test -x "$ARCHIVER_SCRIPT" || die "log archiver is not executable: $ARCHIVER_SCRIPT"
  if squeue -h -u "${USER:?USER is required}" -o '%j' \
    | awk -v name="$ARCHIVER_JOB_NAME" '$1 == name {found=1} END {exit found ? 0 : 1}'; then
    echo "log-archiver=already-running"
    return
  fi

  local job_id
  job_id=$(sbatch --parsable \
    --job-name="$ARCHIVER_JOB_NAME" \
    --export="ALL,ICE_REPO=$REPO,ICE_LOG_SKYNET_HOST=$ARCHIVER_HOST,ICE_LOG_SKYNET_ROOT=$ARCHIVER_ROOT" \
    "$ARCHIVER_SCRIPT") || die "could not submit log archiver"
  echo "log-archiver=submitted job=$job_id host=$ARCHIVER_HOST root=$ARCHIVER_ROOT"
}

RUN_ROOT=${ICE_RUN_ROOT:-"$OUTPUT_ROOT/${EXPERIMENT//\//_}-$(date -u +%Y%m%dT%H%M%SZ)"}
mkdir -p "$RUN_ROOT"

echo "repo=$REPO"
echo "dataset=$DATASET"
echo "output=$RUN_ROOT"
echo "experiment=$EXPERIMENT"
echo "launcher=$LAUNCHER"
echo "account=${PACE_ACCOUNT:-gts-dxu345-rl2}"
echo "qos=${PACE_QOS:-inferno}"

ensure_log_archiver

cd "$REPO"
export EGOVERSE_ABC_DATASET_DIR="$DATASET"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -m egomimic.trainHydra -m \
  "+experiment=$EXPERIMENT" \
  "hydra/launcher=$LAUNCHER" \
  "paths.dataset_dir=$DATASET" \
  "hydra.sweep.dir=$RUN_ROOT" \
  'hydra.sweep.subdir=${hydra.job.num}'
