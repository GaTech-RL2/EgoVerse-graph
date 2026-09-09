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

RUN_ROOT=${ICE_RUN_ROOT:-"$OUTPUT_ROOT/${EXPERIMENT//\//_}-$(date -u +%Y%m%dT%H%M%SZ)"}
mkdir -p "$RUN_ROOT"

echo "repo=$REPO"
echo "dataset=$DATASET"
echo "output=$RUN_ROOT"
echo "experiment=$EXPERIMENT"
echo "launcher=$LAUNCHER"
echo "account=${PACE_ACCOUNT:-gts-dxu345-rl2}"
echo "qos=${PACE_QOS:-inferno}"

cd "$REPO"
export EGOVERSE_ABC_DATASET_DIR="$DATASET"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON" -m egomimic.trainHydra -m \
  "+experiment=$EXPERIMENT" \
  "hydra/launcher=$LAUNCHER" \
  "paths.dataset_dir=$DATASET" \
  "hydra.sweep.dir=$RUN_ROOT" \
  'hydra.sweep.subdir=${hydra.job.num}'
