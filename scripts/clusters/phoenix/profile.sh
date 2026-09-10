#!/usr/bin/env bash

: "${PACE_SCRATCH:?set PACE_SCRATCH from the current Phoenix storage report}"
: "${PUSHSHAPES_DATA_ROOT:?set PUSHSHAPES_DATA_ROOT to a freshly verified Phoenix dataset root}"

EGOVERSE_CLUSTER_PROFILE=phoenix
EGOVERSE_RUN_ROOT=${EGOVERSE_RUN_ROOT:-$PACE_SCRATCH/egoverse/runs}
EGOVERSE_SOURCE_ROOT=${EGOVERSE_SOURCE_ROOT:-$PACE_SCRATCH/egoverse/source}
