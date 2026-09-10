#!/usr/bin/env bash

: "${PACE_SCRATCH:?set PACE_SCRATCH from the current ICE pace-whoami output}"
: "${PUSHSHAPES_DATA_ROOT:?set PUSHSHAPES_DATA_ROOT to a freshly verified ICE dataset root}"

EGOVERSE_CLUSTER_PROFILE=ice
EGOVERSE_RUN_ROOT=${EGOVERSE_RUN_ROOT:-$PACE_SCRATCH/egoverse/runs}
EGOVERSE_SOURCE_ROOT=${EGOVERSE_SOURCE_ROOT:-$PACE_SCRATCH/egoverse/source}
