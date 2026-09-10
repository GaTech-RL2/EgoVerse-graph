#!/usr/bin/env bash
set -euo pipefail

# Move finished Hydra/Submitit run directories off the ICE home filesystem.
# Active or recently touched runs remain in place.

REPO_ROOT=${ICE_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
SOURCE_ROOT=${ICE_LOG_ROOT:-"$REPO_ROOT/logs"}
SKYNET_HOST=${ICE_LOG_SKYNET_HOST:-sky2.cc.gatech.edu}
SKYNET_ROOT=${ICE_LOG_SKYNET_ROOT:-/coc/flash7/acheluva3/ice_runs}
MIN_AGE_MINUTES=${ICE_LOG_MIN_AGE_MINUTES:-30}
STATE_ROOT=${ICE_LOG_ARCHIVER_STATE_ROOT:-"${XDG_STATE_HOME:-$HOME/.local/state}/egoverse-ice-log-archiver"}
LOCK_FILE="$STATE_ROOT/lock"
ARCHIVER_LOG="$STATE_ROOT/archive.log"

# Keep the ICE-side high-level logging folder present for Hydra/Submitit.
mkdir -p "$SOURCE_ROOT" "$STATE_ROOT"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  exit 0
fi

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >>"$ARCHIVER_LOG"
}

remote_root_q=$(printf '%q' "$SKYNET_ROOT")
if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$SKYNET_HOST" \
  "mkdir -p -- $remote_root_q" >>"$ARCHIVER_LOG" 2>&1; then
  log "destination-unavailable host=$SKYNET_HOST root=$SKYNET_ROOT"
  exit 0
fi

while IFS= read -r -d '' run_dir; do
  # Directory mtimes are insufficient because a log file may still be changing.
  if find "$run_dir" -type f -mmin "-$MIN_AGE_MINUTES" -print -quit | grep -q .; then
    log "skip-active path=${run_dir#$SOURCE_ROOT/}"
    continue
  fi

  relative=${run_dir#"$SOURCE_ROOT"/}
  remote_dir="$SKYNET_ROOT/$relative"
  remote_dir_q=$(printf '%q' "$remote_dir")
  if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$SKYNET_HOST" \
    "mkdir -p -- $remote_dir_q" >>"$ARCHIVER_LOG" 2>&1; then
    log "mkdir-failed path=$relative"
    continue
  fi

  if rsync -a --checksum --delete --protect-args \
    "$run_dir/" "$SKYNET_HOST:$remote_dir/" >>"$ARCHIVER_LOG" 2>&1; then
    rm -rf -- "$run_dir"
    log "archived path=$relative"
  else
    log "transfer-failed path=$relative"
  fi
done < <(find "$SOURCE_ROOT" -mindepth 2 -maxdepth 2 -type d -print0 | sort -z)
