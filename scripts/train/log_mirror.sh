#!/bin/bash
# Source this reusable helper from a training launcher on Linux.
# Requires bash, flock, GNU timeout, rsync, and standard POSIX file utilities.
#
# mirror_once LOCAL_LOG_DIR DESTINATION [ATTEMPTS=3]
#   Final flush: bounded retries, nonzero on failure. The caller owns its exit policy.
# watch_log_mirror CHILD_PID LOCAL_LOG_DIR DESTINATION ADMISSION_FILE [INTERVAL=60]
#   Periodic best-effort copies after admission. Never signals the training process.
#   Run in the background; reap it, then call mirror_once after training exits.
#
# Local files and existing destination history are retained. W&B spool is copied;
# only the disposable root wandb-cache directory and internal lock/temp files are
# excluded. Health is updated atomically locally after each transfer, so a copied
# health receipt describes the preceding attempt, not proof of this transfer.

_mirror_health() {
  local source=${1:?} code=${2:?} previous="$1/.mirror-health.json"
  local last_success=0 failures=0 now temporary
  if [[ -f "$previous" ]]; then
    last_success=$(sed -n 's/.*"last_success_unix":\([0-9][0-9]*\).*/\1/p' "$previous")
    failures=$(sed -n 's/.*"consecutive_failures":\([0-9][0-9]*\).*/\1/p' "$previous")
  fi
  [[ "$last_success" =~ ^[0-9]+$ ]] || last_success=0
  [[ "$failures" =~ ^[0-9]+$ ]] || failures=0
  now=$(date +%s)
  if [[ "$code" == 0 ]]; then last_success=$now; failures=0; else failures=$((failures + 1)); fi
  temporary=$(mktemp "$source/.mirror-health.tmp.XXXXXX") || return
  if ! printf '{"schema":"training-log-mirror-health-v1","checked_at_unix":%s,"last_success_unix":%s,"consecutive_failures":%s,"last_exit_code":%s}\n' \
      "$now" "$last_success" "$failures" "$code" > "$temporary"; then
    rm -f "$temporary"
    return 1
  fi
  mv -f "$temporary" "$previous"
}

# Default is the final flush: at most three bounded attempts, requiring exit 0.
# The periodic watcher passes attempts=1 and retries on its next interval.
mirror_once() (
  local source=${1:?} destination=${2:?} attempts=${3:-3}
  local attempt code=1
  [[ -d "$source" ]] || { printf 'TRAINING LOG MIRROR ERROR: local log directory missing\n' >&2; return 2; }
  [[ "$attempts" =~ ^[123]$ ]] || { printf 'TRAINING LOG MIRROR ERROR: attempts must be 1..3\n' >&2; return 2; }
  # The lock belongs to this subshell for the entire copy/retry/health update.
  # It is outside the transfer and cannot be copied into its own destination.
  exec 9> "$source/.mirror.lock" || return
  flock --wait 330 9 || { printf 'TRAINING LOG MIRROR WARNING: lock wait expired\n' >&2; return 75; }
  for ((attempt=1; attempt<=attempts; attempt++)); do
    code=0
    # Bound mkdir as well as rsync: either can stall on the shared filesystem.
    timeout --kill-after=15s 300s bash -c '
      mkdir -p "$2" && exec rsync -a --no-owner --no-group --delay-updates \
        --exclude=/wandb-cache/ --exclude=/.mirror.lock --exclude=/.mirror-health.tmp.\* \
        -- "$1/" "$2/"
    ' training-log-mirror "$source" "$destination" || code=$?
    if ! _mirror_health "$source" "$code"; then
      printf 'TRAINING LOG MIRROR ERROR: could not write local health receipt\n' >&2
      return 74
    fi
    [[ "$code" == 0 ]] && return 0
    printf 'TRAINING LOG MIRROR WARNING: attempt %s/%s exited %s; local files retained\n' "$attempt" "$attempts" "$code" >&2
    [[ "$attempt" == "$attempts" ]] || sleep 2
  done
  return "$code"
)

watch_log_mirror() {
  local child=${1:?} source=${2:?} destination=${3:?} admission=${4:?}
  # Transient periodic errors do not signal the trainer or poison future attempts.
  local interval=${5:-60}
  while kill -0 "$child" 2>/dev/null; do
    sleep "$interval"
    kill -0 "$child" 2>/dev/null || break
    [[ -f "$admission" ]] || continue
    if ! mirror_once "$source" "$destination" 1; then
      printf 'TRAINING LOG MIRROR WARNING: periodic copy incomplete; retrying next interval, training continues\n' >&2
    fi
  done
}
