#!/bin/bash
# Resume-only operational helper. Original/live launchers remain unchanged.
fail() { printf 'PIXEL STARTUP ERROR: %s\n' "$*" >&2; exit 2; }
need_command() { command -v "$1" >/dev/null 2>&1 || fail "Required command unavailable: $1"; }

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
  if ! printf '{"schema":"pixel-log-mirror-health-v1","checked_at_unix":%s,"last_success_unix":%s,"consecutive_failures":%s,"last_exit_code":%s}\n' \
      "$now" "$last_success" "$failures" "$code" > "$temporary"; then
    rm -f "$temporary"
    return 1
  fi
  mv -f "$temporary" "$previous"
}

# Default is the final flush: at most three bounded attempts, requiring exit0.
# The periodic watcher passes attempts=1 and retries on its next interval.
mirror_once() (
  local source=${1:?} destination=${2:?} attempts=${3:-3}
  local attempt code=1
  [[ -d "$source" ]] || { printf 'PIXEL LOG MIRROR ERROR: local log directory missing\n' >&2; return 2; }
  [[ "$attempts" =~ ^[123]$ ]] || { printf 'PIXEL LOG MIRROR ERROR: attempts must be1..3\n' >&2; return 2; }
  # The lock belongs to this subshell for the entire copy/retry/health update.
  # It is outside the transfer and cannot be copied into its own destination.
  exec 9> "$source/.mirror.lock" || return
  flock --wait 330 9 || { printf 'PIXEL LOG MIRROR WARNING: lock wait expired\n' >&2; return 75; }
  for ((attempt=1; attempt<=attempts; attempt++)); do
    code=0
    # Bound mkdir as well as rsync: either can stall on the shared filesystem.
    timeout --kill-after=15s 300s bash -c '
      mkdir -p "$2" && exec rsync -a --no-owner --no-group --delay-updates \
        --exclude=/wandb-cache/ --exclude=/.mirror.lock --exclude=/.mirror-health.tmp.\* \
        -- "$1/" "$2/"
    ' pixel-log-mirror "$source" "$destination" || code=$?
    if ! _mirror_health "$source" "$code"; then
      printf 'PIXEL LOG MIRROR ERROR: could not write local health receipt\n' >&2
      return 74
    fi
    [[ "$code" == 0 ]] && return 0
    printf 'PIXEL LOG MIRROR WARNING: attempt%s/%s exited%s; local files retained\n' "$attempt" "$attempts" "$code" >&2
    [[ "$attempt" == "$attempts" ]] || sleep 2
  done
  return "$code"
)

watch_log_mirror() {
  local child=${1:?} source=${2:?} destination=${3:?} admission=${4:?}
  # Keep the old fifth argument for dispatch compatibility. Transient periodic
  # errors must not create a permanent failure marker or signal the trainer.
  local unused_failure_path=${5:?} interval=${6:-60}
  while kill -0 "$child" 2>/dev/null; do
    sleep "$interval"
    kill -0 "$child" 2>/dev/null || break
    [[ -f "$admission" ]] || continue
    if ! mirror_once "$source" "$destination" 1; then
      printf 'PIXEL LOG MIRROR WARNING: periodic copy incomplete; retrying next interval, training continues\n' >&2
    fi
  done
}
