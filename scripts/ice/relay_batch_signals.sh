#!/usr/bin/env bash
# Source this helper when post-run verification prevents exec-ing the runner.
# The runner publishes ICE_RUNNER_SIGNAL_READY_FILE after installing handlers.
ice_run_with_signal_relay() {
  local ready_file=$1
  shift
  test ! -e "$ready_file" || return 73
  ICE_BATCH_BOUNDARY_FORWARDED=0
  local relay_pid= relay_usr1=0 relay_cancel= relay_interrupted=0 relay_status=0
  _ice_relay_pending() {
    test -n "$relay_pid" || return 0
    test -f "$ready_file" || return 0
    if test -n "$relay_cancel"; then
      kill -s "$relay_cancel" "$relay_pid" 2>/dev/null || true
      relay_cancel=
      relay_usr1=0
    elif test "$relay_usr1" = 1; then
      if kill -USR1 "$relay_pid" 2>/dev/null; then
        ICE_BATCH_BOUNDARY_FORWARDED=1
      fi
      relay_usr1=0
    fi
  }
  trap 'relay_interrupted=1; relay_usr1=1; _ice_relay_pending' USR1
  trap 'relay_interrupted=1; relay_cancel=TERM; _ice_relay_pending' TERM
  trap 'relay_interrupted=1; relay_cancel=INT; _ice_relay_pending' INT
  ICE_RUNNER_SIGNAL_READY_FILE="$ready_file" "$@" &
  relay_pid=$!
  # Latch an early boundary instead of killing Python before its handlers exist.
  while test ! -f "$ready_file" && kill -0 "$relay_pid" 2>/dev/null; do
    sleep 0.02
  done
  _ice_relay_pending
  while :; do
    relay_interrupted=0
    if wait "$relay_pid"; then relay_status=0; else relay_status=$?; fi
    # A trapped signal interrupts bash wait, not necessarily the child. A
    # repeated wait also retains an already-exited child's actual status.
    test "$relay_interrupted" = 1 || break
  done
  trap - USR1 TERM INT
  unset -f _ice_relay_pending
  return "$relay_status"
}
